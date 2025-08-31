from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import spacy
import pandas as pd
import random
from urllib.parse import urlparse, urljoin
from collections import OrderedDict
from tqdm import tqdm
import streamlit as st
import torch
from transformers import AutoTokenizer
import transformers.models.bert.modeling_bert as modeling_bert
from lxt.efficient import monkey_patch
from lxt.utils import clean_tokens, pdf_heatmap

import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

from pathlib import Path
import hashlib
import json

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

nlp = spacy.load("de_core_news_sm")
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.39 Safari/537.36",
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7"
})

id2label = {
    0: "Veranstalter",
    1: "Locations und Räume",
    2: "Catering",
    3: "Agenturen",
    4: "Technik, Bühnen- und Messebau",
    5: "Service",
    6: "Vermittler/Management",
    7: "Kreativ",
    8: "Entertainment",
    9: "Hersteller"
}


def crawl_links(url):
    links = []
    start_domain = urlparse(url).netloc
    try:
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all("a", href=True):
            href = link.get("href")
            full_url = urljoin(url, href)
            link_domain = urlparse(full_url).netloc
            if link_domain == start_domain:
                link_text = link.get_text(strip=True)
                links.append({"url": full_url, "link_text": link_text if link_text else "N/A"})
        time.sleep(random.uniform(1, 3))
    except requests.RequestException as e:
        print(f"Fehler beim Abrufen der URL {url}: {e}")
    return links

def extract_page_information(url):
    try:
        time.sleep(1)
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string if soup.title else "N/A"
        meta_description = soup.find("meta", {"name": "description"})
        description = meta_description["content"] if meta_description else "N/A"
        h1_text = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        full_text = soup.get_text(separator=". ", strip=True)
        return pd.Series({
            "meta-title": title,
            "meta-description": description,
            "h1": h1_text,
            "content": full_text,
        })
    except requests.RequestException as e:
        print(f"Error retrieving URL {url}: {e}")
        return pd.Series({
            "meta-title": "N/A",
            "meta-description": "N/A",
            "h1": "N/A",
            "content": "N/A",
        })


# Select top segments function
def select_top_segments(website_url, model, tokenizer, max_segments=3):
    # Check cache
    cache_key = hashlib.md5(website_url.encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cached = json.load(f)
        return tokenizer(cached["text"], return_tensors="pt", truncation=True, padding=True)["input_ids"], cached["text"]

    # Crawl all links on the homepage (1st level)
    links = crawl_links(website_url)
    if not links:
        # fallback: just use the homepage
        links = [{"url": website_url, "link_text": "Homepage"}]
    # Extract info for each link
    df = pd.DataFrame([extract_page_information(l["url"]) for l in links])
    # Use only content column for filtering
    scored = []
    for idx, row in df.iterrows():
        text = row["content"]
        if not text or text == "N/A":
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze()
        score = probs.max().item()
        scored.append((score, text, row))
    scored.sort(reverse=True, key=lambda x: x[0])
    top_segments = scored[:max_segments]
    # Concatenate top segments
    filtered_text = "\n\n".join([t[1] for t in top_segments])
    # For visualization, return input_ids and filtered_text
    if filtered_text:
        inputs = tokenizer(filtered_text, return_tensors="pt", truncation=True, padding=True)
        with open(cache_file, "w") as f:
            json.dump({"text": filtered_text}, f)
        return inputs["input_ids"], filtered_text
    else:
        return None, ""

st.set_page_config(page_title="XAI-Assisted Annotation", layout="wide")

checkpoint_path = "/Users/davidbirkenberger/Documents/rifel_models/final/checkpoint-378"
monkey_patch(modeling_bert, verbose=True)
model = modeling_bert.BertForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model.eval()
for param in model.parameters():
    param.requires_grad = False


# Sidebar UI for URL-based XAI Filtering
st.sidebar.header("URL-based XAI Filtering")
website_url = st.sidebar.text_input("Enter a website URL (e.g. homepage)", "")
if st.sidebar.button("Extract and Filter Website"):
    with st.spinner("Crawling and analyzing website..."):
        input_ids, filtered_text = select_top_segments(website_url, model, tokenizer)
        st.session_state["probs"] = None
        st.session_state["input_text"] = filtered_text
        st.rerun()

st.title("XAI-Assisted Annotation using Gradient × Input")

# Text input
if "input_text" in st.session_state and st.session_state["input_text"]:
    text = st.session_state["input_text"]
else:
    text = st.text_area("Enter a document snippet:", "Wir bieten Event-Catering an.")

def compute_relevance(text, label_idx):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] > 512:
        input_ids = input_ids[:, :512]
    inputs_embeds = model.bert.get_input_embeddings()(input_ids)
    inputs_embeds.requires_grad_(True)
    logits = model(inputs_embeds=inputs_embeds).logits
    logit = logits[0, label_idx]
    model.zero_grad()
    logit.backward()
    relevance = (inputs_embeds * inputs_embeds.grad).sum(-1).detach().cpu()[0]
    relevance = relevance / relevance.abs().max()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)
    return relevance, tokens

if st.button("Explain predictions"):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Run prediction
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze()

    st.session_state["probs"] = probs
    st.session_state["input_text"] = text

if "probs" in st.session_state:
    probs = st.session_state["probs"]
    text = st.session_state["input_text"]

    # Guard clause: Ensure probs is set
    if probs is None:
        st.warning("No predictions available. Please click 'Explain predictions' first.")
        st.stop()

    # Show all label scores and let user choose one to explain
    label_options = []
    for i, score in enumerate(probs):
        label = id2label.get(i, f"Label {i}")
        label_options.append(f"{label} ({score:.3f})")

    label_descriptions = {
        0: "Plant und führt Veranstaltungen durch (z.B. Messen, Konzerte, Feste, Sportevents).",
        1: "Stellt Räume und Locations für Veranstaltungen bereit (z.B. Hallen, Hotels, Stadien).",
        2: "Versorgt Veranstaltungen mit Speisen und Getränken (z.B. Event-Catering, Food-Trucks).",
        3: "Kreative Planung und Konzeption durch Agenturen (z.B. Event-, Kommunikations-, Messeagenturen).",
        4: "Technische Umsetzung und Infrastruktur (z.B. Licht, Ton, Bühnenbau, Sicherheit).",
        5: "Dienstleistungen wie Personal, Sicherheit, Logistik, Ticketing, Hygiene.",
        6: "Künstlervermittlung und Management (z.B. Booking, Tourmanagement).",
        7: "Künstlerisch-kreative Produktion (z.B. Film, Design, Pressearbeit).",
        8: "Live-Performer auf Events (z.B. Musiker, Schauspieler, Comedians, Moderatoren).",
        9: "Produziert technisches Equipment und Infrastruktur für Events."
    }

    # --- Multi-label explanatory template ---
    top_k = 3
    threshold = 0.5
    predicted_indices = [i for i, p in enumerate(probs) if p > threshold]
    predicted_names = [id2label[i] for i in predicted_indices]

    # Collect top-k tokens per label
    explanations = {}
    for i in predicted_indices:
        rel, tok = compute_relevance(text, i)
        top_tokens = [t for _, t in sorted(zip(rel, tok), key=lambda x: abs(x[0]), reverse=True)[:top_k]]
        explanations[id2label[i]] = top_tokens

    # Format template
    template = "### Prediction: \n"
    for i in predicted_indices:
        label = id2label[i]
        score = probs[i].item()
        top = explanations[label]
        template += f" **{label}** ({score:.2f}) durch Tokens: {', '.join(top)} "
        template += " \n "
    st.markdown(template)

    st.subheader("Explanation:")
    selected_idx = st.session_state.get("selected_idx", 0)
    selected_label = st.pills(
        "Wähle einen Cluster zur Erklärung:",
        options=label_options,
        help="Hover über die Einträge für Beschreibung."
    ) or label_options[selected_idx]
    selected_idx = label_options.index(selected_label)
    st.session_state["selected_idx"] = selected_idx

    label_idx = torch.tensor([selected_idx])
    label_name = id2label.get(label_idx.item(), f"Label {label_idx.item()}")

    relevance, tokens = compute_relevance(text, selected_idx)



    st.markdown("**Highlighted tokens:**")
    html = ""
    for token, score in zip(tokens, relevance):
        intensity = min(1.0, abs(score.item()))
        r, g, b = (255, 0, 0) if score > 0 else (0, 0, 255)
        bg = f"rgba({r},{g},{b},{intensity})"
        html += f"<span title='{score.item():.3f}' style='background-color:{bg}; padding:2px; margin:1px; border-radius:4px;'>{token} </span>"

    st.markdown(f"<div style='line-height:2em; font-size:1.1em'>{html}</div>", unsafe_allow_html=True)