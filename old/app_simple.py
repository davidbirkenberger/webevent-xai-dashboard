import streamlit as st
import torch
from transformers import AutoTokenizer
import transformers.models.bert.modeling_bert as modeling_bert
from lxt.efficient import monkey_patch
from lxt.utils import clean_tokens, pdf_heatmap

import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

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

st.set_page_config(page_title="XAI-Assisted Annotation", layout="wide")

checkpoint_path = "/Users/davidbirkenberger/Documents/rifel_models/final/checkpoint-378"
monkey_patch(modeling_bert, verbose=True)
model = modeling_bert.BertForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model.eval()
for param in model.parameters():
    param.requires_grad = False

st.title("XAI-Assisted Annotation using Gradient × Input")

# Text input
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