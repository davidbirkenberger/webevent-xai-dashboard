import re
import spacy

nlp = spacy.load("de_core_news_sm")


def clean_text(text):
    if not text or not isinstance(text, str) or text.strip() == "":
        return ""
    text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß.,!?;:()\"'\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    unique_sentences = list(dict.fromkeys(sentences))
    cleaned = []
    for s in unique_sentences:
        s = re.sub(r"[.!?]+$", lambda m: m.group(0)[0], s.strip())
        if s and not s.endswith((".", "!", "?")):
            s += "."
        cleaned.append(s)
    return " ".join(cleaned)


def is_complete_sentence(sentence):
    doc = nlp(sentence)
    has_subject = any(token.dep_ in {"sb", "nsubj", "csubj", "cj"} for token in doc)
    has_verb = any(token.pos_ in {"VERB", "AUX"} for token in doc)
    return has_subject and has_verb


def syntax_filter(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(s for s in sentences if is_complete_sentence(s)).strip()


def shorten_content(text, num_tokens=200):
    doc = nlp(text)
    tokens = [t.text for t in doc][:num_tokens]
    return "".join(t if t in ".,;!?:\"'" else f" {t}" for t in tokens).strip()


def merge_text(meta_title, meta_desc, h1, content):
    return " ".join([meta_title, meta_desc, h1, content]).strip()


def merge_text_with_segments(meta_title, meta_desc, main_h1, main_content, about_h1="", about_content=""):
    parts = {
        "meta_title": meta_title,
        "meta_desc": meta_desc,
        "main_h1": main_h1,
        "main_content": main_content,
        "about_h1": about_h1,
        "about_content": about_content,
    }

    text_segments = {}
    full_text = ""
    current_start = 0

    for key, value in parts.items():
        segment = value.strip()
        if segment:
            full_text += segment + " "
            start = current_start
            end = start + len(segment.strip().split())  # word-level fallback
            text_segments[key] = (start, end)
            current_start = end

    return full_text.strip(), text_segments

