import torch
from transformers import AutoTokenizer
import transformers.models.bert.modeling_bert as modeling_bert
from lxt.efficient import monkey_patch
from lxt.utils import clean_tokens
import numpy as np
from typing import Optional

# Monkey-patching for efficient relevance
monkey_patch(modeling_bert, verbose=False)

# Optimized thresholds
THRESHOLDS = np.array([0.21, 0.76, 0.66, 0.21, 0.51, 0.11, 0.46, 0.11, 0.16, 0.26], dtype=np.float64)


def load_model_and_tokenizer(checkpoint_path: str, device: Optional[str] = None):
    model = modeling_bert.BertForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if device is not None:
        model.to(device)

    # Sicherheitsnetz: pad_token_id
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer



def predict_probs(text: str, model, tokenizer):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    )
    dev = next(model.parameters(), None).device if any(True for _ in model.parameters()) else torch.device("cpu")
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        probs = torch.sigmoid(logits).squeeze(0)  # [num_labels]
    return probs


def predict_labels(text: str, model, tokenizer, thresholds=THRESHOLDS):
    probs = predict_probs(text, model, tokenizer)  # probs auf model.device
    thr = torch.as_tensor(thresholds, device=probs.device, dtype=probs.dtype)
    labels = (probs >= thr).int()
    return labels, probs


def compute_relevance(text: str, label_indices, model, tokenizer):
    # label_indices -> Liste
    if isinstance(label_indices, int):
        label_indices = [label_indices]

    # Tokenize EINMAL mit festen Grenzen
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    input_ids = enc["input_ids"]            # [1, L]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    token_type_ids = enc.get("token_type_ids", None)

    # Device
    first_param = next(model.parameters(), None)
    dev = first_param.device if first_param is not None else torch.device("cpu")
    input_ids = input_ids.to(dev)
    attention_mask = attention_mask.to(dev)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(dev)

    # Embeddings mit Grad
    emb_layer = model.bert.get_input_embeddings()
    inputs_embeds = emb_layer(input_ids).detach()       # [1, L, H]
    inputs_embeds.requires_grad_(True)

    # Forward mit EMBEDS + Masken
    kwargs = dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    if token_type_ids is not None:
        kwargs["token_type_ids"] = token_type_ids

    model.eval()
    # Wichtig: zero_grad vor backward, None-Set vermeidet Akkumulation
    model.zero_grad(set_to_none=True)
    outputs = model(**kwargs)
    logits = outputs.logits  # [1, num_labels]

    # Ziel-Logit(e)
    target = logits[0, label_indices].sum()
    target.backward()

    # Grad×Input im Embedding-Raum
    rel = (inputs_embeds * inputs_embeds.grad).sum(dim=-1)  # [1, L]
    rel = rel * attention_mask  # PAD und evtl. abgeschnittene Plätze nullen
    rel = rel.detach().cpu()[0]  # [L]

    # Numerik: Divide-by-zero vermeiden
    max_abs = rel.abs().max()
    if max_abs > 0:
        rel = rel / max_abs

    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
    try:
        tokens = clean_tokens(raw_tokens)
    except Exception:
        # Fallback: akzeptiere Tokens ohne Markierungen
        tokens = []
        for t in raw_tokens:
            if t in ("[CLS]", "[SEP]", "[PAD]"):
                tokens.append(t)
                continue
            t = t.replace("##", "").replace("Ġ", " ").replace("▁", " ")
            tokens.append(t)

    return rel, tokens