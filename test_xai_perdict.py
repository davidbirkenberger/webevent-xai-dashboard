import torch
import pytest
from transformers import AutoTokenizer
from xai_predict import load_model_and_tokenizer, predict_probs, predict_labels, compute_relevance

# Beispielmodell: Mini-BERT (ggf. ersetzen durch deinen Checkpoint)
CHECKPOINT_PATH = "models/rifel_multilabel"


@pytest.fixture(scope="session")
def model_and_tokenizer():
    model, tokenizer = load_model_and_tokenizer(CHECKPOINT_PATH)
    return model, tokenizer


def test_predict_probs_range(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "Das ist ein Testsatz."
    probs = predict_probs(text, model, tokenizer)

    # Werte zwischen 0 und 1
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
    # Shape = [num_labels]
    assert probs.ndim == 1
    assert probs.numel() == model.config.num_labels


def test_predict_labels_thresholds(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "Noch ein Testsatz"

    labels, probs = predict_labels(text, model, tokenizer, thresholds=[0.5] * model.config.num_labels)

    # Labels binär
    assert set(labels.tolist()) <= {0, 1}
    # Konsistenz: label=1 <=> prob >= threshold
    for l, p in zip(labels, probs):
        assert (l == 1 and p >= 0.5) or (l == 0 and p < 0.5)


def test_compute_relevance_outputs(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "Hallo Welt."

    relevance, tokens = compute_relevance(text, 0, model, tokenizer)

    # Länge stimmt mit Tokens überein
    assert len(relevance) == len(tokens)

    # Keine NaNs
    assert not torch.isnan(relevance).any()

    # Normierung: max(abs)=1 oder 0
    max_abs = relevance.abs().max().item()
    assert 0 <= max_abs <= 1 + 1e-6

    # PAD-Tokens sollten 0-Relevanz haben
    if tokenizer.pad_token_id is not None:
        for tok, score in zip(tokens, relevance):
            if tok == "[PAD]":
                assert score.item() == 0.0