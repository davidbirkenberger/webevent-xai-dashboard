import torch
import pytest
from transformers import BertTokenizerFast
from segment_shap import exact_segment_shap_for_sample

# ----- Dummy-Modell -----
class SumModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # Embedding, die einfach die Token-ID als Wert zurückgibt
        self.emb = torch.arange(vocab_size, dtype=torch.float).unsqueeze(1)
        self._dummy = torch.nn.Parameter(torch.empty(0))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # input_ids: [B, L], attention_mask: [B, L]
        vals = []
        for i in range(input_ids.shape[0]):
            ids = input_ids[i]
            mask = attention_mask[i]
            # Summe aller Token-IDs, die sichtbar sind
            vals.append((ids * mask).sum().unsqueeze(0))
        logits = torch.stack(vals).unsqueeze(-1)  # [B, 1]
        return type("O", (), {"logits": logits})

# ----- Test -----
def test_shap_additivity():
    # Mini-Tokenizer (hier BERT für Demo)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    text = "hello world test"
    # Segmente in Wortindizes: "hello"=0, "world"=1, "test"=2
    word_segments = {
        "seg_hello": (0, 1),
        "seg_world": (1, 2),
        "seg_test": (2, 3),
    }

    model = SumModel(vocab_size=30522)
    shap_values, seg_names = exact_segment_shap_for_sample(
        model, tokenizer, text, word_segments, label_index=0
    )

    # Additivitäts-Eigenschaft: v(full) - v(empty) = Sum phi_i
    total_phi = sum(shap_values.values())
    assert abs(total_phi) > 0.0  # sollte nicht trivial null sein

    # Jede Phi sollte ca. der "Wertsteigerung" durch die Token-IDs entsprechen
    for seg, phi in shap_values.items():
        print(f"{seg}: {phi:.4f}")

    # Harte Additivitätsprüfung ist schon in der Funktion

def test_shap_two_segments_exact():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    text = "hello world"
    word_segments = {
        "seg_hello": (0, 1),
        "seg_world": (1, 2),
    }
    model = SumModel(vocab_size=30522)
    shap_values, seg_names = exact_segment_shap_for_sample(
        model, tokenizer, text, word_segments, label_index=0
    )

    # Exakte Erwartungen
    assert round(shap_values["seg_hello"]) == 7592
    assert round(shap_values["seg_world"]) == 2088
    assert round(sum(shap_values.values())) == 9680

# test_segment_shap.py (zusätzlich)
def _sum_segment_token_ids(tokenizer, text, word_segments):
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    offsets = enc["offset_mapping"][0].tolist()
    # Wortindex-Rekonstruktion analog zur Funktion
    words = text.split()
    starts, pos = [], 0
    for w in words:
        starts.append(pos); pos += len(w) + 1
    def char_to_word_id(c):
        last = 0
        for i, s in enumerate(starts):
            if s <= c: last = i
            else: break
        return last
    token_word_ids = [None if (s==e==0) else char_to_word_id(s) for (s, e) in offsets]

    # Specials markieren (damit wir sie nicht in Segment-Summen zählen)
    sp_mask = tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
    is_special = [bool(m) for m in sp_mask]

    expected = {}
    for name, (w0, w1) in word_segments.items():
        s = 0
        for t_idx, wid in enumerate(token_word_ids):
            if wid is None or is_special[t_idx]:
                continue
            if w0 <= wid < w1:
                s += ids[t_idx]
        expected[name] = s
    return expected

def test_shap_three_segments_exact():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    text = "hello world test"
    word_segments = {"seg_hello": (0,1), "seg_world": (1,2), "seg_test": (2,3)}
    model = SumModel(vocab_size=30522)

    shap_values, _ = exact_segment_shap_for_sample(
        model, tokenizer, text, word_segments, label_index=0
    )
    expected = _sum_segment_token_ids(tokenizer, text, word_segments)

    # Exakt gleich (SumModel ist additiv)
    for k in expected:
        assert round(shap_values[k]) == expected[k]

    # Additivität (wird zusätzlich in der Funktion gecheckt)
    assert round(sum(shap_values.values())) == sum(expected.values())