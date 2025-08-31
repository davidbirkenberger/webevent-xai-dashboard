# segment_shap.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import itertools
import math
import torch


@dataclass
class SegmentTokenSpans:
    """
    Holds token spans for each segment.
    Example: {"main_h1": [(5, 8)], "about_content": [(42, 91)]}
    Spans are [start, end) in token index space (inclusive start, exclusive end).
    """
    spans: Dict[str, List[Tuple[int, int]]]

    def segment_names(self) -> List[str]:
        return list(self.spans.keys())

    def tokens_for(self, name: str) -> List[Tuple[int, int]]:
        return self.spans.get(name, [])


def _enumerate_coalitions(names: List[str]) -> List[Tuple[str, ...]]:
    """All subsets (including empty set) of the feature list."""
    coalitions = []
    for r in range(len(names) + 1):
        coalitions.extend(itertools.combinations(names, r))
    return coalitions


def _mask_segments_in_inputs(
    inputs: Dict[str, torch.Tensor],
    token_spans_by_segment: Dict[str, List[Tuple[int, int]]],
    keep: Tuple[str, ...],
    *,
    special_positions: Optional[torch.Tensor] = None,  # bool [1, L]
    pad_token_id: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Setzt attention_mask zunächst komplett auf 0, aktiviert dann nur Special-Tokens
    (über `special_positions`) und die Tokens der Segmente in `keep`.
    Optional: maskierte input_ids -> pad_token_id.
    Annahme: batch size == 1.
    """
    masked = {k: v.clone() for k, v in inputs.items()}
    input_ids = masked["input_ids"]
    attention = masked["attention_mask"]

    # 1) Alles aus
    attention.zero_()

    # 2) Specials an
    if special_positions is not None:
        # special_positions: bool [1, L]
        attention[special_positions] = 1

    # 3) kept-Segmente an
    keep_mask = torch.zeros_like(attention, dtype=torch.bool)
    for seg_name in keep:
        for s, e in token_spans_by_segment.get(seg_name, []):
            keep_mask[:, s:e] = True
    attention[keep_mask] = 1

    # 4) Optional: maskierte Tokens auf PAD setzen
    if pad_token_id is not None:
        if special_positions is not None:
            visible = special_positions | keep_mask
        else:
            visible = keep_mask
        input_ids[~visible] = pad_token_id

    return masked


def _value_function(
    model,
    inputs: Dict[str, torch.Tensor],
    device: str,
    label_index: int,
) -> float:
    """
    Returns the scalar value for the coalition: the *logit* for the given label.
    (Use logits to avoid link-function nonlinearity issues.)
    """
    model.eval()
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in inputs.items()}
        out = model(**batch)
        logits = out.logits  # shape [1, num_labels]
        return float(logits[0, label_index].cpu().item())


def exact_segment_shap_for_sample(
    model,
    tokenizer,
    text: str,
    word_segments: Dict[str, Tuple[int, int]],
    label_index: int,
    device: Optional[str] = None,
    max_length: int = 512,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Compute exact Shapley values at the *segment* level for a single text sample.

    Args
    ----
    - text: the concatenated input text shown to the model.
    - word_segments: segment -> (start_word_idx, end_word_idx) in *word* space,
      as produced by your merge_text_with_segments() (inclusive start, exclusive end).
    - label_index: target label logit to explain.
    - device: torch device.
    - max_length: tokenizer truncation.

    Returns
    -------
    - shap_values: dict segment_name -> Shapley value (signed, on the logit scale)
    - segment_order: list of segment names in deterministic order
    """
    # Resolve device: default to the model's own device to avoid mismatches (e.g., MPS)
    model_device = next(model.parameters()).device
    if device is None:
        device = model_device
    else:
        # If a different device was requested, still prefer the model's device to avoid cross-device ops
        try:
            # Normalize to torch.device for comparison
            req = torch.device(device)
        except Exception:
            req = model_device
        if req != model_device:
            device = model_device

    # 1) Tokenize with offsets to map words -> tokens -> segment token spans
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    offsets = enc["offset_mapping"][0].tolist()
    input_ids = enc["input_ids"][0]
    attention = enc["attention_mask"][0]

    # Spezialtoken-Positionen (CLS/SEP/PAD/…)
    ids_list = input_ids.tolist()
    sp_mask_list = tokenizer.get_special_tokens_mask(
        ids_list, already_has_special_tokens=True
    )
    special_positions = torch.tensor(sp_mask_list, dtype=torch.bool).unsqueeze(0)

    # PAD-ID (kann None sein, je nach Tokenizer)
    pad_id = getattr(tokenizer, "pad_token_id", None)

    # Build token->word index mapping
    # We reconstruct word boundaries by splitting on whitespace in the original text.
    words = text.split()
    # Build cumulative char positions of word starts
    word_starts = []
    pos = 0
    for i, w in enumerate(words):
        word_starts.append(pos)
        pos += len(w) + 1  # +1 for the space

    # Function: for a (start,end) char span, find word id by lower_bound on word_starts
    def char_to_word_id(char_pos: int) -> int:
        # largest i s.t. word_starts[i] <= char_pos
        # linear scan is fine for <=512; if needed, bisect can be used
        last = 0
        for i, ws in enumerate(word_starts):
            if ws <= char_pos:
                last = i
            else:
                break
        return last

    token_word_ids = []
    for (s, e) in offsets:
        if s == e == 0:  # special tokens often have (0,0)
            token_word_ids.append(None)
        else:
            token_word_ids.append(char_to_word_id(s))

    # 2) Map word_segments -> token spans per segment
    seg_token_spans: Dict[str, List[Tuple[int, int]]] = {}
    for seg_name, (w_start, w_end) in word_segments.items():
        # Collect contiguous token runs whose token_word_ids lie in [w_start, w_end)
        runs = []
        run_start = None
        for tidx, wid in enumerate(token_word_ids):
            if wid is None:
                continue
            inside = (w_start <= wid < w_end)
            if inside and run_start is None:
                run_start = tidx
            elif not inside and run_start is not None:
                runs.append((run_start, tidx))
                run_start = None
        if run_start is not None:
            runs.append((run_start, len(token_word_ids)))
        if runs:
            seg_token_spans[seg_name] = runs

    seg = SegmentTokenSpans(seg_token_spans)
    seg_names = [n for n in seg.segment_names() if seg.tokens_for(n)]
    if len(seg_names) == 0:
        return {}, []

    # 3) Precompute factorial weights for Shapley
    k = len(seg_names)
    fact = [math.factorial(i) for i in range(k + 1)]
    def shap_weight(s: int) -> float:
        # For subsets S of size s not containing feature i:
        # w = s! (k-s-1)! / k!
        return fact[s] * fact[k - s - 1] / fact[k]

    # 4) Evaluate all coalitions (forward passes)
    # We’ll cache v(S) for each subset S by its bitmask index.
    name_to_idx = {n: i for i, n in enumerate(seg_names)}
    idx_to_name = {i: n for n, i in name_to_idx.items()}

    # Build a quick lookup from bitmask->value and from coalition tuple->bitmask
    coalition_values = {}
    # Base inputs (batch=1)
    # Führe alle relevanten Inputs mit (BERT-Paar-Setups nutzen token_type_ids)
    allowed = {"input_ids", "attention_mask", "token_type_ids"}
    base_inputs = {k: v.clone() for k, v in enc.items() if k in allowed}

    # Enumerate all subsets once; batch in small groups to speed up
    all_subsets = _enumerate_coalitions(seg_names)
    batch_size = 16
    for i in range(0, len(all_subsets), batch_size):
        chunk = all_subsets[i:i + batch_size]
        batch_inputs = []
        for keep_tuple in chunk:
            masked = _mask_segments_in_inputs(
                base_inputs, seg.spans, keep_tuple,
                special_positions=special_positions,  # bool [1, L]
                pad_token_id=pad_id                   # optional; wenn du ohne PAD willst: None
            )
            batch_inputs.append(masked)

        # Collate into tensors (same length, bs=len(chunk))
        # We only need input_ids and attention_mask for HF BERT
        input_ids_batch = torch.cat([bi["input_ids"] for bi in batch_inputs], dim=0)
        attn_batch = torch.cat([bi["attention_mask"] for bi in batch_inputs], dim=0)

        # optional drittes Tensor
        tt_ids_batch = None
        if "token_type_ids" in batch_inputs[0]:
            tt_ids_batch = torch.cat([bi["token_type_ids"] for bi in batch_inputs], dim=0)

        model.eval()
        with torch.inference_mode():
            if tt_ids_batch is not None:
                out = model(
                    input_ids=input_ids_batch.to(device),
                    attention_mask=attn_batch.to(device),
                    token_type_ids=tt_ids_batch.to(device),
                )
            else:
                out = model(
                    input_ids=input_ids_batch.to(device),
                    attention_mask=attn_batch.to(device),
                )
            logits = out.logits[:, label_index].detach().cpu()  # shape [B]
        for j, keep_tuple in enumerate(chunk):
            coalition_values[tuple(sorted(keep_tuple))] = float(logits[j].item())

    # 5) Compute exact Shapley per segment
    shap_values = {n: 0.0 for n in seg_names}
    for i_name in seg_names:
        phi = 0.0
        for S in all_subsets:
            if i_name in S:
                continue
            s_size = len(S)
            w = shap_weight(s_size)
            v_S = coalition_values[tuple(sorted(S))]
            S_plus = tuple(sorted(S + (i_name,)))
            v_S_plus = coalition_values[S_plus]
            phi += w * (v_S_plus - v_S)
        shap_values[i_name] = phi

    v_empty = coalition_values[tuple()]
    v_full  = coalition_values[tuple(sorted(seg_names))]
    total_phi = sum(shap_values.values())
    diff = (v_full - v_empty) - total_phi
    assert abs(diff) < 1e-4 + 1e-6 * abs(v_full), "Additivity violated ..."

    return shap_values, seg_names


# -------- GLOBAL AGGREGATION --------

def aggregate_global_shap(
    per_sample_shap: List[Dict[str, float]],
    segment_names: Optional[List[str]] = None,
    use_abs: bool = True,
) -> Dict[str, float]:
    """
    Aggregate SHAP across samples to a global importance per segment.
    If segment_names is provided, missing segments in a sample contribute 0.
    """
    if segment_names is None:
        names = sorted({k for d in per_sample_shap for k in d.keys()})
    else:
        names = segment_names

    acc = {n: [] for n in names}
    for d in per_sample_shap:
        for n in names:
            v = d.get(n, 0.0)
            acc[n].append(abs(v) if use_abs else v)

    # Mean; you may switch to median if you expect outliers
    return {n: float(sum(vals) / max(1, len(vals))) for n, vals in acc.items()}