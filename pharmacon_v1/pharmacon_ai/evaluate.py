"""
Evaluation utilities for the Pharmacon writer-adaptation experiment.

Public interface:
    cer(reference, hypothesis)        -> float  (0.0–1.0+)
    wer(reference, hypothesis)        -> float  (0.0–1.0+)
    run_predictions(model_or_peft,
                    processor, crops) -> {line_id: str}
    evaluate_condition(predictions,
                       ground_truth)  -> {line_id: {cer, wer}} + "macro"

Intentionally minimal — computes only what the protocol requires for the
Generic vs Adapted comparison on sealed T01–T08 lines.
No WSCMC, no structured extraction (those are Phase 2).
"""

from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# Levenshtein edit distance (pure Python, no extra deps)
# ---------------------------------------------------------------------------

def _levenshtein(a: list, b: list) -> int:
    """Edit distance between two sequences (characters or words)."""
    m, n = len(a), len(b)
    # Use a single row DP for memory efficiency
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1],   # substitution
                                  prev[j],         # deletion
                                  curr[j - 1])     # insertion
        prev = curr
    return prev[n]


# ---------------------------------------------------------------------------
# CER / WER
# ---------------------------------------------------------------------------

def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate.

    CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars).
    Returns 0.0 if reference is empty.
    """
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate.

    WER = edit_distance(ref_words, hyp_words) / len(ref_words).
    Returns 0.0 if reference is empty.
    """
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


# ---------------------------------------------------------------------------
# Prediction runner
# ---------------------------------------------------------------------------

def run_predictions(model, processor, crops: dict,
                    device=None, **generate_kwargs) -> dict[str, str]:
    """Run TrOCR (base or PEFT) on a dict of line crops.

    Args:
        model:     VisionEncoderDecoderModel or PeftModel.
        processor: TrOCRProcessor with XLMRobertaTokenizer.
        crops:     {line_id: BGR ndarray OR PIL Image}.
        device:    torch.device (auto-detected if None).
        **generate_kwargs: Overrides for model.generate (defaults to beam search).

    Returns:
        {line_id: predicted_text}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    gen_params = {
        "max_new_tokens": 64,
        "num_beams": 4,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.3,
    }
    gen_params.update(generate_kwargs)

    predictions = {}
    with torch.no_grad():
        for line_id, image in crops.items():
            pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
            ids = model.generate(pv, **gen_params)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            predictions[line_id] = text

    return predictions


# ---------------------------------------------------------------------------
# Evaluation scorer
# ---------------------------------------------------------------------------

def evaluate_condition(predictions: dict[str, str],
                       ground_truth: dict[str, str]) -> dict:
    """Compute per-line CER/WER and macro average.

    Args:
        predictions:  {line_id: predicted_text}
        ground_truth: {line_id: reference_text}

    Returns:
        {
          "E01": {"cer": 0.12, "wer": 0.20, "ref": "...", "hyp": "..."},
          ...
          "macro": {"cer": 0.15, "wer": 0.25, "n": 8}
        }
    """
    results = {}
    cer_vals, wer_vals = [], []

    for lid, ref in ground_truth.items():
        hyp = predictions.get(lid, "")
        c = cer(ref, hyp)
        w = wer(ref, hyp)
        results[lid] = {"cer": c, "wer": w, "ref": ref, "hyp": hyp}
        cer_vals.append(c)
        wer_vals.append(w)

    if cer_vals:
        results["macro"] = {
            "cer": sum(cer_vals) / len(cer_vals),
            "wer": sum(wer_vals) / len(wer_vals),
            "n": len(cer_vals),
        }
    return results


# ---------------------------------------------------------------------------
# Pretty-print comparison
# ---------------------------------------------------------------------------

def print_comparison(generic_results: dict, adapted_results: dict,
                     label_a: str = "Generic",
                     label_b: str = "Adapted") -> None:
    """Print a side-by-side CER/WER comparison table to stdout."""
    all_ids = sorted(k for k in generic_results if k != "macro")

    header = f"{'Line':<6}  {label_a+' CER':>12}  {label_a+' WER':>12}  " \
             f"{label_b+' CER':>12}  {label_b+' WER':>12}  {'ΔCER':>8}  {'ΔWER':>8}"
    print(header)
    print("-" * len(header))

    for lid in all_ids:
        g = generic_results.get(lid, {})
        a = adapted_results.get(lid, {})
        gc, gw = g.get("cer", float("nan")), g.get("wer", float("nan"))
        ac, aw = a.get("cer", float("nan")), a.get("wer", float("nan"))
        delta_c = ac - gc
        delta_w = aw - gw
        print(f"{lid:<6}  {gc:>12.4f}  {gw:>12.4f}  "
              f"{ac:>12.4f}  {aw:>12.4f}  "
              f"{delta_c:>+8.4f}  {delta_w:>+8.4f}")

    print("-" * len(header))
    gm = generic_results.get("macro", {})
    am = adapted_results.get("macro", {})
    gc_m, gw_m = gm.get("cer", float("nan")), gm.get("wer", float("nan"))
    ac_m, aw_m = am.get("cer", float("nan")), am.get("wer", float("nan"))
    print(f"{'MACRO':<6}  {gc_m:>12.4f}  {gw_m:>12.4f}  "
          f"{ac_m:>12.4f}  {aw_m:>12.4f}  "
          f"{ac_m - gc_m:>+8.4f}  {aw_m - gw_m:>+8.4f}")
    print()
    improvement = "IMPROVED" if ac_m < gc_m else "NO IMPROVEMENT"
    print(f">> Macro CER: {gc_m:.4f} → {ac_m:.4f}  [{improvement}]")
    print(f">> Macro WER: {gw_m:.4f} → {aw_m:.4f}")


def print_three_condition_comparison(
    cond_a_results: dict,
    cond_b_results: dict,
    cond_c_results: dict,
    writer_id: str = "W01"
) -> None:
    """Print the 3-condition benchmark table mandated by Protocol Page 12.
    
    Conditions:
      A. Generic (Zero-shot)
      B. Adapted (E01-E16)
      C. Adapted + Active (E01-E16 + 4 selected R lines)
    """
    all_ids = sorted(k for k in cond_a_results if k != "macro")

    header = (
        f"Line    {'[A] Generic':<17} {'[B] Adapted (E16)':<19} {'[C] Active (WSCMC)':<19} {'Net ΔCER (C-A)':<14}\n"
        f"        {'CER':>7} {'WER':>8}   {'CER':>7} {'WER':>9}   {'CER':>7} {'WER':>9}"
    )
    print("=" * 82)
    print(f"PHARMACON BENCHMARK: 3-CONDITION EVALUATION [{writer_id}] (ON SEALED T01-T08)")
    print("=" * 82)
    print(header)
    print("-" * 82)

    for lid in all_ids:
        a = cond_a_results.get(lid, {})
        b = cond_b_results.get(lid, {})
        c = cond_c_results.get(lid, {})

        ac, aw = a.get("cer", float("nan")), a.get("wer", float("nan"))
        bc, bw = b.get("cer", float("nan")), b.get("wer", float("nan"))
        cc, cw = c.get("cer", float("nan")), c.get("wer", float("nan"))

        delta = cc - ac
        print(
            f"{lid:<6}  {ac:>7.4f} {aw:>8.4f}   "
            f"{bc:>7.4f} {bw:>9.4f}   "
            f"{cc:>7.4f} {cw:>9.4f}   "
            f"{delta:>+12.4f}"
        )

    print("-" * 82)
    am = cond_a_results.get("macro", {})
    bm = cond_b_results.get("macro", {})
    cm = cond_c_results.get("macro", {})

    am_c, am_w = am.get("cer", float("nan")), am.get("wer", float("nan"))
    bm_c, bm_w = bm.get("cer", float("nan")), bm.get("wer", float("nan"))
    cm_c, cm_w = cm.get("cer", float("nan")), cm.get("wer", float("nan"))

    net_delta = cm_c - am_c
    print(
        f"{'MACRO':<6}  {am_c:>7.4f} {am_w:>8.4f}   "
        f"{bm_c:>7.4f} {bm_w:>9.4f}   "
        f"{cm_c:>7.4f} {cm_w:>9.4f}   "
        f"{net_delta:>+12.4f}"
    )
    print("=" * 82)
    rel_gain = ((am_c - cm_c) / am_c * 100) if am_c > 0 else 0.0
    print(f">> Final Macro CER: {am_c:.4f} → {bm_c:.4f} → {cm_c:.4f} ({rel_gain:+.1f}% error reduction)")
    print(f">> Final Macro WER: {am_w:.4f} → {bm_w:.4f} → {cm_w:.4f}")


