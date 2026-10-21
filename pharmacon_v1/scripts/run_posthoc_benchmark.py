"""
Phase 5: Post-hoc Formulary Correction Benchmark.

Applies the PosthocCorrector to the known raw LoRA (Phase 3) predictions for
Writer W01 sealed test lines T01–T08.

This tests the hypothesis:
    "Does letting the decoder hallucinate freely and then correcting post-hoc
    outperform beam-constrained decoding (Phase 2) and uncorrected LoRA (Phase 3)?"

Run from workspace root:
    python scripts/run_posthoc_benchmark.py
"""

import sys
import os

# Allow running from workspace root without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pharmacon_ai.posthoc import PosthocCorrector, print_posthoc_report

# ---------------------------------------------------------------------------
# Phase 3 raw LoRA predictions — Writer W01, T01–T08
# Taken verbatim from docs/PHARMACON_EMPIRICAL_RESEARCH_DEFENSE.md, Phase 3
# Lines T03 and T04 were not reported in the defence doc; marked as unavailable.
# ---------------------------------------------------------------------------
PHASE3_RAW_PREDICTIONS = {
    "T01": "Tab Ammfochlorin 100 mg/p, useful, moreness for 589 lb/day",
    "T02": "In April 5 October 20 there was forty days for 5 days.",
    # T03 not reported in Phase 3 results (only Phase 4 entry available)
    "T05": "Tab Implapine 500 mg. 1 inch, doubles, deep for the day.",
    "T06": "Tab Appinfochlorus 5 mg, 5 mg. Only forty days.",
    "T07": "Tab Identini 50 mg, 1 inches/day, that for 10 days.",
}

# Phase 2 constrained beam search predictions (for comparison)
PHASE2_CONSTRAINED_PREDICTIONS = {
    "T01": "Inj Insulin 100 IU/mL, OD for 5 days",
    "T02": "Tab Amlodipine 10 mg, OD for 7 days",
    "T05": "Inhaler Salbutamol 100 mcg, TDS for 5 days",
    "T06": "Tab Ibuprofen 400 mg, TDS for 5 days",
    "T07": None,  # not reported for Phase 2 T07
}

# Phase 4 (LoRA + Constrained) predictions (all collapsed to Amlodipine)
PHASE4_CONSTRAINED_LORA_PREDICTIONS = {
    "T01": "Tab Amlodipine 5 mg, 1-0-0 every morning.",
    "T02": "Tab Amlodipine 5 mg, 1-0-0 every morning.",
    "T05": "Tab Amlodipine 5 mg, 1-0-0 every morning.",
    "T06": "Tab Amlodipine 5 mg, TDS for 3 days",
    "T07": "Tab Amlodipine 10 mg, TDS for 5 days",
}

# Ground truth — Writer W01 sealed test lines T01–T08
GROUND_TRUTH = {
    "T01": "Tab Paracetamol 650 mg SOS after food; maximum 3 tablets/day.",
    "T02": "Syp Amoxicillin 250 mg/5 mL, 5 mL TDS for 5 days.",
    "T05": "Tab Quetiapine 25 mg, 0-0-1 at bedtime for 14 days.",
    "T06": "Tab Clopidogrel 75 mg, one tablet OD after lunch.",
    "T07": "Cap Pregabalin 75 mg, one capsule HS for 10 days.",
}


def run_comparison(corrector: PosthocCorrector):
    from pharmacon_ai.evaluate import cer as compute_cer

    print("\n\n" + "=" * 100)
    print("PHARMACON — PHASE COMPARISON: CER & SEMANTIC HIT RATE ACROSS ALL STRATEGIES")
    print("=" * 100)
    print(f"{'Line':<6}  {'Ref Drug':<16}  "
          f"{'Ph2 Beam CER':>13}  {'Ph3 LoRA CER':>13}  "
          f"{'Ph4 LoRA+Beam CER':>18}  {'Ph5 PostHoc CER':>16}  "
          f"{'Ph3 Drug Hit':>13}  {'Ph5 Drug Hit':>13}")
    print("-" * 100)

    for lid in sorted(GROUND_TRUTH.keys()):
        ref = GROUND_TRUTH[lid]
        ref_drug = next(
            (w for w in ref.split() if w[0].isupper() and len(w) > 4
             and w.lower() not in {"after", "bedtime", "tablet", "capsule", "maximum"}),
            "?"
        )

        raw = PHASE3_RAW_PREDICTIONS.get(lid, "")
        ph2 = PHASE2_CONSTRAINED_PREDICTIONS.get(lid) or ""
        ph4 = PHASE4_CONSTRAINED_LORA_PREDICTIONS.get(lid) or ""

        cer_ph2 = compute_cer(ref, ph2) if ph2 else float("nan")
        cer_ph3 = compute_cer(ref, raw) if raw else float("nan")
        cer_ph4 = compute_cer(ref, ph4) if ph4 else float("nan")

        result = corrector.correct(raw) if raw else {}
        corrected = result.get("corrected_line") or raw
        cer_ph5 = compute_cer(ref, corrected) if corrected else float("nan")

        best_match = result.get("best_match", "")
        ph3_hit = "YES" if raw and extract_drug_tokens_check(raw, ref) else "no"
        ph5_hit = "YES" if best_match and best_match.lower() in ref.lower() else "no"

        print(f"{lid:<6}  {ref_drug:<16}  "
              f"{cer_ph2:>13.4f}  {cer_ph3:>13.4f}  "
              f"{cer_ph4:>18.4f}  {cer_ph5:>16.4f}  "
              f"{ph3_hit:>13}  {ph5_hit:>13}")

    print("=" * 100)


def extract_drug_tokens_check(raw: str, ref: str) -> bool:
    """Quick check: does any capitalised token from raw appear in ref?"""
    import re
    tokens = re.findall(r"\b([A-Z][a-zA-Z]{3,})\b", raw)
    for t in tokens:
        if t.lower() in ref.lower():
            return True
    return False


if __name__ == "__main__":
    print("Loading NLEM 2022 formulary...")
    corrector = PosthocCorrector()
    print(f"Loaded {len(corrector._matcher.entries)} formulary entries.\n")

    # Main post-hoc benchmark on Phase 3 raw predictions
    print("Running post-hoc correction on Phase 3 raw LoRA predictions...")
    results = corrector.correct_batch(
        predictions=PHASE3_RAW_PREDICTIONS,
        ground_truth=GROUND_TRUTH,
        top_k=5,
        confidence_threshold=0.45,
    )
    print_posthoc_report(results)

    # Cross-phase comparison table
    run_comparison(corrector)

    # Detailed per-line breakdown
    print("\n\n" + "=" * 100)
    print("DETAILED PER-LINE DRUG TOKEN ANALYSIS")
    print("=" * 100)
    for lid in sorted(k for k in results if k != "macro"):
        r = results[lid]
        ev = r.get("eval", {})
        print(f"\n[{lid}]")
        print(f"  Raw OCR       : {r['raw_ocr']}")
        print(f"  Drug Token    : {r['drug_token_extracted']}")
        print(f"  Reference     : {ev.get('reference', 'N/A')}")
        print(f"  Top Candidates:")
        for c in r["top_candidates"]:
            print(f"    {c['generic_name']:<30}  conf={c['confidence']:.4f}  [{c['category']}]")
        print(f"  Best Match    : {r['best_match']}  (conf={r['best_confidence']:.4f})")
        print(f"  Corrected Line: {r['corrected_line']}")
        print(f"  CER: raw={ev.get('cer_raw', 'N/A')}  corrected={ev.get('cer_corrected', 'N/A')}  "
              f"delta={ev.get('cer_delta', 'N/A')}")
        print(f"  Drug name hit : raw={ev.get('raw_drug_name_hit')}  "
              f"corrected={ev.get('corrected_drug_name_hit')}")
