"""
Post-hoc Formulary Correction Engine (Stage 5b).

Decouples the formulary matching step from the beam-search constraint step.
Instead of restricting generation mid-decode (which collapses to training-set
priors), this module corrects the raw autoregressive output *after* generation.

Pipeline:
    raw_ocr_text  ->  entity_parser  ->  drug_token_extractor  ->
    FormularyMatcher  ->  top_k ranked corrections  +  safety gate

Key difference from constrained_decoding.py:
    constrained_decoding.py: forces the decoder to only emit valid Trie paths
    during generation. Side-effect: collapses to highest-frequency prior drug.

    posthoc.py: lets the decoder hallucinate freely, then corrects the output
    using multi-metric fuzzy matching. The decoder can express what the visual
    encoder actually saw (e.g. 'Implapine'), and the corrector maps it to the
    nearest real drug (e.g. 'Quetiapine').

This module is intentionally pure-Python (no torch dependency) so it can be
run offline, on CPU, or as a post-processing step in the pharmacist dashboard.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pharmacon_ai.formulary import (
    FormularyMatcher,
    extract_prescription_entities,
    levenshtein_similarity,
    jaro_winkler_similarity,
    consonant_skeleton,
    _levenshtein_distance,
)
from pharmacon_ai.evaluate import cer as compute_cer


# ---------------------------------------------------------------------------
# Drug-token extractor: pull the most drug-name-like token from hallucinated text
# ---------------------------------------------------------------------------

# Drug name heuristics: capitalised word(s) after a dosage-form prefix,
# or any capitalised word longer than 4 chars that is NOT a common English word.
_FORM_PREFIX_RE = re.compile(
    r"\b(tab|cap|syp|inj|eye\s+drops?|ear\s+drops?|inhaler|drops?|oint|cream|gel|sachet)\b",
    re.IGNORECASE,
)

_COMMON_ENGLISH = {
    "the", "and", "for", "was", "with", "from", "that", "this", "have", "been",
    "which", "their", "there", "were", "they", "what", "when", "will", "more",
    "also", "into", "than", "then", "some", "after", "about", "before",
    "april", "october", "january", "february", "march", "june", "july",
    "august", "september", "november", "december",
    "morning", "evening", "night", "daily", "days", "months", "weeks",
    "inches", "only", "forty", "twice", "once", "three", "four", "five",
    "useful", "moreness", "doubles", "deep", "thirty",
}


def extract_drug_tokens(raw_text: str) -> str:
    """
    Pull the most drug-name-like substring from a raw (potentially hallucinated)
    OCR string.

    Strategy (in priority order):
    1. If a dosage-form prefix (Tab, Cap, Syp, etc.) is found, take the
       immediately following capitalised word as the drug candidate.
    2. Otherwise take the longest capitalised token that isn't a common English word.
    3. Fall back to the full residual text after stripping numbers and punctuation.
    """
    text = raw_text.strip()

    # Strategy 1: word after form prefix
    m = _FORM_PREFIX_RE.search(text)
    if m:
        after = text[m.end():].strip()
        # Take first capitalised token (could be multi-word like 'Amoxicillin')
        word_match = re.match(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)", after)
        if word_match:
            candidate = word_match.group(1).strip()
            if len(candidate) > 3:
                return candidate

    # Strategy 2: longest capitalised non-English token
    caps_tokens = re.findall(r"\b([A-Z][a-zA-Z]{3,})\b", text)
    non_english = [t for t in caps_tokens if t.lower() not in _COMMON_ENGLISH]
    if non_english:
        return max(non_english, key=len)

    # Strategy 3: full residual after stripping numerics and punctuation
    residual = re.sub(r"[0-9;:,./()\-—]", " ", text)
    residual = " ".join(residual.split())
    return residual


# ---------------------------------------------------------------------------
# Post-hoc corrector
# ---------------------------------------------------------------------------

class PosthocCorrector:
    """
    Apply formulary fuzzy correction to a raw OCR string without constraining
    the decoder during generation.
    """

    def __init__(self, formulary_path: Optional[str] = None):
        self._matcher = FormularyMatcher(formulary_path=formulary_path)

    def correct(
        self,
        raw_ocr: str,
        top_k: int = 3,
        confidence_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Correct a single raw OCR string.

        Returns:
            {
              "raw_ocr": str,
              "drug_token_extracted": str,
              "top_candidates": [
                  {"generic_name": str, "confidence": float, "name_score": float,
                   "skel_score": float, "jw_score": float},
                  ...
              ],
              "best_match": str | None,
              "best_confidence": float,
              "corrected_line": str | None,  # reconstructed prescription line
              "abstain": bool,
            }
        """
        drug_token = extract_drug_tokens(raw_ocr)

        # Score every formulary entry directly against the extracted token
        q_lower = drug_token.lower()
        q_skel = consonant_skeleton(drug_token)

        scored = []
        for entry in self._matcher.entries:
            best_score = 0.0
            best_name = entry["generic_name"]

            for alias in entry["_aliases"]:
                lev = levenshtein_similarity(q_lower, alias)
                jw = jaro_winkler_similarity(q_lower, alias)
                skel = levenshtein_similarity(q_skel, consonant_skeleton(alias))
                sub_bonus = 0.15 if (q_lower in alias or alias in q_lower) and len(q_lower) >= 4 else 0.0

                score = min(1.0, (0.45 * jw) + (0.35 * lev) + (0.20 * skel) + sub_bonus)
                if score > best_score:
                    best_score = score
                    best_name = entry["generic_name"]

            scored.append({
                "generic_name": entry["generic_name"],
                "confidence": round(best_score, 4),
                "category": entry.get("category", ""),
                "allowed_forms": entry.get("allowed_forms", []),
                "allowed_strengths": entry.get("allowed_strengths", []),
            })

        scored.sort(key=lambda x: -x["confidence"])
        top = scored[:top_k]

        best = top[0] if top else None
        best_conf = best["confidence"] if best else 0.0
        abstain = best_conf < confidence_threshold

        # Try to reconstruct a plausible corrected line using entity parser
        corrected_line = None
        if best and not abstain:
            entities = extract_prescription_entities(raw_ocr)
            form = entities.get("form") or ""
            strength = entities.get("strength") or ""
            freq = entities.get("frequency") or ""
            duration = entities.get("duration") or ""
            parts = [p for p in [form, best["generic_name"], strength, freq,
                                  (f"for {duration}" if duration else "")] if p]
            corrected_line = " ".join(parts).strip()

        return {
            "raw_ocr": raw_ocr,
            "drug_token_extracted": drug_token,
            "top_candidates": top,
            "best_match": best["generic_name"] if best else None,
            "best_confidence": best_conf,
            "corrected_line": corrected_line,
            "abstain": abstain,
        }

    def correct_batch(
        self,
        predictions: Dict[str, str],
        ground_truth: Optional[Dict[str, str]] = None,
        top_k: int = 3,
        confidence_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Run post-hoc correction on a batch of {line_id: raw_ocr} predictions.
        Optionally compares corrected drug name against ground truth drug name
        and computes CER improvement.

        Returns per-line results + macro summary.
        """
        results = {}
        cer_raw_vals, cer_corrected_vals = [], []
        name_hits_raw, name_hits_corrected = 0, 0
        n = 0

        for line_id, raw_ocr in predictions.items():
            result = self.correct(raw_ocr, top_k=top_k,
                                  confidence_threshold=confidence_threshold)

            ref = ground_truth.get(line_id, "") if ground_truth else ""
            cer_raw = compute_cer(ref, raw_ocr) if ref else None
            cer_corrected = compute_cer(ref, result["corrected_line"] or raw_ocr) if ref else None

            # Semantic drug name hit: did the best_match appear in the reference?
            raw_drug_tok = extract_drug_tokens(raw_ocr)
            best_match = result["best_match"] or ""

            raw_name_hit = (
                raw_drug_tok.lower() in ref.lower() or ref.lower() in raw_drug_tok.lower()
            ) if ref else None

            corrected_name_hit = (
                best_match.lower() in ref.lower()
            ) if ref and best_match else None

            result["eval"] = {
                "reference": ref,
                "cer_raw": round(cer_raw, 4) if cer_raw is not None else None,
                "cer_corrected": round(cer_corrected, 4) if cer_corrected is not None else None,
                "cer_delta": round(cer_corrected - cer_raw, 4) if (
                    cer_raw is not None and cer_corrected is not None) else None,
                "raw_drug_name_hit": raw_name_hit,
                "corrected_drug_name_hit": corrected_name_hit,
            }

            if cer_raw is not None:
                cer_raw_vals.append(cer_raw)
                cer_corrected_vals.append(cer_corrected)
            if raw_name_hit is not None:
                name_hits_raw += int(raw_name_hit)
                name_hits_corrected += int(corrected_name_hit or False)
                n += 1

            results[line_id] = result

        if cer_raw_vals:
            results["macro"] = {
                "n": len(cer_raw_vals),
                "macro_cer_raw": round(sum(cer_raw_vals) / len(cer_raw_vals), 4),
                "macro_cer_corrected": round(sum(cer_corrected_vals) / len(cer_corrected_vals), 4),
                "macro_cer_delta": round(
                    sum(cer_corrected_vals) / len(cer_corrected_vals)
                    - sum(cer_raw_vals) / len(cer_raw_vals), 4
                ),
                "semantic_hit_rate_raw": round(name_hits_raw / n, 4) if n else None,
                "semantic_hit_rate_corrected": round(name_hits_corrected / n, 4) if n else None,
            }

        return results


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_posthoc_report(results: Dict[str, Any]) -> None:
    """Print a formatted post-hoc correction benchmark report."""
    macro = results.get("macro", {})
    line_ids = sorted(k for k in results if k != "macro")

    print("=" * 100)
    print("PHARMACON — POST-HOC FORMULARY CORRECTION BENCHMARK")
    print("=" * 100)
    print(f"{'Line':<6}  {'Raw OCR (truncated)':<40}  {'Drug Token':<18}  "
          f"{'Best Match':<20}  {'Conf':>6}  {'CER Raw':>8}  {'CER Corr':>9}  {'ΔCER':>7}  "
          f"{'Raw Hit':>8}  {'Corr Hit':>9}")
    print("-" * 100)

    for lid in line_ids:
        r = results[lid]
        ev = r.get("eval", {})
        raw_trunc = (r["raw_ocr"][:38] + "..") if len(r["raw_ocr"]) > 40 else r["raw_ocr"]
        tok = (r["drug_token_extracted"][:16] + "..") if len(r["drug_token_extracted"]) > 18 else r["drug_token_extracted"]
        best = r["best_match"] or "ABSTAIN"
        conf = r["best_confidence"]
        cer_r = f"{ev['cer_raw']:.4f}" if ev.get("cer_raw") is not None else "  N/A  "
        cer_c = f"{ev['cer_corrected']:.4f}" if ev.get("cer_corrected") is not None else "  N/A  "
        delta = f"{ev['cer_delta']:+.4f}" if ev.get("cer_delta") is not None else "  N/A "
        rh = "YES" if ev.get("raw_drug_name_hit") else "no"
        ch = "YES" if ev.get("corrected_drug_name_hit") else "no"
        print(f"{lid:<6}  {raw_trunc:<40}  {tok:<18}  {best:<20}  {conf:>6.4f}  "
              f"{cer_r:>8}  {cer_c:>9}  {delta:>7}  {rh:>8}  {ch:>9}")

    print("=" * 100)
    if macro:
        print(f"MACRO   CER: {macro['macro_cer_raw']:.4f} (raw) → {macro['macro_cer_corrected']:.4f} (corrected)"
              f"  [{macro['macro_cer_delta']:+.4f}]")
        print(f"        Semantic Drug Name Hit Rate:  "
              f"raw={macro['semantic_hit_rate_raw']:.1%}  "
              f"corrected={macro['semantic_hit_rate_corrected']:.1%}")
    print("=" * 100)
