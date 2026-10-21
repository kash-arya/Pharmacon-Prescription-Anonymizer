"""
Writer-Specific Confusion-Matrix Calibration (WSCMC).

Implements Phase 4 / Section 3-4 of Pharmacon_Writer_Calibration_Protocol.pdf:
  1. Levenshtein character alignment between baseline predictions and ground truth on E01-E16.
  2. Character substitution, insertion, and deletion accounting -> confusion probability profile.
  3. Weighted coverage scoring for active refinement prompt selection:
       - Weak-character coverage (weight: 0.45)
       - Confusion-pair coverage (weight: 0.35)
       - Medical-token coverage (weight: 0.20)
  4. Greedy active prompt selection from candidate lines (R01-R04).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


def align_characters(ref: str, hyp: str) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Produce character-level Levenshtein alignment between reference and hypothesis.
    Returns list of (ref_char, hyp_char) pairs.
    (char, None) = deletion, (None, char) = insertion, (c1, c2) = match or substitution.
    """
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    # Backtrace
    i, j = m, n
    alignment = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and (dp[i][j] == dp[i - 1][j - 1] and ref[i - 1] == hyp[j - 1] or
                                dp[i][j] == dp[i - 1][j - 1] + 1 and ref[i - 1] != hyp[j - 1]):
            alignment.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((ref[i - 1], None))
            i -= 1
        else:
            alignment.append((None, hyp[j - 1]))
            j -= 1

    alignment.reverse()
    return alignment


class WriterConfusionProfile:
    """
    Builds character confusion statistics and selects optimal refinement prompts.
    """

    # Common ambiguous handwriting pairs from clinical prescription literature
    FREQUENT_PAIRS = [
        ("m", "n"), ("u", "v"), ("0", "o"), ("0", "O"),
        ("1", "l"), ("1", "I"), ("c", "e"), ("a", "o"),
        ("r", "v"), ("f", "t"), ("d", "cl"), ("g", "q")
    ]

    # Clinical dosage forms, units, frequencies, and administration routes
    MEDICAL_TOKENS = [
        "tab", "cap", "syp", "inj", "drops", "ointment", "inhaler", "ors",
        "mg", "mcg", "g", "ml", "iu", "%",
        "1-0-1", "1-0-0", "0-0-1", "od", "bd", "tds", "qid", "sos", "hs", "ac", "pc",
        "oral", "sc", "im", "iv", "topical",
    ]

    def __init__(self):
        self.truth_counts = defaultdict(int)
        self.substitutions = defaultdict(lambda: defaultdict(int))
        self.deletions = defaultdict(int)
        self.insertions = defaultdict(int)
        self.char_error_rates: Dict[str, float] = {}

    def update_from_enrollment(
        self,
        predictions: Dict[str, str],
        ground_truth: Dict[str, str]
    ) -> None:
        """
        Ingest predictions on E01-E16 and calculate normalized confusion profile.
        """
        for lid, ref in ground_truth.items():
            hyp = predictions.get(lid, "")
            aligned = align_characters(ref, hyp)

            for ref_c, hyp_c in aligned:
                if ref_c is not None:
                    self.truth_counts[ref_c] += 1

                if ref_c is not None and hyp_c is not None:
                    if ref_c != hyp_c:
                        self.substitutions[ref_c][hyp_c] += 1
                elif ref_c is not None and hyp_c is None:
                    self.deletions[ref_c] += 1
                elif ref_c is None and hyp_c is not None:
                    self.insertions[hyp_c] += 1

        # Calculate per-character error rate
        for ch, count in self.truth_counts.items():
            errs = sum(self.substitutions[ch].values()) + self.deletions[ch]
            self.char_error_rates[ch] = errs / count if count > 0 else 0.0

    def get_frequent_confusions(self, min_count: int = 2) -> List[Dict[str, Any]]:
        """Return substitutions observed at least min_count times."""
        confusions = []
        for ref_c, hyp_dict in self.substitutions.items():
            total = self.truth_counts.get(ref_c, 1)
            for hyp_c, count in hyp_dict.items():
                if count >= min_count:
                    confusions.append({
                        "truth": ref_c,
                        "predicted": hyp_c,
                        "count": count,
                        "prob": count / total,
                        "status": "confirmed"
                    })
                else:
                    confusions.append({
                        "truth": ref_c,
                        "predicted": hyp_c,
                        "count": count,
                        "prob": count / total,
                        "status": "provisional"
                    })
        confusions.sort(key=lambda x: -x["count"])
        return confusions

    def score_candidate(
        self,
        sentence: str,
        covered_chars: Set[str],
        covered_tokens: Set[str]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score a candidate sentence using the protocol's weighted coverage:
          - Weak-character coverage (0.45)
          - Confusion-pair coverage (0.35)
          - Medical-token coverage (0.20)
        """
        s_lower = sentence.lower()
        chars_in_s = set(sentence)

        # 1. Weak-character coverage: sum error rate of newly covered weak characters
        weak_char_sum = 0.0
        for ch in chars_in_s:
            if ch not in covered_chars:
                weak_char_sum += self.char_error_rates.get(ch, 0.0)

        # 2. Confusion-pair coverage: reward containing both sides of frequent pairs
        pair_reward = 0.0
        for c1, c2 in self.FREQUENT_PAIRS:
            if c1 in s_lower and c2 in s_lower:
                pair_reward += 1.0

        # 3. Medical-token coverage: reward newly covered dosage forms/units/frequencies
        med_reward = 0.0
        for token in self.MEDICAL_TOKENS:
            if token not in covered_tokens and re.search(r"\b" + re.escape(token) + r"\b", s_lower):
                med_reward += 1.0

        # Normalize components roughly into [0, 1] range
        weak_score = min(1.0, weak_char_sum / 3.0)
        pair_score = min(1.0, pair_reward / 2.0)
        med_score = min(1.0, med_reward / 3.0)

        total_score = (0.45 * weak_score) + (0.35 * pair_score) + (0.20 * med_score)

        breakdown = {
            "weak_char_score": weak_score,
            "pair_score": pair_score,
            "med_score": med_score,
            "total_score": round(total_score, 4)
        }
        return total_score, breakdown

    def select_refinement_lines(
        self,
        candidate_pool: Dict[str, str],
        k: int = 4
    ) -> List[Tuple[str, str, Dict[str, float]]]:
        """
        Greedily select k refinement lines from candidate_pool.
        Returns list of (line_id, sentence, breakdown).
        """
        selected = []
        covered_chars: Set[str] = set()
        covered_tokens: Set[str] = set()

        remaining = dict(candidate_pool)

        for _ in range(min(k, len(candidate_pool))):
            best_id = None
            best_score = -1.0
            best_breakdown = {}

            for lid, sent in remaining.items():
                score, breakdown = self.score_candidate(sent, covered_chars, covered_tokens)
                if score > best_score:
                    best_score = score
                    best_id = lid
                    best_breakdown = breakdown

            if best_id is None:
                break

            chosen_sent = remaining.pop(best_id)
            selected.append((best_id, chosen_sent, best_breakdown))

            # Update coverage
            covered_chars.update(set(chosen_sent))
            for token in self.MEDICAL_TOKENS:
                if re.search(r"\b" + re.escape(token) + r"\b", chosen_sent.lower()):
                    covered_tokens.add(token)

        return selected
