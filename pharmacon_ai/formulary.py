"""
Formulary Resolver and Prescription Entity Extractor (Stage 5 & Stage 6).

Implements the clinical safety layer mandated by the Pharmacon protocol:
  - Regex & rule-based entity parsing (Form, Strength, Frequency, Duration, Route).
  - Multi-tier fuzzy matching against NLEM 2022 and essential Indian formularies:
      * Normalized Levenshtein distance
      * Jaro-Winkler prefix-weighted metric
      * Consonant skeleton matching (critical for cursive medical shorthand)
      * Form & strength clinical compatibility verification
  - Decision Support Safety Gate:
      * Top-K ranked candidate drugs with confidence scores.
      * Safe abstention flag when confidence falls below clinical threshold.
      * Mandatory human confirmation requirement before dispensing.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Path to Curated Indian Formulary Database
# ---------------------------------------------------------------------------

DEFAULT_FORMULARY_PATH = os.path.join(
    os.path.dirname(__file__), "data", "nlem_2022.json"
)


# ---------------------------------------------------------------------------
# Core String & Phonetic Metrics (Pure Python, Zero Dependency)
# ---------------------------------------------------------------------------

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized similarity in [0.0, 1.0]. 1.0 = identical."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    dist = _levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    return max(0.0, 1.0 - (dist / max_len))


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """
    Jaro-Winkler similarity giving bonus weight to matching prefixes.
    Crucial for clinical pharmacology where drug stems start identically
    (e.g., 'Paracet...', 'Amoxi...', 'Metfor...').
    """
    s1, s2 = s1.lower(), s2.lower()
    len1, len2 = len(s1), len(s2)
    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = (max(len1, len2) // 2) - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    t = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1
    transpositions = t / 2.0

    jaro = ((matches / len1) + (matches / len2) + ((matches - transpositions) / matches)) / 3.0

    # Winkler prefix bonus (up to 4 matching chars)
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + (prefix * p * (1.0 - jaro))


def consonant_skeleton(text: str) -> str:
    """
    Extract consonant skeleton. In cursive doctor handwriting, vowels often
    collapse into baseline ripples, while ascenders/descenders (consonants)
    remain visually salient. Consecutive duplicate consonants are collapsed.
    Example: 'Paracetamol' -> 'prctml', 'Amoxicillin' -> 'amxcln', 'Metformin' -> 'mtfrmn'
    """
    cleaned = re.sub(r"[^a-zA-Z]", "", text).lower()
    if not cleaned:
        return ""
    first_char = cleaned[0]
    rest = re.sub(r"[aeiouy]", "", cleaned[1:])
    skeleton = first_char + rest
    return re.sub(r"(.)\1+", r"\1", skeleton)


# ---------------------------------------------------------------------------
# Stage 5: Clinical Entity Parser
# ---------------------------------------------------------------------------

KNOWN_FORMS = {
    "tab": "Tab", "tablet": "Tab", "tablets": "Tab",
    "cap": "Cap", "capsule": "Cap", "capsules": "Cap",
    "syp": "Syp", "syrup": "Syp",
    "inj": "Inj", "injection": "Inj",
    "eye drops": "Eye drops", "eyedrops": "Eye drops", "drops": "Drops",
    "ear drops": "Ear drops",
    "inhaler": "Inhaler", "puffs": "Inhaler", "respules": "Respules",
    "oint": "Ointment", "ointment": "Ointment", "cream": "Cream", "gel": "Gel",
    "ors": "ORS", "sachet": "Sachet", "powder": "Powder",
}

FREQUENCY_PATTERNS = [
    (r"\b1-0-1\b", "1-0-1"),
    (r"\b1-0-0\b", "1-0-0"),
    (r"\b0-0-1\b", "0-0-1"),
    (r"\b1-1-1\b", "1-1-1"),
    (r"\b0-1-0\b", "0-1-0"),
    (r"\bOD\b", "OD"),
    (r"\bBD\b", "BD"),
    (r"\bTDS\b", "TDS"),
    (r"\bQID\b", "QID"),
    (r"\bSOS\b", "SOS"),
    (r"\bHS\b", "HS"),
    (r"\bAC\b", "AC"),
    (r"\bPC\b", "PC"),
    (r"\bonce daily\b", "once daily"),
    (r"\bonce weekly\b", "once weekly"),
]

STRENGTH_REGEX = re.compile(
    r"\b(\d+(?:\.\d+)?(?:\s*,\s*\d+)?)\s*(mg/5\s*mL|mg/mL|mcg|mg|IU/mL|IU|g/L|g|%|mL)(?![a-zA-Z0-9])",
    re.IGNORECASE
)

DURATION_REGEX = re.compile(
    r"\b(?:for\s+)?(\d+)\s*(days?|nights?|weeks?|months?)\b",
    re.IGNORECASE
)


def extract_prescription_entities(text: str) -> Dict[str, Any]:
    """
    Extract structured medical entities from raw prescription transcription.

    Returns dict containing:
      - form: dosage form (Tab, Cap, Syp, etc.)
      - strength: numerical strength with unit (e.g., '650 mg', '250 mg/5 mL')
      - frequency: dosage regimen ('1-0-1', 'OD', 'TDS', 'SOS')
      - duration: duration of treatment ('5 days', '8 weeks')
      - instructions: supplementary clinical directions ('after food', 'before breakfast')
      - drug_candidate_tokens: residual text string representing the drug name
    """
    cleaned = text.strip()

    # 1. Extract Form
    form = None
    form_lower = cleaned.lower()
    for pattern, canonical in sorted(KNOWN_FORMS.items(), key=lambda x: -len(x[0])):
        regex = rf"\b{re.escape(pattern)}\b"
        if re.search(regex, form_lower):
            form = canonical
            # Mask out the form keyword
            cleaned = re.sub(regex, " ", cleaned, flags=re.IGNORECASE)
            break

    # 2. Extract Strength
    strength = None
    strength_match = STRENGTH_REGEX.search(cleaned)
    if strength_match:
        val = strength_match.group(1).replace(" ", "")
        unit = strength_match.group(2).replace(" ", "")
        # Normalize unit spacing
        if "5ml" in unit.lower() or "5 mL" in strength_match.group(0):
            strength = f"{val} mg/5 mL"
        elif "%" in unit:
            strength = f"{val}%"
        else:
            strength = f"{val} {unit}"
        cleaned = STRENGTH_REGEX.sub(" ", cleaned, count=1)

    # 3. Extract Frequency
    frequency = None
    for pattern, canonical in FREQUENCY_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            frequency = canonical
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
            break

    # 4. Extract Duration
    duration = None
    duration_match = DURATION_REGEX.search(cleaned)
    if duration_match:
        duration = f"{duration_match.group(1)} {duration_match.group(2).lower()}"
        cleaned = DURATION_REGEX.sub(" ", cleaned, count=1)

    # 5. Extract Common Clinical Instructions
    instructions = []
    instr_patterns = [
        r"after food", r"before breakfast", r"with meals", r"with the evening meal",
        r"at bedtime", r"empty stomach", r"rinse after use", r"apply locally",
        r"maximum \d+ doses?/day", r"maximum \d+ tablets?/day"
    ]
    for ip in instr_patterns:
        m = re.search(ip, cleaned, re.IGNORECASE)
        if m:
            instructions.append(m.group(0).strip())
            cleaned = re.sub(ip, " ", cleaned, flags=re.IGNORECASE)

    # 6. Residual Candidate Tokens (Drug Name Target)
    residual = re.sub(r"^[A-Z]\d{2}[.:]?\s*", "", cleaned)
    residual = re.sub(r"[;:,.\-—()\/]", " ", residual)
    residual = " ".join(residual.split())

    return {
        "raw_text": text,
        "form": form,
        "strength": strength,
        "frequency": frequency,
        "duration": duration,
        "instructions": "; ".join(instructions) if instructions else None,
        "drug_candidate_text": residual,
    }


# ---------------------------------------------------------------------------
# Stage 6: Formulary Matching Engine
# ---------------------------------------------------------------------------

class FormularyMatcher:
    """
    Multi-metric clinical formulary resolver with decision-support safety gates.
    """

    def __init__(self, formulary_path: Optional[str] = None):
        path = formulary_path or DEFAULT_FORMULARY_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Formulary database not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

        # Pre-compute skeletons and normalized aliases for fast retrieval
        for entry in self.entries:
            entry["_gen_lower"] = entry["generic_name"].lower()
            entry["_gen_skel"] = consonant_skeleton(entry["generic_name"])
            entry["_aliases"] = [entry["_gen_lower"]] + [b.lower() for b in entry.get("common_brands", [])]

    def _score_name_match(self, query: str, entry: Dict[str, Any]) -> Tuple[float, str]:
        """Compute highest similarity against generic name or brand aliases."""
        if not query:
            return 0.0, "Empty query"

        q_lower = query.lower()
        q_skel = consonant_skeleton(query)

        best_score = 0.0
        best_match_name = entry["generic_name"]

        for alias in entry["_aliases"]:
            lev = levenshtein_similarity(q_lower, alias)
            jw = jaro_winkler_similarity(q_lower, alias)
            a_skel = consonant_skeleton(alias)
            skel_score = levenshtein_similarity(q_skel, a_skel)
            sub_bonus = 0.15 if (q_lower in alias or alias in q_lower) and len(q_lower) >= 4 else 0.0

            score = (0.45 * jw) + (0.35 * lev) + (0.20 * skel_score) + sub_bonus
            score = min(1.0, score)

            if score > best_score:
                best_score = score
                best_match_name = alias.title()

        return best_score, best_match_name

    def _score_form_compatibility(self, detected_form: Optional[str], entry: Dict[str, Any]) -> float:
        """Verify if detected dosage form matches approved formulary forms."""
        if not detected_form:
            return 0.6

        allowed = [f.lower() for f in entry.get("allowed_forms", [])]
        df = detected_form.lower()
        if df in allowed:
            return 1.0
        if any(df in a or a in df for a in allowed):
            return 0.85
        return 0.15

    def _score_strength_compatibility(self, detected_strength: Optional[str], entry: Dict[str, Any]) -> float:
        """Verify if detected strength matches approved formulary strengths."""
        if not detected_strength:
            return 0.6

        allowed = entry.get("allowed_strengths", [])
        ds_clean = detected_strength.replace(" ", "").lower()
        for s in allowed:
            if ds_clean == s.replace(" ", "").lower():
                return 1.0

        val_match = re.search(r"\d+(?:\.\d+)?", detected_strength)
        if val_match:
            v = val_match.group(0)
            if any(v in s for s in allowed):
                return 0.8
        return 0.25

    def resolve(
        self,
        prescription_text: str,
        top_k: int = 3,
        confidence_threshold: float = 0.60
    ) -> Dict[str, Any]:
        """
        Parse raw transcription, match against formulary, and apply clinical safety gate.
        """
        entities = extract_prescription_entities(prescription_text)
        query = entities["drug_candidate_text"]

        scored_candidates = []
        for entry in self.entries:
            name_score, matched_term = self._score_name_match(query, entry)
            form_score = self._score_form_compatibility(entities["form"], entry)
            strength_score = self._score_strength_compatibility(entities["strength"], entry)

            total_confidence = (0.55 * name_score) + (0.25 * form_score) + (0.20 * strength_score)

            scored_candidates.append({
                "generic_name": entry["generic_name"],
                "category": entry["category"],
                "confidence": round(total_confidence, 4),
                "matched_term": matched_term,
                "nlem_status": entry.get("nlem_status", False),
                "allowed_forms": entry.get("allowed_forms", []),
                "allowed_strengths": entry.get("allowed_strengths", []),
                "scores": {
                    "name_score": round(name_score, 4),
                    "form_score": round(form_score, 4),
                    "strength_score": round(strength_score, 4),
                }
            })

        scored_candidates.sort(key=lambda x: -x["confidence"])
        top_matches = scored_candidates[:top_k]

        top_conf = top_matches[0]["confidence"] if top_matches else 0.0
        second_conf = top_matches[1]["confidence"] if len(top_matches) > 1 else 0.0
        ambiguity_gap = round(top_conf - second_conf, 4)

        if top_conf >= 0.75 and ambiguity_gap >= 0.08:
            gate_status = "HIGH_CONFIDENCE_MATCH"
            abstain = False
            audit_message = f"High-confidence match with {top_matches[0]['generic_name']} (confidence {top_conf:.2f})."
        elif top_conf >= confidence_threshold:
            gate_status = "AMBIGUOUS_MATCH_REQUIRES_CONFIRMATION"
            abstain = False
            audit_message = f"Moderate match; alternative candidates close in score (margin: {ambiguity_gap:.2f}). Human review required."
        else:
            gate_status = "ABSTAIN_UNRECOGNIZED_PRESCRIPTION"
            abstain = True
            audit_message = f"Top candidate confidence ({top_conf:.2f}) below safety threshold ({confidence_threshold:.2f}). Autonomous recommendation blocked."

        return {
            "raw_input": prescription_text,
            "parsed_entities": entities,
            "top_candidates": top_matches,
            "safety_gate": {
                "status": gate_status,
                "confidence": top_conf,
                "ambiguity_gap": ambiguity_gap,
                "abstain": abstain,
                "requires_human_confirmation": True,
                "audit_message": audit_message,
            }
        }

    def resolve_prescription_line(
        self,
        line_text: str,
        top_k: int = 3,
        confidence_threshold: float = 0.60
    ) -> List[Dict[str, Any]]:
        """
        Handle both single and compound prescription lines.
        """
        parts = [p.strip() for p in re.split(r";|\band\b(?=\s+[A-Z])", line_text) if p.strip()]
        has_multi = len(parts) > 1 and all(
            re.search(r"\d+\s*(?:mg|mcg|IU|%|g|mL)", p, re.I) or
            any(f.lower() in p.lower() for f in ["tab", "cap", "syp", "inj", "drops"])
            for p in parts
        )

        if has_multi:
            return [self.resolve(part, top_k=top_k, confidence_threshold=confidence_threshold) for part in parts]
        return [self.resolve(line_text, top_k=top_k, confidence_threshold=confidence_threshold)]

