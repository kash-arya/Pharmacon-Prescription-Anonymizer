"""
Clinical document gatekeeper and binary signature validation module.
Enforces file signature integrity and scores clinical/pharmacological context.
"""

import os
import re
from typing import Tuple, List

try:
    from src.entities import is_whitelisted
except ImportError:
    from prescription_anonymizer.src.entities import is_whitelisted


SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}

# Category A: Hard Clinical & Provider Markers (+2.0 pts each, cap at 6.0 pts)
CAT_A_PATTERNS = [
    r"\bRx\b",
    r"℞",
    r"\bDr\.?\b",
    r"\bDoctor\b",
    r"\bHospital\b",
    r"\bClinic\b",
    r"\bMedical\b",
    r"\bPharmacy\b",
    r"\bDiagnosis\b",
    r"\bProvisional\s+Diagnosis\b",
    r"\bLab\s+Tests?\b",
    r"\bMedicines?\b",
    r"\bRegistration\s+No\b",
    r"\bReg\.?\s*No\b",
    r"\bUHID\b",
    r"\bMBBS\b",
    r"\bMD\b",
    r"\bFRCS\b",
    r"\bMRCP\b",
    r"\bPBS\b",
    r"\bNHS\b",
    r"\bDEA\b",
    r"\bVersicherten\b",
    r"\bArzt\b",
]

# Category B: Pharmacological & Dosage Markers (+1.5 pts each, cap at 6.0 pts)
CAT_B_PATTERNS = [
    r"\b\d+\s*(?:mg|ml|mcg|gm|iu)\b",
    r"\bTablet\b",
    r"\bTab\.?\b",
    r"\bCapsule\b",
    r"\bCap\.?\b",
    r"\bSyrup\b",
    r"\bSyp\.?\b",
    r"\bInjection\b",
    r"\bInj\.?\b",
    r"\bSolution\b",
    r"\bOintment\b",
    r"\bMorning\b",
    r"\bNight\b",
    r"\bAfter\s+food\b",
    r"\bBefore\s+food\b",
    r"\b(?:BD|TDS|OD|QID|PRN|SOS)\b",
]

# Category C: Demographic Context (+0.5 pts each, cap at 1.5 pts)
CAT_C_PATTERNS = [
    r"\bPatient\b",
    r"\bAge\b",
    r"\bSex\b",
    r"\bGender\b",
    r"\bWeight\b",
    r"\bBP\b",
    r"\bRef\.?\s*no\b",
]

# Category E: Non-Medical Noise Penalty (-2.5 pts each)
CAT_E_PATTERNS = [
    r"\bTax\s+Invoice\b",
    r"\bGSTIN\b",
    r"\bShipping\s+Address\b",
    r"\bBill\s+To\b",
    r"\bCurriculum\s+Vitae\b",
    r"\bResume\b",
    r"\bimport\s+[a-zA-Z_]+\b",
    r"\bdef\s+[a-zA-Z_]+\(",
]

_COMPILED_CAT_A = [re.compile(p, re.IGNORECASE) for p in CAT_A_PATTERNS]
_COMPILED_CAT_B = [re.compile(p, re.IGNORECASE) for p in CAT_B_PATTERNS]
_COMPILED_CAT_C = [re.compile(p, re.IGNORECASE) for p in CAT_C_PATTERNS]
_COMPILED_CAT_E = [re.compile(p, re.IGNORECASE if "import" not in p and "def" not in p else 0) for p in CAT_E_PATTERNS]


def validate_file_signature(filename: str, header_bytes: bytes) -> Tuple[bool, str]:
    """
    Validates file extension and binary magic bytes.
    Accepts PDF, PNG, JPEG, and WEBP.
    Tolerates client canvas re-encoding where extension might not match
    the underlying image payload.
    Returns (True, format_str) on success, or (False, 'UNSUPPORTED_FILE_FORMAT').
    """
    if not filename or not header_bytes:
        return False, "UNSUPPORTED_FILE_FORMAT"

    _, ext = os.path.splitext(filename)
    ext_lower = ext.lower()

    if ext_lower not in SUPPORTED_EXTENSIONS:
        return False, "UNSUPPORTED_FILE_FORMAT"

    if header_bytes.startswith(b"%PDF-"):
        return True, "PDF"
    elif header_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return True, "PNG"
    elif header_bytes.startswith(b"\xff\xd8\xff"):
        return True, "JPEG"
    elif header_bytes.startswith(b"RIFF") and len(header_bytes) >= 12 and header_bytes[8:12] == b"WEBP":
        return True, "WEBP"

    return False, "UNSUPPORTED_FILE_FORMAT"


def evaluate_prescription_gate(
    extracted_text: str,
    visual_stamps_count: int = 0,
    barcode_count: int = 0
) -> Tuple[bool, float, List[str]]:
    """
    Scores whether extracted text and visual markers indicate a medical prescription.
    Returns (is_prescription, total_score, review_reasons).
    """
    if not extracted_text:
        extracted_text = ""

    # Category A: Hard Clinical & Provider Markers (+2.0 pts each, cap at 6.0 pts)
    matches_a = sum(1 for rx in _COMPILED_CAT_A if rx.search(extracted_text))
    cat_a = min(6.0, matches_a * 2.0)

    # Category B: Pharmacological & Dosage Markers (+1.5 pts each, cap at 6.0 pts)
    matches_b = sum(1 for rx in _COMPILED_CAT_B if rx.search(extracted_text))
    words = re.findall(r"[A-Za-z]{3,}", extracted_text)
    excluded_common = {"date", "name", "age", "male", "female", "hospital", "clinic", "doctor", "patient"}
    wl_words = set(w.lower() for w in words if is_whitelisted(w) and w.lower() not in excluded_common)
    cat_b = min(6.0, (matches_b + len(wl_words)) * 1.5)

    # Category C: Demographic Context (+0.5 pts each, cap at 1.5 pts, only if A + B >= 1.5)
    if cat_a + cat_b >= 1.5:
        matches_c = sum(1 for rx in _COMPILED_CAT_C if rx.search(extracted_text))
        cat_c = min(1.5, matches_c * 0.5)
    else:
        cat_c = 0.0

    # Category D: Visual Clinical Fallback (+2.0 pts)
    if visual_stamps_count > 0 or (barcode_count > 0 and (cat_a + cat_b + cat_c >= 1.0)):
        cat_d = 2.0
    else:
        cat_d = 0.0

    # Category E: Non-Medical Noise Penalty (-2.5 pts each)
    matches_e = sum(1 for rx in _COMPILED_CAT_E if rx.search(extracted_text))
    penalty_e = matches_e * 2.5

    total_score = max(0.0, cat_a + cat_b + cat_c + cat_d - penalty_e) if penalty_e <= (cat_a + cat_b + cat_c + cat_d) else (cat_a + cat_b + cat_c + cat_d - penalty_e)
    is_prescription = (cat_a + cat_b + cat_d >= 2.0) and (total_score >= 2.5)

    final_score = round(float(total_score), 2)
    if is_prescription:
        return True, final_score, []
    else:
        return False, final_score, ["Document lacks clinical, pharmacological, or provider markers."]


def record_gate_override_telemetry(
    format_type: str,
    ocr_word_count: int,
    gate_score: float,
    redacted_count: int
) -> None:
    """
    Appends non-PII structural telemetry to logs/gate_overrides.jsonl.
    Contains strictly zero PII/PHI.
    """
    import datetime
    import json

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "gate_overrides.jsonl")

    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "format": format_type,
        "ocr_word_count": int(ocr_word_count),
        "gate_score_before_override": round(float(gate_score), 2),
        "redacted_count": int(redacted_count)
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

