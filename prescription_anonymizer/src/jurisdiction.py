"""
Multi-National Jurisdiction Engine.
Detects national healthcare regulatory frameworks based on statutory tokens and layout signatures.

Supported Jurisdictions:
- US: United States (HIPAA, DEA, NPI, SSN)
- UK: United Kingdom (NHS, GMC, FP10)
- AU: Australia (Medicare, PBS, AHPRA)
- EU: European Union / Germany (GDPR, Muster 16, BSNR, LANR, Krankenkasse)
- IN: India (DPDP Act, ABHA, NMC/MCI, UHID)
- GLOBAL: Fallback running unified superset of all privacy rules
"""

import re
from typing import Dict, List, Literal

JurisdictionType = Literal["AUTO", "US", "UK", "AU", "EU", "IN", "GLOBAL"]

JURISDICTION_SIGNATURES: Dict[str, List[re.Pattern]] = {
    "UK": [
        re.compile(r"\b(?:NHS|FP10|GMC|National\s*Health\s*Service|GP10|WP10)\b", re.IGNORECASE),
        re.compile(r"\bPharmacist(?:'s|’s)?\s*Signature\b", re.IGNORECASE),
        re.compile(r"\bAge\s*D\.O\.B\b", re.IGNORECASE),
    ],
    "US": [
        re.compile(r"\b(?:DEA(?:\s*No\.?)?|NPI|DEA#|Security\s*Prescription|Tamper\s*Resistant)\b", re.IGNORECASE),
        re.compile(r"\b(?:Refills?\s*[:\-=]?\s*(?:None|0|1|2|3|4|5|NR))\b", re.IGNORECASE),
        re.compile(r"\b(?:Dispense\s*as\s*Written|DAW)\b", re.IGNORECASE),
    ],
    "AU": [
        re.compile(r"\b(?:PBS|Medicare|AHPRA|Pharmaceutical\s*Benefits\s*Scheme)\b", re.IGNORECASE),
        re.compile(r"\bPrescriber\s*No\.?\b", re.IGNORECASE),
    ],
    "EU": [
        re.compile(r"\b(?:Krankenkasse|Muster\s*16|Rezept|Versicherten-Nr|BSNR|LANR|Arzt-Nr)\b", re.IGNORECASE),
        re.compile(r"\bAusstellungsdatum\b", re.IGNORECASE),
        re.compile(r"\bWohnanschrift\b", re.IGNORECASE),
    ],
    "IN": [
        re.compile(r"\b(?:UHID|OPD|IPD|ABHA|NMC|MCI|\+91|PIN(?:\s*Code)?)\b", re.IGNORECASE),
        re.compile(r"\b(?:Consultant\s*Physician|Regn?\s*No\.?)\b", re.IGNORECASE),
    ]
}


class JurisdictionDetector:
    """Detects national healthcare regulatory framework from extracted document text."""

    @classmethod
    def detect(cls, text: str) -> str:
        if not text:
            return "GLOBAL"

        scores: Dict[str, int] = {"UK": 0, "US": 0, "AU": 0, "EU": 0, "IN": 0}
        for jurisdiction, patterns in JURISDICTION_SIGNATURES.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                scores[jurisdiction] += len(matches)

        top_jur, top_score = max(scores.items(), key=lambda item: item[1])
        return top_jur if top_score > 0 else "GLOBAL"

    @classmethod
    def resolve_jurisdiction(cls, text: str, requested: str = "AUTO") -> str:
        req_clean = (requested or "AUTO").upper().strip()
        if req_clean in ["US", "UK", "AU", "EU", "IN", "GLOBAL"]:
            return req_clean
        return cls.detect(text)
