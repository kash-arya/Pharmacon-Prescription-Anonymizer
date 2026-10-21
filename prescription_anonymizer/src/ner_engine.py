"""
Local NER Engine using spaCy for Clinical Prescriptions.
Integrates with en_core_web_sm to detect PERSON, LOCATION (GPE, LOC, FAC),
and healthcare facility ORGs locally without sending data to external APIs.
"""

import re
from typing import List, Dict, Any, Optional
import spacy
from .entities import is_whitelisted


class LocalNEREngine:
    """Singleton/cached local NER engine powered by spaCy en_core_web_sm."""

    _instance: Optional["LocalNEREngine"] = None
    _nlp = None

    def __new__(cls) -> "LocalNEREngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                import en_core_web_sm
                cls._nlp = en_core_web_sm.load()
            except Exception:
                cls._nlp = spacy.load("en_core_web_sm")
        return cls._instance

    @property
    def nlp(self):
        return self._nlp

    def find_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Runs NER across text and returns filtered candidate entities:
        - PERSON -> PATIENT_NAME / DOCTOR_NAME
        - GPE, LOC, FAC -> ADDRESS / LOCATION
        - ORG -> CLINIC_FACILITY / ORG (when not whitelisted clinical terms)
        Guaranteed to filter out all whitelisted clinical pharmacology, anatomy, and dosages.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        results = []

        for ent in doc.ents:
            ent_text = ent.text.strip()
            # Skip short noise, single characters or pure digits
            if len(ent_text) < 3 or re.match(r"^\d+$", ent_text):
                continue

            # Strict Clinical Whitelist Filter: never redact medical terms
            if is_whitelisted(ent_text):
                continue

            # Check if any individual word in the entity is a protected clinical term
            words = ent_text.split()
            if any(is_whitelisted(w) for w in words):
                continue

            label = ent.label_
            category = None

            if label == "PERSON":
                # Exclude form labels, common document tokens, or noisy abbreviations misclassified as PERSON
                if len(ent_text) < 4 or ent_text.upper() in {"REG", "TAB", "SEX", "AGE", "DATE", "MALE", "FEMALE", "NAME", "TOTAL", "PAGE", "REF", "SIGN", "TIME"}:
                    continue
                category = "PATIENT_NAME"
            elif label in ("GPE", "LOC", "FAC"):
                category = "ADDRESS"
            elif label == "ORG":
                # For organizations, verify genuine healthcare facility context via facility keywords
                if re.search(r"\b(?:hospital|hospitals|clinic|clinics|diagnostics|diagnostic|imaging|laborator(?:y|ies)|centre|center|centres|centers|health|pharmacy|institute|infirmary|medical|care|dispensary|nursing|maternity|healthcare|foundation|trust)\b", ent_text, re.I):
                    if not re.search(r"\b(?:Surgery|Radiology|Pediatrics|Medicine|Cardiology|Ultrasound|Impression|Diagnosis|Examination|History)\b", ent_text, re.I):
                        category = "CLINIC_FACILITY"

            if category:
                results.append({
                    "text": ent_text,
                    "label": label,
                    "category": category,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                })

        return results
