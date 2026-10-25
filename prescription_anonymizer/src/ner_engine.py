"""
Local NER Engine using spaCy for Clinical Prescriptions.
Integrates with en_core_web_sm and xx_ent_wiki_sm to detect PERSON, LOCATION (GPE, LOC, FAC),
and healthcare facility ORGs locally without sending data to external APIs.
"""

import os
import re
from typing import List, Dict, Any, Optional
import spacy
from .entities import is_whitelisted


class LocalNEREngine:
    """Singleton/cached local NER engine powered by spaCy en_core_web_sm and xx_ent_wiki_sm."""

    _instance: Optional["LocalNEREngine"] = None
    _nlp_en = None
    _nlp_xx = None

    def __new__(cls) -> "LocalNEREngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                import en_core_web_sm
                cls._nlp_en = en_core_web_sm.load()
            except Exception:
                try:
                    cls._nlp_en = spacy.load("en_core_web_sm")
                except Exception:
                    cls._nlp_en = None

            cls._nlp_xx = None

        return cls._instance

    @property
    def nlp(self):
        return self._nlp_en

    @property
    def nlp_xx(self):
        if self._nlp_xx is None and os.environ.get("RENDER_LOW_MEMORY") != "1":
            try:
                import xx_ent_wiki_sm
                self._nlp_xx = xx_ent_wiki_sm.load()
            except Exception:
                try:
                    self._nlp_xx = spacy.load("xx_ent_wiki_sm")
                except Exception:
                    self._nlp_xx = None
        return self._nlp_xx

    def find_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Runs English and multilingual NER across text and returns filtered candidate entities:
        - PERSON, PER -> PATIENT_NAME
        - GPE, LOC, FAC -> ADDRESS / LOCATION
        - ORG -> CLINIC_FACILITY (when matching healthcare facility terms)
        Guaranteed to filter out all whitelisted clinical pharmacology, anatomy, and dosages.
        """
        if not text or not text.strip():
            return []

        models = []
        if self._nlp_en is not None:
            models.append((self._nlp_en, "en"))

        has_eu_markers = bool(re.search(
            r"[äöüßéèêëàâîïôûùçñáíóúÁÉÍÓÚÑÄÖÜ]|\b(?:Rezept|Krankenkasse|Geb|Ausstellungsdatum|Vorname|Unterschrift|Arzt|Receta|Ordonnance|Médicament|Apotheke)\b",
            text,
            re.IGNORECASE
        ))
        if has_eu_markers and os.environ.get("RENDER_LOW_MEMORY") != "1":
            if self._nlp_xx is None:
                try:
                    import xx_ent_wiki_sm
                    self._nlp_xx = xx_ent_wiki_sm.load()
                except Exception:
                    try:
                        self._nlp_xx = spacy.load("xx_ent_wiki_sm")
                    except Exception:
                        self._nlp_xx = None
            if self._nlp_xx is not None:
                models.append((self._nlp_xx, "xx"))

        facility_regex = re.compile(
            r"\b(?:hospital|hospitals|clinic|clinics|diagnostics|diagnostic|imaging|laborator(?:y|ies)|centre|center|centres|centers|health|pharmacy|institute|infirmary|medical|care|dispensary|nursing|maternity|healthcare|foundation|trust|klinik|klinikum|krankenhaus|spital|praxis|hopital|hôpital|clinique|clinica|clínica|farmacia|apotheke|pharmacie|ambulatorio)\b",
            re.IGNORECASE
        )
        non_facility_regex = re.compile(
            r"\b(?:Surgery|Radiology|Pediatrics|Medicine|Cardiology|Ultrasound|Impression|Diagnosis|Examination|History|Treatment|Rezept|Prescription)\b",
            re.IGNORECASE
        )
        stop_person = {
            "REG", "TAB", "SEX", "AGE", "DATE", "MALE", "FEMALE", "NAME", "TOTAL",
            "PAGE", "REF", "SIGN", "TIME", "DOB", "YEARS", "YEAR", "YRS", "REFILL",
            "PRESCRIPTION", "VERORDNUNG", "RECETA", "REGULAR"
        }
        stop_address = {
            "TAB", "TABS", "CAP", "CAPS", "SYP", "INJ", "MG", "ML", "MCG", "GM",
            "PHONE", "TEL", "MOBILE", "FAX", "EMAIL", "URL", "HTTP", "HTTPS",
            "DAILY", "DATE", "REG", "NO", "AGE", "SEX", "MALE", "FEMALE",
            "HOSPITAL", "CLINIC", "PRESCRIPTION", "REFILL", "SIG", "DOSE", "BP",
            "SOS", "OD", "BD", "TID", "QID", "STAT", "PRN", "HS", "MEALS", "MONTHS",
            "YEARS", "YRS", "REGULAR", "NAME", "PT", "PATIENT", "DR", "DOCTOR"
        }

        results: List[Dict[str, Any]] = []
        seen_spans = set()

        for nlp_model, lang in models:
            doc = nlp_model(text)
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

                if label in ("PERSON", "PER"):
                    if len(ent_text) < 4 or ent_text.upper() in stop_person:
                        continue
                    if lang == "xx":
                        line_start = text.rfind("\n", 0, ent.start_char)
                        line_start = 0 if line_start == -1 else line_start + 1
                        line_end = text.find("\n", ent.end_char)
                        line_end = len(text) if line_end == -1 else line_end
                        line_ctx = text[line_start:line_end]

                        if re.search(r"\b(?:Rx|Rp\.?|Disp\.?|Take|Medikament|Tablette|Kapsel|Dosage|Sig\.?)\b", line_ctx, re.IGNORECASE):
                            continue

                        cap_words = sum(1 for w in ent_text.split() if w and w[0].isupper())
                        is_name_header = bool(re.search(r"\b(?:Name|Patient|Herr|Frau|Dr|Doctor|Arzt|Unterschrift|Signature|Prescriber)\b", line_ctx, re.IGNORECASE))
                        if cap_words < 2 and not is_name_header:
                            continue
                    category = "PATIENT_NAME"
                elif label in ("GPE", "LOC", "FAC"):
                    if len(ent_text) < 3 or ent_text.upper() in stop_address:
                        continue
                    category = "ADDRESS"
                elif label == "ORG":
                    if facility_regex.search(ent_text) and not non_facility_regex.search(ent_text):
                        category = "CLINIC_FACILITY"

                if category:
                    # Deduplicate overlapping spans
                    is_sub = False
                    for (s, e, c) in seen_spans:
                        if c == category and s <= ent.start_char and ent.end_char <= e:
                            is_sub = True
                            break
                    if not is_sub:
                        seen_spans.add((ent.start_char, ent.end_char, category))
                        results.append({
                            "text": ent_text,
                            "label": label,
                            "category": category,
                            "start_char": ent.start_char,
                            "end_char": ent.end_char,
                        })

        return results
