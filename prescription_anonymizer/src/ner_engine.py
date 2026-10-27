"""
Local NER Engine using spaCy for Clinical Prescriptions.
Integrates with en_core_web_sm and xx_ent_wiki_sm to detect PERSON, LOCATION (GPE, LOC, FAC),
and healthcare facility ORGs locally without sending data to external APIs.
"""

import os
import re
from typing import List, Dict, Any, Optional
from .entities import is_whitelisted, is_clinical_term


class LocalNEREngine:
    """Singleton/cached local NER engine powered by spaCy en_core_web_sm and xx_ent_wiki_sm."""

    _instance: Optional["LocalNEREngine"] = None
    _nlp_en = None
    _nlp_xx = None

    def __new__(cls) -> "LocalNEREngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_models()
        return cls._instance

    def _init_models(self) -> None:
        if os.environ.get("RENDER_LOW_MEMORY") == "1":
            self._nlp_en = None
            self._nlp_xx = None
            return

        try:
            import en_core_web_sm
            self._nlp_en = en_core_web_sm.load(exclude=["parser", "lemmatizer", "attribute_ruler", "tagger"])
        except Exception:
            try:
                import spacy
                self._nlp_en = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "attribute_ruler", "tagger"])
            except Exception:
                self._nlp_en = None

        self._nlp_xx = None

    @property
    def nlp(self):
        if os.environ.get("RENDER_LOW_MEMORY") == "1":
            return None
        if self._nlp_en is None:
            self._init_models()
        return self._nlp_en

    @property
    def nlp_xx(self):
        if os.environ.get("RENDER_LOW_MEMORY") == "1":
            return None
        if self._nlp_xx is None:
            try:
                import xx_ent_wiki_sm
                self._nlp_xx = xx_ent_wiki_sm.load()
            except Exception:
                try:
                    import spacy
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

        # Zero-spaCy deterministic NER fast-path when RENDER_LOW_MEMORY=1
        if os.environ.get("RENDER_LOW_MEMORY") == "1" or self._nlp_en is None:
            results: List[Dict[str, Any]] = []
            seen_spans = set()

            # 1. Titled/honorific names
            honorific_re = re.compile(
                r"\b(?:Dr|Mr|Mrs|Ms|Prof|Shri|Smt|Kumari|Master|Miss)\.?[^\S\r\n]+[A-Z][a-zA-Z.]*(?:[^\S\r\n]+[A-Z][a-zA-Z.]*){0,3}",
                re.IGNORECASE
            )
            for m in honorific_re.finditer(text):
                ent_text = m.group(0).strip()
                if (
                    len(ent_text) >= 4
                    and not is_clinical_term(ent_text)
                    and not any(is_clinical_term(w) for w in ent_text.split())
                ):
                    seen_spans.add((m.start(), m.end(), "PATIENT_NAME"))
                    results.append({
                        "text": ent_text,
                        "label": "PERSON",
                        "category": "PATIENT_NAME",
                        "start_char": m.start(),
                        "end_char": m.end(),
                    })

            # 2. Label : Value name patterns
            label_name_re = re.compile(
                r"(?:Patient[^\S\r\n]*Name|Name|Pt\.?[^\S\r\n]*Name|Ref\.?[^\S\r\n]*By|Referred[^\S\r\n]*By|Consultant|Doctor|Physician)[^\S\r\n]*[:\-][^\S\r\n]*([A-Z][A-Za-z.]*(?:[^\S\r\n]+[A-Z][A-Za-z.]*){0,3})",
                re.IGNORECASE
            )
            for m in label_name_re.finditer(text):
                ent_text = m.group(1).strip()
                start_c = m.start(1)
                end_c = m.end(1)
                if (
                    len(ent_text) >= 3
                    and not is_clinical_term(ent_text)
                    and not any(is_clinical_term(w) for w in ent_text.split())
                ):
                    if not any(s <= start_c and end_c <= e and c == "PATIENT_NAME" for (s, e, c) in seen_spans):
                        seen_spans.add((start_c, end_c, "PATIENT_NAME"))
                        results.append({
                            "text": ent_text,
                            "label": "PERSON",
                            "category": "PATIENT_NAME",
                            "start_char": start_c,
                            "end_char": end_c,
                        })

            # 3. Institutional phrases
            inst_re = re.compile(
                r"\b[A-Z][A-Za-z&.]*(?:[^\S\r\n]+(?:&|[A-Z][A-Za-z&.]*)){0,4}[^\S\r\n]+(?:Hospital|Hospitals|Clinic|Diagnostics|Imaging|Medical[^\S\r\n]+Centre|Medical[^\S\r\n]+Center|Healthcare|Laboratory|Labs|Pharmacy|Institute|Nursing[^\S\r\n]+Home)(?:[^\S\r\n]+(?:&|[A-Z][A-Za-z&.]*)){0,3}\b",
                re.IGNORECASE
            )
            for m in inst_re.finditer(text):
                ent_text = m.group(0).strip()
                start_c = m.start()
                end_c = m.end()
                if (
                    len(ent_text) >= 3
                    and not is_clinical_term(ent_text)
                    and not any(is_clinical_term(w) for w in ent_text.split())
                ):
                    if not any(s <= start_c and end_c <= e and c == "CLINIC_FACILITY" for (s, e, c) in seen_spans):
                        seen_spans.add((start_c, end_c, "CLINIC_FACILITY"))
                        results.append({
                            "text": ent_text,
                            "label": "ORG",
                            "category": "CLINIC_FACILITY",
                            "start_char": start_c,
                            "end_char": end_c,
                        })

            return results

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


MedicalNEREngine = LocalNEREngine

