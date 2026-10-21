"""
Native Vector PDF Redaction Engine.

Extracts PII bounding boxes directly from PDF text streams and applies
cryptographic/physical text deletion + black-box visual masking using PyMuPDF.
"""

from typing import List, Dict, Any, Tuple
import re
import pymupdf

try:
    from prescription_anonymizer.src.entities import (
        BoundingBox,
        RedactionTarget,
        PATIENT_NAME_ANCHORS,
        DEMOGRAPHIC_ANCHORS,
        IDENTIFIER_ANCHORS,
        CONTACT_ANCHORS,
        DOCTOR_ANCHORS,
        ADDRESS_ANCHORS,
        DATE_ANCHORS,
        STATUTORY_ID_PATTERNS,
        is_whitelisted,
    )
    from prescription_anonymizer.src.jurisdiction import JurisdictionDetector
    from prescription_anonymizer.src.ner_engine import LocalNEREngine
except ImportError:
    from src.entities import (
        BoundingBox,
        RedactionTarget,
        PATIENT_NAME_ANCHORS,
        DEMOGRAPHIC_ANCHORS,
        IDENTIFIER_ANCHORS,
        CONTACT_ANCHORS,
        DOCTOR_ANCHORS,
        ADDRESS_ANCHORS,
        DATE_ANCHORS,
        STATUTORY_ID_PATTERNS,
        is_whitelisted,
    )
    from src.jurisdiction import JurisdictionDetector
    from src.ner_engine import LocalNEREngine



class VectorPDFRedactor:
    """
    High-precision redactor for digitally generated PDFs with selectable text layers.
    """

    def __init__(self, expand_box: bool = True, padding_x: float = 4.0, padding_y: float = 2.0):
        self.expand_box = expand_box
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.ner_engine = LocalNEREngine()

    def find_pii_targets(self, doc: pymupdf.Document) -> List[RedactionTarget]:
        """Scan all pages and return bounding boxes of identified PII entities."""
        targets: List[RedactionTarget] = []

        all_rules = [
            ("PATIENT_NAME", PATIENT_NAME_ANCHORS),
            ("DEMOGRAPHIC", DEMOGRAPHIC_ANCHORS),
            ("IDENTIFIER", IDENTIFIER_ANCHORS),
            ("CONTACT", CONTACT_ANCHORS),
            ("DOCTOR_INFO", DOCTOR_ANCHORS),
            ("DATE", DATE_ANCHORS),
            ("ADDRESS", ADDRESS_ANCHORS),
        ]

        for page_idx, page in enumerate(doc):
            # Extract word-level text with bboxes: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            words = page.get_text("words")
            text = page.get_text()

            # 1. Regex pattern scanning on line-level text
            for category, patterns in all_rules:
                for pattern in patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        matched_str = match.group(0).strip()
                        # Search for instances of the matched phrase on the page
                        rects = page.search_for(matched_str)
                        for r in rects:
                            bbox = BoundingBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)
                            targets.append(RedactionTarget(
                                page_number=page_idx + 1,
                                category=category,
                                raw_text=matched_str,
                                bbox=bbox,
                                confidence=0.98,
                                engine="vector_rule"
                            ))

            # 2. Statutory National Identifiers (US, UK, AU, EU, IN)
            for pattern, category in STATUTORY_ID_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    matched_str = match.group(0).strip()
                    rects = page.search_for(matched_str)
                    if not rects and match.groups():
                        rects = page.search_for(match.group(1).strip())
                    for r in rects:
                        bbox = BoundingBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)
                        targets.append(RedactionTarget(
                            page_number=page_idx + 1,
                            category=category,
                            raw_text=matched_str,
                            bbox=bbox,
                            confidence=0.99,
                            engine="statutory_rule"
                        ))

            # 3. Specific field-value boundary expansion:
            # A. Rightward expansion for patient name labels
            name_label_rects = (
                page.search_for("Patient Name:") +
                page.search_for("PATIENT NAME:") +
                page.search_for("Name:") +
                page.search_for("Surname:") +
                page.search_for("Forename(s):") +
                page.search_for("Forename:")
            )
            for n_rect in name_label_rects:
                # Value area: from end of label up to 200 pts rightward on same line
                val_rect = pymupdf.Rect(n_rect.x1 + 2, n_rect.y0 - 2, min(page.rect.width - 40, n_rect.x1 + 180), n_rect.y1 + 2)
                targets.append(RedactionTarget(
                    page_number=page_idx + 1,
                    category="PATIENT_NAME_VALUE",
                    raw_text="[NAME_FIELD_VALUE]",
                    bbox=BoundingBox(x0=val_rect.x0, y0=val_rect.y0, x1=val_rect.x1, y1=val_rect.y1),
                    confidence=0.99,
                    engine="vector_geometry"
                ))

            # B. Downward expansion for German Muster 16 (patient name printed beneath label)
            german_name_rects = page.search_for("Name, Vorname")
            for g_rect in german_name_rects:
                val_below = pymupdf.Rect(g_rect.x0, g_rect.y1, min(page.rect.width - 20, g_rect.x0 + 200), g_rect.y1 + 18)
                targets.append(RedactionTarget(
                    page_number=page_idx + 1,
                    category="PATIENT_NAME_VALUE",
                    raw_text="[MUSTER16_NAME_VALUE]",
                    bbox=BoundingBox(x0=val_below.x0, y0=val_below.y0, x1=val_below.x1, y1=val_below.y1),
                    confidence=0.99,
                    engine="vector_geometry"
                ))

            # C. Prescriber / Doctor / Pharmacist Signatures
            sig_labels = [
                "PRESCRIBER SIGNATURE",
                "Prescriber Signature",
                "Doctor Signature",
                "Doctor's Signature",
                "Doctor Signature & Stamp",
                "Pharmacist Signature",
                "Unterschrift des Arztes",
                "Authorized Prescriber Signature"
            ]
            for sig_lbl in sig_labels:
                for s_rect in page.search_for(sig_lbl):
                    sig_box = pymupdf.Rect(s_rect.x0 - 15, s_rect.y0 - 26, max(s_rect.x1 + 15, s_rect.x0 + 120), s_rect.y0 + 2)
                    targets.append(RedactionTarget(
                        page_number=page_idx + 1,
                        category="DOCTOR_SIGNATURE",
                        raw_text="[SIGNATURE_VALUE]",
                        bbox=BoundingBox(x0=sig_box.x0, y0=sig_box.y0, x1=sig_box.x1, y1=sig_box.y1),
                        confidence=0.99,
                        engine="signature_detector"
                    ))

            # D. Local spaCy NER for unanchored Persons, Locations, and Organizations
            ner_candidates = self.ner_engine.find_entities(text)
            for cand in ner_candidates:
                matched_str = cand["text"]
                if not is_whitelisted(matched_str) and len(matched_str) >= 3:
                    rects = page.search_for(matched_str)
                    for r in rects:
                        bbox = BoundingBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)
                        targets.append(RedactionTarget(
                            page_number=page_idx + 1,
                            category=cand["category"],
                            raw_text=matched_str,
                            bbox=bbox,
                            confidence=0.95,
                            engine="spacy_ner"
                        ))

        return targets

    def redact(self, input_pdf_path: str, output_pdf_path: str, jurisdiction: str = "AUTO") -> Dict[str, Any]:
        """
        Executes complete text-deletion and visual masking on the PDF.
        Returns audit summary with redacted entities and detected jurisdiction.
        """
        doc = pymupdf.open(input_pdf_path)
        full_doc_text = " ".join(page.get_text() for page in doc)
        resolved_jur = JurisdictionDetector.resolve_jurisdiction(full_doc_text, jurisdiction)

        targets = self.find_pii_targets(doc)

        # Apply redactions page by page
        for t in targets:
            page = doc[t.page_number - 1]
            rect = pymupdf.Rect(
                max(0, t.bbox.x0 - self.padding_x),
                max(0, t.bbox.y0 - self.padding_y),
                t.bbox.x1 + self.padding_x,
                t.bbox.y1 + self.padding_y
            )
            # Add redaction annotation (solid black fill, zero stroke)
            page.add_redact_annot(rect, fill=(0, 0, 0))

        # Physically erase text streams and render black rectangles
        for page in doc:
            page.apply_redactions()

        # Wipe document metadata to prevent file-property leaks
        doc.set_metadata({
            "title": "Redacted Medical Document",
            "author": "Pharmacon Anonymizer",
            "subject": "De-identified Prescription",
            "keywords": "anonymized, de-identified",
            "creator": "Pharmacon Anonymizer Engine",
            "producer": "PyMuPDF",
            "creationDate": "",
            "modDate": ""
        })

        doc.save(output_pdf_path, garbage=4, deflate=True)
        doc.close()

        return {
            "input_file": input_pdf_path,
            "output_file": output_pdf_path,
            "jurisdiction": resolved_jur,
            "total_entities_redacted": len(targets),
            "total_redactions_applied": len(targets),
            "redacted_targets": [t.model_dump() for t in targets]
        }
