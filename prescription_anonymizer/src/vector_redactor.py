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
        KEY_VALUE_ANCHORS,
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
        KEY_VALUE_ANCHORS,
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

    def _reconstruct_visual_lines(self, words: List[Tuple]) -> List[Dict[str, Any]]:
        visual_lines: List[List[Tuple[pymupdf.Rect, str]]] = []
        for w in words:
            rect = pymupdf.Rect(w[0], w[1], w[2], w[3])
            txt = w[4]
            mid = (rect.y0 + rect.y1) / 2.0
            h = rect.height
            placed = False
            for line in visual_lines:
                l_mid = sum((item[0].y0 + item[0].y1) / 2.0 for item in line) / len(line)
                l_h = sum(item[0].height for item in line) / len(line)
                min_h = min(h, l_h)
                v_overlap = min(rect.y1, max(item[0].y1 for item in line)) - max(rect.y0, min(item[0].y0 for item in line))
                if v_overlap > min_h * 0.45 and abs(mid - l_mid) < min_h * 0.35:
                    line.append((rect, txt))
                    placed = True
                    break
            if not placed:
                visual_lines.append([(rect, txt)])

        visual_lines.sort(key=lambda l: sum((item[0].y0 + item[0].y1) / 2.0 for item in l) / len(l))

        line_objects = []
        for l in visual_lines:
            sw = sorted(l, key=lambda item: item[0].x0)
            u_rect = sw[0][0]
            for it in sw[1:]:
                u_rect = u_rect | it[0]
            line_str = " ".join(item[1] for item in sw)
            line_objects.append({
                "rect": u_rect,
                "text": line_str,
                "words": sw
            })
        return line_objects

    def _expand_person_span_on_line(self, rect: pymupdf.Rect, line_objects: List[Dict[str, Any]]) -> pymupdf.Rect:
        if not line_objects:
            return rect

        rect_mid_y = (rect.y0 + rect.y1) / 2.0
        best_line = None
        min_v_dist = float("inf")
        for line in line_objects:
            l_rect = line["rect"]
            l_mid_y = (l_rect.y0 + l_rect.y1) / 2.0
            v_dist = abs(rect_mid_y - l_mid_y)
            if v_dist < max(18.0, l_rect.height * 0.6) and v_dist < min_v_dist:
                min_v_dist = v_dist
                best_line = line

        if best_line is None:
            return rect

        words = best_line["words"]
        if not words:
            return rect

        intersecting_indices = []
        for i, (w_rect, w_txt) in enumerate(words):
            inter = rect & w_rect
            if not inter.is_empty and inter.get_area() > 0.15 * min(rect.get_area(), w_rect.get_area()):
                intersecting_indices.append(i)
            elif abs((w_rect.y0 + w_rect.y1)/2.0 - rect_mid_y) < max(12.0, w_rect.height * 0.5):
                if not (w_rect.x1 < rect.x0 - 5 or w_rect.x0 > rect.x1 + 5):
                    intersecting_indices.append(i)

        if not intersecting_indices:
            closest_idx = None
            closest_dist = float("inf")
            for i, (w_rect, w_txt) in enumerate(words):
                dist = abs((w_rect.x0 + w_rect.x1)/2.0 - (rect.x0 + rect.x1)/2.0)
                if dist < closest_dist and dist < 40.0:
                    closest_dist = dist
                    closest_idx = i
            if closest_idx is not None:
                intersecting_indices = [closest_idx]
            else:
                return rect

        min_idx = min(intersecting_indices)
        max_idx = max(intersecting_indices)

        honorific_re = re.compile(
            r"^(?:Mr|Mrs|Ms|Miss|Master|Baby|Shri|Smt|Herr|Frau|Fr\.|M\.|Mme|Mlle|Monsieur|Madame|Mademoiselle|Sr\.|Sra\.|Srta\.|Señor|Señora|Señorita|Sig\.|Sig\.ra|Signor|Signora|Dott\.|Dott\.ssa|Dr|Prof)\.?$",
            re.IGNORECASE
        )
        initial_re = re.compile(r"^[A-ZÄÖÜÁÉÍÓÚÑ]\.?$")
        name_token_re = re.compile(r"^[A-ZÄÖÜÁÉÍÓÚÑ][a-zA-Zäöüßáéíóúñ\-'.]*$")
        stopwords = {
            "AGE", "SEX", "MALE", "FEMALE", "DATE", "ADDRESS", "PHONE", "HOSPITAL", "CLINIC",
            "RX", "RP", "TAB", "TABS", "TABLET", "TABLETS", "REFILL", "SIG", "DOSE", "DAILY",
            "DIRECTIONS", "PATIENT", "NAME", "PT", "MRN", "UHID", "OPD", "IPD", "YEARS", "YRS",
            "Y", "M", "F", "NORMAL", "DIAGNOSIS", "DEPARTMENT", "SURGERY", "MEDICINE", "REF", "BY"
        }

        def is_valid_name_token(token_str: str) -> bool:
            clean = token_str.strip()
            clean_alnum = re.sub(r"^[^\w]+|[^\w]+$", "", clean)
            if not clean_alnum:
                return False
            upper = clean_alnum.upper()
            if upper in stopwords:
                return False
            if is_whitelisted(clean):
                return False
            for patterns in KEY_VALUE_ANCHORS.values():
                for pat in patterns:
                    if re.search(pat, clean, re.IGNORECASE):
                        if not honorific_re.match(clean):
                            return False
            if honorific_re.match(clean) or initial_re.match(clean) or name_token_re.match(clean):
                return True
            return False

        curr_min = min_idx
        for idx in range(min_idx - 1, -1, -1):
            w_rect, w_txt = words[idx]
            next_rect, _ = words[idx + 1]
            gap = next_rect.x0 - w_rect.x1
            if gap > 35:
                break
            if is_valid_name_token(w_txt):
                curr_min = idx
            else:
                break

        curr_max = max_idx
        for idx in range(max_idx + 1, len(words)):
            w_rect, w_txt = words[idx]
            prev_rect, _ = words[idx - 1]
            gap = w_rect.x0 - prev_rect.x1
            if gap > 35:
                break
            if is_valid_name_token(w_txt):
                curr_max = idx
            else:
                break

        expanded_rect = pymupdf.Rect(rect)
        for idx in range(curr_min, curr_max + 1):
            expanded_rect = expanded_rect | words[idx][0]

        return expanded_rect

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
            line_objects = self._reconstruct_visual_lines(words)

            # 1. Regex pattern scanning on line-level text
            for category, patterns in all_rules:
                for pattern in patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        matched_str = match.group(0).strip()
                        # Search for instances of the matched phrase on the page
                        rects = page.search_for(matched_str)
                        for r in rects:
                            if category in ("PATIENT_NAME", "DOCTOR_INFO"):
                                r = self._expand_person_span_on_line(r, line_objects)
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
                        if cand["category"] == "PATIENT_NAME":
                            r = self._expand_person_span_on_line(r, line_objects)
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
