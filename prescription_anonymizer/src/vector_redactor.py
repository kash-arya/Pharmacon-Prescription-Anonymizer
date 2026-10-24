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
        _COMPILED_WHITELIST,
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
        _COMPILED_WHITELIST,
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
        all_whitelisted_rects: List[pymupdf.Rect] = []

        for page_idx, page in enumerate(doc):
            words = page.get_text("words")
            full_text = page.get_text()
            line_objects = self._reconstruct_visual_lines(words)

            # Detect Clinical Zone (Rx / Rp.) boundaries
            clinical_hdr_re = re.compile(
                r"^\s*(?:Rx\b|Rp\.?\b|Verordnungen\b|Medications?\b|Prescription\b|Diagnosis\b)",
                re.IGNORECASE
            )
            numbered_med_re = re.compile(
                r"^\s*1\.\s+(?:Tab\.?|Cap\.?|Capsule|Tablet|Inhaler|[A-Z][a-z]{3,})\b",
                re.IGNORECASE
            )
            clinical_y0 = None
            for l in line_objects:
                if clinical_hdr_re.search(l["text"]) or numbered_med_re.search(l["text"]):
                    clinical_y0 = l["rect"].y0
                    break
            if clinical_y0 is None:
                for w in words:
                    if re.match(r"^(?:Rx|Rp\.?)$", w[4], re.I):
                        clinical_y0 = w[1]
                        break
            if clinical_y0 is None:
                clinical_y0 = page.rect.height * 0.40

            footer_marker_re = re.compile(
                r"\b(?:Prescriber(?:'s|’s)?\s*Signature|Signature\s+of\s+Prescriber|Doctor(?:'s|’s)?\s*Signature|"
                r"Physician(?:'s|’s)?\s*Signature|Doctor\s+Name\b|Doctor's\s+Stamp|Doctor\s+Signature\s*&|"
                r"Unterschrift(?:\s+des\s+Arztes)?\b|Firma(?:\s+(?:del\s+)?m[eé]dico)?|Certifying\s+Practitioner|"
                r"Pharmacy\s+Dispenser|For\s+Dispenser|Dispense\s+Date|Date\s+of\s+Prescribing|GMC\s*No|AHPRA\s*No)\b",
                re.IGNORECASE
            )
            footer_candidates = [
                l["rect"].y0 for l in line_objects
                if l["rect"].y0 > clinical_y0 + 15 and footer_marker_re.search(l["text"])
            ]
            footer_y0 = min(footer_candidates) if footer_candidates else page.rect.height * 0.82

            stamp_box_re = re.compile(
                r"\b(?:Vertragsarztstempel|Arztstempel|Praxisstempel|Stamp\s*&\s*Signature|Doctor(?:'s|’s)?\s*Stamp)\b",
                re.IGNORECASE
            )
            right_stamp_column_x0 = page.rect.width
            stamp_box_y0 = None
            for l in line_objects:
                if stamp_box_re.search(l["text"]) and l["rect"].x0 > page.rect.width * 0.45:
                    right_stamp_column_x0 = l["rect"].x0 - 15.0
                    stamp_box_y0 = l["rect"].y0 - 5.0
                    break

            def is_in_clinical_zone(r: pymupdf.Rect) -> bool:
                return (clinical_y0 - 4.0 <= r.y0 and r.y1 <= footer_y0 + 4.0 and r.x0 < right_stamp_column_x0)

            page_whitelisted: List[pymupdf.Rect] = []
            for l in line_objects:
                if is_in_clinical_zone(l["rect"]):
                    page_whitelisted.append(l["rect"])
            all_whitelisted_rects.extend(page_whitelisted)

            def add_target(rect: pymupdf.Rect, category: str, text: str, engine: str, conf: float = 0.98):
                if is_in_clinical_zone(rect):
                    return
                for wr in page_whitelisted:
                    inter = rect & wr
                    if not inter.is_empty and inter.get_area() > 0.05 * min(rect.get_area(), wr.get_area()):
                        return
                for existing in targets:
                    if existing.page_number == page_idx + 1:
                        ex_r = pymupdf.Rect(existing.bbox.x0, existing.bbox.y0, existing.bbox.x1, existing.bbox.y1)
                        inter = rect & ex_r
                        if not inter.is_empty and inter.get_area() > 0.75 * max(rect.get_area(), ex_r.get_area()):
                            return
                targets.append(RedactionTarget(
                    page_number=page_idx + 1,
                    category=category,
                    raw_text=text.strip() if text else category,
                    bbox=BoundingBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1),
                    confidence=conf,
                    engine=engine
                ))

            # 1. Doctor Stamp Box Target (Muster 16 right-side column)
            if stamp_box_y0 is not None:
                for l in line_objects:
                    if l["rect"].x0 >= right_stamp_column_x0 - 5.0 and l["rect"].y1 >= stamp_box_y0:
                        add_target(l["rect"], "DOCTOR_STAMP", l["text"], engine="stamp_box_detector")

            # 2. Key-Value Anchor Processing outside Clinical Zone
            for l_idx, line in enumerate(line_objects):
                if is_in_clinical_zone(line["rect"]):
                    continue
                if stamp_box_y0 is not None and line["rect"].x0 >= right_stamp_column_x0 - 5.0 and line["rect"].y1 >= stamp_box_y0:
                    continue

                l_text = line["text"]
                l_words = line["words"]
                line_mid = (line["rect"].y0 + line["rect"].y1) / 2.0

                found_anchors = []
                for cat, patterns in KEY_VALUE_ANCHORS.items():
                    for pat in patterns:
                        for m in re.finditer(pat, l_text, re.IGNORECASE):
                            found_anchors.append((m.start(), m.end(), cat, m.group(0)))

                found_anchors.sort(key=lambda x: (x[0], -(x[1] - x[0])))
                non_overlapping = []
                last_end = -1
                for s, e, cat, kw in found_anchors:
                    if s >= last_end:
                        non_overlapping.append((s, e, cat, kw))
                        last_end = e

                if non_overlapping:
                    for i, (s, e, cat, kw) in enumerate(non_overlapping):
                        next_bound = len(l_text)
                        if i + 1 < len(non_overlapping):
                            next_bound = min(next_bound, non_overlapping[i+1][0])

                        # Segment end stops at next pipe delimiter or next anchor
                        pipe_pos = l_text.find("|", e)
                        if pipe_pos != -1 and pipe_pos < next_bound:
                            seg_end = pipe_pos + 1
                        else:
                            seg_end = next_bound

                        seg_text = l_text[s:seg_end].strip()
                        raw_rects = page.search_for(seg_text)
                        matched_r = [r for r in raw_rects if abs((r.y0 + r.y1)/2.0 - line_mid) < 8.0]
                        if matched_r:
                            u_r = matched_r[0]
                            for mr in matched_r[1:]:
                                u_r = u_r | mr
                            add_target(u_r, cat, seg_text, engine="vector_kv_anchor")
                        else:
                            # Fallback keyword + value bounding box
                            kw_rects = [r for r in page.search_for(kw) if abs((r.y0 + r.y1)/2.0 - line_mid) < 8.0]
                            if kw_rects:
                                ak_r = kw_rects[0]
                                end_x = line["rect"].x1
                                if i + 1 < len(non_overlapping):
                                    nxt_rects = [r for r in page.search_for(non_overlapping[i+1][3]) if abs((r.y0 + r.y1)/2.0 - line_mid) < 8.0]
                                    if nxt_rects:
                                        end_x = nxt_rects[0].x0 - 4.0
                                for w in l_words:
                                    if w[1] == "|" and w[0].x0 > ak_r.x1 and w[0].x0 < end_x:
                                        end_x = w[0].x1 + 2.0
                                        break
                                u_r = pymupdf.Rect(ak_r.x0 - 2, line["rect"].y0 - 2, end_x, line["rect"].y1 + 2)
                                add_target(u_r, cat, seg_text, engine="vector_kv_anchor")

                        # Multi-line value under label (e.g. Patient: \n Name)
                        val_str = l_text[e:seg_end].strip().lstrip(":-=~+ ")
                        is_sublabel = bool(re.search(r"^\s*\(.*?(?:forename|address|vorname|geb|name|anschrift).*?\)\s*$", val_str, re.IGNORECASE))
                        if not val_str or is_sublabel:
                            if l_idx + 1 < len(line_objects):
                                nxt_l = line_objects[l_idx + 1]
                                if not is_in_clinical_zone(nxt_l["rect"]):
                                    nxt_txt = nxt_l["text"].strip()
                                    if not any(pat.search(nxt_txt) for pat in _COMPILED_WHITELIST) and not clinical_hdr_re.search(nxt_txt):
                                        add_target(nxt_l["rect"], cat, nxt_txt, engine="vector_multiline_anchor")

            # 3. Header Facility & Provider Rule (above clinical_y0)
            facility_re = re.compile(
                r"\b(?:Hospital|Clinic|Imaging|Diagnostics?|Institute|Centre|Center|Medical|Healthcare|Alliance|Department|Ambulatory|Surgery|Practice|Optical|Optometry|Vision|Eye\s+Care|Dental|Pharmacy|Chemist|Praxis|Klinik|Apotheke)\b",
                re.IGNORECASE
            )
            degree_re = re.compile(
                r"\b(?:M\.?B\.?B\.?S|M\.?D|M\.?S|D\.?M\.?R\.?D|D\.?N\.?B|F\.?R\.?C\.?R|Regn?\.?|MCI|NMC|MBChB|MRCGP|FRACGP|AHPRA)\b",
                re.IGNORECASE
            )
            for line in line_objects:
                if line["rect"].y1 < clinical_y0:
                    l_text = line["text"]
                    if facility_re.search(l_text) or degree_re.search(l_text):
                        if not any(pat.search(l_text) for pat in _COMPILED_WHITELIST):
                            add_target(line["rect"], "CLINIC_FACILITY", l_text, engine="header_facility_rule")

            # 4. Statutory National Identifiers
            for reg, cat in STATUTORY_ID_PATTERNS:
                for match in re.finditer(reg, full_text, re.IGNORECASE):
                    m_str = match.group(0).strip()
                    if not is_whitelisted(m_str) and len(m_str) > 2:
                        rects = page.search_for(m_str)
                        if not rects and match.groups():
                            rects = page.search_for(match.group(1).strip())
                        for r in rects:
                            add_target(r, cat, m_str, engine="statutory_rule")

            # 5. Full address lines and standalone regexes outside clinical zone
            address_line_re = re.compile(
                r"\b\d+\s+[A-Za-z0-9\-.,]+(?:\s+[A-Za-z0-9\-.,]+)*\s+(?:Street|St|Avenue|Ave|Drive|Dr|Road|Rd|Boulevard|Blvd|Court|Ct|Lane|Ln|Terrace|Colony|Market|Chowk|Way|Square|Sq|Place|Pl|Straße|Str|Weg|Platz|Allee|Gasse|Rue|Bd|Calle|Paseo|Via|Piazza)\b",
                re.IGNORECASE
            )
            for line in line_objects:
                if not is_in_clinical_zone(line["rect"]) and address_line_re.search(line["text"]):
                    if not any(pat.search(line["text"]) for pat in _COMPILED_WHITELIST):
                        add_target(line["rect"], "ADDRESS", line["text"], engine="address_line_rule")

            standalone_regs = [
                ("URL", re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)),
                ("CONTACT_FAX_PHONE", re.compile(r"\b(?:Fax|Tel|Phone|Mobile)\.?\s*[:\-=~+]?\s*[\d\-\s/()+]{5,20}\b|\b\d{3}[\-\s.]\d{3}[\-\s.]\d{4}\b", re.IGNORECASE)),
                ("ADDRESS_POSTCODE", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b|\bVIC\s+\d{4}\b|\b\d{5}(?:-\d{4})?\b")),
                ("POLICY_NUMBER", re.compile(r"\bPolicy\s*[:\-=~+]?\s*[A-Z0-9-]+\b", re.IGNORECASE)),
                ("SCHEME_HEADER", re.compile(r"\b(?:AUSTRALIAN\s+PHARMACEUTICAL\s+BENEFITS\s+SCHEME\s*\(PBS\)|Standard\s+Ambulatory\s+Prescription\s+Form\s*[—·]\s*Medicare\s+Australia|PRESCRIPTION\s+FORM\s*-\s*FP10SS)\b", re.IGNORECASE)),
            ]
            for cat, reg in standalone_regs:
                for match in reg.finditer(full_text):
                    m_str = match.group(0).strip()
                    if not any(pat.search(m_str) for pat in _COMPILED_WHITELIST) and len(m_str) > 2:
                        for r in page.search_for(m_str):
                            add_target(r, cat, m_str, engine="standalone_regex")

            # 6. Prescriber / Doctor / Pharmacist Signatures & Footers
            for line in line_objects:
                if line["rect"].y0 >= footer_y0 - 5.0:
                    l_text = line["text"]
                    if not any(pat.search(l_text) for pat in _COMPILED_WHITELIST):
                        add_target(line["rect"], "DOCTOR_INFO", l_text, engine="footer_zone_rule")
                elif not is_in_clinical_zone(line["rect"]):
                    if re.match(r"^[\s|•·—_-]+$", line["text"]):
                        add_target(line["rect"], "DELIMITER", line["text"], engine="delimiter_cleanup")

            # 7. Local spaCy NER strictly outside Clinical Zone
            ner_candidates = self.ner_engine.find_entities(full_text)
            for cand in ner_candidates:
                m_str = cand["text"]
                if any(pat.search(m_str) for pat in _COMPILED_WHITELIST) or is_whitelisted(m_str) or len(m_str) < 3:
                    continue
                for r in page.search_for(m_str):
                    if is_in_clinical_zone(r):
                        continue
                    if cand["category"] == "PATIENT_NAME":
                        r = self._expand_person_span_on_line(r, line_objects)
                    add_target(r, cand["category"], m_str, engine="spacy_ner")

        self._last_whitelisted_rects = all_whitelisted_rects
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
        whitelisted_rects = getattr(self, "_last_whitelisted_rects", [])

        # Apply redactions page by page with 0 pt overlap against whitelisted clinical text
        for t in targets:
            page = doc[t.page_number - 1]
            rect = pymupdf.Rect(
                max(0, t.bbox.x0 - self.padding_x),
                max(0, t.bbox.y0 - self.padding_y),
                t.bbox.x1 + self.padding_x,
                t.bbox.y1 + self.padding_y
            )

            # Clamp rect away from whitelisted_rects
            skip = False
            for wr in whitelisted_rects:
                inter = rect & wr
                if inter.is_empty:
                    continue
                if inter.get_area() >= 0.85 * rect.get_area():
                    skip = True
                    break
                if rect.y0 < wr.y0 and rect.y1 > wr.y0:
                    rect.y1 = max(rect.y0, wr.y0 - 0.5)
                elif rect.y1 > wr.y1 and rect.y0 < wr.y1:
                    rect.y0 = min(rect.y1, wr.y1 + 0.5)
                if rect.x0 < wr.x0 and rect.x1 > wr.x0:
                    rect.x1 = max(rect.x0, wr.x0 - 0.5)
                elif rect.x1 > wr.x1 and rect.x0 < wr.x1:
                    rect.x0 = min(rect.x1, wr.x1 + 0.5)

            if not skip and rect.width > 2 and rect.height > 2:
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

        from .forensics import ForensicAuditor
        conf = ForensicAuditor.evaluate_redaction_confidence(
            report={"redacted_targets": [t.model_dump() for t in targets]},
            extracted_text=full_doc_text,
            is_raster=False
        )

        return {
            "input_file": input_pdf_path,
            "output_file": output_pdf_path,
            "jurisdiction": resolved_jur,
            "total_entities_redacted": len(targets),
            "total_redactions_applied": len(targets),
            "redacted_targets": [t.model_dump() for t in targets],
            "confidence_score": conf["confidence_score"],
            "triage_status": conf["triage_status"],
            "review_reasons": conf["review_reasons"]
        }
