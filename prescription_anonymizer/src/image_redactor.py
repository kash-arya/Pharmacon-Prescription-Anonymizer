"""
Local Image Redactor Module for Scanned Prescriptions and Medical Forms.

Orchestrates local OCR, multi-column form parsing, regex entity extraction,
local spaCy NER, and permanent black-box visual masking across scanned raster PDFs
and smartphone camera images, strictly preserving clinical pharmacology and findings.
Delegates specialized subtasks to focused single-responsibility modules:
- ocr_preprocess: Tessdata resolution, multilingual OCR, visual line reconstruction, CLAHE contrast enhancement
- barcode_detector: 2D QR and 1D barcode detection with Sobel gradient validation
- form_parser: 2D multi-row column table parsing, spatial handwriting-ink detection, and name-token expansion
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import re
import io
import gc
import pymupdf
import cv2

cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
import numpy as np
from PIL import Image

try:
    from .entities import (
        BoundingBox,
        RedactionTarget,
        KEY_VALUE_ANCHORS,
        CLINICAL_WHITELIST_PATTERNS,
        STATUTORY_ID_PATTERNS,
        is_whitelisted,
        _COMPILED_WHITELIST,
        is_clinical_barrier,
        is_legal_disclaimer,
        BOUNDARY_LABEL_RE,
    )
    from .vlm_grounder import MultimodalVisualGrounder
    from .jurisdiction import JurisdictionDetector
    from .ner_engine import LocalNEREngine
    from .ocr_preprocess import (
        resolve_tessdata_path,
        get_ocr_languages,
        reconstruct_visual_lines,
        get_enhanced_fullpage_ocr,
        normalize_ink_contrast,
    )
    from .barcode_detector import detect_qr_and_barcodes
    from .form_parser import (
        expand_person_span_on_line,
        detect_handwritten_fill_in_zone,
        parse_form_anchors_on_lines,
    )
    from .forensics import ForensicAuditor
except (ImportError, ValueError):
    from src.entities import (
        BoundingBox,
        RedactionTarget,
        KEY_VALUE_ANCHORS,
        CLINICAL_WHITELIST_PATTERNS,
        STATUTORY_ID_PATTERNS,
        is_whitelisted,
        _COMPILED_WHITELIST,
        is_clinical_barrier,
        is_legal_disclaimer,
        BOUNDARY_LABEL_RE,
    )
    from src.vlm_grounder import MultimodalVisualGrounder
    from src.jurisdiction import JurisdictionDetector
    from src.ner_engine import LocalNEREngine
    from src.ocr_preprocess import (
        resolve_tessdata_path,
        get_ocr_languages,
        reconstruct_visual_lines,
        get_enhanced_fullpage_ocr,
        normalize_ink_contrast,
    )
    from src.barcode_detector import detect_qr_and_barcodes
    from src.form_parser import (
        expand_person_span_on_line,
        detect_handwritten_fill_in_zone,
        parse_form_anchors_on_lines,
    )
    from src.forensics import ForensicAuditor


class LocalImageRedactor:
    """
    Resolution-invariant OCR & visual redaction orchestrator for scanned raster PDFs
    and smartphone prescription images.
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

    def __init__(
        self,
        tessdata_dir: Optional[str] = None,
        padding_ratio: float = 0.05,
        dpi: int = 150,
        enable_visual_grounder: bool = True
    ):
        self.tessdata_dir = resolve_tessdata_path(tessdata_dir)
        self.padding_ratio = padding_ratio
        self.dpi = dpi
        self.enable_visual_grounder = enable_visual_grounder
        self.grounder = MultimodalVisualGrounder() if enable_visual_grounder else None
        self.ner_engine = LocalNEREngine()

    def is_whitelisted(self, text: str) -> bool:
        """Determines if a candidate text string contains protected clinical pharmacology or findings."""
        return is_whitelisted(text)

    # Backward-compatibility helper delegates
    def _detect_qr_and_barcodes(
        self,
        page: pymupdf.Page,
        bgr: Optional[np.ndarray] = None,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None
    ) -> List[Tuple[pymupdf.Rect, str]]:
        return detect_qr_and_barcodes(page, bgr=bgr, scale_x=scale_x, scale_y=scale_y)

    def _get_enhanced_fullpage_ocr(
        self,
        page: pymupdf.Page,
        h_ratio: float = 0.45,
        lang: Optional[str] = None
    ) -> Tuple[List[Any], pymupdf.Page, pymupdf.TextPage, pymupdf.Document]:
        return get_enhanced_fullpage_ocr(
            page, dpi=self.dpi, h_ratio=h_ratio, lang=lang, tessdata_dir=self.tessdata_dir
        )

    _get_enhanced_header_ocr = _get_enhanced_fullpage_ocr

    def _expand_person_span_on_line(self, rect: pymupdf.Rect, line_objects: List[Dict[str, Any]]) -> pymupdf.Rect:
        return expand_person_span_on_line(rect, line_objects, is_whitelisted_fn=self.is_whitelisted)

    def _reconstruct_visual_lines(self, words: List[Tuple]) -> List[Dict[str, Any]]:
        return reconstruct_visual_lines(words)

    def _detect_handwritten_fill_in_zone(
        self,
        page_bgr: np.ndarray,
        zone_rect: pymupdf.Rect,
        scale_px_x: float,
        scale_px_y: float,
        whitelisted_rects: Optional[List[pymupdf.Rect]] = None
    ) -> Optional[pymupdf.Rect]:
        return detect_handwritten_fill_in_zone(
            page_bgr, zone_rect, scale_px_x, scale_px_y,
            whitelisted_rects=whitelisted_rects if whitelisted_rects is not None else getattr(self, "_last_whitelisted_rects", [])
        )

    def _parse_form_anchors_on_lines(
        self,
        line_objects: List[Dict[str, Any]],
        search_page: pymupdf.Page,
        textpage: pymupdf.TextPage,
        page_bgr: np.ndarray,
        scale_px_x: float,
        scale_px_y: float,
        add_target_fn: Any,
        suffix: str = ""
    ):
        return parse_form_anchors_on_lines(
            line_objects=line_objects,
            search_page=search_page,
            textpage=textpage,
            page_bgr=page_bgr,
            scale_px_x=scale_px_x,
            scale_px_y=scale_px_y,
            add_target_fn=add_target_fn,
            is_whitelisted_fn=self.is_whitelisted,
            whitelisted_rects=getattr(self, "_last_whitelisted_rects", []),
            suffix=suffix
        )

    def find_pii_targets_for_page(self, page: pymupdf.Page, page_num: int) -> List[RedactionTarget]:
        """
        Scans a single raster page using dynamic multi-column layout parsing,
        regex entity recognition, and strict clinical whitelist protection.
        """
        max_pt = max(page.rect.width, page.rect.height)
        is_low_mem = os.environ.get("RENDER_LOW_MEMORY") == "1"
        if is_low_mem:
            effective_dpi = 135
        else:
            if max_pt < 550:
                effective_dpi = 300
            elif max_pt < 900:
                effective_dpi = self.dpi if (page.rect.width >= 550 and page.rect.height >= 700) else 200
            elif max_pt <= 2000:
                effective_dpi = self.dpi
            else:
                effective_dpi = 72

        tmp_ocr_doc = None
        if getattr(self, "_current_image_ocr_page", None) is not None:
            search_page_ref = self._current_image_ocr_page
            tp = search_page_ref.get_textpage_ocr(language="eng", dpi=effective_dpi, tessdata=self.tessdata_dir)
            words = search_page_ref.get_text("words", textpage=tp)
            full_text = search_page_ref.get_text(textpage=tp)
        elif is_low_mem and not getattr(self, "_is_current_file_image", False):
            # Fast Pre-OCR Contrast Normalization on page raster
            pix = page.get_pixmap(dpi=effective_dpi)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                page_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                page_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                page_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            del pix, arr

            page_bgr = normalize_ink_contrast(page_bgr, gamma=1.6)

            lab = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
            bg = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
            norm_l = np.uint8(np.clip((l.astype(np.float32) / np.maximum(1.0, bg.astype(np.float32))) * 255.0, 0, 255))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(norm_l)
            lab = cv2.merge((cl, a, b))
            page_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            del lab, l, a, b, cl, kernel, bg, norm_l

            _, enc = cv2.imencode(".png", page_bgr)
            tmp_ocr_doc = pymupdf.open()
            tmp_ocr_page = tmp_ocr_doc.new_page(width=page.rect.width, height=page.rect.height)
            tmp_ocr_page.insert_image(page.rect, stream=enc.tobytes())
            del enc

            search_page_ref = tmp_ocr_page
            tp = tmp_ocr_page.get_textpage_ocr(language="eng", dpi=effective_dpi, tessdata=self.tessdata_dir)
            words = tmp_ocr_page.get_text("words", textpage=tp)
            full_text = tmp_ocr_page.get_text(textpage=tp)
        else:
            search_page_ref = page
            tp = page.get_textpage_ocr(language="eng", dpi=effective_dpi, tessdata=self.tessdata_dir)
            words = page.get_text("words", textpage=tp)
            full_text = page.get_text(textpage=tp)

        eu_marker_regex = re.compile(
            r"[äöüßéèêëàâîïôûùçñáíóúÁÉÍÓÚÑÄÖÜ]|\b(?:Rezept|Krankenkasse|Geb|Ausstellungsdatum|Vorname|Unterschrift|Arzt|Receta|Ordonnance|Nom|Prénom|Calle|Straße)\b",
            re.IGNORECASE
        )
        is_eu = bool(eu_marker_regex.search(full_text))
        all_langs = "eng"
        if not is_low_mem and (len(words) < 20 or is_eu):
            all_langs = get_ocr_languages(self.tessdata_dir)
            if all_langs != "eng":
                tp = search_page_ref.get_textpage_ocr(language=all_langs, dpi=effective_dpi, tessdata=self.tessdata_dir)
                words = search_page_ref.get_text("words", textpage=tp)
                full_text = search_page_ref.get_text(textpage=tp)

        self._last_extracted_text = full_text
        self._last_extracted_words = words

        # Build list of protected clinical bounding boxes
        whitelisted_rects: List[pymupdf.Rect] = []
        for w in words:
            if self.is_whitelisted(w[4]):
                whitelisted_rects.append(pymupdf.Rect(w[0], w[1], w[2], w[3]))

        for rx in _COMPILED_WHITELIST:
            for match in rx.finditer(full_text):
                match_str = match.group(0).strip()
                if match_str:
                    for r in search_page_ref.search_for(match_str, textpage=tp):
                        whitelisted_rects.append(r)
        self._last_whitelisted_rects = whitelisted_rects

        targets: List[RedactionTarget] = []
        recorded_rects: List[pymupdf.Rect] = []
        active_lines: List[Dict[str, Any]] = []

        def can_redact(rect: pymupdf.Rect, raw_text: str) -> bool:
            if raw_text.startswith("http") or raw_text == "QR_CODE":
                return True
            if raw_text in ("BARCODE", "BARCODE_OR_QR"):
                # NEVER allow an un-decoded BARCODE candidate to bypass whitelisted_rects.
                for wr in whitelisted_rects:
                    if not (rect & wr).is_empty and (rect & wr).get_area() > 0:
                        return False
                letter_word_count = 0
                for w in words:
                    w_rect = pymupdf.Rect(w[0], w[1], w[2], w[3])
                    inter = rect & w_rect
                    if not inter.is_empty and inter.get_area() > 0.25 * w_rect.get_area():
                        char_w = w_rect.width / max(1, len(w[4]))
                        if re.search(r"[A-Za-z]{3,}", w[4]) and char_w < 35:
                            letter_word_count += 1
                            if letter_word_count >= 2:
                                return False
                return True
            if self.is_whitelisted(raw_text):
                return False
            rect_mid = (rect.y0 + rect.y1) / 2.0
            for wr in whitelisted_rects:
                wr_mid = (wr.y0 + wr.y1) / 2.0
                inter = rect & wr
                if inter.is_empty:
                    continue
                # If on different lines vertically, bounding box overlap is an OCR box artifact
                if abs(rect_mid - wr_mid) > max(15.0, min(rect.height, wr.height) * 0.35):
                    split_y = (rect_mid + wr_mid) / 2.0
                    if rect_mid > wr_mid:
                        rect.y0 = max(rect.y0, split_y)
                    else:
                        rect.y1 = min(rect.y1, split_y)
                    continue
                # If visual target (signature or stamp) overlaps whitelisted printed text horizontally, clip it instead of dropping
                if raw_text in ("[HANDWRITTEN_SIGNATURE]", "[CHROMATIC_STAMP_SEAL]"):
                    if rect.x1 - wr.x1 > 20:
                        rect.x0 = max(rect.x0, wr.x1 + 2)
                        continue
                    elif wr.x0 - rect.x0 > 20:
                        rect.x1 = min(rect.x1, wr.x0 - 2)
                        continue

                if inter.get_area() > 0.15 * min(rect.get_area(), wr.get_area()):
                    return False
            return True

        def add_target(rect: pymupdf.Rect, category: str, text: str, engine: str = "ocr_dynamic"):
            if not rect.is_valid or rect.get_area() <= 1.0:
                return

            if category in ("PATIENT_NAME", "PATIENT_NAME_VALUE", "PATIENT_NAME_VALUE_CLAHE",
                            "PATIENT_NAME_VALUE_2D", "PATIENT_HONORIFIC", "PATIENT_HONORIFIC_ENHANCED",
                            "DOCTOR_NAME", "DOCTOR_NAME_ENHANCED") or \
               (category.startswith("PATIENT_NAME") and not category.endswith("_LABEL")) or \
               (category.startswith("DOCTOR_NAME") and not category.endswith("_LABEL")):
                rect = expand_person_span_on_line(rect, active_lines, is_whitelisted_fn=self.is_whitelisted)

            # Mechanism 3: Bidirectional Segment Completion & Scale-Invariant Glyph Padding
            pii_prefixes = (
                "PATIENT", "DOCTOR", "CLINIC", "ADDRESS", "CONTACT", "IDENTIFIER",
                "PHONE", "EMAIL", "UHID", "MRN", "OPD", "IPD", "STATUTORY", "ID_TOKEN",
                "HEADER_PROVIDER", "HEADER_INSTITUTIONAL", "DISPENSING_PHARMACY", "PERSON",
                "CITY_STATE", "PROVIDER_BLOCK"
            )
            is_pii_rule = any(category.startswith(pfx) or pfx in category for pfx in pii_prefixes)
            if is_pii_rule and not category.endswith("_LABEL") and active_lines and text not in ("[HANDWRITTEN_SIGNATURE]", "[CHROMATIC_STAMP_SEAL]"):
                r_mid_y = (rect.y0 + rect.y1) / 2.0
                for line in active_lines:
                    l_rect = line["rect"]
                    if abs((l_rect.y0 + l_rect.y1) / 2.0 - r_mid_y) < max(12.0, l_rect.height * 0.6):
                        for seg in line.get("segments", []):
                            if not (seg["x1"] < rect.x0 - 4 or seg["x0"] > rect.x1 + 4):
                                new_x0 = min(rect.x0, seg["x0"])
                                new_x1 = max(rect.x1, seg["x1"])

                                # Minus protected clinical tokens
                                for wr in whitelisted_rects:
                                    if abs((wr.y0 + wr.y1) / 2.0 - r_mid_y) < max(10.0, l_rect.height * 0.5):
                                        if wr.x0 < new_x1 and wr.x1 > new_x0:
                                            if (rect.x0 + rect.x1) / 2.0 < (wr.x0 + wr.x1) / 2.0:
                                                new_x1 = min(new_x1, wr.x0 - 2.0)
                                            else:
                                                new_x0 = max(new_x0, wr.x1 + 2.0)

                                if new_x1 > new_x0:
                                    rect.x0 = new_x0
                                    rect.x1 = new_x1

                                med_lh = seg.get("line_height", l_rect.height)
                                pad_y = max(3.0, 0.22 * med_lh)
                                rect.y0 = max(0.0, rect.y0 - pad_y)
                                rect.y1 = min(page.rect.height, rect.y1 + pad_y)
                                break
                        break

            if not can_redact(rect, text):
                return
            if not rect.is_valid or rect.get_area() <= 0:
                return
            for idx, r in enumerate(recorded_rects):
                inter = rect & r
                if not inter.is_empty:
                    inter_a = inter.get_area()
                    if inter_a > 0.70 * r.get_area() and rect.get_area() > r.get_area() * 1.3:
                        merged = r | rect
                        recorded_rects[idx] = merged
                        targets[idx].bbox = BoundingBox(x0=merged.x0, y0=merged.y0, x1=merged.x1, y1=merged.y1)
                        if "LABEL" in targets[idx].category and "LABEL" not in category:
                            targets[idx].category = category
                            targets[idx].raw_text = f"{targets[idx].raw_text} {text}".strip()
                        return
                    if inter_a > 0.75 * min(rect.get_area(), r.get_area()):
                        if text and text not in (targets[idx].raw_text or ""):
                            if targets[idx].raw_text in ("[HANDWRITTEN_INK]", None, ""):
                                targets[idx].raw_text = text
                            else:
                                targets[idx].raw_text = f"{targets[idx].raw_text} {text}".strip()
                        return
            recorded_rects.append(rect)
            targets.append(RedactionTarget(
                page_number=page_num,
                category=category,
                raw_text=text.strip() if text else category,
                bbox=BoundingBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1),
                confidence=0.98,
                engine=engine
            ))

        # Render page pixmap once for visual analysis (barcodes, QR, stamps, signatures)
        if 'page_bgr' not in locals() or page_bgr is None:
            if is_low_mem:
                vis_dpi = 72
            else:
                vis_dpi = 72 if max_pt > 2000 else 150
            pix = page.get_pixmap(dpi=vis_dpi)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                page_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                page_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                page_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

            scale_px_x = page.rect.width / max(1.0, pix.width)
            scale_px_y = page.rect.height / max(1.0, pix.height)
            del pix, arr
        else:
            scale_px_x = page.rect.width / max(1.0, page_bgr.shape[1])
            scale_px_y = page.rect.height / max(1.0, page_bgr.shape[0])

        # 0. Automatic QR Code and Barcode Detection
        code_boxes = detect_qr_and_barcodes(page, bgr=page_bgr, scale_x=scale_px_x, scale_y=scale_px_y)
        for c_rect, payload in code_boxes:
            add_target(c_rect, "BARCODE_OR_QR", payload or "BARCODE_OR_QR", engine="opencv_detector")

        # Reconstruct logical visual lines
        line_objects = reconstruct_visual_lines(words)
        active_lines = line_objects

        # Detect Clinical Zone (Rx/Rp.) and Right-hand Doctor Stamp Box boundaries
        clinical_hdr_re = re.compile(
            r"^\s*(?:Rx\b|Rp\.?\b|R:\b|Disp\.?\b|Sig:?\b|Medikament\b|Verordnungen\b|Medications?\b|Prescription\b|Diagnosis\b)",
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
                if re.match(r"^(?:Rx|Rp\.?|R:|Disp\.?|Sig:?|Medikament)$", w[4], re.IGNORECASE):
                    clinical_y0 = w[1]
                    break

        stamp_box_re = re.compile(
            r"\b(?:Vertragsarztstempel|Arztstempel|Praxisstempel|Stamp\s*&\s*Signature|Doctor(?:'s|’s)?\s*Stamp)\b",
            re.IGNORECASE
        )
        right_stamp_column_x0 = page.rect.width
        for l in line_objects:
            if stamp_box_re.search(l["text"]) and l["rect"].x0 > page.rect.width * 0.45:
                right_stamp_column_x0 = l["rect"].x0 - 15.0
                break

        footer_marker_re = re.compile(
            r"\b(?:Signature|Prescriber|Unterschrift|For\s+Dispenser|Authority\s+approval|Community\s+Pharmacy|"
            r"Doctor(?:'s|’s)?\s*Signature|Signature\s+of\s+Prescriber|Doctor\s+Name\b|GMC\s*No|AHPRA\s*No)\b",
            re.IGNORECASE
        )
        footer_candidates = [
            l["rect"].y0 for l in line_objects
            if l["rect"].y0 > (clinical_y0 or 0) + 15 and footer_marker_re.search(l["text"])
        ]
        footer_y0 = min(footer_candidates) if footer_candidates else page.rect.height * 0.75

        def is_in_clinical_zone(r: pymupdf.Rect) -> bool:
            if clinical_y0 is None:
                return False
            return (clinical_y0 - 5.0 <= r.y0 and r.y1 <= footer_y0 + 5.0 and r.x0 < right_stamp_column_x0)

        # -------------------------------------------------------------------
        # 1. Unified 2D Multi-Column Key-Value Form Field & Ink-Fill Parser
        # -------------------------------------------------------------------
        parse_form_anchors_on_lines(
            line_objects=line_objects,
            search_page=search_page_ref,
            textpage=tp,
            page_bgr=page_bgr,
            scale_px_x=scale_px_x,
            scale_px_y=scale_px_y,
            add_target_fn=add_target,
            is_whitelisted_fn=self.is_whitelisted,
            whitelisted_rects=whitelisted_rects,
            suffix=""
        )

        # Inline pharmacy/clinic dispense rule
        inline_dispense_re = re.compile(
            r"\b(?:dispensing\s+only\s+at|dispensed\s+by)\s+[A-Z][A-Za-z0-9\s',.]+(?:Pharmacy|Clinic|Hospital|Chemist)[A-Za-z0-9\s',.]*",
            re.IGNORECASE
        )
        for match in inline_dispense_re.finditer(full_text):
            m_str = match.group(0).strip()
            if not self.is_whitelisted(m_str) and len(m_str) > 5:
                for r in search_page_ref.search_for(m_str, textpage=tp):
                    add_target(r, "DISPENSING_PHARMACY", m_str, engine="inline_dispense_rule")

        # Anywhere-on-page Pharmacy/Clinic/Healthcare blocks
        pharmacy_block_re = re.compile(
            r"\b(?:[A-ZÄÖÜ][A-Za-z0-9\s',.&-]{2,40}\s+(?:Community\s+Pharmacy|Pharmacy|Chemist|Medical\s+Centre|Medical|Clinic|Hospital|Optical|Optometry|Vision|Eye\s+Care|Dental|Praxis|Klinik|Apotheke))\b",
            re.IGNORECASE
        )
        for match in pharmacy_block_re.finditer(full_text):
            m_str = match.group(0).strip()
            if not self.is_whitelisted(m_str) and len(m_str) > 5:
                for r in search_page_ref.search_for(m_str, textpage=tp):
                    if not is_in_clinical_zone(r) or r.y0 >= footer_y0 - 20:
                        add_target(r, "CLINIC_FACILITY", m_str, engine="pharmacy_block_rule")

        # -------------------------------------------------------------------
        # 2. Universal Standalone Regex Entity Scanners
        # -------------------------------------------------------------------
        standalone_regexes = [
            ("PHONE", re.compile(
                r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b|\b0\d{2,4}[\-\s]?\d{6,8}\b|"
                r"\b(?:Ph|Tel|Phone)?\.?\s*0[23478]\s*\d{4}\s*\d{4}\b|\b\(\d{2}\)\s*\d{4}\s*\d{4}\b|"
                r"\b\(\d{3}\)[\s\-]?[0-9O]{3}[\-\s]?\d{4}\b|\b\d{3}\)[\s\-]?[0-9O]{3}[\-\s]?\d{4}\b|\b\d{3}[\-\s.]\d{3}[\-\s.]\d{4}\b|"
                r"(?:Tel(?:efon)?|Phone|Mobile|Kontakt|Ph)\.?\s*[:\-=~+]?\s*[\d\-\s/()]{5,18}",
                re.IGNORECASE
            )),
            ("DATE", re.compile(r"\b\d{1,2}[/\-.][0-9]{1,2}[/\-.][0-9]{2,4}\b|\b\d{1,2}\.\s*\d{1,2}\.\s*\d{2,4}\b|\b\d{1,2}[\s\-_](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Januar|Februar|März|Mai|Juni|Juli|Oktober|Dezember|Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre|Janvier|Février|Mars|Avril|Mai|Juin|Juillet|Août|Septembre|Octobre|Novembre|Décembre)[a-z]*[\s\-_]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Januar|Februar|März|Mai|Juni|Juli|Oktober|Dezember|Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre|Janvier|Février|Mars|Avril|Mai|Juin|Juillet|Août|Septembre|Octobre|Novembre|Décembre)[a-z]*\s+\d{1,2},?\s+\d{4}\b", re.IGNORECASE)),
            ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE)),
            ("URL", re.compile(r"\bhttps?://[^\s/$.?#].[^\s]*\b", re.IGNORECASE)),
            ("PATIENT_HONORIFIC", re.compile(r"(?i:\b(?:Mr|Mrs|Ms|Miss|Master|Baby|Shri|Smt|Herr|Frau|Fr\.|M\.|Mme|Mlle|Monsieur|Madame|Mademoiselle|Sr\.|Sra\.|Srta\.|Señor|Señora|Señorita|Sig\.|Sig\.ra|Signor|Signora|Dott\.|Dott\.ssa)\.?)[^\S\r\n]+[A-ZÄÖÜÁÉÍÓÚÑ][a-zA-Zäöüßáéíóúñ\-'.]+(?:[^\S\r\n]+(?:[A-ZÄÖÜÁÉÍÓÚÑ]\.?|[A-ZÄÖÜÁÉÍÓÚÑ][a-zA-Zäöüßáéíóúñ\-'.]+))*")),
            ("DOCTOR_NAME", re.compile(
                r"(?i:\b(?:Dr\.|Dr|Prof\.|Prof|Herr\s+Dr\.|Frau\s+Dr\.|Prof\.\s*Dr\.|Dr\.\s*med\.|Dott\.|Dott\.ssa|Or\.))[^\S\r\n]+(?:[A-ZÄÖÜÁÉÍÓÚÑ]\.?[^\S\r\n]+)?[A-ZÄÖÜÁÉÍÓÚÑ][a-zA-Zäöüßáéíóúñ\-'.]+(?:[^\S\r\n]+(?:[A-ZÄÖÜÁÉÍÓÚÑ]\.?|[A-ZÄÖÜÁÉÍÓÚÑ][a-zA-Zäöüßáéíóúñ\-'.]+))*(?:\s*\([A-Za-z\s,.]+\))?|"
                r"\b(?:Dr|Prof)[A-ZÄÖÜÁÉÍÓÚÑ][a-zA-Zäöüßáéíóúñ]+\b|"
                r"\b[A-ZÄÖÜÁÉÍÓÚÑ][a-zA-Zäöüßáéíóúñ\-'.]+(?:[^\S\r\n]+(?:[A-ZÄÖÜÁÉÍÓÚÑ]\.?|[A-ZÄÖÜÁÉÍÓÚÑ][a-zA-Zäöüßáéíóúñ\-'.]+))*[,\s]+(?:M\.?B\.?B\.?S|M\.?D\.?|M\.?S\.?|D\.?M\.?R\.?D|D\.?N\.?B|F\.?R\.?C\.?R|F\.?R\.?C\.?S|FRACGP|MRCGP|MBChB|FRACP|[O0]\.?D\.?|D\.?O\.?|PharmD)\b"
            )),
            ("DOCTOR_CREDENTIALS", re.compile(r"\b(?:MBBS|M\.B\.B\.S|MD|M\.D\.?|MS|M\.S\.?|DMRD|D\.M\.R\.D|DNB|D\.N\.B|FRCR|FRCS|FRACGP|MRCGP|MBChB|FRACP|[O0]\.?D\.?|D\.?O\.?|PharmD)\b")),
            ("ID_TOKEN", re.compile(r"\b[A-Z]{2,4}[-/]\d{4,12}(?:[-/]\d+)?\b|\b\d{8,12}\b")),
            ("ADDRESS_STREET", re.compile(
                r"\b\d+\s+[A-Za-z0-9\-.,]+(?:\s+[A-Za-z0-9\-.,]+)*\s+(?:Street|St|Avenue|Ave|Drive|Dr|Road|Rd|Boulevard|Blvd|Court|Ct|Lane|Ln|Terrace|Colony|Market|Chowk|Way|Square|Sq|Place|Pl|Straße|Str|Weg|Platz|Allee|Gasse|Rue|Bd|Calle|Paseo|Via|Piazza)\b|"
                r"\b(?:Lvl|Level)\s+\d+[,\s]+[A-Za-z0-9\-.,\s]+(?:Street|St|Road|Rd|Ave)\b|"
                r"\b[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b|"
                r"\b[A-ZÄÖÜÁÉÍÓÚÑ][a-zäöüßáéíóúñ]+(?:straße|str\.|strasse|weg|platz|allee|gasse)\s+\d+[a-zA-Z]?\b|"
                r"\b(?:Rue|Boulevard|Bd|Calle|Paseo|Via|Piazza)\s+(?:de\s+|del\s+|della\s+|di\s+)?[A-ZÄÖÜÁÉÍÓÚÑ][a-zäöüßáéíóúñ]+(?:[ \t]+[A-ZÄÖÜÁÉÍÓÚÑ][a-zäöüßáéíóúñ]+){0,3}(?:\s+\d+[a-zA-Z]?)?\b|"
                r"\b(?:NSW|VIC|QLD|WA|SA|TAS|ACT|NT)\s+\d{4}\b|"
                r"\bSEC(?:TOR)?\s*\d+[\s,]+[A-Za-z]+\b",
                re.IGNORECASE
            )),
            ("CITY_STATE", re.compile(r"\b(?:City|State|Country|District|Town|Location)\s*[:\-=~+]?\s*[A-Za-z\s]+\b", re.IGNORECASE)),
        ]

        for cat, reg in standalone_regexes:
            for match in reg.finditer(full_text):
                match_str = match.group(0).strip()
                if not self.is_whitelisted(match_str) and len(match_str) > 2:
                    rects = search_page_ref.search_for(match_str, textpage=tp)
                    for r in rects:
                        if cat in ("ID_TOKEN", "CITY_STATE") and is_in_clinical_zone(r):
                            continue
                        add_target(r, cat, match_str, engine="regex_scanner")
                        # If a doctor name is in the footer authorization area, check for a handwritten signature above it
                        if cat == "DOCTOR_NAME" and r.y0 > page.rect.height * 0.75:
                            try:
                                bottom_y = r.y0 - 10.0
                                top_y = max(page.rect.height * 0.70, r.y0 - 110.0)
                                if bottom_y - top_y > 30.0:
                                    sig_rect = pymupdf.Rect(r.x0 - 20, top_y, r.x1 + 60, bottom_y)
                                    already_detected = any(
                                        t.category == "DOCTOR_SIGNATURE" and
                                        not (sig_rect & pymupdf.Rect(t.bbox.x0, t.bbox.y0, t.bbox.x1, t.bbox.y1)).is_empty
                                        for t in targets
                                    )
                                    if not already_detected:
                                        if page_bgr is not None:
                                            ix0, iy0 = int(sig_rect.x0 / scale_px_x), int(sig_rect.y0 / scale_px_y)
                                            ix1, iy1 = int(sig_rect.x1 / scale_px_x), int(sig_rect.y1 / scale_px_y)
                                            roi = cv2.cvtColor(page_bgr[iy0:iy1, ix0:ix1], cv2.COLOR_BGR2GRAY)
                                        else:
                                            p_pix = page.get_pixmap()
                                            p_arr = np.frombuffer(p_pix.samples, dtype=np.uint8).reshape((p_pix.height, p_pix.width, p_pix.n))
                                            p_gray = cv2.cvtColor(p_arr, cv2.COLOR_RGBA2GRAY) if p_pix.n == 4 else (cv2.cvtColor(p_arr, cv2.COLOR_RGB2GRAY) if p_pix.n == 3 else p_arr)
                                            scale_x = p_pix.width / page.rect.width
                                            scale_y = p_pix.height / page.rect.height
                                            ix0, iy0 = int(sig_rect.x0 * scale_x), int(sig_rect.y0 * scale_y)
                                            ix1, iy1 = int(sig_rect.x1 * scale_x), int(sig_rect.y1 * scale_y)
                                            roi = p_gray[iy0:iy1, ix0:ix1]
                                            del p_pix, p_arr
                                        if roi.size > 0:
                                            thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
                                            ink_pixels = cv2.countNonZero(thresh)
                                            if ink_pixels > 450 and (ink_pixels / roi.size) > 0.05:
                                                add_target(sig_rect, "DOCTOR_SIGNATURE", "[HANDWRITTEN_SIGNATURE]", engine="doctor_footer_signature")
                            except Exception:
                                pass

        # Statutory National Identifiers (US, UK, AU, EU, IN)
        for reg, cat in STATUTORY_ID_PATTERNS:
            for match in re.finditer(reg, full_text, re.IGNORECASE):
                match_str = match.group(0).strip()
                if not self.is_whitelisted(match_str) and len(match_str) > 2:
                    rects = search_page_ref.search_for(match_str, textpage=tp)
                    for r in rects:
                        add_target(r, cat, match_str, engine="statutory_scanner")

        # Generic Header-Zone Facility & Provider Rule
        header_zone_y = page.rect.height * 0.35
        facility_keywords = re.compile(
            r"\b(?:Hospital|Clinic|Imaging|Diagnostics?|Institute|Centre|Center|Medical\s+Centre|Medical|Superspecialit(?:y|ies)|Laborator(?:y|ies)|Pharmacy|Chemist|Optical|Optometry|Vision|Eye\s+Care|Dental|Healthcare|Praxis|Klinik|Apotheke)\b",
            re.IGNORECASE
        )
        provider_creds = re.compile(
            r"\b(?:M\.?B\.?B\.?S|M\.?D|M\.?S|D\.?M\.?R\.?D|D\.?N\.?B|F\.?R\.?C\.?R|F\.?R\.?C\.?S|Reg(?:n|\.)?\s*(?:No|Num|#|:|\d)|MCI|NMC|FRACGP|MRCGP|MBChB|FRACP|O\.?D\.?|D\.?O\.?|PharmD)\b",
            re.IGNORECASE
        )
        # Rule 1B: Top Institutional Banner & Contact Block Sweep (Top 22% of Page)
        top_22_y = page.rect.height * 0.22
        inst_keywords = re.compile(
            r"\b(?:IMAGING|DIAGNOSTICS|HOSPITAL|CLINIC|CENTRE|CENTER|LABORATORY|MEDICAL|NURSING|INSTITUTE)\b",
            re.IGNORECASE
        )
        contact_keywords = re.compile(
            r"\b(?:Phone|Ph:|Mob|Tel|E-?mail|Email|@|gmail|Market|Colony|Chowk|Road|Pin|Haryana|Punjab|Delhi)\b",
            re.IGNORECASE
        )
        for idx, line in enumerate(line_objects):
            line_str = line["text"].strip()
            if self.is_whitelisted(line_str) or len(line_str) < 3:
                continue
            if line["rect"].y0 <= top_22_y and (inst_keywords.search(line_str) or contact_keywords.search(line_str)):
                l_words = line.get("words", [])
                if l_words:
                    min_x = min(w[0].x0 for w in l_words)
                    max_x = max(w[0].x1 for w in l_words)
                    min_y = min(w[0].y0 for w in l_words)
                    max_y = max(w[0].y1 for w in l_words)
                else:
                    min_x, max_x = line["rect"].x0, line["rect"].x1
                    min_y, max_y = line["rect"].y0, line["rect"].y1
                hdr_rect = pymupdf.Rect(max(0, min_x - 4), max(0, min_y - 2), min(page.rect.width, max_x + 4), max_y + 2)
                add_target(hdr_rect, "HEADER_INSTITUTIONAL_BANNER", line_str, engine="header_banner_sweep")
            elif line["rect"].y1 <= header_zone_y:
                if facility_keywords.search(line_str):
                    add_target(line["rect"], "CLINIC_FACILITY", line_str, engine="header_facility_rule")
                elif provider_creds.search(line_str) and not re.search(r"\b(?:patient|uhid|mrn|ipd|opd|admission|pat\b)", line_str, re.I):
                    add_target(line["rect"], "HEADER_PROVIDER", line_str, engine="header_provider_rule")
            elif line["rect"].y0 >= page.rect.height * 0.50 and not is_in_clinical_zone(line["rect"]):
                if facility_keywords.search(line_str):
                    adj_text = " ".join([
                        line_objects[j]["text"] for j in range(max(0, idx - 2), min(len(line_objects), idx + 3))
                    ])
                    has_contact = bool(re.search(r"\b(?:\d{3}[\-\s.]?\d{3}[\-\s.]?\d{4}|\d{8,12}|0[23478]\s*\d{4}\s*\d{4}|Street|St|Rd|Road|Ave|Lvl|NSW|VIC|QLD|WA|SA|TAS|ACT|NT)\b", adj_text, re.IGNORECASE))
                    if has_contact:
                        add_target(line["rect"], "CLINIC_FACILITY", line_str, engine="footer_facility_rule")

        # -------------------------------------------------------------------
        # 2b. Local spaCy NER for unanchored Persons, Locations, and Organizations
        # -------------------------------------------------------------------
        def is_on_barrier_or_disclaimer_line(r: pymupdf.Rect, lines_list: List[Dict[str, Any]]) -> bool:
            for l in lines_list:
                if is_clinical_barrier(l["text"]) or is_legal_disclaimer(l["text"]):
                    v_inter = min(r.y1, l["rect"].y1) - max(r.y0, l["rect"].y0)
                    if v_inter > 0.4 * min(max(1.0, r.height), max(1.0, l["rect"].height)):
                        return True
            return False

        ner_candidates = self.ner_engine.find_entities(full_text)
        for cand in ner_candidates:
            matched_str = cand["text"]
            if not self.is_whitelisted(matched_str) and len(matched_str) >= 3:
                rects = search_page_ref.search_for(matched_str, textpage=tp)
                for r in rects:
                    if is_in_clinical_zone(r) or is_on_barrier_or_disclaimer_line(r, line_objects):
                        continue
                    add_target(r, cand["category"], matched_str, engine="spacy_ner")
                if not rects and len(matched_str.split()) > 1:
                    for part in matched_str.split():
                        if len(part) >= 4 and not self.is_whitelisted(part):
                            for pr in search_page_ref.search_for(part, textpage=tp):
                                if is_in_clinical_zone(pr) or is_on_barrier_or_disclaimer_line(pr, line_objects):
                                    continue
                                add_target(pr, cand["category"], part, engine="spacy_ner_part")

        # -------------------------------------------------------------------
        # 3. CLAHE Contrast-Enhanced Pass (Faint Headers, Glare & Blurry Text)
        # -------------------------------------------------------------------
        tmp_doc_enh = None
        line_objects_enh: List[Dict[str, Any]] = []
        skip_step_3 = is_low_mem
        if not skip_step_3:
            try:
                if is_low_mem:
                    h_ratio = 0.28
                    enh_dpi = 120
                else:
                    is_std_sheet = (page.rect.width >= 550 and page.rect.height >= 700)
                    h_ratio = 0.45 if is_std_sheet else (1.0 if (max_pt < 800 or len(words) < 45) else 0.45)
                    enh_dpi = self.dpi
                words_enh, p_enh, tp_enh, tmp_doc_enh = get_enhanced_fullpage_ocr(
                    page, dpi=enh_dpi, h_ratio=h_ratio, lang=all_langs if (is_eu or len(words) < 20) else "eng", tessdata_dir=self.tessdata_dir
                )
                full_enh = p_enh.get_text(textpage=tp_enh)

                # 3a. Reconstruct visual lines and run unified form anchor parsing on CLAHE pass
                line_objects_enh = reconstruct_visual_lines(words_enh)
                active_lines = line_objects_enh
                parse_form_anchors_on_lines(
                    line_objects=line_objects_enh,
                    search_page=p_enh,
                    textpage=tp_enh,
                    page_bgr=page_bgr,
                    scale_px_x=scale_px_x,
                    scale_px_y=scale_px_y,
                    add_target_fn=add_target,
                    is_whitelisted_fn=self.is_whitelisted,
                    whitelisted_rects=whitelisted_rects,
                    suffix="_CLAHE"
                )

                # Institutional header & banner sweep on CLAHE pass
                if line_objects_enh:
                    for line in line_objects_enh:
                        if line["rect"].y0 <= top_22_y:
                            line_str = line["text"].strip()
                            if inst_keywords.search(line_str) or contact_keywords.search(line_str):
                                l_words = line.get("words", [])
                                if l_words:
                                    min_x = min(w[0].x0 for w in l_words)
                                    max_x = max(w[0].x1 for w in l_words)
                                    min_y = min(w[0].y0 for w in l_words)
                                    max_y = max(w[0].y1 for w in l_words)
                                else:
                                    min_x, max_x = line["rect"].x0, line["rect"].x1
                                    min_y, max_y = line["rect"].y0, line["rect"].y1
                                hdr_rect = pymupdf.Rect(max(0, min_x - 4), max(0, min_y - 2), min(page.rect.width, max_x + 4), max_y + 2)
                                add_target(hdr_rect, "HEADER_INSTITUTIONAL_BANNER", line_str, engine="header_banner_sweep_clahe")

                    # 3b. Standalone regexes on CLAHE pass
                    for cat, reg in standalone_regexes:
                        for match in reg.finditer(full_enh):
                            match_str = match.group(0).strip()
                            if not self.is_whitelisted(match_str) and len(match_str) > 2:
                                rects = p_enh.search_for(match_str, textpage=tp_enh)
                                for r in rects:
                                    if is_in_clinical_zone(r):
                                        continue
                                    add_target(r, f"{cat}_ENHANCED", match_str, engine="ocr_contrast_enhanced")

                    # 3c. Generic Healthcare Facility & Provider Rule on CLAHE pass
                    for idx, line in enumerate(line_objects_enh):
                        line_str = line["text"].strip()
                        if self.is_whitelisted(line_str) or len(line_str) < 3:
                            continue
                        if line["rect"].y1 <= header_zone_y:
                            if facility_keywords.search(line_str):
                                add_target(line["rect"], "CLINIC_FACILITY", line_str, engine="header_facility_rule_clahe")
                            elif provider_creds.search(line_str) and not re.search(r"\b(?:patient|uhid|mrn|ipd|opd|admission|pat\b)", line_str, re.I):
                                add_target(line["rect"], "HEADER_PROVIDER", line_str, engine="header_provider_rule_clahe")
                        elif line["rect"].y0 >= page.rect.height * 0.50 and not is_in_clinical_zone(line["rect"]):
                            if facility_keywords.search(line_str):
                                adj_text = " ".join([
                                    line_objects_enh[j]["text"] for j in range(max(0, idx - 2), min(len(line_objects_enh), idx + 3))
                                ])
                                has_contact = bool(re.search(r"\b(?:\d{3}[\-\s.]?\d{3}[\-\s.]?\d{4}|\d{8,12}|0[23478]\s*\d{4}\s*\d{4}|Street|St|Rd|Road|Ave|Lvl|NSW|VIC|QLD|WA|SA|TAS|ACT|NT)\b", adj_text, re.IGNORECASE))
                                if has_contact:
                                    add_target(line["rect"], "CLINIC_FACILITY", line_str, engine="footer_facility_rule_clahe")

                    # 3d. spaCy NER on enhanced OCR text
                    ner_candidates_enh = self.ner_engine.find_entities(full_enh)
                    for cand in ner_candidates_enh:
                        matched_str = cand["text"]
                        if not self.is_whitelisted(matched_str) and len(matched_str) >= 3:
                            rects = p_enh.search_for(matched_str, textpage=tp_enh)
                            for r in rects:
                                if is_in_clinical_zone(r) or is_on_barrier_or_disclaimer_line(r, line_objects_enh):
                                    continue
                                add_target(r, f"{cand['category']}_ENHANCED", matched_str, engine="spacy_ner_clahe")
                            if not rects and len(matched_str.split()) > 1:
                                for part in matched_str.split():
                                    if len(part) >= 4 and not self.is_whitelisted(part):
                                        for pr in p_enh.search_for(part, textpage=tp_enh):
                                            if is_in_clinical_zone(pr) or is_on_barrier_or_disclaimer_line(pr, line_objects_enh):
                                                continue
                                            add_target(pr, f"{cand['category']}_ENHANCED", part, engine="spacy_ner_clahe_part")
            except Exception:
                pass
            finally:
                active_lines = line_objects
                if tmp_doc_enh is not None:
                    tmp_doc_enh.close()

        # 3e. Multi-line Patient Demographic / Address Block expansion under PATIENT_NAME
        patient_name_targets = [
            t for t in targets
            if (t.category.startswith("PATIENT_NAME") or t.category.startswith("PATIENT_HONORIFIC") or t.category in ("PERSON", "PERSON_ENHANCED"))
            and t.bbox.y1 <= page.rect.height * 0.45
            and (clinical_y0 is None or t.bbox.y1 <= clinical_y0)
        ]
        all_candidate_lines = list(line_objects)
        if 'line_objects_enh' in locals() and line_objects_enh:
            all_candidate_lines.extend(line_objects_enh)

        addr_line_re = re.compile(r"[A-Za-zäöüß]+\s*\d+|\b\d{4,6}\s+[A-Za-zäöüß]+|\b(?:NSW|VIC|QLD|WA|SA|TAS|ACT|NT)\s+\d{4}\b", re.IGNORECASE)
        street_keyword_re = re.compile(
            r"\b(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Way|Lane|Ln|Boulevard|Blvd|Court|Ct|Straße|Str|Weg|Platz|Allee|Gasse|Rue|Bd|Calle|Paseo|Via|Postcode|SYDNEY|Sydney|Melbourne|Brisbane|Perth|Adelaide)\b",
            re.IGNORECASE
        )
        for pt in patient_name_targets:
            curr_y1 = pt.bbox.y1
            curr_x0 = pt.bbox.x0
            for _ in range(4):
                found_next_line = False
                for l in all_candidate_lines:
                    v_gap = l["rect"].y0 - curr_y1
                    x0_diff = abs(l["rect"].x0 - curr_x0)
                    if -5 <= v_gap < 35 and x0_diff < 150 and l["rect"].y1 <= page.rect.height * 0.50:
                        if clinical_y0 is not None and l["rect"].y1 > clinical_y0:
                            continue
                        l_text = l["text"].strip()
                        if self.is_whitelisted(l_text) or len(l_text) < 3:
                            continue
                        if addr_line_re.search(l_text) or street_keyword_re.search(l_text) or re.search(r"\b\d{4,6}\b", l_text):
                            add_target(l["rect"], "ADDRESS_BLOCK", l_text, engine="patient_name_adjacent_address")
                            curr_y1 = l["rect"].y1
                            curr_x0 = l["rect"].x0
                            found_next_line = True
                            break
                if not found_next_line:
                    break

        # -------------------------------------------------------------------
        # 4. Zero-keyword Visual Grounder (chromatic stamps and handwritten signatures)
        # -------------------------------------------------------------------
        if self.enable_visual_grounder and self.grounder is not None:
            try:
                if is_low_mem:
                    bg_h, bg_w = page_bgr.shape[:2]
                    max_bg = max(bg_w, bg_h)
                    if max_bg > 850:
                        v_scale = 850.0 / max_bg
                        down_w = max(1, int(round(bg_w * v_scale)))
                        down_h = max(1, int(round(bg_h * v_scale)))
                        grounder_bgr = cv2.resize(page_bgr, (down_w, down_h), interpolation=cv2.INTER_AREA)
                        v_ratio_x = bg_w / float(down_w)
                        v_ratio_y = bg_h / float(down_h)
                    else:
                        grounder_bgr = page_bgr
                        v_ratio_x = 1.0
                        v_ratio_y = 1.0

                    visual_targets = self.grounder.detect_visual_pii_from_bgr(grounder_bgr, page_num=page_num)
                    for vt in visual_targets:
                        scaled_rect = pymupdf.Rect(
                            vt.bbox.x0 * v_ratio_x * scale_px_x,
                            vt.bbox.y0 * v_ratio_y * scale_px_y,
                            vt.bbox.x1 * v_ratio_x * scale_px_x,
                            vt.bbox.y1 * v_ratio_y * scale_px_y
                        )
                        add_target(scaled_rect, vt.category, vt.raw_text, engine=vt.engine)
                    if grounder_bgr is not page_bgr:
                        del grounder_bgr
                    gc.collect()
                else:
                    visual_targets = self.grounder.detect_visual_pii_from_bgr(page_bgr, page_num=page_num)
                    for vt in visual_targets:
                        scaled_rect = pymupdf.Rect(
                            vt.bbox.x0 * scale_px_x,
                            vt.bbox.y0 * scale_px_y,
                            vt.bbox.x1 * scale_px_x,
                            vt.bbox.y1 * scale_px_y
                        )
                        add_target(scaled_rect, vt.category, vt.raw_text, engine=vt.engine)
            except Exception:
                pass

        # -------------------------------------------------------------------
        # 5. Rule 1A: Geometric Same-Line Name Completion Sweep
        # -------------------------------------------------------------------
        name_initial_re = re.compile(r"^[‘']?[A-ZÄÖÜÁÉÍÓÚÑ]\.?$")
        name_token_re = re.compile(r"^[‘']?[A-ZÄÖÜÁÉÍÓÚÑ][a-zA-Zäöüßáéíóúñ\-'.]*$")
        field_label_re = re.compile(
            r"\b(?:Age|Sex|Gender|DOB|Date|Reg\.?|Time|Collected|Received|Report|Print|Diagnosis|Rx|Rp\.?|Disp\.?|Ref\.?\s*By|MRN|UHID|OPD|IPD)\b",
            re.IGNORECASE
        )
        all_words_pool = list(words)
        if 'line_objects_enh' in locals() and line_objects_enh:
            for lo in line_objects_enh:
                for w in lo.get("words", []):
                    all_words_pool.append((w[0].x0, w[0].y0, w[0].x1, w[0].y1, w[1]))

        for t in targets:
            if (t.category.startswith("PATIENT_NAME") or t.category.startswith("DOCTOR") or "HONORIFIC" in t.category) and not t.category.endswith("_LABEL"):
                t_cy = (t.bbox.y0 + t.bbox.y1) / 2.0
                t_h = max(14.0, t.bbox.y1 - t.bbox.y0)

                # Check words to the right up to column gap (dx < page.rect.width * 0.18)
                curr_right_x = t.bbox.x1
                while True:
                    candidates = []
                    for w in all_words_pool:
                        w_x0, w_y0, w_x1, w_y1, w_txt = w[0], w[1], w[2], w[3], w[4]
                        w_cy = (w_y0 + w_y1) / 2.0
                        if abs(w_cy - t_cy) < t_h * 0.85:
                            dx = w_x0 - curr_right_x
                            if 0 < dx < page.rect.width * 0.18:
                                candidates.append((dx, w))
                    if not candidates:
                        break
                    candidates.sort(key=lambda item: item[0])
                    closest_dx, closest_w = candidates[0]
                    w_x0, w_y0, w_x1, w_y1, w_txt = closest_w[0], closest_w[1], closest_w[2], closest_w[3], closest_w[4]
                    clean_w = w_txt.strip(".,;:()[]{}#~*")
                    if field_label_re.search(w_txt) or self.is_whitelisted(clean_w):
                        break
                    if name_initial_re.match(clean_w) or name_token_re.match(clean_w) or (clean_w.isupper() and len(clean_w) <= 6):
                        t.bbox.x1 = max(t.bbox.x1, w_x1 + 3)
                        t.bbox.y0 = min(t.bbox.y0, w_y0 - 2)
                        t.bbox.y1 = max(t.bbox.y1, w_y1 + 2)
                        t.raw_text = f"{t.raw_text} {w_txt}".strip()
                        curr_right_x = t.bbox.x1
                    else:
                        break

        # -------------------------------------------------------------------
        # 5b. Scale-Invariant Vertical PII Block Propagation (With Clinical Barrier)
        # -------------------------------------------------------------------
        if line_objects:
            all_segs = []
            for l_idx, line in enumerate(line_objects):
                l_text = line["text"]
                l_is_barrier = is_clinical_barrier(l_text)
                for seg in line.get("segments", []):
                    seg_rect = seg["rect"]
                    seg_text = seg["text"]
                    seg_is_barrier = l_is_barrier or is_clinical_barrier(seg_text)

                    has_pii = False
                    pii_cats = []
                    for t in targets:
                        t_r = pymupdf.Rect(t.bbox.x0, t.bbox.y0, t.bbox.x1, t.bbox.y1)
                        inter = seg_rect & t_r
                        if not inter.is_empty and inter.get_area() > 0.15 * min(seg_rect.get_area(), t_r.get_area()):
                            has_pii = True
                            pii_cats.append(t.category)

                    all_segs.append({
                        "line_idx": l_idx,
                        "seg": seg,
                        "rect": seg_rect,
                        "text": seg_text,
                        "line_height": seg.get("line_height", seg_rect.height),
                        "has_pii": has_pii,
                        "pii_cats": pii_cats,
                        "is_barrier": seg_is_barrier,
                    })

            # Cluster consecutive segments into Vertical Layout Blocks
            blocks: List[List[Dict[str, Any]]] = []
            visited = set()

            for i, seg_a in enumerate(all_segs):
                if i in visited or seg_a["is_barrier"]:
                    continue

                curr_block = [seg_a]
                visited.add(i)
                curr_last = seg_a

                curr_line_idx = curr_last["line_idx"]
                while True:
                    next_line_idx = curr_line_idx + 1
                    candidates = [
                        (idx, s) for idx, s in enumerate(all_segs)
                        if s["line_idx"] == next_line_idx and idx not in visited
                    ]
                    if not candidates:
                        break

                    matched_cand = None
                    for c_idx, seg_b in candidates:
                        # STRICT CLINICAL BARRIER: Immediately HALT if next line is a clinical barrier
                        if seg_b["is_barrier"]:
                            matched_cand = None
                            break

                        # 1. Vertical gap < 1.35 * max(h_curr, h_next)
                        h_curr = curr_last["line_height"]
                        h_next = seg_b["line_height"]
                        max_h = max(h_curr, h_next)
                        v_gap = seg_b["rect"].y0 - curr_last["rect"].y1
                        if not (-5.0 <= v_gap < 1.35 * max_h):
                            continue

                        # 2. Alignment: Left, Right, OR Center within 1.2 * max(h_curr, h_next)
                        align_tol = 1.2 * max_h
                        left_align = abs(curr_last["rect"].x0 - seg_b["rect"].x0) <= align_tol
                        right_align = abs(curr_last["rect"].x1 - seg_b["rect"].x1) <= align_tol
                        center_curr = (curr_last["rect"].x0 + curr_last["rect"].x1) / 2.0
                        center_b = (seg_b["rect"].x0 + seg_b["rect"].x1) / 2.0
                        center_align = abs(center_curr - center_b) <= align_tol

                        if left_align or right_align or center_align:
                            matched_cand = (c_idx, seg_b)
                            break

                    if matched_cand is not None:
                        c_idx, seg_b = matched_cand
                        curr_block.append(seg_b)
                        visited.add(c_idx)
                        curr_last = seg_b
                        curr_line_idx = next_line_idx
                    else:
                        break

                if len(curr_block) >= 2:
                    blocks.append(curr_block)

            # Safe Propagation Rule: If block contains >= 2 distinct PII lines, propagate masking
            for block in blocks:
                distinct_pii_lines = set(s["line_idx"] for s in block if s["has_pii"])
                if len(distinct_pii_lines) >= 2:
                    block_x0 = min(s["rect"].x0 for s in block)
                    block_x1 = max(s["rect"].x1 for s in block)
                    is_provider_block = any(any("DOCTOR" in c or "PROVIDER" in c for c in s["pii_cats"]) for s in block)
                    for s in block:
                        s_text = s["text"].strip()
                        if self.is_whitelisted(s_text) or s["is_barrier"]:
                            continue
                        pad_y = max(3.0, 0.22 * s["line_height"])
                        p_rect = pymupdf.Rect(
                            max(0, min(s["rect"].x0, block_x0) - 2),
                            max(0, s["rect"].y0 - pad_y),
                            min(page.rect.width, max(s["rect"].x1, block_x1) + 2),
                            min(page.rect.height, s["rect"].y1 + pad_y)
                        )
                        has_clin_overlap = False
                        for wr in whitelisted_rects:
                            inter = p_rect & wr
                            if not inter.is_empty and inter.get_area() > 0.15 * min(p_rect.get_area(), wr.get_area()):
                                has_clin_overlap = True
                                break
                        if not has_clin_overlap:
                            prop_cat = "PROVIDER_BLOCK" if is_provider_block else "PII_BLOCK_PROPAGATED"
                            add_target(p_rect, prop_cat, s_text, engine="vertical_block_propagation")

        # -------------------------------------------------------------------
        # 5c. Vocabulary-Gated Legal Disclaimer Immunity
        # -------------------------------------------------------------------
        disclaimer_lines = [
            l for l in line_objects
            if is_legal_disclaimer(l["text"])
        ]
        if disclaimer_lines:
            disclaimer_zones = []
            for dl in disclaimer_lines:
                d_rect = dl["rect"]
                d_y0 = max(0, d_rect.y0 - 4.0)
                d_y1 = min(page.rect.height, d_rect.y1 + 4.0)
                for l in line_objects:
                    if l["rect"].y0 >= d_rect.y0 and l["rect"].y1 <= d_rect.y1 + dl.get("line_height", 14.0) * 3.5:
                        d_y1 = max(d_y1, l["rect"].y1 + 4.0)
                disclaimer_zones.append(pymupdf.Rect(0, d_y0, page.rect.width, d_y1))

            strict_regex_categories = {
                "PHONE", "CONTACT_PHONE", "EMAIL", "CONTACT_EMAIL", "URL", "BARCODE", "BARCODE_OR_QR"
            }
            filtered_targets = []
            for t in targets:
                t_r = pymupdf.Rect(t.bbox.x0, t.bbox.y0, t.bbox.x1, t.bbox.y1)
                in_disclaimer = any(not (t_r & dz).is_empty for dz in disclaimer_zones)
                if in_disclaimer:
                    raw_t = t.raw_text or ""
                    is_strict_contact = (
                        t.category in strict_regex_categories or
                        bool(re.search(r"@|\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b|\b\d{3}[\-\s.]\d{3}[\-\s.]\d{4}\b|https?://", raw_t))
                    )
                    if is_strict_contact:
                        filtered_targets.append(t)
                else:
                    filtered_targets.append(t)
            targets = filtered_targets

        try:
            if tmp_ocr_doc is not None:
                tmp_ocr_doc.close()
                del tmp_ocr_doc
        except Exception:
            pass
        try:
            del page_bgr, pix, arr
        except Exception:
            pass
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        return targets

    def find_pii_targets(self, doc: pymupdf.Document) -> List[RedactionTarget]:
        """Scans all pages of a document and returns detected PII targets."""
        all_targets: List[RedactionTarget] = []
        for idx, page in enumerate(doc):
            targets = self.find_pii_targets_for_page(page, idx + 1)
            all_targets.extend(targets)
        return all_targets

    def redact(self, input_path: str, output_path: Optional[str] = None, jurisdiction: str = "AUTO", force_prescription: bool = False) -> Dict[str, Any]:
        """
        Executes complete, irreversible redaction on scanned raster PDF or image.
        Returns audit summary dictionary with detected jurisdiction.
        """
        input_path = os.path.abspath(input_path)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        in_base, in_ext = os.path.splitext(input_path)
        in_ext_lower = in_ext.lower()
        is_image_input = in_ext_lower in self.IMAGE_EXTENSIONS

        if output_path is None:
            output_path = f"{in_base}_anonymized{in_ext}"
        output_path = os.path.abspath(output_path)
        out_ext_lower = os.path.splitext(output_path)[1].lower()

        # Handle Image Input
        if is_image_input:
            self._is_current_file_image = True
            is_low_mem = os.environ.get("RENDER_LOW_MEMORY") == "1"

            # Probe true image dimensions without full uncompressed decode
            with Image.open(input_path) as probe:
                orig_w, orig_h = probe.size

            # In low-memory mode on large photos (>= 2000px), use reduced decode to cap peak allocation < 20MB
            if is_low_mem and max(orig_w, orig_h) >= 2000:
                img_bgr = cv2.imread(input_path, cv2.IMREAD_REDUCED_COLOR_2)
                if img_bgr is None:
                    with Image.open(input_path) as pil_img:
                        pil_img.draft("RGB", (orig_w // 2, orig_h // 2))
                        img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.imread(input_path)
                if img_bgr is None:
                    with Image.open(input_path) as pil_img:
                        img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

            cur_h, cur_w = img_bgr.shape[:2]
            target_max = 1350 if is_low_mem else None
            if target_max and max(cur_w, cur_h) > target_max:
                scale = float(target_max) / max(cur_w, cur_h)
                new_w = max(1, int(round(cur_w * scale)))
                new_h = max(1, int(round(cur_h * scale)))
                img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            elif is_low_mem and max(cur_w, cur_h) < 700:
                target_dim = 1785 if min(cur_w, cur_h) < 400 else 1000
                scale = float(target_dim) / max(cur_w, cur_h)
                new_w = max(1, int(round(cur_w * scale)))
                new_h = max(1, int(round(cur_h * scale)))
                img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Create pristine PDF canvas from img_bgr (un-normalized)
            _, enc_pristine = cv2.imencode(".png", img_bgr)
            with pymupdf.open("png", enc_pristine.tobytes()) as img_doc:
                pdf_bytes = img_doc.convert_to_pdf()
            del enc_pristine
            doc = pymupdf.open("pdf", pdf_bytes)
            page = doc[0]

            # Decoupled Pre-OCR Ink Normalization on a separate scratch canvas solely for OCR
            ocr_scratch = normalize_ink_contrast(img_bgr.copy(), gamma=1.6)
            del img_bgr

            if is_low_mem:
                lab = cv2.cvtColor(ocr_scratch, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
                bg = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
                norm_l = cv2.divide(l, np.maximum(1, bg), scale=255)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(norm_l)
                lab = cv2.merge((cl, a, b))
                proc_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                del lab, l, a, b, cl, kernel, bg, norm_l, ocr_scratch
            else:
                proc_bgr = ocr_scratch

            _, enc_ocr = cv2.imencode(".png", proc_bgr)
            del proc_bgr
            with pymupdf.open("png", enc_ocr.tobytes()) as img_doc:
                ocr_pdf_bytes = img_doc.convert_to_pdf()
            del enc_ocr
            ocr_doc = pymupdf.open("pdf", ocr_pdf_bytes)
            self._current_image_ocr_page = ocr_doc[0]

            # 1. OCR, form parsing, and visual grounder targets
            try:
                targets = self.find_pii_targets_for_page(page, 1)
            finally:
                self._current_image_ocr_page = None
                ocr_doc.close()
                del ocr_doc

            # Resolve jurisdiction using actual extracted OCR text
            ocr_text = getattr(self, "_last_extracted_text", "") or page.get_text()
            resolved_jur = JurisdictionDetector.resolve_jurisdiction(ocr_text, jurisdiction)

            from .document_validator import evaluate_prescription_gate, record_gate_override_telemetry
            visual_stamps_count = sum(1 for t in targets if t.category == "STAMP" or "STAMP" in t.raw_text or t.engine == "visual_grounder_stamp")
            barcode_count = sum(1 for t in targets if t.category in ("BARCODE", "BARCODE_OR_QR", "QR_CODE"))
            is_rx, gate_score, gate_reasons = evaluate_prescription_gate(
                ocr_text,
                visual_stamps_count=visual_stamps_count,
                barcode_count=barcode_count
            )

            if not force_prescription and not is_rx:
                doc.close()
                if is_low_mem:
                    self._last_extracted_words = []
                    self._last_whitelisted_rects = []
                    self._last_extracted_text = ""
                return {
                    "input_file": input_path,
                    "output_file": output_path,
                    "jurisdiction": resolved_jur,
                    "format_type": "SCANNED_RASTER_IMAGE",
                    "total_entities_redacted": 0,
                    "extracted_text": ocr_text,
                    "redacted_targets": [],
                    "confidence_score": 0.0,
                    "triage_status": "NOT_A_PRESCRIPTION",
                    "gate_score": gate_score,
                    "review_reasons": ["Not recognized as a medical prescription. Click 'Force Anonymize' if this is a handwritten or non-standard slip."]
                }

            whitelisted_rects = getattr(self, "_last_whitelisted_rects", [])
            for t in targets:
                pad_x = max(3.0, t.bbox.width * self.padding_ratio)
                pad_y = max(2.0, t.bbox.height * self.padding_ratio)
                rect = pymupdf.Rect(
                    max(0, t.bbox.x0 - pad_x),
                    max(0, t.bbox.y0 - pad_y),
                    min(page.rect.width, t.bbox.x1 + pad_x),
                    min(page.rect.height, t.bbox.y1 + pad_y)
                )
                for wr in whitelisted_rects:
                    inter = rect & wr
                    if not inter.is_empty:
                        rect_mid = (rect.y0 + rect.y1) / 2.0
                        wr_mid = (wr.y0 + wr.y1) / 2.0
                        split_y = (rect_mid + wr_mid) / 2.0
                        if rect_mid > wr_mid:
                            rect.y0 = max(rect.y0, split_y)
                        else:
                            rect.y1 = min(rect.y1, split_y)
                if rect.is_valid and rect.get_area() > 0:
                    if is_low_mem:
                        page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))
                    else:
                        page.add_redact_annot(rect, fill=(0, 0, 0))

            if not is_low_mem:
                page.apply_redactions()

            scale_back_x = orig_w / page.rect.width
            scale_back_y = orig_h / page.rect.height

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if out_ext_lower in self.IMAGE_EXTENSIONS:
                if is_low_mem and max(orig_w, orig_h) > 1800:
                    scale = 1800.0 / max(orig_w, orig_h)
                    scale_x = (orig_w * scale) / page.rect.width
                    scale_y = (orig_h * scale) / page.rect.height
                else:
                    scale_x = orig_w / page.rect.width
                    scale_y = orig_h / page.rect.height
                pix = page.get_pixmap(matrix=pymupdf.Matrix(scale_x, scale_y))
                if out_ext_lower in {".jpg", ".jpeg"}:
                    pix.pil_save(output_path, format="JPEG", quality=92)
                elif out_ext_lower == ".png":
                    png_bytes = pix.tobytes("png")
                    if len(png_bytes) <= 3 * 1024 * 1024:
                        with open(output_path, "wb") as f:
                            f.write(png_bytes)
                        del png_bytes
                    else:
                        del png_bytes
                        pix.pil_save(output_path, format="JPEG", quality=92)
                elif out_ext_lower == ".webp":
                    pix.pil_save(output_path, format="WEBP", quality=92)
                else:
                    pix.save(output_path, jpg_quality=92)
                del pix
            else:
                out_doc = pymupdf.open()
                render_dpi = 100 if is_low_mem else 150
                pix = page.get_pixmap(dpi=render_dpi)
                jpg_bytes = pix.tobytes("jpeg", jpg_quality=85)
                new_p = out_doc.new_page(width=page.rect.width, height=page.rect.height)
                new_p.insert_image(page.rect, stream=jpg_bytes)
                del pix, jpg_bytes
                out_doc.set_metadata({
                    "title": "Redacted Medical Document",
                    "author": "Pharmacon Anonymizer",
                    "subject": "De-identified Prescription",
                    "keywords": "anonymized, de-identified",
                    "creator": "Pharmacon Anonymizer Engine",
                    "producer": "PyMuPDF",
                    "creationDate": "",
                    "modDate": ""
                })
                out_doc.save(output_path, garbage=4, deflate=True)
                out_doc.close()
                del out_doc

            final_extracted_text = getattr(self, "_last_extracted_text", "")
            if not final_extracted_text and not doc.is_closed:
                final_extracted_text = page.get_text()

            doc.close()
            gc.collect()

            reported_targets = []
            for t in targets:
                cat = t.category
                raw_l = (t.raw_text or "").lower()
                if "PATIENT_NAME" in cat and any(kw in raw_l for kw in ["address", "anschrift", "sec ", "sector", "kurukshetra"]):
                    cat = "ADDRESS_BLOCK"
                reported_targets.append(RedactionTarget(
                    page_number=t.page_number,
                    category=cat,
                    raw_text=t.raw_text,
                    bbox=BoundingBox(
                        x0=t.bbox.x0 * scale_back_x,
                        y0=t.bbox.y0 * scale_back_y,
                        x1=t.bbox.x1 * scale_back_x,
                        y1=t.bbox.y1 * scale_back_y
                    ),
                    confidence=t.confidence,
                    engine=t.engine
                ))

            scaled_words = [
                (w[0] * scale_back_x, w[1] * scale_back_y, w[2] * scale_back_x, w[3] * scale_back_y, w[4])
                for w in (getattr(self, "_last_extracted_words", []) or [])
            ]
            conf = ForensicAuditor.evaluate_redaction_confidence(
                report={"redacted_targets": [t.model_dump() for t in reported_targets]},
                extracted_text=final_extracted_text,
                is_raster=True,
                extracted_words=scaled_words
            )

            if is_low_mem:
                self._last_extracted_words = []
                self._last_whitelisted_rects = []
                self._last_extracted_text = ""
            try:
                pymupdf.TOOLS.store_shrink(100)
            except Exception:
                pass
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

            if force_prescription:
                record_gate_override_telemetry(
                    format_type="SCANNED_RASTER_IMAGE",
                    ocr_word_count=len(final_extracted_text.split()),
                    gate_score=gate_score if 'gate_score' in locals() else 0.0,
                    redacted_count=len(reported_targets)
                )

            return {
                "input_file": input_path,
                "output_file": output_path,
                "jurisdiction": resolved_jur,
                "format_type": "SCANNED_RASTER_IMAGE",
                "total_entities_redacted": len(reported_targets),
                "extracted_text": final_extracted_text,
                "redacted_targets": [t.model_dump() for t in reported_targets],
                "confidence_score": conf["confidence_score"],
                "triage_status": conf["triage_status"],
                "review_reasons": conf["review_reasons"]
            }

        # Handle PDF Input
        self._is_current_file_image = False
        doc = pymupdf.open(input_path)
        all_targets: List[RedactionTarget] = []
        collected_ocr_text: List[str] = []

        is_low_mem = os.environ.get("RENDER_LOW_MEMORY") == "1"
        for p_idx, page in enumerate(doc):
            orig_dpi = self.dpi
            if is_low_mem:
                self.dpi = 135
            try:
                targets = self.find_pii_targets_for_page(page, p_idx + 1)
            finally:
                self.dpi = orig_dpi
            all_targets.extend(targets)
            if hasattr(self, "_last_extracted_text") and self._last_extracted_text:
                collected_ocr_text.append(self._last_extracted_text)

            if p_idx == 0:
                first_page_text = self._last_extracted_text if (hasattr(self, "_last_extracted_text") and self._last_extracted_text) else page.get_text()
                from .document_validator import evaluate_prescription_gate, record_gate_override_telemetry
                visual_stamps_count = sum(1 for t in targets if t.category == "STAMP" or "STAMP" in t.raw_text or t.engine == "visual_grounder_stamp")
                barcode_count = sum(1 for t in targets if t.category in ("BARCODE", "BARCODE_OR_QR", "QR_CODE"))
                is_rx, gate_score, gate_reasons = evaluate_prescription_gate(
                    first_page_text,
                    visual_stamps_count=visual_stamps_count,
                    barcode_count=barcode_count
                )
                if not force_prescription and not is_rx:
                    doc.close()
                    if is_low_mem:
                        self._last_extracted_words = []
                        self._last_whitelisted_rects = []
                        self._last_extracted_text = ""
                    return {
                        "input_file": input_path,
                        "output_file": output_path,
                        "jurisdiction": jurisdiction,
                        "format_type": "SCANNED_RASTER",
                        "total_entities_redacted": 0,
                        "extracted_text": first_page_text,
                        "redacted_targets": [],
                        "confidence_score": 0.0,
                        "triage_status": "NOT_A_PRESCRIPTION",
                        "gate_score": gate_score,
                        "review_reasons": ["Not recognized as a medical prescription. Click 'Force Anonymize' if this is a handwritten or non-standard slip."]
                    }

            whitelisted_rects = getattr(self, "_last_whitelisted_rects", [])
            for t in targets:
                pad_x = max(2.0, t.bbox.width * self.padding_ratio)
                pad_y = max(1.5, t.bbox.height * self.padding_ratio)
                rect = pymupdf.Rect(
                    max(0, t.bbox.x0 - pad_x),
                    max(0, t.bbox.y0 - pad_y),
                    min(page.rect.width, t.bbox.x1 + pad_x),
                    min(page.rect.height, t.bbox.y1 + pad_y)
                )
                for wr in whitelisted_rects:
                    inter = rect & wr
                    if not inter.is_empty:
                        rect_mid = (rect.y0 + rect.y1) / 2.0
                        wr_mid = (wr.y0 + wr.y1) / 2.0
                        split_y = (rect_mid + wr_mid) / 2.0
                        if rect_mid > wr_mid:
                            rect.y0 = max(rect.y0, split_y)
                        else:
                            rect.y1 = min(rect.y1, split_y)
                if rect.is_valid and rect.get_area() > 0:
                    if is_low_mem:
                        page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))
                    else:
                        page.add_redact_annot(rect, fill=(0, 0, 0))

            if not is_low_mem:
                page.apply_redactions()

        full_doc_text = " ".join(collected_ocr_text) if collected_ocr_text else " ".join(p.get_text() for p in doc)
        resolved_jur = JurisdictionDetector.resolve_jurisdiction(full_doc_text, jurisdiction)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if out_ext_lower in self.IMAGE_EXTENSIONS:
            pix = doc[0].get_pixmap(dpi=self.dpi)
            pix.save(output_path)
            del pix
            doc.close()
        else:
            out_doc = pymupdf.open()
            render_dpi = 100 if is_low_mem else 150
            for page in doc:
                pix = page.get_pixmap(dpi=render_dpi)
                jpg_bytes = pix.tobytes("jpeg", jpg_quality=85)
                new_p = out_doc.new_page(width=page.rect.width, height=page.rect.height)
                new_p.insert_image(page.rect, stream=jpg_bytes)
                del pix, jpg_bytes
            out_doc.set_metadata({
                "title": "Redacted Medical Document",
                "author": "Pharmacon Anonymizer",
                "subject": "De-identified Prescription",
                "keywords": "anonymized, de-identified",
                "creator": "Pharmacon Anonymizer Engine",
                "producer": "PyMuPDF",
                "creationDate": "",
                "modDate": ""
            })
            out_doc.save(output_path, garbage=4, deflate=True)
            out_doc.close()
            doc.close()
            del out_doc

        if is_low_mem:
            self._last_extracted_words = []
            self._last_whitelisted_rects = []
            self._last_extracted_text = ""
        try:
            pymupdf.TOOLS.store_shrink(100)
        except Exception:
            pass
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        conf = ForensicAuditor.evaluate_redaction_confidence(
            report={"redacted_targets": [t.model_dump() for t in all_targets]},
            extracted_text="\n".join(collected_ocr_text),
            is_raster=True,
            extracted_words=getattr(self, "_last_extracted_words", None)
        )

        if is_low_mem:
            self._last_extracted_words = []
            self._last_whitelisted_rects = []
            self._last_extracted_text = ""
        try:
            pymupdf.TOOLS.store_shrink(100)
        except Exception:
            pass
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        if force_prescription:
            from .document_validator import record_gate_override_telemetry
            record_gate_override_telemetry(
                format_type="SCANNED_RASTER_PDF",
                ocr_word_count=sum(len(txt.split()) for txt in collected_ocr_text),
                gate_score=gate_score if 'gate_score' in locals() else 0.0,
                redacted_count=len(all_targets)
            )

        return {
            "input_file": input_path,
            "output_file": output_path,
            "jurisdiction": resolved_jur,
            "format_type": "SCANNED_RASTER_PDF",
            "total_entities_redacted": len(all_targets),
            "extracted_text": "\n".join(collected_ocr_text),
            "redacted_targets": [t.model_dump() for t in all_targets],
            "confidence_score": conf["confidence_score"],
            "triage_status": conf["triage_status"],
            "review_reasons": conf["review_reasons"]
        }
