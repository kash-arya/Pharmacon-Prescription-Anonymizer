"""
Local Image Redactor Module for Scanned Prescriptions and Medical Forms.

Applies local OCR (via PyMuPDF and embedded Tesseract engine), demographic/PII
anchor detection, 2D multi-column form parsing, and permanent black-box visual masking
to scanned raster PDFs and image files (PNG, JPG, TIFF, etc.), strictly preserving
clinical pharmacology and diagnostic findings.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import re
import io
import pymupdf
import cv2
import numpy as np
from PIL import Image, ImageEnhance

try:
    from prescription_anonymizer.src.entities import (
        BoundingBox,
        RedactionTarget,
        KEY_VALUE_ANCHORS,
        CLINICAL_WHITELIST_PATTERNS,
        STATUTORY_ID_PATTERNS,
        is_whitelisted,
        _COMPILED_WHITELIST,
    )
    from prescription_anonymizer.src.vlm_grounder import MultimodalVisualGrounder
    from prescription_anonymizer.src.jurisdiction import JurisdictionDetector
    from prescription_anonymizer.src.ner_engine import LocalNEREngine
except ImportError:
    from src.entities import (
        BoundingBox,
        RedactionTarget,
        KEY_VALUE_ANCHORS,
        CLINICAL_WHITELIST_PATTERNS,
        STATUTORY_ID_PATTERNS,
        is_whitelisted,
        _COMPILED_WHITELIST,
    )
    from src.vlm_grounder import MultimodalVisualGrounder
    from src.jurisdiction import JurisdictionDetector
    from src.ner_engine import LocalNEREngine


def resolve_tessdata_path(custom_dir: Optional[str] = None) -> str:
    """Finds valid directory containing eng.traineddata."""
    candidates = []
    if custom_dir:
        candidates.append(os.path.abspath(custom_dir))

    env_prefix = os.environ.get("TESSDATA_PREFIX")
    if env_prefix:
        candidates.append(os.path.abspath(env_prefix))
        candidates.append(os.path.abspath(os.path.join(env_prefix, "tessdata")))

    # Project local tessdata
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(repo_root, "data", "tessdata"))
    candidates.append(os.path.join(os.getcwd(), "data", "tessdata"))

    # System tessdata paths
    candidates.extend([
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tessdata",
        "/usr/local/share/tessdata"
    ])

    for path in candidates:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "eng.traineddata")):
            return path

    for path in candidates:
        if os.path.isdir(path):
            return path

    return candidates[0] if candidates else "/usr/share/tessdata"


def get_ocr_languages(tessdata_dir: Optional[str]) -> str:
    """Returns available OCR languages (eng+deu+fra+spa or available subset)."""
    if not tessdata_dir or not os.path.isdir(tessdata_dir):
        return "eng"
    langs = ["eng"]
    for lang in ["deu", "fra", "spa"]:
        if os.path.exists(os.path.join(tessdata_dir, f"{lang}.traineddata")):
            langs.append(lang)
    return "+".join(langs)


class LocalImageRedactor:
    """
    Resolution-invariant OCR & visual redaction engine for scanned raster PDFs
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

    def _detect_qr_and_barcodes(
        self,
        page: pymupdf.Page,
        bgr: Optional[np.ndarray] = None,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None
    ) -> List[Tuple[pymupdf.Rect, str]]:
        """
        Detects 2D QR codes and 1D barcodes using OpenCV QRCodeDetector and BarcodeDetector.
        Returns list of (clamped_bounding_box, detected_payload_or_label).
        """
        if bgr is None:
            pix = page.get_pixmap()
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            elif pix.n == 3:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:
                gray = arr
                bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            scale_x = page.rect.width / max(1.0, pix.width)
            scale_y = page.rect.height / max(1.0, pix.height)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if scale_x is None:
                scale_x = page.rect.width / max(1.0, bgr.shape[1])
            if scale_y is None:
                scale_y = page.rect.height / max(1.0, bgr.shape[0])

        boxes: List[Tuple[pymupdf.Rect, str]] = []

        # 1. OpenCV QRCodeDetector
        try:
            qd = cv2.QRCodeDetector()
            ok_q, decoded_info, points_q, _ = qd.detectAndDecodeMulti(bgr)
            if ok_q and points_q is not None and len(points_q) > 0:
                for i, pts in enumerate(points_q):
                    pts = pts.reshape(-1, 2)
                    pad = 12.0
                    x0 = max(0.0, float(np.min(pts[:, 0])) * scale_x - pad)
                    y0 = max(0.0, float(np.min(pts[:, 1])) * scale_y - pad)
                    x1 = min(page.rect.width, float(np.max(pts[:, 0])) * scale_x + pad)
                    y1 = min(page.rect.height, float(np.max(pts[:, 1])) * scale_y + pad)
                    qw = x1 - x0
                    qh = y1 - y0
                    if 20 < qw < page.rect.width * 0.40 and 20 < qh < page.rect.height * 0.40 and (0.5 <= qw / max(1.0, qh) <= 2.0):
                        info = decoded_info[i] if i < len(decoded_info) and decoded_info[i] else "QR_CODE"
                        boxes.append((pymupdf.Rect(x0, y0, x1, y1), info))
            else:
                ok_d, pts_d = qd.detect(bgr)
                if ok_d and pts_d is not None:
                    pts_d = pts_d.reshape(-1, 2)
                    pad = 12.0
                    x0 = max(0.0, float(np.min(pts_d[:, 0])) * scale_x - pad)
                    y0 = max(0.0, float(np.min(pts_d[:, 1])) * scale_y - pad)
                    x1 = min(page.rect.width, float(np.max(pts_d[:, 0])) * scale_x + pad)
                    y1 = min(page.rect.height, float(np.max(pts_d[:, 1])) * scale_y + pad)
                    qw = x1 - x0
                    qh = y1 - y0
                    if 20 < qw < page.rect.width * 0.40 and 20 < qh < page.rect.height * 0.40 and (0.5 <= qw / max(1.0, qh) <= 2.0):
                        boxes.append((pymupdf.Rect(x0, y0, x1, y1), "QR_CODE"))
        except Exception:
            pass

        # 2. OpenCV BarcodeDetector
        try:
            bd = cv2.barcode.BarcodeDetector()
            detected_candidates = []

            # Pass 1: Full image
            ok_b, points_b = bd.detectMulti(gray)
            if not ok_b or points_b is None or len(points_b) == 0:
                ok_b, points_b = bd.detect(gray)
            if ok_b and points_b is not None and len(points_b) > 0:
                pts_list = points_b if points_b.ndim == 3 else [points_b]
                for pts in pts_list:
                    detected_candidates.append((pts.reshape(-1, 2), 0.0, 0.0))

            # Pass 2: Overlapping horizontal sliding window across 100% of the page.
            # Prevents downsampling loss on high-res images (e.g. 30MP phone scans) and eliminates blind spots.
            h_gray = gray.shape[0]
            tile_h = max(200, int(h_gray * 0.30))
            step_h = max(100, int(h_gray * 0.20))
            for y_start in range(0, h_gray, step_h):
                y_end = min(h_gray, y_start + tile_h)
                tile = gray[y_start:y_end, :]
                ok_t, pts_t = bd.detectMulti(tile)
                if ok_t and pts_t is not None and len(pts_t) > 0:
                    pts_list_t = pts_t if pts_t.ndim == 3 else [pts_t]
                    for pts in pts_list_t:
                        detected_candidates.append((pts.reshape(-1, 2), 0.0, float(y_start)))

            for pts, off_x, off_y in detected_candidates:
                # Validate 1D barcode candidates: vertical-bar barcodes have mean(|dx|) > 2.2 * mean(|dy|)
                pts_x = pts[:, 0] + off_x
                pts_y = pts[:, 1] + off_y
                roi_x0 = max(0, int(np.min(pts_x)))
                roi_y0 = max(0, int(np.min(pts_y)))
                roi_x1 = min(gray.shape[1], int(np.max(pts_x)))
                roi_y1 = min(gray.shape[0], int(np.max(pts_y)))
                if roi_x1 > roi_x0 + 10 and roi_y1 > roi_y0 + 10:
                    roi = gray[roi_y0:roi_y1, roi_x0:roi_x1]
                    sobel_x = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
                    sobel_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
                    mean_dx = float(np.mean(np.abs(sobel_x)))
                    mean_dy = float(np.mean(np.abs(sobel_y)))
                    if mean_dy > 0 and (mean_dx <= 2.2 * mean_dy):
                        continue

                pad = 10.0
                x0 = max(0.0, float(np.min(pts[:, 0]) + off_x) * scale_x - pad)
                y0 = max(0.0, float(np.min(pts[:, 1]) + off_y) * scale_y - pad)
                x1 = min(page.rect.width, float(np.max(pts[:, 0]) + off_x) * scale_x + pad)
                y1 = min(page.rect.height, float(np.max(pts[:, 1]) + off_y) * scale_y + pad)
                bw = x1 - x0
                bh = y1 - y0
                # Barcode plausibility: must not exceed 22% of page height or 12% of total page area
                if 20 < bw < page.rect.width * 0.95 and 10 < bh < page.rect.height * 0.22 and (bw * bh) < (page.rect.width * page.rect.height * 0.12):
                    rect = pymupdf.Rect(x0, y0, x1, y1)
                    is_dup = False
                    for i, (existing_rect, label) in enumerate(boxes):
                        if label == "BARCODE":
                            inter = (rect & existing_rect).get_area()
                            union = rect.get_area() + existing_rect.get_area() - inter
                            if union > 0 and (inter / union > 0.4 or inter / min(rect.get_area(), existing_rect.get_area()) > 0.6):
                                boxes[i] = (rect | existing_rect, label)
                                is_dup = True
                                break
                    if not is_dup:
                        boxes.append((rect, "BARCODE"))
        except Exception:
            pass

        return boxes

    def _get_enhanced_fullpage_ocr(self, page: pymupdf.Page, h_ratio: float = 0.45, lang: Optional[str] = None) -> Tuple[List[Any], pymupdf.Page, pymupdf.TextPage, pymupdf.Document]:
        """Performs CLAHE adaptive contrast preprocessing, denoising, and unsharp masking."""
        max_pt = max(page.rect.width, page.rect.height)
        if max_pt < 550:
            effective_dpi = 300
        elif max_pt < 900:
            effective_dpi = self.dpi if (page.rect.width >= 550 and page.rect.height >= 700) else 200
        elif max_pt <= 2000:
            effective_dpi = self.dpi
        else:
            effective_dpi = 72
        pix = page.get_pixmap(dpi=effective_dpi, clip=pymupdf.Rect(0, 0, page.rect.width, page.rect.height * h_ratio))
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        elif pix.n == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr

        # 1. Denoising + CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        cl_img = clahe.apply(gray)

        # 2. Unsharp masking to sharpen blurry camera text
        blurred = cv2.GaussianBlur(cl_img, (0, 0), 3)
        sharpened = cv2.addWeighted(cl_img, 1.5, blurred, -0.5, 0)

        is_success, buf = cv2.imencode('.png', sharpened)
        tmp_doc = pymupdf.open()
        p_enh = tmp_doc.new_page(width=page.rect.width, height=page.rect.height * h_ratio)
        p_enh.insert_image(pymupdf.Rect(0, 0, page.rect.width, page.rect.height * h_ratio), stream=buf.tobytes())
        ocr_lang = lang if lang is not None else get_ocr_languages(self.tessdata_dir)
        tp_enh = p_enh.get_textpage_ocr(language=ocr_lang, dpi=effective_dpi, tessdata=self.tessdata_dir)
        words_enh = p_enh.get_text("words", textpage=tp_enh)

        return words_enh, p_enh, tp_enh, tmp_doc

    _get_enhanced_header_ocr = _get_enhanced_fullpage_ocr

    def _expand_person_span_on_line(self, rect: pymupdf.Rect, line_objects: List[Dict[str, Any]]) -> pymupdf.Rect:
        """
        Expands a person bounding box horizontally to adjacent name tokens, honorifics,
        and initials on the same visual line within 35 pt gap, excluding form labels,
        stopwords, and clinical whitelist terms.
        """
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
            if self.is_whitelisted(clean):
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

    def _reconstruct_visual_lines(self, words: List[Tuple]) -> List[Dict[str, Any]]:
        """
        Groups word tokens into non-overlapping horizontal lines using running vertical centers (y_mid).
        Prevents vertical chaining / line ballooning across multi-column forms.
        """
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

        # Sort lines vertically by average y_mid
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

    def _detect_handwritten_fill_in_zone(
        self,
        page_bgr: np.ndarray,
        zone_rect: pymupdf.Rect,
        scale_px_x: float,
        scale_px_y: float,
        whitelisted_rects: Optional[List[pymupdf.Rect]] = None
    ) -> Optional[pymupdf.Rect]:
        """
        Detects handwritten cursive ink strokes inside a designated form-fill zone.
        Removes printed form lines (horizontal/vertical grid borders) via morphological filters
        and returns the tight bounding box of significant ink strokes in PDF point coordinates.
        """
        if zone_rect.width <= 4 or zone_rect.height <= 4:
            return None

        ix0 = max(0, int(zone_rect.x0 / scale_px_x))
        iy0 = max(0, int(zone_rect.y0 / scale_px_y))
        ix1 = min(page_bgr.shape[1], int(zone_rect.x1 / scale_px_x))
        iy1 = min(page_bgr.shape[0], int(zone_rect.y1 / scale_px_y))

        if ix1 <= ix0 + 10 or iy1 <= iy0 + 6:
            return None

        roi = page_bgr[iy0:iy1, ix0:ix1]
        if roi.size == 0:
            return None

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )

        # Remove printed horizontal and vertical form grid/rule lines
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)
        vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)
        grid_lines = cv2.bitwise_or(horiz_lines, vert_lines)
        grid_lines = cv2.dilate(grid_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        # Non-line handwritten ink strokes
        ink = cv2.bitwise_and(thresh, cv2.bitwise_not(grid_lines))
        ink_pixels = int(cv2.countNonZero(ink))
        total_pixels = roi.shape[0] * roi.shape[1]
        ink_ratio = float(ink_pixels) / max(1, total_pixels)

        if ink_pixels > 80 and 0.012 < ink_ratio < 0.65:
            pts = cv2.findNonZero(ink)
            if pts is not None:
                bx, by, bw, bh = cv2.boundingRect(pts)
                stroke_rect = pymupdf.Rect(
                    max(zone_rect.x0, (ix0 + bx) * scale_px_x - 2),
                    max(zone_rect.y0, (iy0 + by) * scale_px_y - 2),
                    min(zone_rect.x1, (ix0 + bx + bw) * scale_px_x + 2),
                    min(zone_rect.y1, (iy0 + by + bh) * scale_px_y + 2)
                )
                wl = whitelisted_rects or getattr(self, "_last_whitelisted_rects", [])
                for wr in wl:
                    inter = stroke_rect & wr
                    if not inter.is_empty and inter.get_area() > 0.10 * min(stroke_rect.get_area(), wr.get_area()):
                        return None
                return stroke_rect
        return None

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
        """
        Unified multi-column key-value form parser running across OCR passes.
        Supports:
        - 1D horizontal proximity & fused token extraction
        - 2D multi-row column table parsing for multi-line patient/address blocks
        - Doctor/prescriber signature adjacent lines & ink box search
        - OCR-independent spatial handwritten ink-fill detection
        """
        anchor_patterns = list(KEY_VALUE_ANCHORS.items())

        for l_idx, l_obj in enumerate(line_objects):
            l_text = l_obj["text"]
            l_words = l_obj["words"]
            line_rect = l_obj["rect"]

            detected_anchors = []
            line_mid = (line_rect.y0 + line_rect.y1) / 2.0
            for cat, patterns in anchor_patterns:
                for pat in patterns:
                    for m in re.finditer(pat, l_text, re.IGNORECASE):
                        kw = m.group(0)
                        raw_rects = search_page.search_for(kw, textpage=textpage)
                        kw_rects = sorted(
                            [r for r in raw_rects if abs((r.y0 + r.y1) / 2.0 - line_mid) < max(20.0, line_rect.height * 0.4)],
                            key=lambda r: r.x0
                        )
                        if kw_rects:
                            kw_clusters = []
                            curr_cluster = [kw_rects[0]]
                            for r in kw_rects[1:]:
                                if abs(r.x0 - curr_cluster[-1].x1) < 40:
                                    curr_cluster.append(r)
                                else:
                                    kw_clusters.append(curr_cluster)
                                    curr_cluster = [r]
                            kw_clusters.append(curr_cluster)

                            for cluster in kw_clusters:
                                u_kw = cluster[0]
                                for kr in cluster[1:]:
                                    u_kw = u_kw | kr
                                detected_anchors.append({
                                    "cat": cat,
                                    "kw": kw,
                                    "rect": u_kw
                                })

            # Deduplicate sub-anchors
            dedup_anchors = []
            for a in detected_anchors:
                is_sub = False
                for other in detected_anchors:
                    if a is not other:
                        if (a["rect"] in other["rect"] and len(other["kw"]) > len(a["kw"])) or \
                           (other["kw"].endswith(a["kw"]) and len(other["kw"]) > len(a["kw"])):
                            is_sub = True
                            break
                if not is_sub:
                    dedup_anchors.append(a)
            detected_anchors = dedup_anchors
            detected_anchors.sort(key=lambda a: a["rect"].x0)

            # Clamping line rect with adjacent lines to prevent vertical line bleed
            l_mid = sum((w[0].y0 + w[0].y1) / 2 for w in l_words) / len(l_words) if l_words else line_rect.y0
            split_top = line_rect.y0
            if l_idx > 0:
                prev_words = line_objects[l_idx - 1]["words"]
                prev_mid = sum((w[0].y0 + w[0].y1) / 2 for w in prev_words) / len(prev_words) if prev_words else line_objects[l_idx - 1]["rect"].y0
                if l_mid > prev_mid:
                    split_top = max(line_rect.y0, (prev_mid + l_mid) / 2.0)

            split_bottom = line_rect.y1
            if l_idx + 1 < len(line_objects):
                nxt_words = line_objects[l_idx + 1]["words"]
                nxt_mid = sum((w[0].y0 + w[0].y1) / 2 for w in nxt_words) / len(nxt_words) if nxt_words else line_objects[l_idx + 1]["rect"].y0
                if nxt_mid > l_mid:
                    split_bottom = min(line_rect.y1, (l_mid + nxt_mid) / 2.0)

            for a_idx, anchor in enumerate(detected_anchors):
                a_rect = anchor["rect"]
                a_cat = anchor["cat"]
                clamped_a_rect = pymupdf.Rect(
                    max(0, a_rect.x0 - 2),
                    max(split_top, a_rect.y0 - 2),
                    a_rect.x1 + 2,
                    min(split_bottom, a_rect.y1 + 2)
                )
                eng_anchor = f"form_anchor{suffix.lower()}"
                add_target_fn(clamped_a_rect, f"{a_cat}_LABEL{suffix}", anchor["kw"], engine=eng_anchor)

                # Prescriber/Doctor signature adjacent lines & signature ink
                if re.search(r"\b(?:Signature\s+of\s+Prescriber|Unterschrift(?:\s+des\s+Arztes)?|Prescriber(?:'s|’s)?\s*Signature|Doctor(?:'s|’s)?\s*Signature|Firma(?:\s+(?:del\s+)?médico)?)\b", anchor["kw"], re.I):
                    if l_idx + 1 < len(line_objects):
                        nxt_line = line_objects[l_idx + 1]
                        if nxt_line["rect"].y0 - line_rect.y1 <= 60:
                            if not self.is_whitelisted(nxt_line["text"]) and re.search(r"[A-Za-z]{2,}", nxt_line["text"]):
                                add_target_fn(nxt_line["rect"], "DOCTOR_NAME", nxt_line["text"], engine=f"doctor_signature_adjacent{suffix.lower()}")
                    if l_idx > 0:
                        prev_line = line_objects[l_idx - 1]
                        if line_rect.y0 - prev_line["rect"].y1 <= 60:
                            if not self.is_whitelisted(prev_line["text"]) and re.search(r"[A-Za-z]{2,}", prev_line["text"]):
                                add_target_fn(prev_line["rect"], "DOCTOR_NAME", prev_line["text"], engine=f"doctor_signature_adjacent{suffix.lower()}")

                    sig_boxes = [
                        pymupdf.Rect(max(0, clamped_a_rect.x0 - 40), max(0, clamped_a_rect.y0 - 100), min(search_page.rect.width, clamped_a_rect.x1 + 120), clamped_a_rect.y0),
                        pymupdf.Rect(max(0, clamped_a_rect.x0 - 40), clamped_a_rect.y1, min(search_page.rect.width, clamped_a_rect.x1 + 120), min(search_page.rect.height, clamped_a_rect.y1 + 100)),
                    ]
                    for s_box in sig_boxes:
                        if s_box.width > 20 and s_box.height > 15:
                            ix0 = max(0, int(s_box.x0 / scale_px_x))
                            iy0 = max(0, int(s_box.y0 / scale_px_y))
                            ix1 = min(page_bgr.shape[1], int(s_box.x1 / scale_px_x))
                            iy1 = min(page_bgr.shape[0], int(s_box.y1 / scale_px_y))
                            roi = page_bgr[iy0:iy1, ix0:ix1]
                            if roi.size > 0:
                                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
                                ink_pixels = cv2.countNonZero(thresh)
                                if ink_pixels > 200 and (ink_pixels / roi.size) > 0.02:
                                    add_target_fn(s_box, "DOCTOR_SIGNATURE", "[HANDWRITTEN_SIGNATURE]", engine=f"signature_anchor_ink{suffix.lower()}")

                start_x = a_rect.x1 + 2
                if a_idx + 1 < len(detected_anchors):
                    end_x = max(start_x, detected_anchors[a_idx + 1]["rect"].x0 - 4)
                else:
                    end_x = line_rect.x1 + 4

                # Check for OCR token fusion
                fused_token = None
                for w in l_words:
                    if w[0].x0 <= a_rect.x0 + 8 and w[0].x1 > a_rect.x1 + 6:
                        fused_rect = pymupdf.Rect(start_x, w[0].y0, min(end_x, w[0].x1), w[0].y1)
                        fused_val = re.sub(r"^.*?[:\-=~+/|]+", "", w[1]).strip()
                        clean_fused = re.sub(r"[:\-=~+._\s|]", "", fused_val)
                        if len(clean_fused) > 1 and not self.is_whitelisted(fused_val):
                            fused_token = (fused_rect, fused_val)
                        break

                normal_words = [w for w in l_words if start_x - 4 <= w[0].x0 <= end_x]
                if fused_token:
                    candidate_words = [fused_token] + normal_words
                else:
                    candidate_words = normal_words

                same_line_words = []
                prev_x = a_rect.x1
                for sw_rect, sw_txt in candidate_words:
                    gap = sw_rect.x0 - prev_x
                    if same_line_words and gap > 120:
                        break
                    if not same_line_words and gap > 180:
                        break
                    same_line_words.append((sw_rect, sw_txt))
                    prev_x = sw_rect.x1

                same_line_text = " ".join(w[1] for w in same_line_words)
                clean_val = re.sub(r"[:\-=~+._\s|]", "", same_line_text)

                is_emergency_header = bool(re.search(r"In\s+Cas[ce]\s+of\s+Emergency", anchor["kw"], re.I))
                is_sublabel_instruction = bool(re.search(r"^\s*\(.*?(?:forename|address|vorname|geb|geboren|name|anschrift).*?\)\s*$", same_line_text, re.I)) or bool(re.search(r"^\s*\(.*?\)\s*$", same_line_text))

                if same_line_words and is_sublabel_instruction and a_cat in ("PATIENT_NAME", "ADDRESS"):
                    sublabel_rect = same_line_words[0][0]
                    for sw in same_line_words[1:]:
                        sublabel_rect = sublabel_rect | sw[0]
                    add_target_fn(sublabel_rect, f"{a_cat}_LABEL{suffix}", same_line_text, engine=f"form_sublabel{suffix.lower()}")
                    has_same_line_val = False
                elif same_line_words and len(clean_val) > 2 and not is_emergency_header and not re.match(r"^(?:oe|—|_|-|~)+$", clean_val, re.I):
                    val_rect = same_line_words[0][0]
                    for sw in same_line_words[1:]:
                        val_rect = val_rect | sw[0]
                    val_rect = pymupdf.Rect(
                        max(0, val_rect.x0 - 2),
                        max(split_top, val_rect.y0 - 2),
                        val_rect.x1 + 2,
                        min(split_bottom, val_rect.y1 + 2)
                    )
                    add_target_fn(val_rect, f"{a_cat}_VALUE{suffix}", same_line_text, engine=f"form_field_expansion{suffix.lower()}")
                    has_same_line_val = True
                else:
                    has_same_line_val = False

                if not has_same_line_val or (is_sublabel_instruction and a_cat in ("PATIENT_NAME", "ADDRESS")):
                    # 2D Column Table Parsing: Check rows directly below for this anchor column
                    col_x0 = a_rect.x0 - 25
                    col_x1 = max(a_rect.x1 + 180, end_x)
                    is_multi_row = a_cat in ("PATIENT_NAME", "ADDRESS")
                    prev_y1 = line_rect.y1
                    name_x0 = a_rect.x0

                    for offset in range(1, 4 if is_multi_row else 3):
                        if l_idx + offset < len(line_objects):
                            nxt_l = line_objects[l_idx + offset]
                            clean_next = re.sub(r"[-_=~a.,\s—|]", "", nxt_l["text"]).strip()
                            if not clean_next or re.match(r"^(?:oe|—|_|-|~|aaa)+$", nxt_l["text"].strip(), re.I):
                                continue

                            if is_multi_row:
                                if nxt_l["rect"].y0 - prev_y1 > 28:
                                    break
                                if nxt_l["rect"].y1 >= search_page.rect.height * 0.45:
                                    break
                            else:
                                if nxt_l["rect"].y0 - line_rect.y1 > 110:
                                    break

                            col_words = [w for w in nxt_l["words"] if col_x0 <= w[0].x0 <= col_x1]
                            if col_words:
                                col_txt = " ".join(w[1] for w in col_words)
                                clean_col = re.sub(r"[-_=~a.,\s—|]", "", col_txt).strip()
                                if re.search(r"\b(?:Rx|Rp\.?|Disp\.?|Medikament|Tablette|Kapsel|Age|Sex|Gender|DOB|Date|Telefon|Phone)\b", col_txt, re.I):
                                    break
                                if clean_col and not self.is_whitelisted(col_txt) and not re.match(r"^(?:oe|—|_|-|~|aaa)+$", clean_col, re.I):
                                    if is_multi_row and abs(col_words[0][0].x0 - name_x0) > 35:
                                        break

                                    valid = False
                                    if a_cat == "CONTACT":
                                        valid = bool(re.search(r"[\d@]", col_txt))
                                    elif a_cat == "DATE":
                                        valid = bool(re.search(r"\d|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", col_txt, re.I))
                                    elif a_cat == "IDENTIFIER":
                                        valid = bool(re.search(r"\d", col_txt))
                                    elif a_cat == "DEMOGRAPHIC":
                                        valid = bool(re.search(r"\d|Male|Female|M|F|Normal", col_txt, re.I))
                                    elif a_cat == "INSURANCE":
                                        valid = bool(re.search(r"[A-Za-z]", col_txt))
                                    elif a_cat == "PATIENT_NAME":
                                        valid = bool(re.search(r"(?:[A-Z]{2,}|[A-Z][a-zA-Z]{2,})", col_txt)) and not self.is_whitelisted(col_txt)
                                    elif a_cat == "ADDRESS":
                                        valid = bool(re.search(r"[A-Za-z0-9]", col_txt)) and not self.is_whitelisted(col_txt)

                                    if valid:
                                        c_rect = col_words[0][0]
                                        for cw in col_words[1:]:
                                            c_rect = c_rect | cw[0]
                                        c_rect = pymupdf.Rect(
                                            max(0, c_rect.x0 - 2),
                                            max(0, c_rect.y0 - 2),
                                            c_rect.x1 + 2,
                                            c_rect.y1 + 2
                                        )
                                        target_cat = "ADDRESS_BLOCK" if (is_multi_row and offset > 1) else f"{a_cat}_VALUE_2D{suffix}"
                                        add_target_fn(c_rect, target_cat, col_txt, engine=f"table_column_2d{suffix.lower()}")
                                        prev_y1 = nxt_l["rect"].y1
                                        if not is_multi_row:
                                            break

                    # OCR-independent handwritten ink-fill detection
                    # 1. Right-side fill zone
                    next_anchor_x0 = detected_anchors[a_idx + 1]["rect"].x0 - 4 if a_idx + 1 < len(detected_anchors) else line_rect.x1 + 220
                    right_end_x = min(next_anchor_x0, anchor["rect"].x1 + 220)
                    if right_end_x > anchor["rect"].x1 + 10:
                        right_zone = pymupdf.Rect(
                            anchor["rect"].x1 + 4,
                            max(0, anchor["rect"].y0 - 6),
                            right_end_x,
                            min(search_page.rect.height, anchor["rect"].y1 + 6)
                        )
                        ink_rect = self._detect_handwritten_fill_in_zone(
                            page_bgr, right_zone, scale_px_x, scale_px_y, self._last_whitelisted_rects
                        )
                        if ink_rect:
                            add_target_fn(ink_rect, f"{a_cat}_HANDWRITTEN_FILL{suffix}", "[HANDWRITTEN_INK]", engine="spatial_ink_fill")

                    # 2. Under-label box zone
                    clinical_boundary = search_page.rect.height * 0.45
                    if anchor["rect"].y1 < clinical_boundary:
                        under_zone = pymupdf.Rect(
                            anchor["rect"].x0,
                            anchor["rect"].y1 + 2,
                            min(search_page.rect.width * 0.85, anchor["rect"].x1 + 180),
                            min(clinical_boundary, anchor["rect"].y1 + 65)
                        )
                        if under_zone.height > 10:
                            under_fill = self._detect_handwritten_fill_in_zone(
                                page_bgr, under_zone, scale_px_x, scale_px_y, self._last_whitelisted_rects
                            )
                            if under_fill:
                                add_target_fn(under_fill, f"{a_cat}_HANDWRITTEN_FILL{suffix}", "[HANDWRITTEN_INK]", engine="spatial_ink_fill")

    def find_pii_targets_for_page(self, page: pymupdf.Page, page_num: int) -> List[RedactionTarget]:
        """
        Scans a single raster page using dynamic multi-column layout parsing,
        regex entity recognition, and strict clinical whitelist protection.
        """
        max_pt = max(page.rect.width, page.rect.height)
        if max_pt < 550:
            effective_dpi = 300
        elif max_pt < 900:
            effective_dpi = self.dpi if (page.rect.width >= 550 and page.rect.height >= 700) else 200
        elif max_pt <= 2000:
            effective_dpi = self.dpi
        else:
            effective_dpi = 72
        tp = page.get_textpage_ocr(language="eng", dpi=effective_dpi, tessdata=self.tessdata_dir)
        words = page.get_text("words", textpage=tp)
        full_text = page.get_text(textpage=tp)

        eu_marker_regex = re.compile(
            r"[äöüßéèêëàâîïôûùçñáíóúÁÉÍÓÚÑÄÖÜ]|\b(?:Rezept|Krankenkasse|Geb|Ausstellungsdatum|Vorname|Unterschrift|Arzt|Receta|Ordonnance|Nom|Prénom|Calle|Straße)\b",
            re.IGNORECASE
        )
        is_eu = bool(eu_marker_regex.search(full_text))
        all_langs = "eng"
        if len(words) < 20 or is_eu:
            all_langs = get_ocr_languages(self.tessdata_dir)
            if all_langs != "eng":
                tp = page.get_textpage_ocr(language=all_langs, dpi=effective_dpi, tessdata=self.tessdata_dir)
                words = page.get_text("words", textpage=tp)
                full_text = page.get_text(textpage=tp)

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
                    for r in page.search_for(match_str, textpage=tp):
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
                # If a BARCODE box overlaps with any whitelisted_rects or overlaps with >= 2 OCR words
                # containing letters ([A-Za-z]{3,}), reject it!
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
                    # Clamp rect away from wr along inter-line boundary so redaction never covers adjacent line's clinical text
                    split_y = (rect_mid + wr_mid) / 2.0
                    if rect_mid > wr_mid:
                        rect.y0 = max(rect.y0, split_y)
                    else:
                        rect.y1 = min(rect.y1, split_y)
                    continue
                if inter.get_area() > 0.15 * min(rect.get_area(), wr.get_area()):
                    return False
            return True

        def add_target(rect: pymupdf.Rect, category: str, text: str, engine: str = "ocr_dynamic"):
            if category in ("PATIENT_NAME", "PATIENT_NAME_VALUE", "PATIENT_NAME_VALUE_CLAHE",
                            "PATIENT_NAME_VALUE_2D", "PATIENT_HONORIFIC", "PATIENT_HONORIFIC_ENHANCED",
                            "DOCTOR_NAME", "DOCTOR_NAME_ENHANCED") or \
               category.startswith("PATIENT_NAME") or category.startswith("DOCTOR_NAME"):
                rect = self._expand_person_span_on_line(rect, active_lines)

            if not can_redact(rect, text):
                return
            if rect.get_area() <= 0:
                return
            for idx, r in enumerate(recorded_rects):
                inter = rect & r
                if not inter.is_empty:
                    inter_a = inter.get_area()
                    if inter_a > 0.70 * r.get_area() and rect.get_area() > r.get_area() * 1.3:
                        merged = r | rect
                        recorded_rects[idx] = merged
                        targets[idx].bbox = BoundingBox(x0=merged.x0, y0=merged.y0, x1=merged.x1, y1=merged.y1)
                        return
                    if inter_a > 0.75 * min(rect.get_area(), r.get_area()):
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

        # 0. Automatic QR Code and Barcode Detection
        code_boxes = self._detect_qr_and_barcodes(page, bgr=page_bgr, scale_x=scale_px_x, scale_y=scale_px_y)
        for c_rect, payload in code_boxes:
            add_target(c_rect, "BARCODE_OR_QR", payload or "BARCODE_OR_QR", engine="opencv_detector")

        # Reconstruct logical visual lines
        line_objects = self._reconstruct_visual_lines(words)
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
        self._parse_form_anchors_on_lines(
            line_objects=line_objects,
            search_page=page,
            textpage=tp,
            page_bgr=page_bgr,
            scale_px_x=scale_px_x,
            scale_px_y=scale_px_y,
            add_target_fn=add_target,
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
                for r in page.search_for(m_str, textpage=tp):
                    add_target(r, "DISPENSING_PHARMACY", m_str, engine="inline_dispense_rule")

        # Anywhere-on-page Pharmacy/Clinic/Healthcare blocks
        pharmacy_block_re = re.compile(
            r"\b(?:[A-ZÄÖÜ][A-Za-z0-9\s',.&-]{2,40}\s+(?:Community\s+Pharmacy|Pharmacy|Chemist|Medical\s+Centre|Medical|Clinic|Hospital|Optical|Optometry|Vision|Eye\s+Care|Dental|Praxis|Klinik|Apotheke))\b",
            re.IGNORECASE
        )
        for match in pharmacy_block_re.finditer(full_text):
            m_str = match.group(0).strip()
            if not self.is_whitelisted(m_str) and len(m_str) > 5:
                for r in page.search_for(m_str, textpage=tp):
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
                r"\b[A-ZÄÖÜÁÉÍÓÚÑ][a-zA-Zäöüßáéíóúñ\-'.]+(?:[^\S\r\n]+(?:[A-ZÄÖÜÁÉÍÓÚÑ]\.?|[A-ZÄÖÜÁÉÍÓÚÑ][a-zA-Zäöüßáéíóúñ\-'.]+))*[,\s]+(?:M\.?B\.?B\.?S|M\.?D|M\.?S|D\.?M\.?R\.?D|D\.?N\.?B|F\.?R\.?C\.?R|F\.?R\.?C\.?S|FRACGP|MRCGP|MBChB|FRACP|O\.?D\.?|D\.?O\.?|PharmD)\b"
            )),
            ("DOCTOR_CREDENTIALS", re.compile(r"\b(?:MBBS|M\.B\.B\.S|MD|M\.D|MS|M\.S|DMRD|D\.M\.R\.D|DNB|D\.N\.B|FRCR|FRCS|FRACGP|MRCGP|MBChB|FRACP|O\.?D\.?|D\.?O\.?|PharmD)\b")),
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
                    rects = page.search_for(match_str, textpage=tp)
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
                                    # Deduplicate with existing signature boxes
                                    already_detected = any(
                                        t.category == "DOCTOR_SIGNATURE" and
                                        not (sig_rect & pymupdf.Rect(t.bbox.x0, t.bbox.y0, t.bbox.x1, t.bbox.y1)).is_empty
                                        for t in targets
                                    )
                                    if not already_detected:
                                        p_pix = page.get_pixmap()
                                        p_arr = np.frombuffer(p_pix.samples, dtype=np.uint8).reshape((p_pix.height, p_pix.width, p_pix.n))
                                        p_gray = cv2.cvtColor(p_arr, cv2.COLOR_RGBA2GRAY) if p_pix.n == 4 else (cv2.cvtColor(p_arr, cv2.COLOR_RGB2GRAY) if p_pix.n == 3 else p_arr)
                                        scale_x = p_pix.width / page.rect.width
                                        scale_y = p_pix.height / page.rect.height
                                        ix0, iy0 = int(sig_rect.x0 * scale_x), int(sig_rect.y0 * scale_y)
                                        ix1, iy1 = int(sig_rect.x1 * scale_x), int(sig_rect.y1 * scale_y)
                                        roi = p_gray[iy0:iy1, ix0:ix1]
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
                    rects = page.search_for(match_str, textpage=tp)
                    for r in rects:
                        add_target(r, cat, match_str, engine="statutory_scanner")

        # Generic Header-Zone Facility & Provider Rule (Top 35% of Document and lower-half dispensing facility)
        header_zone_y = page.rect.height * 0.35
        facility_keywords = re.compile(
            r"\b(?:Hospital|Clinic|Imaging|Diagnostics?|Institute|Centre|Center|Medical\s+Centre|Medical|Superspecialit(?:y|ies)|Laborator(?:y|ies)|Pharmacy|Chemist|Optical|Optometry|Vision|Eye\s+Care|Dental|Healthcare|Praxis|Klinik|Apotheke)\b",
            re.IGNORECASE
        )
        provider_creds = re.compile(
            r"\b(?:M\.?B\.?B\.?S|M\.?D|M\.?S|D\.?M\.?R\.?D|D\.?N\.?B|F\.?R\.?C\.?R|F\.?R\.?C\.?S|Regn?\.?|MCI|NMC|FRACGP|MRCGP|MBChB|FRACP|O\.?D\.?|D\.?O\.?|PharmD)\b",
            re.IGNORECASE
        )
        for idx, line in enumerate(line_objects):
            line_str = line["text"].strip()
            if self.is_whitelisted(line_str) or len(line_str) < 3:
                continue
            if line["rect"].y1 <= header_zone_y:
                if facility_keywords.search(line_str):
                    add_target(line["rect"], "CLINIC_FACILITY", line_str, engine="header_facility_rule")
                elif provider_creds.search(line_str):
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
        ner_candidates = self.ner_engine.find_entities(full_text)
        for cand in ner_candidates:
            matched_str = cand["text"]
            if not self.is_whitelisted(matched_str) and len(matched_str) >= 3:
                rects = page.search_for(matched_str, textpage=tp)
                for r in rects:
                    if is_in_clinical_zone(r):
                        continue
                    add_target(r, cand["category"], matched_str, engine="spacy_ner")
                if not rects and len(matched_str.split()) > 1:
                    for part in matched_str.split():
                        if len(part) >= 4 and not self.is_whitelisted(part):
                            p_rects = page.search_for(part, textpage=tp)
                            for pr in p_rects:
                                if is_in_clinical_zone(pr):
                                    continue
                                add_target(pr, cand["category"], part, engine="spacy_ner")

        # -------------------------------------------------------------------
        # 3. CLAHE Contrast-Enhanced Pass (Faint Headers, Glare & Blurry Text)
        # -------------------------------------------------------------------
        tmp_doc_enh = None
        try:
            is_std_sheet = (page.rect.width >= 550 and page.rect.height >= 700)
            h_ratio = 0.45 if is_std_sheet else (1.0 if (max_pt < 800 or len(words) < 45) else 0.45)
            words_enh, p_enh, tp_enh, tmp_doc_enh = self._get_enhanced_fullpage_ocr(page, h_ratio=h_ratio, lang=all_langs if (is_eu or len(words) < 20) else "eng")
            full_enh = p_enh.get_text(textpage=tp_enh)

            # 3a. Reconstruct visual lines and run unified form anchor parsing on CLAHE pass
            line_objects_enh = self._reconstruct_visual_lines(words_enh)
            active_lines = line_objects_enh
            self._parse_form_anchors_on_lines(
                line_objects=line_objects_enh,
                search_page=p_enh,
                textpage=tp_enh,
                page_bgr=page_bgr,
                scale_px_x=scale_px_x,
                scale_px_y=scale_px_y,
                add_target_fn=add_target,
                suffix="_CLAHE"
            )

            # 3b. Standalone regexes on CLAHE pass
            for cat, reg in standalone_regexes:
                for match in reg.finditer(full_enh):
                    match_str = match.group(0).strip()
                    if not self.is_whitelisted(match_str) and len(match_str) > 2:
                        rects = p_enh.search_for(match_str, textpage=tp_enh)
                        for r in rects:
                            if cat in ("ID_TOKEN", "CITY_STATE") and is_in_clinical_zone(r):
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
                    elif provider_creds.search(line_str):
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
                        if is_in_clinical_zone(r):
                            continue
                        add_target(r, f"{cand['category']}_ENHANCED", matched_str, engine="spacy_ner_clahe")
                    if not rects and len(matched_str.split()) > 1:
                        for part in matched_str.split():
                            if len(part) >= 4 and not self.is_whitelisted(part):
                                p_rects = p_enh.search_for(part, textpage=tp_enh)
                                for pr in p_rects:
                                    if is_in_clinical_zone(pr):
                                        continue
                                    add_target(pr, f"{cand['category']}_ENHANCED", part, engine="spacy_ner_clahe")
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

        addr_line_re = re.compile(r"[A-Za-zäöüß]+\s*\d+|\b\d{4,6}\s+[A-Za-zäöüß]+", re.IGNORECASE)
        street_keyword_re = re.compile(
            r"\b(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Way|Lane|Ln|Boulevard|Blvd|Court|Ct|Straße|Str|Weg|Platz|Allee|Gasse|Rue|Bd|Calle|Paseo|Via)\b",
            re.IGNORECASE
        )
        for pt in patient_name_targets:
            curr_y1 = pt.bbox.y1
            curr_x0 = pt.bbox.x0
            for _ in range(3):
                found_next_line = False
                for l in all_candidate_lines:
                    v_gap = l["rect"].y0 - curr_y1
                    x0_diff = abs(l["rect"].x0 - curr_x0)
                    if 0 < v_gap < 25 and x0_diff < 30 and l["rect"].y1 <= page.rect.height * 0.45:
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

        return targets

    def find_pii_targets(self, doc: pymupdf.Document) -> List[RedactionTarget]:
        """Scans all pages of a document and returns detected PII targets."""
        all_targets: List[RedactionTarget] = []
        for idx, page in enumerate(doc):
            targets = self.find_pii_targets_for_page(page, idx + 1)
            all_targets.extend(targets)
        return all_targets

    def redact(self, input_path: str, output_path: Optional[str] = None, jurisdiction: str = "AUTO") -> Dict[str, Any]:
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
            with Image.open(input_path) as orig_img:
                orig_w, orig_h = orig_img.size

            with pymupdf.open(input_path) as img_doc:
                pdf_bytes = img_doc.convert_to_pdf()
            doc = pymupdf.open("pdf", pdf_bytes)
            page = doc[0]

            # 1. OCR, form parsing, and visual grounder targets
            targets = self.find_pii_targets_for_page(page, 1)

            # Resolve jurisdiction using actual extracted OCR text (fixes empty string bug)
            ocr_text = getattr(self, "_last_extracted_text", "") or page.get_text()
            resolved_jur = JurisdictionDetector.resolve_jurisdiction(ocr_text, jurisdiction)

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
                    page.add_redact_annot(rect, fill=(0, 0, 0))

            page.apply_redactions()

            scale_back_x = orig_w / page.rect.width
            scale_back_y = orig_h / page.rect.height

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if out_ext_lower in self.IMAGE_EXTENSIONS:
                scale_x = orig_w / page.rect.width
                scale_y = orig_h / page.rect.height
                pix = page.get_pixmap(matrix=pymupdf.Matrix(scale_x, scale_y))
                pix.save(output_path)
            else:
                doc.save(output_path, garbage=4, deflate=True)

            doc.close()

            reported_targets = []
            for t in targets:
                reported_targets.append(RedactionTarget(
                    page_number=t.page_number,
                    category=t.category,
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

            from .forensics import ForensicAuditor
            conf = ForensicAuditor.evaluate_redaction_confidence(
                report={"redacted_targets": [t.model_dump() for t in reported_targets]},
                extracted_text=getattr(self, "_last_extracted_text", "") or page.get_text(),
                is_raster=True
            )

            return {
                "input_file": input_path,
                "output_file": output_path,
                "jurisdiction": resolved_jur,
                "format_type": "SCANNED_RASTER_IMAGE",
                "total_entities_redacted": len(reported_targets),
                "redacted_targets": [t.model_dump() for t in reported_targets],
                "confidence_score": conf["confidence_score"],
                "triage_status": conf["triage_status"],
                "review_reasons": conf["review_reasons"]
            }

        # Handle PDF Input
        doc = pymupdf.open(input_path)
        all_targets: List[RedactionTarget] = []
        collected_ocr_text: List[str] = []

        for p_idx, page in enumerate(doc):
            targets = self.find_pii_targets_for_page(page, p_idx + 1)
            all_targets.extend(targets)
            if hasattr(self, "_last_extracted_text") and self._last_extracted_text:
                collected_ocr_text.append(self._last_extracted_text)

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
                    page.add_redact_annot(rect, fill=(0, 0, 0))

            page.apply_redactions()

        full_doc_text = " ".join(collected_ocr_text) if collected_ocr_text else " ".join(p.get_text() for p in doc)
        resolved_jur = JurisdictionDetector.resolve_jurisdiction(full_doc_text, jurisdiction)

        # Wipe metadata to eliminate digital trace leakage
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

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if out_ext_lower in self.IMAGE_EXTENSIONS:
            pix = doc[0].get_pixmap(dpi=self.dpi)
            pix.save(output_path)
        else:
            doc.save(output_path, garbage=4, deflate=True)

        doc.close()

        from .forensics import ForensicAuditor
        conf = ForensicAuditor.evaluate_redaction_confidence(
            report={"redacted_targets": [t.model_dump() for t in all_targets]},
            extracted_text="\n".join(collected_ocr_text),
            is_raster=True
        )

        return {
            "input_file": input_path,
            "output_file": output_path,
            "jurisdiction": resolved_jur,
            "format_type": "SCANNED_RASTER_PDF",
            "total_entities_redacted": len(all_targets),
            "redacted_targets": [t.model_dump() for t in all_targets],
            "confidence_score": conf["confidence_score"],
            "triage_status": conf["triage_status"],
            "review_reasons": conf["review_reasons"]
        }
