"""
2D Multi-Column Key-Value Form & Spatial Ink Parser Module.

Performs robust form field extraction across structured healthcare forms and prescriptions:
- Horizontal 1D proximity and fused OCR token recovery
- 2D multi-row column table parsing for multi-line patient demographics and addresses
- Prescriber/physician signature detection on adjacent lines & ink box search
- Optical handwritten cursive ink stroke detection in blank fill-in zones
- Horizontal name-token and honorific expansion across visual lines
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
import re
import pymupdf
import cv2
import numpy as np

try:
    from prescription_anonymizer.src.entities import (
        KEY_VALUE_ANCHORS,
        is_whitelisted as default_is_whitelisted,
    )
except ImportError:
    from src.entities import (
        KEY_VALUE_ANCHORS,
        is_whitelisted as default_is_whitelisted,
    )


def expand_person_span_on_line(
    rect: pymupdf.Rect,
    line_objects: List[Dict[str, Any]],
    is_whitelisted_fn: Optional[Callable[[str], bool]] = None
) -> pymupdf.Rect:
    """
    Expands a person bounding box horizontally to adjacent name tokens, honorifics,
    and initials on the same visual line within 35 pt gap, excluding form labels,
    stopwords, and clinical whitelist terms.
    """
    if not line_objects:
        return rect

    check_whitelist = is_whitelisted_fn or default_is_whitelisted

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
        elif abs((w_rect.y0 + w_rect.y1) / 2.0 - rect_mid_y) < max(12.0, w_rect.height * 0.5):
            if not (w_rect.x1 < rect.x0 - 5 or w_rect.x0 > rect.x1 + 5):
                intersecting_indices.append(i)

    if not intersecting_indices:
        closest_idx = None
        closest_dist = float("inf")
        for i, (w_rect, w_txt) in enumerate(words):
            dist = abs((w_rect.x0 + w_rect.x1) / 2.0 - (rect.x0 + rect.x1) / 2.0)
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
        if check_whitelist(clean):
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


def detect_handwritten_fill_in_zone(
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
            wl = whitelisted_rects or []
            for wr in wl:
                inter = stroke_rect & wr
                if not inter.is_empty and inter.get_area() > 0.10 * min(stroke_rect.get_area(), wr.get_area()):
                    return None
            return stroke_rect
    return None


def parse_form_anchors_on_lines(
    line_objects: List[Dict[str, Any]],
    search_page: pymupdf.Page,
    textpage: pymupdf.TextPage,
    page_bgr: np.ndarray,
    scale_px_x: float,
    scale_px_y: float,
    add_target_fn: Any,
    is_whitelisted_fn: Optional[Callable[[str], bool]] = None,
    whitelisted_rects: Optional[List[pymupdf.Rect]] = None,
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
    check_whitelist = is_whitelisted_fn or default_is_whitelisted

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
                        if not check_whitelist(nxt_line["text"]) and re.search(r"[A-Za-z]{2,}", nxt_line["text"]):
                            add_target_fn(nxt_line["rect"], "DOCTOR_NAME", nxt_line["text"], engine=f"doctor_signature_adjacent{suffix.lower()}")
                if l_idx > 0:
                    prev_line = line_objects[l_idx - 1]
                    if line_rect.y0 - prev_line["rect"].y1 <= 60:
                        if not check_whitelist(prev_line["text"]) and re.search(r"[A-Za-z]{2,}", prev_line["text"]):
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
                    if len(clean_fused) > 1 and not check_whitelist(fused_val):
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
            elif same_line_words and len(clean_val) > 2 and not is_emergency_header and not check_whitelist(same_line_text) and not re.match(r"^(?:oe|—|_|-|~)+$", clean_val, re.I):
                val_rect = same_line_words[0][0]
                for sw in same_line_words[1:]:
                    val_rect = val_rect | sw[0]
                val_rect = pymupdf.Rect(
                    max(0, val_rect.x0 - 2),
                    max(split_top, val_rect.y0 - 2),
                    val_rect.x1 + 2,
                    min(split_bottom, val_rect.y1 + 2)
                )
                is_clinical_encroachment = False
                if whitelisted_rects:
                    for wr in whitelisted_rects:
                        inter = val_rect & wr
                        if not inter.is_empty and inter.get_area() > 0.15 * min(val_rect.get_area(), wr.get_area()):
                            is_clinical_encroachment = True
                            break
                if not is_clinical_encroachment:
                    add_target_fn(val_rect, f"{a_cat}_VALUE{suffix}", same_line_text, engine=f"form_field_expansion{suffix.lower()}")
                    has_same_line_val = True
                else:
                    has_same_line_val = False
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
                            if clean_col and not check_whitelist(col_txt) and not re.match(r"^(?:oe|—|_|-|~|aaa)+$", clean_col, re.I):
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
                                    valid = bool(re.search(r"(?:[A-Z]{2,}|[A-Z][a-zA-Z]{2,})", col_txt)) and not check_whitelist(col_txt)
                                elif a_cat == "ADDRESS":
                                    valid = bool(re.search(r"[A-Za-z0-9]", col_txt)) and not check_whitelist(col_txt)

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
                    ink_rect = detect_handwritten_fill_in_zone(
                        page_bgr, right_zone, scale_px_x, scale_px_y, whitelisted_rects
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
                        under_fill = detect_handwritten_fill_in_zone(
                            page_bgr, under_zone, scale_px_x, scale_px_y, whitelisted_rects
                        )
                        if under_fill:
                            add_target_fn(under_fill, f"{a_cat}_HANDWRITTEN_FILL{suffix}", "[HANDWRITTEN_INK]", engine="spatial_ink_fill")
