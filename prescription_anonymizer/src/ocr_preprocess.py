"""
OCR Preprocessing & Visual Line Reconstruction Module.

Provides tessdata path resolution, multilingual OCR configuration,
visual line grouping with vertical overlap gating, and CLAHE adaptive
contrast enhancement for degraded smartphone photos and scanned PDFs.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import pymupdf
import cv2
import numpy as np


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


def reconstruct_visual_lines(words: List[Tuple]) -> List[Dict[str, Any]]:
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


def get_enhanced_fullpage_ocr(
    page: pymupdf.Page,
    dpi: int = 150,
    h_ratio: float = 0.45,
    lang: Optional[str] = None,
    tessdata_dir: Optional[str] = None
) -> Tuple[List[Any], pymupdf.Page, pymupdf.TextPage, pymupdf.Document]:
    """Performs CLAHE adaptive contrast preprocessing, denoising, and unsharp masking."""
    max_pt = max(page.rect.width, page.rect.height)
    is_low_mem = os.environ.get("RENDER_LOW_MEMORY") == "1"
    if is_low_mem:
        effective_dpi = 200 if max_pt < 550 else 120
    else:
        if max_pt < 550:
            effective_dpi = 300
        elif max_pt < 900:
            effective_dpi = dpi if (page.rect.width >= 550 and page.rect.height >= 700) else 200
        elif max_pt <= 2500:
            effective_dpi = dpi
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
    ocr_lang = lang if lang is not None else get_ocr_languages(tessdata_dir)
    tp_enh = p_enh.get_textpage_ocr(language=ocr_lang, dpi=effective_dpi, tessdata=tessdata_dir)
    words_enh = p_enh.get_text("words", textpage=tp_enh)

    try:
        del pix, arr, gray, cl_img, blurred, sharpened, buf
    except Exception:
        pass

    return words_enh, p_enh, tp_enh, tmp_doc


get_enhanced_header_ocr = get_enhanced_fullpage_ocr
