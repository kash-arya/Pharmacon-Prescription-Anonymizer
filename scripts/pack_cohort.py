#!/usr/bin/env python3
"""
Automated Lightweight Cohort Packager for Pharmacon.
Supports both image folders (.jpg, .png) and multi-page/single-page scanned PDFs (.pdf).
Scans raw writer calibration sources, crops lines via SIFT homography,
and packages them into a compact ZIP for Google Colab execution.
"""

import os, sys, glob, re, zipfile
import cv2
import numpy as np
import pymupdf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from pharmacon_ai.crop import process_photo

SEARCH_BASE = os.path.expanduser("~/Documents/From S23 by GS-KDE Connect")
PDF_PATH    = os.path.join(REPO_ROOT, "Pharmacon_Writer_Calibration_Protocol.pdf")
ZIP_OUTPUT  = os.path.join(REPO_ROOT, "colab_packages", "pharmacon_cohort_crops.zip")

def render_pdf_page(page, dpi=150):
    """Render a PyMuPDF page into a BGR numpy array for OpenCV/SIFT processing."""
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _map_multipage_pdf(pdf_file):
    """Extract sheets from a multi-page PDF (scanned or digital)."""
    doc = pymupdf.open(pdf_file)
    n_pages = len(doc)
    mapping = {}

    # Check if PDF text allows semantic matching
    text_matched = False
    for p_no in range(n_pages):
        page = doc[p_no]
        txt = page.get_text()
        if "E01." in txt or "brown fox" in txt: mapping[2] = render_pdf_page(page); text_matched = True
        elif "E05." in txt or "Azithromycin" in txt: mapping[3] = render_pdf_page(page); text_matched = True
        elif "E09." in txt or "Cetirizine 5 mg" in txt: mapping[4] = render_pdf_page(page); text_matched = True
        elif "E13." in txt or "Ibuprofen 400" in txt: mapping[5] = render_pdf_page(page); text_matched = True
        elif "V01." in txt or "Pantoprazole" in txt: mapping[6] = render_pdf_page(page); text_matched = True
        elif "T01." in txt or "Paracetamol 650" in txt: mapping[7] = render_pdf_page(page); text_matched = True
        elif "T05." in txt or "Quetiapine 25" in txt: mapping[8] = render_pdf_page(page); text_matched = True

    # Fallback to sequential index mapping if scanned images have no text layer
    if not text_matched or len(mapping) < 7:
        mapping.clear()
        if n_pages == 7:
            for i in range(7):
                mapping[i + 2] = render_pdf_page(doc[i])
        elif n_pages >= 8:
            # Assume page 0 is cover/instruction page
            for i in range(7):
                mapping[i + 2] = render_pdf_page(doc[i + 1])
        else:
            # Map whatever pages exist sequentially
            for i in range(min(n_pages, 7)):
                mapping[i + 2] = render_pdf_page(doc[i])
    return mapping

def map_writer_source(writer_source):
    """
    Map writer source (directory or standalone PDF) to {ref_page_idx: img_or_path}.
    Returns a dictionary of reference page indices (2..8).
    """
    if os.path.isfile(writer_source) and writer_source.lower().endswith(".pdf"):
        return _map_multipage_pdf(writer_source)

    if not os.path.isdir(writer_source):
        return {}

    # Check if directory contains a single multi-page PDF
    pdf_files = [f for f in os.listdir(writer_source) if f.lower().endswith(".pdf")]
    if len(pdf_files) == 1:
        full_pdf = os.path.join(writer_source, pdf_files[0])
        try:
            doc = pymupdf.open(full_pdf)
            if len(doc) >= 7:
                return _map_multipage_pdf(full_pdf)
        except Exception:
            pass

    # Directory with individual files (images or single-page PDFs)
    files = sorted(os.listdir(writer_source))
    mapping = {}
    for f in files:
        fl = f.lower()
        full = os.path.join(writer_source, f)
        
        # Decide if this is a valid sheet file
        if not (fl.endswith((".jpg", ".jpeg", ".png")) or fl.endswith(".pdf")):
            continue

        # If individual PDF, render page 0 to numpy array; otherwise keep file path
        if fl.endswith(".pdf"):
            try:
                doc = pymupdf.open(full)
                sheet_data = render_pdf_page(doc[0])
            except Exception as e:
                print(f"    Failed rendering PDF {f}: {e}")
                continue
        else:
            sheet_data = full

        # Match target sheet
        if "ts1" in fl or "trains1" in fl or ("train" in fl and "1" in fl): mapping[2] = sheet_data
        elif "ts2" in fl or "trains2" in fl or ("train" in fl and "2" in fl): mapping[3] = sheet_data
        elif "ts3" in fl or "trains3" in fl or ("train" in fl and "3" in fl): mapping[4] = sheet_data
        elif "ts4" in fl or "trains4" in fl or ("train" in fl and "4" in fl): mapping[5] = sheet_data
        elif "vs" in fl or "valid" in fl: mapping[6] = sheet_data
        elif "hs1" in fl or "heldouts1" in fl or ("held" in fl and "1" in fl): mapping[7] = sheet_data
        elif "hs2" in fl or "heldouts2" in fl or ("held" in fl and "2" in fl): mapping[8] = sheet_data

    return mapping

def discover_candidates():
    """Discover writer folders and standalone PDF files in dataset/ or search path."""
    local_dataset = os.path.join(REPO_ROOT, "dataset")

    def _wid_num(path):
        m = re.search(r"[wW](\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 999

    candidates = []
    if os.path.isdir(local_dataset):
        for item in os.listdir(local_dataset):
            full = os.path.join(local_dataset, item)
            # Exclude blank reference/template PDFs
            if "blank" in item.lower() or "protocol" in item.lower():
                continue
            if os.path.isdir(full) or (os.path.isfile(full) and full.lower().endswith(".pdf")):
                candidates.append(full)

    if not candidates and os.path.isdir(SEARCH_BASE):
        for item in glob.glob(os.path.join(SEARCH_BASE, "*CS*")):
            if os.path.isdir(item) or (os.path.isfile(item) and item.lower().endswith(".pdf")):
                candidates.append(item)

    return sorted(candidates, key=_wid_num)

def main():
    print("=" * 70)
    print("PHARMACON — DUAL-MODE COHORT PACKAGER (IMAGE FOLDERS + SCANNED PDFS)")
    print("=" * 70)

    candidate_sources = discover_candidates()
    print(f"Detected {len(candidate_sources)} writer source(s) in workspace.")

    os.makedirs(os.path.dirname(ZIP_OUTPUT), exist_ok=True)
    if os.path.exists(ZIP_OUTPUT):
        os.remove(ZIP_OUTPUT)

    total_lines = 0
    with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Package core code
        for root, dirs, files in os.walk(os.path.join(REPO_ROOT, "pharmacon_ai")):
            for file in files:
                if not file.endswith(".pyc") and "__pycache__" not in root:
                    full = os.path.join(root, file)
                    rel = os.path.relpath(full, REPO_ROOT)
                    zf.write(full, arcname=rel)

        # 2. Package protocol PDF
        zf.write(PDF_PATH, arcname=os.path.basename(PDF_PATH))

        # 3. Crop and pack each writer
        for idx, w_src in enumerate(candidate_sources, 1):
            source_name = os.path.basename(w_src)
            match = re.search(r"[wW](\d+)", source_name)
            w_id = f"W{int(match.group(1)):02d}" if match else f"W{idx:02d}"

            mapping = map_writer_source(w_src)
            mode_desc = "PDF" if (os.path.isfile(w_src) and w_src.lower().endswith('.pdf')) else "Folder"
            print(f"[{w_id}] Processing {mode_desc} '{source_name}' ({len(mapping)} sheets)...")

            w_lines = 0
            for page_idx, p_data in sorted(mapping.items()):
                try:
                    lines = process_photo(p_data, PDF_PATH, page_idx, writer_id=w_id)
                    for lid, data in lines.items():
                        _, buf = cv2.imencode(".jpg", data["image"], [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                        zf.writestr(f"crops/{w_id}/{lid}.jpg", buf.tobytes())
                        w_lines += 1
                except Exception as e:
                    print(f"    Error cropping page {page_idx}: {e}")

            total_lines += w_lines
            print(f"    Cropped {w_lines} lines cleanly.")

    size_mb = os.path.getsize(ZIP_OUTPUT) / (1024 * 1024)
    print("\n" + "=" * 70)
    print(f"SUCCESS! Output package created:")
    print(f"  Path      : {ZIP_OUTPUT}")
    print(f"  Writers   : {len(candidate_sources)}")
    print(f"  Lines     : {total_lines}")
    print(f"  File Size : {size_mb:.2f} MB  (Uploads to Colab in ~2-3 seconds!)")
    print("=" * 70)

if __name__ == "__main__":
    main()
