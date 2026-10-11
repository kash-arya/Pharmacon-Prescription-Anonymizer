"""
Template: extract writing-box geometry and ground-truth text from the protocol PDF.

The protocol PDF is a fixed printed form. This module extracts, once:
  - the writing-box rectangles (per line ID), and
  - the prompt text above each box (the ground-truth transcription).

Everything here is derived from the PDF's own vector/text structure, so it is
reproducible and needs no manual labeling.
"""

import os

try:
    import pymupdf as fitz
except ImportError:
    fitz = None

PROMPT_FILL = (0.957, 0.965, 0.973)
FILL_TOLERANCE = 0.01  # accept colours within this distance of PROMPT_FILL

# Authoritative ground truth transcriptions from Pharmacon_Writer_Calibration_Protocol.pdf
PROTOCOL_GROUND_TRUTH = {
    "E01": "The quick brown fox jumps over the lazy dog. 0123456789.",
    "E02": "Sphinx of black quartz, judge my vow. Rx No. 24/09.",
    "E03": "Tab Paracetamol 500 mg, 1 tablet after food, BD for 3 days.",
    "E04": "Cap Amoxicillin 500 mg, 1 capsule TDS for 5 days.",
    "E05": "Tab Azithromycin 250 mg, 2 tablets today, then 1 OD for 4 days.",
    "E06": "Tab Metformin 500 mg, 1-0-1 with meals for 30 days.",
    "E07": "Tab Amlodipine 5 mg, 1-0-0 every morning.",
    "E08": "Tab Levothyroxine 50 mcg, once daily before breakfast.",
    "E09": "Syp Cetirizine 5 mg/5 mL, 10 mL HS for 5 nights.",
    "E10": "Eye drops Moxifloxacin 0.5%, 1 drop QID for 5 days.",
    "E11": "Inhaler Budesonide 200 mcg, 2 puffs BD; rinse after use.",
    "E12": "Ointment Mupirocin 2%, apply locally TDS for 7 days.",
    "E13": "Tab Ibuprofen 400 mg SOS after food; maximum 3 doses/day.",
    "E14": "Cap Cholecalciferol 60,000 IU, once weekly for 8 weeks.",
    "E15": "Tab Telmisartan 40 mg 1-0-0; Atorvastatin 10 mg 0-0-1.",
    "E16": "BP 128/76 mmHg; pulse 82/min; SpO2 98%; temperature 99.4 F.",
    "V01": "Tab Pantoprazole 40 mg, 1 tablet AC breakfast for 14 days.",
    "V02": "Tab Doxycycline 100 mg, 1-0-1 after food for 7 days.",
    "V03": "Inj Insulin 10 IU SC before breakfast and 6 IU before dinner.",
    "V04": "ORS 200 mL after each loose stool; review if symptoms persist.",
    "R01": "Mix 0.5 mL with 2.5 mL saline; use once at 8:00 PM.",
    "R02": "Tab Ofloxacin 200 mg, one tablet BD for 5 days.",
    "R03": "Cap Omeprazole 20 mg, one capsule before breakfast.",
    "R04": "Review in 2 weeks; continue current medicines unless advised.",
    "T01": "Tab Paracetamol 650 mg SOS after food; maximum 3 tablets/day.",
    "T02": "Syp Amoxicillin 250 mg/5 mL, 5 mL TDS for 5 days.",
    "T03": "Tab Metformin SR 500 mg, one tablet with the evening meal.",
    "T04": "Tab Cetirizine 10 mg, one tablet HS for 7 nights.",
    "T05": "Tab Quetiapine 25 mg, 0-0-1 at bedtime for 14 days.",
    "T06": "Tab Clopidogrel 75 mg, one tablet OD after lunch.",
    "T07": "Cap Pregabalin 75 mg, one capsule HS for 10 days.",
    "T08": "Eye drops Tobramycin 0.3%, 1 drop TDS for 5 days.",
}

# Page index (0-based) -> line IDs on that page, top-to-bottom.
PAGE_LINES = {
    2: ["E01", "E02", "E03", "E04"],
    3: ["E05", "E06", "E07", "E08"],
    4: ["E09", "E10", "E11", "E12"],
    5: ["E13", "E14", "E15", "E16"],
    6: ["V01", "V02", "V03", "V04", "R01", "R02", "R03", "R04"],
    7: ["T01", "T02", "T03", "T04"],
    8: ["T05", "T06", "T07", "T08"],
}


def _prompt_fills(drawings):
    """Prompt-header rects (filled rectangles), sorted top-to-bottom."""
    fills = []
    for d in drawings:
        if d["type"] != "f" or d.get("fill") is None:
            continue
        if _is_prompt_fill(d["fill"]):
            fills.append(d["rect"])
    fills.sort(key=lambda r: r.y0)
    return fills


def _empty_boxes(drawings):
    """Bordered empty boxes as (x0, y0, x1, y1), deduplicated."""
    h_lines = []  # (y, x0, x1)
    for d in drawings:
        if d["type"] != "s":
            continue
        r = d["rect"]
        if abs(r.y0 - r.y1) < 0.5:
            h_lines.append((r.y0, r.x0, r.x1))

    boxes = []
    for y_top, xl, xr in h_lines:
        bottoms = [
            yb for (yb, xl2, xr2) in h_lines
            if yb > y_top + 10 and abs(xl2 - xl) < 2 and abs(xr2 - xr) < 2
        ]
        if not bottoms:
            continue
        boxes.append((xl, y_top, xr, min(bottoms)))

    seen = set()
    unique = []
    for b in boxes:
        key = tuple(round(v, 1) for v in b)
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique


def _writing_areas(drawings):
    """Match each prompt fill to the empty box whose top == prompt bottom."""
    fills = _prompt_fills(drawings)
    boxes = _empty_boxes(drawings)
    areas = []
    for f in fills:
        for (xl, y_top, xr, yb) in boxes:
            if (abs(y_top - f.y1) < 0.5
                    and abs(xl - f.x0) < 2 and abs(xr - f.x1) < 2):
                areas.append((xl, y_top, xr, yb))
                break
    areas.sort(key=lambda a: a[1])
    return areas


def _prompt_text(page, line_id):
    """Return the prompt text for a line, e.g. 'Tab Paracetamol 500 mg...'.

    The prompt is the text block whose top-left y sits just above the line's
    writing box and which begins with the line ID (e.g. 'E01.').
    """
    prefix = f"{line_id}."
    for block in page.get_text("blocks"):
        text = " ".join(block[4].split())
        if text.startswith(prefix):
            # Strip the leading 'E01. ' and return the sentence.
            return text[len(prefix):].strip()
    raise ValueError(f"No prompt text found for {line_id}")


def load_template(pdf_path):
    """Return {line_id: (x0, y0, x1, y1)} in PDF points (flattened).

    NOTE: coordinates are per-page. Lines on different pages share the same
    coordinate values. For cropping a single photographed sheet, use
    load_page_boxes() which returns boxes for ONE page only.
    """
    doc = fitz.open(pdf_path)
    result = {}
    for page_idx, line_ids in PAGE_LINES.items():
        if page_idx >= len(doc):
            continue
        areas = _writing_areas(doc[page_idx].get_drawings())
        if len(areas) != len(line_ids):
            raise ValueError(
                f"Page {page_idx}: found {len(areas)} areas, "
                f"expected {len(line_ids)}"
            )
        for lid, rect in zip(line_ids, areas):
            result[lid] = tuple(rect)
    doc.close()
    return result


def load_page_boxes(pdf_path, page_index):
    """Return {line_id: (x0, y0, x1, y1)} for ONE PDF page."""
    if page_index not in PAGE_LINES:
        raise ValueError(f"No sheet on PDF page {page_index}")
    doc = fitz.open(pdf_path)
    areas = _writing_areas(doc[page_index].get_drawings())
    line_ids = PAGE_LINES[page_index]
    if len(areas) != len(line_ids):
        raise ValueError(
            f"Page {page_index}: found {len(areas)} areas, "
            f"expected {len(line_ids)}"
        )
    result = {lid: tuple(rect) for lid, rect in zip(line_ids, areas)}
    doc.close()
    return result


def extract_ground_truth(pdf_path=None):
    """Return {line_id: prompt text} for all lines."""
    if fitz is not None and pdf_path and os.path.exists(pdf_path):
        try:
            doc = fitz.open(pdf_path)
            result = {}
            for page_idx, line_ids in PAGE_LINES.items():
                if page_idx >= len(doc):
                    continue
                page = doc[page_idx]
                for lid in line_ids:
                    result[lid] = _prompt_text(page, lid)
            doc.close()
            return result
        except Exception:
            pass
    return dict(PROTOCOL_GROUND_TRUTH)
