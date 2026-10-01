"""
Template: extract writing-box geometry and ground-truth text from the protocol PDF.

The protocol PDF is a fixed printed form. This module extracts, once:
  - the writing-box rectangles (per line ID), and
  - the prompt text above each box (the ground-truth transcription).

Everything here is derived from the PDF's own vector/text structure, so it is
reproducible and needs no manual labeling.
"""

import pymupdf as fitz

PROMPT_FILL = (0.957, 0.965, 0.973)
FILL_TOLERANCE = 0.01  # accept colours within this distance of PROMPT_FILL


def _is_prompt_fill(color):
    """True if `color` is PROMPT_FILL within a small tolerance."""
    if color is None:
        return False
    return all(abs(a - b) <= FILL_TOLERANCE
               for a, b in zip(color, PROMPT_FILL))

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


def extract_ground_truth(pdf_path):
    """Return {line_id: prompt text} for all lines."""
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
