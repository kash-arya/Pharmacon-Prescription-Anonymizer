"""
Photo crop: turn a photo of a printed sheet into per-line crops + labels.

One deep function for the backend and CLI:

    process_photo(photo, pdf_path) -> {line_id: {"image": ndarray, "label": str}}

It loads the template and ground truth from the PDF automatically, finds the
document corners (Canny -> largest 4-sided contour), warps to a top-down A4
view, and crops each known writing box.
"""

import cv2
import numpy as np

from .template import load_page_boxes, extract_ground_truth

A4_WIDTH_PT = 595.28
A4_HEIGHT_PT = 841.89
ZOOM = 2.0  # pixels per PDF point when rasterizing

# Cache template + ground truth per PDF path (avoids reloading on every call).
_template_cache = {}


def _get_labels(pdf_path):
    if pdf_path not in _template_cache:
        _template_cache[pdf_path] = extract_ground_truth(pdf_path)
    return _template_cache[pdf_path]


def _order_points(pts):
    """Order 4 points as TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def find_document_corners(image):
    """Return 4 ordered corners of the sheet, or image bounds if already cropped."""
    h, w = image.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(contour) > 0.60 * img_area:
            return _order_points(approx)

    # Fallback: Photo is already cropped to the page
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")


def _decode_image(image):
    """Accept bytes, a file path, or an ndarray; return a BGR ndarray."""
    if isinstance(image, (bytes, bytearray)):
        buf = np.frombuffer(image, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if isinstance(image, str):
        return cv2.imread(image)
    return image


def _split_of(line_id):
    """Return the split letter (E/V/R/T) for a line ID."""
    return line_id[0]


def process_photo(photo, pdf_path, page_index, writer_id=None, session_id=None):
    """
    Crop ONE photographed sheet into per-line images with labels.

    Args:
        photo: bytes, file path, or BGR ndarray of the sheet photo.
        pdf_path: path to the protocol PDF (template + ground truth source).
        page_index: which PDF page this photo shows (0-based). Determines the
            line IDs on this sheet (e.g. page 2 -> E01-E04).
        writer_id: optional pseudonymous writer ID (manifest requirement).
        session_id: optional session ID (manifest requirement).

    Returns:
        {line_id: {"image": BGR ndarray, "label": str, "split": "E|V|R|T",
                   "writer_id": ..., "session_id": ...}}
    """
    image = _decode_image(photo)
    if image is None:
        raise ValueError("Could not decode the photo.")

    corners = find_document_corners(image)
    if corners is None:
        raise ValueError("Could not find a four-sided document contour.")

    boxes = load_page_boxes(pdf_path, page_index)
    labels = _get_labels(pdf_path)

    target_w = int(A4_WIDTH_PT * ZOOM)
    target_h = int(A4_HEIGHT_PT * ZOOM)
    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1],
    ], dtype="float32")
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, matrix, (target_w, target_h))

    wh, ww = warped.shape[:2]
    lines = {}
    for lid, (x0, y0, x1, y1) in boxes.items():
        px0, py0 = round(x0 * ZOOM), round(y0 * ZOOM)
        px1, py1 = round(x1 * ZOOM), round(y1 * ZOOM)
        if not (0 <= py0 < py1 <= wh and 0 <= px0 < px1 <= ww):
            raise ValueError(
                f"Box {lid} out of bounds after warp "
                f"(crop {px0},{py0},{px1},{py1}; image {ww}x{wh})"
            )
        crop = warped[py0:py1, px0:px1]
        lines[lid] = {
            "image": crop,
            "label": labels.get(lid, ""),
            "split": _split_of(lid),
            "writer_id": writer_id,
            "session_id": session_id,
        }
    return lines
