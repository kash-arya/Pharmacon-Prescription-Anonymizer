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

# Cache template, ground truth, and precomputed PDF keypoints
_template_cache = {}
_pdf_sift_cache = {}

DPI_REF = 150
SCALE_REF = DPI_REF / 72.0  # PDF points to pixels at reference DPI


def _get_labels(pdf_path):
    if pdf_path not in _template_cache:
        _template_cache[pdf_path] = extract_ground_truth(pdf_path)
    return _template_cache[pdf_path]


def _get_pdf_sift_ref(pdf_path, page_index):
    """Return (gray_img, keypoints, descriptors) for the reference PDF page."""
    import pymupdf as fitz
    cache_key = (pdf_path, page_index)
    if cache_key not in _pdf_sift_cache:
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        pix = page.get_pixmap(dpi=DPI_REF)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            (pix.height, pix.width, pix.n)
        )
        if pix.n == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        doc.close()

        sift = cv2.SIFT_create(nfeatures=2500)
        kp, des = sift.detectAndCompute(gray, None)
        _pdf_sift_cache[cache_key] = (kp, des)
    return _pdf_sift_cache[cache_key]


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


def _crop_via_sift(image, pdf_path, page_index, boxes, labels, writer_id, session_id):
    """Register photo against the reference PDF page via SIFT homography."""
    kp_pdf, des_pdf = _get_pdf_sift_ref(pdf_path, page_index)
    if des_pdf is None or len(kp_pdf) < 10:
        return None

    ph_h, ph_w = image.shape[:2]
    # Resize photo for fast feature detection if resolution is high
    max_dim = 1600.0
    scale = max_dim / max(ph_h, ph_w) if max(ph_h, ph_w) > max_dim else 1.0
    if scale < 1.0:
        small_photo = cv2.resize(image, (int(ph_w * scale), int(ph_h * scale)))
        small_gray = cv2.cvtColor(small_photo, cv2.COLOR_BGR2GRAY)
    else:
        small_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=2500)
    kp_photo, des_photo = sift.detectAndCompute(small_gray, None)
    if des_photo is None or len(kp_photo) < 10:
        return None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des_pdf, des_photo, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 15:
        return None

    src_pts = np.float32([kp_pdf[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # Rescale photo points back to original image space
    dst_pts = np.float32([np.array(kp_photo[m.trainIdx].pt) / scale for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None

    inliers = int(np.sum(mask))
    if inliers < 15:
        return None

    # Homography found! Extract each writing box directly
    lines = {}
    for lid, (x0, y0, x1, y1) in boxes.items():
        # Slight horizontal breathing room (+2 pt left/right) so border-adjacent ink isn't clipped
        px0 = (x0 - 2.0) * SCALE_REF
        py0 = y0 * SCALE_REF
        px1 = (x1 + 3.0) * SCALE_REF
        py1 = y1 * SCALE_REF

        box_pts_pdf = np.float32([
            [px0, py0],
            [px1, py0],
            [px1, py1],
            [px0, py1]
        ]).reshape(-1, 1, 2)

        box_pts_photo = cv2.perspectiveTransform(box_pts_pdf, H).reshape(4, 2)

        target_w = int((x1 - x0 + 5.0) * ZOOM)
        target_h = int((y1 - y0) * ZOOM)
        rect_dst = np.float32([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ])

        M = cv2.getPerspectiveTransform(box_pts_photo, rect_dst)
        crop = cv2.warpPerspective(image, M, (target_w, target_h))

        lines[lid] = {
            "image": crop,
            "label": labels.get(lid, ""),
            "split": _split_of(lid),
            "writer_id": writer_id,
            "session_id": session_id,
        }
    return lines


def process_photo(photo, pdf_path, page_index, writer_id=None, session_id=None):
    """
    Crop ONE photographed sheet into per-line images with labels.

    Uses SIFT feature-homography to align precisely to the printed template,
    immune to camera angle, framing, or margins. Falls back to contour warping
    if SIFT has insufficient features.

    Args:
        photo: bytes, file path, or BGR ndarray of the sheet photo.
        pdf_path: path to the protocol PDF (template + ground truth source).
        page_index: which PDF page this photo shows (0-based).
        writer_id: optional pseudonymous writer ID.
        session_id: optional session ID.

    Returns:
        {line_id: {"image": BGR ndarray, "label": str, "split": "E|V|R|T",
                   "writer_id": ..., "session_id": ...}}
    """
    image = _decode_image(photo)
    if image is None:
        raise ValueError("Could not decode the photo.")

    boxes = load_page_boxes(pdf_path, page_index)
    labels = _get_labels(pdf_path)

    # 1. Primary method: SIFT Homography against reference PDF page
    try:
        sift_crops = _crop_via_sift(image, pdf_path, page_index, boxes, labels,
                                     writer_id, session_id)
        if sift_crops is not None and len(sift_crops) == len(boxes):
            return sift_crops
    except Exception:
        pass

    # 2. Fallback method: 4-corner document detection and A4 perspective warp
    corners = find_document_corners(image)
    if corners is None:
        raise ValueError("Could not find a four-sided document contour.")

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
