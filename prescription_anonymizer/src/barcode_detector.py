"""
Barcode & QR Code Detection Module for Medical Prescriptions.

Detects 2D QR codes and 1D barcodes across prescription specimens using OpenCV
QRCodeDetector and BarcodeDetector with sliding-window multi-pass tile scanning
and Sobel gradient validation to eliminate false positives on clinical text.
"""

from typing import List, Tuple, Optional
import os
import pymupdf
import cv2
import numpy as np


def detect_qr_and_barcodes(
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
        # Skip when RENDER_LOW_MEMORY=1 to stay within Render 512MB RAM constraints
        if os.environ.get("RENDER_LOW_MEMORY") != "1":
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
