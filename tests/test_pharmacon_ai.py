"""Tests for the pharmacon_ai package."""

import os
import sys

import cv2
import numpy as np
import pymupdf as fitz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pharmacon_ai import (process_photo, load_template, load_page_boxes,
                          extract_ground_truth, cer, wer, evaluate_condition)

PDF = os.path.join(os.path.dirname(__file__), "..",
                   "Pharmacon_Writer_Calibration_Protocol.pdf")


def test_template_has_32_boxes():
    t = load_template(PDF)
    assert len(t) == 32, f"expected 32, got {len(t)}"
    for lid in ["E01", "E16", "V04", "R04", "T08"]:
        assert lid in t, f"missing {lid}"


def test_page_boxes_are_scoped_to_one_page():
    boxes = load_page_boxes(PDF, 2)
    assert set(boxes.keys()) == {"E01", "E02", "E03", "E04"}
    boxes = load_page_boxes(PDF, 6)
    assert set(boxes.keys()) == {"V01", "V02", "V03", "V04",
                                 "R01", "R02", "R03", "R04"}


def test_ground_truth_has_32_labels():
    gt = extract_ground_truth(PDF)
    assert len(gt) == 32
    assert gt["E01"] == "The quick brown fox jumps over the lazy dog. 0123456789."
    assert gt["E16"] == "BP 128/76 mmHg; pulse 82/min; SpO2 98%; temperature 99.4 F."


def test_cer_wer_metrics():
    # Exact matches
    assert cer("Tab Paracetamol 500 mg", "Tab Paracetamol 500 mg") == 0.0
    assert wer("Tab Paracetamol 500 mg", "Tab Paracetamol 500 mg") == 0.0

    # Single character substitution (1 / 22)
    ref = "Tab Paracetamol 500 mg"
    hyp = "Tab Paracetomol 500 mg"
    assert abs(cer(ref, hyp) - 1.0 / len(ref)) < 1e-6
    # 1 word changed out of 4
    assert abs(wer(ref, hyp) - 1.0 / 4.0) < 1e-6

    # Macro evaluation
    preds = {"T01": "Tab Metformin 500 mg", "T02": "Wrong"}
    labels = {"T01": "Tab Metformin 500 mg", "T02": "Right"}
    eval_res = evaluate_condition(preds, labels)
    assert eval_res["T01"]["cer"] == 0.0
    assert eval_res["macro"]["n"] == 2
    assert eval_res["macro"]["cer"] > 0.0


def _warped_photo(page_idx):
    doc = fitz.open(PDF)
    page = doc[page_idx]
    zoom = 2.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()

    h, w = img.shape[:2]
    pad = 120
    src = np.array([[60, 60], [w - 40, 90], [w - 80, h - 50], [70, h - 40]],
                   dtype="float32")
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(dst, src)
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 30, dtype=np.uint8)
    warped = cv2.warpPerspective(img, M, (w + 2 * pad, h + 2 * pad))
    mask = cv2.warpPerspective(np.ones_like(img) * 255, M,
                               (w + 2 * pad, h + 2 * pad))
    canvas[mask > 0] = warped[mask > 0]
    return canvas


def test_process_photo_returns_only_that_page():
    photo = _warped_photo(2)  # page 2 = E01-E04
    lines = process_photo(photo, PDF, page_index=2)
    assert set(lines.keys()) == {"E01", "E02", "E03", "E04"}
    assert lines["E01"]["label"] == (
        "The quick brown fox jumps over the lazy dog. 0123456789.")
    assert lines["E01"]["image"].size > 0


if __name__ == "__main__":
    test_template_has_32_boxes()
    test_page_boxes_are_scoped_to_one_page()
    test_ground_truth_has_32_labels()
    test_cer_wer_metrics()
    test_process_photo_returns_only_that_page()
    print("All pharmacon_ai tests passed.")
