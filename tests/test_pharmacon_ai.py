"""Unit tests for the pharmacon_ai package."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pharmacon_ai import (
    cer, wer, evaluate_condition,
    align_characters, WriterConfusionProfile
)

try:
    import cv2
    import numpy as np
    import pymupdf as fitz
    from pharmacon_ai import load_template, load_page_boxes, extract_ground_truth, process_photo
    HAS_CV_DEPS = True
except ImportError:
    HAS_CV_DEPS = False

PDF = os.path.join(os.path.dirname(__file__), "..",
                   "Pharmacon_Writer_Calibration_Protocol.pdf")


class TestPharmaconAI(unittest.TestCase):

    def test_cer_wer_metrics(self):
        # Exact matches
        self.assertEqual(cer("Tab Paracetamol 500 mg", "Tab Paracetamol 500 mg"), 0.0)
        self.assertEqual(wer("Tab Paracetamol 500 mg", "Tab Paracetamol 500 mg"), 0.0)

        # Single character substitution (1 / 22)
        ref = "Tab Paracetamol 500 mg"
        hyp = "Tab Paracetomol 500 mg"
        self.assertAlmostEqual(cer(ref, hyp), 1.0 / len(ref), places=5)
        # 1 word changed out of 4
        self.assertAlmostEqual(wer(ref, hyp), 1.0 / 4.0, places=5)

        # Macro evaluation
        preds = {"T01": "Tab Metformin 500 mg", "T02": "Wrong"}
        labels = {"T01": "Tab Metformin 500 mg", "T02": "Right"}
        eval_res = evaluate_condition(preds, labels)
        self.assertEqual(eval_res["T01"]["cer"], 0.0)
        self.assertEqual(eval_res["macro"]["n"], 2)
        self.assertGreater(eval_res["macro"]["cer"], 0.0)

    @unittest.skipUnless(HAS_CV_DEPS and os.path.exists(PDF), "Requires cv2/pymupdf and protocol PDF")
    def test_template_has_32_boxes(self):
        t = load_template(PDF)
        self.assertEqual(len(t), 32)
        for lid in ["E01", "E16", "V04", "R04", "T08"]:
            self.assertIn(lid, t)

    @unittest.skipUnless(HAS_CV_DEPS and os.path.exists(PDF), "Requires cv2/pymupdf and protocol PDF")
    def test_page_boxes_are_scoped_to_one_page(self):
        boxes = load_page_boxes(PDF, 2)
        self.assertEqual(set(boxes.keys()), {"E01", "E02", "E03", "E04"})
        boxes = load_page_boxes(PDF, 6)
        self.assertEqual(set(boxes.keys()), {"V01", "V02", "V03", "V04",
                                             "R01", "R02", "R03", "R04"})

    @unittest.skipUnless(HAS_CV_DEPS and os.path.exists(PDF), "Requires cv2/pymupdf and protocol PDF")
    def test_ground_truth_has_32_labels(self):
        gt = extract_ground_truth(PDF)
        self.assertEqual(len(gt), 32)
        self.assertEqual(gt["E01"], "The quick brown fox jumps over the lazy dog. 0123456789.")
        self.assertEqual(gt["E16"], "BP 128/76 mmHg; pulse 82/min; SpO2 98%; temperature 99.4 F.")


if __name__ == "__main__":
    unittest.main()
