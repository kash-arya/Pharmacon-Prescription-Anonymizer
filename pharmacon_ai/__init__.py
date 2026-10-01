"""
Pharmacon AI pipeline — writer-adaptive prescription OCR.

This package exposes the AI pipeline to the team's backend as a small set of
deep functions. The backend calls one function, gets crops + labels back.

Public interface (everything a teammate needs):
    from pharmacon_ai import process_photo
    lines = process_photo(photo_bytes, pdf_path, page_index)
    # -> {"E01": {"image": <ndarray>, "label": "...", "split": "E", ...}}
"""

from .template import load_template, load_page_boxes, extract_ground_truth
from .crop import process_photo
from .adapt import load_trOCR, make_writer_adapter, train_adapter, predict
from .evaluate import cer, wer, run_predictions, evaluate_condition, print_comparison

__all__ = ["process_photo", "load_template", "load_page_boxes",
           "extract_ground_truth", "load_trOCR", "make_writer_adapter",
           "train_adapter", "predict",
           "cer", "wer", "run_predictions", "evaluate_condition",
           "print_comparison"]
