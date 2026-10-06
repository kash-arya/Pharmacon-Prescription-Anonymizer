"""
Pharmacon AI pipeline — writer-adaptive prescription OCR.
"""

__all__ = [
    # Metrics & Evaluator
    "cer", "wer", "run_predictions", "evaluate_condition", "print_comparison",
    # Clinical Entity Extractor & Formulary Matcher
    "FormularyMatcher", "extract_prescription_entities", "levenshtein_similarity",
    "jaro_winkler_similarity", "consonant_skeleton",
    # Template & Fixed-form Crop Engine
    "load_template", "load_page_boxes", "extract_ground_truth", "process_photo",
    # TrOCR LoRA Model & Training
    "load_trOCR", "make_writer_adapter", "train_adapter", "predict",
]

# 1. Metrics & Evaluator (Zero heavy deps)
from .evaluate import cer, wer, run_predictions, evaluate_condition, print_comparison

# 2. Formulary Matcher & Entity Parser (Zero heavy deps)
from .formulary import (FormularyMatcher, extract_prescription_entities,
                        levenshtein_similarity, jaro_winkler_similarity,
                        consonant_skeleton)

# 3. Template & Cropping (Requires pymupdf / cv2)
try:
    from .template import load_template, load_page_boxes, extract_ground_truth
    from .crop import process_photo
except ImportError:
    pass

# 4. Neural Network Adaptations (Requires torch / transformers / peft)
try:
    from .adapt import load_trOCR, make_writer_adapter, train_adapter, predict
except ImportError:
    pass
