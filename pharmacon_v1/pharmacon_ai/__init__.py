"""
Pharmacon AI pipeline — writer-adaptive prescription OCR.
"""

__all__ = [
    # Metrics & Evaluator
    "cer", "wer", "run_predictions", "evaluate_condition", "print_comparison",
    "print_three_condition_comparison",
    # WSCMC Active Refinement
    "align_characters", "WriterConfusionProfile",
    # Formulary-Constrained Beam Search Decoding
    "FormularyTrie", "build_clinical_lexicon", "build_formulary_trie", "make_prefix_allowed_tokens_fn",
    # Clinical Entity Extractor & Formulary Matcher
    "FormularyMatcher", "extract_prescription_entities", "levenshtein_similarity",
    "jaro_winkler_similarity", "consonant_skeleton",
    # Clinical Inventory & Pharmacist Dispensation Engine
    "init_db", "get_inventory_items", "get_patient_medications", "dispense_prescription",
    # Template & Fixed-form Crop Engine
    "load_template", "load_page_boxes", "extract_ground_truth", "process_photo",
    # TrOCR LoRA Model & Training
    "load_trOCR", "make_writer_adapter", "train_adapter", "predict",
]

# 1. Metrics & Evaluator (Zero heavy deps)
from .evaluate import (cer, wer, run_predictions, evaluate_condition,
                       print_comparison, print_three_condition_comparison)

# 2. WSCMC Active Refinement (Zero heavy deps)
from .wscmc import align_characters, WriterConfusionProfile

# 3. Formulary-Constrained Beam Search Decoding (Zero heavy deps)
from .constrained_decoding import (FormularyTrie, build_clinical_lexicon,
                                   build_formulary_trie, make_prefix_allowed_tokens_fn)

# 4. Formulary Matcher & Entity Parser (Zero heavy deps)
from .formulary import (FormularyMatcher, extract_prescription_entities,
                        levenshtein_similarity, jaro_winkler_similarity,
                        consonant_skeleton)

# 5. Clinical Inventory & Dispensation Engine (Zero heavy deps)
from .inventory import (init_db, get_inventory_items, get_patient_medications,
                        dispense_prescription)

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
