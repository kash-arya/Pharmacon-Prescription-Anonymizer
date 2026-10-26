# Pharmacon AI: Medical Prescription Anonymizer

Local, CPU-first de-identification system for medical prescriptions, clinical intake records, and lab slips. Complies with HIPAA Safe Harbor (45 CFR § 164.514(b)), GDPR Article 9, and the Indian DPDP Act 2023 without transmitting sensitive patient records to third-party cloud APIs.

---

## Repository Structure

- **[`prescription_anonymizer/`](prescription_anonymizer/)** — The production anonymization engine, REST microservice, CLI tools, benchmarks, and test suites.
  - `src/` — Modular 11-file redaction core:
    - `raster_checker.py` — Structural document classification (`VECTOR_DIGITAL` vs `SCANNED_RASTER`).
    - `vector_redactor.py` — Fast byte-level vector stream redactor and metadata sanitizer (<50ms/page).
    - `image_redactor.py` — Multi-pass OCR and vision redaction pipeline for scanned PDFs and photos.
    - `ocr_preprocess.py` — Line reconstruction, CLAHE adaptive contrast, and unsharp masking.
    - `form_parser.py` — 1D/2D table anchor parser and spatial ink-density detector for handwritten signatures and fill zones.
    - `barcode_detector.py` — OpenCV morphological & Sobel scanner for 1D barcodes and QR codes.
    - `ner_engine.py` — Contextual spaCy NER (`en_core_web_sm`, lazy-loaded `xx_ent_wiki_sm`).
    - `jurisdiction.py` — Automatic healthcare statutory framework detector (`US`, `UK`, `AU`, `EU`, `IN`, `GLOBAL`).
    - `forensics.py` — Adversarial attack simulator and strict Residual and Partial-Masking confidence scoring.
    - `vlm_grounder.py` — Zero-keyword spatial ink grounder and clinical Rx baseline demarcation guard.
    - `entities.py` — Pydantic schemas, bounding boxes, and comprehensive clinical medication whitelists.
  - `data/` — Authentic clinical specimens across digital vector PDFs, scanned documents, and raw smartphone camera photos.
  - `outputs/` — High-contrast side-by-side visual audit images (`<name>_SIDE_BY_SIDE.jpg`) and automated triage summary report (`SUMMARY_REPORT.md`).
  - `benchmarks/` — Automated evaluation suite measuring PII recall, zero-leak guarantees, and clinical entity preservation.
  - `tests/` — End-to-end unit and integration test suite (23/23 tests passing).
- **[`pharmacon_v1/`](pharmacon_v1/)** — Historical phase 1 handwriting recognition and writer adaptation exploration artifacts.

---

## Quick Start

```bash
cd prescription_anonymizer

# Set up environment
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm

# Redact a single prescription
python anonymize.py --input data/samples/sample_digital_vector_prescription.pdf

# Run full evaluation benchmark (8/8 PASS, 100% PII Recall, 0 Leaks)
python benchmarks/benchmark.py

# Run unit and integration tests (23/23 PASS)
python -m unittest tests/test_prescription_anonymizer.py
```

For complete architectural details, REST API documentation, and technical defense notes, see [`prescription_anonymizer/README.md`](prescription_anonymizer/README.md).
