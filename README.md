# Pharmacon AI: Medical Prescription Anonymizer

Offline, CPU-first de-identification system for medical prescriptions, clinical intake records, and lab slips. Complies with HIPAA Safe Harbor, GDPR Article 9, and the Indian DPDP Act 2023 without sending patient health information to cloud APIs.

---

## Repository Structure

- **[`prescription_anonymizer/`](prescription_anonymizer/)** — The production anonymization engine, REST microservice, CLI tools, benchmarks, and test suites.
  - `src/` — Modular redaction core (vector redactor, image redactor, form parser, barcode detector, OCR preprocessor, NER engine).
  - `data/` — Authentic clinical specimens across digital vector PDFs, scanned documents, and raw smartphone camera photos.
  - `outputs/` — High-contrast side-by-side visual audit images (`<name>_SIDE_BY_SIDE.jpg`) and automated triage summary report.
  - `benchmarks/` — Automated evaluation suite measuring PII recall, zero-leak guarantees, and clinical entity preservation.
  - `tests/` — End-to-end unit and integration test suite.
  - `viewer/` — Interactive web viewer for side-by-side verification and inspection.
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
