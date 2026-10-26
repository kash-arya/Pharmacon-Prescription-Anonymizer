# Pharmacon Prescription Anonymizer

Local de-identification engine for medical prescriptions, clinical intake records, and lab slips. Runs offline on standard CPU hardware to comply with HIPAA Safe Harbor (45 CFR § 164.514(b)), GDPR Article 9, and the Indian DPDP Act 2023 without sending patient records to third-party cloud APIs.

---

## Modular System Architecture (`src/`)

The core de-identification pipeline is modularized into 11 specialized components:

| Module | Core Responsibility |
| :--- | :--- |
| [`src/raster_checker.py`](src/raster_checker.py) | Structural classification of documents (`VECTOR_DIGITAL` vs `SCANNED_RASTER`). |
| [`src/vector_redactor.py`](src/vector_redactor.py) | Byte-level stream redactor for digital PDFs (underlying bytecode sanitization in <50ms). |
| [`src/image_redactor.py`](src/image_redactor.py) | Multi-pass vision redactor for scanned PDFs and smartphone camera photos. |
| [`src/ocr_preprocess.py`](src/ocr_preprocess.py) | Line clustering, CLAHE adaptive contrast, and unsharp masking for phone clicks. |
| [`src/form_parser.py`](src/form_parser.py) | 1D/2D table anchor parser and spatial ink-density detector for handwritten signatures and fill zones. |
| [`src/barcode_detector.py`](src/barcode_detector.py) | OpenCV morphological & Sobel scanner for 1D barcodes and QR codes. |
| [`src/ner_engine.py`](src/ner_engine.py) | Contextual spaCy NER (`en_core_web_sm`, lazy-loaded `xx_ent_wiki_sm`) with clinical whitelist gating. |
| [`src/jurisdiction.py`](src/jurisdiction.py) | Automatic healthcare statutory framework detector (`US`, `UK`, `AU`, `EU`, `IN`, `GLOBAL`). |
| [`src/forensics.py`](src/forensics.py) | Adversarial reconstruction simulator & strict Residual and Partial-Masking Audit confidence scoring. |
| [`src/vlm_grounder.py`](src/vlm_grounder.py) | Zero-keyword spatial ink grounder and clinical Rx baseline demarcation guard. |
| [`src/entities.py`](src/entities.py) | Pydantic data schemas, bounding boxes, and comprehensive clinical medication whitelists. |

---

## Visual Audit Artifacts (`outputs/`)

Each specimen processed across all 4 dataset suites (`samples/`, `real_phone_prescriptions/`, `raw_public_slips/`, `unseen_holdout/`) produces an auditable visual pair:
- **`outputs/<subfolder>/<stem>_SIDE_BY_SIDE.jpg`**: High-contrast, side-by-side composite image displaying the original prescription on the left and the masked output on the right, topped with a real-time audit triage banner (`AUTO_APPROVED` vs `NEEDS_HUMAN_REVIEW`).
- **`outputs/SUMMARY_REPORT.md`**: Complete markdown audit report detailing jurisdiction detection, format classification, redacted entity counts, confidence scores, and triage review reasons.

---

## Production REST API Specification

A high-performance FastAPI microservice engineered for low-memory container environments (Render 512MB RAM + Vercel frontend):

| Method & Route | Description |
| :--- | :--- |
| `GET /health` | Health and readiness probe returning engine status and loaded models. |
| `POST /anonymize/preview` | Returns unified JSON payload with Left PNG (`original_image_base64`), Right PNG (`redacted_image_base64`), Side-by-Side JPEG (`side_by_side_image_base64`), and direct download binary (`download_file_base64`) in original uploaded format. |
| `POST /anonymize/file` | De-identifies and streams the sanitized document in its exact original format (`.pdf`, `.jpg`, `.png`, `.webp`) with attachment download headers. |
| `POST /api/v1/anonymize` | Multipart endpoint supporting `return_type=file` (binary stream) or `return_type=json` (metadata + base64). |
| `POST /api/v1/anonymize/batch` | High-throughput batch redaction accepting a `.zip` archive and streaming back a sanitized `.zip` with an audit summary. |

---

## Quick Start

### 1. Setup Environment
```bash
source .venv/bin/activate
# or install dependencies:
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm
```

### 2. Anonymize Prescriptions (CLI)
Works on digital vector PDFs, scanned PDFs, raw smartphone photos (`.jpg`, `.png`), or entire folders:
```bash
# Single file mode
python anonymize.py --input data/samples/sample_digital_vector_prescription.pdf

# Parallel multi-core batch mode
python anonymize.py --input data/samples/ --output /tmp/sanitized_prescriptions/ --workers 8
```

### 3. Generate Visual Audit Comparisons
Batch-processes all specimens across dataset folders and generates side-by-side comparisons in `outputs/`:
```bash
python scripts/export_all_masked_results.py
```

### 4. Run API Server
```bash
python -m api.server
```
- Interactive Web App: `http://localhost:8000/`
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`
- Node.js & React Integration: See [`docs/NODEJS_INTEGRATION_GUIDE.md`](docs/NODEJS_INTEGRATION_GUIDE.md) and [`scripts/node_client_example.js`](scripts/node_client_example.js).

### 5. Run Evaluation Benchmark
```bash
python benchmarks/benchmark.py
```

### 6. Run Test Suite
```bash
python -m unittest tests/test_prescription_anonymizer.py
```

---

## Benchmark Results (Authentic Clinical Documents)

| Document Category | Document Specimen | Format Type | PII Fields | Leaks | Clinical Preservation | Latency | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Digital Vector EMR** | Apollo Hospitals EMR Slip | Vector PDF | 9 | **0** | **100.0%** | $362\text{ ms}$ | **PASS** |
| **Scanned Intake Slip** | Doctor Intake Slip #1 | Scanned PDF | 7 | **0** | **100.0%** | $3.40\text{ s}$ | **PASS** |
| **Scanned Intake Slip** | Doctor Intake Slip #2 | Scanned PDF | 7 | **0** | **100.0%** | $8.70\text{ s}$ | **PASS** |
| **Multi-page Intake** | Scanned Multi-page Record | Scanned PDF | 11 | **0** | **100.0%** | $2.97\text{ s}$ | **PASS** |
| **Scanned Image** | Standalone Doctor Slip PNG | Scanned Image | 6 | **0** | **100.0%** | $1.71\text{ s}$ | **PASS** |
| **Smartphone Photo** | Akash Imaging Ultrasound (S23) | JPEG Camera | 13 | **0** | **100.0%** | $8.98\text{ s}$ | **PASS** |
| **Smartphone Photo** | SIMSS Hospital OPD Click (S23) | JPEG Camera | 11 | **0** | **100.0%** | $6.61\text{ s}$ | **PASS** |
| **Smartphone Photo** | Akash Imaging Report 2 (S23) | JPEG Camera | 12 | **0** | **100.0%** | $9.64\text{ s}$ | **PASS** |
| **Aggregate Summary** | **Authentic Document Cohort** | **Hybrid Pipeline** | **100.0% Recall** | **0 Leaks** | **100.0% Preserved** | **5.30 s avg** | **8 / 8 (100% PASS)** |

**Unit & Integration Test Suite:** `23 / 23 PASS (100.0%)`

---

## Technical Defense & Architecture
For architectural proofs, threat models, and oral defense talking points, see [`docs/PRESCRIPTION_ANONYMIZER_RESEARCH_DEFENSE.md`](docs/PRESCRIPTION_ANONYMIZER_RESEARCH_DEFENSE.md).
