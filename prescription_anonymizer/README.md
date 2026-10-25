# Pharmacon Prescription Anonymizer

Local de-identification engine for medical prescriptions, clinical intake records, and lab slips. Runs offline on standard CPU hardware to comply with HIPAA Safe Harbor (45 CFR § 164.514(b)), GDPR Article 9, and the Indian DPDP Act 2023 without sending patient records to third-party cloud APIs.

---

## How It Works

Prescriptions arrive in two distinct formats: clean digital PDFs exported from hospital EMR systems, or messy scans and camera clicks taken on the ward. The engine routes each document through specialized pipelines:

### 1. Digital Vector PDFs (`src/vector_redactor.py`)
Parses text spans and drawing instructions directly via PyMuPDF. It strips underlying text streams, clears embedded document metadata, and draws opaque vector redaction rectangles over identified PII in under 50ms per page.

### 2. Scanned Documents & Photos (`src/image_redactor.py`)
Applies a modular vision pipeline designed for CPU throughput:
- **Image Preprocessing (`src/ocr_preprocess.py`):** Visual line reconstruction, CLAHE contrast enhancement, and unsharp masking for faint, crumpled, or low-light camera clicks.
- **Form & Table Parsing (`src/form_parser.py`):** Extracts multi-line patient demographic blocks (name, street, postal code) across 2D table cells. Detects handwritten cursive fills and doctor signatures in blank zones using spatial ink density.
- **Barcode & QR Detection (`src/barcode_detector.py`):** Sliding-window OpenCV scanner with Sobel gradient validation to isolate patient barcodes without catching printed horizontal divider rules.
- **Named Entity Recognition (`src/ner_engine.py`):** SpaCy's `en_core_web_sm` and `xx_ent_wiki_sm` models recognize free-text patient names, locations, and facilities without requiring fixed label anchors.
- **Clinical Term Preservation:** Regex boundary clamping and morphological suffix whitelists (`-cillin`, `-prazole`, `-statin`, `-olol`, `-mab`, dosages, units) guarantee that medications and diagnoses are never blacked out.

---

## Supported Jurisdictions

The engine infers jurisdiction dynamically and sanitizes statutory identifier patterns across 5 healthcare frameworks:

- **United States:** HIPAA Safe Harbor, DEA registration numbers, 10-digit NPIs, state medical licenses.
- **United Kingdom:** NHS England FP10 slips, 10-digit NHS numbers, GMC registrations.
- **Australia:** 10-digit Medicare numbers, PBS prescriber numbers, AHPRA practitioner IDs.
- **Germany / EU:** GDPR Article 9, statutory insurance IDs (`Versicherten-Nr`), `BSNR` practice numbers, and `LANR` doctor IDs.
- **India:** DPDP Act 2023, 14-digit ABHA IDs, hospital UHID/OPD barcodes, and NMC/MCI registrations.

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

# Parallel multi-core batch mode (defaults to max available CPU cores)
python anonymize.py --input data/samples/ --output /tmp/sanitized_prescriptions/ --workers 8
```

### 3. Generate Visual Audit Comparisons
Batch-processes all specimens across dataset folders and generates high-contrast visual comparisons in `outputs/`:
```bash
python scripts/export_all_masked_results.py
```
Generated artifacts:
- `outputs/<subfolder>/<name>_SIDE_BY_SIDE.jpg` (Side-by-side original vs masked JPEG per specimen)
- `outputs/SUMMARY_REPORT.md` (Triage confidence scores and jurisdiction audit table)

### 4. Run API Server & Web Interface
```bash
# Starts FastAPI server on port 8000
python -m api.server
```
- Interactive Web App (Drag & Drop + Live Split Slider): Open `http://localhost:8000/`
- Interactive Swagger UI: Open `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`
- Node.js Integration: See [`docs/NODEJS_INTEGRATION_GUIDE.md`](docs/NODEJS_INTEGRATION_GUIDE.md) and [`scripts/node_client_example.js`](scripts/node_client_example.js).

### 5. Run Evaluation Benchmark
```bash
# Verify recall and zero PII leaks across the baseline and smartphone cohort
python benchmarks/benchmark.py
```

### 6. Interactive Inspection Viewer
```bash
# Open static viewer in browser
xdg-open viewer/viewer.html

# Or spin up local server on port 8080
python viewer/viewer.py --serve --port 8080
```

### 7. Run Test Suite
```bash
python -m unittest tests/test_prescription_anonymizer.py
```

---

## Benchmark Results (Authentic Clinical Documents)

| Document Category | Document Specimen | Format Type | PII Fields | Leaks | Clinical Preservation | Latency | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Digital Vector EMR** | Apollo Hospitals EMR Slip | Vector PDF | 9 | **0** | **100.0%** | $568\text{ ms}$ | **PASS** |
| **Scanned Intake Slip** | Doctor Intake Slip #1 | Scanned PDF | 7 | **0** | **100.0%** | $4.46\text{ s}$ | **PASS** |
| **Scanned Intake Slip** | Doctor Intake Slip #2 | Scanned PDF | 7 | **0** | **100.0%** | $4.80\text{ s}$ | **PASS** |
| **Multi-page Intake** | Scanned Multi-page Record | Scanned PDF | 11 | **0** | **100.0%** | $3.77\text{ s}$ | **PASS** |
| **Scanned Image** | Standalone Doctor Slip PNG | Scanned Image | 6 | **0** | **100.0%** | $2.54\text{ s}$ | **PASS** |
| **Smartphone Photo** | Akash Imaging Ultrasound (S23) | JPEG Camera | 13 | **0** | **100.0%** | $10.71\text{ s}$ | **PASS** |
| **Smartphone Photo** | SIMSS Hospital OPD Click (S23) | JPEG Camera | 11 | **0** | **100.0%** | $14.05\text{ s}$ | **PASS** |
| **Smartphone Photo** | Akash Imaging Report 2 (S23) | JPEG Camera | 12 | **0** | **100.0%** | $12.94\text{ s}$ | **PASS** |
| **Aggregate Summary** | **Authentic Document Cohort** | **Hybrid Pipeline** | **100.0% Recall** | **0 Leaks** | **100.0% Preserved** | **6.73 s avg** | **8 / 8 (100% PASS)** |

**Unit & Integration Test Suite:** `23 / 23 PASS (100.0%)`

---

## Technical Defense & Architecture
For architectural proofs, threat models, and oral defense talking points, see [`docs/PRESCRIPTION_ANONYMIZER_RESEARCH_DEFENSE.md`](docs/PRESCRIPTION_ANONYMIZER_RESEARCH_DEFENSE.md).
