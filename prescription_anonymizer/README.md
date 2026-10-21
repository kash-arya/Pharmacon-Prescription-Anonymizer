# Pharmacon Prescription Anonymizer

> Reliable, local de-identification for medical prescriptions and clinical intake forms.  
> Built to satisfy HIPAA Safe Harbor (45 CFR § 164.514(b)), GDPR Article 9, and the Indian DPDP Act 2023 without relying on cloud APIs or leaking clinical text.

---

## Core Capabilities

- **Hybrid Processing Pipeline:** Native vector redaction for digital EMR exports (runs in under 50ms) paired with local Tesseract OCR + CLAHE preprocessing for scanned papers and camera clicks.
- **Local spaCy NER Engine:** Integrates `LocalNEREngine` powered by spaCy's `en_core_web_sm` model to recognize unstructured patient names, locations, and healthcare organizations without relying on rigid keyword anchors or hardcoded lists.
- **Morphological Clinical Protection Whitelist:** Protects pharmacological entities using morphological medical suffixes (`-cillin`, `-mycin`, `-prazole`, `-sartan`, `-dipine`, `-statin`, `-olol`, `-pril`, `-zole`, `-mab`, `-nib`, `-tide`, `-profen`, `-parin`, etc.), diagnostic findings, dosages, and unit metrics.
- **Unified Visual Stamp & Signature Grounding:** Resolution-normalized `detect_visual_pii_from_bgr()` runs across both standalone raster images and scanned PDF pages to localize colored rubber clinic stamps (HSV segmentation) and freeform doctor signatures without explicit label anchors.
- **Zero-Blindspot Barcode Coverage:** Full-surface vertical sliding-window barcode and QR code detector running in <60ms with IoU deduplication.
- **Multi-National Regulatory Profiles:** Automated jurisdiction detection and statutory identifier sanitization across 5 global legal frameworks:
  - 🇺🇸 **United States:** HIPAA Safe Harbor, DEA registration numbers, 10-digit NPIs, state licenses.
  - 🇬🇧 **United Kingdom:** NHS England FP10 standard, 10-digit NHS numbers (3-3-4 spaced), GMC registrations.
  - 🇦🇺 **Australia:** 10-digit Medicare numbers, PBS prescriber numbers, AHPRA practitioner IDs.
  - 🇪🇺 **EU / Germany:** GDPR Article 9, statutory health insurance IDs (`Versicherten-Nr`), `BSNR` practice numbers, and `LANR` doctor IDs.
  - 🇮🇳 **India:** DPDP Act 2023, 14-digit ABHA IDs, hospital UHID/OPD barcodes, and NMC/MCI doctor registrations.
- **Clinical Anti-Encroachment Guard:** Midpoint boundary clamping stops bounding boxes from bleeding into adjacent table cells so prescribed drugs, dosages, and diagnostic findings remain intact.
- **Byte-Level PDF Sterilization:** Strips underlying text streams, clears PDF metadata, and overwrites pixels completely ($\sigma^2 = 0.0000$), blocking adversarial contrast or high-pass edge attacks.
- **REST Microservice & Node.js Integration:** Production-ready FastAPI server with interactive Swagger UI (`/docs`) and ready-to-paste Node.js client helpers for web applications.
- **Self-Contained Inspection Viewer:** Generates a lightweight, single-file HTML viewer with split-slider overlays to review redacted boundaries side-by-side.

---

## Quick Start

### 1. Environment Setup
```bash
source .venv/bin/activate
# or install directly:
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Anonymize a File or Directory (CLI)
Works on digital vector PDFs, scanned PDFs, raw smartphone photos (`.jpg`, `.png`), or entire folders:
```bash
# Single file mode
python anonymize.py --input data/samples/sample_digital_vector_prescription.pdf

# Parallel multi-core batch mode (defaults to max available CPU cores)
python anonymize.py --input data/samples/ --output /tmp/sanitized_prescriptions/ --workers 8
```

### 3. Launch REST Microservice & Web Interface
```bash
# Starts high-performance server on port 8000
python -m api.server
```
- Interactive Web App (Drag & Drop + Live Split Slider): Open `http://localhost:8000/`
- Interactive Swagger UI: Open `http://localhost:8000/docs` in your browser.
- Health Probe: `http://localhost:8000/health`
- Node.js Integration Guide: See [`docs/NODEJS_INTEGRATION_GUIDE.md`](docs/NODEJS_INTEGRATION_GUIDE.md) and [`scripts/node_client_example.js`](scripts/node_client_example.js).

### 4. Run Benchmark Suite
```bash
# Verify recall and zero PII leaks on genuine baseline and smartphone cohort
python benchmarks/benchmark.py
```

### 5. Interactive Inspection Viewer
```bash
# Open static viewer in browser
xdg-open viewer/viewer.html

# Or spin up local server on port 8080
python viewer/viewer.py --serve --port 8080
```

### 6. Automated Test Suite
```bash
python -m unittest tests/test_prescription_anonymizer.py
```

---

## Baseline Evaluation Results (Authentic Clinical Documents)

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

**Unit & Integration Test Suite:** `20 / 20 PASS (100.0%)`

---

## Architecture & Defense Notes
For technical breakdowns, mathematical guarantees, and oral defense talking points, read:  
[`docs/PRESCRIPTION_ANONYMIZER_RESEARCH_DEFENSE.md`](docs/PRESCRIPTION_ANONYMIZER_RESEARCH_DEFENSE.md).
