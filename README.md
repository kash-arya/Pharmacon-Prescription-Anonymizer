# Pharmacon Prescription Anonymizer

> **Healthcare Document De-Identification Engine**  
> Dual-engine pipeline for digital vector PDFs (`VECTOR_DIGITAL`) and scanned/camera prescription images (`SCANNED_RASTER` / `IMAGE_RASTER`).

---

## 1. Overview & Core Design Principle

Clinical prescriptions have a strict de-identification requirement: you must remove every trace of patient identity while leaving every medical instruction intact. If an anonymizer blacks out an active drug, dosage, or frequency instruction, the resulting document is clinically useless for research and training.

Pharmacon enforces an asymmetric redaction boundary:

- **100% PII Recall**: Redacts patient names, dates of birth, ages, phone numbers, statutory IDs (BSNR, LANR, NHS, Medicare, ABHA, Aadhaar, MRN), physical addresses, administrative staff attributions ("Typed By", "Verified By"), rubber stamps, signatures, and scannable 1D/2D barcodes/QR codes.
- **Strict Clinical Preservation**: Preserves 100% of medication names, strengths, dosage units, frequency regimens, routes of administration, and diagnostic findings.

The engine meets statutory de-identification standards across multiple healthcare jurisdictions:
- **HIPAA Safe Harbor** (45 CFR § 164.514(b))
- **GDPR Article 9 & Article 89** (Health data processing for scientific research)
- **Indian Digital Personal Data Protection (DPDP) Act 2023** & **ABHA / ABDM**
- **UK NHS & Australian Medicare/PBS Prescribing Standards**

---

## 2. System Architecture & Pipeline Stages

```
                        ┌──────────────────────────────────────────────┐
                        │        Input Document (PDF / Image)          │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │  Stage 1: Intake & Clinical Document Gate    │
                        │    - 16-byte magic-header check              │
                        │    - Weighted clinical keyword scoring       │
                        │    - Force Anonymize override check          │
                        └───────┬──────────────────────────────┬───────┘
                                │                              │
                [Gate Rejected] │                              │ [Gate Passed]
                                ▼                              ▼
                 HTTP 422 / NOT_A_PRESCRIPTION    ┌────────────────────────────┐
                                                  │ Document Type Classifier   │
                                                  │ (raster_checker.py)        │
                                                  └──────┬──────────────┬──────┘
                                                         │              │
                                        [VECTOR_DIGITAL] │              │ [SCANNED_RASTER]
                                                         ▼              ▼
┌──────────────────────────────────────┐     ┌───────────────────────────────────┐
│ Vector Stream Engine                 │     │ Stage 2: Pre-OCR Normalization    │
│ (vector_redactor.py)                 │     │ (ocr_preprocess.py)               │
│  - PyMuPDF text & rect extraction    │     │  - Decoupled scratch canvas       │
│  - Stream replacement & metadata zap │     │  - Gamma 1.6 + LAB division       │
│  - Latency: < 50 ms/page             │     │  - Explicit `del ocr_scratch`     │
└──────────────────┬───────────────────┘     └─────────────────┬─────────────────┘
                   │                                           │
                   │                                           ▼
                   │                         ┌───────────────────────────────────┐
                   │                         │ Stage 3: Hybrid PII Detection     │
                   │                         │  - Key-value anchor parser        │
                   │                         │  - Collision-safe address logic   │
                   │                         │  - QR/Barcode detector (cv2/zbar) │
                   │                         │  - Visual stamp/ink grounder      │
                   │                         └─────────────────┬─────────────────┘
                   │                                           │
                   ▼                                           ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Audit Triage & Confidence Assessment                                  │
│  - Verification against clinical barriers & protected pharmacological tokens   │
│  - Classification: AUTO_APPROVED (PASS) vs. NEEDS_HUMAN_REVIEW                 │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ Pristine Output Generation                                                     │
│  - Black-box redaction overlay on pristine working canvas                      │
│  - Resolution preserved (up to 1800 px max dimension, JPEG quality >= 92)      │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Intake & Clinical Document Gate (`src/document_validator.py`)
- **Binary Header Validation**: Reads the first 16 bytes of every upload to verify file magic headers (`%PDF-`, `\xFF\xD8\xFF`, `\x89PNG\r\n\x1a\n`, `II*\x00`, `MM\x00*`). Non-matching files fail immediately before the server allocates image decoding buffers.
- **Weighted Clinical Scoring**: Fast regex pre-filter scans for standard medical tokens (`Rx`, `Disp`, `Tab`, `Cap`, `mg`, `ml`, `Sig`, `Diagnosis`). Files scoring below the clinical threshold return `NOT_A_PRESCRIPTION` (`HTTP 422`).
- **Manual Gate Bypass**: Real-world low-contrast slips, receipts, or unusual layouts can bypass the validator using `force_prescription=true` in the API or clicking `"Force Anonymize"` in the web interface.

### Stage 2: Pre-OCR Normalization (`src/ocr_preprocess.py`)
Faint dot-matrix printing and uneven smartphone lighting require aggressive contrast boosting for OCR to read cleanly. However, applying these filters directly to the output image degrades paper texture and leaves text harsh and jagged. 

Pharmacon separates the reading canvas from the rendering canvas:
- **Scratch Buffer Preprocessing**: Applies non-linear gamma darkening ($\gamma = 1.6$), morphological background division in CIELAB color space (`cv2.morphologyEx` close with a $41 \times 41$ kernel), and CLAHE (`clipLimit=2.0, tileGridSize=(8, 8)`).
- **Zero Memory Bleed**: Normalization runs entirely on a disposable scratch buffer that is discarded (`del ocr_scratch`, `del ocr_pdf_bytes`) before launching OCR. Bounding boxes are stamped onto the untouched working canvas (`<= 1800px`, JPEG `q92`), preserving natural background contrast.

### Stage 3: Hybrid PII Detection Engine
- **Universal Form Anchors (`src/form_parser.py`, `src/entities.py`)**:
  - Key-value pattern matching for patient demographics (`Patient Name`, `Age`, `Sex`, `DOB`, `UHID`, `MRN`).
  - Provider details (`Consultant`, `Doctor`, `Dr.`, `Practitioner`, `Reg No`).
  - Administrative staff footers: `{"Typed By", "Prepared By", "Verified By", "Checked By", "Reported By", "Entered By", "Printed By", "Created By"}`.
- **Collision-Safe Address vs. Dosage Precedence**:
  Addresses on medical slips often collide with drug lines. For example, street names like `"12 Cap Road"` or `"Tab Street 5"` contain drug-form words (`Cap`, `Tab`) that naive filters mistake for medications.
  - **Strict Numeric Gate**: Dosage units (`mg`, `ml`, `mcg`, `g`, `gm`, `iu`, `my`, `mp`, `mgs`) only count as medications when preceded by digits (`\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|...)\b`). Bare unit words without digits never block redaction.
  - **Unambiguous Postal Nouns**: Words like `road`, `street`, `avenue`, `lane`, `nagar`, `colony`, `sector`, `marg`, `apartment`, `flat`, `house`, `plot`, `straße`, `gasse`, and `weg` override bare drug-form tokens. `"12 Cap Road"` and `"Drops Avenue 7"` stay masked, while real prescriptions like `"Tab. Paracetamol 500 mg"` remain unmasked.
- **2D Layout Block Propagation**:
  - Downward sweep tracks contiguous patient demographic rows and multi-line clinic headers.
  - The sweep stops the instant it encounters clinical demarcation lines (`Rx:`, `Disp:`, `Diagnosis:`, `Advice:`, `Clinical History:`).
- **Barcode & Visual Grounder (`src/barcode_detector.py`, `src/vlm_grounder.py`)**:
  - Detects 1D Code-128/EAN and 2D QR codes directly on the working canvas. Decoded payloads are discarded immediately without being logged or stored.
  - Morphological contour filtering flags doctor signatures and colored clinic rubber stamps.

### Stage 4: Automated Audit Triage
- Evaluates residual unredacted text, boundary sharpness, and target distribution.
- Assigns automated triage tags:
  - `PASS` / `AUTO_APPROVED`: Clean redaction, 0 detected PII leaks, all clinical tokens intact.
  - `NEEDS_HUMAN_REVIEW`: Raised on high-tilt camera photos, severe motion blur, or ambiguous field boundaries.

---

## 3. Resource Optimization for Cloud Deployment (Render 512 MB Free Tier)

Deploying document OCR to free cloud instances (like Render's **512 MB RAM / 0.1 vCPU** container) typically fails due to memory spikes. A single 12-megapixel phone photo can spike past 600 MB during OpenCV decode and Tesseract execution. 

Pharmacon runs reliably under 310 MB peak RSS through four specific constraints:

1. **Client-Side Downscaling**:
   - The browser scales large images down to $\max(\text{width}, \text{height}) \le 1800\text{ px}$ at JPEG quality `0.92` via an HTML5 `<canvas>` before uploading. This avoids sending 15–20 MB files over the network and halves server-side memory consumption.
   - Files larger than 25 MB are rejected on the client before upload.
2. **Strict Queue Concurrency**:
   - The web UI serializes batch uploads one at a time.
   - An `asyncio.Semaphore(1)` locks the processing pipeline on the server so that only one document is analyzed at any moment. Incoming parallel requests receive `HTTP 503 (Retry-After: 5)`.
3. **Explicit Memory Compaction**:
   - Disposable buffers are unlinked immediately after use (`del ocr_scratch`).
   - After each document finishes, the worker triggers Python garbage collection (`gc.collect()`) followed by `ctypes.CDLL("libc.so.6").malloc_trim(0)`. This forces glibc to return freed heap memory back to the OS instead of holding it.
   - Tested peak RSS on high-resolution camera prescriptions is **307.79 MB**, safely below the 512 MB hard ceiling.

---

## 4. Benchmark & Performance Metrics

### Primary Benchmark Suite (`benchmarks/benchmark.py`)
Tested across representative clinical documents, including digital vector EMRs, scanned doctor slips, multi-page intake forms, and high-resolution camera photos:

```
================================================================================
PHARMACON PRESCRIPTION ANONYMIZER: AUTOMATED EVALUATION BENCHMARK
================================================================================
Total Test Cases Evaluated : 8
Total Passed Cases         : 8 / 8 (100.0%)
Average PII Recall Rate    : 100.00% (Target: 100.0%)
Total PII Leaks Detected   : 0 (Target: 0)
Average Clinical Preserved : 100.00% (Target: 100.0%)
Total Pipeline Latency     : 59,043.7 ms (Avg: 7,380.5 ms/doc)
================================================================================
```

| Document | Format Type | Redacted Targets | Leaked PII | PII Recall | Clinical Preserved | Latency | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sample_digital_vector_prescription.pdf` | `VECTOR_DIGITAL` | 20 | 0 / 9 | 100.0% | 5 / 5 (100.0%) | 776 ms | **PASS** |
| `sample_prescription_1.pdf` | `SCANNED_RASTER` | 10 | 0 / 7 | 100.0% | 5 / 5 (100.0%) | 1,487 ms | **PASS** |
| `sample_prescription_2.pdf` | `SCANNED_RASTER` | 6 | 0 / 7 | 100.0% | 3 / 3 (100.0%) | 4,583 ms | **PASS** |
| `sample_patient_details_1.pdf` | `SCANNED_RASTER` | 42 | 0 / 11 | 100.0% | 5 / 5 (100.0%) | 3,129 ms | **PASS** |
| `sample_prescription_1_p1.png` | `SCANNED_RASTER` | 10 | 0 / 6 | 100.0% | 5 / 5 (100.0%) | 2,174 ms | **PASS** |
| `Aryan March 25.jpg` (Samsung S23) | `SCANNED_RASTER` | 34 | 0 / 13 | 100.0% | 8 / 8 (100.0%) | 8,555 ms | **PASS** |
| `Aryan March 26.jpg` (Samsung S23) | `SCANNED_RASTER` | 30 | 0 / 11 | 100.0% | 8 / 8 (100.0%) | 15,013 ms | **PASS** |
| `Aryan Sept 24.jpg` (Samsung S23) | `SCANNED_RASTER` | 41 | 0 / 12 | 100.0% | 8 / 8 (100.0%) | 23,328 ms | **PASS** |

### Multi-Jurisdiction Unseen Holdout Suite (`data/unseen_holdout/`)
- `commons_eu_de_kassenrezept-at.jpg` (EU/DE GKV Austrian Kassenrezept): **29 targets | PASS**
- `prscrptn1.jpeg` (UK NHS Scanned Rx): **17 targets | PASS**
- `prscrptn2.jpg` (US Outpatient Clinic): **14 targets | PASS**
- `vector_au_pbs_rx.pdf` (AU PBS Digital Vector): **41 targets | PASS**
- `vector_eu_de_muster16.pdf` (DE Muster 16 GKV): **35 targets | PASS**
- `vector_uk_nhs_fp10.pdf` (UK NHS FP10 Vector): **30 targets | PASS**
- `vector_us_hipaa_rx.pdf` (US HIPAA Digital Standard): **32 targets | PASS**

### Unit & Regression Suite
- **Command**: `python3 -m unittest discover tests`
- **Result**: **36 / 36 PASS** (`OK`, 0 errors, 0 failures).
- Regression coverage includes:
  - Address vs. dosage collisions (`"12 Cap Road"`, `"Tab Street 5"`, and `"Drops Avenue 7"` masked; `"Tab. Paracetamol 500 mg"` preserved).
  - Administrative staff attributions (`"Typed By:- KHUSHBOO"` masked).
  - Unscannable synthetic QR codes verified with zero payload leaks.
  - Background pixel brightness preservation ($\Delta \le \pm 3.0$ gray levels).

### Production Resource Benchmark (`Aryan Sept 24.jpg` under 512 MB cgroup)
- **Peak Process RSS (`VmHWM`)**: **307.79 MB** (Ceiling: $< 365\text{ MB}$, Safety Margin: $204.21\text{ MB}$)
- **End-to-End Latency**: **5.07 s**
- **Output Artifact**: $1800 \times 1426\text{ px}$, $401.56\text{ KB}$, JPEG `q92`

---

## 5. Repository Structure

```
SE_Lab_Project/
├── README.md                           # Master documentation & architecture
├── render.yaml                         # Render deployment specification
├── .gitignore                          # Clean repository ignore patterns
└── prescription_anonymizer/            # Core application package
    ├── Dockerfile                      # Production container image
    ├── requirements.txt                # Python dependencies
    ├── anonymize.py                    # Standalone CLI redactor
    ├── api/
    │   ├── server.py                   # FastAPI application & endpoints
    │   ├── web_app.py                  # Entrypoint module
    │   └── frontend.py                 # Web interface & client canvas scaler
    ├── src/
    │   ├── document_validator.py       # Header validation & clinical gate
    │   ├── raster_checker.py           # Document format classifier
    │   ├── vector_redactor.py          # Native digital PDF redaction
    │   ├── image_redactor.py           # OCR & vision redaction pipeline
    │   ├── ocr_preprocess.py           # Pre-OCR contrast & scratch buffer
    │   ├── form_parser.py              # Key-value anchor & demographic parser
    │   ├── entities.py                 # PII schemas & pharmacology whitelist
    │   ├── barcode_detector.py         # Barcode and QR code detector
    │   ├── vlm_grounder.py             # Ink signature & clinic stamp detector
    │   ├── ner_engine.py               # spaCy biomedical named entity recognizer
    │   ├── jurisdiction.py             # Multi-jurisdiction rules resolver
    │   └── forensics.py                # Audit log & verification tracking
    ├── benchmarks/
    │   └── benchmark.py                # 8-document evaluation harness
    ├── tests/
    │   └── test_prescription_anonymizer.py # 36-test unit & integration suite
    └── data/
        ├── samples/                    # Public evaluation samples
        ├── unseen_holdout/             # Multi-jurisdiction test files
        └── tessdata/                   # Tesseract language models
```

---

## 6. Local Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.11 and 3.14)
- Tesseract OCR engine (`tesseract-ocr`, `tesseract-ocr-eng`, `tesseract-ocr-deu`, `tesseract-ocr-fra`, `tesseract-ocr-spa`)
- System libraries: `libgl1`, `libglib2.0-0`

### Setup
```bash
cd prescription_anonymizer

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and spacy model
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Web Application
```bash
# Start server with live reloading on port 8000
python -m uvicorn api.web_app:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) to view the application.

### Tests & Benchmarks
```bash
# Run unit and regression suite (36 tests)
python -m unittest discover tests

# Run benchmark harness (8 test cases)
python benchmarks/benchmark.py

# CLI redaction of a single file
python anonymize.py --input data/samples/sample_digital_vector_prescription.pdf --output /tmp/anonymized.pdf
```

### Docker
```bash
# Build container image
docker build -t pharmacon-anonymizer .

# Run with 512 MB memory limit matching Render deployment
docker run -p 10000:10000 --memory=512m pharmacon-anonymizer
```
Health endpoint is available at `GET /health`.

---

## 7. Compliance & Regulatory Alignment

| Standard | Requirement | Pharmacon Implementation |
| :--- | :--- | :--- |
| **HIPAA 45 CFR § 164.514(b)** | Redaction of 18 Safe Harbor identifier classes | Complete masking of patient names, dates, geographic data below state level, contact numbers, statutory IDs, and device identifiers. |
| **GDPR Art. 9 & 89** | Processing of special category health data for research | Zero third-party network egress. Local vector stream sanitation and destructive pixel masking. |
| **DPDP Act 2023 (India)** | Purpose limitation & data minimization | Automated removal of ABHA IDs, Aadhaar numbers, and clinic patient barcodes with zero persistent storage of unmasked originals. |
| **Clinical Safety** | Uncompromised pharmacological integrity | Whitelist guards ensure zero false-positive redaction of drug molecules, strengths, dosage frequencies, and diagnostic notes. |

---

## 8. License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
