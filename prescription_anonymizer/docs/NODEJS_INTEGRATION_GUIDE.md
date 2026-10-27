# Pharmacon Microservice: Node.js & Web Integration Guide

This guide describes how to connect the **Pharmacon Prescription Anonymizer Microservice** with your team's **Node.js backend and web frontend**.

---

## 1. Quick Start the Microservice

```bash
cd prescription_anonymizer
./.venv/bin/python -m api.server
```

- Server starts at: `http://localhost:8000` (or `http://localhost:10000` when running in Docker/Render)
- Interactive Web App (Split Slider UI): **`http://localhost:8000/`** (or `/app`)
- Interactive OpenAPI / Swagger UI: **`http://localhost:8000/docs`**
- Health Probe: `http://localhost:8000/health`

---

## 2. API Contract

### A. `POST /api/v1/anonymize`

#### Option 1: Stream Sanitized File Binary (Recommended for file download)
- **Query Parameter:** `?return_type=file&jurisdiction=AUTO` (or `US`, `UK`, `AU`, `EU`, `IN`, `GLOBAL`)
- **Request Body:** `multipart/form-data` with field `file`
- **Response Headers:**
  - `X-Redacted-Count`: Integer count of masked entities.
  - `X-Format-Type`: `VECTOR_DIGITAL` or `SCANNED_RASTER`.
  - `X-Jurisdiction`: Detected or applied regulatory jurisdiction (`US`, `UK`, `AU`, `EU`, `IN`, `GLOBAL`).
- **Response Body:** Binary stream of anonymized `.pdf` or `.jpg`.

#### Option 2: JSON Audit Metadata + Base64 (Recommended for web preview / UI inspector)
- **Query Parameter:** `?return_type=json&jurisdiction=AUTO`
- **Request Body:** `multipart/form-data` with field `file`
- **Response Body (`application/json`):**
```json
{
  "status": "SUCCESS",
  "filename": "rx_01_anonymized.pdf",
  "jurisdiction": "US",
  "format_type": "VECTOR_DIGITAL",
  "engine": "Native Vector Stream Redactor (PyMuPDF)",
  "total_entities_redacted": 15,
  "redacted_targets": [
    {
      "page_number": 1,
      "category": "PATIENT_NAME",
      "raw_text": "Maria Sharapova",
      "bbox": { "x0": 55.0, "y0": 125.0, "x1": 210.0, "y1": 140.0 },
      "confidence": 0.98,
      "engine": "vector_rule"
    }
  ],
  "data_uri": "data:application/pdf;base64,JVBERi0xLjQK..."
}
```

### B. `POST /api/v1/anonymize/preview` (Side-by-Side Visual Split & Immediate Download)
- **Query Parameter:** `?page=0&jurisdiction=AUTO`
- **Request Body:** `multipart/form-data` with field `file`
- **Response Body (`application/json`):**
  Returns:
  - `original_image_base64`: PNG data URI of the original prescription page.
  - `redacted_image_base64`: PNG data URI of the redacted page.
  - `side_by_side_image_base64`: Side-by-side composite comparison.
  - `download_file_base64`: Base64 data URI of the sanitized document in its original file format, ready for direct client-side download without a second network roundtrip.
  - `confidence_score`: Forensic redaction confidence (0.0 to 1.0).
  - `triage_status`: `AUTO_APPROVED` or `NEEDS_HUMAN_REVIEW`.
  - `review_reasons`: Detailed list of risk factors if human review is advised.

### C. `POST /api/v1/anonymize/file` (Direct Streaming File Download)
- **Query Parameter:** `?jurisdiction=AUTO`
- **Request Body:** `multipart/form-data` with field `file`
- **Response Body:** Binary stream of the anonymized file with `Content-Disposition: attachment; filename="..._anonymized.<ext>"` headers.

### D. `POST /api/v1/anonymize/batch` (Bulk Processing)
- **Query Parameter:** `?jurisdiction=AUTO` (supports `US`, `UK`, `AU`, `EU`, `IN`, `GLOBAL`)
- **Request Body:** `multipart/form-data` with field `file` (`.zip` archive containing multiple PDF/PNG/JPG prescriptions)
- **Response Headers:**
  - `X-Total-Documents`: Number of processed documents
  - `X-Total-Entities-Redacted`: Total count of redacted PII fields
  - `X-Jurisdiction`: Applied regulatory mode
- **Response Body:** Downloadable `.zip` archive containing:
  - All sanitized documents (e.g. `case_01_anonymized.pdf`)
  - `batch_audit_summary.json` containing exhaustive compliance logs and redaction stats.

---

## 3. Node.js Express Controller Integration

Copy and paste into your teammate's Express router:

```javascript
const express = require('express');
const router = express.Router();
const multer = require('multer');
const upload = multer({ dest: '/tmp/uploads/' });
const { anonymizeWithAuditJSON, anonymizeFileStream } = require('./node_client_example');

// Route for Web App (Frontend displays redacted file + entity statistics)
router.post('/upload-prescription', upload.single('prescription'), async (req, res) => {
  try {
    const filePath = req.file.path;
    const auditData = await anonymizeWithAuditJSON(filePath);
    
    // Return audit metadata and data_uri for live React/Vue preview
    res.json({
      success: true,
      data: auditData
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
```

---

## 4. Docker Deployment

Launch both services with Docker Compose:

```bash
docker compose up --build
```
