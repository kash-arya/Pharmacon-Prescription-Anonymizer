#!/usr/bin/env python3
"""
Pharmacon Medical Prescription Anonymizer REST Microservice.

Production-grade FastAPI server providing HTTP endpoints for Node.js, Python,
and web frontend integration.

Endpoints:
    GET  /health              -> Health check & engine status
    GET  /docs                -> Interactive OpenAPI Swagger UI
    POST /api/v1/anonymize    -> Multipart prescription redaction endpoint
"""

import os
import sys
import uuid
import base64
import shutil
import zipfile
import json
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from prescription_anonymizer.src.raster_checker import inspect_pdf_structure
    from prescription_anonymizer.src.vector_redactor import VectorPDFRedactor
    from prescription_anonymizer.src.image_redactor import LocalImageRedactor, resolve_tessdata_path
    from prescription_anonymizer.api.web_app import get_web_app_html
except ImportError:
    from src.raster_checker import inspect_pdf_structure
    from src.vector_redactor import VectorPDFRedactor
    from src.image_redactor import LocalImageRedactor, resolve_tessdata_path
    from api.web_app import get_web_app_html

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


# --- OpenAPI / Pydantic Typed Response Models ---

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status", examples=["healthy"])
    service: str = Field(..., description="Service name", examples=["Pharmacon Anonymizer"])
    version: str = Field(..., description="Service version", examples=["2.0.0"])
    tessdata_directory: str = Field(..., description="Path to resolved Tesseract training assets")
    tesseract_model_present: bool = Field(..., description="Whether eng.traineddata exists")
    supported_extensions: List[str] = Field(..., description="Whitelisted document file extensions")


class RedactedTargetModel(BaseModel):
    page_number: Optional[int] = Field(1, description="Page index where entity was detected")
    category: str = Field(..., description="HIPAA/DPDP entity category", examples=["PATIENT_NAME", "DOCTOR_NAME"])
    raw_text: Optional[str] = Field("", description="Detected plaintext entity payload")
    bbox: Optional[Dict[str, float]] = Field(None, description="Bounding box dictionary [x0, y0, x1, y1]")
    confidence: Optional[float] = Field(1.0, description="Detection confidence score")
    engine: Optional[str] = Field("vector_rule", description="Subsystem that triggered redaction")


class AnonymizeJsonResponse(BaseModel):
    status: str = Field("SUCCESS", examples=["SUCCESS"])
    filename: str = Field(..., description="Sanitized document file name")
    format_type: str = Field(..., description="Detected format category", examples=["VECTOR_DIGITAL", "SCANNED_RASTER"])
    jurisdiction: Optional[str] = Field("GLOBAL", description="Regulatory jurisdiction applied", examples=["US", "UK", "IN"])
    engine: str = Field(..., description="De-identification pipeline utilized")
    total_entities_redacted: int = Field(..., description="Total count of masked entities", examples=[14])
    redacted_targets: List[Dict[str, Any]] = Field(..., description="List of all redacted entity descriptors")
    data_uri: str = Field(..., description="Base64 Data URI of the sanitized binary")


class PreviewResponse(BaseModel):
    status: str = Field("SUCCESS", examples=["SUCCESS"])
    filename: str = Field(..., description="Sanitized document file name")
    format_type: str = Field(..., description="Detected format category", examples=["VECTOR_DIGITAL", "SCANNED_RASTER"])
    jurisdiction: Optional[str] = Field("GLOBAL", description="Regulatory jurisdiction applied", examples=["US", "UK", "IN"])
    engine: str = Field(..., description="De-identification pipeline utilized")
    total_entities_redacted: int = Field(..., description="Total count of masked entities", examples=[14])
    redacted_targets: List[Dict[str, Any]] = Field(..., description="List of all redacted entity descriptors")
    original_image: str = Field(..., description="High-resolution JPEG Base64 Data URI of original page")
    redacted_image: str = Field(..., description="High-resolution JPEG Base64 Data URI of redacted page")


class BatchFileReport(BaseModel):
    original_file: str = Field(..., description="Original path inside ZIP archive")
    sanitized_file: Optional[str] = Field(None, description="Sanitized output filename")
    status: str = Field(..., description="Processing status", examples=["SUCCESS", "ERROR"])
    format_type: Optional[str] = Field(None, description="Detected format category")
    engine: Optional[str] = Field(None, description="Engine used")
    entities_redacted: Optional[int] = Field(0, description="Count of redacted entities")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchAuditSummary(BaseModel):
    service: str = "Pharmacon Medical Prescription Anonymizer"
    version: str = "2.0.0"
    total_documents: int
    successful_redactions: int
    total_entities_redacted: int
    files: List[BatchFileReport]

app = FastAPI(
    title="Pharmacon Prescription Anonymizer API",
    description="High-precision layout-agnostic de-identification microservice for medical prescriptions.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for seamless integration with Node.js and frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Redacted-Count", "X-Format-Type", "X-Jurisdiction", "Content-Disposition"]
)

# Shared redactor singletons to avoid repeated instantiation overhead
vector_engine = VectorPDFRedactor()
image_engine = LocalImageRedactor()


def _cleanup_files(*paths: str):
    """Safely removes temporary files after response transmission."""
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _cleanup_dir(dir_path: str):
    """Safely removes temporary directories and all their contents."""
    if dir_path and os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)


def _render_page_to_base64_jpeg(file_path: str, page_num: int = 0, dpi: int = 120) -> str:
    """Renders a single page of a PDF or image into a base64-encoded JPEG Data URI."""
    _, ext = os.path.splitext(file_path)
    ext_lower = ext.lower()
    if ext_lower == ".pdf":
        import pymupdf
        doc = pymupdf.open(file_path)
        try:
            if len(doc) == 0:
                raise ValueError("PDF is empty.")
            page_idx = min(page_num, len(doc) - 1)
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("jpeg")
        finally:
            doc.close()
    else:
        import cv2
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Failed to read image at {file_path}")
        h, w = img.shape[:2]
        max_dim = 1600
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode image to JPEG")
        img_bytes = buf.tobytes()

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serves the interactive client-side web application interface."""
    return HTMLResponse(get_web_app_html())


@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
async def web_app():
    """Explicit route for the interactive client-side web application."""
    return HTMLResponse(get_web_app_html())


@app.get("/api", tags=["System"])
async def api_info():
    """Microservice metadata and documentation index."""
    return {
        "service": "Pharmacon Prescription Anonymizer Microservice",
        "version": "2.0.0",
        "documentation": "/docs",
        "health": "/health",
        "interactive_ui": "/"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health and readiness probe."""
    tess_path = resolve_tessdata_path()
    return {
        "status": "healthy",
        "service": "Pharmacon Anonymizer",
        "version": "2.0.0",
        "tessdata_directory": tess_path,
        "tesseract_model_present": os.path.exists(os.path.join(tess_path, "eng.traineddata")),
        "supported_extensions": sorted(list(SUPPORTED_EXTENSIONS))
    }


@app.post("/api/v1/anonymize", tags=["Redaction"])
async def anonymize_prescription(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Prescription file (PDF, PNG, JPG, JPEG)"),
    return_type: Literal["file", "json"] = Query(
        "file",
        description="'file' streams the sanitized binary directly; 'json' returns base64 and audit metadata."
    ),
    jurisdiction: Literal["AUTO", "US", "UK", "AU", "EU", "IN", "GLOBAL"] = Query(
        "AUTO",
        description="Target healthcare regulatory framework (AUTO detects automatically)"
    )
):
    """
    De-identifies a medical prescription:
    - Auto-classifies format: Digital Vector vs Scanned Raster.
    - Masks all 18 HIPAA Safe Harbor and DPDP PII/PHI categories.
    - Strictly protects clinical pharmacology and diagnostic findings.
    """
    filename = file.filename or "prescription.pdf"
    _, ext = os.path.splitext(filename)
    ext_lower = ext.lower()

    if ext_lower not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext_lower}'. Supported extensions: {sorted(list(SUPPORTED_EXTENSIONS))}"
        )

    # Write upload to isolated temporary file
    req_id = uuid.uuid4().hex
    temp_dir = f"/tmp/pharmacon_api_{req_id}"
    os.makedirs(temp_dir, exist_ok=True)

    input_path = os.path.join(temp_dir, f"input_{filename}")
    output_filename = f"{os.path.splitext(filename)[0]}_anonymized{ext_lower}"
    output_path = os.path.join(temp_dir, output_filename)

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")
        with open(input_path, "wb") as f:
            f.write(content)
    except HTTPException:
        _cleanup_dir(temp_dir)
        raise
    except Exception as e:
        _cleanup_dir(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {str(e)}")

    # Stage 1: Document Structure Inspection
    try:
        structure = inspect_pdf_structure(input_path)
        fmt = structure["format_type"]
    except Exception as e:
        _cleanup_dir(temp_dir)
        raise HTTPException(status_code=422, detail=f"Failed to parse document structure: {str(e)}")

    # Stage 2: Redaction Routing & Execution
    try:
        if fmt == "VECTOR_DIGITAL":
            engine_name = "Native Vector Stream Redactor (PyMuPDF)"
            report = vector_engine.redact(input_path, output_path, jurisdiction=jurisdiction)
        else:
            engine_name = "Scanned Raster Image Redactor (Local OCR Anchor Engine)"
            report = image_engine.redact(input_path, output_path, jurisdiction=jurisdiction)
    except Exception as e:
        _cleanup_dir(temp_dir)
        raise HTTPException(status_code=500, detail=f"Redaction execution error: {str(e)}")

    redacted_count = report.get("total_entities_redacted", len(report.get("redacted_targets", [])))
    active_jurisdiction = report.get("jurisdiction", "GLOBAL")

    # Return Mode A: Stream sanitized file binary
    if return_type == "file":
        media_type = "application/pdf" if ext_lower == ".pdf" else f"image/{ext_lower.lstrip('.')}"
        background_tasks.add_task(_cleanup_dir, temp_dir)
        return FileResponse(
            output_path,
            media_type=media_type,
            filename=output_filename,
            headers={
                "X-Redacted-Count": str(redacted_count),
                "X-Format-Type": fmt,
                "X-Jurisdiction": active_jurisdiction,
                "X-Engine": engine_name
            }
        )

    # Return Mode B: Return JSON metadata with embedded Base64 output
    try:
        with open(output_path, "rb") as out_f:
            out_bytes = out_f.read()
        b64_str = base64.b64encode(out_bytes).decode("utf-8")
        media_type = "application/pdf" if ext_lower == ".pdf" else f"image/{ext_lower.lstrip('.')}"
        data_uri = f"data:{media_type};base64,{b64_str}"
    finally:
        _cleanup_dir(temp_dir)

    return JSONResponse(content={
        "status": "SUCCESS",
        "filename": output_filename,
        "format_type": fmt,
        "jurisdiction": active_jurisdiction,
        "engine": engine_name,
        "total_entities_redacted": redacted_count,
        "redacted_targets": report.get("redacted_targets", []),
        "data_uri": data_uri
    })


@app.post("/api/v1/anonymize/preview", response_model=PreviewResponse, tags=["Redaction"])
async def anonymize_prescription_preview(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Prescription file (PDF, PNG, JPG, JPEG)"),
    page: int = Query(0, ge=0, description="Zero-indexed page number to preview"),
    jurisdiction: Literal["AUTO", "US", "UK", "AU", "EU", "IN", "GLOBAL"] = Query(
        "AUTO",
        description="Target healthcare regulatory framework (AUTO detects automatically)"
    )
):
    """
    De-identifies a prescription and generates base64 JPEG preview images of page 1 (or specified page)
    for both original and sanitized versions, suitable for side-by-side visual inspection in frontend UIs.
    """
    filename = file.filename or "prescription.pdf"
    _, ext = os.path.splitext(filename)
    ext_lower = ext.lower()

    if ext_lower not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext_lower}'. Supported extensions: {sorted(list(SUPPORTED_EXTENSIONS))}"
        )

    req_id = uuid.uuid4().hex
    temp_dir = f"/tmp/pharmacon_preview_{req_id}"
    os.makedirs(temp_dir, exist_ok=True)

    input_path = os.path.join(temp_dir, f"input_{filename}")
    output_filename = f"{os.path.splitext(filename)[0]}_anonymized{ext_lower}"
    output_path = os.path.join(temp_dir, output_filename)

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")
        with open(input_path, "wb") as f:
            f.write(content)
    except HTTPException:
        _cleanup_dir(temp_dir)
        raise
    except Exception as e:
        _cleanup_dir(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {str(e)}")

    try:
        structure = inspect_pdf_structure(input_path)
        fmt = structure["format_type"]
    except Exception as e:
        _cleanup_dir(temp_dir)
        raise HTTPException(status_code=422, detail=f"Failed to parse document structure: {str(e)}")

    try:
        if fmt == "VECTOR_DIGITAL":
            engine_name = "Native Vector Stream Redactor (PyMuPDF)"
            report = vector_engine.redact(input_path, output_path, jurisdiction=jurisdiction)
        else:
            engine_name = "Scanned Raster Image Redactor (Local OCR Anchor Engine)"
            report = image_engine.redact(input_path, output_path, jurisdiction=jurisdiction)
    except Exception as e:
        _cleanup_dir(temp_dir)
        raise HTTPException(status_code=500, detail=f"Redaction execution error: {str(e)}")

    redacted_count = report.get("total_entities_redacted", len(report.get("redacted_targets", [])))
    active_jurisdiction = report.get("jurisdiction", "GLOBAL")

    try:
        orig_b64 = _render_page_to_base64_jpeg(input_path, page_num=page)
        red_b64 = _render_page_to_base64_jpeg(output_path, page_num=page)
    except Exception as e:
        _cleanup_dir(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to render preview images: {str(e)}")
    finally:
        _cleanup_dir(temp_dir)

    return JSONResponse(content={
        "status": "SUCCESS",
        "filename": output_filename,
        "format_type": fmt,
        "jurisdiction": active_jurisdiction,
        "engine": engine_name,
        "total_entities_redacted": redacted_count,
        "redacted_targets": report.get("redacted_targets", []),
        "original_image": orig_b64,
        "redacted_image": red_b64
    })


@app.post("/api/v1/anonymize/batch", tags=["Redaction"])
async def anonymize_batch_archive(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="ZIP archive containing prescription files (PDF, PNG, JPG)"),
    jurisdiction: Literal["AUTO", "US", "UK", "AU", "EU", "IN", "GLOBAL"] = Query(
        "AUTO",
        description="Target healthcare regulatory framework (AUTO detects automatically)"
    )
):
    """
    Batch de-identifies an entire ZIP archive of prescriptions.
    Returns a ZIP archive containing all sanitized documents along with 'batch_audit_summary.json'.
    """
    filename = file.filename or "batch.zip"
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .zip archive.")

    req_id = uuid.uuid4().hex
    temp_dir = f"/tmp/pharmacon_batch_{req_id}"
    in_extract_dir = os.path.join(temp_dir, "extracted")
    out_redact_dir = os.path.join(temp_dir, "sanitized")
    os.makedirs(in_extract_dir, exist_ok=True)
    os.makedirs(out_redact_dir, exist_ok=True)

    archive_input_path = os.path.join(temp_dir, "upload.zip")
    archive_output_path = os.path.join(temp_dir, f"anonymized_batch_{req_id[:8]}.zip")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded zip archive is empty (0 bytes).")
        with open(archive_input_path, "wb") as f:
            f.write(content)
        with zipfile.ZipFile(archive_input_path, "r") as zf:
            for member in zf.infolist():
                # Prevent Zip-Slip directory traversal vulnerability
                resolved_target = os.path.abspath(os.path.join(in_extract_dir, member.filename))
                if not (resolved_target == os.path.abspath(in_extract_dir) or resolved_target.startswith(os.path.abspath(in_extract_dir) + os.sep)):
                    raise HTTPException(status_code=400, detail=f"Archive contains illegal path (Zip-Slip attempt): {member.filename}")
            zf.extractall(in_extract_dir)
    except HTTPException:
        _cleanup_dir(temp_dir)
        raise
    except zipfile.BadZipFile:
        _cleanup_dir(temp_dir)
        raise HTTPException(status_code=400, detail="Invalid or corrupt ZIP archive.")
    except Exception as e:
        _cleanup_dir(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to extract ZIP archive: {str(e)}")

    found_files = []
    for root_dir, _, filenames in os.walk(in_extract_dir):
        for fname in filenames:
            if fname.startswith(".") or "__MACOSX" in root_dir:
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_p = os.path.join(root_dir, fname)
                rel_p = os.path.relpath(full_p, in_extract_dir)
                found_files.append((full_p, rel_p, fname, ext))

    if not found_files:
        _cleanup_dir(temp_dir)
        raise HTTPException(
            status_code=400,
            detail=f"No supported prescription files found in archive. Supported formats: {sorted(list(SUPPORTED_EXTENSIONS))}"
        )

    batch_reports = []
    total_redactions = 0

    # Detect filename collisions across subdirectories
    name_counts: Dict[str, int] = {}
    for full_p, rel_p, fname, ext in found_files:
        base_cand = f"{os.path.splitext(fname)[0]}_anonymized{ext}"
        name_counts[base_cand] = name_counts.get(base_cand, 0) + 1

    for full_p, rel_p, fname, ext in found_files:
        name_no_ext = os.path.splitext(fname)[0]
        flat_name = f"{name_no_ext}_anonymized{ext}"

        # If collision exists across folders, preserve subfolder path; keep flat when unique
        if name_counts[flat_name] > 1:
            rel_dir = os.path.dirname(rel_p)
            out_fname = os.path.join(rel_dir, flat_name) if rel_dir else flat_name
        else:
            out_fname = flat_name

        out_file_path = os.path.join(out_redact_dir, out_fname)
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)

        try:
            structure = inspect_pdf_structure(full_p)
            fmt = structure["format_type"]
            if fmt == "VECTOR_DIGITAL":
                engine_name = "Native Vector Stream Redactor (PyMuPDF)"
                rep = vector_engine.redact(full_p, out_file_path, jurisdiction=jurisdiction)
            else:
                engine_name = "Scanned Raster Image Redactor (Local OCR Anchor Engine)"
                rep = image_engine.redact(full_p, out_file_path, jurisdiction=jurisdiction)

            cnt = rep.get("total_entities_redacted", len(rep.get("redacted_targets", [])))
            doc_jur = rep.get("jurisdiction", "GLOBAL")
            total_redactions += cnt
            batch_reports.append({
                "original_file": rel_p,
                "sanitized_file": out_fname,
                "status": "SUCCESS",
                "format_type": fmt,
                "jurisdiction": doc_jur,
                "engine": engine_name,
                "entities_redacted": cnt
            })
        except Exception as err:
            batch_reports.append({
                "original_file": rel_p,
                "status": "ERROR",
                "error": str(err)
            })

    summary_data = {
        "service": "Pharmacon Medical Prescription Anonymizer",
        "version": "2.0.0",
        "total_documents": len(found_files),
        "successful_redactions": len([r for r in batch_reports if r.get("status") == "SUCCESS"]),
        "total_entities_redacted": total_redactions,
        "files": batch_reports
    }
    summary_json_path = os.path.join(out_redact_dir, "batch_audit_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as jf:
        json.dump(summary_data, jf, indent=2)

    with zipfile.ZipFile(archive_output_path, "w", compression=zipfile.ZIP_DEFLATED) as out_zf:
        for root_dir, _, filenames in os.walk(out_redact_dir):
            for fn in filenames:
                file_to_zip = os.path.join(root_dir, fn)
                arcname = os.path.relpath(file_to_zip, out_redact_dir)
                out_zf.write(file_to_zip, arcname)

    background_tasks.add_task(_cleanup_dir, temp_dir)
    return FileResponse(
        archive_output_path,
        media_type="application/zip",
        filename=os.path.basename(archive_output_path),
        headers={
            "X-Total-Documents": str(len(found_files)),
            "X-Total-Entities-Redacted": str(total_redactions)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
