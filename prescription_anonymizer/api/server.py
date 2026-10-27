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
import io
import gc
import ctypes
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
import pymupdf

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks, Response
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

MIME_MAP = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp"
}


# --- OpenAPI / Pydantic Typed Response Models ---

class HealthResponse(BaseModel):
    status: str = Field("ok", description="Service status", examples=["ok"])
    service: str = Field("Pharmacon Anonymizer", description="Service name", examples=["Pharmacon Anonymizer"])
    version: str = Field("2.0.0", description="Service version", examples=["2.0.0"])
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
    confidence_score: Optional[float] = Field(1.0, description="Self-audit confidence score")
    triage_status: Optional[str] = Field("AUTO_APPROVED", description="Triage verdict: AUTO_APPROVED or NEEDS_HUMAN_REVIEW")
    review_reasons: Optional[List[str]] = Field(default_factory=list, description="Triage review reasons")


class PreviewResponse(BaseModel):
    status: str = Field("SUCCESS", examples=["SUCCESS"])
    filename: str = Field(..., description="Original prescription file name")
    anonymized_filename: str = Field(..., description="Sanitized document file name")
    mime_type: str = Field(..., description="MIME type of sanitized document", examples=["application/pdf", "image/jpeg"])
    format_type: str = Field(..., description="Detected format category", examples=["VECTOR_DIGITAL", "SCANNED_RASTER"])
    jurisdiction: Optional[str] = Field("GLOBAL", description="Regulatory jurisdiction applied", examples=["US", "UK", "IN"])
    engine: str = Field(..., description="De-identification pipeline utilized")
    total_entities_redacted: int = Field(..., description="Total count of masked entities", examples=[14])
    confidence_score: Optional[float] = Field(1.0, description="Self-audit confidence score (0.0 to 0.99)")
    triage_status: Optional[str] = Field("AUTO_APPROVED", description="Triage verdict: AUTO_APPROVED or NEEDS_HUMAN_REVIEW")
    review_reasons: Optional[List[str]] = Field(default_factory=list, description="Triage review reasons")
    original_image_base64: str = Field(..., description="High-resolution PNG Base64 Data URI of original page for left pane")
    redacted_image_base64: str = Field(..., description="High-resolution PNG Base64 Data URI of redacted page for right pane")
    side_by_side_image_base64: str = Field(..., description="Combined side-by-side comparison JPEG Base64 Data URI")
    download_file_base64: str = Field(..., description="Complete anonymized file payload Base64 Data URI in original format")
    redacted_targets: List[Dict[str, Any]] = Field(default_factory=list, description="List of all redacted entity descriptors")
    # Legacy backward-compatibility aliases
    original_image: Optional[str] = Field(None, description="Legacy alias for original_image_base64")
    redacted_image: Optional[str] = Field(None, description="Legacy alias for redacted_image_base64")
    data_uri: Optional[str] = Field(None, description="Legacy alias for download_file_base64")


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

# Enable CORS for seamless integration with Vercel frontend, Node.js, and external clients
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS")
if allowed_origins_env:
    allow_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
else:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
        "X-Redacted-Count",
        "X-Jurisdiction",
        "X-Confidence-Score",
        "X-Triage-Status",
        "X-Format-Type"
    ]
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


def _purge_system_memory():
    """Purges PyMuPDF global cache, trims malloc arenas, and clears engine state."""
    try:
        pymupdf.TOOLS.store_shrink(100)
    except Exception:
        pass
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    if os.environ.get("RENDER_LOW_MEMORY") == "1":
        image_engine._last_extracted_words = []
        image_engine._last_whitelisted_rects = []
        image_engine._last_extracted_text = ""


def _run_redaction(fmt: str, input_path: str, output_path: str, jurisdiction: str) -> Tuple[str, Dict[str, Any]]:
    """Executes redaction pipeline directly in-process (single process)."""
    if fmt == "VECTOR_DIGITAL":
        return "Native Vector Stream Redactor (PyMuPDF)", vector_engine.redact(input_path, output_path, jurisdiction=jurisdiction)
    else:
        return "Scanned Raster Image Redactor (Local OCR Anchor Engine)", image_engine.redact(input_path, output_path, jurisdiction=jurisdiction)


def _render_page_to_pil(file_path: str, page_num: int = 0, dpi: int = 150, max_height: Optional[int] = None) -> Image.Image:
    """Renders a single page of a PDF or loads an image into a clean RGB PIL Image."""
    _, ext = os.path.splitext(file_path)
    ext_lower = ext.lower()
    is_low_mem = os.environ.get("RENDER_LOW_MEMORY") == "1"
    img = None
    if ext_lower == ".pdf":
        doc = pymupdf.open(file_path)
        try:
            if len(doc) == 0:
                raise ValueError("PDF is empty.")
            page_idx = min(page_num, len(doc) - 1)
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            try:
                if pix.n >= 4:
                    img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples).convert("RGB")
                else:
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            finally:
                del pix
                try:
                    pymupdf.TOOLS.store_shrink(100)
                except Exception:
                    pass
        finally:
            doc.close()
            try:
                pymupdf.TOOLS.store_shrink(100)
            except Exception:
                pass
    else:
        try:
            with Image.open(file_path) as im:
                if is_low_mem and max(im.size) > 700:
                    im.draft("RGB", (max_height or 700, max_height or 700))
                img = im.convert("RGB")
        except Exception:
            doc = pymupdf.open(file_path)
            try:
                pix = doc[0].get_pixmap(dpi=dpi)
                try:
                    if pix.n >= 4:
                        img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples).convert("RGB")
                    else:
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                finally:
                    del pix
                    try:
                        pymupdf.TOOLS.store_shrink(100)
                    except Exception:
                        pass
            finally:
                doc.close()
                try:
                    pymupdf.TOOLS.store_shrink(100)
                except Exception:
                    pass

    if img is not None and max_height is not None and img.height > max_height:
        scale = float(max_height) / img.height
        new_w = max(1, int(img.width * scale))
        resample_mode = Image.Resampling.BILINEAR if is_low_mem else Image.Resampling.LANCZOS
        img = img.resize((new_w, max_height), resample_mode)

    return img


def _pil_to_base64_data_uri(img: Image.Image, format: str = "PNG", quality: int = 85) -> str:
    """Encodes a PIL Image into a base64 Data URI."""
    buf = io.BytesIO()
    if format.upper() in ("JPG", "JPEG"):
        img.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _generate_side_by_side_base64(
    orig_img: Image.Image,
    red_img: Image.Image,
    triage_status: str = "AUTO_APPROVED",
    confidence: float = 0.96
) -> str:
    """
    Assembles a high-contrast side-by-side comparison JPEG with an audit banner:
    Left: Original Prescription
    Right: Anonymized Output with Triage Status & Confidence Badge
    """
    is_low_mem = os.environ.get("RENDER_LOW_MEMORY") == "1"
    max_h_limit = 700 if is_low_mem else 1200
    target_h = max(450 if is_low_mem else 700, min(max_h_limit, max(orig_img.height, red_img.height)))
    w_orig = max(1, int(orig_img.width * (target_h / orig_img.height)))
    w_red = max(1, int(red_img.width * (target_h / red_img.height)))

    resample_mode = Image.Resampling.BILINEAR if is_low_mem else Image.Resampling.LANCZOS
    orig_scaled = orig_img.resize((w_orig, target_h), resample_mode)
    red_scaled = red_img.resize((w_red, target_h), resample_mode)

    header_h = 55
    divider_w = 4
    canvas_w = w_orig + w_red + divider_w
    canvas_h = header_h + target_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 23, 42))  # Slate 900
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Left: ORIGINAL
    draw.rectangle([0, 0, w_orig, header_h], fill=(30, 41, 59))
    draw.text((16, 18), "ORIGINAL PRESCRIPTION (PRE-REDACTION)", fill=(203, 213, 225), font=font)

    # Right: ANONYMIZED
    rx0 = w_orig + divider_w
    bg_right = (6, 78, 59) if triage_status == "AUTO_APPROVED" else (120, 53, 15)
    draw.rectangle([rx0, 0, canvas_w, header_h], fill=bg_right)
    draw.text(
        (rx0 + 16, 18),
        f"ANONYMIZED OUTPUT | {triage_status} ({int(confidence * 100)}% CONFIDENCE)",
        fill=(240, 253, 244) if triage_status == "AUTO_APPROVED" else (254, 243, 199),
        font=font
    )

    # Divider line
    draw.rectangle([w_orig, 0, w_orig + divider_w, canvas_h], fill=(59, 130, 246))

    # Composite pages
    canvas.paste(orig_scaled, (0, header_h))
    canvas.paste(red_scaled, (rx0, header_h))

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=75 if is_low_mem else 85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    del canvas, orig_scaled, red_scaled
    return f"data:image/jpeg;base64,{b64}"


def _render_page_to_base64_jpeg(file_path: str, page_num: int = 0, dpi: int = 120) -> str:
    """Renders a single page of a PDF or image into a base64-encoded JPEG Data URI."""
    img = _render_page_to_pil(file_path, page_num=page_num, dpi=dpi)
    return _pil_to_base64_data_uri(img, format="JPEG", quality=85)


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
    _purge_system_memory()
    tess_path = resolve_tessdata_path()
    return {
        "status": "ok",
        "service": "Pharmacon Anonymizer",
        "version": "2.0.0",
        "tessdata_directory": tess_path,
        "tesseract_model_present": os.path.exists(os.path.join(tess_path, "eng.traineddata")),
        "supported_extensions": sorted(list(SUPPORTED_EXTENSIONS))
    }


@app.post("/anonymize", tags=["Redaction"])
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

        structure = inspect_pdf_structure(input_path)
        fmt = structure["format_type"]

        engine_name, report = _run_redaction(fmt, input_path, output_path, jurisdiction)

        redacted_count = report.get("total_entities_redacted", len(report.get("redacted_targets", [])))
        active_jurisdiction = report.get("jurisdiction", "GLOBAL")
        conf_score = float(report.get("confidence_score", 0.96))
        triage_stat = str(report.get("triage_status", "AUTO_APPROVED"))
        reasons = report.get("review_reasons", [])
        media_type = MIME_MAP.get(ext_lower, "application/octet-stream")

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise HTTPException(status_code=500, detail="Sanitized file output was not generated or is empty.")

        # Return Mode A: Stream sanitized file binary
        if return_type == "file":
            with open(output_path, "rb") as out_f:
                file_bytes = out_f.read()
            return Response(
                content=file_bytes,
                media_type=media_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{output_filename}"',
                    "X-Redacted-Count": str(redacted_count),
                    "X-Format-Type": fmt,
                    "X-Jurisdiction": active_jurisdiction,
                    "X-Engine": engine_name,
                    "X-Confidence-Score": str(conf_score),
                    "X-Triage-Status": triage_stat
                }
            )

        # Return Mode B: Return JSON metadata with embedded Base64 output
        with open(output_path, "rb") as out_f:
            out_bytes = out_f.read()
        b64_str = base64.b64encode(out_bytes).decode("utf-8")
        del out_bytes
        data_uri = f"data:{media_type};base64,{b64_str}"
        del b64_str

        return JSONResponse(content={
            "status": "SUCCESS",
            "filename": output_filename,
            "format_type": fmt,
            "jurisdiction": active_jurisdiction,
            "engine": engine_name,
            "total_entities_redacted": redacted_count,
            "redacted_targets": report.get("redacted_targets", []),
            "data_uri": data_uri,
            "confidence_score": conf_score,
            "triage_status": triage_stat,
            "review_reasons": reasons
        })
    finally:
        _cleanup_dir(temp_dir)
        _purge_system_memory()


@app.post("/anonymize/file", tags=["Redaction"])
@app.post("/api/v1/anonymize/file", tags=["Redaction"])
async def anonymize_prescription_file_download(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Prescription file (PDF, PNG, JPG, JPEG, WEBP)"),
    jurisdiction: Literal["AUTO", "US", "UK", "AU", "EU", "IN", "GLOBAL"] = Query(
        "AUTO",
        description="Target healthcare regulatory framework (AUTO detects automatically)"
    )
):
    """
    De-identifies a prescription and streams back the sanitized file in the exact same
    file extension and MIME type as uploaded, with Content-Disposition attachment header.
    """
    filename = file.filename or "prescription.pdf"
    stem, ext = os.path.splitext(filename)
    ext_lower = ext.lower()

    if ext_lower not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext_lower}'. Supported extensions: {sorted(list(SUPPORTED_EXTENSIONS))}"
        )

    req_id = uuid.uuid4().hex
    temp_dir = f"/tmp/pharmacon_file_{req_id}"
    os.makedirs(temp_dir, exist_ok=True)

    input_path = os.path.join(temp_dir, f"input_{filename}")
    output_filename = f"{stem}_anonymized{ext_lower}"
    output_path = os.path.join(temp_dir, output_filename)

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")
        with open(input_path, "wb") as f:
            f.write(content)

        structure = inspect_pdf_structure(input_path)
        fmt = structure["format_type"]

        engine_name, report = _run_redaction(fmt, input_path, output_path, jurisdiction)

        redacted_count = report.get("total_entities_redacted", len(report.get("redacted_targets", [])))
        active_jurisdiction = report.get("jurisdiction", "GLOBAL")
        conf_score = float(report.get("confidence_score", 0.96))
        triage_stat = str(report.get("triage_status", "AUTO_APPROVED"))
        media_type = MIME_MAP.get(ext_lower, "application/octet-stream")

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise HTTPException(status_code=500, detail="Sanitized file output was not generated or is empty.")

        with open(output_path, "rb") as out_f:
            file_bytes = out_f.read()

        return Response(
            content=file_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "X-Redacted-Count": str(redacted_count),
                "X-Format-Type": fmt,
                "X-Jurisdiction": active_jurisdiction,
                "X-Engine": engine_name,
                "X-Confidence-Score": str(conf_score),
                "X-Triage-Status": triage_stat
            }
        )
    finally:
        _cleanup_dir(temp_dir)
        _purge_system_memory()


@app.post("/anonymize/preview", response_model=PreviewResponse, tags=["Redaction"])
@app.post("/api/v1/anonymize/preview", response_model=PreviewResponse, tags=["Redaction"])
async def anonymize_prescription_preview(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Prescription file (PDF, PNG, JPG, JPEG, WEBP)"),
    page: int = Query(0, ge=0, description="Zero-indexed page number to preview"),
    jurisdiction: Literal["AUTO", "US", "UK", "AU", "EU", "IN", "GLOBAL"] = Query(
        "AUTO",
        description="Target healthcare regulatory framework (AUTO detects automatically)"
    )
):
    """
    De-identifies a prescription and generates a unified preview payload for the Vercel frontend:
    - original_image_base64: PNG/JPEG Base64 Data URI of original document for the Left pane
    - redacted_image_base64: PNG/JPEG Base64 Data URI of redacted document for the Right pane
    - side_by_side_image_base64: JPEG Base64 Data URI with comparative audit banner
    - download_file_base64: Exact anonymized binary in original format (.pdf/.jpg/.png/.webp)
      allowing one-click instant client-side downloads without a second request
    - Deterministic audit confidence score, triage verdict, and entity count.
    """
    filename = file.filename or "prescription.pdf"
    stem, ext = os.path.splitext(filename)
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
    output_filename = f"{stem}_anonymized{ext_lower}"
    output_path = os.path.join(temp_dir, output_filename)

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")
        with open(input_path, "wb") as f:
            f.write(content)

        structure = inspect_pdf_structure(input_path)
        fmt = structure["format_type"]

        engine_name, report = _run_redaction(fmt, input_path, output_path, jurisdiction)

        redacted_count = report.get("total_entities_redacted", len(report.get("redacted_targets", [])))
        active_jurisdiction = report.get("jurisdiction", "GLOBAL")
        conf_score = float(report.get("confidence_score", 0.96))
        triage_stat = str(report.get("triage_status", "AUTO_APPROVED"))
        reasons = report.get("review_reasons", [])
        mime_type = MIME_MAP.get(ext_lower, "application/octet-stream")

        is_low_mem = os.environ.get("RENDER_LOW_MEMORY") == "1"
        preview_dpi = 80 if is_low_mem else 150
        preview_max_h = 700 if is_low_mem else None
        preview_fmt = "JPEG" if is_low_mem else "PNG"
        preview_quality = 72 if is_low_mem else 85

        orig_pil = _render_page_to_pil(input_path, page_num=page, dpi=preview_dpi, max_height=preview_max_h)
        red_pil = _render_page_to_pil(output_path, page_num=page, dpi=preview_dpi, max_height=preview_max_h)

        side_by_side_b64 = _generate_side_by_side_base64(
            orig_pil, red_pil,
            triage_status=triage_stat,
            confidence=conf_score
        )
        orig_preview_b64 = _pil_to_base64_data_uri(orig_pil, format=preview_fmt, quality=preview_quality)
        red_preview_b64 = _pil_to_base64_data_uri(red_pil, format=preview_fmt, quality=preview_quality)
        del orig_pil, red_pil

        try:
            pymupdf.TOOLS.store_shrink(100)
        except Exception:
            pass
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        if is_low_mem and ext_lower in {".jpg", ".jpeg", ".png", ".webp"}:
            download_b64 = red_preview_b64
        else:
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError(f"Redacted output file is missing or empty at {output_path}")
            with open(output_path, "rb") as out_f:
                file_bytes = out_f.read()
            file_b64_str = base64.b64encode(file_bytes).decode("utf-8")
            del file_bytes
            download_b64 = f"data:{mime_type};base64,{file_b64_str}"
            del file_b64_str

        try:
            pymupdf.TOOLS.store_shrink(100)
        except Exception:
            pass
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        return JSONResponse(content={
            "status": "SUCCESS",
            "filename": filename,
            "anonymized_filename": output_filename,
            "mime_type": mime_type,
            "format_type": fmt,
            "jurisdiction": active_jurisdiction,
            "engine": engine_name,
            "total_entities_redacted": redacted_count,
            "confidence_score": conf_score,
            "triage_status": triage_stat,
            "review_reasons": reasons,
            "original_image_base64": orig_preview_b64,
            "redacted_image_base64": red_preview_b64,
            "side_by_side_image_base64": side_by_side_b64,
            "download_file_base64": download_b64,
            "redacted_targets": report.get("redacted_targets", []),
            # Legacy backward-compatibility aliases
            "original_image": orig_preview_b64,
            "redacted_image": red_preview_b64,
            "data_uri": download_b64
        })
    finally:
        _cleanup_dir(temp_dir)
        _purge_system_memory()


@app.post("/anonymize/batch", tags=["Redaction"])
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
            raise
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid or corrupt ZIP archive.")
        except Exception as e:
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
                engine_name, rep = _run_redaction(fmt, full_p, out_file_path, jurisdiction)

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

        if not os.path.exists(archive_output_path) or os.path.getsize(archive_output_path) == 0:
            raise HTTPException(status_code=500, detail="Batch output archive could not be generated.")

        with open(archive_output_path, "rb") as azf:
            zip_bytes = azf.read()

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{os.path.basename(archive_output_path)}"',
                "X-Total-Documents": str(len(found_files)),
                "X-Total-Entities-Redacted": str(total_redactions)
            }
        )
    finally:
        _cleanup_dir(temp_dir)
        _purge_system_memory()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
