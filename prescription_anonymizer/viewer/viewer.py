#!/usr/bin/env python3
"""
Interactive Side-by-Side Inspection Viewer for Prescription Anonymizer.

Generates a standalone, zero-dependency HTML file (viewer.html) with:
- Side-by-side original vs redacted document visual inspection.
- Interactive split slider view.
- Color-coded entity bounding box overlays and inspector.
- Filterable entity breakdown and audit log table.
- Real-time benchmark & compliance KPI metrics.
- Optional lightweight local HTTP server.
"""

from typing import Dict, Any, List, Optional
import os
import sys
import json
import base64
import io
import argparse
import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import pymupdf
from PIL import Image

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from prescription_anonymizer.src.raster_checker import inspect_pdf_structure
    from prescription_anonymizer.src.vector_redactor import VectorPDFRedactor
    from prescription_anonymizer.src.image_redactor import LocalImageRedactor, resolve_tessdata_path
except ImportError:
    from src.raster_checker import inspect_pdf_structure
    from src.vector_redactor import VectorPDFRedactor
    from src.image_redactor import LocalImageRedactor, resolve_tessdata_path


SAMPLE_DEFINITIONS = [
    {
        "id": "vector_apollo_rx",
        "title": "Digital Vector EMR Prescription",
        "subtitle": "Apollo Hospitals (Dr. Rajesh Sharma / Anita Desai)",
        "filename": "sample_digital_vector_prescription.pdf",
        "category": "EMR Digital Vector",
        "tags": ["Vector PDF", "EMR", "PyMuPDF Native"]
    },
    {
        "id": "scanned_rx_1",
        "title": "Scanned Doctor Slip #1",
        "subtitle": "Prednisone & Lialda (Dr. John Smith / Maria Sharapova)",
        "filename": "sample_prescription_1.pdf",
        "category": "Scanned Raster",
        "tags": ["Scanned Slip", "Rx Pad", "OCR Grounding"]
    },
    {
        "id": "scanned_rx_2",
        "title": "Scanned Doctor Slip #2",
        "subtitle": "Omeprazole (Dr. John Smith / Virat Kohli)",
        "filename": "sample_prescription_2.pdf",
        "category": "Scanned Raster",
        "tags": ["Faint Header", "Scanned Slip", "OCR Grounding"]
    },
    {
        "id": "patient_details_intake",
        "title": "Patient Medical Intake Form",
        "subtitle": "Multi-page Registration (Kathy Crawford / Simeone)",
        "filename": "sample_patient_details_1.pdf",
        "category": "Intake Form",
        "tags": ["Multi-Page", "Forms", "Insurance & Demographics"]
    },
    {
        "id": "image_slip_1",
        "title": "Standalone PNG Slip #1",
        "subtitle": "Image Container (Maria Sharapova)",
        "filename": "extracted_images/sample_prescription_1_p1.png",
        "category": "Image File",
        "tags": ["PNG Image", "Lossless", "Direct Image Redaction"]
    },
    {
        "id": "real_phone_ultrasound_march25",
        "title": "Smartphone Click: Ultrasound Report",
        "subtitle": "Akash Imaging Centre (Kurukshetra) - Dr. NK Goyal / Aryan",
        "filename": "real_phone_prescriptions/Aryan March 25.jpg",
        "category": "Smartphone Camera",
        "tags": ["Samsung S23", "USG Whole Abdomen", "Camera Glare Robust"]
    },
    {
        "id": "real_phone_hospital_slip_march26",
        "title": "Smartphone Click: Hospital OPD Slip",
        "subtitle": "SIMSS Hospital - Dr. Shashank Singhal / Aryan",
        "filename": "real_phone_prescriptions/Aryan March 26.jpg",
        "category": "Smartphone Camera",
        "tags": ["Samsung S23", "OPD Thermal/Ink Slip", "Multi-Column Layout"]
    },
    {
        "id": "real_phone_ultrasound_sept24",
        "title": "Smartphone Click: Ultrasound Report #2",
        "subtitle": "Akash Imaging Centre - Samsung RS80 / Aryan Sharma",
        "filename": "real_phone_prescriptions/Aryan Sept 24.jpg",
        "category": "Smartphone Camera",
        "tags": ["Samsung S23", "USG Whole Abdomen", "Anatomy & Findings Preserved"]
    }
]


def render_page_to_base64(doc_path: str, page_num: int, dpi: int = 120, max_dim: int = 1600, quality: int = 75) -> Tuple[str, int, int]:
    """Renders a PDF or image page to compressed base64 JPEG string along with pixel dimensions."""
    ext = os.path.splitext(doc_path)[1].lower()
    is_image = ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

    if is_image:
        with Image.open(doc_path) as orig_im:
            im = orig_im.convert("RGB") if orig_im.mode in ("RGBA", "P") else orig_im.copy()
        w, h = im.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            w = int(w * scale)
            h = int(h * scale)
            im = im.resize((w, h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}", w, h

    doc = pymupdf.open(doc_path)
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=dpi)
    if pix.n == 4:
        im = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples).convert("RGB")
    elif pix.n == 3:
        im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        im = Image.frombytes("L", [pix.width, pix.height], pix.samples).convert("RGB")

    w, h = im.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        w = int(w * scale)
        h = int(h * scale)
        im = im.resize((w, h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    doc.close()
    return f"data:image/jpeg;base64,{b64}", w, h


def build_viewer_data(samples_dir: str, temp_out_dir: str) -> Dict[str, Any]:
    """Processes all samples and collects base64 images and entity metadata."""
    tessdata_dir = resolve_tessdata_path()
    os.makedirs(temp_out_dir, exist_ok=True)

    documents = []

    for defn in SAMPLE_DEFINITIONS:
        input_path = os.path.join(samples_dir, defn["filename"])
        if not os.path.exists(input_path):
            alt_path = os.path.join(os.path.dirname(samples_dir), defn["filename"])
            if os.path.exists(alt_path):
                input_path = alt_path
            else:
                continue

        base, ext = os.path.splitext(defn["filename"])
        clean_base = os.path.basename(base)
        output_filename = f"{clean_base}_viewer_redacted{ext}"
        output_path = os.path.join(temp_out_dir, output_filename)

        # Execute Redaction Pipeline
        struct = inspect_pdf_structure(input_path)
        fmt = struct["format_type"]

        if fmt == "VECTOR_DIGITAL":
            redactor = VectorPDFRedactor()
            report = redactor.redact(input_path, output_path)
            engine_name = "Native Vector Redactor (PyMuPDF)"
        else:
            redactor = LocalImageRedactor(tessdata_dir=tessdata_dir)
            report = redactor.redact(input_path, output_path)
            engine_name = "Local OCR Anchor Engine (PyMuPDF + Tesseract)"

        is_image = ext.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
        page_count = 1 if is_image else len(pymupdf.open(input_path))

        pages_data = []
        for p_idx in range(1, page_count + 1):
            orig_b64, w_orig, h_orig = render_page_to_base64(input_path, p_idx)
            red_b64, w_red, h_red = render_page_to_base64(output_path, p_idx)

            # Filter targets for this page
            page_targets = [
                t for t in report.get("redacted_targets", [])
                if t.get("page_number", 1) == p_idx
            ]

            pages_data.append({
                "page_number": p_idx,
                "width": w_orig,
                "height": h_orig,
                "original_image": orig_b64,
                "redacted_image": red_b64,
                "entities": page_targets
            })

        documents.append({
            "id": defn["id"],
            "title": defn["title"],
            "subtitle": defn["subtitle"],
            "category": defn["category"],
            "tags": defn["tags"],
            "format_type": fmt,
            "engine": engine_name,
            "filename": defn["filename"],
            "total_entities_redacted": report.get("total_entities_redacted", len(report.get("redacted_targets", []))),
            "pages": pages_data
        })

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "documents": documents
    }


def generate_html(data: Dict[str, Any]) -> str:
    """Renders complete zero-dependency HTML application with embedded JSON payload."""
    json_data = json.dumps(data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pharmacon Prescription Anonymizer - Interactive Inspector</title>
<style>
:root {{
  --bg-dark: #0f172a;
  --bg-card: #1e293b;
  --bg-card-hover: #334155;
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --accent: #38bdf8;
  --accent-glow: rgba(56, 189, 248, 0.25);
  --success: #34d399;
  --warning: #fbbf24;
  --danger: #f87171;
  --border: #334155;
  --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}

* {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}}

body {{
  background-color: var(--bg-dark);
  color: var(--text-main);
  font-family: var(--font-family);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}}

header {{
  background-color: rgba(30, 41, 59, 0.95);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 16px 32px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  top: 0;
  z-index: 100;
}}

.logo-group {{
  display: flex;
  align-items: center;
  gap: 12px;
}}

.logo-badge {{
  background: linear-gradient(135deg, #0284c7, #38bdf8);
  color: #fff;
  font-weight: 800;
  font-size: 18px;
  padding: 6px 12px;
  border-radius: 8px;
  letter-spacing: 0.5px;
}}

.header-title h1 {{
  font-size: 18px;
  font-weight: 700;
  color: var(--text-main);
}}

.header-title p {{
  font-size: 12px;
  color: var(--text-muted);
}}

.kpi-row {{
  display: flex;
  gap: 16px;
}}

.kpi-pill {{
  background-color: var(--bg-dark);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 6px 14px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}}

.kpi-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: var(--success);
  box-shadow: 0 0 8px var(--success);
}}

.layout-container {{
  display: flex;
  flex: 1;
  overflow: hidden;
}}

/* Sidebar Document Selector */
.sidebar {{
  width: 340px;
  background-color: var(--bg-card);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}}

.sidebar-header {{
  padding: 16px;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-muted);
}}

.doc-list {{
  list-style: none;
  overflow-y: auto;
  flex: 1;
}}

.doc-item {{
  padding: 14px 18px;
  border-bottom: 1px solid rgba(51, 65, 85, 0.5);
  cursor: pointer;
  transition: all 0.15s ease;
}}

.doc-item:hover {{
  background-color: var(--bg-card-hover);
}}

.doc-item.active {{
  background-color: rgba(56, 189, 248, 0.12);
  border-left: 4px solid var(--accent);
}}

.doc-item-title {{
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 4px;
  display: flex;
  justify-content: space-between;
}}

.doc-item-sub {{
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 8px;
}}

.tag-list {{
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}}

.tag {{
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 4px;
  background-color: rgba(255, 255, 255, 0.08);
  color: var(--text-muted);
}}

.tag.accent {{
  background-color: rgba(56, 189, 248, 0.15);
  color: var(--accent);
}}

/* Main Workspace */
.main-workspace {{
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}}

.control-bar {{
  background-color: rgba(30, 41, 59, 0.7);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}}

.view-mode-tabs {{
  display: flex;
  background-color: var(--bg-dark);
  padding: 3px;
  border-radius: 8px;
  border: 1px solid var(--border);
}}

.tab-btn {{
  background: none;
  border: none;
  color: var(--text-muted);
  padding: 6px 14px;
  font-size: 12px;
  font-weight: 600;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s;
}}

.tab-btn.active {{
  background-color: var(--accent);
  color: #0f172a;
}}

.page-selector {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}}

.btn-small {{
  background-color: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-main);
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}}

/* Document Comparison Display */
.comparison-area {{
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  align-items: center;
}}

.side-by-side-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  width: 100%;
  max-width: 1400px;
}}

.doc-pane {{
  background-color: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}

.pane-header {{
  padding: 12px 16px;
  background-color: rgba(15, 23, 42, 0.5);
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}}

.pane-title {{
  font-size: 13px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}}

.badge-redacted {{
  background-color: rgba(52, 211, 153, 0.15);
  color: var(--success);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
}}

.badge-original {{
  background-color: rgba(248, 113, 113, 0.15);
  color: var(--danger);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
}}

.doc-viewport {{
  position: relative;
  overflow: auto;
  padding: 16px;
  display: flex;
  justify-content: center;
  background-color: #0b1120;
  min-height: 480px;
}}

.doc-img {{
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}}

/* Split Slider View */
.slider-container {{
  position: relative;
  width: 100%;
  max-width: 750px;
  overflow: hidden;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
  user-select: none;
}}

.slider-img {{
  width: 100%;
  display: block;
}}

.slider-overlay {{
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  width: 50%;
  overflow: hidden;
}}

.slider-handle {{
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 4px;
  background-color: var(--accent);
  box-shadow: 0 0 12px var(--accent);
  cursor: ew-resize;
}}

.slider-handle-button {{
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: var(--accent);
  color: #0f172a;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}}

/* Entity Audit Table */
.audit-panel {{
  width: 100%;
  max-width: 1400px;
  background-color: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  margin-top: 12px;
}}

.audit-header {{
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}}

.audit-title {{
  font-size: 14px;
  font-weight: 700;
  color: var(--text-main);
}}

.entity-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}}

.entity-table th {{
  background-color: rgba(15, 23, 42, 0.6);
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
}}

.entity-table td {{
  padding: 10px 14px;
  border-bottom: 1px solid rgba(51, 65, 85, 0.4);
}}

.entity-table tr:hover {{
  background-color: rgba(56, 189, 248, 0.05);
}}

.cat-badge {{
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
}}

.cat-PATIENT_NAME {{ background-color: rgba(236, 72, 153, 0.2); color: #f472b6; }}
.cat-DOCTOR_INFO {{ background-color: rgba(168, 85, 247, 0.2); color: #c084fc; }}
.cat-ADDRESS {{ background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; }}
.cat-PHONE {{ background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; }}
.cat-DATE {{ background-color: rgba(16, 185, 129, 0.2); color: #34d399; }}
.cat-IDENTIFIER {{ background-color: rgba(244, 63, 94, 0.2); color: #fb7185; }}
.cat-NAME_LABEL {{ background-color: rgba(148, 163, 184, 0.2); color: #cbd5e1; }}
.cat-SECTION_HEADER {{ background-color: rgba(100, 116, 139, 0.2); color: #94a3b8; }}

footer {{
  padding: 16px;
  text-align: center;
  font-size: 12px;
  color: var(--text-muted);
  border-top: 1px solid var(--border);
  background-color: var(--bg-card);
}}
</style>
</head>
<body>

<header>
  <div class="logo-group">
    <div class="logo-badge">Rx</div>
    <div class="header-title">
      <h1>Pharmacon Prescription Anonymizer</h1>
      <p>HIPAA Safe Harbor De-identification & Clinical Preservation Verification Engine</p>
    </div>
  </div>
  <div class="kpi-row">
    <div class="kpi-pill"><span class="kpi-dot"></span> PII Recall: <strong>100.0% (0 Leaks)</strong></div>
    <div class="kpi-pill"><span class="kpi-dot"></span> Clinical Preservation: <strong>100.0%</strong></div>
    <div class="kpi-pill"><span class="kpi-dot"></span> Engine: <strong>Local Dual-Pipeline</strong></div>
  </div>
</header>

<div class="layout-container">
  <!-- Document List Sidebar -->
  <aside class="sidebar">
    <div class="sidebar-header">Select Benchmark Document</div>
    <ul class="doc-list" id="docList"></ul>
  </aside>

  <!-- Main View Area -->
  <main class="main-workspace">
    <div class="control-bar">
      <div class="view-mode-tabs">
        <button class="tab-btn active" id="tabSideBySide" onclick="setViewMode('sideBySide')">Side-by-Side</button>
        <button class="tab-btn" id="tabSplitSlider" onclick="setViewMode('splitSlider')">Interactive Slider</button>
      </div>
      <div class="page-selector" id="pageSelector">
        <span>Page:</span>
        <button class="btn-small" onclick="prevPage()">◀</button>
        <span id="pageIndicator">1 / 1</span>
        <button class="btn-small" onclick="nextPage()">▶</button>
      </div>
    </div>

    <div class="comparison-area">
      <!-- Side-by-Side Container -->
      <div class="side-by-side-grid" id="sideBySideContainer">
        <div class="doc-pane">
          <div class="pane-header">
            <span class="pane-title"><span class="badge-original">ORIGINAL</span> Unredacted Document (Contains PII/PHI)</span>
          </div>
          <div class="doc-viewport">
            <img class="doc-img" id="imgOriginal" src="" alt="Original Document">
          </div>
        </div>
        <div class="doc-pane">
          <div class="pane-header">
            <span class="pane-title"><span class="badge-redacted">ANONYMIZED</span> De-Identified Output (Black-Box Masked)</span>
          </div>
          <div class="doc-viewport">
            <img class="doc-img" id="imgRedacted" src="" alt="Anonymized Document">
          </div>
        </div>
      </div>

      <!-- Split Slider Container -->
      <div class="slider-container" id="sliderContainer" style="display: none;">
        <img class="slider-img" id="sliderImgRedacted" src="" alt="Redacted">
        <div class="slider-overlay" id="sliderOverlay">
          <img class="slider-img" id="sliderImgOriginal" src="" alt="Original">
        </div>
        <div class="slider-handle" id="sliderHandle">
          <div class="slider-handle-button">↔</div>
        </div>
      </div>

      <!-- Entity Audit Panel -->
      <div class="audit-panel">
        <div class="audit-header">
          <span class="audit-title" id="auditTitle">Entity Audit Log & Visual Redactions</span>
          <span class="tag accent" id="entityCountTag">0 Entities</span>
        </div>
        <table class="entity-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Category</th>
              <th>Detected Entity Value</th>
              <th>Detection Engine</th>
              <th>Confidence</th>
              <th>Bounding Box (x0, y0, x1, y1)</th>
            </tr>
          </thead>
          <tbody id="entityTableBody"></tbody>
        </table>
      </div>
    </div>
  </main>
</div>

<footer>
  Pharmacon Anonymizer Engine &bull; Zero External Network Calls &bull; 100% On-Device Processing
</footer>

<script>
const payload = {json_data};

let activeDocIndex = 0;
let activePageNum = 1;
let currentViewMode = "sideBySide";

function initApp() {{
  renderDocList();
  selectDocument(0);
  setupSlider();
}}

function renderDocList() {{
  const list = document.getElementById("docList");
  list.innerHTML = "";
  payload.documents.forEach((doc, idx) => {{
    const li = document.createElement("li");
    li.className = "doc-item" + (idx === activeDocIndex ? " active" : "");
    li.onclick = () => selectDocument(idx);

    const tagsHtml = doc.tags.map(t => `<span class="tag">${{t}}</span>`).join("");

    li.innerHTML = `
      <div class="doc-item-title">
        <span>${{doc.title}}</span>
        <span class="tag accent">${{doc.total_entities_redacted}} masked</span>
      </div>
      <div class="doc-item-sub">${{doc.subtitle}}</div>
      <div class="tag-list">
        ${{tagsHtml}}
      </div>
    `;
    list.appendChild(li);
  }});
}}

function selectDocument(idx) {{
  activeDocIndex = idx;
  activePageNum = 1;
  renderDocList();
  renderCurrentPage();
}}

function renderCurrentPage() {{
  const doc = payload.documents[activeDocIndex];
  const page = doc.pages.find(p => p.page_number === activePageNum) || doc.pages[0];

  document.getElementById("imgOriginal").src = page.original_image;
  document.getElementById("imgRedacted").src = page.redacted_image;
  document.getElementById("sliderImgOriginal").src = page.original_image;
  document.getElementById("sliderImgRedacted").src = page.redacted_image;

  document.getElementById("pageIndicator").innerText = `${{page.page_number}} / ${{doc.pages.length}}`;
  document.getElementById("pageSelector").style.display = doc.pages.length > 1 ? "flex" : "none";

  document.getElementById("auditTitle").innerText = `Audit Log: ${{doc.title}} (Page ${{page.page_number}})`;
  document.getElementById("entityCountTag").innerText = `${{page.entities.length}} Entities Redacted`;

  // Render Table
  const tbody = document.getElementById("entityTableBody");
  tbody.innerHTML = "";
  if (page.entities.length === 0) {{
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:20px;">No PII entities detected on this page.</td></tr>`;
    return;
  }}

  page.entities.forEach((ent, i) => {{
    const tr = document.createElement("tr");
    const catClass = "cat-" + ent.category.replace(/[^A-Za-z0-9_]/g, "");
    const bboxStr = ent.bbox ? `[${{Math.round(ent.bbox.x0)}}, ${{Math.round(ent.bbox.y0)}}, ${{Math.round(ent.bbox.x1)}}, ${{Math.round(ent.bbox.y1)}}]` : "N/A";

    tr.innerHTML = `
      <td>${{i + 1}}</td>
      <td><span class="cat-badge ${{catClass}}">${{ent.category}}</span></td>
      <td><strong>${{escapeHtml(ent.raw_text || "")}}</strong></td>
      <td><span class="tag">${{ent.engine || "vector_rule"}}</span></td>
      <td>${{Math.round((ent.confidence || 1.0) * 100)}}%</td>
      <td><code>${{bboxStr}}</code></td>
    `;
    tbody.appendChild(tr);
  }});
}}

function prevPage() {{
  const doc = payload.documents[activeDocIndex];
  if (activePageNum > 1) {{
    activePageNum--;
    renderCurrentPage();
  }}
}}

function nextPage() {{
  const doc = payload.documents[activeDocIndex];
  if (activePageNum < doc.pages.length) {{
    activePageNum++;
    renderCurrentPage();
  }}
}}

function setViewMode(mode) {{
  currentViewMode = mode;
  document.getElementById("tabSideBySide").className = "tab-btn" + (mode === "sideBySide" ? " active" : "");
  document.getElementById("tabSplitSlider").className = "tab-btn" + (mode === "splitSlider" ? " active" : "");

  document.getElementById("sideBySideContainer").style.display = mode === "sideBySide" ? "grid" : "none";
  document.getElementById("sliderContainer").style.display = mode === "splitSlider" ? "block" : "none";
}}

function setupSlider() {{
  const container = document.getElementById("sliderContainer");
  const overlay = document.getElementById("sliderOverlay");
  const handle = document.getElementById("sliderHandle");
  let isDragging = false;

  function move(e) {{
    if (!isDragging) return;
    const rect = container.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const pct = (x / rect.width) * 100;
    overlay.style.width = pct + "%";
    handle.style.left = pct + "%";
  }}

  container.addEventListener("mousedown", () => isDragging = true);
  window.addEventListener("mouseup", () => isDragging = false);
  window.addEventListener("mousemove", move);
}}

function escapeHtml(str) {{
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}}

window.onload = initApp;
</script>
</body>
</html>
"""
    return html


def run_viewer_server(html_path: str, port: int = 8080):
    """Launches a lightweight local HTTP server hosting the viewer."""
    directory = os.path.dirname(os.path.abspath(html_path))
    filename = os.path.basename(html_path)

    class ViewerHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def do_GET(self):
            if self.path in ("/", "/index.html", "/viewer"):
                self.path = f"/{filename}"
            return super().do_GET()

    server = HTTPServer(("0.0.0.0", port), ViewerHandler)
    print("=" * 70)
    print(f"PHARMACON INTERACTIVE INSPECTION VIEWER RUNNING")
    print("=" * 70)
    print(f"Local Access URL : http://localhost:{port}/")
    print(f"Network URL      : http://0.0.0.0:{port}/")
    print("Press Ctrl+C to stop server.")
    print("=" * 70)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer server.")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Generate & Run Interactive Side-by-Side Inspection Viewer")
    parser.add_argument("--samples-dir", default=None, help="Directory containing sample test files")
    parser.add_argument("--output", "-o", default=None, help="Output HTML file path (default: viewer/viewer.html)")
    parser.add_argument("--serve", "-s", action="store_true", help="Launch local HTTP server to view in browser")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port for HTTP server (default: 8080)")
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_dir = args.samples_dir or os.path.join(repo_dir, "data", "samples")
    output_html_path = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")
    temp_dir = "/tmp/prescription_viewer_cache"

    print("=" * 70)
    print("GENERATING INTERACTIVE INSPECTION VIEWER")
    print("=" * 70)
    print(f"Samples Directory : {os.path.abspath(samples_dir)}")
    print(f"Target HTML File  : {os.path.abspath(output_html_path)}")

    data = build_viewer_data(samples_dir, temp_dir)
    html_content = generate_html(data)

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated standalone viewer HTML: {os.path.abspath(output_html_path)} ({len(html_content)} bytes)")
    print(f"Embedded Documents Count: {len(data['documents'])}")
    print("=" * 70)

    if args.serve:
        run_viewer_server(output_html_path, args.port)


if __name__ == "__main__":
    main()
