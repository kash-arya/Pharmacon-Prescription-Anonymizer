#!/usr/bin/env python3
"""
Interactive Multinational Prescription Holdout Inspection Viewer.

Dynamically scans, anonymizes, and visualizes unseen multinational prescription
specimens from `data/unseen_holdout/` (including real Wikimedia Commons specimens
and user-supplied holdout files).

Generates `viewer/holdout_viewer.html` with:
- Zero-hardcoded specimen list (dynamic directory discovery).
- Jurisdiction detection badges (US, UK, AU, EU, IN, GLOBAL).
- Real-time CPU processing latencies (ms).
- Format type detection (SCANNED_RASTER vs VECTOR_DIGITAL).
- Side-by-side original vs redacted comparison and interactive split-slider.
- Entity bounding box inspection and forensic audit log table.
- Standalone local HTTP inspection server support (--serve --port 8081).
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import json
import base64
import io
import time
import argparse
import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import pymupdf
from PIL import Image

# Ensure project root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

try:
    from prescription_anonymizer.src.raster_checker import inspect_pdf_structure
    from prescription_anonymizer.src.vector_redactor import VectorPDFRedactor
    from prescription_anonymizer.src.image_redactor import LocalImageRedactor, resolve_tessdata_path
    from prescription_anonymizer.src.jurisdiction import JurisdictionDetector
except ImportError:
    from src.raster_checker import inspect_pdf_structure
    from src.vector_redactor import VectorPDFRedactor
    from src.image_redactor import LocalImageRedactor, resolve_tessdata_path
    from src.jurisdiction import JurisdictionDetector


SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}


def render_page_to_base64(doc_path: str, page_num: int, dpi: int = 120, max_dim: int = 1600, quality: int = 75) -> Tuple[str, int, int]:
    """Renders a PDF or image page to compressed base64 JPEG string along with pixel dimensions."""
    ext = os.path.splitext(doc_path)[1].lower()
    is_image = ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

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


def discover_holdout_files(holdout_dir: str) -> List[str]:
    """Dynamically locates all supported prescription files in holdout directory."""
    files = []
    if not os.path.exists(holdout_dir):
        return []
    for root, _, filenames in os.walk(holdout_dir):
        for fn in sorted(filenames):
            if fn.startswith(".") or fn == "holdout_manifest.json":
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, fn))
    return sorted(files)


def humanize_title(filename: str) -> str:
    """Creates a readable human title from a filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    clean = base.replace("commons_", "").replace("_", " ").strip()
    return clean.title()


def build_holdout_data(holdout_dir: str, temp_out_dir: str) -> Dict[str, Any]:
    """Scans and anonymizes all holdout specimens, generating visual comparison payload."""
    tessdata_dir = resolve_tessdata_path()
    os.makedirs(temp_out_dir, exist_ok=True)

    holdout_files = discover_holdout_files(holdout_dir)
    print(f"[+] Found {len(holdout_files)} holdout files in {holdout_dir}")

    documents = []

    for idx, input_path in enumerate(holdout_files, 1):
        fname = os.path.basename(input_path)
        base, ext = os.path.splitext(fname)
        out_fname = f"{base}_holdout_redacted{ext}"
        output_path = os.path.join(temp_out_dir, out_fname)

        print(f"  [{idx}/{len(holdout_files)}] Processing: {fname}...")
        t0 = time.time()
        struct = inspect_pdf_structure(input_path)
        fmt = struct["format_type"]

        if fmt == "VECTOR_DIGITAL":
            redactor = VectorPDFRedactor()
            report = redactor.redact(input_path, output_path, jurisdiction="AUTO")
            engine_name = "Native Vector Redactor (PyMuPDF)"
        else:
            redactor = LocalImageRedactor(tessdata_dir=tessdata_dir)
            report = redactor.redact(input_path, output_path, jurisdiction="AUTO")
            engine_name = "Local OCR Anchor Engine (PyMuPDF + Tesseract)"

        t1 = time.time()
        latency_ms = round((t1 - t0) * 1000.0, 1)

        detected_jur = report.get("jurisdiction", "GLOBAL")
        total_redacted = report.get("total_entities_redacted", len(report.get("redacted_targets", [])))

        is_image = ext.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
        page_count = 1 if is_image else len(pymupdf.open(input_path))

        pages_data = []
        for p_idx in range(1, page_count + 1):
            orig_b64, w_orig, h_orig = render_page_to_base64(input_path, p_idx)
            red_b64, w_red, h_red = render_page_to_base64(output_path, p_idx)

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

        title = humanize_title(fname)
        tags = [detected_jur, fmt.replace("_", " ")]
        if is_image:
            tags.append("Raster Specimen")
        else:
            tags.append("PDF Document")

        documents.append({
            "id": f"holdout_{idx}_{base[:20]}",
            "title": title,
            "filename": fname,
            "subtitle": f"{fmt} • {latency_ms} ms",
            "jurisdiction": detected_jur,
            "latency_ms": latency_ms,
            "format_type": fmt,
            "engine": engine_name,
            "category": "Unseen Holdout Specimen",
            "tags": tags,
            "total_entities_redacted": total_redacted,
            "pages": pages_data
        })

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_documents": len(documents),
        "total_redactions": sum(d["total_entities_redacted"] for d in documents),
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
<title>Pharmacon Prescription Anonymizer - Multinational Holdout Inspector</title>
<style>
:root {{
  --bg-dark: #090d16;
  --bg-card: #131c2e;
  --bg-card-hover: #1e293b;
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --accent: #38bdf8;
  --accent-glow: rgba(56, 189, 248, 0.25);
  --success: #34d399;
  --warning: #fbbf24;
  --danger: #f87171;
  --border: #1e293b;
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
  background-color: rgba(19, 28, 46, 0.95);
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
  font-size: 16px;
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
  align-items: center;
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

.sidebar {{
  width: 360px;
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
  display: flex;
  justify-content: space-between;
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
  align-items: center;
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
  padding: 2px 8px;
  border-radius: 12px;
  background-color: rgba(51, 65, 85, 0.8);
  color: #cbd5e1;
}}

.jur-badge {{
  font-size: 10px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 10px;
}}

.jur-US {{ background-color: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; }}
.jur-UK {{ background-color: #581c87; color: #d8b4fe; border: 1px solid #a855f7; }}
.jur-AU {{ background-color: #78350f; color: #fde68a; border: 1px solid #f59e0b; }}
.jur-EU {{ background-color: #064e3b; color: #a7f3d0; border: 1px solid #10b981; }}
.jur-IN {{ background-color: #7c2d12; color: #fed7aa; border: 1px solid #f97316; }}
.jur-GLOBAL {{ background-color: #1e293b; color: #cbd5e1; border: 1px solid #475569; }}

.main-content {{
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  background-color: var(--bg-dark);
}}

.toolbar {{
  background-color: var(--bg-card);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  top: 0;
  z-index: 10;
}}

.tabs {{
  display: flex;
  gap: 8px;
}}

.tab-btn {{
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-muted);
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.2s ease;
}}

.tab-btn:hover {{
  color: var(--text-main);
  border-color: var(--accent);
}}

.tab-btn.active {{
  background-color: var(--accent);
  color: #000;
  font-weight: 600;
  border-color: var(--accent);
}}

.pagination {{
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 13px;
}}

.page-btn {{
  background-color: var(--bg-dark);
  border: 1px solid var(--border);
  color: var(--text-main);
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
}}

.page-btn:disabled {{
  opacity: 0.3;
  cursor: not-allowed;
}}

.stage-wrapper {{
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}}

.view-side-by-side {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}}

.view-card {{
  background-color: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
}}

.view-card-header {{
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  color: var(--text-muted);
}}

.image-canvas-container {{
  position: relative;
  background-color: #020617;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 480px;
}}

.image-canvas-container img {{
  max-width: 100%;
  max-height: 720px;
  display: block;
  object-fit: contain;
}}

.view-slider-container {{
  background-color: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  display: none;
}}

.split-slider {{
  position: relative;
  max-width: 800px;
  margin: 0 auto;
  overflow: hidden;
  user-select: none;
  border-radius: 8px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.5);
}}

.split-slider img {{
  width: 100%;
  height: auto;
  display: block;
}}

.slider-overlay {{
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  width: 50%;
  overflow: hidden;
  z-index: 2;
}}

.slider-overlay img {{
  position: absolute;
  top: 0;
  left: 0;
  max-width: none;
  height: 100%;
}}

.slider-handle {{
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 3px;
  background-color: var(--accent);
  z-index: 3;
  cursor: ew-resize;
  box-shadow: 0 0 12px var(--accent);
}}

.slider-handle::after {{
  content: "↔";
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 30px;
  height: 30px;
  background-color: var(--accent);
  color: #000;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 14px;
}}

.audit-table-card {{
  background-color: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
}}

.audit-table-header {{
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  display: flex;
  justify-content: space-between;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}

th, td {{
  text-align: left;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
}}

th {{
  color: var(--text-muted);
  font-weight: 600;
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.5px;
}}

tr:hover {{
  background-color: var(--bg-card-hover);
}}

.cat-badge {{
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  background-color: rgba(56, 189, 248, 0.15);
  color: var(--accent);
}}

code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
  background-color: rgba(15, 23, 42, 0.8);
  padding: 2px 6px;
  border-radius: 4px;
}}
</style>
</head>
<body>

<header>
  <div class="logo-group">
    <div class="logo-badge">HOLDOUT</div>
    <div class="header-title">
      <h1>Pharmacon Multinational Holdout Inspector</h1>
      <p>Zero-Trust Blind Evaluation on Unseen Global Prescription Specimens</p>
    </div>
  </div>
  <div class="kpi-row">
    <div class="kpi-pill">
      <span class="kpi-dot"></span>
      <span>Total Documents: <strong id="kpiTotalDocs">{data['total_documents']}</strong></span>
    </div>
    <div class="kpi-pill">
      <span>Total Entities Redacted: <strong id="kpiTotalRedactions">{data['total_redactions']}</strong></span>
    </div>
  </div>
</header>

<div class="layout-container">
  <div class="sidebar">
    <div class="sidebar-header">
      <span>Discovered Specimens</span>
      <span id="docCountBadge">{len(data['documents'])} Files</span>
    </div>
    <ul class="doc-list" id="docList"></ul>
  </div>

  <div class="main-content">
    <div class="toolbar">
      <div class="tabs">
        <button id="tabSideBySide" class="tab-btn active" onclick="setViewMode('sideBySide')">Side-by-Side View</button>
        <button id="tabSplitSlider" class="tab-btn" onclick="setViewMode('splitSlider')">Interactive Split Slider</button>
      </div>
      <div class="pagination">
        <button class="page-btn" id="prevPageBtn" onclick="prevPage()">‹ Prev</button>
        <span id="pageIndicator">Page 1 / 1</span>
        <button class="page-btn" id="nextPageBtn" onclick="nextPage()">Next ›</button>
      </div>
    </div>

    <div class="stage-wrapper">
      <!-- Side by Side View -->
      <div id="sideBySideContainer" class="view-side-by-side">
        <div class="view-card">
          <div class="view-card-header">
            <span>Original Unredacted Specimen</span>
            <span id="origDimensions">-</span>
          </div>
          <div class="image-canvas-container">
            <img id="imgOriginal" alt="Original Specimen">
          </div>
        </div>

        <div class="view-card">
          <div class="view-card-header">
            <span>De-Identified Specimen (100% Cryptographic Burn-In)</span>
            <span id="redDimensions">-</span>
          </div>
          <div class="image-canvas-container">
            <img id="imgRedacted" alt="Redacted Specimen">
          </div>
        </div>
      </div>

      <!-- Split Slider View -->
      <div id="sliderContainer" class="view-slider-container">
        <div class="split-slider" id="splitSliderWrapper">
          <img id="sliderRedactedImg" alt="Redacted Base">
          <div class="slider-overlay" id="sliderOverlay">
            <img id="sliderOriginalImg" alt="Original Overlay">
          </div>
          <div class="slider-handle" id="sliderHandle"></div>
        </div>
      </div>

      <!-- Forensic Audit & Bounding Box Table -->
      <div class="audit-table-card">
        <div class="audit-table-header">
          <div>
            <span>Redacted Entities on Current Page</span>
            <span style="font-size: 12px; color: var(--text-muted); font-weight: normal; margin-left: 8px;" id="entityCountSub"></span>
          </div>
          <div id="activeDocBadgeGroup"></div>
        </div>

        <table>
          <thead>
            <tr>
              <th style="width: 40px;">#</th>
              <th style="width: 180px;">Category</th>
              <th>Detected Raw Value</th>
              <th style="width: 140px;">Detection Engine</th>
              <th style="width: 90px;">Confidence</th>
              <th style="width: 220px;">Target Bounding Box</th>
            </tr>
          </thead>
          <tbody id="entitiesTbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
const payload = {json_data};
let activeDocIndex = 0;
let activePageNum = 1;
let currentViewMode = "sideBySide";

function initApp() {{
  renderSidebar();
  if (payload.documents.length > 0) {{
    selectDocument(0);
  }}
  setupSlider();
}}

function renderSidebar() {{
  const list = document.getElementById("docList");
  list.innerHTML = "";
  payload.documents.forEach((doc, idx) => {{
    const li = document.createElement("li");
    li.className = "doc-item" + (idx === activeDocIndex ? " active" : "");
    li.onclick = () => selectDocument(idx);

    const jurClass = "jur-" + (doc.jurisdiction || "GLOBAL");
    li.innerHTML = `
      <div class="doc-item-title">
        <span>${{escapeHtml(doc.title)}}</span>
        <span class="jur-badge ${{jurClass}}">${{doc.jurisdiction}}</span>
      </div>
      <div class="doc-item-sub">${{doc.filename}} • <strong>${{doc.latency_ms}} ms</strong></div>
      <div class="tag-list">
        <span class="tag">${{doc.format_type}}</span>
        <span class="tag">${{doc.total_entities_redacted}} masked</span>
      </div>
    `;
    list.appendChild(li);
  }});
}}

function selectDocument(idx) {{
  activeDocIndex = idx;
  activePageNum = 1;
  const items = document.querySelectorAll(".doc-item");
  items.forEach((item, i) => {{
    item.className = "doc-item" + (i === idx ? " active" : "");
  }});
  renderCurrentPage();
}}

function renderCurrentPage() {{
  const doc = payload.documents[activeDocIndex];
  if (!doc) return;
  const page = doc.pages[activePageNum - 1];
  if (!page) return;

  document.getElementById("pageIndicator").innerText = `Page ${{activePageNum}} / ${{doc.pages.length}}`;
  document.getElementById("prevPageBtn").disabled = (activePageNum === 1);
  document.getElementById("nextPageBtn").disabled = (activePageNum === doc.pages.length);

  document.getElementById("imgOriginal").src = page.original_image;
  document.getElementById("imgRedacted").src = page.redacted_image;
  document.getElementById("sliderRedactedImg").src = page.redacted_image;
  document.getElementById("sliderOriginalImg").src = page.original_image;

  document.getElementById("origDimensions").innerText = `${{page.width}} × ${{page.height}} px`;
  document.getElementById("redDimensions").innerText = `${{page.width}} × ${{page.height}} px`;

  const jurClass = "jur-" + (doc.jurisdiction || "GLOBAL");
  document.getElementById("activeDocBadgeGroup").innerHTML = `
    <span class="jur-badge ${{jurClass}}">${{doc.jurisdiction}} Profile</span>
    <span class="tag" style="margin-left: 8px;">Latency: ${{doc.latency_ms}} ms</span>
    <span class="tag" style="margin-left: 4px;">Engine: ${{doc.engine}}</span>
  `;

  document.getElementById("entityCountSub").innerText = `(${{page.entities.length}} targets identified on page ${{activePageNum}})`;

  const tbody = document.getElementById("entitiesTbody");
  tbody.innerHTML = "";
  if (page.entities.length === 0) {{
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="6" style="text-align: center; color: var(--text-muted); padding: 24px;">Zero PII entities detected on this page.</td>`;
    tbody.appendChild(tr);
    return;
  }}

  page.entities.forEach((ent, i) => {{
    const tr = document.createElement("tr");
    const b = ent.bbox || {{}};
    const bboxStr = `[${{Math.round(b.x0 || 0)}}, ${{Math.round(b.y0 || 0)}}, ${{Math.round(b.x1 || 0)}}, ${{Math.round(b.y1 || 0)}}]`;
    tr.innerHTML = `
      <td>${{i + 1}}</td>
      <td><span class="cat-badge">${{escapeHtml(ent.category)}}</span></td>
      <td><strong>${{escapeHtml(ent.raw_text || "")}}</strong></td>
      <td><span class="tag">${{escapeHtml(ent.engine || "detector")}}</span></td>
      <td>${{Math.round((ent.confidence || 0.98) * 100)}}%</td>
      <td><code>${{bboxStr}}</code></td>
    `;
    tbody.appendChild(tr);
  }});
}}

function prevPage() {{
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
  const container = document.getElementById("splitSliderWrapper");
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
  if (!str) return "";
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}}

window.onload = initApp;
</script>
</body>
</html>
"""
    return html


def run_viewer_server(html_path: str, port: int = 8081):
    """Launches a lightweight local HTTP server hosting the holdout viewer."""
    directory = os.path.dirname(os.path.abspath(html_path))
    filename = os.path.basename(html_path)

    class ViewerHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def do_GET(self):
            if self.path in ("/", "/index.html", "/viewer", "/holdout"):
                self.path = f"/{filename}"
            return super().do_GET()

    server = HTTPServer(("0.0.0.0", port), ViewerHandler)
    print("=" * 75)
    print("PHARMACON MULTINATIONAL HOLDOUT INSPECTION VIEWER RUNNING")
    print("=" * 75)
    print(f"Local Access URL : http://localhost:{port}/")
    print(f"Network URL      : http://0.0.0.0:{port}/")
    print("Press Ctrl+C to stop server.")
    print("=" * 75)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping holdout viewer server.")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Multinational Holdout Inspection Viewer")
    parser.add_argument("--holdout-dir", default=None, help="Directory containing holdout prescription files")
    parser.add_argument("--output", "-o", default=None, help="Output HTML file path (default: viewer/holdout_viewer.html)")
    parser.add_argument("--serve", "-s", action="store_true", help="Launch local HTTP server to view in browser")
    parser.add_argument("--port", "-p", type=int, default=8081, help="Port for HTTP server (default: 8081)")
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    holdout_dir = args.holdout_dir or os.path.join(repo_dir, "data", "unseen_holdout")
    output_html_path = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdout_viewer.html")
    temp_dir = "/tmp/holdout_viewer_cache"

    print("=" * 75)
    print("PHARMACON: GENERATING MULTINATIONAL HOLDOUT INSPECTION VIEWER")
    print("=" * 75)
    print(f"Holdout Directory : {os.path.abspath(holdout_dir)}")
    print(f"Target HTML File  : {os.path.abspath(output_html_path)}")

    data = build_holdout_data(holdout_dir, temp_dir)
    html_content = generate_html(data)

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[+] Generated standalone holdout viewer HTML: {os.path.abspath(output_html_path)}")
    print(f"[+] Total Holdout Documents Processed: {len(data['documents'])}")
    print(f"[+] Total Target Entities Masked: {data['total_redactions']}")
    print("=" * 75)

    if args.serve:
        run_viewer_server(output_html_path, args.port)


if __name__ == "__main__":
    main()
