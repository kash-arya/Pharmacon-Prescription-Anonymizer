"""
Pharmacon Web Application UI.

Provides standalone client-side HTML/JS application served directly by FastAPI.
Features:
- Dual modes: Single Document Split-Slider Inspector & Bulk Batch ZIP Processor.
- Drag-and-drop prescription uploader (PDF/JPG/PNG/ZIP).
- Live interactive side-by-side split slider with coordinate visualization.
- Multi-page pagination support for intake records.
- JSON Audit Log exporter with single-click clipboard copy.
- Bulk ZIP de-identification with instant download and tabular reporting.
"""

def get_web_app_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pharmacon - Medical Prescription De-Identifier</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .split-container {
      position: relative;
      overflow: hidden;
      user-select: none;
    }
    .split-overlay {
      position: absolute;
      top: 0;
      left: 0;
      height: 100%;
      overflow: hidden;
    }
    .slider-handle {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 3px;
      background: #38bdf8;
      cursor: ew-resize;
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.7);
    }
    .slider-handle::after {
      content: '↔';
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 28px;
      height: 28px;
      background: #0284c7;
      color: white;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 13px;
      font-weight: bold;
      border: 2px solid white;
    }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 antialiased min-h-screen flex flex-col font-sans">

  <!-- Top Navigation Bar -->
  <header class="bg-slate-900/80 backdrop-blur border-b border-slate-800 sticky top-0 z-50 px-6 py-4">
    <div class="max-w-7xl mx-auto flex items-center justify-between flex-wrap gap-4">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-sky-500 to-emerald-400 flex items-center justify-center font-bold text-white shadow-lg shadow-sky-500/20">
          Rx
        </div>
        <div>
          <h1 class="font-bold text-base text-slate-100">Pharmacon Anonymizer</h1>
          <p class="text-xs text-slate-400">Multi-National Healthcare De-Identification Engine</p>
        </div>
      </div>
      <div class="flex items-center gap-3 flex-wrap">
        <!-- Jurisdiction Selector -->
        <div class="flex items-center gap-1.5 bg-slate-950 border border-slate-700 px-2.5 py-1 rounded-lg">
          <label for="jurisdiction-select" class="text-xs text-slate-400 font-medium">Profile:</label>
          <select id="jurisdiction-select" class="bg-transparent text-xs text-sky-400 font-medium focus:outline-none cursor-pointer">
            <option value="AUTO" class="bg-slate-900 text-slate-200">🌐 Auto-Detect</option>
            <option value="US" class="bg-slate-900 text-slate-200">🇺🇸 United States (HIPAA / DEA / NPI)</option>
            <option value="UK" class="bg-slate-900 text-slate-200">🇬🇧 United Kingdom (NHS / GMC / FP10)</option>
            <option value="AU" class="bg-slate-900 text-slate-200">🇦🇺 Australia (PBS / Medicare / AHPRA)</option>
            <option value="EU" class="bg-slate-900 text-slate-200">🇪🇺 European Union (GDPR / Muster 16)</option>
            <option value="IN" class="bg-slate-900 text-slate-200">🇮🇳 India (DPDP Act / ABHA / NMC)</option>
            <option value="GLOBAL" class="bg-slate-900 text-slate-200">🌍 Global Unified</option>
          </select>
        </div>
        <a href="/docs" target="_blank" class="text-xs font-mono bg-slate-800 hover:bg-slate-700 text-sky-400 px-3 py-1.5 rounded-lg border border-slate-700 transition-colors">
          Swagger /docs
        </a>
        <a href="/health" target="_blank" class="text-xs font-mono bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 px-3 py-1.5 rounded-lg border border-emerald-500/30 transition-colors">
          ● Health: Online
        </a>
      </div>
    </div>
  </header>

  <!-- Main Content Container -->
  <main class="flex-1 max-w-7xl mx-auto w-full p-6 space-y-6">

    <!-- Mode Selector Tabs -->
    <div class="flex items-center gap-2 border-b border-slate-800 pb-3">
      <button id="tab-single-btn" onclick="switchMode('single')" class="px-4 py-2 text-xs font-semibold rounded-lg bg-sky-500 text-slate-950 transition-all shadow-sm">
        Single Document Inspector
      </button>
      <button id="tab-batch-btn" onclick="switchMode('batch')" class="px-4 py-2 text-xs font-semibold rounded-lg bg-slate-900 text-slate-400 hover:text-slate-200 transition-all">
        Bulk Batch ZIP Processor
      </button>
    </div>

    <!-- VIEW 1: SINGLE DOCUMENT INSPECTOR -->
    <div id="view-single" class="space-y-6">

      <!-- Upload Card -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div class="flex flex-col md:flex-row items-center justify-between gap-4 mb-4">
          <div>
            <h2 class="text-lg font-semibold text-slate-100">Upload Prescription Document</h2>
            <p class="text-xs text-slate-400">Supports Digital Vector EMRs (PDF), Scanned Papers (PDF/PNG), and Smartphone Camera Photos (JPG).</p>
          </div>
          <div id="status-badge" class="hidden text-xs font-mono px-3 py-1.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300">
            Ready
          </div>
        </div>

        <!-- Dropzone -->
        <div id="dropzone" class="border-2 border-dashed border-slate-700 hover:border-sky-500 rounded-xl p-8 text-center cursor-pointer transition-all bg-slate-950/40 hover:bg-slate-950/80">
          <input type="file" id="file-input" class="hidden" accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp">
          <div class="flex flex-col items-center justify-center gap-2">
            <svg class="w-10 h-10 text-sky-400 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
            </svg>
            <div class="text-sm font-medium text-slate-200">Drag & Drop prescription here, or <span class="text-sky-400 hover:underline">browse file</span></div>
            <div class="text-xs text-slate-500">PDF, PNG, JPG up to 50MB (Processed 100% locally on CPU)</div>
          </div>
        </div>
      </div>

      <!-- Viewer & Inspector Section (Hidden until file uploaded) -->
      <div id="result-section" class="hidden space-y-6">

        <!-- Summary KPI Bar -->
        <div class="grid grid-cols-2 md:grid-cols-6 gap-4">
          <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div class="text-xs text-slate-400">Entities Redacted</div>
            <div id="kpi-count" class="text-2xl font-bold text-emerald-400 mt-1">0</div>
          </div>
          <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div class="text-xs text-slate-400">Document Format</div>
            <div id="kpi-format" class="text-sm font-mono text-sky-400 mt-2 truncate">-</div>
          </div>
          <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div class="text-xs text-slate-400">Jurisdiction Profile</div>
            <div id="kpi-jurisdiction" class="text-sm font-mono text-amber-400 mt-2 truncate">-</div>
          </div>
          <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div class="text-xs text-slate-400">Triage Verdict</div>
            <div id="kpi-triage" class="text-xs font-bold font-mono text-emerald-400 mt-2 truncate">-</div>
          </div>
          <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div class="text-xs text-slate-400">Processing Engine</div>
            <div id="kpi-engine" class="text-xs font-mono text-purple-400 mt-2 truncate">-</div>
          </div>
          <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center justify-center">
            <button id="download-btn" class="w-full bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold text-xs py-3 px-4 rounded-xl transition-all shadow-lg shadow-sky-500/20">
              Download Redacted File
            </button>
          </div>
        </div>

        <!-- Split Slider Inspection Viewer -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
          <div class="flex items-center justify-between mb-4 flex-wrap gap-2">
            <div>
              <h3 class="text-base font-semibold text-slate-100">Side-by-Side Interactive Split Inspector</h3>
              <p class="text-xs text-slate-400">Slide left or right to inspect redaction boundaries against the original prescription.</p>
            </div>
            <div class="flex items-center gap-3">
              <div class="flex items-center gap-1.5 bg-slate-950 border border-slate-800 px-2 py-1 rounded-lg text-xs">
                <button id="prev-page-btn" onclick="changePage(-1)" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300">‹ Prev</button>
                <span id="page-label" class="font-mono text-slate-400">Page 1</span>
                <button id="next-page-btn" onclick="changePage(1)" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300">Next ›</button>
              </div>
              <div class="flex items-center gap-2 text-xs font-mono">
                <span class="text-slate-400">Original (Left)</span>
                <span class="text-sky-400">|</span>
                <span class="text-emerald-400">Redacted (Right)</span>
              </div>
            </div>
          </div>

          <div class="flex justify-center bg-slate-950 rounded-xl p-4 border border-slate-800">
            <div id="split-box" class="split-container relative rounded-lg border border-slate-700 max-w-full inline-block">
              <!-- Redacted Image (Base) -->
              <img id="img-redacted" class="block max-h-[750px] w-auto pointer-events-none" alt="Redacted">
              <!-- Original Image (Clipped Overlay) -->
              <div id="split-clip" class="split-overlay pointer-events-none" style="width: 50%;">
                <img id="img-original" class="block max-h-[750px] w-auto max-w-none pointer-events-none" alt="Original">
              </div>
              <!-- Interactive Handle -->
              <div id="slider-handle" class="slider-handle" style="left: 50%;"></div>
            </div>
          </div>
        </div>

        <!-- Masked Entities Audit Table -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-base font-semibold text-slate-100">Redacted PII / PHI Entity Audit Log</h3>
            <button id="copy-json-btn" onclick="copyAuditJson()" class="text-xs font-mono bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition-colors">
              📋 Copy Audit JSON
            </button>
          </div>
          <div class="overflow-x-auto max-h-72 overflow-y-auto">
            <table class="w-full text-left text-xs">
              <thead class="bg-slate-950 text-slate-400 uppercase text-[10px] tracking-wider sticky top-0">
                <tr>
                  <th class="py-2.5 px-4">#</th>
                  <th class="py-2.5 px-4">Category</th>
                  <th class="py-2.5 px-4">Detected Text / Payload</th>
                  <th class="py-2.5 px-4">Engine</th>
                  <th class="py-2.5 px-4">Coordinates (x0, y0, x1, y1)</th>
                </tr>
              </thead>
              <tbody id="entities-tbody" class="divide-y divide-slate-800 font-mono text-[11px]">
                <!-- Injected dynamically -->
              </tbody>
            </table>
          </div>
        </div>

      </div>

    </div>

    <!-- VIEW 2: BULK BATCH ZIP PROCESSOR -->
    <div id="view-batch" class="hidden space-y-6">

      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h2 class="text-lg font-semibold text-slate-100 mb-1">Bulk ZIP Batch Anonymizer</h2>
        <p class="text-xs text-slate-400 mb-4">Upload a .zip archive containing any combination of digital PDF, scanned PDF, or JPG prescriptions. The server will sanitize all files in parallel and return a bundled ZIP archive with compliance summaries.</p>

        <!-- Batch Dropzone -->
        <div id="batch-dropzone" class="border-2 border-dashed border-slate-700 hover:border-emerald-500 rounded-xl p-8 text-center cursor-pointer transition-all bg-slate-950/40 hover:bg-slate-950/80">
          <input type="file" id="batch-file-input" class="hidden" accept=".zip">
          <div class="flex flex-col items-center justify-center gap-2">
            <svg class="w-10 h-10 text-emerald-400 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4"/>
            </svg>
            <div class="text-sm font-medium text-slate-200">Drag & Drop prescription ZIP archive here, or <span class="text-emerald-400 hover:underline">browse ZIP</span></div>
            <div class="text-xs text-slate-500">ZIP archives containing multiple documents (up to 500MB)</div>
          </div>
        </div>

        <div id="batch-status-badge" class="hidden mt-4 text-xs font-mono px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">
          Ready
        </div>
      </div>

      <!-- Batch Result Card -->
      <div id="batch-result-section" class="hidden bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div class="flex items-center justify-between">
          <div>
            <h3 class="text-base font-semibold text-slate-100">Batch Processing Summary</h3>
            <p class="text-xs text-slate-400">All sanitized documents have been packaged into your downloaded archive.</p>
          </div>
          <div id="batch-kpis" class="text-xs font-mono text-emerald-400"></div>
        </div>
      </div>

    </div>

  </main>

  <script>
    // Tab switching
    function switchMode(mode) {
      const isSingle = mode === 'single';
      document.getElementById('view-single').classList.toggle('hidden', !isSingle);
      document.getElementById('view-batch').classList.toggle('hidden', isSingle);

      const singleBtn = document.getElementById('tab-single-btn');
      const batchBtn = document.getElementById('tab-batch-btn');

      if (isSingle) {
        singleBtn.className = 'px-4 py-2 text-xs font-semibold rounded-lg bg-sky-500 text-slate-950 transition-all shadow-sm';
        batchBtn.className = 'px-4 py-2 text-xs font-semibold rounded-lg bg-slate-900 text-slate-400 hover:text-slate-200 transition-all';
      } else {
        batchBtn.className = 'px-4 py-2 text-xs font-semibold rounded-lg bg-emerald-500 text-slate-950 transition-all shadow-sm';
        singleBtn.className = 'px-4 py-2 text-xs font-semibold rounded-lg bg-slate-900 text-slate-400 hover:text-slate-200 transition-all';
      }
    }

    // Single Document State
    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const resultSection = document.getElementById('result-section');
    const statusBadge = document.getElementById('status-badge');
    const splitBox = document.getElementById('split-box');
    const splitClip = document.getElementById('split-clip');
    const sliderHandle = document.getElementById('slider-handle');
    const imgOrig = document.getElementById('img-original');
    const imgRed = document.getElementById('img-redacted');
    const downloadBtn = document.getElementById('download-btn');
    const pageLabel = document.getElementById('page-label');

    let currentFile = null;
    let currentPage = 0;
    let lastAuditData = null;
    let isDragging = false;

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('border-sky-500', 'bg-slate-950'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('border-sky-500', 'bg-slate-950'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('border-sky-500', 'bg-slate-950');
      if (e.dataTransfer.files.length > 0) processUpload(e.dataTransfer.files[0], 0);
    });
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) processUpload(e.target.files[0], 0);
    });

    async function changePage(delta) {
      if (!currentFile) return;
      const targetPage = Math.max(0, currentPage + delta);
      if (targetPage === currentPage) return;
      await processUpload(currentFile, targetPage);
    }

    async function processUpload(file, page = 0) {
      currentFile = file;
      currentPage = page;
      pageLabel.innerText = `Page ${page + 1}`;

      statusBadge.classList.remove('hidden');
      statusBadge.innerText = `Processing page ${page + 1}...`;
      statusBadge.className = 'text-xs font-mono px-3 py-1.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30';

      const formData = new FormData();
      formData.append('file', file);
      const jur = document.getElementById('jurisdiction-select').value;

      try {
        const res = await fetch(`/api/v1/anonymize/preview?page=${page}&jurisdiction=${jur}`, {
          method: 'POST',
          body: formData
        });

        const rawText = await res.text();
        let data = null;
        try {
          data = JSON.parse(rawText);
        } catch (_) {
          data = null;
        }

        if (!res.ok) {
          const errMsg = (data && data.detail) ? data.detail : (res.status === 502 || res.status === 504 ? `Server resource limit reached (HTTP ${res.status}). The container was restarted due to memory constraints.` : `Server error (HTTP ${res.status}): ${rawText.slice(0, 160)}`);
          throw new Error(errMsg);
        }

        if (!data) {
          throw new Error(`Invalid response received from server (HTTP ${res.status}).`);
        }
        lastAuditData = data;
        renderResults(data);

        statusBadge.innerText = `Redaction Successful (${data.jurisdiction || 'GLOBAL'})`;
        statusBadge.className = 'text-xs font-mono px-3 py-1.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
      } catch (err) {
        statusBadge.innerText = 'Failed: ' + err.message;
        statusBadge.className = 'text-xs font-mono px-3 py-1.5 rounded-full bg-red-500/20 text-red-300 border border-red-500/30';
      }
    }

    function renderResults(data) {
      document.getElementById('kpi-count').innerText = data.total_entities_redacted;
      document.getElementById('kpi-format').innerText = data.format_type;
      document.getElementById('kpi-jurisdiction').innerText = data.jurisdiction || 'GLOBAL';
      document.getElementById('kpi-engine').innerText = data.engine;

      const triageEl = document.getElementById('kpi-triage');
      if (triageEl) {
        const status = data.triage_status || 'AUTO_APPROVED';
        const score = data.confidence_score !== undefined ? data.confidence_score.toFixed(2) : '1.00';
        triageEl.innerText = `${status} (${score})`;
        if (status === 'AUTO_APPROVED') {
          triageEl.className = 'text-xs font-bold font-mono text-emerald-400 mt-2 truncate';
        } else {
          triageEl.className = 'text-xs font-bold font-mono text-amber-400 mt-2 truncate';
        }
      }

      imgOrig.src = data.original_image;
      imgRed.src = data.redacted_image;

      const tbody = document.getElementById('entities-tbody');
      tbody.innerHTML = '';
      (data.redacted_targets || []).forEach((t, i) => {
        const row = document.createElement('tr');
        const b = t.bbox || {};
        const coords = `[${Math.round(b.x0 || 0)}, ${Math.round(b.y0 || 0)}, ${Math.round(b.x1 || 0)}, ${Math.round(b.y1 || 0)}]`;
        const cat = escapeHtml(t.category);
        const raw = escapeHtml(t.raw_text || '-');
        const eng = escapeHtml(t.engine);
        row.innerHTML = `
          <td class="py-2 px-4 text-slate-500">${i + 1}</td>
          <td class="py-2 px-4 font-semibold text-sky-400">${cat}</td>
          <td class="py-2 px-4 text-slate-300 truncate max-w-xs font-sans">${raw}</td>
          <td class="py-2 px-4 text-purple-400">${eng}</td>
          <td class="py-2 px-4 text-slate-400 font-mono text-[10px]">${coords}</td>
        `;
        tbody.appendChild(row);
      });

      resultSection.classList.remove('hidden');

      downloadBtn.onclick = async () => {
        const fd = new FormData();
        fd.append('file', currentFile);
        const jur = document.getElementById('jurisdiction-select').value;
        const fileRes = await fetch(`/api/v1/anonymize?return_type=file&jurisdiction=${jur}`, { method: 'POST', body: fd });
        const blob = await fileRes.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = data.filename || 'anonymized_prescription.pdf';
        document.body.appendChild(a);
        a.click();
        a.remove();
      };

      setSplitPos(50);
    }

    function copyAuditJson() {
      if (!lastAuditData) return;
      navigator.clipboard.writeText(JSON.stringify(lastAuditData, null, 2)).then(() => {
        const btn = document.getElementById('copy-json-btn');
        const prev = btn.innerText;
        btn.innerText = '✓ Copied!';
        setTimeout(() => btn.innerText = prev, 2000);
      });
    }

    function setSplitPos(pct) {
      pct = Math.max(0, Math.min(100, pct));
      splitClip.style.width = pct + '%';
      sliderHandle.style.left = pct + '%';
    }

    function onPointerMove(e) {
      if (!isDragging) return;
      const rect = splitBox.getBoundingClientRect();
      const x = (e.clientX || (e.touches && e.touches[0].clientX)) - rect.left;
      const pct = (x / rect.width) * 100;
      setSplitPos(pct);
    }

    sliderHandle.addEventListener('mousedown', () => isDragging = true);
    sliderHandle.addEventListener('touchstart', () => isDragging = true);
    window.addEventListener('mouseup', () => isDragging = false);
    window.addEventListener('touchend', () => isDragging = false);
    window.addEventListener('mousemove', onPointerMove);
    window.addEventListener('touchmove', onPointerMove);

    // Batch Processing State
    const batchDropzone = document.getElementById('batch-dropzone');
    const batchFileInput = document.getElementById('batch-file-input');
    const batchStatusBadge = document.getElementById('batch-status-badge');
    const batchResultSection = document.getElementById('batch-result-section');

    batchDropzone.addEventListener('click', () => batchFileInput.click());
    batchDropzone.addEventListener('dragover', (e) => { e.preventDefault(); batchDropzone.classList.add('border-emerald-500', 'bg-slate-950'); });
    batchDropzone.addEventListener('dragleave', () => batchDropzone.classList.remove('border-emerald-500', 'bg-slate-950'));
    batchDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      batchDropzone.classList.remove('border-emerald-500', 'bg-slate-950');
      if (e.dataTransfer.files.length > 0) processBatchUpload(e.dataTransfer.files[0]);
    });
    batchFileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) processBatchUpload(e.target.files[0]);
    });

    async function processBatchUpload(file) {
      batchStatusBadge.classList.remove('hidden');
      batchStatusBadge.innerText = 'Processing batch archive (sanitizing documents in parallel)...';
      batchStatusBadge.className = 'mt-4 text-xs font-mono px-3 py-2 rounded-lg bg-amber-500/20 text-amber-300 border border-amber-500/30';

      const formData = new FormData();
      formData.append('file', file);
      const jur = document.getElementById('jurisdiction-select').value;

      try {
        const res = await fetch(`/api/v1/anonymize/batch?jurisdiction=${jur}`, {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          const rawText = await res.text();
          let err = null;
          try {
            err = JSON.parse(rawText);
          } catch (_) {
            err = null;
          }
          const errMsg = (err && err.detail) ? err.detail : (res.status === 502 || res.status === 504 ? `Server resource limit reached (HTTP ${res.status}). The container was restarted due to memory constraints.` : `Server error (HTTP ${res.status}): ${rawText.slice(0, 160)}`);
          throw new Error(errMsg);
        }

        const totalDocs = res.headers.get('x-total-documents') || '0';
        const totalEntities = res.headers.get('x-total-entities-redacted') || '0';

        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `sanitized_batch_${Date.now()}.zip`;
        document.body.appendChild(a);
        a.click();
        a.remove();

        batchStatusBadge.innerText = `✓ Batch Redaction Complete: Sanitized ${totalDocs} documents (${totalEntities} entities masked). Archive downloaded.`;
        batchStatusBadge.className = 'mt-4 text-xs font-mono px-3 py-2 rounded-lg bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';

        batchResultSection.classList.remove('hidden');
        document.getElementById('batch-kpis').innerText = `Total Documents: ${totalDocs} | Total Redactions: ${totalEntities}`;
      } catch (err) {
        batchStatusBadge.innerText = 'Batch Processing Failed: ' + err.message;
        batchStatusBadge.className = 'mt-4 text-xs font-mono px-3 py-2 rounded-lg bg-red-500/20 text-red-300 border border-red-500/30';
      }
    }
  </script>
</body>
</html>
"""
