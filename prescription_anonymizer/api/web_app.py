"""
Pharmacon Web Application UI.

Provides standalone client-side HTML/JS application served directly by FastAPI.
Phase 2 Features:
- Dual-mode dropzone supporting mixed drag-and-drop of multiple files & full directories.
- Exhaustive recursive directory traversal with Chromium 100-entry batch while-loop fix.
- Zero-bandwidth client-side pre-filter (extension whitelist + 25MB cap).
- Collapsible warning banner for ignored/oversized files.
- Fault-tolerant sequential queue worker (concurrency = 1) with AbortController cancellation.
- Memory-safe Blob storage: Base64 data URIs converted immediately to Blobs, raw strings discarded.
- Batch Document Strip / Gallery with card selection and live queue indicators.
- Clinical Gate rejection cards (NOT_A_PRESCRIPTION) with direct "Actually a prescription? Force Anonymize" action.
- Interactive Side-by-Side Split Slider Inspector for active document.
- Redacted PII / PHI Entity Audit Log with JSON clipboard export.
- Anti-blocker bulk download with sequential 350ms delay excluding non-prescription/failed items.
- Full ObjectURL cleanup on session reset and beforeunload.
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
    .scrollbar-thin::-webkit-scrollbar {
      height: 6px;
      width: 6px;
    }
    .scrollbar-thin::-webkit-scrollbar-track {
      background: #0f172a;
    }
    .scrollbar-thin::-webkit-scrollbar-thumb {
      background: #334155;
      border-radius: 3px;
    }
    .scrollbar-thin::-webkit-scrollbar-thumb:hover {
      background: #475569;
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
        Document Inspector &amp; Queue
      </button>
      <button id="tab-batch-btn" onclick="switchMode('batch')" class="px-4 py-2 text-xs font-semibold rounded-lg bg-slate-900 text-slate-400 hover:text-slate-200 transition-all">
        Bulk Batch ZIP Processor
      </button>
    </div>

    <!-- VIEW 1: DOCUMENT INSPECTOR & QUEUE -->
    <div id="view-single" class="space-y-6">

      <!-- Upload Card -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <h2 class="text-lg font-semibold text-slate-100">Upload Prescriptions &amp; Clinical Documents</h2>
            <p class="text-xs text-slate-400">Supports Digital Vector EMRs (PDF), Scanned Papers (PDF/PNG), and Camera Photos (JPG). Mixed multi-file &amp; full directory drops supported.</p>
          </div>
          <div class="flex items-center gap-2">
            <button id="btn-browse-files" type="button" class="px-3.5 py-1.5 bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold rounded-lg text-xs shadow-md transition-all flex items-center gap-1.5">
              <span>📄</span> Select Files
            </button>
            <button id="btn-browse-folder" type="button" class="px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-bold rounded-lg text-xs transition-all flex items-center gap-1.5">
              <span>📁</span> Select Folder
            </button>
          </div>
        </div>

        <!-- Hidden inputs for multi-file and folder selection -->
        <input type="file" id="file-input" class="hidden" multiple accept=".pdf,.png,.jpg,.jpeg">
        <input type="file" id="folder-input" class="hidden" webkitdirectory directory multiple>

        <!-- Dropzone -->
        <div id="dropzone" class="border-2 border-dashed border-slate-700 hover:border-sky-500 rounded-xl p-8 text-center cursor-pointer transition-all bg-slate-950/40 hover:bg-slate-950/80">
          <div class="flex flex-col items-center justify-center gap-2 pointer-events-none">
            <svg class="w-10 h-10 text-sky-400 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
            </svg>
            <div class="text-sm font-medium text-slate-200">
              Drag &amp; Drop mixed prescription files or complete folders here, or click to browse
            </div>
            <div class="text-xs text-slate-500">
              PDF, PNG, JPG under 25 MB • Zero-Bandwidth Pre-Filter • Flat Memory Blob Store • Local Concurrency = 1
            </div>
          </div>
        </div>

        <!-- Ignored Files Collapsible Banner -->
        <div id="ignored-banner" class="hidden bg-amber-500/10 border border-amber-500/30 rounded-xl p-3 text-xs text-amber-300">
          <div class="flex items-center justify-between cursor-pointer" onclick="toggleIgnoredDetails()">
            <div class="flex items-center gap-2">
              <span class="text-amber-400 font-bold">⚠️ Notice:</span>
              <span id="ignored-summary-text">Ignored 0 unsupported or oversized file(s) — Only PDF, PNG, and JPG files under 25 MB are queued.</span>
            </div>
            <span id="ignored-toggle-btn" class="font-mono text-[11px] underline text-amber-400 hover:text-amber-300">View Details</span>
          </div>
          <div id="ignored-details" class="hidden mt-2 pt-2 border-t border-amber-500/20 max-h-40 overflow-y-auto space-y-1 font-mono text-[11px]">
            <!-- Injected dynamically -->
          </div>
        </div>

      </div>

      <!-- Queue Progress Card -->
      <div id="queue-card" class="hidden bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
        <div class="flex items-center justify-between flex-wrap gap-3">
          <div class="flex items-center gap-3">
            <div id="queue-spinner" class="hidden w-4 h-4 border-2 border-sky-400 border-t-transparent rounded-full animate-spin"></div>
            <span id="queue-status-text" class="font-mono text-xs font-semibold text-slate-200">Ready</span>
          </div>
          <div class="flex items-center gap-2">
            <button id="btn-cancel-queue" onclick="cancelProcessingQueue()" class="hidden px-3 py-1 bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/40 rounded-lg text-xs font-mono transition-colors">
              ⏹ Cancel Processing
            </button>
            <button id="btn-download-all" onclick="downloadAllRedacted()" class="hidden px-3.5 py-1 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold rounded-lg text-xs font-mono shadow-md transition-all flex items-center gap-1.5">
              <span>⬇</span> <span id="download-all-text">Download All Redacted (0)</span>
            </button>
            <button id="btn-clear-queue" onclick="clearSession()" class="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 border border-slate-700 rounded-lg text-xs font-mono transition-colors">
              ✕ Clear Batch
            </button>
          </div>
        </div>

        <!-- Progress Bar -->
        <div class="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
          <div id="queue-progress-bar" class="bg-gradient-to-r from-sky-500 to-emerald-400 h-2.5 transition-all duration-300" style="width: 0%;"></div>
        </div>

        <!-- Queue Stats Summary -->
        <div class="flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span id="queue-stats">Total: 0 | Redacted: 0 | Skipped: 0 | Failed: 0</span>
          <span id="queue-pct">0%</span>
        </div>
      </div>

      <!-- Batch Document Strip / Gallery -->
      <div id="gallery-container" class="hidden space-y-2">
        <div class="flex items-center justify-between">
          <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Document Gallery</h3>
          <span id="gallery-count" class="text-xs font-mono text-slate-500">0 Documents</span>
        </div>
        <div id="gallery-strip" class="flex gap-3 overflow-x-auto pb-2 scrollbar-thin">
          <!-- Cards injected dynamically -->
        </div>
      </div>

      <!-- Active Document Inspector Section -->
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
            <button id="download-btn" class="w-full bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold text-xs py-3 px-4 rounded-xl transition-all shadow-lg shadow-sky-500/20 disabled:opacity-40 disabled:cursor-not-allowed">
              Download Redacted File
            </button>
          </div>
        </div>

        <!-- Clinical Gate Rejection Card (Visible when triage_status === "NOT_A_PRESCRIPTION") -->
        <div id="gate-alert-card" class="hidden bg-gradient-to-r from-amber-950/50 to-slate-900 border-2 border-amber-500/60 rounded-2xl p-6 shadow-xl space-y-4">
          <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div class="space-y-1">
              <div class="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                <span>⚠️</span> NOT A PRESCRIPTION — SKIPPED
              </div>
              <h3 class="text-base font-bold text-slate-100">Clinical Gate Exclusion</h3>
              <p id="gate-reason-text" class="text-xs text-slate-300 font-mono">
                Document scored below clinical prescription gate threshold.
              </p>
              <p class="text-[11px] text-slate-400">
                Non-clinical files are automatically excluded to preserve system integrity, low latency, and zero-PII compliance.
              </p>
            </div>
            <button id="btn-force-anonymize-active" onclick="forceAnonymizeActiveItem()" class="px-5 py-3 bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold rounded-xl text-xs shadow-lg shadow-amber-500/20 transition-all flex items-center gap-2 whitespace-nowrap">
              <span>⚡</span> Actually a prescription? Force Anonymize
            </button>
          </div>
        </div>

        <!-- Failure Details Card (Visible when status === "failed") -->
        <div id="failed-alert-card" class="hidden bg-gradient-to-r from-red-950/50 to-slate-900 border-2 border-red-500/60 rounded-2xl p-6 shadow-xl space-y-3">
          <div class="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full text-xs font-bold bg-red-500/20 text-red-300 border border-red-500/40">
            <span>✕</span> PROCESSING FAILED
          </div>
          <h3 class="text-base font-bold text-slate-100">Document Processing Error</h3>
          <p id="failed-reason-text" class="text-xs text-red-300 font-mono">
            An error occurred while processing this document.
          </p>
        </div>

        <!-- Split Slider Inspection Viewer (Visible when document is redacted) -->
        <div id="split-inspection-card" class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div class="flex items-center justify-between mb-2 flex-wrap gap-2">
            <div>
              <h3 class="text-base font-semibold text-slate-100">Side-by-Side Interactive Split Inspector</h3>
              <p class="text-xs text-slate-400">Slide left or right to inspect redaction boundaries against the original prescription.</p>
            </div>
            <div class="flex items-center gap-3">
              <div id="page-nav-controls" class="flex items-center gap-1.5 bg-slate-950 border border-slate-800 px-2 py-1 rounded-lg text-xs">
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
        <div id="audit-table-card" class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-base font-semibold text-slate-100">Redacted PII / PHI Entity Audit Log</h3>
            <button id="copy-json-btn" onclick="copyAuditJson()" class="text-xs font-mono bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition-colors">
              📋 Copy Audit JSON
            </button>
          </div>
          <div class="overflow-x-auto max-h-72 overflow-y-auto scrollbar-thin">
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
            <div class="text-sm font-medium text-slate-200">Drag &amp; Drop prescription ZIP archive here, or <span class="text-emerald-400 hover:underline">browse ZIP</span></div>
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

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function formatFileSize(bytes) {
      if (!bytes || bytes === 0) return '0 B';
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // Memory-Safe Blob Storage: Converts incoming Base64 Data URI into a Blob and Object URL
    const trackedObjectUrls = new Set();

    function base64ToBlobUrl(base64Data, defaultMime = 'image/png') {
      if (!base64Data) return null;
      try {
        let mime = defaultMime;
        let b64 = base64Data;
        if (base64Data.startsWith('data:')) {
          const parts = base64Data.split(',');
          const mimeMatch = parts[0].match(/:(.*?);/);
          if (mimeMatch) mime = mimeMatch[1];
          b64 = parts[1];
        }
        const binStr = atob(b64);
        const len = binStr.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
          bytes[i] = binStr.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: mime });
        const url = URL.createObjectURL(blob);
        trackedObjectUrls.add(url);
        return url;
      } catch (err) {
        console.warn('base64ToBlobUrl conversion error:', err);
        return null;
      }
    }

    // Client-Side Pre-Filter: Extension Whitelist & 25MB Cap
    const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25 MB
    const ALLOWED_EXTS = new Set(['pdf', 'png', 'jpg', 'jpeg']);

    function preFilterFiles(fileList) {
      const validFiles = [];
      const ignoredFiles = [];

      for (const file of fileList) {
        const name = file.name || 'unnamed';
        const lowerName = name.toLowerCase();
        const extMatch = lowerName.match(/\\.([a-z0-9]+)$/);
        const ext = extMatch ? extMatch[1] : '';

        if (lowerName === '.ds_store' || lowerName === 'thumbs.db' || name.startsWith('.')) {
          ignoredFiles.push({ name, reason: 'Hidden / system file (ignored)' });
          continue;
        }
        if (file.size === 0) {
          ignoredFiles.push({ name, reason: 'Empty file (0 bytes)' });
          continue;
        }
        if (file.size > MAX_FILE_SIZE) {
          const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
          ignoredFiles.push({ name, reason: `Exceeds 25 MB cap (${sizeMb} MB)` });
          continue;
        }
        if (!ALLOWED_EXTS.has(ext)) {
          ignoredFiles.push({ name, reason: `Unsupported format (.${ext || 'unknown'})` });
          continue;
        }
        validFiles.push(file);
      }
      return { validFiles, ignoredFiles };
    }

    function toggleIgnoredDetails() {
      const details = document.getElementById('ignored-details');
      const btn = document.getElementById('ignored-toggle-btn');
      if (details.classList.contains('hidden')) {
        details.classList.remove('hidden');
        btn.innerText = 'Hide Details';
      } else {
        details.classList.add('hidden');
        btn.innerText = 'View Details';
      }
    }

    function renderIgnoredBanner(ignoredList) {
      const banner = document.getElementById('ignored-banner');
      const summaryText = document.getElementById('ignored-summary-text');
      const details = document.getElementById('ignored-details');

      if (!ignoredList || ignoredList.length === 0) {
        banner.classList.add('hidden');
        return;
      }

      summaryText.innerText = `Ignored ${ignoredList.length} unsupported or oversized file(s) — Only PDF, PNG, and JPG files under 25 MB are queued.`;
      details.innerHTML = ignoredList.map(item => `
        <div class="flex items-center justify-between py-1 border-b border-amber-500/10 last:border-0">
          <span class="truncate max-w-md text-amber-200">${escapeHtml(item.name)}</span>
          <span class="text-amber-400/80 text-[10px] uppercase font-bold">${escapeHtml(item.reason)}</span>
        </div>
      `).join('');

      banner.classList.remove('hidden');
    }

    // Exhaustive Recursive Directory Traversal with Chromium 100-Entry While-Loop Fix
    async function readAllDirectoryEntries(directoryReader) {
      const entries = [];
      const readBatch = () => {
        return new Promise((resolve) => {
          directoryReader.readEntries(
            (batch) => resolve(batch || []),
            () => resolve([])
          );
        });
      };
      // Chromium caps single readEntries() call at 100 items. Loop until empty array.
      while (true) {
        const batch = await readBatch();
        if (!batch || batch.length === 0) break;
        for (const entry of batch) {
          entries.push(entry);
        }
      }
      return entries;
    }

    async function getFilesFromEntry(entry) {
      if (entry.isFile) {
        return new Promise((resolve) => {
          entry.file(
            (f) => resolve(f ? [f] : []),
            () => resolve([])
          );
        });
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        const entries = await readAllDirectoryEntries(reader);
        const nested = await Promise.all(entries.map((e) => getFilesFromEntry(e)));
        return nested.flat();
      }
      return [];
    }

    async function extractFilesFromDataTransfer(dataTransfer) {
      const files = [];
      if (dataTransfer.items && dataTransfer.items.length > 0) {
        const promises = [];
        for (let i = 0; i < dataTransfer.items.length; i++) {
          const item = dataTransfer.items[i];
          const entry = (item.webkitGetAsEntry && item.webkitGetAsEntry()) || (item.getAsEntry && item.getAsEntry());
          if (entry) {
            promises.push(getFilesFromEntry(entry));
          } else {
            const file = item.getAsFile ? item.getAsFile() : null;
            if (file) files.push(file);
          }
        }
        if (promises.length > 0) {
          const nested = await Promise.all(promises);
          files.push(...nested.flat());
        }
      } else if (dataTransfer.files && dataTransfer.files.length > 0) {
        for (let i = 0; i < dataTransfer.files.length; i++) {
          files.push(dataTransfer.files[i]);
        }
      }
      return files;
    }

    // Queue State & Data Model
    let fileQueue = [];
    let isQueueRunning = false;
    let abortController = null;
    let selectedQueueIndex = -1;

    // DOM References
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const folderInput = document.getElementById('folder-input');
    const btnBrowseFiles = document.getElementById('btn-browse-files');
    const btnBrowseFolder = document.getElementById('btn-browse-folder');

    const queueCard = document.getElementById('queue-card');
    const queueStatusText = document.getElementById('queue-status-text');
    const queueProgressBar = document.getElementById('queue-progress-bar');
    const queueStats = document.getElementById('queue-stats');
    const queuePct = document.getElementById('queue-pct');
    const queueSpinner = document.getElementById('queue-spinner');
    const btnCancelQueue = document.getElementById('btn-cancel-queue');
    const btnDownloadAll = document.getElementById('btn-download-all');
    const downloadAllText = document.getElementById('download-all-text');

    const galleryContainer = document.getElementById('gallery-container');
    const galleryStrip = document.getElementById('gallery-strip');
    const galleryCount = document.getElementById('gallery-count');

    const resultSection = document.getElementById('result-section');
    const gateAlertCard = document.getElementById('gate-alert-card');
    const gateReasonText = document.getElementById('gate-reason-text');
    const failedAlertCard = document.getElementById('failed-alert-card');
    const failedReasonText = document.getElementById('failed-reason-text');
    const splitInspectionCard = document.getElementById('split-inspection-card');
    const auditTableCard = document.getElementById('audit-table-card');

    const kpiCount = document.getElementById('kpi-count');
    const kpiFormat = document.getElementById('kpi-format');
    const kpiJurisdiction = document.getElementById('kpi-jurisdiction');
    const kpiTriage = document.getElementById('kpi-triage');
    const kpiEngine = document.getElementById('kpi-engine');
    const downloadBtn = document.getElementById('download-btn');

    const splitBox = document.getElementById('split-box');
    const splitClip = document.getElementById('split-clip');
    const sliderHandle = document.getElementById('slider-handle');
    const imgOrig = document.getElementById('img-original');
    const imgRed = document.getElementById('img-redacted');
    const pageLabel = document.getElementById('page-label');

    // Wire Upload Buttons & Inputs
    btnBrowseFiles.addEventListener('click', () => fileInput.click());
    btnBrowseFolder.addEventListener('click', () => folderInput.click());
    dropzone.addEventListener('click', (e) => {
      if (e.target !== btnBrowseFiles && e.target !== btnBrowseFolder) {
        fileInput.click();
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleIncomingFiles(Array.from(e.target.files));
        fileInput.value = '';
      }
    });

    folderInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleIncomingFiles(Array.from(e.target.files));
        folderInput.value = '';
      }
    });

    // Drag-and-Drop Handling
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('border-sky-500', 'bg-slate-950');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('border-sky-500', 'bg-slate-950');
    });

    dropzone.addEventListener('drop', async (e) => {
      e.preventDefault();
      dropzone.classList.remove('border-sky-500', 'bg-slate-950');
      const files = await extractFilesFromDataTransfer(e.dataTransfer);
      if (files.length > 0) {
        handleIncomingFiles(files);
      }
    });

    // Handle Incoming Files
    async function handleIncomingFiles(rawFiles) {
      if (!rawFiles || rawFiles.length === 0) return;
      const { validFiles, ignoredFiles } = preFilterFiles(rawFiles);

      if (ignoredFiles.length > 0) {
        renderIgnoredBanner(ignoredFiles);
      } else {
        document.getElementById('ignored-banner').classList.add('hidden');
      }

      if (validFiles.length === 0) {
        if (ignoredFiles.length > 0) {
          alert(`All ${ignoredFiles.length} file(s) were ignored (unsupported extension or oversized > 25MB).`);
        }
        return;
      }

      for (const f of validFiles) {
        fileQueue.push({
          file: f,
          status: 'queued', // queued | processing | completed | not_a_prescription | failed | cancelled
          page: 0,
          origBlobUrl: null,
          redBlobUrl: null,
          downloadBlobUrl: null,
          localPreviewUrl: null,
          anonymizedFilename: '',
          triageStatus: '',
          gateScore: 0.0,
          confidenceScore: 0.0,
          totalEntities: 0,
          reviewReasons: [],
          gateReason: '',
          error: null,
          resultData: null
        });
      }

      renderQueueGallery();
      updateQueueProgress();

      if (!isQueueRunning) {
        runSequentialQueue();
      }
    }

    // Sequential Queue Worker (CONCURRENCY = 1)
    async function runSequentialQueue() {
      if (isQueueRunning) return;
      isQueueRunning = true;

      queueCard.classList.remove('hidden');
      btnCancelQueue.classList.remove('hidden');
      queueSpinner.classList.remove('hidden');

      const jurSelect = document.getElementById('jurisdiction-select');
      const jur = jurSelect ? jurSelect.value : 'AUTO';

      for (let i = 0; i < fileQueue.length; i++) {
        if (!isQueueRunning) break;
        const item = fileQueue[i];
        if (item.status !== 'queued') continue;

        item.status = 'processing';
        renderQueueGallery();
        updateQueueProgress();

        abortController = new AbortController();

        try {
          const formData = new FormData();
          formData.append('file', item.file);

          const res = await fetch(`/api/v1/anonymize/preview?page=0&jurisdiction=${encodeURIComponent(jur)}`, {
            method: 'POST',
            body: formData,
            signal: abortController.signal
          });

          const rawText = await res.text();
          let data = null;
          try {
            data = JSON.parse(rawText);
          } catch (_) {
            data = null;
          }

          if (res.status === 422 || (data && data.triage_status === 'NOT_A_PRESCRIPTION')) {
            item.status = 'not_a_prescription';
            item.triageStatus = 'NOT_A_PRESCRIPTION';
            item.gateScore = data && data.gate_score !== undefined ? data.gate_score : 0.0;
            item.confidenceScore = data && data.confidence_score !== undefined ? data.confidence_score : 0.0;
            item.reviewReasons = (data && data.review_reasons && data.review_reasons.length > 0)
              ? data.review_reasons
              : [(data && data.detail) || 'Not recognized as a medical prescription. Click "Force Anonymize" if this is a handwritten or non-standard slip.'];
            item.gateReason = item.reviewReasons[0];

            if (item.file.type.startsWith('image/')) {
              item.localPreviewUrl = URL.createObjectURL(item.file);
              trackedObjectUrls.add(item.localPreviewUrl);
            }
            item.resultData = data || { triage_status: 'NOT_A_PRESCRIPTION', review_reasons: item.reviewReasons };
          } else if (!res.ok) {
            const errMsg = (data && data.detail) ? data.detail : (res.status === 502 || res.status === 504 ? `Server memory limit exceeded (HTTP ${res.status}).` : `Server error (HTTP ${res.status}): ${rawText.slice(0, 140)}`);
            throw new Error(errMsg);
          } else if (data) {
            item.status = 'completed';
            item.triageStatus = data.triage_status || 'AUTO_APPROVED';
            item.confidenceScore = data.confidence_score !== undefined ? data.confidence_score : 1.0;
            item.totalEntities = data.total_entities_redacted || 0;
            item.anonymizedFilename = data.anonymized_filename || ('anonymized_' + item.file.name);

            // Memory-Safe Blob Conversion: discard Base64 strings immediately
            item.origBlobUrl = base64ToBlobUrl(data.original_image_base64 || data.original_image, 'image/png');
            item.redBlobUrl = base64ToBlobUrl(data.redacted_image_base64 || data.redacted_image, 'image/png');
            item.downloadBlobUrl = base64ToBlobUrl(data.download_file_base64 || data.data_uri, data.mime_type || 'application/pdf');

            delete data.original_image_base64;
            delete data.redacted_image_base64;
            delete data.side_by_side_image_base64;
            delete data.download_file_base64;
            delete data.original_image;
            delete data.redacted_image;
            delete data.data_uri;

            item.resultData = data;
          } else {
            throw new Error('Empty response received from server.');
          }
        } catch (err) {
          if (err.name === 'AbortError') {
            item.status = 'cancelled';
            item.error = 'Processing cancelled by user.';
            break;
          } else {
            item.status = 'failed';
            item.error = err.message || 'Unknown processing error';
          }
        } finally {
          abortController = null;
        }

        if (selectedQueueIndex === -1) {
          selectQueueItem(i);
        } else if (selectedQueueIndex === i) {
          renderActiveInspector(item);
        }

        renderQueueGallery();
        updateQueueProgress();
      }

      isQueueRunning = false;
      btnCancelQueue.classList.add('hidden');
      queueSpinner.classList.add('hidden');
      updateQueueProgress();
      renderQueueGallery();
    }

    // Force Anonymize Action for Single Item
    async function forceAnonymizeItem(index) {
      const item = fileQueue[index];
      if (!item) return;

      item.status = 'processing';
      item.error = null;
      renderQueueGallery();
      updateQueueProgress();
      if (selectedQueueIndex === index) {
        renderActiveInspector(item);
      }

      const jurSelect = document.getElementById('jurisdiction-select');
      const jur = jurSelect ? jurSelect.value : 'AUTO';
      const formData = new FormData();
      formData.append('file', item.file);

      try {
        const res = await fetch(`/api/v1/anonymize/preview?page=0&jurisdiction=${encodeURIComponent(jur)}&force_prescription=true`, {
          method: 'POST',
          body: formData
        });

        const rawText = await res.text();
        let data = null;
        try { data = JSON.parse(rawText); } catch (_) {}

        if (!res.ok) {
          throw new Error((data && data.detail) || `Server error (HTTP ${res.status}): ${rawText.slice(0, 140)}`);
        }

        if (data) {
          item.status = 'completed';
          item.triageStatus = data.triage_status || 'AUTO_APPROVED';
          item.confidenceScore = data.confidence_score !== undefined ? data.confidence_score : 1.0;
          item.totalEntities = data.total_entities_redacted || 0;
          item.anonymizedFilename = data.anonymized_filename || ('anonymized_' + item.file.name);

          item.origBlobUrl = base64ToBlobUrl(data.original_image_base64 || data.original_image, 'image/png');
          item.redBlobUrl = base64ToBlobUrl(data.redacted_image_base64 || data.redacted_image, 'image/png');
          item.downloadBlobUrl = base64ToBlobUrl(data.download_file_base64 || data.data_uri, data.mime_type || 'application/pdf');

          delete data.original_image_base64;
          delete data.redacted_image_base64;
          delete data.side_by_side_image_base64;
          delete data.download_file_base64;
          delete data.original_image;
          delete data.redacted_image;
          delete data.data_uri;

          item.resultData = data;
        }
      } catch (err) {
        item.status = 'failed';
        item.error = 'Force anonymize failed: ' + err.message;
      }

      renderQueueGallery();
      updateQueueProgress();
      if (selectedQueueIndex === index) {
        selectQueueItem(index);
      }
    }

    function forceAnonymizeActiveItem() {
      if (selectedQueueIndex >= 0) {
        forceAnonymizeItem(selectedQueueIndex);
      }
    }

    // Cancel Active Processing Queue
    function cancelProcessingQueue() {
      if (abortController) {
        abortController.abort();
      }
      isQueueRunning = false;
      for (const item of fileQueue) {
        if (item.status === 'queued') {
          item.status = 'cancelled';
        }
      }
      queueStatusText.innerText = 'Queue Cancelled by User';
      btnCancelQueue.classList.add('hidden');
      queueSpinner.classList.add('hidden');
      renderQueueGallery();
      updateQueueProgress();
    }

    // Progress Calculation & Counters
    function updateQueueProgress() {
      const total = fileQueue.length;
      if (total === 0) {
        queueCard.classList.add('hidden');
        galleryContainer.classList.add('hidden');
        return;
      }

      galleryContainer.classList.remove('hidden');
      galleryCount.innerText = `${total} Document${total === 1 ? '' : 's'}`;

      let completedCount = 0;
      let skippedCount = 0;
      let failedCount = 0;
      let queuedCount = 0;
      let processingCount = 0;
      let activeItemName = '';
      let activeIndex = -1;

      for (let i = 0; i < total; i++) {
        const s = fileQueue[i].status;
        if (s === 'completed') completedCount++;
        else if (s === 'not_a_prescription') skippedCount++;
        else if (s === 'failed') failedCount++;
        else if (s === 'queued') queuedCount++;
        else if (s === 'processing') {
          processingCount++;
          activeItemName = fileQueue[i].file.name;
          activeIndex = i + 1;
        }
      }

      const processed = completedCount + skippedCount + failedCount;
      const pct = Math.round((processed / total) * 100);

      queueProgressBar.style.width = pct + '%';
      queuePct.innerText = pct + '%';

      if (isQueueRunning && activeIndex > 0) {
        queueStatusText.innerText = `Processing ${activeIndex} of ${total}: ${activeItemName} [${pct}%]`;
      } else if (isQueueRunning) {
        queueStatusText.innerText = `Processing ${processed} of ${total} [${pct}%]...`;
      } else if (processed === total) {
        queueStatusText.innerText = `✓ Batch Processing Complete (${completedCount} Redacted, ${skippedCount} Skipped, ${failedCount} Failed)`;
      } else {
        queueStatusText.innerText = `Queue Idle (${processed} of ${total} processed)`;
      }

      queueStats.innerText = `Total: ${total} | Redacted: ${completedCount} | Skipped (Gate): ${skippedCount} | Failed: ${failedCount}`;

      // Update Bulk Download Button
      if (completedCount >= 1) {
        btnDownloadAll.classList.remove('hidden');
        downloadAllText.innerText = `Download All Redacted (${completedCount})`;
      } else {
        btnDownloadAll.classList.add('hidden');
      }
    }

    // Render Batch Document Gallery Strip
    function renderQueueGallery() {
      if (fileQueue.length === 0) {
        galleryStrip.innerHTML = '';
        return;
      }

      galleryStrip.innerHTML = fileQueue.map((item, idx) => {
        const isSelected = idx === selectedQueueIndex;
        const selectedClass = isSelected
          ? 'ring-2 ring-sky-400 bg-slate-800 shadow-sky-500/20 shadow-md'
          : 'bg-slate-900/90 hover:bg-slate-800/70 border border-slate-800';

        let badgeHtml = '';
        let previewHtml = '';

        if (item.status === 'completed') {
          badgeHtml = `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">✓ ${item.totalEntities} Redacted</span>`;
          if (item.redBlobUrl) {
            previewHtml = `<img src="${item.redBlobUrl}" class="w-full h-24 object-cover rounded bg-slate-950" alt="Thumb">`;
          } else {
            previewHtml = `<div class="w-full h-24 rounded bg-slate-950 flex items-center justify-center text-xs font-mono text-emerald-400">PDF REDACTED</div>`;
          }
        } else if (item.status === 'not_a_prescription') {
          badgeHtml = `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">⚠️ Gate: Skipped</span>`;
          if (item.localPreviewUrl) {
            previewHtml = `
              <div class="relative w-full h-24 rounded overflow-hidden bg-slate-950">
                <img src="${item.localPreviewUrl}" class="w-full h-full object-cover opacity-50" alt="Thumb">
                <div class="absolute inset-0 flex flex-col items-center justify-center p-1 bg-amber-950/60 text-center">
                  <span class="text-[9px] font-bold text-amber-300">NOT A PRESCRIPTION</span>
                  <button onclick="event.stopPropagation(); forceAnonymizeItem(${idx});" class="mt-1 px-1.5 py-0.5 rounded bg-amber-500 hover:bg-amber-400 text-slate-950 text-[9px] font-bold shadow">
                    Force Anonymize
                  </button>
                </div>
              </div>`;
          } else {
            previewHtml = `
              <div class="relative w-full h-24 rounded bg-slate-950 flex flex-col items-center justify-center p-2 text-center border border-amber-500/30">
                <span class="text-lg">📄</span>
                <span class="text-[9px] font-bold text-amber-300 mt-1">NON-PRESCRIPTION PDF</span>
                <button onclick="event.stopPropagation(); forceAnonymizeItem(${idx});" class="mt-1 px-1.5 py-0.5 rounded bg-amber-500 hover:bg-amber-400 text-slate-950 text-[9px] font-bold shadow">
                  Force Anonymize
                </button>
              </div>`;
          }
        } else if (item.status === 'processing') {
          badgeHtml = `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-sky-500/20 text-sky-400 border border-sky-500/30">● Processing</span>`;
          previewHtml = `
            <div class="w-full h-24 rounded bg-slate-950 flex flex-col items-center justify-center gap-1">
              <div class="w-5 h-5 border-2 border-sky-400 border-t-transparent rounded-full animate-spin"></div>
              <span class="text-[10px] font-mono text-slate-400">Sanitizing...</span>
            </div>`;
        } else if (item.status === 'failed') {
          badgeHtml = `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-400 border border-red-500/30">✕ Failed</span>`;
          previewHtml = `
            <div class="w-full h-24 rounded bg-slate-950 flex flex-col items-center justify-center gap-1 p-2 text-center border border-red-500/30">
              <span class="text-base text-red-400">⚠️</span>
              <span class="text-[9px] font-mono text-red-300 truncate max-w-full">${escapeHtml(item.error || 'Failed')}</span>
            </div>`;
        } else {
          badgeHtml = `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400">Queued</span>`;
          previewHtml = `
            <div class="w-full h-24 rounded bg-slate-950 flex items-center justify-center text-xs font-mono text-slate-500">
              ⏱ Waiting
            </div>`;
        }

        return `
          <div onclick="selectQueueItem(${idx})" class="flex-shrink-0 w-48 rounded-xl p-2.5 cursor-pointer transition-all ${selectedClass} space-y-2">
            ${previewHtml}
            <div class="space-y-1">
              <div class="text-xs font-medium text-slate-200 truncate" title="${escapeHtml(item.file.name)}">
                ${escapeHtml(item.file.name)}
              </div>
              <div class="flex items-center justify-between">
                <span class="text-[10px] font-mono text-slate-500">${formatFileSize(item.file.size)}</span>
                ${badgeHtml}
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    // Select Document from Gallery
    function selectQueueItem(index) {
      if (index < 0 || index >= fileQueue.length) return;
      selectedQueueIndex = index;
      renderQueueGallery();
      renderActiveInspector(fileQueue[index]);
    }

    // Render Active Document in Inspector & Split Viewer
    function renderActiveInspector(item) {
      if (!item) return;
      resultSection.classList.remove('hidden');

      const isCompleted = item.status === 'completed';
      const isSkipped = item.status === 'not_a_prescription';
      const isFailed = item.status === 'failed';

      // KPI Bar Values
      kpiCount.innerText = item.totalEntities || (item.resultData && item.resultData.total_entities_redacted) || 0;
      kpiFormat.innerText = (item.resultData && item.resultData.format_type) || (item.file.type.includes('pdf') ? 'PDF_DOCUMENT' : 'IMAGE_RASTER');
      kpiJurisdiction.innerText = (item.resultData && item.resultData.jurisdiction) || 'GLOBAL';
      kpiEngine.innerText = (item.resultData && item.resultData.engine) || (item.file.type.includes('pdf') ? 'Vector-PyMuPDF' : 'Raster-Tesseract');

      const conf = item.confidenceScore !== undefined ? item.confidenceScore.toFixed(2) : '1.00';
      const verdict = item.triageStatus || (isSkipped ? 'NOT_A_PRESCRIPTION' : (isCompleted ? 'AUTO_APPROVED' : '-'));
      kpiTriage.innerText = `${verdict} (${conf})`;

      if (isSkipped) {
        kpiTriage.className = 'text-xs font-bold font-mono text-amber-400 mt-2 truncate';
      } else if (isCompleted) {
        kpiTriage.className = 'text-xs font-bold font-mono text-emerald-400 mt-2 truncate';
      } else {
        kpiTriage.className = 'text-xs font-bold font-mono text-slate-400 mt-2 truncate';
      }

      // Download Button State
      if (isCompleted && item.downloadBlobUrl) {
        downloadBtn.disabled = false;
        downloadBtn.onclick = () => {
          const a = document.createElement('a');
          a.href = item.downloadBlobUrl;
          a.download = item.anonymizedFilename || ('anonymized_' + item.file.name);
          document.body.appendChild(a);
          a.click();
          a.remove();
        };
      } else {
        downloadBtn.disabled = true;
        downloadBtn.onclick = null;
      }

      // Card Visibility Logic
      if (isSkipped) {
        gateAlertCard.classList.remove('hidden');
        failedAlertCard.classList.add('hidden');
        splitInspectionCard.classList.add('hidden');
        auditTableCard.classList.add('hidden');

        gateReasonText.innerText = item.gateReason || 'Not recognized as a medical prescription (Gate score < 3.0).';
      } else if (isFailed) {
        gateAlertCard.classList.add('hidden');
        failedAlertCard.classList.remove('hidden');
        splitInspectionCard.classList.add('hidden');
        auditTableCard.classList.add('hidden');

        failedReasonText.innerText = item.error || 'Document failed during de-identification.';
      } else if (isCompleted) {
        gateAlertCard.classList.add('hidden');
        failedAlertCard.classList.add('hidden');
        splitInspectionCard.classList.remove('hidden');
        auditTableCard.classList.remove('hidden');

        // Populate Split Slider Images
        imgOrig.src = item.origBlobUrl || '';
        imgRed.src = item.redBlobUrl || '';
        pageLabel.innerText = `Page ${(item.page || 0) + 1}`;
        setSplitPos(50);

        // Populate Audit Table
        const tbody = document.getElementById('entities-tbody');
        tbody.innerHTML = '';
        const targets = (item.resultData && item.resultData.redacted_targets) || [];

        if (targets.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" class="py-4 text-center text-slate-500 font-sans">No sensitive entities detected in this document.</td></tr>';
        } else {
          targets.forEach((t, i) => {
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
        }
      } else {
        // Queued or Processing
        gateAlertCard.classList.add('hidden');
        failedAlertCard.classList.add('hidden');
        splitInspectionCard.classList.add('hidden');
        auditTableCard.classList.add('hidden');
      }
    }

    // Multi-page Pagination for Active Document
    async function changePage(delta) {
      if (selectedQueueIndex < 0) return;
      const item = fileQueue[selectedQueueIndex];
      if (!item || !item.file) return;

      const targetPage = Math.max(0, (item.page || 0) + delta);
      if (targetPage === (item.page || 0)) return;

      item.page = targetPage;
      pageLabel.innerText = `Page ${targetPage + 1}`;

      const jurSelect = document.getElementById('jurisdiction-select');
      const jur = jurSelect ? jurSelect.value : 'AUTO';
      const formData = new FormData();
      formData.append('file', item.file);

      try {
        const res = await fetch(`/api/v1/anonymize/preview?page=${targetPage}&jurisdiction=${encodeURIComponent(jur)}`, {
          method: 'POST',
          body: formData
        });
        if (res.ok) {
          const data = await res.json();
          item.origBlobUrl = base64ToBlobUrl(data.original_image_base64 || data.original_image, 'image/png');
          item.redBlobUrl = base64ToBlobUrl(data.redacted_image_base64 || data.redacted_image, 'image/png');
          item.resultData = data;
          imgOrig.src = item.origBlobUrl || '';
          imgRed.src = item.redBlobUrl || '';
          renderActiveInspector(item);
        }
      } catch (err) {
        console.warn('Page fetch error:', err);
      }
    }

    // Split Slider Drag Mechanics
    let isDragging = false;

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

    // Copy JSON Audit Log
    function copyAuditJson() {
      if (selectedQueueIndex < 0) return;
      const item = fileQueue[selectedQueueIndex];
      if (!item || !item.resultData) return;
      navigator.clipboard.writeText(JSON.stringify(item.resultData, null, 2)).then(() => {
        const btn = document.getElementById('copy-json-btn');
        const prev = btn.innerText;
        btn.innerText = '✓ Copied!';
        setTimeout(() => btn.innerText = prev, 2000);
      });
    }

    // Anti-Blocker Bulk Download (350ms delay between downloads, strictly excludes skipped/failed)
    async function downloadAllRedacted() {
      const validItems = fileQueue.filter((item) => item.status === 'completed' && item.downloadBlobUrl);
      if (validItems.length === 0) return;

      btnDownloadAll.disabled = true;
      const prevText = downloadAllText.innerText;

      try {
        for (let i = 0; i < validItems.length; i++) {
          const item = validItems[i];
          downloadAllText.innerText = `Downloading ${i + 1}/${validItems.length}...`;

          const a = document.createElement('a');
          a.href = item.downloadBlobUrl;
          a.download = item.anonymizedFilename || ('anonymized_' + item.file.name);
          document.body.appendChild(a);
          a.click();
          a.remove();

          if (i < validItems.length - 1) {
            await new Promise((r) => setTimeout(r, 350));
          }
        }
      } finally {
        downloadAllText.innerText = prevText;
        btnDownloadAll.disabled = false;
      }
    }

    // ObjectURL Cleanup & Session Reset
    function clearSession() {
      if (abortController) {
        abortController.abort();
      }
      isQueueRunning = false;

      for (const url of trackedObjectUrls) {
        try {
          URL.revokeObjectURL(url);
        } catch (_) {}
      }
      trackedObjectUrls.clear();

      fileQueue = [];
      selectedQueueIndex = -1;

      queueCard.classList.add('hidden');
      galleryContainer.classList.add('hidden');
      resultSection.classList.add('hidden');
      document.getElementById('ignored-banner').classList.add('hidden');
      fileInput.value = '';
      folderInput.value = '';

      renderQueueGallery();
    }

    window.addEventListener('beforeunload', () => {
      for (const url of trackedObjectUrls) {
        try {
          URL.revokeObjectURL(url);
        } catch (_) {}
      }
    });

    // Batch Processing Tab Logic (ZIP Archive)
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
        const res = await fetch(`/api/v1/anonymize/batch?jurisdiction=${encodeURIComponent(jur)}`, {
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
        setTimeout(() => window.URL.revokeObjectURL(url), 1000);

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
