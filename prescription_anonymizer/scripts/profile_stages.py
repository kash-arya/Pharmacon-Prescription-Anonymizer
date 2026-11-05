import os
import sys
import time
import psutil

# Ensure path is set
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
for p in (current_dir, project_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

# Set RENDER_LOW_MEMORY=1 to replicate production
os.environ["RENDER_LOW_MEMORY"] = "1"

def get_total_rss_mb():
    p = psutil.Process()
    rss = p.memory_info().rss
    try:
        for child in p.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass
    return rss / (1024 * 1024)

# Profiling step recordings
step_timings = {}
step_rss = {}

def record_step(step_name, duration_ms, rss_mb):
    if step_name not in step_timings:
        step_timings[step_name] = 0.0
    step_timings[step_name] += duration_ms
    step_rss[step_name] = rss_mb

counters = {
    "tesseract_calls": 0,
    "barcode_calls": 0,
    "vlm_calls": 0,
    "normalize_ink_calls": 0,
    "clinical_gate_calls": 0
}

# 1. Instrument Tesseract OCR
import pymupdf
orig_get_textpage_ocr = pymupdf.Page.get_textpage_ocr
def instrumented_get_textpage_ocr(self, *args, **kwargs):
    counters["tesseract_calls"] += 1
    t0 = time.perf_counter()
    m0 = get_total_rss_mb()
    res = orig_get_textpage_ocr(self, *args, **kwargs)
    t1 = time.perf_counter()
    m1 = get_total_rss_mb()
    record_step("Tesseract OCR", (t1 - t0) * 1000, m1)
    return res
pymupdf.Page.get_textpage_ocr = instrumented_get_textpage_ocr

# 2. Instrument Barcode Sweep
import src.barcode_detector as bc_mod
orig_detect = bc_mod.detect_qr_and_barcodes
def instrumented_barcode(*args, **kwargs):
    counters["barcode_calls"] += 1
    t0 = time.perf_counter()
    res = orig_detect(*args, **kwargs)
    t1 = time.perf_counter()
    m1 = get_total_rss_mb()
    record_step("Barcode Sweep", (t1 - t0) * 1000, m1)
    return res
bc_mod.detect_qr_and_barcodes = instrumented_barcode

# 3. Instrument VLM Stamp Grounder
import src.vlm_grounder as vlm_mod
orig_ground = vlm_mod.MultimodalVisualGrounder.detect_visual_pii_from_bgr
def instrumented_ground(self, *args, **kwargs):
    counters["vlm_calls"] += 1
    t0 = time.perf_counter()
    res = orig_ground(self, *args, **kwargs)
    t1 = time.perf_counter()
    m1 = get_total_rss_mb()
    record_step("VLM Stamp Grounder", (t1 - t0) * 1000, m1)
    return res
vlm_mod.MultimodalVisualGrounder.detect_visual_pii_from_bgr = instrumented_ground

# 4. Instrument normalize_ink_contrast
import src.ocr_preprocess as ocr_mod
orig_norm = ocr_mod.normalize_ink_contrast
def instrumented_norm(img, gamma=1.6):
    counters["normalize_ink_calls"] += 1
    t0 = time.perf_counter()
    res = orig_norm(img, gamma=gamma)
    t1 = time.perf_counter()
    m1 = get_total_rss_mb()
    record_step("Pre-OCR Ink Normalization", (t1 - t0) * 1000, m1)
    return res
ocr_mod.normalize_ink_contrast = instrumented_norm

# 5. Instrument Clinical Gate
import src.document_validator as dv_mod
orig_eval_gate = dv_mod.evaluate_prescription_gate
def instrumented_gate(*args, **kwargs):
    counters["clinical_gate_calls"] += 1
    t0 = time.perf_counter()
    res = orig_eval_gate(*args, **kwargs)
    t1 = time.perf_counter()
    m1 = get_total_rss_mb()
    record_step("Clinical Gatekeeper", (t1 - t0) * 1000, m1)
    return res
dv_mod.evaluate_prescription_gate = instrumented_gate

# 6. Instrument Base64 Encoding
import api.server as server_mod
orig_pil_to_b64 = server_mod._pil_to_base64_data_uri
def instrumented_pil_to_b64(*args, **kwargs):
    t0 = time.perf_counter()
    res = orig_pil_to_b64(*args, **kwargs)
    t1 = time.perf_counter()
    m1 = get_total_rss_mb()
    record_step("Base64 Preview Encoding", (t1 - t0) * 1000, m1)
    return res
server_mod._pil_to_base64_data_uri = instrumented_pil_to_b64

orig_sbs_b64 = server_mod._generate_side_by_side_base64
def instrumented_sbs_b64(*args, **kwargs):
    t0 = time.perf_counter()
    res = orig_sbs_b64(*args, **kwargs)
    t1 = time.perf_counter()
    m1 = get_total_rss_mb()
    record_step("Side-by-Side Canvas Composite", (t1 - t0) * 1000, m1)
    return res
server_mod._generate_side_by_side_base64 = instrumented_sbs_b64

from api.server import app
from fastapi.testclient import TestClient

def profile_file(file_path):
    print("=" * 80)
    print(f"PROFILING TARGET: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / 1024:.1f} KB")
    print("=" * 80)

    # Reset metrics
    global step_timings, step_rss
    step_timings = {}
    step_rss = {}
    for k in counters:
        counters[k] = 0

    client = TestClient(app)
    
    m_init = get_total_rss_mb()

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    fname = os.path.basename(file_path)
    mime = "image/png" if file_path.endswith(".png") else "image/jpeg"

    t_req_start = time.perf_counter()
    res = client.post("/api/v1/anonymize/preview", files={"file": (fname, file_bytes, mime)})
    t_req_end = time.perf_counter()
    m_req_end = get_total_rss_mb()

    total_time_ms = (t_req_end - t_req_start) * 1000

    print(f"\nResponse HTTP Status : {res.status_code}")
    if res.status_code == 200:
        data = res.json()
        print(f"Triage Status        : {data.get('triage_status')}")
        print(f"Entities Redacted    : {data.get('total_entities_redacted')}")
        print(f"Confidence Score     : {data.get('confidence_score')}")
        print(f"Format Type          : {data.get('format_type')}")
        print(f"Engine               : {data.get('engine')}")
    else:
        print(f"Response Error       : {res.text[:300]}")

    print("\nSTAGE-BY-STAGE LATENCY & MEMORY PROFILE:")
    print("-" * 80)
    print(f"{'Internal Pipeline Stage':<35} | {'Wall-Clock (ms)':<16} | {'RSS (MB)':<12}")
    print("-" * 80)
    for step, dur in step_timings.items():
        rss_val = step_rss.get(step, m_req_end)
        print(f"{step:<35} | {dur:>13.2f} ms | {rss_val:>10.2f} MB")

    tracked_sum = sum(step_timings.values())
    untracked_time = max(0.0, total_time_ms - tracked_sum)
    print(f"{'NER/Layout/Redaction Burn (Other)':<35} | {untracked_time:>13.2f} ms | {m_req_end:>10.2f} MB")
    print("-" * 80)
    print(f"{'TOTAL PIPELINE PREVIEW':<35} | {total_time_ms:>13.2f} ms | {m_req_end:>10.2f} MB")
    print(f"Memory RSS Initial   : {m_init:.2f} MB")
    print(f"Memory RSS Peak/End  : {m_req_end:.2f} MB (Delta: {m_req_end - m_init:+.2f} MB)")
    print(f"Tesseract OCR Calls  : {counters['tesseract_calls']}")
    print(f"Barcode Detector     : {counters['barcode_calls']}")
    print(f"VLM Grounder         : {counters['vlm_calls']}")
    print(f"Clinical Gate Calls  : {counters['clinical_gate_calls']}")
    print(f"normalize_ink_contrast: {counters['normalize_ink_calls']}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    test_files = [
        "data/samples/extracted_images/sample_prescription_1_p1.png",
        "data/real_phone_prescriptions/Aryan March 26.jpg",
    ]
    for tf in test_files:
        p = os.path.join(current_dir, tf)
        if os.path.exists(p):
            profile_file(p)
        else:
            print(f"File not found: {p}")
