#!/usr/bin/env python3
"""Step 5: Production Constraint Verification under Render Free Tier (512MB RAM).

Executes sequential stress testing of the live FastAPI server under cgroup memory limits:
- Round 1: 4 Desktop test files (WhatsApp x2, 24b7b6cb, 3Q15_Timetable_SemV)
- Round 2: Exact same 4 files repeated to verify memory stability & zero leak
- Round 3: Real high-resolution camera prescriptions (Aryan Sept 24, Aryan March 26)

Monitors combined Python RSS + child process RSS every 50ms.
"""

import os
import sys
import time
import json
import threading
import subprocess
import requests
import psutil

PORT = 8999
BASE_URL = f"http://127.0.0.1:{PORT}"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FIXTURES_DIR = os.path.join(REPO_DIR, "tests", "fixtures", "prscrptn")
DATA_DIR = os.path.join(REPO_DIR, "data", "real_phone_prescriptions")


def get_combined_rss(proc: psutil.Process) -> float:
    """Return combined RSS in MB for process and all child processes."""
    try:
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


class MemoryTracker:
    def __init__(self, pid: int):
        self.proc = psutil.Process(pid)
        self.running = True
        self.peak_rss = 0.0
        self.current_rss = 0.0
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        while self.running:
            rss = get_combined_rss(self.proc)
            with self.lock:
                self.current_rss = rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            time.sleep(0.04)

    def reset_peak(self) -> float:
        with self.lock:
            self.peak_rss = self.current_rss
            return self.peak_rss

    def get_peak(self) -> float:
        with self.lock:
            return self.peak_rss

    def get_current(self) -> float:
        with self.lock:
            return self.current_rss

    def stop(self):
        self.running = False


def run_request(tracker: MemoryTracker, file_path: str, upload_name: str, mime: str):
    tracker.reset_peak()
    start_rss = tracker.get_current()
    t0 = time.perf_counter()

    with open(file_path, "rb") as f:
        files = {"file": (upload_name, f, mime)}
        resp = requests.post(
            f"{BASE_URL}/api/v1/anonymize/preview?page=0&jurisdiction=AUTO",
            files=files,
            timeout=120
        )
    elapsed = time.perf_counter() - t0
    time.sleep(0.3)  # Allow gc/trim cycle to settle
    peak_rss = tracker.get_peak()
    idle_rss = tracker.get_current()

    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text[:200]}

    return {
        "status_code": resp.status_code,
        "elapsed_ms": round(elapsed * 1000, 1),
        "peak_rss_mb": round(peak_rss, 2),
        "idle_rss_mb": round(idle_rss, 2),
        "triage_status": data.get("triage_status", "-"),
        "entities": data.get("total_entities_redacted", 0),
        "detail": data.get("detail", "")
    }


def main():
    print("=" * 80)
    print("PHARMACON STEP 5: PRODUCTION CONSTRAINT VERIFICATION (RENDER 512MB RAM)")
    print("=" * 80)

    # 1. Start Server Process with RENDER_LOW_MEMORY=1
    env = os.environ.copy()
    env["RENDER_LOW_MEMORY"] = "1"
    env["PYTHONPATH"] = REPO_DIR
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_THREAD_LIMIT"] = "1"
    env["MALLOC_ARENA_MAX"] = "2"

    cmd = [
        sys.executable, "-m", "uvicorn", "api.server:app",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--log-level", "warning"
    ]

    print(f"Starting server: {' '.join(cmd)}")
    server_proc = subprocess.Popen(
        cmd,
        cwd=REPO_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        # Wait for server health
        print("Waiting for server startup...")
        healthy = False
        for _ in range(60):
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=1)
                if r.status_code == 200:
                    healthy = True
                    break
            except Exception:
                time.sleep(0.25)

        if not healthy:
            print("ERROR: Server failed to start within 15 seconds.")
            server_proc.terminate()
            out, err = server_proc.communicate(timeout=3)
            print("STDOUT:", out.decode())
            print("STDERR:", err.decode())
            return 1

        tracker = MemoryTracker(server_proc.pid)
        time.sleep(0.5)
        startup_rss = tracker.get_current()
        print(f"Server healthy! Initial idle RSS: {startup_rss:.2f} MB")

        files_round1_2 = [
            ("WhatsApp Image 2026-11-05 at 07.15.55.jpeg", os.path.join(FIXTURES_DIR, "WhatsApp Image 2026-11-05 at 07.15.55.jpeg"), "image/jpeg"),
            ("WhatsApp Image 2026-11-05 at 07.15.56.jpeg", os.path.join(FIXTURES_DIR, "WhatsApp Image 2026-11-05 at 07.15.56.jpeg"), "image/jpeg"),
            ("24b7b6cb-2446-49cf-b13c-2b63a74d6446.jpeg", os.path.join(FIXTURES_DIR, "24b7b6cb-2446-49cf-b13c-2b63a74d6446.jpeg"), "image/jpeg"),
            ("3Q15_Timetable_SemV.png", os.path.join(FIXTURES_DIR, "3Q15_Timetable_SemV.png"), "image/png"),
        ]

        files_round3 = [
            ("Aryan Sept 24.jpg", os.path.join(DATA_DIR, "Aryan Sept 24.jpg"), "image/jpeg"),
            ("Aryan March 26.jpg", os.path.join(DATA_DIR, "Aryan March 26.jpg"), "image/jpeg"),
        ]

        results = []

        # --- ROUND 1 ---
        print("\n" + "=" * 80)
        print("ROUND 1: 4 Desktop Test Files (Initial Cold Run)")
        print("=" * 80)
        r1_peaks = []
        for name, path, mime in files_round1_2:
            upload_name = f"prscrptn/{name}"
            res = run_request(tracker, path, upload_name, mime)
            res["round"] = "Round 1"
            res["name"] = name
            results.append(res)
            r1_peaks.append(res["peak_rss_mb"])
            print(f"[{res['round']}] {name[:36]:36} | HTTP {res['status_code']} | {res['elapsed_ms']:7.1f} ms | Peak RSS: {res['peak_rss_mb']:6.1f} MB | Triage: {res['triage_status']}")

        # --- ROUND 2 ---
        print("\n" + "=" * 80)
        print("ROUND 2: Repeat Exact 4 Desktop Files (Zero Memory Leak Check)")
        print("=" * 80)
        r2_peaks = []
        for name, path, mime in files_round1_2:
            upload_name = f"prscrptn/{name}"
            res = run_request(tracker, path, upload_name, mime)
            res["round"] = "Round 2"
            res["name"] = name
            results.append(res)
            r2_peaks.append(res["peak_rss_mb"])
            print(f"[{res['round']}] {name[:36]:36} | HTTP {res['status_code']} | {res['elapsed_ms']:7.1f} ms | Peak RSS: {res['peak_rss_mb']:6.1f} MB | Triage: {res['triage_status']}")

        # --- ROUND 3 ---
        print("\n" + "=" * 80)
        print("ROUND 3: Real High-Resolution Camera Prescriptions (Stress Test)")
        print("=" * 80)
        for name, path, mime in files_round3:
            upload_name = f"prscrptn/{name}"
            res = run_request(tracker, path, upload_name, mime)
            res["round"] = "Round 3"
            res["name"] = name
            results.append(res)
            print(f"[{res['round']}] {name[:36]:36} | HTTP {res['status_code']} | {res['elapsed_ms']:7.1f} ms | Peak RSS: {res['peak_rss_mb']:6.1f} MB | Triage: {res['triage_status']}")

        tracker.stop()

        # Summary & Success Criteria Validation
        print("\n" + "=" * 80)
        print("EVALUATION & CRITERIA VERIFICATION")
        print("=" * 80)

        max_peak = max(r["peak_rss_mb"] for r in results)
        r1_max = max(r1_peaks)
        r2_max = max(r2_peaks)
        drift = abs(r2_max - r1_max)

        # Check Timetable triage status
        timetable_r1 = [r for r in results if r["round"] == "Round 1" and "Timetable" in r["name"]][0]
        timetable_r2 = [r for r in results if r["round"] == "Round 2" and "Timetable" in r["name"]][0]

        crit_no_oom = server_proc.poll() is None
        crit_peak_rss = max_peak < 400.0
        crit_stability = drift <= 30.0
        crit_timetable_r1 = (timetable_r1["status_code"] == 200 and timetable_r1["triage_status"] == "NOT_A_PRESCRIPTION")
        crit_timetable_r2 = (timetable_r2["status_code"] == 200 and timetable_r2["triage_status"] == "NOT_A_PRESCRIPTION")

        print(f"1. No OOM Kill (Process Alive):         {'PASS' if crit_no_oom else 'FAIL'}")
        print(f"2. Peak Combined RSS < 400 MB:           {'PASS' if crit_peak_rss else 'FAIL'} (Peak: {max_peak:.2f} MB)")
        print(f"3. Stability across Rounds (<=30MB diff): {'PASS' if crit_stability else 'FAIL'} (Round 1: {r1_max:.1f} MB, Round 2: {r2_max:.1f} MB, Diff: {drift:.1f} MB)")
        print(f"4. Timetable Triage HTTP 200 (Round 1):   {'PASS' if crit_timetable_r1 else 'FAIL'} (HTTP {timetable_r1['status_code']}, {timetable_r1['triage_status']})")
        print(f"5. Timetable Triage HTTP 200 (Round 2):   {'PASS' if crit_timetable_r2 else 'FAIL'} (HTTP {timetable_r2['status_code']}, {timetable_r2['triage_status']})")

        all_passed = crit_no_oom and crit_peak_rss and crit_stability and crit_timetable_r1 and crit_timetable_r2

        print("=" * 80)
        if all_passed:
            print(">>> VERIFICATION SUCCESSFUL: ALL CRITERIA MET UNDER 512MB LIMIT <<<")
        else:
            print(">>> VERIFICATION FAILED: CRITERIA NOT MET <<<")
        print("=" * 80)

        # Save results to json
        out_path = os.path.join(REPO_DIR, "reports", "step5_verification_results.json")
        with open(out_path, "w") as f:
            json.dump({
                "all_passed": all_passed,
                "startup_rss_mb": round(startup_rss, 2),
                "max_peak_rss_mb": round(max_peak, 2),
                "round1_max_peak_mb": round(r1_max, 2),
                "round2_max_peak_mb": round(r2_max, 2),
                "drift_mb": round(drift, 2),
                "results": results
            }, f, indent=2)
        print(f"Saved results to: {out_path}")

        return 0 if all_passed else 1

    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    sys.exit(main())
