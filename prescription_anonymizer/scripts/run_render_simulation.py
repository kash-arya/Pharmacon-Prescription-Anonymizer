#!/usr/bin/env python3
"""Render Production Simulation under Dual Constraints (512MB RAM, 10% CPU / 0.1 vCPU).

Runs the server under:
  systemd-run --user --scope -p MemoryMax=512M -p MemorySwapMax=0 -p CPUQuota=10%
Sends requests from outside the scope (simulating real client over network):
  Round 1: 6 Real Test Files (Cold Start + Sequential Stress)
  Round 2: Repeat Exact 6 Files without restart (Memory Stability & Leak Check)
"""

import os
import sys
import time
import json
import threading
import subprocess
import requests
import psutil

session = requests.Session()
session.trust_env = False

PORT = 8999
BASE_URL = f"http://127.0.0.1:{PORT}"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FIXTURES_DIR = os.path.join(REPO_DIR, "tests", "fixtures", "prscrptn")
DATA_DIR = os.path.join(REPO_DIR, "data", "real_phone_prescriptions")


def get_pid_on_port(port: int):
    port_str = str(port)
    for _ in range(120):
        try:
            for p in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmd = ' '.join(p.info['cmdline'] or [])
                    if port_str in cmd and 'uvicorn' in cmd and 'systemd-run' not in cmd:
                        return p.pid
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass
        time.sleep(0.5)
    return None


def get_combined_rss(proc: psutil.Process) -> float:
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
            time.sleep(0.05)

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
    t0 = time.perf_counter()

    with open(file_path, "rb") as f:
        files = {"file": (upload_name, f, mime)}
        resp = session.post(
            f"{BASE_URL}/api/v1/anonymize/preview?page=0&jurisdiction=AUTO",
            files=files,
            timeout=120
        )
    elapsed = time.perf_counter() - t0
    time.sleep(0.5)
    peak_rss = tracker.get_peak()
    idle_rss = tracker.get_current()

    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text[:200]}

    return {
        "status_code": resp.status_code,
        "elapsed_s": round(elapsed, 2),
        "peak_rss_mb": round(peak_rss, 1),
        "idle_rss_mb": round(idle_rss, 1),
        "triage_status": data.get("triage_status", "-"),
        "entities": data.get("total_entities_redacted", 0),
        "error": data.get("detail", "") if resp.status_code != 200 else None
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 80)
    print("RENDER SIMULATION: DUAL CONSTRAINTS (512MB RAM, 10% CPU / 0.1 vCPU)")
    print("=" * 80)

    # 1. Launch Server under systemd scope
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["RENDER_LOW_MEMORY"] = "1"
    env["PYTHONPATH"] = REPO_DIR
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_THREAD_LIMIT"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MALLOC_ARENA_MAX"] = "2"
    env["MALLOC_TRIM_THRESHOLD_"] = "32768"
    env["MALLOC_MMAP_THRESHOLD_"] = "32768"

    venv_python = os.path.join(REPO_DIR, ".venv", "bin", "python3")
    server_cmd = [
        "systemd-run", "--user", "--scope",
        "-p", "MemoryMax=512M",
        "-p", "MemorySwapMax=0",
        "-p", "CPUQuota=10%",
        venv_python, "-m", "uvicorn", "api.server:app",
        "--app-dir", REPO_DIR,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--log-level", "warning"
    ]

    print(f"Starting server in 10% CPU quota scope: {' '.join(server_cmd)}")
    log_file = open("/tmp/render_sim_server.log", "w", buffering=1)
    server_proc = subprocess.Popen(
        server_cmd,
        cwd=REPO_DIR,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )

    try:
        print("Waiting for server startup on port 8999...")
        server_pid = get_pid_on_port(PORT)
        if not server_pid:
            print("ERROR: Could not find server listening on port 8999.")
            server_proc.terminate()
            return 1

        # Poll /health
        healthy = False
        last_err = None
        for _ in range(60):
            try:
                r = session.get(f"{BASE_URL}/health", timeout=15)
                if r.status_code == 200:
                    healthy = True
                    break
                else:
                    last_err = f"HTTP {r.status_code}: {r.text}"
            except Exception as e:
                last_err = str(e)
                time.sleep(1.0)

        if not healthy:
            print(f"ERROR: Health check failed. Last error: {last_err}")
            server_proc.terminate()
            log_file.close()
            with open("/tmp/render_sim_server.log", "r") as f:
                print("SERVER LOG:\n", f.read())
            return 1

        tracker = MemoryTracker(server_pid)
        time.sleep(1.0)
        startup_rss = tracker.get_current()
        print(f"Server healthy! PID: {server_pid}, Baseline Idle RSS: {startup_rss:.1f} MB")

        all_test_files = [
            ("WhatsApp Image 2026-11-05 at 07.15.55.jpeg", os.path.join(FIXTURES_DIR, "WhatsApp Image 2026-11-05 at 07.15.55.jpeg"), "image/jpeg"),
            ("WhatsApp Image 2026-11-05 at 07.15.56.jpeg", os.path.join(FIXTURES_DIR, "WhatsApp Image 2026-11-05 at 07.15.56.jpeg"), "image/jpeg"),
            ("24b7b6cb-2446-49cf-b13c-2b63a74d6446.jpeg", os.path.join(FIXTURES_DIR, "24b7b6cb-2446-49cf-b13c-2b63a74d6446.jpeg"), "image/jpeg"),
            ("3Q15_Timetable_SemV.png", os.path.join(FIXTURES_DIR, "3Q15_Timetable_SemV.png"), "image/png"),
            ("Aryan Sept 24.jpg", os.path.join(DATA_DIR, "Aryan Sept 24.jpg"), "image/jpeg"),
            ("Aryan March 26.jpg", os.path.join(DATA_DIR, "Aryan March 26.jpg"), "image/jpeg"),
        ]

        results = []

        # --- ROUND 1 ---
        print("\n" + "=" * 80)
        print("ROUND 1: 6 Real Test Files (Cold Start + Sequential Stress under 10% CPU)")
        print("=" * 80)
        r1_peaks = []
        for name, path, mime in all_test_files:
            upload_name = f"prscrptn/{name}"
            res = run_request(tracker, path, upload_name, mime)
            res["round"] = "Round 1"
            res["name"] = name
            results.append(res)
            r1_peaks.append(res["peak_rss_mb"])
            print(f"[{res['round']}] {name[:36]:36} | HTTP {res['status_code']} | {res['elapsed_s']:5.2f} s | Peak RSS: {res['peak_rss_mb']:5.1f} MB | Triage: {res['triage_status']}")

        # --- ROUND 2 ---
        print("\n" + "=" * 80)
        print("ROUND 2: Repeat Exact 6 Real Test Files (Memory Leak & Warm Latency Check)")
        print("=" * 80)
        r2_peaks = []
        for name, path, mime in all_test_files:
            upload_name = f"prscrptn/{name}"
            res = run_request(tracker, path, upload_name, mime)
            res["round"] = "Round 2"
            res["name"] = name
            results.append(res)
            r2_peaks.append(res["peak_rss_mb"])
            print(f"[{res['round']}] {name[:36]:36} | HTTP {res['status_code']} | {res['elapsed_s']:5.2f} s | Peak RSS: {res['peak_rss_mb']:5.1f} MB | Triage: {res['triage_status']}")

        tracker.stop()

        # Summary & Criteria
        print("\n" + "=" * 80)
        print("ACCEPTANCE CRITERIA VERIFICATION (DUAL LIMITS: 512MB RAM + 10% CPU)")
        print("=" * 80)

        max_peak = max(r["peak_rss_mb"] for r in results)
        max_time = max(r["elapsed_s"] for r in results)
        r1_max = max(r1_peaks)
        r2_max = max(r2_peaks)
        drift = abs(r2_max - r1_max)

        timetable_r1 = [r for r in results if r["round"] == "Round 1" and "Timetable" in r["name"]][0]
        timetable_r2 = [r for r in results if r["round"] == "Round 2" and "Timetable" in r["name"]][0]

        crit_no_oom = server_proc.poll() is None
        crit_peak_rss = max_peak < 400.0
        crit_stability = drift <= 30.0
        crit_time = max_time < 60.0
        crit_tt_r1 = (timetable_r1["status_code"] == 200 and timetable_r1["triage_status"] == "NOT_A_PRESCRIPTION")
        crit_tt_r2 = (timetable_r2["status_code"] == 200 and timetable_r2["triage_status"] == "NOT_A_PRESCRIPTION")
        crit_all_200 = all(r["status_code"] == 200 for r in results)

        print(f"1. No OOM Kill (Process Alive):         {'PASS' if crit_no_oom else 'FAIL'}")
        print(f"2. Peak Combined RSS < 400 MB:           {'PASS' if crit_peak_rss else 'FAIL'} (Peak: {max_peak:.1f} MB, Target: <400 MB)")
        print(f"3. Stability across Rounds (<=30MB diff): {'PASS' if crit_stability else 'FAIL'} (Round 1: {r1_max:.1f} MB, Round 2: {r2_max:.1f} MB, Drift: {drift:.1f} MB)")
        print(f"4. Max Request Time < 60 s:              {'PASS' if crit_time else 'FAIL'} (Max: {max_time:.2f} s)")
        print(f"5. All 12 Requests Return HTTP 200:      {'PASS' if crit_all_200 else 'FAIL'}")
        print(f"6. Timetable Triage == NOT_A_PRESCRIPTION:{'PASS' if crit_tt_r1 and crit_tt_r2 else 'FAIL'}")

        all_passed = crit_no_oom and crit_peak_rss and crit_stability and crit_time and crit_all_200 and crit_tt_r1 and crit_tt_r2
        print("=" * 80)
        print(f">>> FINAL SIMULATION VERDICT: {'SUCCESS: READY FOR RENDER' if all_passed else 'FAILED'} <<<")
        print("=" * 80)

        out_path = os.path.join(REPO_DIR, "reports", "render_simulation_results.json")
        with open(out_path, "w") as f:
            json.dump({
                "all_passed": all_passed,
                "startup_rss_mb": round(startup_rss, 1),
                "max_peak_rss_mb": round(max_peak, 1),
                "max_elapsed_s": round(max_time, 2),
                "round1_max_peak_mb": round(r1_max, 1),
                "round2_max_peak_mb": round(r2_max, 1),
                "drift_mb": round(drift, 1),
                "results": results
            }, f, indent=2)
        print(f"Saved results to: {out_path}")

        return 0 if all_passed else 1

    finally:
        try:
            if server_proc.poll() is None:
                server_proc.terminate()
                server_proc.wait(timeout=5)
        except Exception:
            server_proc.kill()
        subprocess.run(["pkill", "-9", "-f", "uvicorn api.server:app"], check=False)


if __name__ == "__main__":
    sys.exit(main())
