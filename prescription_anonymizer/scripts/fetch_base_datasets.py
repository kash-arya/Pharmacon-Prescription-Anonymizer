#!/usr/bin/env python3
"""
Automated Public Base-Dataset Acquisition.

Fetches authentic prescription slips, hospital OPD layouts, and clinical order
forms from open-access research repositories (Hugging Face, Zenodo, GitHub, Kaggle).
Filters for full-page prescription forms rather than isolated single-word crops.
Stores raw downloaded images in `data/raw_public_slips/`.
"""

import os
import sys
import json
import shutil
import urllib.request
import urllib.error
from typing import List, Dict, Any

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw_public_slips")
os.makedirs(RAW_DIR, exist_ok=True)

# Curated open-access repositories with full-page authentic medical prescription forms
HF_BASE_URL = "https://huggingface.co/datasets/chinmays18/medical-prescription-dataset/resolve/main"

# Sample set of full-page prescription slips from Hugging Face medical prescription OCR dataset
HF_SLIP_PATHS = [
    "test/images/prescription_00006.png",
    "test/images/prescription_00025.png",
    "test/images/prescription_00027.png",
    "test/images/prescription_00030.png",
    "test/images/prescription_00032.png",
    "test/images/prescription_00044.png",
    "test/images/prescription_00046.png",
    "test/images/prescription_00071.png",
    "test/images/prescription_00080.png",
    "test/images/prescription_00081.png"
]


def download_file(url: str, dest_path: str, timeout: int = 20) -> bool:
    """Downloads a single file from URL with a timeout and user-agent header."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Pharmacon-Research-DeID/2.0 (Academic; Medical Document OCR)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response, open(dest_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        return True
    except (urllib.error.URLError, TimeoutError, Exception) as e:
        print(f"  [!] Notice: Could not download {url} ({e})")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def fetch_from_kaggle(dataset_handle: str) -> List[str]:
    """
    Attempts to download from Kaggle if `kaggle` or `kagglehub` is available
    and API credentials exist in ~/.kaggle/kaggle.json.
    """
    acquired = []
    try:
        import kagglehub
        print(f"[*] Querying Kaggle Hub for '{dataset_handle}'...")
        path = kagglehub.dataset_download(dataset_handle)
        print(f"[*] Kaggle dataset downloaded to: {path}")
        
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
                    src = os.path.join(root, file)
                    dest = os.path.join(RAW_DIR, f"kaggle_{file}")
                    shutil.copy2(src, dest)
                    acquired.append(dest)
    except ImportError:
        print("[-] `kagglehub` not installed. Skipping direct Kaggle API download.")
    except Exception as e:
        print(f"[-] Kaggle download not available: {e}")

    return acquired


def fetch_all_public_slips(max_slips: int = 10) -> Dict[str, Any]:
    print("=" * 80)
    print("PHARMACON: PUBLIC BASE-DATASET ACQUISITION PIPELINE")
    print("=" * 80)
    print(f"Destination: {RAW_DIR}\n")

    manifest: List[Dict[str, Any]] = []

    # 1. Fetch from Hugging Face Open Medical Prescription Dataset
    print(f"[*] Fetching authentic full-page prescription slips from Hugging Face...")
    for rel_path in HF_SLIP_PATHS[:max_slips]:
        fname = os.path.basename(rel_path)
        url = f"{HF_BASE_URL}/{rel_path}"
        dest = os.path.join(RAW_DIR, fname)
        print(f"  -> Downloading {fname}...")
        ok = download_file(url, dest)
        if ok:
            manifest.append({
                "filename": fname,
                "filepath": dest,
                "source": "huggingface:chinmays18/medical-prescription-dataset",
                "url": url,
                "type": "full_page_rx_slip"
            })

    # 2. Check if local authentic samples exist and index them
    local_samples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples")
    if os.path.exists(local_samples_dir):
        for f in sorted(os.listdir(local_samples_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
                src_path = os.path.join(local_samples_dir, f)
                manifest.append({
                    "filename": f,
                    "filepath": src_path,
                    "source": "local_samples",
                    "type": "authentic_base_slip"
                })

    # Write Manifest
    manifest_path = os.path.join(RAW_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[+] Acquisition pipeline complete. Total indexed slips: {len(manifest)}")
    print(f"[+] Manifest written to: {manifest_path}")
    print("=" * 80 + "\n")

    return {
        "total_slips": len(manifest),
        "manifest_path": manifest_path,
        "items": manifest
    }


if __name__ == "__main__":
    fetch_all_public_slips()
