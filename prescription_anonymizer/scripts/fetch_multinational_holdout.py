#!/usr/bin/env python3
"""
Multinational Wikimedia Commons Prescription Holdout Acquisition Pipeline.

Fetches authentic public-domain and Creative Commons medical prescription specimens
across multiple international healthcare frameworks (US, UK, EU/Germany, ES, Global)
directly from the Wikimedia Commons MediaWiki API.

Ensures a zero-trust, blind evaluation dataset stored in `data/unseen_holdout/`.
"""

import os
import sys
import json
import re
import urllib.request
import urllib.parse
import urllib.error
import shutil
from typing import List, Dict, Any, Tuple
from PIL import Image

# Determine holdout directory relative to script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
HOLDOUT_DIR = os.path.join(REPO_ROOT, "data", "unseen_holdout")
os.makedirs(HOLDOUT_DIR, exist_ok=True)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "PharmaconHoldoutEvaluator/2.0 (Medical Document DeID Research; research@example.org)"

SEARCH_QUERIES = [
    "Muster 16 Rezept",
    "Kassenrezept",
    "Privatrezept",
    "FP10 prescription",
    "medical prescription sample",
    "prescription form",
    "medical prescription pad",
    "doctor prescription",
    "receta medica"
]

SUPPORTED_RASTER_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}
EXCLUDE_TITLE_KEYWORDS = {"logo", "icon", "flag", "badge", "coat_of_arms", "coa", "diagram", "map", "signature_only", "symbol"}


def sanitize_filename(name: str) -> str:
    """Creates a clean, safe filesystem slug from a title."""
    name = re.sub(r"^File:", "", name, flags=re.IGNORECASE)
    base, ext = os.path.splitext(name)
    clean_base = re.sub(r"[^a-zA-Z0-9_\-]+", "_", base).strip("_").lower()
    return f"{clean_base[:40]}{ext.lower()}"


def infer_region_prefix(title: str, query: str) -> str:
    """Infers region identifier (de, us, uk, es, jp, global) from file metadata."""
    t_lower = (title + " " + query).lower()
    if any(k in t_lower for k in ["kassenrezept", "privatrezept", "rezept", "muster", "deutsch"]):
        return "eu_de"
    elif any(k in t_lower for k in ["geograph.org.uk", "uk", "nhs", "fp10", "england", "british"]):
        return "uk"
    elif any(k in t_lower for k in ["ny", "prohibition", "whiskey", "cannabis", "usa", "american"]):
        return "us"
    elif any(k in t_lower for k in ["receta", "espanol", "mexico", "espana"]):
        return "es"
    elif any(k in t_lower for k in ["japan", "japanese"]):
        return "jp"
    return "global"


def search_commons_files(query: str, limit: int = 15) -> List[str]:
    """Searches Wikimedia Commons for files matching the given query."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{query} filetype:bitmap",
        "srnamespace": "6",  # File: namespace
        "srlimit": str(limit),
        "format": "json"
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("query", {}).get("search", [])
            return [r["title"] for r in results if "title" in r]
    except Exception as e:
        print(f"  [!] Search error for '{query}': {e}")
        return []


def get_image_info_batch(titles: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetches direct URLs, dimensions, and MIME types for titles in a single batch."""
    if not titles:
        return {}

    info_map = {}
    # Batch in groups of 25
    for i in range(0, len(titles), 25):
        batch = titles[i:i + 25]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "format": "json"
        }
        url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for pid, pdata in pages.items():
                    title = pdata.get("title")
                    img_list = pdata.get("imageinfo", [])
                    if img_list:
                        info_map[title] = img_list[0]
        except Exception as e:
            print(f"  [!] Batch imageinfo error: {e}")

    return info_map


def download_specimen(url: str, dest_path: str, timeout: int = 25) -> bool:
    """Downloads a single image file with timeout and validation."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest_path, "wb") as out:
            shutil.copyfileobj(resp, out)

        # Verify image integrity with PIL
        with Image.open(dest_path) as im:
            im.verify()
        return True
    except Exception as e:
        print(f"  [!] Failed download: {url} ({e})")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def main():
    print("=" * 80)
    print("PHARMACON: MULTINATIONAL COMMONS HOLDOUT ACQUISITION PIPELINE")
    print("=" * 80)
    print(f"Target Directory : {HOLDOUT_DIR}")
    print(f"Source Repository: Wikimedia Commons MediaWiki API")
    print(f"User-Agent       : {USER_AGENT}\n")

    print("[1/3] Executing MediaWiki search queries across multinational layouts...")
    candidate_titles: Dict[str, str] = {}  # title -> query

    for q in SEARCH_QUERIES:
        titles = search_commons_files(q, limit=12)
        print(f"  -> Query \"{q}\": Found {len(titles)} hits")
        for t in titles:
            ext = os.path.splitext(t)[1].lower()
            if ext in SUPPORTED_RASTER_EXTS:
                # Filter out obvious non-document items
                t_clean = t.lower()
                if not any(k in t_clean for k in EXCLUDE_TITLE_KEYWORDS):
                    if t not in candidate_titles:
                        candidate_titles[t] = q

    print(f"\n[+] Total unique candidate raster images gathered: {len(candidate_titles)}")

    print("\n[2/3] Fetching image dimensions and direct download URLs...")
    info_map = get_image_info_batch(list(candidate_titles.keys()))

    # Filter by resolution (width >= 400 and height >= 400)
    valid_specimens = []
    for title, q in candidate_titles.items():
        info = info_map.get(title)
        if not info:
            continue
        width = info.get("width", 0)
        height = info.get("height", 0)
        url = info.get("url", "")
        mime = info.get("mime", "")

        if width >= 400 and height >= 400 and url.startswith("http"):
            region = infer_region_prefix(title, q)
            slug = sanitize_filename(title)
            dest_name = f"commons_{region}_{slug}"
            valid_specimens.append({
                "title": title,
                "query": q,
                "region": region.upper(),
                "width": width,
                "height": height,
                "mime": mime,
                "url": url,
                "dest_name": dest_name,
                "dest_path": os.path.join(HOLDOUT_DIR, dest_name)
            })

    # Prioritize diverse regional coverage up to 10-12 distinct files
    # Group by region to prevent single-region dominance
    region_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for spec in valid_specimens:
        r = spec["region"]
        region_buckets.setdefault(r, []).append(spec)

    selected_specimens = []
    max_total = 12
    # Round-robin selection across available regions
    while len(selected_specimens) < max_total:
        added_in_round = False
        for r, items in list(region_buckets.items()):
            if items and len(selected_specimens) < max_total:
                selected_specimens.append(items.pop(0))
                added_in_round = True
        if not added_in_round:
            break

    print(f"[+] Selected {len(selected_specimens)} diverse specimens across {len(region_buckets)} regional layouts.")

    print(f"\n[3/3] Downloading specimens to {HOLDOUT_DIR}...")
    downloaded_records = []

    for idx, spec in enumerate(selected_specimens, 1):
        dest_p = spec["dest_path"]
        print(f"  [{idx}/{len(selected_specimens)}] Downloading {spec['dest_name']} ({spec['width']}x{spec['height']})...")

        ok = False
        if os.path.exists(dest_p) and os.path.getsize(dest_p) > 5000:
            ok = True
        else:
            ok = download_specimen(spec["url"], dest_p)

        if ok and os.path.exists(dest_p):
            file_kb = round(os.path.getsize(dest_p) / 1024.0, 1)
            downloaded_records.append({
                "filename": spec["dest_name"],
                "region": spec["region"],
                "dimensions": f"{spec['width']} x {spec['height']}",
                "size_kb": f"{file_kb} KB",
                "source_title": spec["title"],
                "source_url": spec["url"]
            })

    # Save manifest.json
    manifest_path = os.path.join(HOLDOUT_DIR, "holdout_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_specimens": len(downloaded_records),
            "source": "Wikimedia Commons",
            "specimens": downloaded_records
        }, f, indent=2)

    # Print clean formatted table
    print("\n" + "=" * 110)
    print("DOWNLOADED MULTINATIONAL WIKIMEDIA COMMONS PRESCRIPTION SPECIMENS")
    print("=" * 110)
    col_fmt = "{:<36} {:<10} {:<14} {:<10} {:<36}"
    print(col_fmt.format("Filename", "Region", "Dimensions", "File Size", "Commons Source Title"))
    print("-" * 110)
    for rec in downloaded_records:
        fn_short = rec["filename"][:35]
        title_short = rec["source_title"][:35]
        print(col_fmt.format(fn_short, rec["region"], rec["dimensions"], rec["size_kb"], title_short))
    print("=" * 110)
    print(f"[+] Download complete: {len(downloaded_records)} holdout specimens verified.")
    print(f"[+] Manifest generated: {manifest_path}\n")


if __name__ == "__main__":
    main()
