#!/usr/bin/env python3
"""
Multi-Core Workspace Masked Results Exporter & Triage Summary Reporter.

Processes all prescription specimens across all 4 dataset folders:
  - data/samples/
  - data/real_phone_prescriptions/
  - data/raw_public_slips/
  - data/unseen_holdout/

Outputs directly into outputs/ inside the workspace:
  1. outputs/<subfolder>/<name>_SIDE_BY_SIDE.jpg (Side-by-side visual audit)
  2. outputs/SUMMARY_REPORT.md (Complete Markdown Audit Table)

Standalone masked documents are rendered via secure temporary files and
deleted immediately after generating the side-by-side comparison image.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import time
import argparse
import datetime
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import pymupdf
from PIL import Image, ImageDraw, ImageFont

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

DATA_FOLDERS = ["samples", "real_phone_prescriptions", "raw_public_slips", "unseen_holdout"]
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}
IGNORE_SUFFIXES = ("_anonymized", "_MASKED", "_SIDE_BY_SIDE", "_holdout_redacted")


def render_page_to_pil(doc_or_image_path: str, page_num: int = 1, dpi: int = 150) -> Image.Image:
    """Renders page 1 of document or loads image into RGB PIL Image."""
    ext = os.path.splitext(doc_or_image_path)[1].lower()
    if ext == ".pdf":
        doc = pymupdf.open(doc_or_image_path)
        page = doc[min(page_num - 1, len(doc) - 1)]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples) if pix.n == 3 else Image.frombytes("RGBA", [pix.width, pix.height], pix.samples).convert("RGB")
        doc.close()
        return img
    else:
        try:
            doc = pymupdf.open(doc_or_image_path)
            pix = doc[0].get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples) if pix.n == 3 else Image.frombytes("RGBA", [pix.width, pix.height], pix.samples).convert("RGB")
            doc.close()
            return img
        except Exception:
            return Image.open(doc_or_image_path).convert("RGB")


def generate_side_by_side_image(
    orig_path: str,
    masked_path: str,
    out_jpg_path: str,
    metadata: Dict[str, Any]
) -> None:
    """Creates a high-contrast side-by-side comparison JPEG with an audit banner."""
    orig_img = render_page_to_pil(orig_path, page_num=1, dpi=150)
    masked_img = render_page_to_pil(masked_path, page_num=1, dpi=150)

    target_h = max(800, min(1400, max(orig_img.height, masked_img.height)))
    w_orig = max(1, int(orig_img.width * (target_h / orig_img.height)))
    w_masked = max(1, int(masked_img.width * (target_h / masked_img.height)))

    orig_scaled = orig_img.resize((w_orig, target_h), Image.Resampling.LANCZOS)
    masked_scaled = masked_img.resize((w_masked, target_h), Image.Resampling.LANCZOS)

    header_h = 75
    subhdr_h = 32
    divider_w = 4
    canvas_w = w_orig + w_masked + divider_w
    canvas_h = header_h + subhdr_h + target_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 23, 42))  # Slate 900
    draw = ImageDraw.Draw(canvas)

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
    ]
    font_bold = None
    font_sub = None
    font_col = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font_bold = ImageFont.truetype(fp, 16)
                font_sub = ImageFont.truetype(fp, 13)
                font_col = ImageFont.truetype(fp, 14)
                break
            except Exception:
                pass
    if font_bold is None:
        font_bold = ImageFont.load_default()
        font_sub = font_bold
        font_col = font_bold

    folder = metadata.get("folder", "")
    filename = metadata.get("filename", "")
    jur = metadata.get("jurisdiction", "GLOBAL")
    redacted = metadata.get("total_entities_redacted", 0)
    conf = metadata.get("confidence_score", 1.0) * 100.0
    triage = metadata.get("triage_status", "AUTO_APPROVED")
    fmt = metadata.get("format_type", "SCANNED_RASTER")

    title_text = f"DATASET: {folder}  |  SPECIMEN: {filename}  |  FORMAT: {fmt}"
    meta_text = f"Jurisdiction: {jur}   |   Entities Redacted: {redacted}   |   Self-Audit Confidence: {conf:.1f}%   |   Triage: "

    draw.text((20, 14), title_text, fill=(248, 250, 252), font=font_bold)
    draw.text((20, 42), meta_text, fill=(203, 213, 225), font=font_sub)

    meta_bbox = draw.textbbox((20, 42), meta_text, font=font_sub)
    badge_x = meta_bbox[2] + 2
    badge_color = (34, 197, 94) if triage == "AUTO_APPROVED" else (239, 68, 68)
    draw.text((badge_x, 42), triage, fill=badge_color, font=font_bold)

    # Sub-header column bars
    draw.rectangle([0, header_h, w_orig, header_h + subhdr_h], fill=(30, 41, 59))
    draw.rectangle([w_orig + divider_w, header_h, canvas_w, header_h + subhdr_h], fill=(30, 41, 59))

    left_lbl = "[ ORIGINAL UNREDACTED SPECIMEN ]"
    right_lbl = "[ MASKED / ANONYMIZED DOCUMENT ]"
    draw.text((max(10, w_orig // 2 - 130), header_h + 8), left_lbl, fill=(241, 245, 249), font=font_col)
    draw.text((w_orig + divider_w + max(10, w_masked // 2 - 130), header_h + 8), right_lbl, fill=(241, 245, 249), font=font_col)

    # Divider bar
    draw.rectangle([w_orig, header_h, w_orig + divider_w, canvas_h], fill=(51, 65, 85))

    # Paste images
    canvas.paste(orig_scaled, (0, header_h + subhdr_h))
    canvas.paste(masked_scaled, (w_orig + divider_w, header_h + subhdr_h))

    # Save
    canvas.save(out_jpg_path, "JPEG", quality=90)


def process_single_specimen(args_tuple: Tuple[str, str, str, str]) -> Dict[str, Any]:
    """Worker process: Anonymizes single document in temporary file and generates side-by-side JPEG."""
    input_path, folder_name, out_root, tessdata_dir = args_tuple
    fname = os.path.basename(input_path)
    base, ext = os.path.splitext(fname)

    sub_dir = os.path.join(out_root, folder_name)
    os.makedirs(sub_dir, exist_ok=True)
    side_by_side_path = os.path.join(sub_dir, f"{base}_SIDE_BY_SIDE.jpg")

    temp_masked_file = None
    t0 = time.time()
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_f:
            temp_masked_file = tmp_f.name

        struct = inspect_pdf_structure(input_path)
        fmt = struct["format_type"]

        if fmt == "VECTOR_DIGITAL":
            redactor = VectorPDFRedactor()
            report = redactor.redact(input_path, temp_masked_file, jurisdiction="AUTO")
        else:
            redactor = LocalImageRedactor(tessdata_dir=tessdata_dir)
            report = redactor.redact(input_path, temp_masked_file, jurisdiction="AUTO")

        elapsed_ms = round((time.time() - t0) * 1000.0, 1)

        detected_jur = report.get("jurisdiction", "GLOBAL")
        total_redacted = report.get("total_entities_redacted", len(report.get("redacted_targets", [])))
        confidence_score = report.get("confidence_score", 1.0)
        triage_status = report.get("triage_status", "AUTO_APPROVED")
        review_reasons = report.get("review_reasons", [])

        meta = {
            "folder": folder_name,
            "filename": fname,
            "format_type": fmt,
            "jurisdiction": detected_jur,
            "total_entities_redacted": total_redacted,
            "confidence_score": confidence_score,
            "triage_status": triage_status,
            "review_reasons": review_reasons,
            "elapsed_ms": elapsed_ms,
            "side_by_side_file": side_by_side_path,
            "status": "SUCCESS"
        }

        generate_side_by_side_image(input_path, temp_masked_file, side_by_side_path, meta)
        return meta

    except Exception as ex:
        return {
            "folder": folder_name,
            "filename": fname,
            "format_type": "UNKNOWN",
            "jurisdiction": "UNKNOWN",
            "total_entities_redacted": 0,
            "confidence_score": 0.0,
            "triage_status": "NEEDS_HUMAN_REVIEW",
            "review_reasons": [str(ex)],
            "elapsed_ms": round((time.time() - t0) * 1000.0, 1),
            "side_by_side_file": "",
            "status": "ERROR",
            "error": str(ex)
        }
    finally:
        if temp_masked_file and os.path.exists(temp_masked_file):
            try:
                os.remove(temp_masked_file)
            except Exception:
                pass


def discover_specimens(data_root: str) -> List[Tuple[str, str]]:
    """Discovers all valid prescription specimens across the 4 dataset folders."""
    specimens: List[Tuple[str, str]] = []
    for fld in DATA_FOLDERS:
        fld_path = os.path.join(data_root, fld)
        if not os.path.isdir(fld_path):
            continue
        for fn in sorted(os.listdir(fld_path)):
            full_p = os.path.join(fld_path, fn)
            if not os.path.isfile(full_p):
                continue
            if fn.startswith(".") or fn.endswith(".json"):
                continue
            base, ext = os.path.splitext(fn)
            if ext.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if any(fn.endswith(s + ext) for s in IGNORE_SUFFIXES):
                continue
            specimens.append((full_p, fld))
    return sorted(specimens, key=lambda x: (x[1], x[0]))


def write_summary_report(results: List[Dict[str, Any]], report_path: str) -> None:
    """Generates the Markdown summary report table at outputs/SUMMARY_REPORT.md."""
    total_docs = len(results)
    successful_docs = sum(1 for r in results if r.get("status") == "SUCCESS")
    total_entities = sum(r.get("total_entities_redacted", 0) for r in results)
    auto_approved = sum(1 for r in results if r.get("triage_status") == "AUTO_APPROVED")
    needs_review = total_docs - auto_approved

    jur_counts: Dict[str, int] = {}
    for r in results:
        j = r.get("jurisdiction", "GLOBAL")
        jur_counts[j] = jur_counts.get(j, 0) + 1

    lines = [
        "# PHARMACON: COMPLETE WORKSPACE MASKED RESULTS & TRIAGE AUDIT",
        "",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total Specimens Processed:** {total_docs}  ",
        f"**Successful Redactions:** {successful_docs} / {total_docs} ({successful_docs/max(1, total_docs)*100:.1f}%)  ",
        f"**Total PII Entities Redacted:** {total_entities:,}  ",
        f"**Automated Triage Verdict:** `{auto_approved}` AUTO_APPROVED  |  `{needs_review}` NEEDS_HUMAN_REVIEW  ",
        f"**Jurisdiction Breakdown:** " + ", ".join(f"`{k}`: {v}" for k, v in sorted(jur_counts.items())),
        "",
        "---",
        "",
        "## Summary Results Table",
        "",
        "| Folder | Document Name | Format | Jurisdiction | Entities Redacted | Confidence | Triage Status | Side-by-Side Comparison |",
        "|:-------|:--------------|:-------|:------------:|:-----------------:|:----------:|:-------------:|:------------------------|"
    ]

    for r in sorted(results, key=lambda x: (x.get("folder", ""), x.get("filename", ""))):
        folder = r.get("folder", "")
        fn = r.get("filename", "")
        fmt = r.get("format_type", "")
        jur = r.get("jurisdiction", "GLOBAL")
        redacted = r.get("total_entities_redacted", 0)
        conf = f"{r.get('confidence_score', 0.0) * 100:.1f}%"
        triage = r.get("triage_status", "UNKNOWN")
        triage_badge = f"**{triage}**" if triage == "AUTO_APPROVED" else f"<span style='color:red;'>**{triage}**</span>"

        base, ext = os.path.splitext(fn)
        side_rel = f"{folder}/{base}_SIDE_BY_SIDE.jpg"
        side_link = f"[`View Side-by-Side`]({side_rel})"

        lines.append(f"| `{folder}` | `{fn}` | `{fmt}` | `{jur}` | {redacted} | {conf} | {triage_badge} | {side_link} |")

    lines.extend([
        "",
        "---",
        "",
        "## Forensic Triage Notes",
        "- Documents flagged as `AUTO_APPROVED` have passed all clinical preservation constraints, demographic block masking, and high self-audit confidence.",
        "- Documents flagged as `NEEDS_HUMAN_REVIEW` generally represent ultra-low resolution historical specimens (< 400 px), non-standard camera angles, or missing national provider identifiers requiring visual pharmacist sign-off.",
        ""
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export all masked side-by-side comparisons into workspace outputs directory.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="Number of parallel worker processes")
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"), help="Root data directory")
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "outputs"), help="Destination directory")
    args = parser.parse_args()

    tessdata_dir = resolve_tessdata_path()
    os.makedirs(args.out_dir, exist_ok=True)

    specimens = discover_specimens(args.data_dir)
    print("=" * 80)
    print("PHARMACON: MULTI-CORE WORKSPACE MASKED RESULTS EXPORTER")
    print("=" * 80)
    print(f"Data Root Directory : {args.data_dir}")
    print(f"Output Directory    : {args.out_dir}")
    print(f"Parallel Workers    : {args.workers}")
    print(f"Total Specimens     : {len(specimens)}")
    print(f"Tessdata Path       : {tessdata_dir}")
    print("-" * 80)

    work_items = [
        (input_p, fld, args.out_dir, tessdata_dir)
        for input_p, fld in specimens
    ]

    t_start = time.time()
    results: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(process_single_specimen, item): item[0]
            for item in work_items
        }

        for idx, future in enumerate(as_completed(future_map), 1):
            src_file = future_map[future]
            try:
                res = future.result()
                results.append(res)
                fname = res.get("filename", os.path.basename(src_file))
                fld = res.get("folder", "")
                jur = res.get("jurisdiction", "")
                red = res.get("total_entities_redacted", 0)
                lat = res.get("elapsed_ms", 0.0)
                triage = res.get("triage_status", "")
                print(f"  [{idx:02d}/{len(work_items):02d}] ({fld}) {fname} -> {jur} | Redacted: {red:2d} | Triage: {triage:18s} ({lat:6.1f} ms)")
            except Exception as ex:
                print(f"  [{idx:02d}/{len(work_items):02d}] FAILED on {src_file}: {ex}")

    total_time = round(time.time() - t_start, 2)
    print("-" * 80)
    print(f"[+] All {len(results)} specimens processed in {total_time}s (Avg: {total_time/max(1, len(results))*1000:.1f} ms/doc)")

    report_path = os.path.join(args.out_dir, "SUMMARY_REPORT.md")
    write_summary_report(results, report_path)
    print(f"[+] Saved complete markdown triage report: {report_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
