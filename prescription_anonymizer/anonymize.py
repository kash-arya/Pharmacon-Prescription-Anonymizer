#!/usr/bin/env python3
"""
Pharmacon Medical Prescription Anonymizer CLI.

Usage:
    python anonymize.py --input <path_to_prescription_pdf_or_image_or_directory> [--output <output_path>] [--json]
"""

import argparse
import os
import sys
import json
import time
import concurrent.futures
from typing import Dict, Any, List, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from prescription_anonymizer.src.raster_checker import inspect_pdf_structure
    from prescription_anonymizer.src.vector_redactor import VectorPDFRedactor
    from prescription_anonymizer.src.image_redactor import LocalImageRedactor
except ImportError:
    from src.raster_checker import inspect_pdf_structure
    from src.vector_redactor import VectorPDFRedactor
    from src.image_redactor import LocalImageRedactor

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


def process_file(input_file: str, output_file: str, vector_redactor: VectorPDFRedactor = None, image_redactor: LocalImageRedactor = None) -> Dict[str, Any]:
    """Inspects and anonymizes a single document or image."""
    structure = inspect_pdf_structure(input_file)
    fmt = structure["format_type"]

    if fmt == "VECTOR_DIGITAL":
        redactor = vector_redactor or VectorPDFRedactor()
        engine_name = "Native Vector Stream Redactor (PyMuPDF)"
        report = redactor.redact(input_file, output_file)
    else:
        redactor = image_redactor or LocalImageRedactor()
        engine_name = "Scanned Raster Image Redactor (Local OCR Anchor Engine)"
        report = redactor.redact(input_file, output_file)

    report["format_type"] = fmt
    report["page_count"] = structure["page_count"]
    report["engine"] = engine_name
    return report


def _batch_worker(task: Tuple[str, str]) -> Dict[str, Any]:
    """Independent worker process function for parallel multi-core batch redaction."""
    input_file, output_file = task
    try:
        rep = process_file(input_file, output_file)
        rep["status"] = "SUCCESS"
        return rep
    except Exception as e:
        return {
            "input_file": input_file,
            "output_file": output_file,
            "status": "ERROR",
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="De-identify medical prescriptions (PDF/Image/Directory)")
    parser.add_argument("--input", "-i", required=True, help="Path to input prescription file or directory")
    parser.add_argument("--output", "-o", default=None, help="Path to output file or directory")
    parser.add_argument("--workers", "-w", type=int, default=min(8, os.cpu_count() or 4), help="Parallel CPU workers for batch processing (default: auto)")
    parser.add_argument("--json", action="store_true", help="Print audit report as JSON")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------
    # Batch Processing: Directory Mode
    # -------------------------------------------------------------------
    if os.path.isdir(input_path):
        files_to_process: List[str] = []
        for root, dirs, filenames in os.walk(input_path):
            dirs[:] = [d for d in dirs if d != "anonymized_output" and not d.startswith(".")]
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_EXTENSIONS and not fname.endswith("_anonymized" + ext):
                    files_to_process.append(os.path.join(root, fname))

        if not files_to_process:
            print(f"No valid prescription files found in directory: {input_path}", file=sys.stderr)
            sys.exit(0)

        out_dir = os.path.abspath(args.output) if args.output else os.path.join(input_path, "anonymized_output")
        os.makedirs(out_dir, exist_ok=True)

        batch_tasks: List[Tuple[str, str]] = []
        for fpath in files_to_process:
            rel = os.path.relpath(fpath, input_path)
            base, ext = os.path.splitext(rel)
            out_file = os.path.join(out_dir, f"{base}_anonymized{ext}")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            batch_tasks.append((fpath, out_file))

        workers = max(1, min(args.workers, len(batch_tasks)))
        t0 = time.perf_counter()

        if workers > 1 and len(batch_tasks) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                batch_reports = list(executor.map(_batch_worker, batch_tasks))
        else:
            batch_reports = [_batch_worker(t) for t in batch_tasks]

        elapsed_sec = time.perf_counter() - t0

        if args.json:
            print(json.dumps({
                "mode": "BATCH_DIRECTORY",
                "input_directory": input_path,
                "output_directory": out_dir,
                "total_files": len(files_to_process),
                "workers_used": workers,
                "elapsed_seconds": round(elapsed_sec, 2),
                "throughput_docs_per_sec": round(len(files_to_process) / max(0.001, elapsed_sec), 2),
                "successful": sum(1 for r in batch_reports if r.get("status") == "SUCCESS"),
                "reports": batch_reports
            }, indent=2))
            return

        print("=" * 75)
        print("PHARMACON MEDICAL PRESCRIPTION ANONYMIZER: BATCH EXECUTION")
        print("=" * 75)
        print(f"Source Directory : {input_path}")
        print(f"Target Directory : {out_dir}")
        print(f"Total Files      : {len(files_to_process)} across {workers} parallel CPU workers")
        print(f"Processing Time  : {elapsed_sec:.2f}s ({len(files_to_process)/max(0.001, elapsed_sec):.1f} docs/sec)")
        print("-" * 75)
        for rep in batch_reports:
            fname = os.path.basename(rep["input_file"])
            if rep["status"] == "SUCCESS":
                print(f"  ✓ {fname:<35} | {rep['format_type']:<15} | Masked {rep['total_entities_redacted']} entities")
            else:
                print(f"  ✗ {fname:<35} | FAILED: {rep.get('error')}")
        print("=" * 75)
        print(f"All anonymized files safely stored in: {out_dir}\n")
        return

    # -------------------------------------------------------------------
    # Single File Mode
    # -------------------------------------------------------------------
    if args.output is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_anonymized{ext}"
    else:
        output_path = os.path.abspath(args.output)

    report = process_file(input_path, output_path)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("=" * 70)
    print("PHARMACON MEDICAL PRESCRIPTION ANONYMIZER")
    print("=" * 70)
    print(f"Input Document : {input_path}")
    print(f"Output Target  : {output_path}")
    print(f"Document Format: {report['format_type']} ({report['page_count']} pages)")
    print(f"Routing to     : {report['engine']}")
    print(f"Status         : SUCCESS")
    print(f"Redacted Units : {report['total_entities_redacted']} PII/PHI entities masked")
    for t in report["redacted_targets"][:20]:
        print(f"  - [{t['category']}] {repr(t.get('raw_text'))}")
    if len(report["redacted_targets"]) > 20:
        print(f"  ... and {len(report['redacted_targets']) - 20} more entities masked.")
    print("=" * 70)
    print(f"Saved anonymized document to: {output_path}\n")


if __name__ == "__main__":
    main()
