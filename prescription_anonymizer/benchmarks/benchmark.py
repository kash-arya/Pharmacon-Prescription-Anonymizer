#!/usr/bin/env python3
"""
Automated Evaluation Benchmark Suite for Pharmacon Prescription Anonymizer.

Evaluates:
1. PII Recall / Leakage Rate (Target: 100% Recall / 0% Leakage)
2. Clinical Pharmacology Preservation Rate (Target: 100% Preserved)
3. Processing Throughput & Redaction Latency

Outputs terminal summary table and saves detailed audit to benchmark_results.json.
"""

from typing import Dict, Any, List
import os
import sys
import time
import json
import argparse
from tabulate import tabulate
import pymupdf

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from prescription_anonymizer.src.raster_checker import inspect_pdf_structure
    from prescription_anonymizer.src.vector_redactor import VectorPDFRedactor
    from prescription_anonymizer.src.image_redactor import LocalImageRedactor, resolve_tessdata_path
except ImportError:
    from src.raster_checker import inspect_pdf_structure
    from src.vector_redactor import VectorPDFRedactor
    from src.image_redactor import LocalImageRedactor, resolve_tessdata_path


BENCHMARK_CASES = [
    {
        "id": "vector_prescription_emr",
        "filename": "sample_digital_vector_prescription.pdf",
        "description": "Digital Vector EMR Prescription (Apollo Hospitals)",
        "expected_format": "VECTOR_DIGITAL",
        "ground_truth_pii": [
            "Rajesh Sharma",
            "MCI-44912",
            "98110 23456",
            "dr.sharma@apollohospitals.com",
            "Anita Desai",
            "42 Y",
            "19/10/2026",
            "APL-2026-994812",
            "98765 43210"
        ],
        "ground_truth_clinical": [
            "Telmisartan 40 mg",
            "Amlodipine 5 mg",
            "Paracetamol 650 mg",
            "Essential Hypertension",
            "Low salt diet"
        ]
    },
    {
        "id": "scanned_slip_rx1",
        "filename": "sample_prescription_1.pdf",
        "description": "Scanned Doctor Slip #1 (Maria Sharapova / Dr John Smith)",
        "expected_format": "SCANNED_RASTER",
        "ground_truth_pii": [
            "John Smith",
            "Non-Important Street",
            "(000)-111-2222",
            "Maria Sharapova",
            "5/11/2022",
            "tennis court",
            "new Russia"
        ],
        "ground_truth_clinical": [
            "Prednisone",
            "20 mg",
            "Lialda",
            "2.4 gram",
            "Refill"
        ]
    },
    {
        "id": "scanned_slip_rx2",
        "filename": "sample_prescription_2.pdf",
        "description": "Scanned Doctor Slip #2 (Virat Kohli / Dr John Smith)",
        "expected_format": "SCANNED_RASTER",
        "ground_truth_pii": [
            "John Smith",
            "Non-Important Street",
            "(000)-123-2222",
            "Virat Kohli",
            "2/05/2",
            "cricket blvd",
            "NewDelhi"
        ],
        "ground_truth_clinical": [
            "Omeprazole",
            "tablets daily for three months",
            "Refill"
        ]
    },
    {
        "id": "scanned_patient_form",
        "filename": "sample_patient_details_1.pdf",
        "description": "Scanned Multi-page Patient Intake Form (Kathy Crawford)",
        "expected_format": "SCANNED_RASTER",
        "ground_truth_pii": [
            "Kathy Crawford",
            "May 6 1972",
            "(737) 988-0851",
            "9264 Ash Dr",
            "Simeone Crawford",
            "9266 Ash Dr",
            "375-4621",
            "Random",
            "4789 Bollinger Rd",
            "7115207313",
            "30 December 2020"
        ],
        "ground_truth_clinical": [
            "IMMUNE",
            "Migraine",
            "Peanuts",
            "Triptans",
            "Medical Insurance Details"
        ]
    },
    {
        "id": "scanned_image_rx1_png",
        "filename": "extracted_images/sample_prescription_1_p1.png",
        "description": "Standalone PNG Image Rx #1 (Maria Sharapova)",
        "expected_format": "SCANNED_RASTER",
        "ground_truth_pii": [
            "John Smith",
            "Non-Important Street",
            "(000)-111-2222",
            "Maria Sharapova",
            "5/11/2022",
            "tennis court"
        ],
        "ground_truth_clinical": [
            "Prednisone",
            "20 mg",
            "Lialda",
            "2.4 gram",
            "Refill"
        ]
    },
    {
        "id": "real_phone_ultrasound_march25",
        "filename": "real_phone_prescriptions/Aryan March 25.jpg",
        "description": "Real Samsung S23 Click #1 (Ultrasound Report - Akash Imaging)",
        "expected_format": "SCANNED_RASTER",
        "ground_truth_pii": [
            "Mr. ARYAN SHARMA",
            "19 Yrs Male",
            "Dr. PAWAN K GOYAL",
            "102438458",
            "NK GOYAL",
            "22/03/2025",
            "92 55 44 22 44",
            "9306036484",
            "Sriparna Dhar",
            "Akash Singh",
            "KHUSHBOO",
            "Kurukshetra",
            "http://103.193.198.232"
        ],
        "ground_truth_clinical": [
            "ABDOMEN",
            "Right kidney",
            "Left kidney",
            "Gall bladder",
            "CBD",
            "calculi",
            "calculus",
            "Urinary Bladder"
        ]
    },
    {
        "id": "real_phone_hospital_slip_march26",
        "filename": "real_phone_prescriptions/Aryan March 26.jpg",
        "description": "Real Samsung S23 Click #2 (Outpatient Slip - SIMSS Hospital)",
        "expected_format": "SCANNED_RASTER",
        "ground_truth_pii": [
            "Mr ARYAN",
            "UD-26270098511",
            "OP-26270005459",
            "31-May-2026",
            "21 Y/Male",
            "9253323609",
            "SEC 4,Kurukshetra",
            "SHASHANK SINGHAL",
            "Singhal",
            "01744-278022",
            "Mr RECEPTION"
        ],
        "ground_truth_clinical": [
            "GENERAL SURGERY",
            "Chief Complaints",
            "DIAGNOSIS",
            "VITALS",
            "ADVICE / TREATMENT",
            "INVESTIGATIONS",
            "mm of Hg",
            "Temp"
        ]
    },
    {
        "id": "real_phone_ultrasound_sept24",
        "filename": "real_phone_prescriptions/Aryan Sept 24.jpg",
        "description": "Real Samsung S23 Click #3 (Ultrasound Report - Akash Imaging)",
        "expected_format": "SCANNED_RASTER",
        "ground_truth_pii": [
            "Mr.ARYAN SHARMA",
            "102417573",
            "04/09/2024",
            "92 55 44 22 44",
            "9306036484",
            "akashimagingcentre@gmail.com",
            "Deepak Garg",
            "Sriparna Dhar",
            "Akash Singh",
            "Kurukshetra",
            "AKASH",
            "http://103.193.198.232"
        ],
        "ground_truth_clinical": [
            "ULTRASOUND",
            "WHOLE ABDOMEN",
            "Samsung RS80",
            "Liver",
            "Gall bladder",
            "Pancreas",
            "Right kidney",
            "Left kidney"
        ]
    }
]


def extract_document_text(file_path: str, format_type: str, tessdata_dir: str) -> str:
    """Extracts all text from document for verification (native or multi-DPI OCR)."""
    ext = os.path.splitext(file_path)[1].lower()
    is_image = ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

    if is_image:
        with pymupdf.open(file_path) as img_doc:
            pdf_bytes = img_doc.convert_to_pdf()
        doc = pymupdf.open("pdf", pdf_bytes)
    else:
        doc = pymupdf.open(file_path)

    full_text = []
    for page in doc:
        if format_type == "VECTOR_DIGITAL":
            text = page.get_text()
            full_text.append(text)
        else:
            tp200 = page.get_textpage_ocr(language="eng", dpi=200, tessdata=tessdata_dir)
            t200 = page.get_text(textpage=tp200)
            tp150 = page.get_textpage_ocr(language="eng", dpi=150, tessdata=tessdata_dir)
            t150 = page.get_text(textpage=tp150)
            full_text.append(t200 + "\n" + t150)

    doc.close()
    return "\n".join(full_text)


def evaluate_test_case(case: Dict[str, Any], samples_dir: str, output_dir: str, tessdata_dir: str) -> Dict[str, Any]:
    """Runs pipeline on a single test case and scores PII recall, leakage, and clinical preservation."""
    import re
    import cv2
    import numpy as np

    input_path = os.path.join(samples_dir, case["filename"])
    if not os.path.exists(input_path):
        alt_path = os.path.join(os.path.dirname(samples_dir), case["filename"])
        if os.path.exists(alt_path):
            input_path = alt_path
        elif os.path.exists(case["filename"]):
            input_path = case["filename"]
        else:
            return {
                "id": case["id"],
                "filename": case["filename"],
                "status": "SKIPPED_NOT_FOUND",
                "error": f"File not found at {input_path}"
            }

    base, ext = os.path.splitext(case["filename"])
    clean_base = os.path.basename(base)
    output_filename = f"{clean_base}_anonymized{ext}"
    output_path = os.path.join(output_dir, output_filename)

    start_time = time.perf_counter()

    # 1. Inspection & Routing
    struct = inspect_pdf_structure(input_path)
    fmt = struct["format_type"]

    if fmt == "VECTOR_DIGITAL":
        redactor = VectorPDFRedactor()
        report = redactor.redact(input_path, output_path)
        engine_name = "Native Vector Redactor"
    else:
        redactor = LocalImageRedactor(tessdata_dir=tessdata_dir)
        report = redactor.redact(input_path, output_path)
        engine_name = "Local OCR Image Redactor"

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # 2. Text Extraction on Redacted Document
    redacted_text = extract_document_text(output_path, fmt, tessdata_dir)
    redacted_text_lower = redacted_text.lower()
    norm_redacted = re.sub(r"[\s\-_:,./]+", "", redacted_text_lower)

    # 3. Strict Token-Level PII Leakage Check
    gt_pii = case["ground_truth_pii"]
    leaked_pii: List[str] = []

    STOPWORDS = {
        "mr", "mrs", "ms", "miss", "master", "dr", "or", "prof",
        "yrs", "yr", "male", "female", "y", "m", "f",
        "of", "in", "the", "and", "at", "by", "for", "to",
        "street", "st", "rd", "road", "ave", "avenue", "blvd",
        "court", "drive", "no", "date", "reg", "pt", "op",
        "com", "net", "org", "edu", "gov", "http", "https", "www"
    }

    for pii_item in gt_pii:
        norm_full = re.sub(r"[\s\-_:,./]+", "", pii_item.lower())
        if len(norm_full) >= 4 and norm_full in norm_redacted:
            leaked_pii.append(pii_item)
            continue

        tokens = [t for t in re.split(r"[\s\-_:,./]+", pii_item) if t]
        item_leaked = False
        for tok in tokens:
            tok_lower = tok.lower()
            if tok_lower in STOPWORDS:
                continue
            if re.match(r"^\d+$", tok):
                if len(tok) >= 4 and tok in norm_redacted:
                    item_leaked = True
                    break
            elif len(tok) >= 3:
                norm_tok = tok_lower
                if (norm_tok in norm_redacted) or re.search(r"\b" + re.escape(tok) + r"\b", redacted_text, re.IGNORECASE):
                    item_leaked = True
                    break

        if item_leaked and pii_item not in leaked_pii:
            leaked_pii.append(pii_item)

    # 4. QR Code & Barcode Scannability Leak Check
    out_ext_lower = os.path.splitext(output_path)[1].lower()
    is_img_out = out_ext_lower in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    try:
        qd = cv2.QRCodeDetector()
        if is_img_out:
            im_cv = cv2.imread(output_path)
            if im_cv is not None:
                ok_qr, decoded_info, _, _ = qd.detectAndDecodeMulti(im_cv)
                if ok_qr and any(decoded_info):
                    leaked_pii.append(f"SCANNABLE_QR_CODE:{decoded_info[0]}")
        else:
            chk_doc = pymupdf.open(output_path)
            for chk_page in chk_doc:
                chk_pix = chk_page.get_pixmap()
                chk_arr = np.frombuffer(chk_pix.samples, dtype=np.uint8).reshape((chk_pix.height, chk_pix.width, chk_pix.n))
                if chk_pix.n == 4:
                    chk_bgr = cv2.cvtColor(chk_arr, cv2.COLOR_RGBA2BGR)
                elif chk_pix.n == 3:
                    chk_bgr = cv2.cvtColor(chk_arr, cv2.COLOR_RGB2BGR)
                else:
                    chk_bgr = cv2.cvtColor(chk_arr, cv2.COLOR_GRAY2BGR)
                ok_qr, decoded_info, _, _ = qd.detectAndDecodeMulti(chk_bgr)
                if ok_qr and any(decoded_info):
                    leaked_pii.append(f"SCANNABLE_QR_CODE:{decoded_info[0]}")
                    break
            chk_doc.close()
    except Exception:
        pass

    pii_recall_pct = ((len(gt_pii) - len(leaked_pii)) / len(gt_pii)) * 100.0 if gt_pii else 100.0
    pii_leakage_rate = (len(leaked_pii) / len(gt_pii)) * 100.0 if gt_pii else 0.0

    # 5. Clinical Preservation Check
    gt_clinical = case["ground_truth_clinical"]
    preserved_clinical: List[str] = []
    missing_clinical: List[str] = []
    for clin_item in gt_clinical:
        norm_clin = re.sub(r"[\s\-_:,.]+", "", clin_item.lower())
        if (norm_clin in norm_redacted or
            clin_item.lower() in redacted_text_lower or
            re.search(re.escape(clin_item), redacted_text, re.IGNORECASE) or
            (len(norm_clin) >= 4 and (norm_clin[:-1] in norm_redacted or
             re.search(r"\b" + re.escape(clin_item[:-1]) + r"[a-z]?\b", redacted_text, re.IGNORECASE)))):
            preserved_clinical.append(clin_item)
        else:
            missing_clinical.append(clin_item)

    clinical_preservation_pct = (len(preserved_clinical) / len(gt_clinical)) * 100.0 if gt_clinical else 100.0

    # Status: PASS requires 0 PII leaks and 100% clinical preservation
    passed = (len(leaked_pii) == 0) and (clinical_preservation_pct >= 99.0)


    return {
        "id": case["id"],
        "filename": case["filename"],
        "description": case["description"],
        "format_type": fmt,
        "engine": engine_name,
        "input_path": input_path,
        "output_path": output_path,
        "elapsed_ms": round(elapsed_ms, 2),
        "total_entities_redacted": report.get("total_entities_redacted", len(report.get("redacted_targets", []))),
        "ground_truth_pii_count": len(gt_pii),
        "leaked_pii_count": len(leaked_pii),
        "leaked_pii_items": leaked_pii,
        "pii_recall_pct": round(pii_recall_pct, 2),
        "pii_leakage_rate_pct": round(pii_leakage_rate, 2),
        "ground_truth_clinical_count": len(gt_clinical),
        "preserved_clinical_count": len(preserved_clinical),
        "missing_clinical_items": missing_clinical,
        "clinical_preservation_pct": round(clinical_preservation_pct, 2),
        "status": "PASS" if passed else "FAIL"
    }


def run_benchmark(samples_dir: str, output_dir: str, results_json_path: str):
    """Executes the complete benchmark suite and prints formatted summary."""
    tessdata_dir = resolve_tessdata_path()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("PHARMACON PRESCRIPTION ANONYMIZER: AUTOMATED EVALUATION BENCHMARK")
    print("=" * 80)
    print(f"Samples Directory : {os.path.abspath(samples_dir)}")
    print(f"Output Directory  : {os.path.abspath(output_dir)}")
    print(f"Tessdata Path     : {tessdata_dir}\n")

    results: List[Dict[str, Any]] = []

    for case in BENCHMARK_CASES:
        print(f"Evaluating: {case['filename']} ({case['description']})...")
        res = evaluate_test_case(case, samples_dir, output_dir, tessdata_dir)
        results.append(res)
        print(f" -> Status: {res['status']} | PII Recall: {res.get('pii_recall_pct', 0)}% | Clinical Preserved: {res.get('clinical_preservation_pct', 0)}% ({res.get('elapsed_ms', 0)} ms)")

    # Compute Aggregates
    valid_results = [r for r in results if r["status"] in ("PASS", "FAIL")]
    total_cases = len(valid_results)
    passed_cases = sum(1 for r in valid_results if r["status"] == "PASS")
    avg_pii_recall = sum(r["pii_recall_pct"] for r in valid_results) / total_cases if total_cases else 0.0
    total_leaks = sum(r["leaked_pii_count"] for r in valid_results)
    avg_clinical_pres = sum(r["clinical_preservation_pct"] for r in valid_results) / total_cases if total_cases else 0.0
    total_latency_ms = sum(r["elapsed_ms"] for r in valid_results)

    # Tabulate Terminal Table
    table_data = []
    for r in valid_results:
        table_data.append([
            r["filename"],
            r["format_type"],
            r["total_entities_redacted"],
            f"{r['leaked_pii_count']}/{r['ground_truth_pii_count']}",
            f"{r['pii_recall_pct']:.1f}%",
            f"{r['preserved_clinical_count']}/{r['ground_truth_clinical_count']}",
            f"{r['clinical_preservation_pct']:.1f}%",
            f"{r['elapsed_ms']:.0f} ms",
            r["status"]
        ])

    headers = [
        "Document", "Type", "Redacted", "PII Leaks", "PII Recall", "Clin. Preserved", "Clin. Rate", "Latency", "Status"
    ]

    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION RESULTS TABLE")
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    print("\n" + "=" * 80)
    print("BENCHMARK AGGREGATE SUMMARY")
    print("=" * 80)
    print(f"Total Test Cases Evaluated : {total_cases}")
    print(f"Total Passed Cases         : {passed_cases} / {total_cases} ({passed_cases/total_cases*100:.1f}%)")
    print(f"Average PII Recall Rate    : {avg_pii_recall:.2f}% (Target: 100.0%)")
    print(f"Total PII Leaks Detected   : {total_leaks} (Target: 0)")
    print(f"Average Clinical Preserved : {avg_clinical_pres:.2f}% (Target: 100.0%)")
    print(f"Total Pipeline Latency     : {total_latency_ms:.1f} ms (Avg: {total_latency_ms/total_cases:.1f} ms/doc)")
    print("=" * 80)

    # Save to benchmark_results.json
    final_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "pass_rate_pct": round(passed_cases / total_cases * 100.0, 2) if total_cases else 0.0,
            "average_pii_recall_pct": round(avg_pii_recall, 2),
            "total_pii_leaks": total_leaks,
            "average_clinical_preservation_pct": round(avg_clinical_pres, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "status": "PASS" if passed_cases == total_cases else "FAIL"
        },
        "test_cases": results
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nSaved benchmark results to: {os.path.abspath(results_json_path)}\n")

    return final_output


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_json = os.path.join(repo_root, "reports", "benchmark_results.json")
    parser = argparse.ArgumentParser(description="Automated Evaluation Benchmark for Prescription Anonymizer")
    parser.add_argument("--samples-dir", default=None, help="Directory containing sample test files")
    parser.add_argument("--output-dir", default="/tmp/prescription_benchmark_outputs", help="Directory to save redacted test files")
    parser.add_argument("--output-json", default=default_json, help="Path to save benchmark_results.json")
    args = parser.parse_args()

    default_samples = os.path.join(repo_root, "data", "samples")
    samples_dir = args.samples_dir or default_samples

    run_benchmark(
        samples_dir=samples_dir,
        output_dir=args.output_dir,
        results_json_path=args.output_json
    )


if __name__ == "__main__":
    main()
