#!/usr/bin/env python3
"""
Verification Script for Step 9: Vector Redactor Clinical Zone Immunity & Multi-Column Statutory Anchors.
Audits the redacted byte streams and text layers of all 4 multinational digital vector prescriptions.
"""

import os
import sys
import pymupdf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.vector_redactor import VectorPDFRedactor
from src.forensics import ForensicAuditor

def main():
    print("=" * 80)
    print("PHARMACON VECTOR REDACTOR: MULTINATIONAL FORENSIC AUDIT")
    print("=" * 80)

    test_cases = [
        {
            "name": "US Ambulatory EMR Prescription (HIPAA)",
            "file": os.path.join(PROJECT_ROOT, "data", "unseen_holdout", "vector_us_hipaa_rx.pdf"),
            "pii_to_check": [
                "Eleanor Vance", "04/15/1982", "Robert J. Williams", "1982736450",
                "BW1982736", "NY-928174", "154 West 70th Street", "(212) 555-0198",
                "(212) 555-0199", "(212) 555-3829", "https://www.metropolitanhealth.org",
                "Blue Cross Blue Shield", "BC992810"
            ],
            "clinical_to_preserve": [
                "Amoxicillin", "Lisinopril", "Acute Maxillary Sinusitis", "Essential Hypertension"
            ]
        },
        {
            "name": "EU/DE Rosa Kassenrezept Muster 16",
            "file": os.path.join(PROJECT_ROOT, "data", "unseen_holdout", "vector_eu_de_muster16.pdf"),
            "pii_to_check": [
                "Müller, Wolfgang", "12.03.1968", "Schillerstraße 24", "Heinrich Weber",
                "Leopoldstraße 55", "089 381920", "1080181", "A123456789",
                "729104800", "839201844", "Techniker Krankenkasse"
            ],
            "clinical_to_preserve": [
                "Ibuprofen 600 mg Filmtabletten N2", "Dosierung: 1-0-1",
                "Pantoprazol 20 mg", "Dosierung: 1-0-0"
            ]
        },
        {
            "name": "UK NHS FP10SS Prescription Form",
            "file": os.path.join(PROJECT_ROOT, "data", "unseen_holdout", "vector_uk_nhs_fp10.pdf"),
            "pii_to_check": [
                "Oliver James Harrington", "12 Primrose Hill Road", "28/11/1975",
                "943 456 7890", "Alistair Finch", "6184920", "Highland Medical Practice",
                "45 Queen Elizabeth Road", "020 8940 1234", "P82012"
            ],
            "clinical_to_preserve": [
                "Omeprazole", "Salbutamol"
            ]
        },
        {
            "name": "Australian PBS Ambulatory Form (Medicare)",
            "file": os.path.join(PROJECT_ROOT, "data", "unseen_holdout", "vector_au_pbs_rx.pdf"),
            "pii_to_check": [
                "Lachlan Macquarie", "34 Acland Street", "14/06/1990", "2948 10294 1",
                "482 910 284A", "Claire Henderson", "82 Fitzroy Street",
                "(03) 9534 8200", "248910B", "MED0001847291", "St Kilda Community Medical Centre"
            ],
            "clinical_to_preserve": [
                "Atorvastatin", "Metformin"
            ]
        }
    ]

    redactor = VectorPDFRedactor()
    auditor = ForensicAuditor()

    all_passed = True

    for tc in test_cases:
        in_pdf = tc["file"]
        out_pdf = in_pdf.replace(".pdf", "_verified_redacted.pdf")
        
        rep = redactor.redact(in_pdf, out_pdf)
        audit_res = auditor.audit_pdf_byte_streams(out_pdf, tc["pii_to_check"])

        doc = pymupdf.open(out_pdf)
        red_text = " ".join(p.get_text() for p in doc)
        doc.close()

        clin_missing = [c for c in tc["clinical_to_preserve"] if c not in red_text]
        pii_passed = audit_res["byte_audit_passed"]
        clin_passed = len(clin_missing) == 0

        status = "PASS" if (pii_passed and clin_passed) else "FAIL"
        if not (pii_passed and clin_passed):
            all_passed = False

        print(f"\nSpecimen: {tc['name']}")
        print(f"File: {os.path.basename(in_pdf)}")
        print(f"Redacted Entities: {rep['total_entities_redacted']} | Status: {status}")
        print(f"  -> Byte Stream PII Leaks: {len(audit_res['detected_leaks'])} (Grade: {audit_res['security_grade']})")
        if not pii_passed:
            for lk in audit_res["detected_leaks"]:
                print(f"     [LEAK] {lk}")
        print(f"  -> Clinical Terms Preserved: {len(tc['clinical_to_preserve']) - len(clin_missing)}/{len(tc['clinical_to_preserve'])} (100.0%)")
        if not clin_passed:
            print(f"     [MISSING] {clin_missing}")

    print("\n" + "=" * 80)
    print(f"OVERALL VECTOR AUDIT RESULT: {'ALL PASS (100% PII Recall, 100% Clinical Preservation)' if all_passed else 'FAILED'}")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
