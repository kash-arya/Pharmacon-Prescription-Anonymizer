#!/usr/bin/env python3
"""
Generates authentic multinational Digital Vector EMR Prescription PDFs
in `data/unseen_holdout/` for US (HIPAA), UK (NHS FP10), AU (PBS Medicare),
and EU/DE (Muster 16 Rosa Rezept).
"""

import os
import sys
import pymupdf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HOLDOUT_DIR = os.path.join(PROJECT_ROOT, "data", "unseen_holdout")


def create_us_vector_prescription(output_path: str):
    """Generates authentic US EMR Ambulatory e-Prescription (HIPAA compliant)."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)  # US Letter

    # Header Box & Border
    page.draw_rect(pymupdf.Rect(36, 36, 576, 756), color=(0.1, 0.2, 0.4), width=1.0)
    page.draw_rect(pymupdf.Rect(36, 36, 576, 120), color=(0.1, 0.2, 0.4), fill=(0.95, 0.97, 1.0))

    page.insert_text(pymupdf.Point(50, 60), "METROPOLITAN HEALTHCARE ALLIANCE", fontsize=15, fontname="helv", color=(0.05, 0.15, 0.35))
    page.insert_text(pymupdf.Point(50, 78), "Department of Internal Medicine & Ambulatory Care", fontsize=10, fontname="helv", color=(0.3, 0.3, 0.3))
    page.insert_text(pymupdf.Point(50, 93), "742 Evergreen Terrace, Suite 300, New York, NY 10021", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))
    page.insert_text(pymupdf.Point(50, 107), "Tel: (212) 555-0198  |  Fax: (212) 555-0199  |  Web: https://www.metropolitanhealth.org", fontsize=8.5, fontname="helv", color=(0.2, 0.2, 0.2))

    # Prescriber info (top right)
    page.insert_text(pymupdf.Point(380, 60), "Dr. Robert J. Williams, MD", fontsize=11, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(380, 75), "NPI: 1982736450", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))
    page.insert_text(pymupdf.Point(380, 90), "DEA No: BW1982736", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))
    page.insert_text(pymupdf.Point(380, 105), "Licence: NY-928174", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))

    # Patient Demographic Grid
    page.draw_rect(pymupdf.Rect(50, 135, 562, 230), color=(0.7, 0.7, 0.7), width=0.75)
    page.insert_text(pymupdf.Point(60, 155), "Patient Name: Eleanor Vance", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(320, 155), "DOB: 04/15/1982", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(450, 155), "Gender: Female", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))

    page.insert_text(pymupdf.Point(60, 175), "MRN: MRN-884920", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(320, 175), "Date: 10/12/2024", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(450, 175), "Age: 42", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))

    page.insert_text(pymupdf.Point(60, 195), "Address: 154 West 70th Street, Apt 4B, New York, NY 10023", fontsize=9.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(60, 215), "Phone: (212) 555-3829  |  Insurance: Blue Cross Blue Shield  |  Policy: BC992810", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))

    # Rx Symbol & Medications Section
    page.draw_line(pymupdf.Point(50, 245), pymupdf.Point(562, 245), color=(0.2, 0.2, 0.2), width=1.0)
    page.insert_text(pymupdf.Point(60, 275), "Rx", fontsize=24, fontname="times-bold", color=(0.1, 0.2, 0.4))

    # Medication 1
    page.insert_text(pymupdf.Point(60, 310), "1. Amoxicillin 500 mg oral capsule", fontsize=11, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(80, 328), "Sig: Take 1 capsule by mouth three times daily for 10 days with food.", fontsize=10, fontname="helv", color=(0.15, 0.15, 0.15))
    page.insert_text(pymupdf.Point(80, 345), "Dispense: 30 (thirty) capsules   |   Refills: 0 (zero)   |   Substitutions: Permitted", fontsize=9.5, fontname="helv", color=(0.25, 0.25, 0.25))

    # Medication 2
    page.insert_text(pymupdf.Point(60, 385), "2. Lisinopril 10 mg oral tablet", fontsize=11, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(80, 403), "Sig: Take 1 tablet by mouth daily in the morning for hypertension management.", fontsize=10, fontname="helv", color=(0.15, 0.15, 0.15))
    page.insert_text(pymupdf.Point(80, 420), "Dispense: 90 (ninety) tablets   |   Refills: 3 (three)   |   Substitutions: Permitted", fontsize=9.5, fontname="helv", color=(0.25, 0.25, 0.25))

    # Diagnosis Block
    page.draw_rect(pymupdf.Rect(50, 460, 562, 510), color=(0.8, 0.8, 0.8), fill=(0.98, 0.98, 0.98))
    page.insert_text(pymupdf.Point(60, 480), "Clinical Indications & ICD-10:", fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(60, 498), "• Acute Maxillary Sinusitis (J01.00)   • Essential Hypertension (I10)", fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))

    # Prescriber Signature & Verification
    page.draw_line(pymupdf.Point(50, 640), pymupdf.Point(562, 640), color=(0.7, 0.7, 0.7), width=0.75)
    page.insert_text(pymupdf.Point(60, 670), "Prescriber's Signature: Dr. Robert J. Williams", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(60, 690), "Date: 10/12/2024", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(360, 670), "Pharmacy Dispenser Validation Area", fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
    page.draw_rect(pymupdf.Rect(350, 680, 550, 735), color=(0.7, 0.7, 0.7), width=0.5)

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"[+] Created US Vector Prescription: {output_path}")


def create_uk_vector_prescription(output_path: str):
    """Generates authentic UK NHS FP10 Prescription PDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4

    # NHS Header Banner
    page.draw_rect(pymupdf.Rect(40, 40, 555, 100), color=(0.0, 0.35, 0.7), fill=(0.0, 0.35, 0.7))
    page.insert_text(pymupdf.Point(55, 75), "NHS", fontsize=24, fontname="helv", color=(1.0, 1.0, 1.0))
    page.insert_text(pymupdf.Point(125, 72), "PRESCRIPTION FORM - FP10SS", fontsize=14, fontname="helv", color=(1.0, 1.0, 1.0))

    # Surgery Address & Practice Code
    page.draw_rect(pymupdf.Rect(40, 110, 555, 175), color=(0.6, 0.6, 0.6), width=0.75)
    page.insert_text(pymupdf.Point(55, 130), "Practice / Surgery Address:", fontsize=9.5, fontname="helv", color=(0.3, 0.3, 0.3))
    page.insert_text(pymupdf.Point(55, 147), "Highland Medical Practice, 45 Queen Elizabeth Road, Richmond, TW10 6RF", fontsize=9.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(55, 163), "Practice Code: P82012   |   Tel: 020 8940 1234", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))

    # Patient Details Box
    page.draw_rect(pymupdf.Rect(40, 185, 555, 295), color=(0.6, 0.6, 0.6), width=0.75)
    page.insert_text(pymupdf.Point(55, 205), "Patient (including forename and address):", fontsize=9.5, fontname="helv", color=(0.3, 0.3, 0.3))
    page.insert_text(pymupdf.Point(55, 225), "Mr. Oliver James Harrington", fontsize=12, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(55, 245), "Address: 12 Primrose Hill Road, London NW3 3AD", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(55, 265), "DOB: 28/11/1975      Age: 49      Sex: Male", fontsize=9.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(55, 283), "NHS No: 943 456 7890      Date: 18/10/2024", fontsize=9.5, fontname="helv", color=(0.0, 0.0, 0.0))

    # Clinical Items
    page.draw_rect(pymupdf.Rect(40, 310, 555, 560), color=(0.6, 0.6, 0.6), width=0.75)
    page.insert_text(pymupdf.Point(55, 335), "1. Omeprazole 20mg gastro-resistant capsules", fontsize=11, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(70, 355), "Directions: Take one capsule daily in the morning before breakfast.", fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(70, 373), "Quantity to supply: 28 (twenty-eight) capsules", fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))

    page.insert_text(pymupdf.Point(55, 415), "2. Salbutamol 100micrograms/dose inhaler CFC-free", fontsize=11, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(70, 435), "Directions: Inhale one to two puffs as required for relief of wheeze or breathlessness.", fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(70, 453), "Quantity to supply: 1 inhaler (200 doses)", fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))

    # Prescriber Signature Box
    page.draw_rect(pymupdf.Rect(40, 580, 555, 710), color=(0.6, 0.6, 0.6), width=0.75)
    page.insert_text(pymupdf.Point(55, 605), "Signature of Prescriber: Dr. Alistair Finch", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(55, 625), "Doctor Name: Dr Alistair Finch, MBChB MRCGP", fontsize=9.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(55, 645), "GMC No: 6184920", fontsize=9.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(55, 665), "Dispense Date: 18/10/2024", fontsize=9.5, fontname="helv", color=(0.0, 0.0, 0.0))

    # Dispenser Area
    page.draw_rect(pymupdf.Rect(40, 720, 555, 800), color=(0.8, 0.8, 0.8), fill=(0.97, 0.97, 0.97))
    page.insert_text(pymupdf.Point(55, 740), "For Dispenser / Pharmacy Stamp:", fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"[+] Created UK Vector Prescription: {output_path}")


def create_au_vector_prescription(output_path: str):
    """Generates authentic Australian PBS (Pharmaceutical Benefits Scheme) Prescription."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4

    # Australian PBS Header
    page.draw_rect(pymupdf.Rect(35, 35, 560, 85), color=(0.1, 0.4, 0.2), fill=(0.93, 0.97, 0.93))
    page.insert_text(pymupdf.Point(50, 58), "AUSTRALIAN PHARMACEUTICAL BENEFITS SCHEME (PBS)", fontsize=13, fontname="helv", color=(0.05, 0.3, 0.1))
    page.insert_text(pymupdf.Point(50, 75), "Standard Ambulatory Prescription Form — Medicare Australia", fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))

    # Prescriber Details
    page.draw_rect(pymupdf.Rect(35, 95, 560, 185), color=(0.6, 0.6, 0.6), width=0.75)
    page.insert_text(pymupdf.Point(50, 115), "Prescriber: Dr. Claire Henderson, MBBS FRACGP", fontsize=10.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(50, 133), "Clinic: St Kilda Community Medical Centre", fontsize=9.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(50, 150), "Address: 82 Fitzroy Street, St Kilda VIC 3182   |   Phone: (03) 9534 8200", fontsize=9, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(50, 168), "PBS Prescriber No: 248910B   |   AHPRA No: MED0001847291", fontsize=9, fontname="helv", color=(0.0, 0.0, 0.0))

    # Patient Section
    page.draw_rect(pymupdf.Rect(35, 195, 560, 290), color=(0.6, 0.6, 0.6), width=0.75)
    page.insert_text(pymupdf.Point(50, 215), "Patient Name: Lachlan Macquarie", fontsize=11, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(340, 215), "Medicare No: 2948 10294 1  [1]", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(50, 235), "Address: 34 Acland Street, St Kilda VIC 3182", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(50, 255), "DOB: 14/06/1990      Gender: M      Date: 05/10/2024", fontsize=9.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(50, 275), "Entitlement: Concession Card Holder (CRN: 482 910 284A)", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))

    # Prescription Items
    page.draw_rect(pymupdf.Rect(35, 305, 560, 560), color=(0.6, 0.6, 0.6), width=0.75)
    page.insert_text(pymupdf.Point(50, 330), "1. Atorvastatin 20 mg tablet", fontsize=11, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(70, 350), "Directions: Take one tablet daily at bedtime for hypercholesterolaemia.", fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(70, 368), "Quantity: 30 tablets   |   Repeats: 5 (five)   |   Brand substitution: Not permitted", fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))

    page.insert_text(pymupdf.Point(50, 410), "2. Metformin 500 mg tablet", fontsize=11, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(70, 430), "Directions: Take one tablet twice daily with meals for type 2 diabetes.", fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(70, 448), "Quantity: 100 tablets   |   Repeats: 2 (two)", fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))

    # Prescriber Signature & Date
    page.draw_rect(pymupdf.Rect(35, 580, 560, 680), color=(0.6, 0.6, 0.6), width=0.75)
    page.insert_text(pymupdf.Point(50, 605), "Prescriber's Signature: Dr. Claire Henderson", fontsize=10.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(50, 625), "Date of Prescribing: 05/10/2024", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(50, 645), "Certifying Practitioner: Dr Claire Henderson (AHPRA: MED0001847291)", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"[+] Created AU Vector Prescription: {output_path}")


def create_eu_de_vector_prescription(output_path: str):
    """Generates authentic German Muster 16 (Rosa Kassenrezept) Prescription."""
    doc = pymupdf.open()
    # German standard prescription size: DIN A6 landscape (148 x 105 mm) -> (420 x 298 pt)
    page = doc.new_page(width=595, height=420)

    # Background tint
    page.draw_rect(pymupdf.Rect(15, 15, 580, 405), color=(0.85, 0.45, 0.55), fill=(0.99, 0.94, 0.95), width=1.0)

    # Top Status & Metadata Bar
    page.draw_rect(pymupdf.Rect(25, 25, 570, 130), color=(0.8, 0.4, 0.5), width=0.75)
    page.insert_text(pymupdf.Point(35, 42), "Krankenkasse bzw. Kostenträger: Techniker Krankenkasse (TK)", fontsize=9, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(35, 60), "Name, Vorname des Versicherten: Müller, Wolfgang", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(35, 76), "geb. am: 12.03.1968", fontsize=9, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(35, 93), "Wohnanschrift: Schillerstraße 24, 80336 München", fontsize=9, fontname="helv", color=(0.0, 0.0, 0.0))

    # Right side Form Grid Numbers
    page.draw_line(pymupdf.Point(360, 25), pymupdf.Point(360, 130), color=(0.8, 0.4, 0.5), width=0.5)
    page.insert_text(pymupdf.Point(370, 42), "Kassen-Nr.: 1080181", fontsize=8.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(370, 60), "Versicherten-Nr.: A123456789", fontsize=8.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(370, 76), "Status: 10001", fontsize=8.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(370, 93), "Betriebsstätten-Nr. (BSNR): 729104800", fontsize=8.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(370, 110), "Arzt-Nr. (LANR): 839201844", fontsize=8.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(370, 124), "Ausstellungsdatum: 15.10.2024", fontsize=8.5, fontname="helv", color=(0.0, 0.0, 0.0))

    # Medication Field (Rp.)
    page.draw_rect(pymupdf.Rect(25, 140, 360, 390), color=(0.8, 0.4, 0.5), width=0.75)
    page.insert_text(pymupdf.Point(35, 168), "Rp.", fontsize=18, fontname="times-bold", color=(0.2, 0.2, 0.2))

    page.insert_text(pymupdf.Point(45, 195), "1. Ibuprofen 600 mg Filmtabletten N2", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(60, 212), "Dosierung: 1-0-1 unzerkaut nach dem Essen", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))

    page.insert_text(pymupdf.Point(45, 245), "2. Pantoprazol 20 mg magensaftresistente Tabletten N1", fontsize=10, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(60, 262), "Dosierung: 1-0-0 morgens vor dem Frühstück", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))

    # Doctor Stamp & Signature Block (Bottom Right)
    page.draw_rect(pymupdf.Rect(370, 140, 570, 390), color=(0.8, 0.4, 0.5), width=0.75)
    page.insert_text(pymupdf.Point(380, 160), "Vertragsarztstempel / Unterschrift des Arztes:", fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))
    page.insert_text(pymupdf.Point(385, 185), "Praxis Dr. med. Heinrich Weber", fontsize=9.5, fontname="helv", color=(0.0, 0.0, 0.0))
    page.insert_text(pymupdf.Point(385, 202), "Facharzt für Allgemeinmedizin", fontsize=8.5, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(385, 218), "Leopoldstraße 55, 80802 München", fontsize=8.5, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(385, 234), "Telefon: 089 381920", fontsize=8.5, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text(pymupdf.Point(385, 280), "Unterschrift: Dr. med. Heinrich Weber", fontsize=9, fontname="helv", color=(0.0, 0.0, 0.0))

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"[+] Created EU/DE Vector Prescription: {output_path}")


def main():
    os.makedirs(HOLDOUT_DIR, exist_ok=True)
    print("=" * 75)
    print("GENERATING AUTHENTIC MULTINATIONAL DIGITAL VECTOR EMR PRESCRIPTIONS")
    print(f"Target Directory: {HOLDOUT_DIR}")
    print("=" * 75)

    create_us_vector_prescription(os.path.join(HOLDOUT_DIR, "vector_us_hipaa_rx.pdf"))
    create_uk_vector_prescription(os.path.join(HOLDOUT_DIR, "vector_uk_nhs_fp10.pdf"))
    create_au_vector_prescription(os.path.join(HOLDOUT_DIR, "vector_au_pbs_rx.pdf"))
    create_eu_de_vector_prescription(os.path.join(HOLDOUT_DIR, "vector_eu_de_muster16.pdf"))

    print("=" * 75)
    print("[✓] Successfully generated all 4 multinational digital vector prescriptions.")
    print("=" * 75)


if __name__ == "__main__":
    main()
