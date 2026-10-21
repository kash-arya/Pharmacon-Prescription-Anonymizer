"""
Tests for Formulary Matcher and Clinical Entity Extractor (Stages 5 & 6).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pharmacon_ai.formulary import (
    FormularyMatcher,
    extract_prescription_entities,
    levenshtein_similarity,
    jaro_winkler_similarity,
    consonant_skeleton,
)


class TestFormularyMetrics(unittest.TestCase):
    def test_levenshtein_similarity(self):
        self.assertAlmostEqual(levenshtein_similarity("Paracetamol", "Paracetamol"), 1.0)
        self.assertAlmostEqual(levenshtein_similarity("", ""), 1.0)
        self.assertAlmostEqual(levenshtein_similarity("A", ""), 0.0)
        # 1 edit distance in 11 letters
        sim = levenshtein_similarity("Paracetamol", "Paracetomol")
        self.assertAlmostEqual(sim, 10 / 11)

    def test_jaro_winkler_similarity(self):
        # Prefix match boosts Jaro-Winkler
        jw = jaro_winkler_similarity("Metformin", "Metform")
        lev = levenshtein_similarity("Metformin", "Metform")
        self.assertGreaterEqual(jw, lev)
        self.assertAlmostEqual(jaro_winkler_similarity("Amox", "Amox"), 1.0)

    def test_consonant_skeleton(self):
        # Preserves initial letter, strips subsequent vowels
        self.assertEqual(consonant_skeleton("Paracetamol"), "prctml")
        self.assertEqual(consonant_skeleton("Metformin"), "mtfrmn")
        self.assertEqual(consonant_skeleton("Amoxicillin"), "amxcln")
        self.assertEqual(consonant_skeleton("Cetirizine"), "ctrzn")
        self.assertEqual(consonant_skeleton("Clopidogrel"), "clpdgrl")


class TestEntityParser(unittest.TestCase):
    def test_standard_oral_solid(self):
        text = "Tab Paracetamol 650 mg SOS after food; maximum 3 tablets/day."
        res = extract_prescription_entities(text)
        self.assertEqual(res["form"], "Tab")
        self.assertEqual(res["strength"], "650 mg")
        self.assertEqual(res["frequency"], "SOS")
        self.assertIn("after food", res["instructions"])
        self.assertIn("Paracetamol", res["drug_candidate_text"])

    def test_liquid_suspension(self):
        text = "Syp Amoxicillin 250 mg/5 mL, 5 mL TDS for 5 days."
        res = extract_prescription_entities(text)
        self.assertEqual(res["form"], "Syp")
        self.assertEqual(res["strength"], "250 mg/5 mL")
        self.assertEqual(res["frequency"], "TDS")
        self.assertEqual(res["duration"], "5 days")
        self.assertIn("Amoxicillin", res["drug_candidate_text"])

    def test_ophthalmic_drops(self):
        text = "Eye drops Tobramycin 0.3%, 1 drop TDS for 5 days."
        res = extract_prescription_entities(text)
        self.assertEqual(res["form"], "Eye drops")
        self.assertEqual(res["strength"], "0.3%")
        self.assertEqual(res["frequency"], "TDS")
        self.assertEqual(res["duration"], "5 days")
        self.assertIn("Tobramycin", res["drug_candidate_text"])

    def test_high_potency_vitamins(self):
        text = "Cap Cholecalciferol 60,000 IU, once weekly for 8 weeks."
        res = extract_prescription_entities(text)
        self.assertEqual(res["form"], "Cap")
        self.assertEqual(res["strength"], "60,000 IU")
        self.assertEqual(res["frequency"], "once weekly")
        self.assertEqual(res["duration"], "8 weeks")
        self.assertIn("Cholecalciferol", res["drug_candidate_text"])


class TestFormularyMatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = FormularyMatcher()

    def test_ground_truth_test_lines_top1_resolution(self):
        """All held-out lines T01-T08 must resolve to their correct generic at Top-1."""
        test_lines = {
            "T01": ("Tab Paracetamol 650 mg SOS after food; maximum 3 tablets/day.", "Paracetamol"),
            "T02": ("Syp Amoxicillin 250 mg/5 mL, 5 mL TDS for 5 days.", "Amoxicillin"),
            "T03": ("Tab Metformin SR 500 mg, one tablet with the evening meal.", "Metformin"),
            "T04": ("Tab Cetirizine 10 mg, one tablet HS for 7 nights.", "Cetirizine"),
            "T05": ("Tab Quetiapine 25 mg, 0-0-1 at bedtime for 14 days.", "Quetiapine"),
            "T06": ("Tab Clopidogrel 75 mg, one tablet OD after lunch.", "Clopidogrel"),
            "T07": ("Cap Pregabalin 75 mg, one capsule HS for 10 days.", "Pregabalin"),
            "T08": ("Eye drops Tobramycin 0.3%, 1 drop TDS for 5 days.", "Tobramycin"),
        }

        for lid, (line_text, expected_generic) in test_lines.items():
            res = self.matcher.resolve(line_text, top_k=3)
            top_1 = res["top_candidates"][0]
            self.assertEqual(
                top_1["generic_name"], expected_generic,
                f"Failed for {lid}: expected {expected_generic}, got {top_1['generic_name']} (conf={top_1['confidence']})"
            )
            self.assertGreaterEqual(top_1["confidence"], 0.80)
            self.assertEqual(res["safety_gate"]["status"], "HIGH_CONFIDENCE_MATCH")

    def test_noisy_ocr_recovery(self):
        """Verify that degraded OCR outputs match the intended medication."""
        noisy_samples = [
            ("Tab Paracetmol 650mg SOS", "Paracetamol"),
            ("Cap Amoxcilin 500 mg TDS", "Amoxicillin"),
            ("Tab Metfrmin 500mg 1-0-1", "Metformin"),
            ("Syp Cetrzn 5 mg/5ml HS", "Cetirizine"),
            ("Tab Clopidogrl 75 mg OD", "Clopidogrel"),
            ("Cap Prgbalin 75mg HS", "Pregabalin"),
            ("Eye drops Tobramycn 0.3%", "Tobramycin"),
            ("Tab Amlodpn 5 mg morning", "Amlodipine"),
            ("Cap Omeprzole 20 mg AC", "Omeprazole"),
            ("Tab Pantoprzol 40 mg AC", "Pantoprazole"),
        ]

        for noisy_text, expected in noisy_samples:
            res = self.matcher.resolve(noisy_text, top_k=3)
            top_1 = res["top_candidates"][0]["generic_name"]
            self.assertEqual(
                top_1, expected,
                f"Noisy OCR recovery failed for '{noisy_text}': got {top_1} instead of {expected}"
            )

    def test_brand_name_synonym_resolution(self):
        """Clinical brand names must map to their generic NLEM equivalents."""
        brand_samples = [
            ("Tab Dolo 650 mg SOS", "Paracetamol"),
            ("Cap Calpol 500 mg", "Paracetamol"),
            ("Tab Glycomet 500 mg BD", "Metformin"),
            ("Tab Stamlo 5 mg OD", "Amlodipine"),
            ("Tab Telma 40 mg OD", "Telmisartan"),
            ("Tab Pan 40 mg AC", "Pantoprazole"),
            ("Cap Omez 20 mg", "Omeprazole"),
            ("Tab Cetzine 10 mg HS", "Cetirizine"),
            ("Eye drops Vigamox 0.5%", "Moxifloxacin"),
        ]

        for brand_text, expected in brand_samples:
            res = self.matcher.resolve(brand_text, top_k=3)
            top_1 = res["top_candidates"][0]["generic_name"]
            self.assertEqual(
                top_1, expected,
                f"Brand resolution failed for '{brand_text}': got {top_1} instead of {expected}"
            )

    def test_clinical_form_mismatch_penalty(self):
        """Incompatible dosage form must be heavily penalized."""
        # Eye drops Metformin does not exist clinically
        res = self.matcher.resolve("Eye drops Metformin 500 mg")
        for cand in res["top_candidates"]:
            if cand["generic_name"] == "Metformin":
                self.assertLess(cand["scores"]["form_score"], 0.5)

    def test_abstention_on_gibberish(self):
        """Nonsense OCR or non-medical strings must trigger the safety abstention gate."""
        nonsense = "Sphinx of black quartz judge my vow Rx 24/09"
        res = self.matcher.resolve(nonsense, confidence_threshold=0.60)
        self.assertTrue(
            res["safety_gate"]["abstain"],
            f"Failed to abstain on gibberish. Status was: {res['safety_gate']['status']}"
        )
        self.assertEqual(res["safety_gate"]["status"], "ABSTAIN_UNRECOGNIZED_PRESCRIPTION")


if __name__ == "__main__":
    unittest.main()
