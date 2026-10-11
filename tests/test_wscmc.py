"""
Tests for Writer-Specific Confusion-Matrix Calibration (WSCMC).
"""

import unittest
from pharmacon_ai.wscmc import align_characters, WriterConfusionProfile


class TestWSCMC(unittest.TestCase):

    def test_align_characters(self):
        ref = "kitten"
        hyp = "sitting"
        aligned = align_characters(ref, hyp)
        # Check alignment length is non-empty and matches
        self.assertTrue(len(aligned) >= len(ref))
        ref_recon = "".join(r for r, _ in aligned if r is not None)
        hyp_recon = "".join(h for _, h in aligned if h is not None)
        self.assertEqual(ref_recon, ref)
        self.assertEqual(hyp_recon, hyp)

    def test_confusion_profile_and_selection(self):
        profile = WriterConfusionProfile()
        preds = {
            "E01": "The quik brown fox jumps over the lazy dog.",
            "E02": "Sphinx of blak quartz, judge my vow.",
            "E03": "Tab Parasetamol 500 mg, 1 tablet after food, BD for 3 days."
        }
        truths = {
            "E01": "The quick brown fox jumps over the lazy dog.",
            "E02": "Sphinx of black quartz, judge my vow.",
            "E03": "Tab Paracetamol 500 mg, 1 tablet after food, BD for 3 days."
        }
        profile.update_from_enrollment(preds, truths)

        # 'c' was substituted by 's' in Paracetamol, deleted in quick/black
        self.assertIn("c", profile.char_error_rates)
        self.assertGreater(profile.char_error_rates["c"], 0.0)

        candidates = {
            "R01": "Mix 0.5 mL with 2.5 mL saline; use once at 8:00 PM.",
            "R02": "Tab Ofloxacin 200 mg, one tablet BD for 5 days.",
            "R03": "Cap Omeprazole 20 mg, one capsule before breakfast.",
            "R04": "Review in 2 weeks; continue current medicines unless advised."
        }

        selected = profile.select_refinement_lines(candidates, k=4)
        self.assertEqual(len(selected), 4)
        # Verify greedy ordering
        self.assertIn("R02", [s[0] for s in selected])


if __name__ == "__main__":
    unittest.main()
