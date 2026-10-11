"""
Unit tests for Clinical Inventory & Dispensation Engine.
"""

import os
import tempfile
import unittest
from pharmacon_ai.inventory import init_db, get_inventory_items, dispense_prescription, get_patient_medications


class TestInventory(unittest.TestCase):

    def setUp(self):
        self.tmp_fd, self.tmp_db = tempfile.mkstemp(suffix=".db")
        init_db(self.tmp_db)

    def tearDown(self):
        os.close(self.tmp_fd)
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)

    def test_init_db_and_seeding(self):
        items = get_inventory_items(self.tmp_db)
        self.assertGreater(len(items), 0)
        has_paracetamol = any("Paracetamol" in r["generic_name"] for r in items)
        self.assertTrue(has_paracetamol)

    def test_dispense_and_stock_decrement(self):
        # Initial stock of Paracetamol
        items_before = {r["generic_name"]: r["stock_units"] for r in get_inventory_items(self.tmp_db)}
        pcm_stock_before = items_before.get("Paracetamol", 500)

        # Dispense: 1-0-1 for 3 days = 6 tablets
        rx_data = {
            "generic_name": "Paracetamol",
            "form": "Tab",
            "strength": "500 mg",
            "frequency": "BD",
            "duration": "3 days",
            "gate_status": "HIGH_CONFIDENCE_MATCH"
        }
        res = dispense_prescription("W01", "P-101", rx_data, raw_ocr_text="Tab Paracetamol 500 mg BD", db_path=self.tmp_db)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["quantity_dispensed"], 6)
        self.assertEqual(res["remaining_stock"], pcm_stock_before - 6)

        # Check patient records
        patient_rx = get_patient_medications("P-101", db_path=self.tmp_db)
        self.assertEqual(len(patient_rx), 1)
        self.assertEqual(patient_rx[0]["resolved_generic"], "Paracetamol")


if __name__ == "__main__":
    unittest.main()
