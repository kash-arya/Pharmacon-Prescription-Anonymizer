"""
Unit and Integration Test Suite for Pharmacon Prescription Anonymizer.
"""

import os
import sys
import unittest
import json
import tempfile
import pymupdf
from PIL import Image

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.raster_checker import inspect_pdf_structure
from src.vector_redactor import VectorPDFRedactor
from src.image_redactor import LocalImageRedactor, resolve_tessdata_path
from src.forensics import ForensicAuditor
from src.vlm_grounder import MultimodalVisualGrounder
from benchmarks.benchmark import run_benchmark, BENCHMARK_CASES
from viewer.viewer import build_viewer_data, generate_html
from anonymize import process_file


class TestPrescriptionAnonymizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.samples_dir = os.path.join(cls.repo_dir, "data", "samples")
        cls.vector_pdf = os.path.join(cls.samples_dir, "sample_digital_vector_prescription.pdf")
        cls.scanned_pdf_1 = os.path.join(cls.samples_dir, "sample_prescription_1.pdf")
        cls.scanned_pdf_2 = os.path.join(cls.samples_dir, "sample_prescription_2.pdf")
        cls.patient_pdf = os.path.join(cls.samples_dir, "sample_patient_details_1.pdf")
        cls.sample_image = os.path.join(cls.samples_dir, "extracted_images", "sample_prescription_1_p1.png")
        cls.temp_dir = tempfile.mkdtemp(prefix="test_pharmacon_")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temp_dir") and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Raster Checker Tests
    # -----------------------------------------------------------------------
    def test_raster_checker_digital_vector(self):
        struct = inspect_pdf_structure(self.vector_pdf)
        self.assertEqual(struct["format_type"], "VECTOR_DIGITAL")
        self.assertGreater(struct["total_char_count"], 100)

    def test_raster_checker_scanned_pdf(self):
        struct = inspect_pdf_structure(self.scanned_pdf_1)
        self.assertEqual(struct["format_type"], "SCANNED_RASTER")
        self.assertEqual(struct["total_char_count"], 0)

    def test_raster_checker_image_file(self):
        struct = inspect_pdf_structure(self.sample_image)
        self.assertEqual(struct["format_type"], "SCANNED_RASTER")
        self.assertEqual(struct["page_count"], 1)

    # -----------------------------------------------------------------------
    # Vector Redactor Tests
    # -----------------------------------------------------------------------
    def test_vector_redactor_execution(self):
        redactor = VectorPDFRedactor()
        out_pdf = os.path.join(self.temp_dir, "test_vector_out.pdf")
        report = redactor.redact(self.vector_pdf, out_pdf)

        self.assertGreaterEqual(report["total_entities_redacted"], 10)
        self.assertTrue(os.path.exists(out_pdf))

        # Verify no PII leak in output PDF
        doc = pymupdf.open(out_pdf)
        text = doc[0].get_text().lower()
        doc.close()

        for pii in ["rajesh sharma", "anita desai", "mci-44912", "98110 23456", "dr.sharma@"]:
            self.assertNotIn(pii, text)

        # Verify clinical pharmacology preserved
        for clin in ["telmisartan 40 mg", "amlodipine 5 mg", "paracetamol 650 mg"]:
            self.assertIn(clin, text)

    # -----------------------------------------------------------------------
    # Local Image Redactor Tests
    # -----------------------------------------------------------------------
    def test_tessdata_resolution(self):
        tessdata = resolve_tessdata_path()
        self.assertTrue(os.path.isdir(tessdata))
        self.assertTrue(os.path.exists(os.path.join(tessdata, "eng.traineddata")))

    def test_image_redactor_scanned_pdf(self):
        redactor = LocalImageRedactor()
        out_pdf = os.path.join(self.temp_dir, "test_scanned_out.pdf")
        report = redactor.redact(self.scanned_pdf_1, out_pdf)

        self.assertGreaterEqual(report["total_entities_redacted"], 5)
        self.assertTrue(os.path.exists(out_pdf))

        # Verify OCR on redacted document has zero PII leaks
        doc = pymupdf.open(out_pdf)
        page0 = doc[0]
        tp = page0.get_textpage_ocr(language="eng", dpi=150, tessdata=redactor.tessdata_dir)
        text = page0.get_text(textpage=tp).lower()
        doc.close()

        for pii in ["maria sharapova", "john smith", "non-important street", "5/11/2022"]:
            self.assertNotIn(pii, text)

        # Verify clinical preservation
        for clin in ["prednisone", "20 mg"]:
            self.assertIn(clin, text)
        self.assertTrue("lialda" in text or "liatda" in text or "2.4 gram" in text)

    def test_image_redactor_standalone_image(self):
        redactor = LocalImageRedactor()
        out_img = os.path.join(self.temp_dir, "test_image_out.png")
        report = redactor.redact(self.sample_image, out_img)

        self.assertGreaterEqual(report["total_entities_redacted"], 5)
        self.assertTrue(os.path.exists(out_img))

        # Check preserved dimensions
        with Image.open(self.sample_image) as orig_img, Image.open(out_img) as res_img:
            self.assertEqual(orig_img.size, res_img.size)

    # -----------------------------------------------------------------------
    # Benchmark Evaluation Tests
    # -----------------------------------------------------------------------
    def test_automated_benchmark_all_passed(self):
        out_json = os.path.join(self.temp_dir, "test_bench_results.json")
        out_dir = os.path.join(self.temp_dir, "bench_outputs")

        results = run_benchmark(
            samples_dir=self.samples_dir,
            output_dir=out_dir,
            results_json_path=out_json
        )

        self.assertEqual(results["summary"]["status"], "PASS")
        self.assertEqual(results["summary"]["pass_rate_pct"], 100.0)
        self.assertEqual(results["summary"]["average_pii_recall_pct"], 100.0)
        self.assertEqual(results["summary"]["total_pii_leaks"], 0)
        self.assertEqual(results["summary"]["average_clinical_preservation_pct"], 100.0)

    # -----------------------------------------------------------------------
    # Viewer Generation Tests
    # -----------------------------------------------------------------------
    def test_viewer_html_generation(self):
        temp_cache = os.path.join(self.temp_dir, "viewer_cache")
        data = build_viewer_data(self.samples_dir, temp_cache)
        self.assertGreaterEqual(len(data["documents"]), 4)

        html = generate_html(data)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Pharmacon Prescription Anonymizer", html)
        self.assertTrue("data:image/jpeg;base64," in html or "data:image/png;base64," in html)
        self.assertIn("Side-by-Side", html)
        self.assertIn("Interactive Slider", html)

    # -----------------------------------------------------------------------
    # Forensic Security & Multimodal Tests
    # -----------------------------------------------------------------------
    def test_forensic_auditor_compliance_certificate(self):
        redactor = VectorPDFRedactor()
        out_pdf = os.path.join(self.temp_dir, "test_forensic_out.pdf")
        redactor.redact(self.vector_pdf, out_pdf)

        auditor = ForensicAuditor()
        cert = auditor.generate_compliance_certificate(
            out_pdf,
            known_pii_strings=["Rajesh Sharma", "Anita Desai", "98110 23456", "MCI-44912"]
        )
        self.assertEqual(cert["compliance_status"], "CERTIFIED_SECURE")
        self.assertEqual(cert["byte_stream_leak_count"], 0)
        self.assertTrue(cert["metadata_sanitized"])

    def test_multimodal_visual_grounder(self):
        grounder = MultimodalVisualGrounder()
        # Test on real phone prescription image
        phone_img = os.path.join(self.repo_dir, "data", "real_phone_prescriptions", "Aryan March 25.jpg")
        if os.path.exists(phone_img):
            targets = grounder.detect_visual_pii(phone_img)
            self.assertIsInstance(targets, list)

    def test_batch_directory_processing_logic(self):
        out_batch_dir = os.path.join(self.temp_dir, "test_batch_run")
        os.makedirs(out_batch_dir, exist_ok=True)
        # Test process_file helper
        out_sample = os.path.join(out_batch_dir, "sample_out.pdf")
        rep = process_file(self.vector_pdf, out_sample)
        self.assertEqual(rep["format_type"], "VECTOR_DIGITAL")
        self.assertTrue(os.path.exists(out_sample))

    # -----------------------------------------------------------------------
    # FastAPI Microservice Integration Tests
    # -----------------------------------------------------------------------
    def test_api_health_endpoint(self):
        from fastapi.testclient import TestClient
        from api.server import app
        client = TestClient(app)
        res = client.get("/health")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn(body["status"], ("ok", "healthy"))
        self.assertTrue(body["tesseract_model_present"])

    def test_api_anonymize_json_endpoint(self):
        from fastapi.testclient import TestClient
        from api.server import app
        client = TestClient(app)
        with open(self.vector_pdf, "rb") as f:
            res = client.post("/api/v1/anonymize?return_type=json", files={"file": ("sample.pdf", f, "application/pdf")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["format_type"], "VECTOR_DIGITAL")
        self.assertGreaterEqual(data["total_entities_redacted"], 10)
        self.assertTrue(data["data_uri"].startswith("data:application/pdf;base64,"))

    def test_api_anonymize_file_stream_endpoint(self):
        from fastapi.testclient import TestClient
        from api.server import app
        client = TestClient(app)
        with open(self.vector_pdf, "rb") as f:
            res = client.post("/api/v1/anonymize?return_type=file", files={"file": ("sample.pdf", f, "application/pdf")})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "application/pdf")
        self.assertGreater(len(res.content), 500)
        self.assertIn("x-redacted-count", res.headers)

    def test_api_web_interface_endpoints(self):
        from fastapi.testclient import TestClient
        from api.server import app
        client = TestClient(app)

        res_root = client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("text/html", res_root.headers.get("content-type", ""))
        self.assertIn("Pharmacon Anonymizer", res_root.text)
        self.assertIn("split-container", res_root.text)

        res_app = client.get("/app")
        self.assertEqual(res_app.status_code, 200)
        self.assertIn("text/html", res_app.headers.get("content-type", ""))

        res_api = client.get("/api")
        self.assertEqual(res_api.status_code, 200)
        self.assertEqual(res_api.json()["version"], "2.0.0")

    def test_api_anonymize_preview_endpoint(self):
        from fastapi.testclient import TestClient
        from api.server import app
        client = TestClient(app)
        with open(self.vector_pdf, "rb") as f:
            res = client.post("/api/v1/anonymize/preview", files={"file": ("sample.pdf", f, "application/pdf")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["format_type"], "VECTOR_DIGITAL")
        self.assertGreaterEqual(data["total_entities_redacted"], 10)
        self.assertTrue(data["original_image_base64"].startswith("data:image/png;base64,"))
        self.assertTrue(data["redacted_image_base64"].startswith("data:image/png;base64,"))
        self.assertTrue(data["side_by_side_image_base64"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(data["download_file_base64"].startswith("data:application/pdf;base64,"))
        self.assertTrue(data["original_image"].startswith("data:image/"))
        self.assertTrue(data["redacted_image"].startswith("data:image/"))

    def test_api_anonymize_file_direct_download(self):
        from fastapi.testclient import TestClient
        from api.server import app
        client = TestClient(app)
        with open(self.vector_pdf, "rb") as f:
            res = client.post("/anonymize/file", files={"file": ("prescription_sample.pdf", f, "application/pdf")})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "application/pdf")
        self.assertEqual(res.headers.get("content-disposition"), 'attachment; filename="prescription_sample_anonymized.pdf"')
        self.assertIn("x-redacted-count", res.headers)

    def test_api_anonymize_batch_endpoint(self):
        import io
        import zipfile
        from fastapi.testclient import TestClient
        from api.server import app

        # Prepare in-memory test zip containing vector PDF and raster image
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.write(self.vector_pdf, arcname="prescription_apollo.pdf")
            zf.write(self.sample_image, arcname="images/rx_slip.png")
        zip_buf.seek(0)

        client = TestClient(app)
        res = client.post(
            "/api/v1/anonymize/batch",
            files={"file": ("batch_prescriptions.zip", zip_buf.getvalue(), "application/zip")}
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "application/zip")
        self.assertEqual(res.headers.get("x-total-documents"), "2")
        self.assertGreaterEqual(int(res.headers.get("x-total-entities-redacted", 0)), 15)

        # Inspect returned output ZIP
        out_buf = io.BytesIO(res.content)
        with zipfile.ZipFile(out_buf, "r") as out_zf:
            namelist = out_zf.namelist()
            self.assertIn("batch_audit_summary.json", namelist)
            self.assertIn("prescription_apollo_anonymized.pdf", namelist)
            self.assertIn("rx_slip_anonymized.png", namelist)

            summary = json.loads(out_zf.read("batch_audit_summary.json").decode("utf-8"))
            self.assertEqual(summary["total_documents"], 2)
            self.assertEqual(summary["successful_redactions"], 2)
            self.assertGreaterEqual(summary["total_entities_redacted"], 15)

    def test_api_error_handling(self):
        from fastapi.testclient import TestClient
        from api.server import app
        client = TestClient(app)

        # 1. Unsupported extension
        res_bad_ext = client.post(
            "/api/v1/anonymize",
            files={"file": ("malware.exe", b"test content", "application/octet-stream")}
        )
        self.assertEqual(res_bad_ext.status_code, 400)
        self.assertIn("Unsupported file format", res_bad_ext.json()["detail"])

        # 2. Empty file upload
        res_empty = client.post(
            "/api/v1/anonymize",
            files={"file": ("empty.pdf", b"", "application/pdf")}
        )
        self.assertEqual(res_empty.status_code, 400)
        self.assertIn("empty", res_empty.json()["detail"].lower())

        # 3. Non-zip upload to batch endpoint
        res_bad_batch = client.post(
            "/api/v1/anonymize/batch",
            files={"file": ("not_a_zip.pdf", b"pdf content", "application/pdf")}
        )
        self.assertEqual(res_bad_batch.status_code, 400)
        self.assertIn("must be a .zip archive", res_bad_batch.json()["detail"])


    def test_multinational_jurisdiction_detector(self):
        from src.jurisdiction import JurisdictionDetector
        detector = JurisdictionDetector()
        
        self.assertEqual(detector.detect("DEA REG NO: AB1234567 NPI: 1234567890"), "US")
        self.assertEqual(detector.detect("NHS NUMBER: 485 777 3456 GMC 6123456 FP10"), "UK")
        self.assertEqual(detector.detect("MEDICARE NO: 2345 67890 1 PBS PRESCRIBER NO: 456789A AHPRA"), "AU")
        self.assertEqual(detector.detect("Versicherten-Nr. T123456789 BSNR: 723456700 LANR: 987654321 Muster 16"), "EU")
        self.assertEqual(detector.detect("UHID: AP2024-88912 ABHA ID: 91-4567-8912-3456 NMC-45892"), "IN")
        self.assertEqual(detector.detect("Generic hospital slip without statutory tags"), "GLOBAL")

    def test_ocr_preprocess_module(self):
        from src.ocr_preprocess import resolve_tessdata_path, get_ocr_languages, reconstruct_visual_lines
        tess_dir = resolve_tessdata_path()
        self.assertTrue(os.path.isdir(tess_dir))
        langs = get_ocr_languages(tess_dir)
        self.assertIn("eng", langs)

        words = [
            (10.0, 10.0, 50.0, 25.0, "Patient:"),
            (55.0, 10.0, 100.0, 25.0, "John"),
            (105.0, 10.0, 140.0, 25.0, "Doe"),
            (10.0, 35.0, 40.0, 50.0, "Age:"),
            (45.0, 35.0, 65.0, 50.0, "42")
        ]
        lines = reconstruct_visual_lines(words)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["text"], "Patient: John Doe")
        self.assertEqual(lines[1]["text"], "Age: 42")

    def test_barcode_detector_module(self):
        from src.barcode_detector import detect_qr_and_barcodes
        doc = pymupdf.open(self.vector_pdf)
        page = doc[0]
        boxes = detect_qr_and_barcodes(page)
        self.assertIsInstance(boxes, list)
        doc.close()

    def test_form_parser_module(self):
        from src.form_parser import expand_person_span_on_line, detect_handwritten_fill_in_zone
        import numpy as np

        line_objs = [{
            "rect": pymupdf.Rect(10, 10, 150, 25),
            "text": "Mr. John Doe",
            "words": [
                (pymupdf.Rect(10, 10, 35, 25), "Mr."),
                (pymupdf.Rect(40, 10, 80, 25), "John"),
                (pymupdf.Rect(85, 10, 120, 25), "Doe")
            ]
        }]
        john_rect = pymupdf.Rect(40, 10, 80, 25)
        expanded = expand_person_span_on_line(john_rect, line_objs)
        self.assertLessEqual(expanded.x0, 10.0)
        self.assertGreaterEqual(expanded.x1, 120.0)

        # Blank image detection returns None
        blank_bgr = np.full((200, 200, 3), 255, dtype=np.uint8)
        zone = pymupdf.Rect(10, 10, 100, 50)
        ink = detect_handwritten_fill_in_zone(blank_bgr, zone, 1.0, 1.0)
        self.assertIsNone(ink)

    def test_consecutive_uploads_memory_and_download_integrity(self):
        """
        Verifies that consecutive uploads under RENDER_LOW_MEMORY=1 do not cause RSS memory leak,
        and that download_file_base64 and /anonymize/file produce non-empty valid binary outputs.
        """
        import base64
        from fastapi.testclient import TestClient
        from api.server import app

        def get_current_rss_mb():
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return int(line.split()[1]) / 1024.0
            except Exception:
                pass
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        old_low_mem = os.environ.get("RENDER_LOW_MEMORY")
        os.environ["RENDER_LOW_MEMORY"] = "1"
        try:
            client = TestClient(app)
            phone_dir = os.path.join(self.repo_dir, "data", "real_phone_prescriptions")
            files_to_test = [
                ("Aryan March 26.jpg", os.path.join(phone_dir, "Aryan March 26.jpg"), "image/jpeg"),
                ("Aryan Sept 24.jpg", os.path.join(phone_dir, "Aryan Sept 24.jpg"), "image/jpeg"),
                ("sample_prescription_1.pdf", self.scanned_pdf_1, "application/pdf")
            ]

            rss_readings = []
            for name, path, mime in files_to_test:
                with open(path, "rb") as f:
                    content = f.read()
                res = client.post("/api/v1/anonymize/preview", files={"file": (name, content, mime)})
                self.assertEqual(res.status_code, 200, f"Preview failed for {name}: {res.text}")
                data = res.json()
                self.assertIn("download_file_base64", data)
                dl_b64 = data["download_file_base64"]
                self.assertTrue(dl_b64.startswith("data:"))
                header, b64_data = dl_b64.split(",", 1)
                raw_bytes = base64.b64decode(b64_data)
                self.assertGreater(len(raw_bytes), 5000, f"Download payload too small for {name}")
                self.assertEqual(data["triage_status"], "AUTO_APPROVED", f"Expected AUTO_APPROVED for {name}, got {data['triage_status']}")
                if name.endswith(".pdf"):
                    self.assertLess(len(raw_bytes), 400000, f"Redacted PDF size too large: {len(raw_bytes)} bytes")
                rss = get_current_rss_mb()
                rss_readings.append(rss)

            # Check /anonymize/file streaming binary endpoint
            with open(self.scanned_pdf_1, "rb") as f:
                pdf_bytes = f.read()
            res_file = client.post("/api/v1/anonymize/file", files={"file": ("sample_prescription_1.pdf", pdf_bytes, "application/pdf")})
            self.assertEqual(res_file.status_code, 200)
            self.assertGreater(len(res_file.content), 5000)
            self.assertLess(len(res_file.content), 400000)

            # Verify RSS does not grow significantly between Request 1 and Request 2
            growth_1_to_2 = rss_readings[1] - rss_readings[0]
            self.assertLess(growth_1_to_2, 35.0, f"RSS grew unexpectedly between Request 1 and 2: +{growth_1_to_2:.2f} MB")
        finally:
            if old_low_mem is not None:
                os.environ["RENDER_LOW_MEMORY"] = old_low_mem
            else:
                os.environ.pop("RENDER_LOW_MEMORY", None)


if __name__ == "__main__":
    unittest.main()
