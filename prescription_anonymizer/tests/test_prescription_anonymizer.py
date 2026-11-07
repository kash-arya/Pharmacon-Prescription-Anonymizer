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
        self.assertIn(res_bad_ext.status_code, (400, 415))
        if res_bad_ext.status_code == 415:
            self.assertEqual(res_bad_ext.json().get("triage_status"), "IGNORED_UNSUPPORTED_FORMAT")
        else:
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
            import api.server
            api.server._image_engine = None
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
                del data, raw_bytes, content, res
                import gc
                gc.collect()
                try:
                    import ctypes
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
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
            import api.server
            api.server._image_engine = None
            if old_low_mem is not None:
                os.environ["RENDER_LOW_MEMORY"] = old_low_mem
            else:
                os.environ.pop("RENDER_LOW_MEMORY", None)

    def test_right_aligned_multiline_provider_block_propagation(self):
        """
        Verifies that right-aligned multi-line provider blocks with unlabelled qualification
        and address lines are 100% masked end-to-end via scale-invariant vertical block propagation.
        """
        from PIL import Image, ImageDraw, ImageFont
        img_w, img_h = 1000, 1200
        img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        provider_lines = [
            "Dr. John Doe, M.D.",
            "Consultant Physician",
            "452 Medical Plaza Blvd",
            "Suite 300, New York, NY",
            "Tel: (555) 019-2834"
        ]
        y = 60
        for l in provider_lines:
            bbox = draw.textbbox((0, 0), l, font=font)
            text_w = bbox[2] - bbox[0]
            x = 940 - text_w
            draw.text((x, y), l, fill=(0, 0, 0), font=font)
            y += 35

        test_img = os.path.join(self.temp_dir, "test_provider_block.png")
        out_pdf = os.path.join(self.temp_dir, "test_provider_block_out.pdf")
        img.save(test_img)

        redactor = LocalImageRedactor()
        report = redactor.redact(test_img, out_pdf)
        self.assertGreaterEqual(report["total_entities_redacted"], 3)

        target_rects = [
            pymupdf.Rect(t["bbox"]["x0"], t["bbox"]["y0"], t["bbox"]["x1"], t["bbox"]["y1"])
            for t in report["redacted_targets"]
        ]
        scale = 1000.0 / 750.0
        words = getattr(redactor, "_last_extracted_words", [])

        # Verify key tokens across all 5 lines are masked
        check_tokens = ["Dr.", "John", "Doe,", "Consultant", "Physician", "452", "Medical", "Plaza", "Suite", "York,", "Tel:", "019-2834"]
        for tok in check_tokens:
            matches = [w for w in words if w[4] == tok]
            self.assertTrue(len(matches) > 0, f"Token {tok} not detected by OCR")
            w = matches[0]
            w_r = pymupdf.Rect(w[0] * scale, w[1] * scale, w[2] * scale, w[3] * scale)
            is_masked = any(not (w_r & tr).is_empty and (w_r & tr).get_area() > 0.4 * w_r.get_area() for tr in target_rects)
            self.assertTrue(is_masked, f"Provider token {tok} was not masked by block propagation")

    def test_clinical_barrier_guard_and_disclaimer_immunity(self):
        """
        Verifies that adjacent clinical barrier sections ('Provisional Diagnosis', 'Lab Tests', 'Rx')
        and legal disclaimer footer blocks remain 100% unredacted with zero false positives.
        """
        from PIL import Image, ImageDraw, ImageFont
        img_w, img_h = 1000, 1200
        img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        clinical_lines = [
            "Provisional Diagnosis: Acute Bronchitis",
            "Lab Tests: Complete Blood Count, Chest X-Ray",
            "Rx: Amoxicillin 500mg, 1 tab TID"
        ]
        y = 300
        for l in clinical_lines:
            draw.text((60, y), l, fill=(0, 0, 0), font=font)
            y += 45

        disclaimer_lines = [
            "Disclaimer: This electronic record is provisional in nature and not valid for medico-legal purposes.",
            "Terms & Conditions apply. For recipient's use only. Substitution allowed."
        ]
        y = 950
        for l in disclaimer_lines:
            draw.text((60, y), l, fill=(50, 50, 50), font=font)
            y += 35

        test_img = os.path.join(self.temp_dir, "test_clinical_barrier.png")
        out_pdf = os.path.join(self.temp_dir, "test_clinical_barrier_out.pdf")
        img.save(test_img)

        redactor = LocalImageRedactor()
        report = redactor.redact(test_img, out_pdf)

        target_rects = [
            pymupdf.Rect(t["bbox"]["x0"], t["bbox"]["y0"], t["bbox"]["x1"], t["bbox"]["y1"])
            for t in report["redacted_targets"]
        ]
        scale = 1000.0 / 750.0
        words = getattr(redactor, "_last_extracted_words", [])

        protected_tokens = [
            "Provisional", "Diagnosis:", "Acute", "Bronchitis",
            "Lab", "Tests:", "Complete", "Blood", "Count,", "Chest", "X-Ray",
            "Rx:", "Amoxicillin", "500mg,",
            "Disclaimer:", "provisional", "medico-legal",
            "Terms", "Conditions", "recipient's"
        ]
        for tok in protected_tokens:
            matches = [w for w in words if w[4] == tok]
            self.assertTrue(len(matches) > 0, f"Token {tok} not detected by OCR")
            w = matches[0]
            w_r = pymupdf.Rect(w[0] * scale, w[1] * scale, w[2] * scale, w[3] * scale)
            is_masked = any(not (w_r & tr).is_empty and (w_r & tr).get_area() > 0.4 * w_r.get_area() for tr in target_rects)
            self.assertFalse(is_masked, f"Clinical/Disclaimer token {tok} was erroneously masked")

    def test_magic_byte_rejection(self):
        """
        Verifies that unsupported file extensions and mismatched binary magic bytes
        are immediately rejected with HTTP 415 IGNORED_UNSUPPORTED_FORMAT.
        """
        from src.document_validator import validate_file_signature
        from fastapi.testclient import TestClient
        from api.server import app

        # Direct function validation
        self.assertEqual(validate_file_signature("video.mp4", b"\x00\x00\x00\x18ftypisom"), (False, "UNSUPPORTED_FILE_FORMAT"))
        self.assertEqual(validate_file_signature("malware.exe", b"MZ\x90\x00\x03\x00\x00\x00"), (False, "UNSUPPORTED_FILE_FORMAT"))
        self.assertEqual(validate_file_signature("archive.zip", b"PK\x03\x04\x14\x00\x00\x00"), (False, "UNSUPPORTED_FILE_FORMAT"))
        self.assertEqual(validate_file_signature("fake.pdf", b"\x00\x00\x00\x00\x00\x00"), (False, "UNSUPPORTED_FILE_FORMAT"))
        self.assertEqual(validate_file_signature("fake.png", b"GIF89a\x00\x01\x00\x01"), (False, "UNSUPPORTED_FILE_FORMAT"))
        self.assertEqual(validate_file_signature("valid.pdf", b"%PDF-1.7\n%..."), (True, "PDF"))
        self.assertEqual(validate_file_signature("valid.png", b"\x89PNG\r\n\x1a\n\x00\x00"), (True, "PNG"))
        self.assertEqual(validate_file_signature("valid.jpeg", b"\xff\xd8\xff\xe0\x00\x10"), (True, "JPEG"))

        # API HTTP 415 rejection
        client = TestClient(app)
        res_mp4 = client.post("/api/v1/anonymize/preview", files={"file": ("video.mp4", b"\x00\x00\x00\x18ftypisom", "video/mp4")})
        self.assertEqual(res_mp4.status_code, 415)
        self.assertEqual(res_mp4.json().get("triage_status"), "IGNORED_UNSUPPORTED_FORMAT")

        res_bad_pdf = client.post("/api/v1/anonymize/preview", files={"file": ("corrupt.pdf", b"NOT_A_VALID_PDF_HEADER", "application/pdf")})
        self.assertEqual(res_bad_pdf.status_code, 415)
        self.assertEqual(res_bad_pdf.json().get("triage_status"), "IGNORED_UNSUPPORTED_FORMAT")

    def test_non_prescription_invoice_rejected(self):
        """
        Verifies that non-clinical documents (e.g. tax invoices) lacking medical/pharmacological
        markers are rejected by the clinical gatekeeper without rendering or burning masks.
        """
        from src.document_validator import evaluate_prescription_gate
        from fastapi.testclient import TestClient
        from api.server import app
        from PIL import Image, ImageDraw

        invoice_text = "Tax Invoice #99, Name: Aryan, Address: 123 MD Road, Total: $50"
        is_rx, score, reasons = evaluate_prescription_gate(invoice_text)
        self.assertFalse(is_rx)
        self.assertLess(score, 2.5)
        self.assertIn("Document lacks clinical, pharmacological, or provider markers.", reasons)

        # Test through /api/v1/anonymize/preview
        im = Image.new("RGB", (800, 600), (255, 255, 255))
        d = ImageDraw.Draw(im)
        d.text((50, 50), "Tax Invoice #99", fill=(0, 0, 0))
        d.text((50, 100), "Name: Aryan Sharma", fill=(0, 0, 0))
        d.text((50, 150), "Address: 123 MD Road", fill=(0, 0, 0))
        d.text((50, 200), "Total: $50.00", fill=(0, 0, 0))
        test_inv_path = os.path.join(self.temp_dir, "test_tax_invoice.png")
        im.save(test_inv_path)

        with open(test_inv_path, "rb") as f:
            im_bytes = f.read()

        client = TestClient(app)
        res = client.post("/api/v1/anonymize/preview", files={"file": ("test_tax_invoice.png", im_bytes, "image/png")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["triage_status"], "NOT_A_PRESCRIPTION")
        self.assertEqual(data["total_entities_redacted"], 0)
        self.assertIsNone(data["original_image_base64"])
        self.assertIsNone(data["redacted_image_base64"])
        self.assertIsNone(data["download_file_base64"])

    def test_force_prescription_override(self):
        """
        Verifies that force_prescription=True bypasses the clinical gate on non-standard/low-score slips,
        completes redaction, and appends non-PII structural telemetry to logs/gate_overrides.jsonl.
        """
        import json
        from fastapi.testclient import TestClient
        from api.server import app
        from PIL import Image, ImageDraw

        im = Image.new("RGB", (800, 600), (255, 255, 255))
        d = ImageDraw.Draw(im)
        d.text((50, 50), "Bill To: John Smith", fill=(0, 0, 0))
        d.text((50, 100), "Ship To: 742 Evergreen Terrace", fill=(0, 0, 0))
        d.text((50, 150), "Tel: 555-0199", fill=(0, 0, 0))
        test_bill_path = os.path.join(self.temp_dir, "test_bill.png")
        im.save(test_bill_path)

        with open(test_bill_path, "rb") as f:
            bill_bytes = f.read()

        client = TestClient(app)
        res = client.post("/api/v1/anonymize/preview?force_prescription=true", files={"file": ("test_bill.png", bill_bytes, "image/png")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertNotEqual(data["triage_status"], "NOT_A_PRESCRIPTION")
        self.assertIsNotNone(data["download_file_base64"])

        # Check telemetry file
        candidate_dirs = [
            os.path.join(self.repo_dir, "logs"),
            os.path.join(os.getcwd(), "logs"),
        ]
        log_path = None
        for d in candidate_dirs:
            p = os.path.join(d, "gate_overrides.jsonl")
            if os.path.exists(p):
                log_path = p
                break
        self.assertIsNotNone(log_path, "Telemetry log file gate_overrides.jsonl was not created")

        with open(log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertTrue(len(lines) > 0)
        last_entry = json.loads(lines[-1])

        self.assertIn("timestamp", last_entry)
        self.assertIn("format", last_entry)
        self.assertIn("ocr_word_count", last_entry)
        self.assertIn("gate_score_before_override", last_entry)
        self.assertIn("redacted_count", last_entry)
        # Strict zero-PII check
        raw_json = json.dumps(last_entry)
        self.assertNotIn("John", raw_json)
        self.assertNotIn("Evergreen", raw_json)
        self.assertNotIn("555-0199", raw_json)

    def test_web_app_integration_suite(self):
        """
        Verifies that mounting `app` from `api.web_app` directly provides full
        FastAPI endpoint compatibility for preview, clinical gate, and force override:
        1. A real raster prescription (.png/.jpg) -> HTTP 200, triage != NOT_A_PRESCRIPTION, redacted_count > 0, non-null redacted_image_base64
        2. A non-prescription image (synthetic timetable/invoice .png) -> HTTP 200 (NOT 500!), triage == NOT_A_PRESCRIPTION
        3. The same non-prescription image with ?force_prescription=true -> HTTP 200, triage != NOT_A_PRESCRIPTION, non-null redacted_image_base64
        """
        from api.web_app import app as web_app
        from fastapi.testclient import TestClient
        from PIL import Image, ImageDraw

        client = TestClient(web_app)

        # 1. Real raster prescription (.png)
        real_rx_path = os.path.join(self.repo_dir, "data", "samples", "extracted_images", "sample_prescription_1_p1.png")
        if not os.path.exists(real_rx_path):
            real_rx_path = os.path.join(self.repo_dir, "data", "raw_public_slips", "prescription_00006.png")
        with open(real_rx_path, "rb") as f:
            rx_bytes = f.read()

        res_rx = client.post("/api/v1/anonymize/preview", files={"file": ("prescription_sample.png", rx_bytes, "image/png")})
        self.assertEqual(res_rx.status_code, 200)
        data_rx = res_rx.json()
        self.assertNotEqual(data_rx.get("triage_status"), "NOT_A_PRESCRIPTION")
        self.assertGreater(data_rx.get("total_entities_redacted", 0), 0)
        self.assertIsNotNone(data_rx.get("redacted_image_base64"))
        self.assertIsNotNone(data_rx.get("download_file_base64"))

        # 2. Non-prescription image (synthetic timetable .png)
        tt_img = Image.new("RGB", (900, 700), (255, 255, 255))
        d = ImageDraw.Draw(tt_img)
        d.text((50, 50), "Department of Computer Science - Semester V Timetable", fill=(0, 0, 0))
        d.text((50, 100), "Monday: Distributed Systems (Room 301)", fill=(0, 0, 0))
        d.text((50, 150), "Tuesday: Compiler Design Lab (Lab 4)", fill=(0, 0, 0))
        d.text((50, 200), "Wednesday: Database Management (Room 204)", fill=(0, 0, 0))
        tt_path = os.path.join(self.temp_dir, "3Q15_Timetable_SemV.png")
        tt_img.save(tt_path)
        with open(tt_path, "rb") as f:
            tt_bytes = f.read()

        res_tt = client.post("/api/v1/anonymize/preview", files={"file": ("3Q15_Timetable_SemV.png", tt_bytes, "image/png")})
        self.assertEqual(res_tt.status_code, 200, f"Expected 200, got {res_tt.status_code}: {res_tt.text}")
        data_tt = res_tt.json()
        self.assertEqual(data_tt.get("triage_status"), "NOT_A_PRESCRIPTION")
        self.assertEqual(data_tt.get("total_entities_redacted", 0), 0)
        self.assertIsNone(data_tt.get("redacted_image_base64"))

        # 3. Same non-prescription image with ?force_prescription=true
        res_tt_force = client.post("/api/v1/anonymize/preview?force_prescription=true", files={"file": ("3Q15_Timetable_SemV.png", tt_bytes, "image/png")})
        self.assertEqual(res_tt_force.status_code, 200)
        data_tt_force = res_tt_force.json()
        self.assertNotEqual(data_tt_force.get("triage_status"), "NOT_A_PRESCRIPTION")
        self.assertIsNotNone(data_tt_force.get("redacted_image_base64"))
        self.assertIsNotNone(data_tt_force.get("download_file_base64"))

    def test_concurrent_semaphore_serialization_and_503(self):
        """
        Verifies that PIPELINE_SEMAPHORE serializes concurrent requests cleanly,
        returns HTTP 503 with Retry-After: 5 on timeout, and releases the lock
        on both success and exception paths.
        """
        from api.server import app, PIPELINE_SEMAPHORE
        from api.web_app import PIPELINE_SEMAPHORE as WEB_APP_SEMAPHORE
        from fastapi.testclient import TestClient

        self.assertIsNotNone(PIPELINE_SEMAPHORE)
        self.assertIs(PIPELINE_SEMAPHORE, WEB_APP_SEMAPHORE)

        client = TestClient(app, raise_server_exceptions=False)
        real_rx_path = os.path.join(self.repo_dir, "data", "samples", "extracted_images", "sample_prescription_1_p1.png")
        if not os.path.exists(real_rx_path):
            real_rx_path = os.path.join(self.repo_dir, "data", "raw_public_slips", "prescription_00006.png")
        with open(real_rx_path, "rb") as f:
            valid_bytes = f.read()

        old_timeout = os.environ.get("PIPELINE_SEMAPHORE_TIMEOUT")
        os.environ["PIPELINE_SEMAPHORE_TIMEOUT"] = "0.05"

        try:
            # 1. Simulate semaphore acquisition (lock held by another worker)
            PIPELINE_SEMAPHORE._value = 0
            self.assertTrue(PIPELINE_SEMAPHORE.locked())

            res_busy = client.post(
                "/api/v1/anonymize/preview",
                files={"file": ("sample.png", valid_bytes, "image/png")}
            )
            self.assertEqual(res_busy.status_code, 503)
            self.assertEqual(res_busy.headers.get("retry-after"), "5")
            busy_data = res_busy.json()
            self.assertIn("Server busy processing another document", busy_data.get("detail", ""))

            # Reset semaphore lock
            PIPELINE_SEMAPHORE._value = 1
            self.assertFalse(PIPELINE_SEMAPHORE.locked())

            # 2. Verify lock release on Success path
            os.environ["PIPELINE_SEMAPHORE_TIMEOUT"] = "15.0"
            res_ok = client.post(
                "/api/v1/anonymize/preview",
                files={"file": ("sample.png", valid_bytes, "image/png")}
            )
            self.assertEqual(res_ok.status_code, 200)
            self.assertFalse(PIPELINE_SEMAPHORE.locked(), "Semaphore must be unlocked after success path")

            # 3. Verify lock release on Exception path
            corrupt_png = b"\x89PNG\r\n\x1a\nCORRUPT_PAYLOAD_TRIGGERING_EXCEPTION"
            res_err = client.post(
                "/api/v1/anonymize/preview",
                files={"file": ("corrupt.png", corrupt_png, "image/png")}
            )
            self.assertEqual(res_err.status_code, 500)
            self.assertFalse(PIPELINE_SEMAPHORE.locked(), "Semaphore must be unlocked after exception path")

        finally:
            if old_timeout is not None:
                os.environ["PIPELINE_SEMAPHORE_TIMEOUT"] = old_timeout
            else:
                os.environ.pop("PIPELINE_SEMAPHORE_TIMEOUT", None)
            if PIPELINE_SEMAPHORE.locked():
                PIPELINE_SEMAPHORE._value = 1

    def test_pristine_unredacted_brightness_preservation(self):
        """FIX 2: Ensure unredacted background pixels preserve original brightness within +/-3 gray levels."""
        import cv2
        import numpy as np
        from src.image_redactor import LocalImageRedactor

        # Create a synthetic prescription image with off-white background (gray ~205)
        syn_w, syn_h = 700, 900
        syn_img = np.full((syn_h, syn_w, 3), 205, dtype=np.uint8)

        cv2.putText(syn_img, "PRESCRIPTION / RX SLIP", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(syn_img, "Patient: John Doe, Age: 45 M", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(syn_img, "Rx: Amoxicillin 500 mg po tid", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(syn_img, "Doctor: Dr. Sarah Smith", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        syn_input = os.path.join(self.temp_dir, "synthetic_brightness_rx.jpg")
        syn_output = os.path.join(self.temp_dir, "synthetic_brightness_rx_out.jpg")
        cv2.imwrite(syn_input, syn_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        redactor = LocalImageRedactor()
        redactor.redact(syn_input, syn_output, force_prescription=True)

        self.assertTrue(os.path.exists(syn_output))
        out_img = cv2.imread(syn_output)
        in_img = cv2.imread(syn_input)

        gray_out = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
        gray_in = cv2.resize(cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY), (gray_out.shape[1], gray_out.shape[0]))

        unredacted_mask = gray_out > 10
        self.assertGreater(np.sum(unredacted_mask), 1000)

        in_median = float(np.median(gray_in[unredacted_mask]))
        out_median = float(np.median(gray_out[unredacted_mask]))
        in_mean = float(np.mean(gray_in[unredacted_mask]))
        out_mean = float(np.mean(gray_out[unredacted_mask]))

        median_delta = abs(out_median - in_median)
        mean_delta = abs(out_mean - in_mean)

        self.assertLessEqual(
            median_delta, 3.0,
            f"Unredacted median brightness shifted by {median_delta:.2f} (in: {in_median:.2f}, out: {out_median:.2f}) > 3.0 tolerance"
        )
        self.assertLessEqual(
            mean_delta, 3.0,
            f"Unredacted mean brightness shifted by {mean_delta:.2f} (in: {in_mean:.2f}, out: {out_mean:.2f}) > 3.0 tolerance"
        )


if __name__ == "__main__":
    unittest.main()
