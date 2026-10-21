"""
Raster & Document Type Detection Module.

Inspects incoming PDFs to determine if they contain native digital vector text
or are scanned raster images encapsulated inside a PDF container.
"""

from typing import Dict, Any, List
import os


def inspect_pdf_structure(pdf_path: str) -> Dict[str, Any]:
    """
    Inspects each page of a PDF:
    - Counts characters in native text layer.
    - Counts embedded raster images.
    - Determines whether the document is digital vector or scanned raster.
    """
    import pymupdf as fitz

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found at {pdf_path}")

    ext = os.path.splitext(pdf_path)[1].lower()
    image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
    if ext in image_extensions:
        return {
            "file_path": pdf_path,
            "page_count": 1,
            "total_char_count": 0,
            "total_image_count": 1,
            "format_type": "SCANNED_RASTER",
            "pages": [{
                "page_number": 1,
                "char_count": 0,
                "image_count": 1,
                "sample_text": None,
                "is_raster": True
            }]
        }

    with fitz.open(pdf_path) as doc:
        page_reports: List[Dict[str, Any]] = []
        total_chars = 0
        total_images = 0

        for i, page in enumerate(doc):
            text = page.get_text().strip()
            char_count = len(text)
            images = page.get_images()
            image_count = len(images)

            total_chars += char_count
            total_images += image_count

            # A page with fewer than 20 characters and 1+ full-page images is classified as raster
            is_page_raster = (char_count < 20 and image_count > 0)

            page_reports.append({
                "page_number": i + 1,
                "char_count": char_count,
                "image_count": image_count,
                "sample_text": text[:150] if text else None,
                "is_raster": is_page_raster
            })

        is_overall_raster = all(p["is_raster"] for p in page_reports) or (total_chars < 50 and total_images > 0)
        page_count = len(doc)

    return {
        "file_path": pdf_path,
        "page_count": page_count,
        "total_char_count": total_chars,
        "total_image_count": total_images,
        "format_type": "SCANNED_RASTER" if is_overall_raster else "VECTOR_DIGITAL",
        "pages": page_reports
    }
