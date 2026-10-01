"""Command-line interface for the Pharmacon AI pipeline.

Usage:
    python -m pharmacon_ai.cli crop <photo.jpg> <out_dir> <page_index> [pdf_path]

Examples:
    python -m pharmacon_ai.cli crop photo.jpg ./crops 2
    python -m pharmacon_ai.cli crop photo.jpg ./crops 2 Protocol.pdf

page_index is the PDF page the photo shows (0-based): 2=Sheet1(E01-E04),
3=Sheet2, 4=Sheet3, 5=Sheet4, 6=Validation+Refinement, 7=Test1, 8=Test2.
"""

import os
import sys

import cv2

from .crop import process_photo

DEFAULT_PDF = "Pharmacon_Writer_Calibration_Protocol.pdf"


def _crop(photo, out_dir, page_index, pdf_path):
    lines = process_photo(photo, pdf_path, page_index)
    os.makedirs(out_dir, exist_ok=True)
    for lid, item in lines.items():
        out = os.path.join(out_dir, f"{lid}.png")
        cv2.imwrite(out, item["image"])
    print(f"Cropped {len(lines)} lines to {out_dir}")
    for lid in sorted(lines):
        print(f"  {lid}: {lines[lid]['label']}")


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 1
    cmd, photo, out_dir, page_index = argv[0], argv[1], argv[2], int(argv[3])
    pdf_path = argv[4] if len(argv) > 4 else DEFAULT_PDF
    if cmd == "crop":
        _crop(photo, out_dir, page_index, pdf_path)
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
