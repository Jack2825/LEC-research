#!/usr/bin/env python3
"""Export one manuscript-supplied panel without redrawing its data or images.

The source PDFs remain unchanged. Crops split each page only at the white
gutter between its two labeled panels; embedded images and vector objects
are retained in the panel PDF. PNG and JPG are rendered from that PDF.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DPI = 450
PANELS = {
    "2E": ("2E-F.pdf", "aging", "Fig02E_cell_painting", (0, 0, 625, 351)),
    "2F": ("2E-F.pdf", "aging", "Fig02F_tube_formation", (625, 0, 881, 351)),
    "5I": ("5I-J.pdf", "disease", "Fig05I_lymphatic_vascular_remodeling", (0, 0, 878, 246)),
    "5J": ("5I-J.pdf", "disease", "Fig05J_cardiac_function", (878, 0, 1295, 246)),
}


def export_panel(key: str) -> None:
    filename, branch, stem, coordinates = PANELS[key]
    source = ROOT / filename
    output = ROOT / "new_result2" / branch / stem
    output.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(source) as original:
        if len(original) != 1:
            raise ValueError(f"Expected one source page: {source}")
        clip = pymupdf.Rect(coordinates)
        if not original[0].rect.contains(clip):
            raise ValueError(f"Panel crop exceeds the source page: {key}")
        # No source text span may be cut across a panel boundary.
        for block in original[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    box = pymupdf.Rect(span["bbox"])
                    if box.intersects(clip) and not clip.contains(box):
                        raise ValueError(f"Crop crosses text in {key}: {span['text']}")
        with pymupdf.open() as result:
            page = result.new_page(width=clip.width, height=clip.height)
            page.show_pdf_page(page.rect, original, 0, clip=clip)
            result.save(output / f"{stem}.pdf", garbage=4, deflate=True)
            pixmap = page.get_pixmap(dpi=DPI, alpha=False)
            pixmap.save(output / f"{stem}.png")
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            image.save(output / f"{stem}.jpg", quality=95, subsampling=0, dpi=(DPI, DPI))
            image.close()
    manifest = {
        "panel": key,
        "input_file": filename,
        "input_type": "Completed experimental panel PDF",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_page": 1,
        "crop_points_x0_y0_x1_y1": ";".join(map(str, coordinates)),
        "figure_use": "Preserve supplied microscopy, graphs, labels and statistics",
        "data_handling": "No numerical values reconstructed; source PDF is the original artwork",
        "script": f"new_script2/{branch}/{stem}.py",
    }
    with (output / f"Fig{key.zfill(3)}_input_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest))
        writer.writeheader()
        writer.writerow(manifest)
    print(f"Exported {key}: {output}")
