#!/usr/bin/env python3
"""Assemble current separate panels into main Figures 1-5.

PDF objects are placed directly to retain vector text and the source image
resolution. PNG and JPG are rendered from the assembled PDF. Only outer white
margins are trimmed. Figures 2 and 5 include the supplied experimental panels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops
import pymupdf


ROOT = Path(__file__).resolve().parents[1]
CANVAS_WIDTH = 6000
OUTER_MARGIN = 90
GUTTER = 90
WHITE_THRESHOLD = 8
TRIM_PADDING = 28
OUTPUT_DPI = 450

# These are trusted local figure outputs and several exceed Pillow's default
# decompression-bomb warning threshold because they were exported at 450-600 dpi.
Image.MAX_IMAGE_PIXELS = None


def panel(
    label: str,
    result_dir: str,
    stem: str,
    source_csv: str,
    input_manifest: str,
) -> dict[str, str]:
    directory = ROOT / result_dir
    return {
        "panel": label,
        "png": str(directory / f"{stem}.png"),
        "pdf": str(directory / f"{stem}.pdf"),
        "source_csv": str(directory / source_csv),
        "input_manifest": str(directory / input_manifest),
    }


def experimental_panel(label: str, branch: str, stem: str) -> dict[str, str]:
    item = panel(label, f"new_result2/{branch}/{stem}", stem, "", f"{stem[:6]}_input_manifest.csv")
    item["source_csv"] = ""
    item["original_artwork"] = str(ROOT / ("2E-F.pdf" if stem.startswith("Fig02") else "5I-J.pdf"))
    return item


FIGURES: list[dict[str, Any]] = [
    {
        "figure": "Figure 1",
        "output_dir": ROOT / "new_result2" / "Fig01_combined",
        "stem": "Fig01_combined",
        # The workflow is an overview panel and should not dominate B-C.
        "row_width_fractions": {1: 0.58},
        "rows": [
            [
                panel(
                    "A",
                    "new_result2/Fig01A_research_workflow",
                    "Fig01A_research_workflow",
                    "Fig01A_source_data.csv",
                    "Fig01A_input_manifest.csv",
                )
            ],
            [
                panel(
                    "B",
                    "new_result2/Fig01B_donor_age_and_LEC_counts",
                    "Fig01B_donor_age_and_LEC_counts",
                    "Fig01B_source_data.csv",
                    "Fig01B_input_csv_manifest.csv",
                ),
                panel(
                    "C",
                    "new_result2/Fig01C_two_step_LEC_selection_UMAP",
                    "Fig01C_two_step_LEC_selection_UMAP",
                    "Fig01C_source_data.csv.gz",
                    "Fig01C_input_manifest.csv",
                ),
            ],
        ],
    },
    {
        "figure": "Figure 2",
        "output_dir": ROOT / "new_result2" / "aging" / "Fig02_combined",
        "stem": "Fig02_combined",
        # Match A and B by height so they read as one side-by-side row.
        "equal_height_rows": {1, 4},
        "rows": [
            [
                panel(
                    "A",
                    "new_result2/aging/Fig02A_aging_LEC_state_UMAP",
                    "Fig02A_aging_LEC_state_UMAP",
                    "Fig02A_source_data.csv",
                    "Fig02A_input_manifest.csv",
                ),
                panel(
                    "B",
                    "new_result2/aging/Fig02B_aging_LEC_state_composition",
                    "Fig02B_aging_LEC_state_composition",
                    "Fig02B_source_data.csv",
                    "Fig02B_input_manifest.csv",
                ),
            ],
            [
                panel(
                    "C",
                    "new_result2/aging/Fig02C_aging_marker_GSEA_map",
                    "Fig02C_aging_marker_GSEA_map",
                    "Fig02C_source_data.csv",
                    "Fig02C_input_manifest.csv",
                )
            ],
            [
                panel(
                    "D",
                    "new_result2/aging/Fig02D_aging_representative_state_abundance",
                    "Fig02D_aging_representative_state_abundance",
                    "Fig02D_source_data.csv",
                    "Fig02D_input_manifest.csv",
                )
            ],
            [
                experimental_panel("E", "aging", "Fig02E_cell_painting"),
                experimental_panel("F", "aging", "Fig02F_tube_formation"),
            ],
        ],
    },
    {
        "figure": "Figure 3",
        "output_dir": ROOT / "new_result2" / "aging" / "Fig03_combined",
        "stem": "Fig03_combined",
        "rows": [
            [
                panel(
                    "A",
                    "new_result2/aging/Fig03A_aging_state_specific_regulons",
                    "Fig03A_aging_state_specific_regulons",
                    "Fig03A_source_data.csv",
                    "Fig03A_input_manifest.csv",
                ),
                panel(
                    "B",
                    "new_result2/aging/Fig03B_aging_regulon_robustness",
                    "Fig03B_aging_regulon_robustness",
                    "Fig03B_source_data.csv",
                    "Fig03B_input_manifest.csv",
                ),
            ],
            [
                panel(
                    "C",
                    "new_result2/aging/Fig03C_aging_state_abundance",
                    "Fig03C_aging_state_abundance",
                    "Fig03C_source_data.csv",
                    "Fig03C_input_manifest.csv",
                )
            ],
            [
                panel(
                    "D",
                    "new_result2/aging/Fig03D_aging_ligand_activity",
                    "Fig03D_aging_ligand_activity",
                    "Fig03D_source_data.csv",
                    "Fig03D_input_manifest.csv",
                ),
                panel(
                    "E",
                    "new_result2/aging/Fig03E_aging_sender_ligand_network",
                    "Fig03E_aging_sender_ligand_network",
                    "Fig03E_source_data.csv",
                    "Fig03E_input_manifest.csv",
                ),
            ],
            [
                panel(
                    "F",
                    "new_result2/aging/Fig03F_aging_ligand_receptor_interactions",
                    "Fig03F_aging_ligand_receptor_interactions",
                    "Fig03F_source_data.csv",
                    "Fig03F_input_manifest.csv",
                )
            ],
        ],
    },
    {
        "figure": "Figure 4",
        "output_dir": ROOT / "new_result2" / "disease" / "Fig04_combined",
        "stem": "Fig04_combined",
        # Match A and B by height so they read as one side-by-side row.
        "equal_height_rows": {1},
        "rows": [
            [
                panel(
                    "A",
                    "new_result2/disease/Fig04A_disease_LEC_state_UMAP",
                    "Fig04A_disease_LEC_state_UMAP",
                    "Fig04A_source_data.csv",
                    "Fig04A_input_manifest.csv",
                ),
                panel(
                    "B",
                    "new_result2/disease/Fig04B_disease_LEC_state_composition",
                    "Fig04B_disease_LEC_state_composition",
                    "Fig04B_source_data.csv",
                    "Fig04B_input_manifest.csv",
                ),
            ],
            [
                panel(
                    "C",
                    "new_result2/disease/Fig04C_disease_marker_GSEA_map",
                    "Fig04C_disease_marker_GSEA_map",
                    "Fig04C_source_data.csv",
                    "Fig04C_input_manifest.csv",
                )
            ],
            [
                panel(
                    "D",
                    "new_result2/disease/Fig04D_disease_representative_state_effects",
                    "Fig04D_disease_representative_state_effects",
                    "Fig04D_source_data.csv",
                    "Fig04D_input_manifest.csv",
                )
            ],
        ],
    },
    {
        "figure": "Figure 5",
        "output_dir": ROOT / "new_result2" / "disease" / "Fig05_combined",
        "stem": "Fig05_combined",
        "equal_height_rows": {6},
        "rows": [
            [
                panel(
                    "A",
                    "new_result2/disease/Fig05A_disease_state_specific_regulons",
                    "Fig05A_disease_state_specific_regulons",
                    "Fig05A_source_data.csv",
                    "Fig05A_input_manifest.csv",
                ),
                panel(
                    "B",
                    "new_result2/disease/Fig05B_disease_regulon_robustness",
                    "Fig05B_disease_regulon_robustness",
                    "Fig05B_source_data.csv",
                    "Fig05B_input_manifest.csv",
                ),
            ],
            [
                panel(
                    "C",
                    "new_result2/disease/Fig05C_disease_state_abundance",
                    "Fig05C_disease_state_abundance",
                    "Fig05C_source_data.csv",
                    "Fig05C_input_manifest.csv",
                )
            ],
            [
                panel(
                    "D",
                    "new_result2/disease/Fig05D_disease_communication_modules",
                    "Fig05D_disease_communication_modules",
                    "Fig05D_source_data.csv",
                    "Fig05D_input_manifest.csv",
                )
            ],
            [
                panel(
                    "E",
                    "new_result2/disease/Fig05E_DCM_sender_ligand_network",
                    "Fig05E_DCM_sender_ligand_network",
                    "Fig05E_source_data.csv",
                    "Fig05E_input_manifest.csv",
                ),
                panel(
                    "F",
                    "new_result2/disease/Fig05F_HCM_sender_ligand_network",
                    "Fig05F_HCM_sender_ligand_network",
                    "Fig05F_source_data.csv",
                    "Fig05F_input_manifest.csv",
                ),
            ],
            [
                panel(
                    "G",
                    "new_result2/disease/Fig05G_DoxCM_sender_ligand_network",
                    "Fig05G_DoxCM_sender_ligand_network",
                    "Fig05G_source_data.csv",
                    "Fig05G_input_manifest.csv",
                ),
                panel(
                    "H",
                    "new_result2/disease/Fig05H_HFpEF_sender_ligand_network",
                    "Fig05H_HFpEF_sender_ligand_network",
                    "Fig05H_source_data.csv",
                    "Fig05H_input_manifest.csv",
                ),
            ],
            [
                experimental_panel("I", "disease", "Fig05I_lymphatic_vascular_remodeling"),
                experimental_panel("J", "disease", "Fig05J_cardiac_function"),
            ],
        ],
    },
]

# Current manuscript excludes the former Figure 2 experimental panels E/F.
for _figure in FIGURES:
    if _figure["stem"] == "Fig02_combined":
        _figure["rows"] = [[p for p in row if p["panel"] not in {"E", "F"}] for row in _figure["rows"]]
        _figure["rows"] = [row for row in _figure["rows"] if row]



def relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def validate_panel_files(figure_spec: dict[str, Any]) -> None:
    for row in figure_spec["rows"]:
        for item in row:
            for field in ["png", "pdf", "source_csv", "input_manifest"]:
                if not item.get(field):
                    continue
                path = Path(item[field])
                if not path.exists():
                    raise FileNotFoundError(path)


def trim_white(image: Image.Image) -> Image.Image:
    background = Image.new("RGB", image.size, "white")
    difference = ImageChops.difference(image, background).convert("L")
    mask = difference.point(lambda value: 255 if value > WHITE_THRESHOLD else 0)
    box = mask.getbbox()
    if box is None:
        return image.copy()
    left = max(0, box[0] - TRIM_PADDING)
    top = max(0, box[1] - TRIM_PADDING)
    right = min(image.width, box[2] + TRIM_PADDING)
    bottom = min(image.height, box[3] + TRIM_PADDING)
    return image.crop((left, top, right, bottom))


def prepare_panel(path: str, slot_width: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    # Downsample before detecting the white boundary to keep memory bounded.
    if image.width > slot_width * 1.25:
        preliminary_height = max(1, round(image.height * slot_width * 1.25 / image.width))
        image = image.resize(
            (round(slot_width * 1.25), preliminary_height),
            Image.Resampling.LANCZOS,
        )
    image = trim_white(image)
    target_height = max(1, round(image.height * slot_width / image.width))
    if image.width != slot_width:
        image = image.resize((slot_width, target_height), Image.Resampling.LANCZOS)
    return image


def write_manifest(
    figure_spec: dict[str, Any],
    placements: list[dict[str, object]],
) -> None:
    output_dir: Path = figure_spec["output_dir"]
    manifest_path = output_dir / f"{figure_spec['stem']}_panel_manifest.csv"
    note = str(figure_spec.get("note", ""))
    rows: list[dict[str, object]] = []
    for placement in placements:
        item = placement["item"]
        rows.append(
            {
                "combined_figure": figure_spec["figure"],
                "panel": item["panel"],
                "layout_row": placement["row"],
                "layout_column": placement["column"],
                "source_png": relative(item["png"]),
                "source_pdf": relative(item["pdf"]),
                "source_data_csv": relative(item["source_csv"]) if item["source_csv"] else "",
                "source_input_manifest": relative(item["input_manifest"]),
                "original_artwork": relative(item["original_artwork"]) if item.get("original_artwork") else "",
                "script": "new_script2/make_combined_figures.py",
                "placement_points": ";".join(f"{value:.3f}" for value in placement["rect"]),
                "source_clip_points": ";".join(f"{value:.3f}" for value in placement["clip"]),
                "assembly_note": note,
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pdf_content_clip(document: pymupdf.Document, keep_page: bool = False) -> pymupdf.Rect:
    page = document[0]
    if keep_page:
        return page.rect
    pixmap = page.get_pixmap(dpi=100, alpha=False)
    preview = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    background = Image.new("RGB", preview.size, "white")
    mask = ImageChops.difference(preview, background).convert("L").point(
        lambda value: 255 if value > WHITE_THRESHOLD else 0
    )
    bounds = mask.getbbox()
    preview.close()
    background.close()
    mask.close()
    if bounds is None:
        raise ValueError("Empty source panel")
    sx, sy = page.rect.width / pixmap.width, page.rect.height / pixmap.height
    clip = pymupdf.Rect(bounds[0] * sx, bounds[1] * sy, bounds[2] * sx, bounds[3] * sy)
    # Include complete text bounding boxes as well as visible pixels.
    for block in page.get_text("blocks"):
        if block[6] == 0:
            clip |= pymupdf.Rect(block[:4]) & page.rect
    return (clip + (-4, -4, 4, 4)) & page.rect


def assemble(figure_spec: dict[str, Any]) -> None:
    validate_panel_files(figure_spec)
    unit = 72 / OUTPUT_DPI
    page_width = CANVAS_WIDTH * unit
    margin, gutter = OUTER_MARGIN * unit, GUTTER * unit
    content_width = page_width - 2 * margin
    prepared_rows = []
    row_heights = []
    row_widths = []
    equal_height_rows = set(figure_spec.get("equal_height_rows", set()))
    row_width_fractions = figure_spec.get("row_width_fractions", {})

    for row_index, row in enumerate(figure_spec["rows"], start=1):
        columns = len(row)
        width_fraction = float(row_width_fractions.get(row_index, 1.0))
        row_content_width = content_width * width_fraction
        prepared = []
        for item in row:
            document = pymupdf.open(item["pdf"])
            if len(document) != 1:
                raise ValueError(f"Expected one-page panel: {item['pdf']}")
            clip = pdf_content_clip(document, keep_page=bool(item.get("original_artwork")))
            prepared.append({"item": item, "document": document, "clip": clip})
        ratios = [entry["clip"].width / entry["clip"].height for entry in prepared]
        available_width = row_content_width - gutter * (columns - 1)
        if row_index in equal_height_rows and columns > 1:
            height = available_width / sum(ratios)
            widths = [height * ratio for ratio in ratios]
        else:
            widths = [available_width / columns] * columns
        for entry, width, ratio in zip(prepared, widths, ratios, strict=True):
            entry.update(width=width, height=width / ratio)
        prepared_rows.append(prepared)
        row_heights.append(max(entry["height"] for entry in prepared))
        row_widths.append(sum(widths) + gutter * (columns - 1))

    canvas_height = 2 * margin + sum(row_heights) + gutter * (len(prepared_rows) - 1)
    combined = pymupdf.open()
    page = combined.new_page(width=page_width, height=canvas_height)
    placements: list[dict[str, object]] = []
    y = margin
    for row_index, (prepared, row_height, row_width) in enumerate(
        zip(prepared_rows, row_heights, row_widths, strict=True), start=1
    ):
        x = margin + (content_width - row_width) / 2
        for column_index, entry in enumerate(prepared, start=1):
            item = entry["item"]
            rect = pymupdf.Rect(x, y, x + entry["width"], y + entry["height"])
            page.show_pdf_page(rect, entry["document"], 0, clip=entry["clip"])
            placements.append(
                {
                    "item": item,
                    "row": row_index,
                    "column": column_index,
                    "rect": rect,
                    "clip": entry["clip"],
                }
            )
            x += entry["width"] + gutter
            entry["document"].close()
        y += row_height + gutter

    output_dir: Path = figure_spec["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(figure_spec["stem"])
    combined.save(output_dir / f"{stem}.pdf", garbage=4, deflate=True)
    pixmap = page.get_pixmap(dpi=OUTPUT_DPI, alpha=False)
    pixmap.save(output_dir / f"{stem}.png")
    canvas = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    canvas.save(
        output_dir / f"{stem}.jpg",
        quality=95,
        subsampling=0,
        dpi=(OUTPUT_DPI, OUTPUT_DPI),
    )
    canvas.close()
    combined.close()
    write_manifest(figure_spec, placements)
    print(f"{figure_spec['figure']} combined output written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, choices=range(1, 6), default=list(range(1, 6)))
    args = parser.parse_args()
    for figure_spec in FIGURES:
        if int(figure_spec["figure"].split()[-1]) in args.figures:
            assemble(figure_spec)


if __name__ == "__main__":
    main()
