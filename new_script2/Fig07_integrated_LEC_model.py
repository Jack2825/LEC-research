#!/usr/bin/env python3
"""Create the Graphic Abstract: integrated model of cardiac LEC remodeling.

This is a manuscript-level synthesis schematic. It preserves the supplied
aging, cardiovascular-disease, functional, and spatial-validation narrative
while using the visual language of the other main figures. Computational
evidence files are recorded in the input manifest; experimental conclusions
are explicitly identified as non-computational manuscript synthesis.

Outputs
-------
new_result2/Graphic_Abstract_integrated_LEC_model/
    Graphic_Abstract_integrated_LEC_model.png
    Graphic_Abstract_integrated_LEC_model.pdf
    Graphic_Abstract_integrated_LEC_model.jpg
    Fig07_source_data.csv
    Fig07_input_manifest.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "new_result2" / "Graphic_Abstract_integrated_LEC_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_STEM = OUTPUT_DIR / "Graphic_Abstract_integrated_LEC_model"
SOURCE_CSV = OUTPUT_DIR / "Fig07_source_data.csv"
MANIFEST_CSV = OUTPUT_DIR / "Fig07_input_manifest.csv"
SCRIPT_VERSION = "Fig07_integrated_LEC_model_v3_compact_2026_09_02"

TEXT = "#20252A"
SUBTEXT = "#59636C"
AGING = "#3973A9"
AGING_FILL = "#F4F8FB"
DISEASE = "#C5544D"
DISEASE_FILL = "#FCF6F4"
SHARED_EDGE = "#71808B"
SHARED_FILL = "#F3F6F7"
DIVIDER = "#D8DEE2"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


CONTENT = [
    {
        "content_order": 1,
        "branch": "Aging",
        "stage": "Microenvironmental cues",
        "title": "Aging-associated extracellular signals",
        "body": (
            "Altered inflammatory, lymphangiogenic and matrix-related inputs\n"
            "Examples: VEGFC–NRP2 • ANGPT2–TEK • VCAN–CD44"
        ),
        "badge": "",
        "evidence_type": "computational synthesis",
    },
    {
        "content_order": 2,
        "branch": "Disease",
        "stage": "Microenvironmental cues",
        "title": "Disease-associated microenvironmental\ncues",
        "body": (
            "Disease-specific inflammatory, stromal and growth-factor signaling\n"
            "Prioritized pathways include BMP/TGF-β/activin, PDGF,\n"
            "VEGF, ANGPT, Notch and WNT"
        ),
        "badge": "",
        "evidence_type": "computational synthesis",
    },
    {
        "content_order": 3,
        "branch": "Aging",
        "stage": "Transcriptional regulation",
        "title": "State-associated transcriptional regulators",
        "body": (
            "Distinct regulon programs across aging-sensitive LEC states\n"
            "SCENIC: EPAS1 / TP63 / MEIS2 / TEAD1;\n"
            "KLF2 / NFATC4 / FOXC1"
        ),
        "badge": "",
        "evidence_type": "computational synthesis",
    },
    {
        "content_order": 4,
        "branch": "Disease",
        "stage": "Transcriptional regulation",
        "title": "Disease-associated transcriptional programs",
        "body": (
            "Regulon programs across DCM, HCM, DoxCM and HFpEF\n"
            "SCENIC supports disease-specific transcriptional regulation"
        ),
        "badge": "",
        "evidence_type": "computational synthesis",
    },
    {
        "content_order": 5,
        "branch": "Aging",
        "stage": "LEC state remodeling",
        "title": "Aging-associated LEC state redistribution",
        "body": (
            "State 0 ↑\n"
            "State 4 cytokine-responsive / immune ↓\n"
            "State 5 matrix-interactive / structural ↓"
        ),
        "badge": "",
        "evidence_type": "computational synthesis",
    },
    {
        "content_order": 6,
        "branch": "Disease",
        "stage": "LEC state remodeling",
        "title": "Shared and disease-specific LEC state\nremodeling",
        "body": (
            "Cardiovascular diseases do not produce one uniform\n"
            "pathological LEC state\n"
            "Shared remodeling + disease-specific state changes"
        ),
        "badge": "",
        "evidence_type": "computational synthesis",
    },
    {
        "content_order": 7,
        "branch": "Aging",
        "stage": "Functional consequences",
        "title": "Reduced adaptive inflammatory\nresponsiveness",
        "body": (
            "Impaired response to systemic inflammatory challenge\n"
            "Influenza–plasma assay: young LECs adapt; aged LECs\n"
            "show reduced tube formation and organelle remodeling"
        ),
        "badge": "IN VITRO",
        "evidence_type": "experimental manuscript synthesis",
    },
    {
        "content_order": 8,
        "branch": "Disease",
        "stage": "Functional consequences",
        "title": "Lymphatic and cardiac remodeling",
        "body": (
            "Functional support: endothelial Acvr2a deletion in CMP models\n"
            "Improved cardiac function • increased open lymphatic\n"
            "vessels • no change in blood vessels"
        ),
        "badge": "IN VIVO",
        "evidence_type": "experimental manuscript synthesis",
    },
    {
        "content_order": 9,
        "branch": "Shared",
        "stage": "Spatial validation",
        "title": "",
        "body": (
            "Independent spatial datasets provide tissue-level support for selected "
            "aging- and disease-associated LEC programs"
        ),
        "badge": "",
        "evidence_type": "spatial-validation synthesis",
    },
]


EVIDENCE_INPUTS = [
    {
        "evidence_scope": "aging extracellular signaling",
        "input_file": "new_result2/aging/Fig03D_aging_ligand_activity/Fig03D_source_data.csv",
        "use": "aging-prioritized ligand activity",
    },
    {
        "evidence_scope": "aging ligand-receptor signaling",
        "input_file": "new_result2/aging/Fig03F_aging_ligand_receptor_interactions/Fig03F_source_data.csv",
        "use": "aging focused ligand-receptor axes",
    },
    {
        "evidence_scope": "aging transcriptional regulation",
        "input_file": "new_result2/aging/Fig03A_aging_state_specific_regulons/Fig03A_source_data.csv",
        "use": "state-specific aging regulon programs",
    },
    {
        "evidence_scope": "aging LEC state remodeling",
        "input_file": "new_result2/aging/Fig02D_aging_representative_state_abundance/Fig02D_source_data.csv",
        "use": "donor-level aging state abundance effects",
    },
    {
        "evidence_scope": "disease extracellular signaling",
        "input_file": "new_result2/disease/Fig05D_disease_communication_modules/Fig05D_source_data.csv",
        "use": "disease-specific communication modules",
    },
    {
        "evidence_scope": "disease transcriptional regulation",
        "input_file": "new_result2/disease/Fig05A_disease_state_specific_regulons/Fig05A_source_data.csv",
        "use": "state-specific disease regulon programs",
    },
    {
        "evidence_scope": "disease LEC state remodeling",
        "input_file": "new_result2/disease/Fig04D_disease_representative_state_effects/Fig04D_source_data.csv",
        "use": "shared and disease-specific state abundance effects",
    },
    {
        "evidence_scope": "aging spatial validation",
        "input_file": "new_result2/aging/FigS07_spatial_aging_validation/FigS07_figure_manifest.csv",
        "use": "aging spatial-validation evidence provenance",
    },
    {
        "evidence_scope": "disease spatial validation",
        "input_file": "new_result2/disease/Fig06_spatial_disease_BMP_ACVR2A/Fig06_figure_manifest.csv",
        "use": "disease spatial-validation evidence provenance",
    },
    {
        "evidence_scope": "in vitro functional conclusion",
        "input_file": "not_applicable",
        "use": "non-computational experimental manuscript synthesis",
    },
    {
        "evidence_scope": "in vivo functional conclusion",
        "input_file": "not_applicable",
        "use": "non-computational experimental manuscript synthesis",
    },
]


def validate_inputs() -> None:
    missing = [
        item["input_file"]
        for item in EVIDENCE_INPUTS
        if item["input_file"] != "not_applicable"
        and not (ROOT / item["input_file"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Figure 7 evidence inputs: {missing}")


def add_text(
    axis: plt.Axes,
    artists: list[plt.Text],
    x: float,
    y: float,
    value: str,
    **kwargs: object,
) -> plt.Text:
    artist = axis.text(x, y, value, transform=axis.transAxes, **kwargs)
    artists.append(artist)
    return artist


def add_badge(
    axis: plt.Axes,
    artists: list[plt.Text],
    x: float,
    y: float,
    width: float,
    text: str,
    color: str,
) -> None:
    height = 0.030
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            transform=axis.transAxes,
            boxstyle="round,pad=0.003,rounding_size=0.004",
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=5,
        )
    )
    add_text(
        axis,
        artists,
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=color,
        zorder=6,
    )


def add_card(
    axis: plt.Axes,
    artists: list[plt.Text],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
    fill: str,
    badge: str = "",
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            transform=axis.transAxes,
            boxstyle="round,pad=0.005,rounding_size=0.010",
            facecolor=fill,
            edgecolor=color,
            linewidth=0.95,
            zorder=2,
        )
    )
    axis.plot(
        [x + 0.009, x + 0.009],
        [y + 0.017, y + height - 0.017],
        transform=axis.transAxes,
        color=color,
        linewidth=3.0,
        solid_capstyle="round",
        zorder=3,
    )
    title_lines = title.count("\n") + 1
    title_y = y + height - 0.017
    body_y = y + height - (0.055 if title_lines == 1 else 0.074)
    title_artist = add_text(
        axis,
        artists,
        x + 0.023,
        title_y,
        title,
        ha="left",
        va="top",
        fontsize=12.7,
        fontweight="bold",
        color=color,
        linespacing=1.02,
        zorder=4,
    )
    body_artist = add_text(
        axis,
        artists,
        x + 0.023,
        body_y,
        body,
        ha="left",
        va="top",
        fontsize=10.7,
        color=TEXT,
        linespacing=1.12,
        zorder=4,
    )
    if badge:
        badge_width = 0.070
        badge_x = x + width - badge_width - 0.012
        badge_y = y + height - 0.041
        add_badge(axis, artists, badge_x, badge_y, badge_width, badge, color)

    # Retain references so the renderer-based audit sees both text blocks.
    _ = (title_artist, body_artist)


def add_connector(
    axis: plt.Axes,
    x: float,
    start_y: float,
    end_y: float,
    color: str,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            (x, start_y),
            (x, end_y),
            transform=axis.transAxes,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.05,
            color=color,
            alpha=0.85,
            zorder=1,
        )
    )


def validate_canvas(figure: plt.Figure, artists: list[plt.Text]) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    outside: list[str] = []
    for artist in artists:
        box = artist.get_window_extent(renderer=renderer)
        if (
            box.x0 < canvas.x0 - 1
            or box.y0 < canvas.y0 - 1
            or box.x1 > canvas.x1 + 1
            or box.y1 > canvas.y1 + 1
        ):
            outside.append(artist.get_text().replace("\n", " | "))
    if outside:
        raise ValueError(f"Figure 7 text extends beyond the canvas: {outside}")


def write_traceability() -> None:
    source = pd.DataFrame(CONTENT)
    source.insert(0, "script_version", SCRIPT_VERSION)
    source.to_csv(SOURCE_CSV, index=False)

    manifest = pd.DataFrame(EVIDENCE_INPUTS)
    manifest.insert(0, "script_version", SCRIPT_VERSION)
    manifest.insert(1, "figure_panel", "Figure 7")
    manifest.to_csv(MANIFEST_CSV, index=False)


def main() -> None:
    validate_inputs()
    write_traceability()

    # Keep the established type sizes while reducing excess vertical space.
    figure, axis = plt.subplots(figsize=(14.2, 9.0), facecolor="white")
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    artists: list[plt.Text] = []

    add_text(
        axis,
        artists,
        0.035,
        0.982,
        "GRAPHIC ABSTRACT",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=SUBTEXT,
    )
    add_text(
        axis,
        artists,
        0.035,
        0.950,
        "Integrated model of cardiac LEC remodeling in aging and cardiovascular disease",
        ha="left",
        va="top",
        fontsize=19.0,
        fontweight="bold",
        color=TEXT,
    )
    add_text(
        axis,
        artists,
        0.345,
        0.875,
        "PHYSIOLOGICAL AGING",
        ha="center",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color=AGING,
    )
    add_text(
        axis,
        artists,
        0.762,
        0.875,
        "CARDIOVASCULAR DISEASE",
        ha="center",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color=DISEASE,
    )

    axis.plot(
        [0.542, 0.542],
        [0.112, 0.845],
        transform=axis.transAxes,
        color=DIVIDER,
        linewidth=0.9,
        linestyle=(0, (3, 4)),
        zorder=0,
    )

    row_labels = [
        "MICROENVIRONMENTAL\nCUES",
        "TRANSCRIPTIONAL\nREGULATION",
        "LEC STATE\nREMODELING",
        "FUNCTIONAL\nCONSEQUENCES",
    ]
    axis.plot(
        [0.165, 0.525],
        [0.848, 0.848],
        transform=axis.transAxes,
        color=AGING,
        linewidth=2.0,
        solid_capstyle="round",
    )
    axis.plot(
        [0.560, 0.965],
        [0.848, 0.848],
        transform=axis.transAxes,
        color=DISEASE,
        linewidth=2.0,
        solid_capstyle="round",
    )

    row_bottoms = [0.680, 0.505, 0.330, 0.155]
    card_height = 0.150
    left_x, left_width = 0.165, 0.360
    right_x, right_width = 0.560, 0.405

    content_lookup = {
        (item["branch"], item["stage"]): item
        for item in CONTENT
        if item["branch"] != "Shared"
    }
    stage_names = [
        "Microenvironmental cues",
        "Transcriptional regulation",
        "LEC state remodeling",
        "Functional consequences",
    ]

    for index, (label, y, stage) in enumerate(
        zip(row_labels, row_bottoms, stage_names, strict=True)
    ):
        add_text(
            axis,
            artists,
            0.137,
            y + card_height / 2,
            label,
            ha="right",
            va="center",
            fontsize=9.8,
            fontweight="bold",
            color=SUBTEXT,
            linespacing=1.05,
        )
        aging_item = content_lookup[("Aging", stage)]
        disease_item = content_lookup[("Disease", stage)]
        add_card(
            axis,
            artists,
            x=left_x,
            y=y,
            width=left_width,
            height=card_height,
            title=str(aging_item["title"]),
            body=str(aging_item["body"]),
            badge=str(aging_item["badge"]),
            color=AGING,
            fill=AGING_FILL,
        )
        add_card(
            axis,
            artists,
            x=right_x,
            y=y,
            width=right_width,
            height=card_height,
            title=str(disease_item["title"]),
            body=str(disease_item["body"]),
            badge=str(disease_item["badge"]),
            color=DISEASE,
            fill=DISEASE_FILL,
        )
        if index < len(row_bottoms) - 1:
            next_top = row_bottoms[index + 1] + card_height
            add_connector(axis, left_x + left_width / 2, y, next_top, AGING)
            add_connector(axis, right_x + right_width / 2, y, next_top, DISEASE)

    spatial = next(item for item in CONTENT if item["branch"] == "Shared")
    spatial_x, spatial_y, spatial_width, spatial_height = 0.165, 0.045, 0.800, 0.070
    axis.add_patch(
        FancyBboxPatch(
            (spatial_x, spatial_y),
            spatial_width,
            spatial_height,
            transform=axis.transAxes,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            facecolor=SHARED_FILL,
            edgecolor=SHARED_EDGE,
            linewidth=0.9,
            zorder=2,
        )
    )
    add_text(
        axis,
        artists,
        spatial_x + spatial_width / 2,
        spatial_y + spatial_height / 2,
        str(spatial["body"]),
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="semibold",
        color=SUBTEXT,
        zorder=3,
    )

    validate_canvas(figure, artists)
    figure.savefig(FIGURE_STEM.with_suffix(".png"), dpi=450, facecolor="white")
    figure.savefig(FIGURE_STEM.with_suffix(".pdf"), facecolor="white")
    figure.savefig(
        FIGURE_STEM.with_suffix(".jpg"),
        dpi=450,
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)
    print(f"Graphic Abstract written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
