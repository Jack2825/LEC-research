#!/usr/bin/env python3
"""Create aging Figure 3E: sender-to-ligand-to-LEC network."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from Fig03D_aging_ligand_activity import (
    CELLTYPE_MAP_CSV,
    DONOR_CSV,
    INTERACTION_CSV,
    ROOT,
    TEXT_COLOR,
    build_source_data,
    draw_panel_b,
    load_and_validate,
    select_panel_a,
    select_panels_b_c,
)


OUTPUT_DIR = ROOT / "new_result2" / "aging" / "Fig03E_aging_sender_ligand_network"
PNG = OUTPUT_DIR / "Fig03E_aging_sender_ligand_network.png"
PDF = OUTPUT_DIR / "Fig03E_aging_sender_ligand_network.pdf"
JPG = OUTPUT_DIR / "Fig03E_aging_sender_ligand_network.jpg"
SOURCE_CSV = OUTPUT_DIR / "Fig03E_source_data.csv"
MANIFEST_CSV = OUTPUT_DIR / "Fig03E_input_manifest.csv"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    activity, interactions, donors = load_and_validate()
    panel_a = select_panel_a(activity)
    panel_b, panel_c, ligand_order = select_panels_b_c(interactions)
    build_source_data(panel_a, panel_b, panel_c).query("panel == 'E'").to_csv(
        SOURCE_CSV, index=False
    )
    pd.DataFrame(
        [
            {
                "figure_panel": "E",
                "input_csv": INTERACTION_CSV.relative_to(ROOT).as_posix(),
                "use": "sender-ligand network from finite top-50 interactions",
                "filter_or_selection": "eight highest distinct ligands and all six selected sender types",
            },
            {
                "figure_panel": "E validation",
                "input_csv": CELLTYPE_MAP_CSV.relative_to(ROOT).as_posix(),
                "use": "validate sender identifiers",
                "filter_or_selection": "six selected sender types",
            },
            {
                "figure_panel": "E donor provenance",
                "input_csv": DONOR_CSV.relative_to(ROOT).as_posix(),
                "use": "verify Q1/Q4 donor population",
                "filter_or_selection": ">=25 final LECs per donor",
            },
        ]
    ).to_csv(MANIFEST_CSV, index=False)

    figure, axis = plt.subplots(figsize=(11.4, 7.3), facecolor="white")
    figure.subplots_adjust(left=0.035, right=0.98, bottom=0.08, top=0.79)
    draw_panel_b(axis, panel_b, ligand_order)
    figure.text(0.025, 0.96, "E", fontsize=18, fontweight="bold", color=TEXT_COLOR, va="top")
    figure.text(
        0.08,
        0.96,
        "Sender cell → ligand → LEC network",
        fontsize=17,
        fontweight="bold",
        color=TEXT_COLOR,
        va="top",
    )
    figure.text(
        0.08,
        0.90,
        "Six sender cell types and eight prioritized ligands; line width reflects within-run priority.",
        fontsize=9,
        color="#5A6268",
        va="top",
    )
    figure.savefig(PNG, dpi=450, facecolor="white")
    figure.savefig(PDF, facecolor="white")
    figure.savefig(JPG, dpi=450, facecolor="white", pil_kwargs={"quality": 95, "subsampling": 0})
    plt.close(figure)
    print(f"Figure 3E written to: {OUTPUT_DIR}")
    print(f"Sender types: {panel_b['sender'].nunique()}; ligands: {panel_b['ligand'].nunique()}")


if __name__ == "__main__":
    main()
