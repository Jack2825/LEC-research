#!/usr/bin/env python3
"""Create aging Figure 3F: focused ligand-receptor interactions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from Fig03D_aging_ligand_activity import (
    CELLTYPE_MAP_CSV,
    DONOR_CSV,
    INTERACTION_CSV,
    ROOT,
    TEXT_COLOR,
    build_source_data,
    draw_panel_c,
    load_and_validate,
    select_panel_a,
    select_panels_b_c,
)


OUTPUT_DIR = ROOT / "new_result2" / "aging" / "Fig03F_aging_ligand_receptor_interactions"
PNG = OUTPUT_DIR / "Fig03F_aging_ligand_receptor_interactions.png"
PDF = OUTPUT_DIR / "Fig03F_aging_ligand_receptor_interactions.pdf"
JPG = OUTPUT_DIR / "Fig03F_aging_ligand_receptor_interactions.jpg"
SOURCE_CSV = OUTPUT_DIR / "Fig03F_source_data.csv"
MANIFEST_CSV = OUTPUT_DIR / "Fig03F_input_manifest.csv"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    activity, interactions, donors = load_and_validate()
    panel_a = select_panel_a(activity)
    panel_b, panel_c, ligand_order = select_panels_b_c(interactions)
    build_source_data(panel_a, panel_b, panel_c).query("panel == 'F'").to_csv(
        SOURCE_CSV, index=False
    )
    pd.DataFrame(
        [
            {
                "figure_panel": "F",
                "input_csv": INTERACTION_CSV.relative_to(ROOT).as_posix(),
                "use": "receptor-resolved prioritized interactions",
                "filter_or_selection": "receptor-resolved edges for the eight Figure 3E ligands",
            },
            {
                "figure_panel": "F validation",
                "input_csv": CELLTYPE_MAP_CSV.relative_to(ROOT).as_posix(),
                "use": "validate sender identifiers",
                "filter_or_selection": "six selected sender types",
            },
            {
                "figure_panel": "F donor provenance",
                "input_csv": DONOR_CSV.relative_to(ROOT).as_posix(),
                "use": "verify Q1/Q4 donor population",
                "filter_or_selection": ">=25 final LECs per donor",
            },
        ]
    ).to_csv(MANIFEST_CSV, index=False)

    figure, axis = plt.subplots(figsize=(15.0, 7.4), facecolor="white")
    figure.subplots_adjust(left=0.15, right=0.94, bottom=0.30, top=0.78)
    draw_panel_c(figure, axis, panel_c, ligand_order)
    figure.text(0.025, 0.96, "F", fontsize=18, fontweight="bold", color=TEXT_COLOR, va="top")
    figure.text(
        0.08,
        0.96,
        "Focused ligand–receptor interactions",
        fontsize=17,
        fontweight="bold",
        color=TEXT_COLOR,
        va="top",
    )
    figure.text(
        0.08,
        0.90,
        "Color and inset value show MultiNicheNet prioritization score; an empty cell indicates no selected top-50 edge.",
        fontsize=9,
        color="#5A6268",
        va="top",
    )
    figure.savefig(PNG, dpi=450, facecolor="white")
    figure.savefig(PDF, facecolor="white")
    figure.savefig(JPG, dpi=450, facecolor="white", pil_kwargs={"quality": 95, "subsampling": 0})
    plt.close(figure)
    print(f"Figure 3F written to: {OUTPUT_DIR}")
    print(f"Ligand-receptor pairs: {panel_c['ligand_receptor'].nunique()}")


if __name__ == "__main__":
    main()
