#!/usr/bin/env python3
"""Create disease Figure 5E: representative DCM communication network."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from Fig05D_disease_communication_modules import (
    ROOT,
    RUNS,
    SUBTEXT,
    TEXT,
    draw_network_panel,
    group_csv,
    load_selected_axes,
    summary_csv,
)


def run_network_panel(letter: str, disease: str, stem: str) -> None:
    selected, _ = load_selected_axes()
    displayed = selected.loc[selected["disease"].eq(disease)].copy()
    if displayed.empty:
        raise ValueError(f"No selected axes for {disease}")
    config = RUNS[disease]
    output = ROOT / "new_result2" / "disease" / stem
    output.mkdir(parents=True, exist_ok=True)
    displayed.to_csv(output / f"Fig05{letter}_source_data.csv", index=False)

    manifest_rows = [
        {
            "figure_panel": letter,
            "input_csv": group_csv(str(config["folder"])).relative_to(ROOT).as_posix(),
            "use": f"representative sender-ligand-receptor axes for {disease}",
            "filter_or_selection": f"focused {disease} axes selected for the main figure",
        },
        {
            "figure_panel": f"{letter} provenance",
            "input_csv": summary_csv(str(config["folder"])).relative_to(ROOT).as_posix(),
            "use": "validate completed run and donor counts",
            "filter_or_selection": str(config["tier"]),
        },
    ]
    if disease == "HFpEF":
        sensitivity_folder = str(config["sensitivity_folder"])
        manifest_rows.extend(
            [
                {
                    "figure_panel": f"{letter} sensitivity",
                    "input_csv": group_csv(sensitivity_folder).relative_to(ROOT).as_posix(),
                    "use": "HFpEF min15 receptor sensitivity",
                    "filter_or_selection": ">=15 LECs per donor",
                },
                {
                    "figure_panel": f"{letter} sensitivity provenance",
                    "input_csv": summary_csv(sensitivity_folder).relative_to(ROOT).as_posix(),
                    "use": "validate completed HFpEF min15 run",
                    "filter_or_selection": ">=15 LECs per donor",
                },
            ]
        )
    pd.DataFrame(manifest_rows).to_csv(output / f"Fig05{letter}_input_manifest.csv", index=False)

    figure, axis = plt.subplots(figsize=(10.8, 6.9), facecolor="white")
    figure.subplots_adjust(left=0.035, right=0.98, bottom=0.07, top=0.77)
    draw_network_panel(axis, disease, selected)
    figure.text(0.025, 0.96, letter, fontsize=18, fontweight="bold", color=TEXT, va="top")
    figure.text(
        0.08,
        0.96,
        f"Representative {disease} sender–ligand–LEC receptor axes",
        fontsize=16,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    figure.text(
        0.08,
        0.90,
        "Focused predicted communication network; full ligand–receptor and ligand–target networks remain supplementary.",
        fontsize=8.6,
        color=SUBTEXT,
        va="top",
    )
    figure.savefig(output / f"{stem}.png", dpi=450, facecolor="white")
    figure.savefig(output / f"{stem}.pdf", facecolor="white")
    figure.savefig(
        output / f"{stem}.jpg",
        dpi=450,
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)
    print(f"Figure 5{letter} written to: {output}")


def main() -> None:
    run_network_panel("E", "DCM", "Fig05E_DCM_sender_ligand_network")


if __name__ == "__main__":
    main()
