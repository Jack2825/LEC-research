#!/usr/bin/env python3
"""Create disease Figure 5C: donor-level abundance of regulon-defined states."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from Fig05A_disease_state_specific_regulons import (
    CELL_FILTER_CSV,
    COMPARISON_CSV,
    COMPARISON_ORDER,
    HFPEF_MIN10_SUMMARY_CSV,
    HFPEF_MIN15_SUMMARY_CSV,
    NEUTRAL,
    SUBTEXT,
    TEXT,
    configure_style,
    draw_effect_axis,
    draw_hfpef_effect_axis,
    effect_table,
    effect_xlim,
    hfpef_table,
    load_inputs,
    project_root,
)


SCRIPT_VERSION = "Fig05C_disease_state_abundance_v1_2026_08_31"
OUTPUT_DIR = Path("new_result2/disease/Fig05C_disease_state_abundance")


def main() -> int:
    root = project_root()
    inputs = load_inputs(root)
    effects = effect_table(inputs)
    hfpef = hfpef_table(inputs)
    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    pd.concat([effects, hfpef], ignore_index=True, sort=False).assign(
        script_version=SCRIPT_VERSION
    ).to_csv(output / "Fig05C_source_data.csv", index=False)
    pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "input_file": COMPARISON_CSV.as_posix(),
                "used_fields": "comparison;LEC_state;mean_difference_percentage_points;mean_difference_ci_low;mean_difference_ci_high;global_BH_with_HFpEF_min10",
                "figure_use": "C primary within-dataset disease-minus-NF effects",
            },
            {
                "script_version": SCRIPT_VERSION,
                "input_file": CELL_FILTER_CSV.as_posix(),
                "used_fields": "donor_key;dataset_folder;condition;donor_n_final_LECs;LEC_state",
                "figure_use": "C HFpEF-minus-NF effects at >=10 and >=15",
            },
            {
                "script_version": SCRIPT_VERSION,
                "input_file": HFPEF_MIN10_SUMMARY_CSV.as_posix(),
                "used_fields": "receiver_min_cells;n_reference_donors;n_disease_donors;validation_status;status",
                "figure_use": "C validate exploratory >=10 HFpEF tier",
            },
            {
                "script_version": SCRIPT_VERSION,
                "input_file": HFPEF_MIN15_SUMMARY_CSV.as_posix(),
                "used_fields": "receiver_min_cells;n_reference_donors;n_disease_donors;validation_status;status",
                "figure_use": "C validate >=15 HFpEF sensitivity tier",
            },
        ]
    ).to_csv(output / "Fig05C_input_manifest.csv", index=False)

    configure_style()
    figure = plt.figure(figsize=(16.2, 5.9), facecolor="white")
    grid = GridSpec(
        1,
        4,
        figure=figure,
        width_ratios=[1.0, 1.0, 1.0, 0.86],
        left=0.13,
        right=0.985,
        bottom=0.22,
        top=0.70,
        wspace=0.23,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    hfpef_axis = figure.add_subplot(grid[0, 3], sharey=axes[0])
    limits = effect_xlim(effects)
    for index, comparison in enumerate(COMPARISON_ORDER):
        draw_effect_axis(axes[index], comparison, effects, limits, show_labels=index == 0)
    draw_hfpef_effect_axis(
        hfpef_axis,
        inputs["hfpef_effects10"],  # type: ignore[arg-type]
        inputs["hfpef_effects15"],  # type: ignore[arg-type]
    )
    figure.text(0.025, 0.96, "C", fontsize=18, fontweight="bold", color=TEXT, va="top")
    figure.text(
        0.075,
        0.96,
        "Disease-linked abundance of regulon-defined states",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.075,
        0.895,
        "D01/D08: primary ≥25 LECs/donor. D11 HFpEF vs NF: exploratory ≥10 with ≥15 sensitivity; relaxed thresholds apply to both groups.",
        ha="left",
        va="top",
        fontsize=8.5,
        color=SUBTEXT,
    )
    handles = [
        Line2D([0], [0], marker="o", color=NEUTRAL, markerfacecolor=NEUTRAL, linewidth=1.2, label="global BH q ≤ 0.05"),
        Line2D([0], [0], marker="o", color=NEUTRAL, markerfacecolor="white", linewidth=1.2, label="global BH q > 0.05"),
    ]
    axes[2].legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.0, -0.29),
        ncol=2,
        frameon=False,
        fontsize=7,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    figure.savefig(output / "Fig05C_disease_state_abundance.pdf", bbox_inches="tight")
    figure.savefig(output / "Fig05C_disease_state_abundance.png", dpi=600, bbox_inches="tight")
    figure.savefig(
        output / "Fig05C_disease_state_abundance.jpg",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)
    print(f"Figure 5C written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
