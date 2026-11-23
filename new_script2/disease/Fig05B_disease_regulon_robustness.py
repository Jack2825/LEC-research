#!/usr/bin/env python3
"""Create disease Figure 5B: dataset/donor-bootstrap regulon robustness."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from Fig05A_disease_state_specific_regulons import (
    SCENIC_FULL_CSV,
    SUBTEXT,
    TEXT,
    configure_style,
    draw_panel_b,
    load_inputs,
    project_root,
    robustness_table,
)


SCRIPT_VERSION = "Fig05B_disease_regulon_robustness_v1_2026_08_31"
OUTPUT_DIR = Path("new_result2/disease/Fig05B_disease_regulon_robustness")


def main() -> int:
    root = project_root()
    robust = robustness_table(load_inputs(root))
    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    source = robust.copy()
    source.insert(0, "script_version", SCRIPT_VERSION)
    source.to_csv(output / "Fig05B_source_data.csv", index=False)
    pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "input_file": SCENIC_FULL_CSV.as_posix(),
                "used_fields": "LEC_state;TF;dataset_donor_bootstrap_RSS_median;dataset_donor_bootstrap_RSS_q025;dataset_donor_bootstrap_RSS_q975;dataset_donor_bootstrap_RSS_top_frequency",
                "figure_use": "B cross-dataset donor-bootstrap robustness",
            }
        ]
    ).to_csv(output / "Fig05B_input_manifest.csv", index=False)

    configure_style()
    figure, axis = plt.subplots(figsize=(8.5, 5.3), facecolor="white")
    figure.subplots_adjust(left=0.29, right=0.96, bottom=0.20, top=0.80)
    draw_panel_b(axis, robust)
    figure.text(0.025, 0.965, "B", ha="left", va="top", fontsize=18, fontweight="bold", color=TEXT)
    figure.text(
        0.08,
        0.965,
        "Reproducibility of disease-state regulons",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.96,
        0.025,
        "Strict dataset/donor-balanced Step 11 SCENIC analysis.",
        ha="right",
        va="bottom",
        fontsize=7,
        color=SUBTEXT,
    )
    figure.savefig(output / "Fig05B_disease_regulon_robustness.pdf", bbox_inches="tight")
    figure.savefig(output / "Fig05B_disease_regulon_robustness.png", dpi=600, bbox_inches="tight")
    figure.savefig(
        output / "Fig05B_disease_regulon_robustness.jpg",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)
    print(f"Figure 5B written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
