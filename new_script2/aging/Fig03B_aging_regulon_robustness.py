#!/usr/bin/env python3
"""Create aging Figure 3B: donor-bootstrap regulon robustness."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from Fig03A_aging_state_specific_regulons import (
    SCENIC_FULL_CSV,
    configure_style,
    draw_panel_b,
    load_inputs,
    project_root,
    robustness_table,
)


SCRIPT_VERSION = "Fig03B_aging_regulon_robustness_v1_2026_08_31"
OUTPUT_DIR = Path("new_result2/aging/Fig03B_aging_regulon_robustness")


def main() -> int:
    root = project_root()
    robust = robustness_table(load_inputs(root))
    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)

    source = robust.copy()
    source.insert(0, "script_version", SCRIPT_VERSION)
    source.to_csv(output / "Fig03B_source_data.csv", index=False)
    pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "input_file": SCENIC_FULL_CSV.as_posix(),
                "used_fields": "LEC_state;TF;donor_bootstrap_RSS_median;donor_bootstrap_RSS_q025;donor_bootstrap_RSS_q975;donor_bootstrap_RSS_top_frequency",
                "figure_use": "B donor-bootstrap regulon specificity and robustness",
            }
        ]
    ).to_csv(output / "Fig03B_input_manifest.csv", index=False)

    configure_style()
    figure, axis = plt.subplots(figsize=(8.4, 5.3), facecolor="white")
    figure.subplots_adjust(left=0.29, right=0.96, bottom=0.20, top=0.80)
    draw_panel_b(axis, robust)
    figure.text(0.025, 0.965, "B", ha="left", va="top", fontsize=18, fontweight="bold")
    figure.text(
        0.08,
        0.965,
        "Reproducibility of aging-state regulons across donors",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output / "Fig03B_aging_regulon_robustness.pdf", bbox_inches="tight")
    figure.savefig(output / "Fig03B_aging_regulon_robustness.png", dpi=600, bbox_inches="tight")
    figure.savefig(
        output / "Fig03B_aging_regulon_robustness.jpg",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)
    print(f"Figure 3B written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
