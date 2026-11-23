#!/usr/bin/env python3
"""Create aging Figure 3C: age-dependent abundance of selected LEC states."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Fig03A_aging_state_specific_regulons import (
    AGE_GROUPS,
    AGE_STATES,
    AGE_SUMMARY_CSV,
    AGE_TEST_CSV,
    DONOR_STATE_CSV,
    age_table,
    configure_style,
    draw_age_axis,
    load_inputs,
    project_root,
)


SCRIPT_VERSION = "Fig03C_aging_state_abundance_v1_2026_08_31"
OUTPUT_DIR = Path("new_result2/aging/Fig03C_aging_state_abundance")


def main() -> int:
    root = project_root()
    age = age_table(load_inputs(root))
    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)

    source = age.copy()
    source.insert(0, "script_version", SCRIPT_VERSION)
    source.to_csv(output / "Fig03C_source_data.csv", index=False)
    pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "input_file": DONOR_STATE_CSV.as_posix(),
                "used_fields": "donor_key;LEC_state;age_group;n_total_LEC_cells;cell_state_percentage",
                "figure_use": "C donor-level state abundance after the >=25-LEC filter",
            },
            {
                "script_version": SCRIPT_VERSION,
                "input_file": AGE_SUMMARY_CSV.as_posix(),
                "used_fields": "LEC_state;age_group;n_donors;mean_percentage;sem_percentage",
                "figure_use": "C age-group means and SEM",
            },
            {
                "script_version": SCRIPT_VERSION,
                "input_file": AGE_TEST_CSV.as_posix(),
                "used_fields": "LEC_state;kruskal_p;kruskal_fdr_bh;epsilon_squared",
                "figure_use": "C donor-level omnibus evidence",
            },
        ]
    ).to_csv(output / "Fig03C_input_manifest.csv", index=False)

    configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), facecolor="white")
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.18, top=0.72, wspace=0.29)
    rng = np.random.default_rng(20260823)
    for index, state in enumerate(AGE_STATES):
        draw_age_axis(axes[index], state, age, rng, show_panel_label=index == 0)
    figure.text(
        0.98,
        0.025,
        f"Age groups: {', '.join(AGE_GROUPS)}; donors n=4, 4, and 5.",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#666666",
    )
    figure.savefig(output / "Fig03C_aging_state_abundance.pdf", bbox_inches="tight")
    figure.savefig(output / "Fig03C_aging_state_abundance.png", dpi=600, bbox_inches="tight")
    figure.savefig(
        output / "Fig03C_aging_state_abundance.jpg",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)
    print(f"Figure 3C written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
