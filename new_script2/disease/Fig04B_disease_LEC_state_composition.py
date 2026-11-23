#!/usr/bin/env python3
"""Create Figure 4B: disease LEC-state composition after the >=25-LEC cutoff."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "new_result2_disease" / "08_3_min25_LEC_filtered_analysis" / "tables"
SUMMARY_CSV = INPUT_DIR / "08_3_retained_cohort_state_summary.csv"
COLOR_CSV = INPUT_DIR / "08_3_state_color_mapping.csv"
OUTPUT_DIR = ROOT / "new_result2" / "disease" / "Fig04B_disease_LEC_state_composition"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PNG = OUTPUT_DIR / "Fig04B_disease_LEC_state_composition.png"
PDF = OUTPUT_DIR / "Fig04B_disease_LEC_state_composition.pdf"
JPG = OUTPUT_DIR / "Fig04B_disease_LEC_state_composition.jpg"
SOURCE_CSV = OUTPUT_DIR / "Fig04B_source_data.csv"
MANIFEST_CSV = OUTPUT_DIR / "Fig04B_input_manifest.csv"

STATE_ORDER = [f"state{index}" for index in range(9)]
COHORT_ORDER = [
    "dataset01 | NF",
    "dataset01 | DCM",
    "dataset01 | HCM",
    "dataset08 | NF",
    "dataset08 | DoxCM",
    "dataset11 | HFpEF",
]
COHORT_LABELS = {
    "dataset01 | NF": "D01 NF",
    "dataset01 | DCM": "D01 DCM",
    "dataset01 | HCM": "D01 HCM",
    "dataset08 | NF": "D08 NF",
    "dataset08 | DoxCM": "D08 DoxCM",
    "dataset11 | HFpEF": "D11 HFpEF",
}
SCRIPT_VERSION = "Fig04B_disease_LEC_state_composition_v1_2026_08_31"
TEXT = "#17212B"
SUBTEXT = "#52606D"
GRID = "#DCE3E8"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9.1,
        "axes.titlesize": 11.5,
        "axes.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def state_number(value: object) -> int:
    match = re.fullmatch(r"state(\d+)", str(value).strip())
    if match is None:
        raise ValueError(f"Unexpected state label: {value!r}")
    return int(match.group(1))


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_data() -> tuple[pd.DataFrame, dict[str, str]]:
    for path in [SUMMARY_CSV, COLOR_CSV]:
        if not path.exists():
            raise FileNotFoundError(path)

    summary = pd.read_csv(SUMMARY_CSV)
    colors = pd.read_csv(COLOR_CSV)
    require_columns(
        summary,
        {
            "cohort",
            "dataset_folder",
            "condition",
            "LEC_state",
            "cell_pooled_percentage_within_cohort",
            "n_retained_donors",
            "percent_donors_with_state",
            "mean_within_donor_state_percentage",
        },
        SUMMARY_CSV.name,
    )
    require_columns(colors, {"LEC_state", "color"}, COLOR_CSV.name)

    summary["LEC_state"] = summary["LEC_state"].astype(str)
    summary["state_number"] = summary["LEC_state"].map(state_number)
    summary = summary.loc[
        summary["cohort"].isin(COHORT_ORDER) & (summary["n_retained_donors"] > 0)
    ].copy()
    observed = set(summary["cohort"])
    missing_cohorts = [cohort for cohort in COHORT_ORDER if cohort not in observed]
    if missing_cohorts:
        raise ValueError(f"Expected retained cohorts are missing: {missing_cohorts}")
    summary["cohort_label"] = summary["cohort"].map(COHORT_LABELS)

    state_colors = dict(zip(colors["LEC_state"], colors["color"]))
    missing_colors = [state for state in STATE_ORDER if state not in state_colors]
    if missing_colors:
        raise ValueError(f"Missing state colors: {missing_colors}")
    return summary, state_colors


def complete_matrix(summary: pd.DataFrame, value_column: str) -> pd.DataFrame:
    matrix = summary.pivot(index="cohort", columns="LEC_state", values=value_column)
    return matrix.reindex(index=COHORT_ORDER, columns=STATE_ORDER, fill_value=0).fillna(0)


def draw_stacked(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    state_colors: dict[str, str],
    title: str,
    ylabel: str,
) -> None:
    positions = np.arange(len(COHORT_ORDER))
    bottom = np.zeros(len(COHORT_ORDER), dtype=float)
    for state in STATE_ORDER:
        values = matrix[state].to_numpy(dtype=float)
        axis.bar(
            positions,
            values,
            bottom=bottom,
            width=0.72,
            color=state_colors[state],
            edgecolor="white",
            linewidth=0.3,
        )
        bottom += values
    labels = [COHORT_LABELS[cohort] for cohort in COHORT_ORDER]
    axis.set_xticks(positions, labels, rotation=35, ha="right", rotation_mode="anchor")
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_title(title, loc="left", fontweight="bold", color=TEXT, pad=9)
    axis.set_ylabel(ylabel, color=TEXT)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color=GRID, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#89939D")
    axis.tick_params(colors=TEXT)


def draw_support(axis: plt.Axes, support: pd.DataFrame) -> mpl.image.AxesImage:
    image = axis.imshow(
        support.to_numpy(dtype=float),
        cmap="Blues",
        vmin=0,
        vmax=100,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_title(
        "Donor prevalence of each state",
        loc="left",
        fontweight="bold",
        color=TEXT,
        pad=9,
    )
    axis.set_xticks(np.arange(len(STATE_ORDER)), [f"S{i}" for i in range(9)])
    axis.set_yticks(
        np.arange(len(COHORT_ORDER)),
        [COHORT_LABELS[cohort] for cohort in COHORT_ORDER],
    )
    axis.tick_params(length=0, colors=TEXT)
    for row in range(support.shape[0]):
        for column in range(support.shape[1]):
            value = float(support.iat[row, column])
            axis.text(
                column,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=6.8,
                color="white" if value >= 58 else TEXT,
            )
    for spine in axis.spines.values():
        spine.set_color("#89939D")
        spine.set_linewidth(0.7)
    return image


def write_traceability(summary: pd.DataFrame) -> None:
    source = summary.copy()
    source.insert(0, "script_version", SCRIPT_VERSION)
    source.insert(1, "figure_panel", "4B")
    source.insert(2, "record_type", "displayed_cohort_state_composition")
    source["minimum_LECs_per_donor"] = 25
    source["source_file"] = SUMMARY_CSV.relative_to(ROOT).as_posix()
    source.to_csv(SOURCE_CSV, index=False)

    pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "figure_panel": "4B composition and prevalence",
                "input_file": SUMMARY_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "cohort;dataset_folder;condition;LEC_state;cell_pooled_percentage_within_cohort;n_retained_donors;mean_within_donor_state_percentage;percent_donors_with_state",
                "purpose": "post-cutoff cell-weighted composition, equal-donor composition, and donor prevalence",
            },
            {
                "script_version": SCRIPT_VERSION,
                "figure_panel": "4B colors",
                "input_file": COLOR_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "LEC_state;color",
                "purpose": "state colors consistent with Figure 4A",
            },
        ]
    ).to_csv(MANIFEST_CSV, index=False)


def main() -> None:
    summary, state_colors = load_data()
    pooled = complete_matrix(summary, "cell_pooled_percentage_within_cohort")
    donor = complete_matrix(summary, "mean_within_donor_state_percentage")
    support = complete_matrix(summary, "percent_donors_with_state")
    write_traceability(summary)

    figure = plt.figure(figsize=(17.2, 6.4), facecolor="white")
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=[1.14, 1.14, 1.25, 0.04],
        left=0.065,
        right=0.967,
        bottom=0.28,
        top=0.72,
        wspace=0.33,
    )
    pooled_axis = figure.add_subplot(grid[0, 0])
    donor_axis = figure.add_subplot(grid[0, 1])
    support_axis = figure.add_subplot(grid[0, 2])
    color_axis = figure.add_subplot(grid[0, 3])

    draw_stacked(
        pooled_axis,
        pooled,
        state_colors,
        "Cell-weighted composition",
        "LECs in cohort (%)",
    )
    draw_stacked(
        donor_axis,
        donor,
        state_colors,
        "Equal-donor composition",
        "Mean within-donor abundance (%)",
    )
    image = draw_support(support_axis, support)
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("Donors with state (%)", color=TEXT)
    colorbar.ax.tick_params(labelsize=8, colors=TEXT)
    colorbar.outline.set_linewidth(0.6)

    figure.text(0.018, 0.955, "B", fontsize=18, fontweight="bold", color=TEXT, va="top")
    figure.text(
        0.06,
        0.955,
        "Disease-group composition of LEC states",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    figure.text(
        0.06,
        0.885,
        "Descriptive composition among retained donors (≥25 LECs). Equal-donor estimates prevent large donors from dominating cohort summaries.",
        fontsize=10.0,
        color=SUBTEXT,
        va="top",
    )

    legend_handles = [
        Patch(facecolor=state_colors[state], edgecolor="none", label=f"State {state_number(state)}")
        for state in STATE_ORDER
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=9,
        frameon=False,
        columnspacing=1.45,
        handlelength=1.45,
        fontsize=8.8,
    )

    figure.savefig(PNG, dpi=450, facecolor="white")
    figure.savefig(PDF, facecolor="white")
    figure.savefig(
        JPG,
        dpi=450,
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)
    print(f"Figure 4B written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
