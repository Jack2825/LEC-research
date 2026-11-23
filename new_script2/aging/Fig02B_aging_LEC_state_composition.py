#!/usr/bin/env python3
"""Create Figure 2B: aging LEC-state composition after the >=25-LEC cutoff."""

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
INPUT_DIR = (
    ROOT
    / "new_result2_aging"
    / "08_7_min25_LEC_filtered_age_analysis"
    / "dataset13"
    / "three_groups_Q2_Q3_merged"
    / "donor_level"
)
POOLED_CSV = INPUT_DIR / "08_7_Q2_Q3_merged_pooled_state_composition.csv"
DONOR_CSV = INPUT_DIR / "08_7_Q2_Q3_merged_donor_normalized_state_composition.csv"
SUMMARY_CSV = INPUT_DIR / "08_7_Q2_Q3_merged_state_group_summary_mean_sem.csv"
COLOR_CSV = (
    ROOT
    / "new_result2_aging"
    / "08_combined_LEC_recluster"
    / "dataset13"
    / "08_state_name_mapping.csv"
)

OUTPUT_DIR = ROOT / "new_result2" / "aging" / "Fig02B_aging_LEC_state_composition"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PNG = OUTPUT_DIR / "Fig02B_aging_LEC_state_composition.png"
PDF = OUTPUT_DIR / "Fig02B_aging_LEC_state_composition.pdf"
JPG = OUTPUT_DIR / "Fig02B_aging_LEC_state_composition.jpg"
SOURCE_CSV = OUTPUT_DIR / "Fig02B_source_data.csv"
MANIFEST_CSV = OUTPUT_DIR / "Fig02B_input_manifest.csv"

AGE_ORDER = ["Q1", "Q2+Q3", "Q4"]
STATE_ORDER = [f"state{index}" for index in range(11)]
SCRIPT_VERSION = "Fig02B_aging_LEC_state_composition_v1_2026_08_31"
TEXT = "#17212B"
SUBTEXT = "#52606D"
GRID = "#DCE3E8"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9.2,
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    for path in [POOLED_CSV, DONOR_CSV, SUMMARY_CSV, COLOR_CSV]:
        if not path.exists():
            raise FileNotFoundError(path)

    pooled = pd.read_csv(POOLED_CSV)
    donor = pd.read_csv(DONOR_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    colors = pd.read_csv(COLOR_CSV)

    require_columns(
        pooled,
        {"age_group", "LEC_state", "pooled_state_percentage"},
        POOLED_CSV.name,
    )
    require_columns(
        donor,
        {
            "age_group",
            "LEC_state",
            "n_donors",
            "mean_donor_percentage",
            "percent_donors_with_state",
        },
        DONOR_CSV.name,
    )
    require_columns(
        summary,
        {"age_group", "LEC_state", "n_donors", "percent_donors_with_state"},
        SUMMARY_CSV.name,
    )
    require_columns(colors, {"LEC_state", "color_hex"}, COLOR_CSV.name)

    for frame in [pooled, donor, summary]:
        frame["LEC_state"] = frame["LEC_state"].astype(str)
        frame["state_number"] = frame["LEC_state"].map(state_number)
        unexpected_groups = sorted(set(frame["age_group"]) - set(AGE_ORDER))
        if unexpected_groups:
            raise ValueError(f"Unexpected age groups: {unexpected_groups}")

    state_colors = dict(zip(colors["LEC_state"], colors["color_hex"]))
    missing_colors = [state for state in STATE_ORDER if state not in state_colors]
    if missing_colors:
        raise ValueError(f"Missing state colors: {missing_colors}")

    return pooled, donor, summary, state_colors


def complete_matrix(
    frame: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    matrix = frame.pivot(index="age_group", columns="LEC_state", values=value_column)
    return matrix.reindex(index=AGE_ORDER, columns=STATE_ORDER, fill_value=0).fillna(0)


def draw_stacked(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    state_colors: dict[str, str],
    title: str,
    ylabel: str,
) -> None:
    positions = np.arange(len(AGE_ORDER))
    bottom = np.zeros(len(AGE_ORDER), dtype=float)
    for state in STATE_ORDER:
        values = matrix[state].to_numpy(dtype=float)
        axis.bar(
            positions,
            values,
            bottom=bottom,
            width=0.66,
            color=state_colors[state],
            edgecolor="white",
            linewidth=0.35,
        )
        bottom += values

    axis.set_title(title, loc="left", fontweight="bold", color=TEXT, pad=9)
    axis.set_ylabel(ylabel, color=TEXT)
    axis.set_xticks(positions, AGE_ORDER)
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 25, 50, 75, 100])
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
    axis.set_xticks(np.arange(len(STATE_ORDER)), [f"S{i}" for i in range(11)])
    axis.set_yticks(np.arange(len(AGE_ORDER)), AGE_ORDER)
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
                fontsize=7.4,
                color="white" if value >= 58 else TEXT,
            )
    for spine in axis.spines.values():
        spine.set_color("#89939D")
        spine.set_linewidth(0.7)
    return image


def write_traceability(
    pooled: pd.DataFrame,
    donor: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    source_parts: list[pd.DataFrame] = []
    for record_type, frame, source in [
        ("pooled_cell_composition", pooled, POOLED_CSV),
        ("equal_donor_composition", donor, DONOR_CSV),
        ("donor_state_prevalence", summary, SUMMARY_CSV),
    ]:
        current = frame.copy()
        current.insert(0, "script_version", SCRIPT_VERSION)
        current.insert(1, "figure_panel", "2B")
        current.insert(2, "record_type", record_type)
        current["minimum_LECs_per_donor"] = 25
        current["source_file"] = source.relative_to(ROOT).as_posix()
        source_parts.append(current)
    pd.concat(source_parts, ignore_index=True, sort=False).to_csv(SOURCE_CSV, index=False)

    pd.DataFrame(
        [
            {
                "figure_panel": "2B left",
                "input_file": POOLED_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "age_group;LEC_state;pooled_state_percentage",
                "purpose": "cell-weighted composition after the donor >=25-LEC cutoff",
            },
            {
                "figure_panel": "2B center",
                "input_file": DONOR_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "age_group;LEC_state;mean_donor_percentage",
                "purpose": "equal-donor state composition",
            },
            {
                "figure_panel": "2B right",
                "input_file": SUMMARY_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "age_group;LEC_state;n_donors;percent_donors_with_state",
                "purpose": "state prevalence across retained biological replicates",
            },
            {
                "figure_panel": "2B colors",
                "input_file": COLOR_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "LEC_state;color_hex",
                "purpose": "state colors consistent with Figure 2A",
            },
        ]
    ).assign(script_version=SCRIPT_VERSION).to_csv(MANIFEST_CSV, index=False)


def main() -> None:
    pooled, donor, summary, state_colors = load_data()
    pooled_matrix = complete_matrix(pooled, "pooled_state_percentage")
    donor_matrix = complete_matrix(donor, "mean_donor_percentage")
    support_matrix = complete_matrix(summary, "percent_donors_with_state")

    write_traceability(pooled, donor, summary)

    figure = plt.figure(figsize=(15.6, 5.6), facecolor="white")
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.32, 0.045],
        left=0.07,
        right=0.965,
        bottom=0.25,
        top=0.72,
        wspace=0.34,
    )
    pooled_axis = figure.add_subplot(grid[0, 0])
    donor_axis = figure.add_subplot(grid[0, 1])
    support_axis = figure.add_subplot(grid[0, 2])
    color_axis = figure.add_subplot(grid[0, 3])

    draw_stacked(
        pooled_axis,
        pooled_matrix,
        state_colors,
        "Cell-weighted composition",
        "LECs in age group (%)",
    )
    draw_stacked(
        donor_axis,
        donor_matrix,
        state_colors,
        "Equal-donor composition",
        "Mean within-donor abundance (%)",
    )
    image = draw_support(support_axis, support_matrix)
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("Donors with state (%)", color=TEXT)
    colorbar.ax.tick_params(labelsize=8, colors=TEXT)
    colorbar.outline.set_linewidth(0.6)

    figure.text(0.018, 0.955, "B", fontsize=18, fontweight="bold", color=TEXT, va="top")
    figure.text(
        0.065,
        0.955,
        "Aging LEC-state composition after donor filtering",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    figure.text(
        0.065,
        0.885,
        "Primary analysis uses donors with ≥25 LECs; Q2 and Q3 are combined. Equal-donor estimates are used for biological-replicate interpretation.",
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
        ncol=6,
        frameon=False,
        columnspacing=1.35,
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
    print(f"Figure 2B written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
