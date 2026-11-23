#!/usr/bin/env python3
r"""
Step 08.5: merge the Step 08.3 Q2 and Q3 donor groups and repeat Step 08.4.

This is a read-only downstream branch. It reads the Step 08.3 h5ad, preserves
``age_quartile`` and ``age_quartile_equal_LEC_donors``, and creates an in-memory
three-level label:

    Q1 -> Q1
    Q2 -> Q2+Q3
    Q3 -> Q2+Q3
    Q4 -> Q4

The donor-level analysis mirrors Step 08.4: complete donor x state counts
(including zeros), within-donor state percentages, individual donor points with
mean +/- SEM, Kruskal-Wallis with BH FDR, gated Dunn-Holm post hoc tests, Welch
ANOVA, continuous-age Spearman, leave-top-donor-out, minimum-cell, and
Project/Sex-adjusted sensitivity analyses.

Additional descriptive figures reproduce the useful earlier diagnostics:
age distribution and group transition, donor/cell support, pooled and
donor-normalized state composition, highest single-donor contribution,
Project/Sex balance, and UMAP summaries.

No input h5ad, expression matrix, embedding, cluster, or state label is changed.
The merged cell-level label is exported in a separate metadata CSV.

Default input
-------------
new_result2_aging/08_3_equal_LEC_donor_age_quartiles/dataset13/
    08_3_combined_LEC_equal_donor_age_quartiles.h5ad

Default output
--------------
new_result2_aging/08_5_merge_Q2_Q3_donor_level_analysis/dataset13/

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_aging\08_5_merge_Q2_Q3_donor_level_analysis.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg", force=True)


SCRIPT_VERSION = "08_5_dataset13_merge_Q2_Q3_donor_level_analysis_2026_08_10"
DEFAULT_INPUT = (
    "new_result2_aging/08_3_equal_LEC_donor_age_quartiles/dataset13/"
    "08_3_combined_LEC_equal_donor_age_quartiles.h5ad"
)
DEFAULT_OUTPUT_ROOT = (
    "new_result2_aging/08_5_merge_Q2_Q3_donor_level_analysis/dataset13"
)
DEFAULT_SOURCE_GROUP_KEY = "age_quartile_equal_LEC_donors"
DEFAULT_MERGED_GROUP_KEY = "age_group_Q2_Q3_merged"
DEFAULT_STATE_KEY = "LEC_state"
DEFAULT_DONOR_KEY = "donor_key"
DEFAULT_AGE_KEY = "age_numeric_resolved"
DEFAULT_PROJECT_KEY = "Project"
DEFAULT_SEX_KEY = "Sex"
DEFAULT_ORIGINAL_GROUP_KEY = "age_quartile"
SOURCE_GROUPS = ["Q1", "Q2", "Q3", "Q4"]
MERGED_GROUPS = ["Q1", "Q2+Q3", "Q4"]
GROUP_MAP = {"Q1": "Q1", "Q2": "Q2+Q3", "Q3": "Q2+Q3", "Q4": "Q4"}
SOURCE_COLORS = {
    "Q1": "#1f77b4",
    "Q2": "#ff7f0e",
    "Q3": "#2ca02c",
    "Q4": "#d62728",
}
MERGED_COLORS = {
    "Q1": "#1f77b4",
    "Q2+Q3": "#9467bd",
    "Q4": "#d62728",
}


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Step 08.3 Q2 and Q3 and repeat donor-level state-age "
            "statistics with expanded descriptive figures."
        )
    )
    parser.add_argument("--input_h5ad", default=DEFAULT_INPUT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source_group_key", default=DEFAULT_SOURCE_GROUP_KEY)
    parser.add_argument("--merged_group_key", default=DEFAULT_MERGED_GROUP_KEY)
    parser.add_argument("--state_key", default=DEFAULT_STATE_KEY)
    parser.add_argument("--donor_key", default=DEFAULT_DONOR_KEY)
    parser.add_argument("--age_key", default=DEFAULT_AGE_KEY)
    parser.add_argument("--project_key", default=DEFAULT_PROJECT_KEY)
    parser.add_argument("--sex_key", default=DEFAULT_SEX_KEY)
    parser.add_argument("--original_group_key", default=DEFAULT_ORIGINAL_GROUP_KEY)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min_donor_LECs_sensitivity", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def load_step08_4_module(project_dir: Path):
    path = project_dir / "new_script2_aging" / "08_4_donor_level_LEC_state_age_analysis.py"
    if not path.exists():
        raise FileNotFoundError(f"Required Step 08.4 implementation not found: {path}")
    spec = importlib.util.spec_from_file_location("step08_4_donor_analysis", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Step 08.4 module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_merged_tables(adata: ad.AnnData, args: argparse.Namespace, step08_4):
    # First use the tested Step 08.4 constructor against the original exact-12
    # groups, then merge Q2/Q3. This retains all zero donor-state combinations.
    step08_4.GROUPS = list(SOURCE_GROUPS)
    step08_4.GROUP_COLORS = dict(SOURCE_COLORS)
    donor_meta, donor_state, states, _ = step08_4.prepare_donor_state_table(
        adata.obs,
        args.donor_key,
        args.state_key,
        args.source_group_key,
        args.age_key,
        args.project_key,
        args.sex_key,
        args.original_group_key,
        expected_donors_per_group=12,
    )
    donor_meta["source_equal_donor_quartile"] = donor_meta["age_group"].astype(str)
    donor_state["source_equal_donor_quartile"] = donor_state["age_group"].astype(str)
    donor_meta["age_group"] = pd.Categorical(
        donor_meta["source_equal_donor_quartile"].map(GROUP_MAP),
        categories=MERGED_GROUPS,
        ordered=True,
    )
    donor_state["age_group"] = pd.Categorical(
        donor_state["source_equal_donor_quartile"].map(GROUP_MAP),
        categories=MERGED_GROUPS,
        ordered=True,
    )
    expected = pd.Series({"Q1": 12, "Q2+Q3": 24, "Q4": 12})
    observed = (
        donor_meta.groupby("age_group", observed=False).size().reindex(MERGED_GROUPS)
    )
    if not observed.eq(expected).all():
        raise RuntimeError(
            f"Merged donor-count invariant failed: {observed.to_dict()}"
        )
    range_labels: dict[str, str] = {}
    for group in MERGED_GROUPS:
        frame = donor_meta.loc[donor_meta["age_group"].eq(group)]
        range_labels[group] = (
            f"{group}\n{frame['age_numeric'].min():g}-{frame['age_numeric'].max():g} y"
            f"\nn={len(frame)}"
        )
    cell_merged = normalized_text(adata.obs[args.source_group_key]).map(GROUP_MAP)
    if cell_merged.isna().any():
        bad = normalized_text(adata.obs.loc[cell_merged.isna(), args.source_group_key]).unique()
        raise ValueError(f"Unmapped source age-group labels: {bad.tolist()}")
    return donor_meta, donor_state, states, range_labels, cell_merged


def make_descriptive_tables(
    donor_meta: pd.DataFrame,
    donor_state: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    group_rows: list[dict[str, object]] = []
    for group in MERGED_GROUPS:
        frame = donor_meta.loc[donor_meta["age_group"].eq(group)]
        group_rows.append(
            {
                "merged_age_group": group,
                "age_min": float(frame["age_numeric"].min()),
                "age_max": float(frame["age_numeric"].max()),
                "n_donors": int(len(frame)),
                "n_LEC_cells": int(frame["n_total_LEC_cells"].sum()),
                "median_LECs_per_donor": float(frame["n_total_LEC_cells"].median()),
                "min_LECs_per_donor": int(frame["n_total_LEC_cells"].min()),
                "max_LECs_per_donor": int(frame["n_total_LEC_cells"].max()),
                "n_donors_below_5_LECs": int(frame["n_total_LEC_cells"].lt(5).sum()),
                "n_donors_below_10_LECs": int(frame["n_total_LEC_cells"].lt(10).sum()),
                "n_projects": int(frame["Project"].nunique()),
            }
        )
    group_summary = pd.DataFrame(group_rows)

    transition = pd.crosstab(
        donor_meta["source_equal_donor_quartile"], donor_meta["age_group"]
    ).reindex(index=SOURCE_GROUPS, columns=MERGED_GROUPS, fill_value=0)
    transition.index.name = "source_equal_donor_quartile"
    transition = transition.reset_index()

    cell_composition = (
        donor_state.groupby(["age_group", "LEC_state"], observed=False)[
            "n_state_cells"
        ]
        .sum()
        .rename("n_cells")
        .reset_index()
    )
    group_totals = cell_composition.groupby("age_group", observed=True)[
        "n_cells"
    ].transform("sum")
    cell_composition["fraction_within_merged_group"] = np.where(
        group_totals.gt(0), cell_composition["n_cells"] / group_totals, np.nan
    )
    cell_composition["percent_within_merged_group"] = (
        100.0 * cell_composition["fraction_within_merged_group"]
    )

    donor_composition = (
        donor_state.groupby(["age_group", "LEC_state"], observed=False)
        .agg(
            n_donors=("donor_key", "size"),
            n_donors_with_state=("state_present", "sum"),
            mean_donor_percentage=("cell_state_percentage", "mean"),
            median_donor_percentage=("cell_state_percentage", "median"),
            sd_donor_percentage=("cell_state_percentage", "std"),
        )
        .reset_index()
    )
    donor_composition["sem_donor_percentage"] = (
        donor_composition["sd_donor_percentage"]
        / np.sqrt(donor_composition["n_donors"])
    )
    donor_composition["percent_donors_with_state"] = (
        100.0
        * donor_composition["n_donors_with_state"]
        / donor_composition["n_donors"]
    )

    observed_state = donor_state.loc[donor_state["n_state_cells"].gt(0)].copy()
    observed_state["n_merged_group_state_cells"] = (
        observed_state.groupby(["age_group", "LEC_state"], observed=True)[
            "n_state_cells"
        ].transform("sum")
    )
    observed_state["donor_fraction_of_merged_group_state"] = (
        observed_state["n_state_cells"]
        / observed_state["n_merged_group_state_cells"]
    )
    highest_state = (
        observed_state.sort_values(
            ["age_group", "LEC_state", "n_state_cells", "donor_key"],
            ascending=[True, True, False, True],
        )
        .groupby(["age_group", "LEC_state"], observed=True, sort=False)
        .head(1)
        .rename(
            columns={
                "donor_key": "top_donor_key",
                "n_state_cells": "top_donor_n_cells",
                "donor_fraction_of_merged_group_state": (
                    "top_donor_fraction_of_merged_group_state"
                ),
            }
        )[
            [
                "age_group",
                "LEC_state",
                "top_donor_key",
                "top_donor_n_cells",
                "n_merged_group_state_cells",
                "top_donor_fraction_of_merged_group_state",
            ]
        ]
        .copy()
    )
    highest_state["top_donor_percent_of_merged_group_state"] = (
        100.0 * highest_state["top_donor_fraction_of_merged_group_state"]
    )
    highest_state["top_donor_over_50_percent"] = highest_state[
        "top_donor_fraction_of_merged_group_state"
    ].gt(0.5)

    highest_overall = donor_meta.copy()
    highest_overall["n_merged_group_LECs"] = highest_overall.groupby(
        "age_group", observed=True
    )["n_total_LEC_cells"].transform("sum")
    highest_overall["donor_fraction_of_merged_group_LECs"] = (
        highest_overall["n_total_LEC_cells"]
        / highest_overall["n_merged_group_LECs"]
    )
    highest_overall = (
        highest_overall.sort_values(
            ["age_group", "n_total_LEC_cells", "donor_key"],
            ascending=[True, False, True],
        )
        .groupby("age_group", observed=True, sort=False)
        .head(1)
        .rename(
            columns={
                "donor_key": "top_donor_key",
                "n_total_LEC_cells": "top_donor_n_LEC_cells",
                "donor_fraction_of_merged_group_LECs": (
                    "top_donor_fraction_of_merged_group_LECs"
                ),
            }
        )[
            [
                "age_group",
                "top_donor_key",
                "top_donor_n_LEC_cells",
                "n_merged_group_LECs",
                "top_donor_fraction_of_merged_group_LECs",
            ]
        ]
        .copy()
    )
    highest_overall["top_donor_percent_of_merged_group_LECs"] = (
        100.0 * highest_overall["top_donor_fraction_of_merged_group_LECs"]
    )

    project_counts = (
        donor_meta.groupby(["age_group", "Project"], observed=True)
        .size()
        .rename("n_donors")
        .reset_index()
    )
    sex_counts = (
        donor_meta.groupby(["age_group", "Sex"], observed=True)
        .size()
        .rename("n_donors")
        .reset_index()
    )
    return {
        "group_summary": group_summary,
        "transition": transition,
        "cell_composition": cell_composition,
        "donor_composition": donor_composition,
        "highest_state": highest_state,
        "highest_overall": highest_overall,
        "project_counts": project_counts,
        "sex_counts": sex_counts,
    }


def save_per_state_figures_08_5(
    step08_4,
    donor_state: pd.DataFrame,
    group_summary: pd.DataFrame,
    omnibus: pd.DataFrame,
    posthoc: pd.DataFrame,
    states: list[str],
    range_labels: dict[str, str],
    output_dir: Path,
    dpi: int,
) -> list[dict[str, str]]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for state in states:
        frame = donor_state.loc[donor_state["LEC_state"].eq(state)].copy()
        state_summary = group_summary.loc[group_summary["LEC_state"].eq(state)].copy()
        omnibus_row = omnibus.loc[omnibus["LEC_state"].eq(state)].iloc[0]
        state_posthoc = posthoc.loc[posthoc["LEC_state"].eq(state)].copy()
        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        step08_4.draw_state_panel(
            ax,
            frame,
            state_summary,
            omnibus_row,
            range_labels,
            state_posthoc,
            show_ylabel=True,
            compact=False,
        )
        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=7,
                markerfacecolor="#777777", markeredgecolor="white",
                label="Individual donor",
            ),
            Line2D(
                [0], [0], marker="D", linestyle="none", markersize=6,
                markerfacecolor="white", markeredgecolor="black",
                label="Mean +/- SEM",
            ),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False)
        fig.suptitle("Q2+Q3 merged: donor-level LEC-state percentage", fontsize=13)
        fig.tight_layout()
        path = output_dir / f"08_5_{state}_donor_percentage_by_merged_age_group.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append(
            {
                "figure": str(path),
                "description": f"{state}: donor percentages and mean +/- SEM for Q1/Q2+Q3/Q4",
                "status": "ok",
            }
        )
    return manifest


def save_age_distribution_transition(
    donor_meta: pd.DataFrame,
    transition: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
    for position, group in enumerate(MERGED_GROUPS):
        frame = donor_meta.loc[donor_meta["age_group"].eq(group)].sort_values(
            ["age_numeric", "donor_key"]
        )
        jitter = np.linspace(-0.18, 0.18, len(frame))
        axes[0].scatter(
            frame["age_numeric"],
            np.full(len(frame), position) + jitter,
            s=55,
            color=MERGED_COLORS[group],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.88,
        )
    axes[0].set_yticks(range(len(MERGED_GROUPS)), MERGED_GROUPS)
    axes[0].set_xlabel("Resolved donor age")
    axes[0].set_ylabel("Merged age group")
    axes[0].set_title("Individual donor ages")
    axes[0].grid(axis="x", alpha=0.2)

    matrix = transition.set_index("source_equal_donor_quartile").reindex(
        index=SOURCE_GROUPS, columns=MERGED_GROUPS, fill_value=0
    )
    image = axes[1].imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    axes[1].set_xticks(range(len(MERGED_GROUPS)), MERGED_GROUPS)
    axes[1].set_yticks(range(len(SOURCE_GROUPS)), SOURCE_GROUPS)
    axes[1].set_xlabel("Merged group")
    axes[1].set_ylabel("Step 08.3 equal-donor quartile")
    axes[1].set_title("Deterministic Q2+Q3 merge")
    for row in range(len(SOURCE_GROUPS)):
        for column in range(len(MERGED_GROUPS)):
            axes[1].text(column, row, int(matrix.iloc[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axes[1], label="Number of donors", fraction=0.046)
    fig.suptitle("Merged age-group definition", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_donor_cell_support(
    donor_meta: pd.DataFrame,
    descriptive: dict[str, pd.DataFrame],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    summary = descriptive["group_summary"].set_index("merged_age_group").reindex(MERGED_GROUPS)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    colors = [MERGED_COLORS[group] for group in MERGED_GROUPS]
    bars = axes[0].bar(MERGED_GROUPS, summary["n_donors"], color=colors)
    axes[0].bar_label(bars)
    axes[0].set_title("Donor entries (identity QC pending)")
    axes[0].set_ylabel("Number of donor entries")
    bars = axes[1].bar(MERGED_GROUPS, summary["n_LEC_cells"], color=colors)
    axes[1].bar_label(bars)
    axes[1].set_title("Pooled LEC-cell support")
    axes[1].set_ylabel("Number of LEC cells")
    values = [
        donor_meta.loc[donor_meta["age_group"].eq(group), "n_total_LEC_cells"].to_numpy()
        for group in MERGED_GROUPS
    ]
    axes[2].boxplot(values, tick_labels=MERGED_GROUPS, showfliers=False)
    for position, (group, group_values) in enumerate(zip(MERGED_GROUPS, values), start=1):
        jitter = np.linspace(-0.12, 0.12, len(group_values))
        axes[2].scatter(
            np.full(len(group_values), position) + jitter,
            group_values,
            color=MERGED_COLORS[group],
            edgecolor="white",
            linewidth=0.4,
            s=36,
            alpha=0.85,
        )
    axes[2].axhline(5, color="black", linestyle="--", linewidth=1, label="5 LECs")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("LEC cells per donor (log scale)")
    axes[2].set_title("Unequal donor sampling")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Donor and LEC-cell support after merging Q2 and Q3", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_state_composition(
    descriptive: dict[str, pd.DataFrame],
    states: list[str],
    palette: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    pooled = descriptive["cell_composition"].pivot(
        index="age_group", columns="LEC_state", values="percent_within_merged_group"
    ).reindex(index=MERGED_GROUPS, columns=states, fill_value=0).fillna(0)
    donor = descriptive["donor_composition"].pivot(
        index="age_group", columns="LEC_state", values="mean_donor_percentage"
    ).reindex(index=MERGED_GROUPS, columns=states, fill_value=0).fillna(0)
    presence = descriptive["donor_composition"].pivot(
        index="age_group", columns="LEC_state", values="percent_donors_with_state"
    ).reindex(index=MERGED_GROUPS, columns=states, fill_value=0).fillna(0)
    colors = [palette[state] for state in states]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    pooled.plot(kind="bar", stacked=True, color=colors, ax=axes[0], width=0.72, legend=False)
    donor.plot(kind="bar", stacked=True, color=colors, ax=axes[1], width=0.72, legend=False)
    image = axes[2].imshow(presence.T.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=100)
    axes[0].set_title("Pooled-cell composition")
    axes[0].set_ylabel("Cells within merged group (%)")
    axes[1].set_title("Donor-normalized composition")
    axes[1].set_ylabel("Mean within-donor state percentage")
    for ax in axes[:2]:
        ax.set_xlabel("Merged age group")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.2)
    axes[2].set_title("Donor presence for each state")
    axes[2].set_xticks(range(len(MERGED_GROUPS)), MERGED_GROUPS)
    axes[2].set_yticks(range(len(states)), states)
    axes[2].set_xlabel("Merged age group")
    axes[2].set_ylabel("LEC state")
    fig.colorbar(image, ax=axes[2], label="Donors with state (%)", fraction=0.046)
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=palette[state],
                   markeredgecolor="none", markersize=9, label=state)
        for state in states
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(states)),
               bbox_to_anchor=(0.5, -0.04), title="LEC state")
    fig.suptitle("LEC-state composition after merging Q2 and Q3", fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compact_donor(value: object) -> str:
    return str(value).rsplit("::", maxsplit=1)[-1]


def save_highest_donor_contribution(
    descriptive: dict[str, pd.DataFrame],
    states: list[str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    state_table = descriptive["highest_state"]
    overall = descriptive["highest_overall"].set_index("age_group").reindex(MERGED_GROUPS)
    percent = state_table.pivot(
        index="LEC_state", columns="age_group",
        values="top_donor_percent_of_merged_group_state",
    ).reindex(index=states, columns=MERGED_GROUPS)
    donor = state_table.pivot(
        index="LEC_state", columns="age_group", values="top_donor_key"
    ).reindex(index=states, columns=MERGED_GROUPS)
    fig, axes = plt.subplots(
        1, 2, figsize=(15, max(6.5, 0.62 * len(states) + 2.3)),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    cmap = matplotlib.colormaps["YlOrRd"].copy()
    cmap.set_bad("#eeeeee")
    image = axes[0].imshow(percent.to_numpy(float), aspect="auto", cmap=cmap, vmin=0, vmax=100)
    axes[0].set_xticks(range(len(MERGED_GROUPS)), MERGED_GROUPS)
    axes[0].set_yticks(range(len(states)), states)
    axes[0].set_xlabel("Merged age group")
    axes[0].set_ylabel("LEC state")
    axes[0].set_title("Largest donor share within each group-state pool")
    for row, state in enumerate(states):
        for column, group in enumerate(MERGED_GROUPS):
            value = percent.loc[state, group]
            donor_key = donor.loc[state, group]
            if pd.isna(value):
                axes[0].text(column, row, "no cells", ha="center", va="center", fontsize=7)
                continue
            axes[0].text(
                column, row, f"{value:.1f}%\n{compact_donor(donor_key)}",
                ha="center", va="center", fontsize=7.2,
                color="white" if value >= 52 else "black",
                weight="bold" if value > 50 else "normal",
            )
            if value > 50:
                axes[0].add_patch(
                    Rectangle((column - 0.48, row - 0.48), 0.96, 0.96,
                              fill=False, edgecolor="black", linewidth=2.2)
                )
    fig.colorbar(image, ax=axes[0], label="Largest donor contribution (%)",
                 fraction=0.045, pad=0.03)
    values = overall["top_donor_percent_of_merged_group_LECs"].to_numpy(float)
    bars = axes[1].bar(
        MERGED_GROUPS, values,
        color=[MERGED_COLORS[group] for group in MERGED_GROUPS], width=0.7,
    )
    axes[1].axhline(50, color="black", linestyle="--", linewidth=1)
    for bar, group, value in zip(bars, MERGED_GROUPS, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, value + 2,
            f"{value:.1f}%\n{compact_donor(overall.loc[group, 'top_donor_key'])}",
            ha="center", va="bottom", fontsize=8,
        )
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("Largest donor contribution (%)")
    axes[1].set_title("Largest donor share of all group LECs")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Highest single-donor contribution after merging Q2 and Q3", fontsize=14)
    fig.text(
        0.5, 0.012,
        "Heatmap denominator: all cells in the same merged group and state. Black outline marks >50%.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_project_sex_balance(donor_meta: pd.DataFrame, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    project = pd.crosstab(donor_meta["age_group"], donor_meta["Project"]).reindex(MERGED_GROUPS)
    sex = pd.crosstab(donor_meta["age_group"], donor_meta["Sex"]).reindex(MERGED_GROUPS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    project.plot(kind="bar", stacked=True, ax=axes[0], width=0.75, colormap="tab10")
    sex.plot(kind="bar", stacked=True, ax=axes[1], width=0.75,
             color=["#e377c2", "#17becf", "#7f7f7f"][: len(sex.columns)])
    axes[0].set_title("Project distribution")
    axes[1].set_title("Sex distribution")
    for ax in axes:
        ax.set_xlabel("Merged age group")
        ax.set_ylabel("Number of donor entries")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Donor covariate balance after merging Q2 and Q3", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def categorical_scatter(ax, coordinates, values, palette, title, label_on_data=False):
    text = normalized_text(pd.Series(values)).to_numpy()
    categories = sorted(pd.unique(text), key=lambda value: str(value))
    for category in categories:
        mask = text == category
        ax.scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=9,
            color=palette.get(category, "#808080"), alpha=0.78, linewidth=0,
            label=category,
        )
        if label_on_data and mask.any():
            center = np.median(coordinates[mask], axis=0)
            ax.text(center[0], center[1], category, ha="center", va="center",
                    fontsize=9, weight="bold")
    if not label_on_data:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=2, fontsize=8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def save_umap_summary(
    adata: ad.AnnData,
    cell_merged: pd.Series,
    args: argparse.Namespace,
    state_palette: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    if "X_umap" not in adata.obsm:
        raise KeyError("Input Step 08.3 h5ad has no X_umap embedding.")
    xy = np.asarray(adata.obsm["X_umap"])
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    categorical_scatter(axes[0, 0], xy, cell_merged, MERGED_COLORS, "Merged Q1 / Q2+Q3 / Q4")
    categorical_scatter(
        axes[0, 1], xy, adata.obs[args.source_group_key], SOURCE_COLORS,
        "Preserved Step 08.3 equal-donor quartile",
    )
    categorical_scatter(
        axes[0, 2], xy, adata.obs[args.state_key], state_palette,
        "Unchanged LEC states", label_on_data=True,
    )
    age = pd.to_numeric(adata.obs[args.age_key], errors="coerce").to_numpy(float)
    image = axes[1, 0].scatter(xy[:, 0], xy[:, 1], c=age, s=9, cmap="viridis",
                               alpha=0.8, linewidth=0)
    axes[1, 0].set_title("Resolved donor age")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[1, 0].set_frame_on(False)
    fig.colorbar(image, ax=axes[1, 0], label="Age")
    projects = normalized_text(adata.obs[args.project_key])
    project_categories = sorted(projects.unique())
    base = list(matplotlib.colormaps["tab10"].colors)
    project_palette = {
        value: matplotlib.colors.to_hex(base[index % len(base)])
        for index, value in enumerate(project_categories)
    }
    categorical_scatter(axes[1, 1], xy, projects, project_palette, "Project")
    sex_palette = {"F": "#e377c2", "M": "#17becf", "unknown": "#7f7f7f"}
    categorical_scatter(axes[1, 2], xy, adata.obs[args.sex_key], sex_palette, "Sex")
    fig.suptitle("Step 08.5 metadata-only Q2+Q3 merge; embedding and states unchanged", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    path: Path,
    input_h5ad: Path,
    descriptive: dict[str, pd.DataFrame],
    omnibus: pd.DataFrame,
    posthoc: pd.DataFrame,
    continuous: pd.DataFrame,
    adjusted: pd.DataFrame,
    alpha: float,
    minimum_cells: int,
) -> None:
    summary_text = descriptive["group_summary"].to_csv(index=False).strip()
    primary = omnibus.loc[
        omnibus["kruskal_fdr_bh_significant"],
        ["LEC_state", "kruskal_p", "kruskal_fdr_bh", "epsilon_squared"],
    ]
    primary_text = (
        primary.to_csv(index=False).strip()
        if not primary.empty
        else "No state passed the primary BH-FDR threshold."
    )
    continuous_sig = continuous.loc[continuous["spearman_fdr_bh_significant"]]
    adjusted_sig = adjusted.loc[adjusted["age_group_fdr_bh_HC3_significant"]]
    text = f"""Step 08.5 Q2+Q3 merged donor-level analysis
================================================

Run identity
------------
Script version: {SCRIPT_VERSION}
Input h5ad: {input_h5ad}
Input h5ad modified: no

Merged group rule
-----------------
Step 08.3 Q1 remains Q1; Q2 and Q3 become Q2+Q3; Q4 remains Q4.
The preserved source labels are not overwritten. Because Step 08.3 split tied
ages at 48 and 62.5 to obtain exactly 12 donors per quartile, the merged Q1 and
Q2+Q3 ranges both contain age 48, and Q2+Q3 and Q4 both contain age 62.5.

Observed support
----------------
{summary_text}

Donor-level calculation
-----------------------
For each donor and state, including zero-count combinations:

    percentage = 100 * state cells / all LEC cells from that donor

Figures show individual donor entries and mean +/- SEM. Donor entry, not cell,
is the statistical replicate.

Primary statistics
------------------
Kruskal-Wallis tests compare Q1, Q2+Q3, and Q4 for each LEC state. BH FDR is
applied across states at alpha={alpha:g}. Dunn-Holm post hoc comparisons are run
only for states passing omnibus FDR.

{primary_text}

Post hoc rows generated: {len(posthoc)}

Sensitivity analyses
--------------------
Welch ANOVA, continuous-age Spearman, leave-top-donor-out, donors with at least
{minimum_cells} LECs, and Project/Sex-adjusted HC3 OLS are reported separately.
Continuous-age FDR-significant states: {int(continuous_sig.shape[0])}
Project/Sex-adjusted FDR-significant states: {int(adjusted_sig.shape[0])}

Composition figures
-------------------
Pooled-cell composition weights every cell equally. Donor-normalized composition
first calculates percentages within each donor and then averages donors. The
single-donor contribution heatmap uses all cells in the same merged group-state
pool as its denominator; small state pools can therefore produce large values.

Interpretation safeguards
-------------------------
The merged groups contain 12, 24, and 12 donor entries and are not balanced
tertiles. Q2+Q3 is a broad middle-age group, so this analysis tests young versus
middle versus old rather than a balanced monotonic age trend. Project and low-LEC
donors remain imbalanced. In addition, eight raw donor labels occur in both EGAS
and ERP Project namespaces; their biological identity should be resolved before
final donor-level inference. Use continuous age and the sensitivity tables when
interpreting any apparent group association.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1.")
    if args.min_donor_LECs_sensitivity < 1:
        raise ValueError("--min_donor_LECs_sensitivity must be at least 1.")
    project_dir = find_project_dir()
    input_h5ad = resolve_path(args.input_h5ad, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    sentinel = output_root / "08_5_run_summary.csv"
    if not input_h5ad.exists():
        raise FileNotFoundError(f"Step 08.3 input not found: {input_h5ad}")
    if sentinel.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Step 08.5 output exists: {sentinel}. Use --overwrite.")

    step08_4 = load_step08_4_module(project_dir)
    print(f"Reading Step 08.3 h5ad: {input_h5ad}")
    adata = ad.read_h5ad(input_h5ad)
    donor_meta, donor_state, states, range_labels, cell_merged = prepare_merged_tables(
        adata, args, step08_4
    )
    step08_4.GROUPS = list(MERGED_GROUPS)
    step08_4.GROUP_COLORS = dict(MERGED_COLORS)

    group_state_summary = step08_4.make_group_summary(donor_state)
    omnibus = step08_4.make_omnibus_table(donor_state, states, args.alpha)
    posthoc = step08_4.make_posthoc_table(donor_state, omnibus, args.alpha)
    continuous = step08_4.make_continuous_age_table(donor_state, states, args.alpha)
    leaveout = step08_4.make_leave_top_donor_out_table(
        donor_state, omnibus, states, args.alpha
    )
    minimum_cell = step08_4.make_minimum_cell_sensitivity(
        donor_state, states, args.min_donor_LECs_sensitivity, args.alpha
    )
    adjusted = step08_4.make_adjusted_sensitivity(donor_state, states, args.alpha)
    descriptive = make_descriptive_tables(donor_meta, donor_state)

    print("\nMerged group support:")
    print(descriptive["group_summary"].to_string(index=False))
    print("\nPrimary state-level tests:")
    print(
        omnibus[
            ["LEC_state", "kruskal_H", "kruskal_p", "kruskal_fdr_bh",
             "epsilon_squared", "kruskal_fdr_bh_significant"]
        ].to_string(index=False)
    )
    if args.dry_run:
        print("\nDry run complete. No files were written.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    per_state_dir = figures_dir / "per_state"
    if not args.skip_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    donor_meta.rename(columns={"age_group": args.merged_group_key}).to_csv(
        output_root / "08_5_donor_metadata.csv", index=False
    )
    donor_state.rename(columns={"age_group": args.merged_group_key}).to_csv(
        output_root / "08_5_donor_state_counts_percentages.csv.gz",
        index=False,
        compression="gzip",
    )
    group_state_summary.rename(columns={"age_group": args.merged_group_key}).to_csv(
        output_root / "08_5_state_group_summary_mean_sem.csv", index=False
    )
    omnibus.to_csv(output_root / "08_5_state_omnibus_tests.csv", index=False)
    posthoc.to_csv(output_root / "08_5_state_posthoc_Dunn_Holm.csv", index=False)
    continuous.to_csv(output_root / "08_5_continuous_age_spearman.csv", index=False)
    leaveout.to_csv(
        output_root / "08_5_leave_top_donor_out_sensitivity.csv", index=False
    )
    minimum_cell.to_csv(
        output_root / f"08_5_min{args.min_donor_LECs_sensitivity}_LEC_donor_sensitivity.csv",
        index=False,
    )
    adjusted.to_csv(
        output_root / "08_5_project_sex_adjusted_sensitivity.csv", index=False
    )
    table_paths = {
        "group_summary": "08_5_merged_group_summary.csv",
        "transition": "08_5_source_to_merged_group_transition.csv",
        "cell_composition": "08_5_pooled_state_composition.csv",
        "donor_composition": "08_5_donor_normalized_state_composition.csv",
        "highest_state": "08_5_highest_donor_contribution_by_group_state.csv",
        "highest_overall": "08_5_highest_donor_contribution_by_group.csv",
        "project_counts": "08_5_donor_counts_by_group_project.csv",
        "sex_counts": "08_5_donor_counts_by_group_sex.csv",
    }
    for key, filename in table_paths.items():
        descriptive[key].to_csv(output_root / filename, index=False)

    cell_metadata = adata.obs.copy()
    cell_metadata[args.merged_group_key] = pd.Categorical(
        cell_merged, categories=MERGED_GROUPS, ordered=True
    )
    cell_metadata.reset_index(names="combined_cell_id").to_csv(
        output_root / "08_5_combined_LEC_cell_metadata.csv.gz",
        index=False,
        compression="gzip",
    )

    palette = step08_4.state_color_map(adata, args.state_key, states)
    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        figure_manifest.extend(
            save_per_state_figures_08_5(
                step08_4, donor_state, group_state_summary, omnibus, posthoc,
                states, range_labels, per_state_dir, args.dpi,
            )
        )
        combined_path = figures_dir / "08_5_all_states_donor_percentage_by_merged_group.png"
        step08_4.save_all_states_figure(
            donor_state, group_state_summary, omnibus, states,
            range_labels, combined_path, args.dpi,
        )
        figure_manifest.append(
            {"figure": str(combined_path), "description": "All donor-level state percentages after Q2+Q3 merge", "status": "ok"}
        )
        statistical_path = figures_dir / "08_5_state_age_group_association_summary.png"
        step08_4.save_statistical_summary_figure(
            omnibus, palette, statistical_path, args.dpi, args.alpha
        )
        figure_manifest.append(
            {"figure": str(statistical_path), "description": "Kruskal-Wallis effect size and BH FDR summary", "status": "ok"}
        )
        descriptive_specs = [
            (
                "08_5_age_distribution_and_group_transition.png",
                "Donor ages and Step 08.3-to-merged group transition",
                lambda path: save_age_distribution_transition(
                    donor_meta, descriptive["transition"], path, args.dpi
                ),
            ),
            (
                "08_5_donor_and_LEC_cell_support.png",
                "Donor entries, pooled LEC cells, and LEC cells per donor",
                lambda path: save_donor_cell_support(
                    donor_meta, descriptive, path, args.dpi
                ),
            ),
            (
                "08_5_state_composition_pooled_donor_normalized_presence.png",
                "Pooled, donor-normalized, and donor-presence state composition",
                lambda path: save_state_composition(
                    descriptive, states, palette, path, args.dpi
                ),
            ),
            (
                "08_5_highest_single_donor_contribution.png",
                "Largest donor contribution by merged group-state and group overall",
                lambda path: save_highest_donor_contribution(
                    descriptive, states, path, args.dpi
                ),
            ),
            (
                "08_5_project_sex_balance.png",
                "Project and Sex balance among donor entries",
                lambda path: save_project_sex_balance(donor_meta, path, args.dpi),
            ),
            (
                "08_5_umap_summary.png",
                "Unchanged UMAP colored by merged and preserved metadata",
                lambda path: save_umap_summary(
                    adata, cell_merged, args, palette, path, args.dpi
                ),
            ),
        ]
        for filename, description, function in descriptive_specs:
            path = figures_dir / filename
            function(path)
            figure_manifest.append(
                {"figure": str(path), "description": description, "status": "ok"}
            )
    pd.DataFrame(
        figure_manifest, columns=["figure", "description", "status"]
    ).to_csv(output_root / "08_5_figure_manifest.csv", index=False)

    summary = descriptive["group_summary"].set_index("merged_age_group")
    significant_states = omnibus.loc[
        omnibus["kruskal_fdr_bh_significant"], "LEC_state"
    ].astype(str).tolist()
    run_summary = {
        "script_version": SCRIPT_VERSION,
        "status": "completed",
        "input_h5ad": str(input_h5ad),
        "output_root": str(output_root),
        "n_LEC_cells": int(adata.n_obs),
        "n_donor_entries": int(donor_meta["donor_key"].nunique()),
        "merged_groups": ";".join(MERGED_GROUPS),
        "donor_entries_by_group": ";".join(
            f"{group}:{int(summary.loc[group, 'n_donors'])}" for group in MERGED_GROUPS
        ),
        "LEC_cells_by_group": ";".join(
            f"{group}:{int(summary.loc[group, 'n_LEC_cells'])}" for group in MERGED_GROUPS
        ),
        "n_states": len(states),
        "primary_test": "Kruskal-Wallis on donor-level percentages",
        "multiple_testing": "Benjamini-Hochberg across states",
        "alpha": args.alpha,
        "n_primary_FDR_significant_states": len(significant_states),
        "primary_FDR_significant_states": ";".join(significant_states),
        "n_posthoc_rows": len(posthoc),
        "input_h5ad_modified": False,
    }
    pd.DataFrame([run_summary]).to_csv(sentinel, index=False)
    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "input_h5ad": str(input_h5ad),
        "output_root": str(output_root),
        "source_group_key": args.source_group_key,
        "merged_group_key": args.merged_group_key,
        "group_map": GROUP_MAP,
        "group_order": MERGED_GROUPS,
        "group_range_labels": range_labels,
        "state_key": args.state_key,
        "donor_key": args.donor_key,
        "age_key": args.age_key,
        "project_key": args.project_key,
        "sex_key": args.sex_key,
        "alpha": args.alpha,
        "primary_test": "Kruskal-Wallis with BH FDR",
        "posthoc": "Dunn-Holm only after omnibus FDR",
        "min_donor_LECs_sensitivity": args.min_donor_LECs_sensitivity,
        "figures_enabled": not args.skip_figures,
        "dpi": args.dpi,
        "input_h5ad_modified": False,
    }
    (output_root / "08_5_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    write_notes(
        output_root / "08_5_method_and_interpretation_notes.txt",
        input_h5ad,
        descriptive,
        omnibus,
        posthoc,
        continuous,
        adjusted,
        args.alpha,
        args.min_donor_LECs_sensitivity,
    )
    print(f"\nStep 08.5 outputs written to: {output_root}")
    print("Input h5ad was read only; original age labels, states, and embeddings are unchanged.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
