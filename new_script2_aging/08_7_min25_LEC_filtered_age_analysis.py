#!/usr/bin/env python3
r"""
Step 08.7: donor-level age analysis after requiring at least 25 LECs.

This is a read-only downstream branch of the Step 08.3 h5ad. It preserves the
existing ``age_quartile_equal_LEC_donors`` labels and first analyzes the four
original Step 08.3 groups. It then performs a separate analysis in which Q2 and
Q3 are combined. Donors are never merged with one another; each retained
``donor_key`` remains one biological-replicate entry.

The filter is donor-level:

    retain donor if total observed LEC cells >= 25

For each retained donor and state, including zero-count donor-state pairs:

    state percentage = 100 * state cells / all LEC cells from that donor

Kruskal-Wallis tests compare donor-level percentages, followed by BH correction
across the 11 states. Dunn-Holm post hoc tests are generated only for states
passing omnibus BH FDR. The four-group test is deliberately marked not testable
when any group contains fewer than two donors; it is not replaced by a cell-
pooled test. The Q2+Q3 analysis remains testable with 4/4/5 donor entries.

No cells, labels, expression values, embeddings, clusters, states, or h5ad files
are changed.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import re
import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


matplotlib.use("Agg", force=True)


SCRIPT_VERSION = "08_7_dataset13_min25_LEC_preserved_and_merged_age_analysis_2026_08_10"
DEFAULT_INPUT = (
    "new_result2_aging/08_3_equal_LEC_donor_age_quartiles/dataset13/"
    "08_3_combined_LEC_equal_donor_age_quartiles.h5ad"
)
DEFAULT_OUTPUT_ROOT = (
    "new_result2_aging/08_7_min25_LEC_filtered_age_analysis/dataset13"
)
DEFAULT_GROUP_KEY = "age_quartile_equal_LEC_donors"
DEFAULT_STATE_KEY = "LEC_state"
DEFAULT_DONOR_KEY = "donor_key"
DEFAULT_AGE_KEY = "age_numeric_resolved"
DEFAULT_PROJECT_KEY = "Project"
DEFAULT_SEX_KEY = "Sex"
DEFAULT_ORIGINAL_GROUP_KEY = "age_quartile"
FOUR_GROUPS = ["Q1", "Q2", "Q3", "Q4"]
MERGED_GROUPS = ["Q1", "Q2+Q3", "Q4"]
MERGE_MAP = {"Q1": "Q1", "Q2": "Q2+Q3", "Q3": "Q2+Q3", "Q4": "Q4"}
COLORS = {
    "Q1": "#1f77b4",
    "Q2": "#ff7f0e",
    "Q3": "#2ca02c",
    "Q4": "#d62728",
    "Q2+Q3": "#9467bd",
}


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def natural_key(value: object) -> tuple[tuple[int, object], ...]:
    pieces = re.split(r"(\d+(?:\.\d+)?)", str(value))
    result: list[tuple[int, object]] = []
    for piece in pieces:
        if not piece:
            continue
        try:
            result.append((0, float(piece)))
        except ValueError:
            result.append((1, piece.lower()))
    return tuple(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat donor-level state-age analysis after a >=25 total-LEC donor "
            "filter, preserving four quartiles and separately merging Q2+Q3."
        )
    )
    parser.add_argument("--input_h5ad", default=DEFAULT_INPUT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--group_key", default=DEFAULT_GROUP_KEY)
    parser.add_argument("--state_key", default=DEFAULT_STATE_KEY)
    parser.add_argument("--donor_key", default=DEFAULT_DONOR_KEY)
    parser.add_argument("--age_key", default=DEFAULT_AGE_KEY)
    parser.add_argument("--project_key", default=DEFAULT_PROJECT_KEY)
    parser.add_argument("--sex_key", default=DEFAULT_SEX_KEY)
    parser.add_argument("--original_group_key", default=DEFAULT_ORIGINAL_GROUP_KEY)
    parser.add_argument("--minimum_LECs", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_step08_4(project_dir: Path):
    path = project_dir / "new_script2_aging" / "08_4_donor_level_LEC_state_age_analysis.py"
    if not path.is_file():
        raise FileNotFoundError(f"Required Step 08.4 helper script is missing: {path}")
    spec = importlib.util.spec_from_file_location("step08_4_for_08_7", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Step 08.4 helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_p(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.2e}" if number < 0.001 else f"{number:.3f}"


def prepare_filtered_tables(adata: ad.AnnData, args: argparse.Namespace, step08_4):
    donor_meta, donor_state, states, _ = step08_4.prepare_donor_state_table(
        adata.obs,
        args.donor_key,
        args.state_key,
        args.group_key,
        args.age_key,
        args.project_key,
        args.sex_key,
        args.original_group_key,
        expected_donors_per_group=12,
    )
    full_meta = donor_meta.copy()
    full_meta["age_group"] = full_meta["age_group"].astype(str)
    full_ranges = (
        full_meta.groupby("age_group", observed=True)["age_numeric"]
        .agg(preserved_age_min="min", preserved_age_max="max")
        .reindex(FOUR_GROUPS)
    )
    full_meta["retained_min25_LECs"] = full_meta["n_total_LEC_cells"].ge(
        args.minimum_LECs
    )
    eligible_keys = set(
        full_meta.loc[full_meta["retained_min25_LECs"], "donor_key"].astype(str)
    )
    filtered_meta = full_meta.loc[full_meta["donor_key"].isin(eligible_keys)].copy()
    filtered_state = donor_state.loc[donor_state["donor_key"].isin(eligible_keys)].copy()
    filtered_meta["age_group"] = pd.Categorical(
        filtered_meta["age_group"], categories=FOUR_GROUPS, ordered=True
    )
    filtered_state["age_group"] = pd.Categorical(
        filtered_state["age_group"].astype(str), categories=FOUR_GROUPS, ordered=True
    )
    observed = (
        filtered_meta.groupby("age_group", observed=False).size().reindex(FOUR_GROUPS)
    )
    expected = pd.Series([4, 1, 3, 5], index=FOUR_GROUPS)
    if not observed.astype(int).eq(expected).all():
        raise RuntimeError(
            f">={args.minimum_LECs} retained-count invariant failed: "
            f"observed {observed.to_dict()}, expected {expected.to_dict()}"
        )
    if len(filtered_meta) != 13:
        raise RuntimeError(f"Expected 13 retained donor entries, found {len(filtered_meta)}")
    if int(filtered_meta["n_total_LEC_cells"].sum()) != 818:
        raise RuntimeError("Expected 818 retained LEC cells at the >=25 threshold.")
    return full_meta, filtered_meta, filtered_state, states, full_ranges


def make_scheme_tables(
    filtered_meta: pd.DataFrame,
    filtered_state: pd.DataFrame,
    full_ranges: pd.DataFrame,
    merged: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame, dict[str, str]]:
    meta = filtered_meta.copy()
    state = filtered_state.copy()
    if merged:
        groups = MERGED_GROUPS
        meta["source_preserved_quartile"] = meta["age_group"].astype(str)
        state["source_preserved_quartile"] = state["age_group"].astype(str)
        meta["age_group"] = pd.Categorical(
            meta["source_preserved_quartile"].map(MERGE_MAP),
            categories=groups,
            ordered=True,
        )
        state["age_group"] = pd.Categorical(
            state["source_preserved_quartile"].map(MERGE_MAP),
            categories=groups,
            ordered=True,
        )
        preserved_ranges = pd.DataFrame(
            {
                "preserved_age_min": [
                    full_ranges.loc["Q1", "preserved_age_min"],
                    min(
                        full_ranges.loc["Q2", "preserved_age_min"],
                        full_ranges.loc["Q3", "preserved_age_min"],
                    ),
                    full_ranges.loc["Q4", "preserved_age_min"],
                ],
                "preserved_age_max": [
                    full_ranges.loc["Q1", "preserved_age_max"],
                    max(
                        full_ranges.loc["Q2", "preserved_age_max"],
                        full_ranges.loc["Q3", "preserved_age_max"],
                    ),
                    full_ranges.loc["Q4", "preserved_age_max"],
                ],
            },
            index=groups,
        )
    else:
        groups = FOUR_GROUPS
        meta["age_group"] = pd.Categorical(
            meta["age_group"].astype(str), categories=groups, ordered=True
        )
        state["age_group"] = pd.Categorical(
            state["age_group"].astype(str), categories=groups, ordered=True
        )
        preserved_ranges = full_ranges.reindex(groups).copy()

    rows: list[dict[str, object]] = []
    range_labels: dict[str, str] = {}
    for group in groups:
        frame = meta.loc[meta["age_group"].eq(group)]
        preserved_min = float(preserved_ranges.loc[group, "preserved_age_min"])
        preserved_max = float(preserved_ranges.loc[group, "preserved_age_max"])
        rows.append(
            {
                "age_group": group,
                "preserved_age_min": preserved_min,
                "preserved_age_max": preserved_max,
                "eligible_age_min": float(frame["age_numeric"].min()),
                "eligible_age_max": float(frame["age_numeric"].max()),
                "n_donor_entries": int(len(frame)),
                "n_LEC_cells": int(frame["n_total_LEC_cells"].sum()),
                "median_LECs_per_donor": float(frame["n_total_LEC_cells"].median()),
                "min_LECs_per_donor": int(frame["n_total_LEC_cells"].min()),
                "max_LECs_per_donor": int(frame["n_total_LEC_cells"].max()),
            }
        )
        range_labels[group] = (
            f"{group}\n{preserved_min:g}-{preserved_max:g} y\nn={len(frame)}"
        )
    return meta, state, groups, pd.DataFrame(rows), range_labels


def make_composition_tables(
    donor_meta: pd.DataFrame,
    donor_state: pd.DataFrame,
    groups: list[str],
    states: list[str],
) -> dict[str, pd.DataFrame]:
    pooled = (
        donor_state.groupby(["age_group", "LEC_state"], observed=False)["n_state_cells"]
        .sum()
        .rename("n_state_cells")
        .reset_index()
    )
    pooled["n_group_LEC_cells"] = pooled.groupby("age_group", observed=False)[
        "n_state_cells"
    ].transform("sum")
    pooled["pooled_state_percentage"] = np.where(
        pooled["n_group_LEC_cells"].gt(0),
        100.0 * pooled["n_state_cells"] / pooled["n_group_LEC_cells"],
        np.nan,
    )
    normalized = (
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
    normalized["sem_donor_percentage"] = (
        normalized["sd_donor_percentage"] / np.sqrt(normalized["n_donors"])
    )
    normalized["percent_donors_with_state"] = (
        100.0 * normalized["n_donors_with_state"] / normalized["n_donors"]
    )

    dominance_rows: list[dict[str, object]] = []
    for group in groups:
        for state in states:
            frame = donor_state.loc[
                donor_state["age_group"].eq(group) & donor_state["LEC_state"].eq(state)
            ].copy()
            total = int(frame["n_state_cells"].sum())
            top = frame.sort_values(
                ["n_state_cells", "donor_key"], ascending=[False, True]
            ).iloc[0]
            dominance_rows.append(
                {
                    "age_group": group,
                    "LEC_state": state,
                    "n_group_state_cells": total,
                    "n_donors_with_state": int(frame["state_present"].sum()),
                    "top_donor_key": str(top["donor_key"]),
                    "top_donor_n_state_cells": int(top["n_state_cells"]),
                    "top_donor_percent_of_group_state": (
                        100.0 * int(top["n_state_cells"]) / total if total else np.nan
                    ),
                }
            )
    dominance = pd.DataFrame(dominance_rows)

    overall_rows: list[dict[str, object]] = []
    for group in groups:
        frame = donor_meta.loc[donor_meta["age_group"].eq(group)].copy()
        group_cells = int(frame["n_total_LEC_cells"].sum())
        top = frame.sort_values(
            ["n_total_LEC_cells", "donor_key"], ascending=[False, True]
        ).iloc[0]
        overall_rows.append(
            {
                "age_group": group,
                "n_group_LEC_cells": group_cells,
                "top_donor_key": str(top["donor_key"]),
                "top_donor_n_LEC_cells": int(top["n_total_LEC_cells"]),
                "top_donor_percent_of_group_LECs": (
                    100.0 * int(top["n_total_LEC_cells"]) / group_cells
                ),
            }
        )
    return {
        "pooled": pooled,
        "normalized": normalized,
        "dominance": dominance,
        "overall_dominance": pd.DataFrame(overall_rows),
    }


def add_bh_fdr(
    table: pd.DataFrame,
    p_column: str,
    q_column: str,
    alpha: float,
) -> pd.DataFrame:
    result = table.copy()
    result[q_column] = np.nan
    result[f"{q_column}_significant"] = False
    finite = pd.to_numeric(result[p_column], errors="coerce").notna()
    if finite.any():
        reject, adjusted, _, _ = multipletests(
            result.loc[finite, p_column].astype(float),
            alpha=alpha,
            method="fdr_bh",
        )
        result.loc[finite, q_column] = adjusted
        result.loc[finite, f"{q_column}_significant"] = reject
    return result


def run_cell_level_statistics(
    donor_state: pd.DataFrame,
    groups: list[str],
    states: list[str],
    alpha: float,
) -> dict[str, pd.DataFrame]:
    """Cell-count association tests; cells are explicitly not donor replicates."""
    counts = (
        donor_state.groupby(["age_group", "LEC_state"], observed=False)["n_state_cells"]
        .sum()
        .unstack("LEC_state", fill_value=0)
        .reindex(index=groups, columns=states, fill_value=0)
        .astype(int)
    )
    group_totals = counts.sum(axis=1)
    long = counts.stack().rename("n_state_cells").reset_index()
    long["n_group_LEC_cells"] = long["age_group"].map(group_totals)
    long["pooled_state_percentage"] = (
        100.0 * long["n_state_cells"] / long["n_group_LEC_cells"]
    )

    global_array = counts.to_numpy(int)
    global_chi2, global_p, global_df, global_expected = stats.chi2_contingency(
        global_array, correction=False
    )
    global_v = math.sqrt(
        float(global_chi2)
        / (float(global_array.sum()) * min(len(groups) - 1, len(states) - 1))
    )
    global_test = pd.DataFrame(
        [
            {
                "test": "global_group_by_11_state_composition",
                "n_cells": int(global_array.sum()),
                "n_groups": len(groups),
                "n_states": len(states),
                "chi_square": float(global_chi2),
                "degrees_of_freedom": int(global_df),
                "p_value": float(global_p),
                "cramers_v": global_v,
                "minimum_expected_count": float(global_expected.min()),
                "n_expected_counts_below_5": int((global_expected < 5).sum()),
                "independence_warning": "cells nested within donors; descriptive/sensitivity inference only",
            }
        ]
    )

    rows: list[dict[str, object]] = []
    for state in states:
        state_counts = counts[state].to_numpy(int)
        other_counts = group_totals.to_numpy(int) - state_counts
        contingency = np.vstack([state_counts, other_counts])
        try:
            chi2, p_value, df, expected = stats.chi2_contingency(
                contingency, correction=False
            )
            status = "ok"
            effect = math.sqrt(float(chi2) / float(contingency.sum()))
            minimum_expected = float(expected.min())
            below_five = int((expected < 5).sum())
        except Exception as exc:
            chi2, p_value, df, effect = np.nan, np.nan, len(groups) - 1, np.nan
            minimum_expected, below_five = np.nan, 0
            status = f"not_testable: {exc}"
        row: dict[str, object] = {
            "LEC_state": state,
            "n_state_cells": int(state_counts.sum()),
            "n_other_LEC_cells": int(other_counts.sum()),
            "chi_square": chi2,
            "degrees_of_freedom": df,
            "chi_square_p": p_value,
            "cramers_v_state_vs_other": effect,
            "minimum_expected_count": minimum_expected,
            "n_expected_counts_below_5": below_five,
            "test_status": status,
            "independence_warning": "cells nested within donors",
        }
        for group, state_count, total_count in zip(groups, state_counts, group_totals):
            row[f"n_state_cells_{group}"] = int(state_count)
            row[f"n_group_cells_{group}"] = int(total_count)
            row[f"pooled_percentage_{group}"] = 100.0 * state_count / total_count
        rows.append(row)
    omnibus = add_bh_fdr(
        pd.DataFrame(rows), "chi_square_p", "chi_square_fdr_bh", alpha
    )

    posthoc_rows: list[dict[str, object]] = []
    significant_states = omnibus.loc[
        omnibus["chi_square_fdr_bh_significant"], "LEC_state"
    ].astype(str)
    for state in significant_states:
        state_rows: list[dict[str, object]] = []
        for group1, group2 in itertools.combinations(groups, 2):
            state1 = int(counts.loc[group1, state])
            state2 = int(counts.loc[group2, state])
            total1 = int(group_totals.loc[group1])
            total2 = int(group_totals.loc[group2])
            table = np.array(
                [[state1, total1 - state1], [state2, total2 - state2]], dtype=int
            )
            odds_ratio, p_value = stats.fisher_exact(table, alternative="two-sided")
            state_rows.append(
                {
                    "LEC_state": state,
                    "omnibus_chi_square_fdr_bh": float(
                        omnibus.loc[omnibus["LEC_state"].eq(state), "chi_square_fdr_bh"].iloc[0]
                    ),
                    "group1": group1,
                    "group2": group2,
                    "n_state_cells_group1": state1,
                    "n_all_cells_group1": total1,
                    "pooled_percentage_group1": 100.0 * state1 / total1,
                    "n_state_cells_group2": state2,
                    "n_all_cells_group2": total2,
                    "pooled_percentage_group2": 100.0 * state2 / total2,
                    "odds_ratio_group1_vs_group2": float(odds_ratio),
                    "fisher_exact_p": float(p_value),
                }
            )
        state_table = pd.DataFrame(state_rows)
        reject, adjusted, _, _ = multipletests(
            state_table["fisher_exact_p"], alpha=alpha, method="holm"
        )
        state_table["fisher_holm_p"] = adjusted
        state_table["fisher_holm_significant"] = reject
        posthoc_rows.extend(state_table.to_dict("records"))
    posthoc_columns = [
        "LEC_state", "omnibus_chi_square_fdr_bh", "group1", "group2",
        "n_state_cells_group1", "n_all_cells_group1", "pooled_percentage_group1",
        "n_state_cells_group2", "n_all_cells_group2", "pooled_percentage_group2",
        "odds_ratio_group1_vs_group2", "fisher_exact_p", "fisher_holm_p",
        "fisher_holm_significant",
    ]
    posthoc = pd.DataFrame(posthoc_rows, columns=posthoc_columns)
    return {
        "group_state_counts": long,
        "global_test": global_test,
        "omnibus": omnibus,
        "posthoc": posthoc,
    }


def run_statistics(
    step08_4,
    donor_state: pd.DataFrame,
    states: list[str],
    groups: list[str],
    alpha: float,
) -> dict[str, pd.DataFrame]:
    original_groups = list(step08_4.GROUPS)
    original_colors = dict(step08_4.GROUP_COLORS)
    try:
        step08_4.GROUPS = list(groups)
        step08_4.GROUP_COLORS = {group: COLORS[group] for group in groups}
        summary = step08_4.make_group_summary(donor_state)
        omnibus = step08_4.make_omnibus_table(donor_state, states, alpha)
        posthoc = step08_4.make_posthoc_table(donor_state, omnibus, alpha)
        continuous = step08_4.make_continuous_age_table(donor_state, states, alpha)
        leaveout = step08_4.make_leave_top_donor_out_table(
            donor_state, omnibus, states, alpha
        )
        adjusted = step08_4.make_adjusted_sensitivity(donor_state, states, alpha)
    finally:
        step08_4.GROUPS = original_groups
        step08_4.GROUP_COLORS = original_colors
    return {
        "group_state_summary": summary,
        "omnibus": omnibus,
        "posthoc": posthoc,
        "continuous": continuous,
        "leaveout": leaveout,
        "adjusted": adjusted,
    }


def save_all_state_points(
    donor_state: pd.DataFrame,
    group_summary: pd.DataFrame,
    omnibus: pd.DataFrame,
    groups: list[str],
    states: list[str],
    range_labels: dict[str, str],
    path: Path,
    dpi: int,
    per_state_dir: Path,
) -> list[dict[str, str]]:
    import matplotlib.pyplot as plt

    def draw(ax, state: str, compact: bool) -> None:
        frame = donor_state.loc[donor_state["LEC_state"].eq(state)]
        state_summary = group_summary.loc[group_summary["LEC_state"].eq(state)]
        test = omnibus.loc[omnibus["LEC_state"].eq(state)].iloc[0]
        for position, group in enumerate(groups):
            points = frame.loc[frame["age_group"].eq(group)].sort_values("donor_key")
            jitter = np.linspace(-0.15, 0.15, len(points))
            ax.scatter(
                np.full(len(points), position) + jitter,
                points["cell_state_percentage"],
                s=28 if compact else 48,
                color=COLORS[group],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            row = state_summary.loc[state_summary["age_group"].eq(group)].iloc[0]
            mean = float(row["mean_percentage"])
            sem = float(row["sem_percentage"])
            if np.isfinite(sem):
                ax.errorbar(
                    position, mean, yerr=sem, fmt="D", color="black",
                    markerfacecolor="white", markersize=5 if compact else 7,
                    capsize=4, linewidth=1.4, zorder=5,
                )
            else:
                ax.scatter(
                    [position], [mean], marker="D", facecolor="white",
                    edgecolor="black", s=32 if compact else 52, zorder=5,
                )
        ax.set_xticks(
            range(len(groups)),
            groups if compact else [range_labels[group] for group in groups],
        )
        ax.set_xlim(-0.45, len(groups) - 0.55)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        status = str(test["test_status"])
        if status == "ok":
            title = (
                f"{state}: KW p={format_p(test['kruskal_p'])}, "
                f"FDR={format_p(test['kruskal_fdr_bh'])}"
            )
        else:
            title = f"{state}: KW not testable ({status})"
        ax.set_title(title, fontsize=10 if compact else 12)

    per_state_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for state in states:
        fig, ax = plt.subplots(figsize=(7.6, 5.8), constrained_layout=True)
        draw(ax, state, compact=False)
        ax.set_ylabel("Donor LECs in state (%)")
        fig.suptitle("Retained donors: individual percentages and mean +/- SEM")
        state_path = per_state_dir / f"08_7_{state}_donor_percentage.png"
        fig.savefig(state_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append(
            {"figure": str(state_path), "description": f"{state} retained-donor percentages", "status": "ok"}
        )

    n_columns = 3
    n_rows = math.ceil(len(states) / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 4.0 * n_rows))
    flat = np.atleast_1d(axes).ravel()
    for index, state in enumerate(states):
        draw(flat[index], state, compact=True)
        if index % n_columns == 0:
            flat[index].set_ylabel("Donor state (%)")
    for ax in flat[len(states):]:
        ax.set_visible(False)
    fig.suptitle("Minimum 25 LECs: donor-level state percentages", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    manifest.append(
        {"figure": str(path), "description": "All retained-donor state percentages", "status": "ok"}
    )
    return manifest


def save_group_support(
    donor_meta: pd.DataFrame,
    overview: pd.DataFrame,
    groups: list[str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    table = overview.set_index("age_group").reindex(groups)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    colors = [COLORS[group] for group in groups]
    bars = axes[0].bar(groups, table["n_donor_entries"], color=colors)
    axes[0].bar_label(bars, fmt="%d")
    axes[0].set_ylabel("Retained donor entries")
    axes[0].set_title("Donors after >=25 LEC filter")
    bars = axes[1].bar(groups, table["n_LEC_cells"], color=colors)
    axes[1].bar_label(bars, fmt="%d")
    axes[1].set_ylabel("Retained LEC cells")
    axes[1].set_title("Pooled cell support")
    for position, group in enumerate(groups):
        values = donor_meta.loc[
            donor_meta["age_group"].eq(group), "n_total_LEC_cells"
        ].to_numpy(float)
        jitter = np.linspace(-0.12, 0.12, len(values))
        axes[2].scatter(
            np.full(len(values), position) + jitter,
            values,
            color=COLORS[group], edgecolor="white", s=52,
        )
    axes[2].axhline(25, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_xticks(range(len(groups)), groups)
    axes[2].set_ylabel("LECs per donor (log scale)")
    axes[2].set_title("Individual retained donors")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_composition(
    composition: dict[str, pd.DataFrame],
    groups: list[str],
    states: list[str],
    state_colors: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    pooled = composition["pooled"].pivot(
        index="age_group", columns="LEC_state", values="pooled_state_percentage"
    ).reindex(index=groups, columns=states, fill_value=0)
    normalized = composition["normalized"].pivot(
        index="age_group", columns="LEC_state", values="mean_donor_percentage"
    ).reindex(index=groups, columns=states, fill_value=0)
    presence = composition["normalized"].pivot(
        index="age_group", columns="LEC_state", values="percent_donors_with_state"
    ).reindex(index=groups, columns=states, fill_value=0)
    colors = [state_colors[state] for state in states]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    pooled.plot(kind="bar", stacked=True, color=colors, ax=axes[0], width=0.72, legend=False)
    normalized.plot(kind="bar", stacked=True, color=colors, ax=axes[1], width=0.72, legend=False)
    image = axes[2].imshow(presence.to_numpy(float), aspect="auto", cmap="Blues", vmin=0, vmax=100)
    axes[0].set_title("Pooled-cell composition")
    axes[0].set_ylabel("State percentage of pooled group LECs")
    axes[1].set_title("Donor-normalized composition")
    axes[1].set_ylabel("Mean within-donor state percentage")
    for ax in axes[:2]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    axes[2].set_xticks(range(len(states)), states, rotation=60, ha="right")
    axes[2].set_yticks(range(len(groups)), groups)
    axes[2].set_title("Donors containing each state")
    fig.colorbar(image, ax=axes[2], label="Donors with state (%)", fraction=0.046)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title="LEC state", bbox_to_anchor=(1.01, 0.98), loc="upper left")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compact_donor(value: object) -> str:
    return str(value).split("::")[-1]


def save_dominance(
    composition: dict[str, pd.DataFrame],
    groups: list[str],
    states: list[str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    dominance = composition["dominance"]
    values = dominance.pivot(
        index="LEC_state", columns="age_group", values="top_donor_percent_of_group_state"
    ).reindex(index=states, columns=groups)
    donors = dominance.pivot(
        index="LEC_state", columns="age_group", values="top_donor_key"
    ).reindex(index=states, columns=groups)
    overall = composition["overall_dominance"].set_index("age_group").reindex(groups)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True)
    image = axes[0].imshow(values.to_numpy(float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    axes[0].set_xticks(range(len(groups)), groups)
    axes[0].set_yticks(range(len(states)), states)
    axes[0].set_title("Largest donor share within group-state pool")
    for row, state in enumerate(states):
        for column, group in enumerate(groups):
            value = values.loc[state, group]
            label = "NA" if not np.isfinite(value) else f"{value:.1f}%\n{compact_donor(donors.loc[state, group])}"
            axes[0].text(column, row, label, ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes[0], label="Largest donor contribution (%)", fraction=0.046)
    bars = axes[1].bar(
        groups,
        overall["top_donor_percent_of_group_LECs"],
        color=[COLORS[group] for group in groups],
    )
    for bar, group in zip(bars, groups):
        value = float(overall.loc[group, "top_donor_percent_of_group_LECs"])
        donor = compact_donor(overall.loc[group, "top_donor_key"])
        axes[1].text(bar.get_x() + bar.get_width()/2, value + 1, f"{value:.1f}%\n{donor}", ha="center", va="bottom", fontsize=8)
    axes[1].axhline(50, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, max(105, float(overall["top_donor_percent_of_group_LECs"].max()) + 15))
    axes[1].set_ylabel("Largest donor share of group LECs (%)")
    axes[1].set_title("Overall donor dominance")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_statistical_summary(
    omnibus: pd.DataFrame,
    states: list[str],
    state_colors: dict[str, str],
    path: Path,
    dpi: int,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    frame = omnibus.set_index("LEC_state").reindex(states)
    finite = pd.to_numeric(frame["kruskal_p"], errors="coerce").notna()
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    if finite.any():
        q = pd.to_numeric(frame["kruskal_fdr_bh"], errors="coerce").clip(lower=np.finfo(float).tiny)
        score = -np.log10(q)
        ax.bar(states, score, color=[state_colors[state] for state in states])
        ax.axhline(-math.log10(0.05), color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("-log10(BH FDR)")
        ax.tick_params(axis="x", rotation=55)
    else:
        ax.axis("off")
        ax.text(
            0.5, 0.58,
            "Four-group omnibus tests are not valid:\nQ2 contains only one retained donor.",
            ha="center", va="center", fontsize=16,
        )
        ax.text(
            0.5, 0.36,
            "Descriptive donor points are retained; no pooled-cell substitute was used.",
            ha="center", va="center", fontsize=11,
        )
    ax.set_title(title)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_cell_level_state_panels(
    cell_results: dict[str, pd.DataFrame],
    groups: list[str],
    states: list[str],
    state_colors: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    counts = cell_results["group_state_counts"]
    omnibus = cell_results["omnibus"].set_index("LEC_state")
    n_columns = 3
    n_rows = math.ceil(len(states) / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 3.9 * n_rows))
    flat = np.atleast_1d(axes).ravel()
    for index, state in enumerate(states):
        ax = flat[index]
        frame = counts.loc[counts["LEC_state"].eq(state)].set_index("age_group").reindex(groups)
        bars = ax.bar(
            groups,
            frame["pooled_state_percentage"],
            color=[COLORS[group] for group in groups],
        )
        panel_top = max(5.0, float(frame["pooled_state_percentage"].max()) * 1.23 + 0.5)
        ax.set_ylim(0, panel_top)
        for bar, n_cells in zip(bars, frame["n_state_cells"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + panel_top * 0.018,
                f"n={int(n_cells)}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        row = omnibus.loc[state]
        ax.set_title(
            f"{state}: chi-square p={format_p(row['chi_square_p'])}, "
            f"FDR={format_p(row['chi_square_fdr_bh'])}",
            fontsize=10,
        )
        ax.set_ylabel("Pooled cells in state (%)" if index % n_columns == 0 else "")
        ax.grid(axis="y", alpha=0.2)
    for ax in flat[len(states):]:
        ax.set_visible(False)
    fig.suptitle(
        "Cell-level pooled-state tests (cells are nested within donors; not biological replicates)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_cell_level_summary(
    cell_results: dict[str, pd.DataFrame],
    groups: list[str],
    states: list[str],
    state_colors: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    counts = cell_results["group_state_counts"].pivot(
        index="age_group", columns="LEC_state", values="pooled_state_percentage"
    ).reindex(index=groups, columns=states, fill_value=0)
    omnibus = cell_results["omnibus"].set_index("LEC_state").reindex(states)
    global_row = cell_results["global_test"].iloc[0]
    colors = [state_colors[state] for state in states]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
    counts.plot(
        kind="bar", stacked=True, color=colors, width=0.72, ax=axes[0], legend=False
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("State percentage of pooled group cells")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].set_title(
        "Global cell-level composition\n"
        f"chi-square p={format_p(global_row['p_value'])}, V={global_row['cramers_v']:.3f}"
    )
    q = pd.to_numeric(omnibus["chi_square_fdr_bh"], errors="coerce").clip(
        lower=np.finfo(float).tiny
    )
    axes[1].bar(states, -np.log10(q), color=colors)
    axes[1].axhline(-math.log10(0.05), color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("-log10(BH FDR)")
    axes[1].set_title("State-versus-other-cells association")
    axes[1].tick_params(axis="x", rotation=55)
    fig.suptitle(
        "Cell-level sensitivity only: individual cells are not independent donors",
        fontsize=14,
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        title="LEC state", bbox_to_anchor=(1.01, 0.98), loc="upper left",
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_scheme_outputs(
    output_dir: Path,
    prefix: str,
    donor_meta: pd.DataFrame,
    donor_state: pd.DataFrame,
    overview: pd.DataFrame,
    statistics: dict[str, pd.DataFrame],
    composition: dict[str, pd.DataFrame],
) -> None:
    donor_meta.to_csv(output_dir / f"{prefix}_donor_metadata.csv", index=False)
    donor_state.to_csv(
        output_dir / f"{prefix}_donor_state_counts_percentages.csv.gz",
        index=False,
        compression="gzip",
    )
    overview.to_csv(output_dir / f"{prefix}_group_summary.csv", index=False)
    filenames = {
        "group_state_summary": "state_group_summary_mean_sem.csv",
        "omnibus": "state_omnibus_tests.csv",
        "posthoc": "state_posthoc_Dunn_Holm.csv",
        "continuous": "continuous_age_spearman.csv",
        "leaveout": "leave_top_donor_out_sensitivity.csv",
        "adjusted": "project_sex_adjusted_sensitivity.csv",
    }
    for key, filename in filenames.items():
        statistics[key].to_csv(output_dir / f"{prefix}_{filename}", index=False)
    composition_names = {
        "pooled": "pooled_state_composition.csv",
        "normalized": "donor_normalized_state_composition.csv",
        "dominance": "highest_donor_contribution_by_group_state.csv",
        "overall_dominance": "highest_donor_contribution_by_group.csv",
    }
    for key, filename in composition_names.items():
        composition[key].to_csv(output_dir / f"{prefix}_{filename}", index=False)


def write_cell_level_outputs(
    output_dir: Path,
    prefix: str,
    overview: pd.DataFrame,
    cell_results: dict[str, pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overview.to_csv(output_dir / f"{prefix}_cell_level_group_summary.csv", index=False)
    cell_results["group_state_counts"].to_csv(
        output_dir / f"{prefix}_cell_level_group_state_counts_percentages.csv",
        index=False,
    )
    cell_results["global_test"].to_csv(
        output_dir / f"{prefix}_cell_level_global_composition_chi_square.csv",
        index=False,
    )
    cell_results["omnibus"].to_csv(
        output_dir / f"{prefix}_cell_level_state_omnibus_chi_square.csv",
        index=False,
    )
    cell_results["posthoc"].to_csv(
        output_dir / f"{prefix}_cell_level_state_posthoc_Fisher_Holm.csv",
        index=False,
    )


def write_notes(
    path: Path,
    input_h5ad: Path,
    minimum: int,
    four_overview: pd.DataFrame,
    merged_overview: pd.DataFrame,
    four_stats: dict[str, pd.DataFrame],
    merged_stats: dict[str, pd.DataFrame],
    four_cell_stats: dict[str, pd.DataFrame],
    merged_cell_stats: dict[str, pd.DataFrame],
) -> None:
    merged_significant = merged_stats["omnibus"].loc[
        merged_stats["omnibus"]["kruskal_fdr_bh_significant"], "LEC_state"
    ].astype(str).tolist()
    continuous_significant = merged_stats["continuous"].loc[
        merged_stats["continuous"]["spearman_fdr_bh_significant"], "LEC_state"
    ].astype(str).tolist()
    four_cell_significant = four_cell_stats["omnibus"].loc[
        four_cell_stats["omnibus"]["chi_square_fdr_bh_significant"], "LEC_state"
    ].astype(str).tolist()
    merged_cell_significant = merged_cell_stats["omnibus"].loc[
        merged_cell_stats["omnibus"]["chi_square_fdr_bh_significant"], "LEC_state"
    ].astype(str).tolist()
    text = f"""Step 08.7 minimum-{minimum}-LEC donor-level analysis
==================================================

Input (read only): {input_h5ad}
Source data or labels changed: no
Retained donor entries: 13
Retained LEC cells: 818

Filter
------
A donor entry was retained when it had at least {minimum} total LEC cells. The
filter was learned as an exploratory reliability threshold in Step 08.6. It is
not a minimum donor-state cell threshold and does not validate rare-state
pseudobulk, GSEA, or SCENIC analyses.

Age labels
----------
The existing Step 08.3 age_quartile_equal_LEC_donors label was preserved. Age
ranges were not recalculated after filtering. Tied boundary ages therefore
remain split across Q1/Q2 at 48 years and Q3/Q4 at 62.5 years, as in Step 08.3.

Four preserved groups:
{four_overview.to_csv(index=False).strip()}

The four-group primary Kruskal-Wallis analysis is not testable under the
pre-specified safeguard requiring at least two donor entries in every group,
because Q2 contains one donor. No pooled-cell test was substituted.

Q2+Q3 combined groups:
{merged_overview.to_csv(index=False).strip()}

Statistical analysis
--------------------
Each donor is one biological-replicate entry. All zero donor-state combinations
are retained. The primary merged-group test is Kruskal-Wallis on donor-level
state percentages across Q1, Q2+Q3, and Q4. BH FDR is applied across 11 states.
Dunn-Holm tests are generated only for states passing omnibus FDR. Welch ANOVA,
continuous-age Spearman, leave-top-donor-out, and Project/Sex-adjusted HC3 OLS
are separate sensitivity analyses.

Merged-group primary significant states: {merged_significant or 'none'}
Continuous-age FDR-significant states: {continuous_significant or 'none'}

Separate cell-level sensitivity
-------------------------------
The cell-level result is stored separately from donor-level inference. For each
state, a 2 x K Pearson chi-square test compares state-versus-other-cell counts
across age groups, with BH correction across 11 states. A global K x 11
composition chi-square test is also reported. Fisher exact pairwise tests with
Holm correction are generated only for cell-level states passing omnibus FDR.

Four-group cell-level FDR-significant states: {four_cell_significant or 'none'}
Merged-group cell-level FDR-significant states: {merged_cell_significant or 'none'}

These cell-level p-values assume independent cells, although cells are nested
within donors. They can be much smaller than donor-level p-values and must not
be used as the main biological-replicate inference. The 818 cells do not replace
the 13 donor entries as the experimental sample size.

Interpretation
--------------
The merged analysis has only 4/4/5 donor entries. Percentage estimates are
bounded, zero-inflated, and compositional, and adjusted models have very limited
degrees of freedom. Treat all results as exploratory and compare them with the
unfiltered analyses. The 2,000 Step 08.6 resampling runs are not included as
biological observations in this analysis.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.minimum_LECs != 25:
        warnings.warn(
            "Step 08.7 was designed for the requested 25-LEC threshold; "
            f"running with {args.minimum_LECs} changes the expected invariants."
        )
    if args.minimum_LECs < 1:
        raise ValueError("--minimum_LECs must be positive.")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1.")

    project_dir = find_project_dir()
    input_h5ad = resolve_path(args.input_h5ad, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    if not input_h5ad.is_file():
        raise FileNotFoundError(f"Input h5ad does not exist: {input_h5ad}")
    sentinel = output_root / "08_7_run_summary.csv"
    if sentinel.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"08.7 output exists: {sentinel}. Use --overwrite.")

    step08_4 = load_step08_4(project_dir)
    print(f"Reading Step 08.3 input: {input_h5ad}")
    adata = ad.read_h5ad(input_h5ad)
    full_meta, filtered_meta, filtered_state, states, full_ranges = prepare_filtered_tables(
        adata, args, step08_4
    )
    four_meta, four_state, four_groups, four_overview, four_labels = make_scheme_tables(
        filtered_meta, filtered_state, full_ranges, merged=False
    )
    merged_meta, merged_state, merged_groups, merged_overview, merged_labels = make_scheme_tables(
        filtered_meta, filtered_state, full_ranges, merged=True
    )
    print("Preserved four-group donor counts:", dict(zip(four_overview["age_group"], four_overview["n_donor_entries"])))
    print("Q2+Q3 merged donor counts:", dict(zip(merged_overview["age_group"], merged_overview["n_donor_entries"])))
    print(f"Retained donor entries: {len(filtered_meta)}; retained LECs: {filtered_meta['n_total_LEC_cells'].sum()}")
    if args.dry_run:
        print("Dry run complete; no files written.")
        return 0

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    four_dir = output_root / "four_preserved_quartiles"
    merged_dir = output_root / "three_groups_Q2_Q3_merged"
    four_donor_dir = four_dir / "donor_level"
    four_cell_dir = four_dir / "cell_level"
    merged_donor_dir = merged_dir / "donor_level"
    merged_cell_dir = merged_dir / "cell_level"
    for directory in (four_donor_dir, four_cell_dir, merged_donor_dir, merged_cell_dir):
        directory.mkdir(parents=True, exist_ok=True)

    four_stats = run_statistics(step08_4, four_state, states, four_groups, args.alpha)
    merged_stats = run_statistics(step08_4, merged_state, states, merged_groups, args.alpha)
    four_cell_stats = run_cell_level_statistics(
        four_state, four_groups, states, args.alpha
    )
    merged_cell_stats = run_cell_level_statistics(
        merged_state, merged_groups, states, args.alpha
    )
    four_composition = make_composition_tables(four_meta, four_state, four_groups, states)
    merged_composition = make_composition_tables(merged_meta, merged_state, merged_groups, states)
    write_scheme_outputs(
        four_donor_dir, "08_7_four_groups", four_meta, four_state, four_overview,
        four_stats, four_composition,
    )
    write_scheme_outputs(
        merged_donor_dir, "08_7_Q2_Q3_merged", merged_meta, merged_state, merged_overview,
        merged_stats, merged_composition,
    )
    write_cell_level_outputs(
        four_cell_dir, "08_7_four_groups", four_overview, four_cell_stats
    )
    write_cell_level_outputs(
        merged_cell_dir, "08_7_Q2_Q3_merged", merged_overview, merged_cell_stats
    )
    full_meta.to_csv(output_root / "08_7_all_donor_filter_mapping.csv", index=False)
    filtered_meta.to_csv(output_root / "08_7_retained_donors_min25.csv", index=False)
    pd.DataFrame(
        [
            {
                "grouping": "four_preserved_quartiles",
                "analysis_level": "donor_level",
                "canonical_folder": str(four_donor_dir),
                "inference_unit": "donor entry",
            },
            {
                "grouping": "four_preserved_quartiles",
                "analysis_level": "cell_level",
                "canonical_folder": str(four_cell_dir),
                "inference_unit": "cell (nested within donor; sensitivity only)",
            },
            {
                "grouping": "three_groups_Q2_Q3_merged",
                "analysis_level": "donor_level",
                "canonical_folder": str(merged_donor_dir),
                "inference_unit": "donor entry",
            },
            {
                "grouping": "three_groups_Q2_Q3_merged",
                "analysis_level": "cell_level",
                "canonical_folder": str(merged_cell_dir),
                "inference_unit": "cell (nested within donor; sensitivity only)",
            },
        ]
    ).to_csv(output_root / "08_7_result_folder_map.csv", index=False)

    state_colors = step08_4.state_color_map(adata, args.state_key, states)
    manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        for directory, prefix, meta, state, groups, overview, labels, stats_result, composition in (
            (four_donor_dir, "08_7_four_groups", four_meta, four_state, four_groups, four_overview, four_labels, four_stats, four_composition),
            (merged_donor_dir, "08_7_Q2_Q3_merged", merged_meta, merged_state, merged_groups, merged_overview, merged_labels, merged_stats, merged_composition),
        ):
            figures = directory / "figures"
            per_state = figures / "per_state"
            figures.mkdir(parents=True, exist_ok=True)
            manifest.extend(
                save_all_state_points(
                    state,
                    stats_result["group_state_summary"],
                    stats_result["omnibus"],
                    groups,
                    states,
                    labels,
                    figures / f"{prefix}_all_states_donor_percentages.png",
                    args.dpi,
                    per_state,
                )
            )
            figure_jobs = [
                (
                    figures / f"{prefix}_group_support.png",
                    "Retained donor and cell support",
                    lambda path, m=meta, o=overview, g=groups: save_group_support(m, o, g, path, args.dpi),
                ),
                (
                    figures / f"{prefix}_composition.png",
                    "Pooled, donor-normalized, and donor-presence composition",
                    lambda path, c=composition, g=groups: save_composition(c, g, states, state_colors, path, args.dpi),
                ),
                (
                    figures / f"{prefix}_highest_donor_contribution.png",
                    "Largest donor share by group-state and group overall",
                    lambda path, c=composition, g=groups: save_dominance(c, g, states, path, args.dpi),
                ),
                (
                    figures / f"{prefix}_statistical_summary.png",
                    "Omnibus statistical summary or explicit non-testable notice",
                    lambda path, s=stats_result, p=prefix: save_statistical_summary(s["omnibus"], states, state_colors, path, args.dpi, p.replace("_", " ")),
                ),
            ]
            for path, description, function in figure_jobs:
                function(path)
                manifest.append({"figure": str(path), "description": description, "status": "ok"})

        for directory, prefix, groups, cell_results in (
            (four_cell_dir, "08_7_four_groups", four_groups, four_cell_stats),
            (merged_cell_dir, "08_7_Q2_Q3_merged", merged_groups, merged_cell_stats),
        ):
            figures = directory / "figures"
            figures.mkdir(parents=True, exist_ok=True)
            state_path = figures / f"{prefix}_cell_level_state_percentages.png"
            save_cell_level_state_panels(
                cell_results, groups, states, state_colors, state_path, args.dpi
            )
            manifest.append(
                {
                    "figure": str(state_path),
                    "description": "Cell-level pooled state percentages and chi-square tests",
                    "status": "ok",
                }
            )
            summary_path = figures / f"{prefix}_cell_level_composition_and_FDR.png"
            save_cell_level_summary(
                cell_results, groups, states, state_colors, summary_path, args.dpi
            )
            manifest.append(
                {
                    "figure": str(summary_path),
                    "description": "Cell-level global composition and state FDR summary",
                    "status": "ok",
                }
            )

    pd.DataFrame(manifest, columns=["figure", "description", "status"]).to_csv(
        output_root / "08_7_figure_manifest.csv", index=False
    )
    write_notes(
        output_root / "08_7_method_and_interpretation_notes.txt",
        input_h5ad,
        args.minimum_LECs,
        four_overview,
        merged_overview,
        four_stats,
        merged_stats,
        four_cell_stats,
        merged_cell_stats,
    )
    config = {
        "script_version": SCRIPT_VERSION,
        "input_h5ad": str(input_h5ad),
        "output_root": str(output_root),
        "minimum_total_LECs_per_donor": args.minimum_LECs,
        "preserved_group_key": args.group_key,
        "four_group_counts": dict(zip(four_overview["age_group"], four_overview["n_donor_entries"].astype(int))),
        "merged_group_counts": dict(zip(merged_overview["age_group"], merged_overview["n_donor_entries"].astype(int))),
        "n_retained_donor_entries": int(len(filtered_meta)),
        "n_retained_LEC_cells": int(filtered_meta["n_total_LEC_cells"].sum()),
        "n_states": len(states),
        "primary_unit": "donor entry",
        "primary_test": "Kruskal-Wallis donor-level state percentage with BH FDR across states",
        "cell_level_test": "Pearson chi-square state-versus-other-cells with BH FDR across states",
        "cell_level_independence_warning": "cells nested within donors; sensitivity only",
        "source_modified": False,
    }
    (output_root / "08_7_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    run_summary = pd.DataFrame(
        [
            {
                "minimum_LECs": args.minimum_LECs,
                "n_retained_donor_entries": len(filtered_meta),
                "n_retained_LEC_cells": int(filtered_meta["n_total_LEC_cells"].sum()),
                "four_group_counts": "Q1:4;Q2:1;Q3:3;Q4:5",
                "four_group_primary_test_status": "not_testable_Q2_n_1",
                "merged_group_counts": "Q1:4;Q2+Q3:4;Q4:5",
                "merged_significant_states_FDR": ";".join(
                    merged_stats["omnibus"].loc[
                        merged_stats["omnibus"]["kruskal_fdr_bh_significant"], "LEC_state"
                    ].astype(str)
                ) or "none",
                "four_group_cell_level_significant_states_FDR": ";".join(
                    four_cell_stats["omnibus"].loc[
                        four_cell_stats["omnibus"]["chi_square_fdr_bh_significant"], "LEC_state"
                    ].astype(str)
                ) or "none",
                "merged_group_cell_level_significant_states_FDR": ";".join(
                    merged_cell_stats["omnibus"].loc[
                        merged_cell_stats["omnibus"]["chi_square_fdr_bh_significant"], "LEC_state"
                    ].astype(str)
                ) or "none",
                "source_modified": False,
            }
        ]
    )
    run_summary.to_csv(output_root / "08_7_run_summary.csv", index=False)

    print("\nMerged Q2+Q3 primary results:")
    print(
        merged_stats["omnibus"][["LEC_state", "kruskal_p", "kruskal_fdr_bh", "epsilon_squared", "test_status"]]
        .to_string(index=False, float_format=lambda value: f"{value:.6g}")
    )
    print(f"\nOutputs: {output_root}")
    print("Source h5ad and all previous results were not modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
