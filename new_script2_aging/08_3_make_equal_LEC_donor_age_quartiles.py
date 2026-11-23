#!/usr/bin/env python3
r"""
Step 08.3: add new age quartiles containing exactly 12 LEC-positive donors each.

This is a metadata-only sensitivity analysis built from the primary Step 08
resolution-0.3 LEC object. It does not remove cells, alter expression, rerun
HVG selection/PCA/Harmony/neighbors/UMAP/Leiden, rename LEC states, or overwrite
the original ``age_quartile`` column.

The 48 donors represented by at least one Step 08 LEC are ordered by resolved
age and assigned to four groups of 12. When an identical age crosses a 12-donor
boundary, donors at that age must be split between adjacent groups. Among all
valid age-ordered assignments, the script first minimizes changes relative to
the original quartile label, then minimizes imbalance in total LEC-cell counts,
and finally uses donor_key as a deterministic tie breaker.

Default input
-------------
new_result2_aging/08_combined_LEC_recluster/dataset13/
    08_combined_final_LEC_reclustered.h5ad

Default output
--------------
new_result2_aging/08_3_equal_LEC_donor_age_quartiles/dataset13/

Main outputs
------------
08_3_combined_LEC_equal_donor_age_quartiles.h5ad
08_3_donor_quartile_mapping.csv
08_3_quartile_summary.csv
08_3_original_to_new_quartile_transition.csv
08_3_state_composition_by_new_quartile.csv
08_3_donor_normalized_state_composition.csv
08_3_state_donor_presence_by_new_quartile.csv
08_3_highest_donor_contribution_by_quartile_state.csv
08_3_highest_donor_contribution_by_quartile.csv
08_3_method_and_interpretation_notes.txt
figures/*.png

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_aging\08_3_make_equal_LEC_donor_age_quartiles.py

Validate and print the proposed mapping without writing files:

    .\.venv\Scripts\python.exe `
      .\new_script2_aging\08_3_make_equal_LEC_donor_age_quartiles.py --dry_run
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg", force=True)


SCRIPT_VERSION = (
    "08_3_dataset13_equal_12_LEC_donors_per_age_quartile_"
    "with_donor_dominance_2026_08_09"
)
DEFAULT_INPUT = (
    "new_result2_aging/08_combined_LEC_recluster/dataset13/"
    "08_combined_final_LEC_reclustered.h5ad"
)
DEFAULT_OUTPUT_ROOT = (
    "new_result2_aging/08_3_equal_LEC_donor_age_quartiles/dataset13"
)
DEFAULT_OUT_H5AD = "08_3_combined_LEC_equal_donor_age_quartiles.h5ad"
DEFAULT_ORIGINAL_KEY = "age_quartile"
DEFAULT_NEW_KEY = "age_quartile_equal_LEC_donors"
DEFAULT_STATE_KEY = "LEC_state"
DEFAULT_DONOR_KEY = "donor_key"
DEFAULT_AGE_KEY = "age_numeric_resolved"
DEFAULT_PROJECT_KEY = "Project"
DEFAULT_SEX_KEY = "Sex"
N_QUARTILES = 4
EXPECTED_DONORS = 48
DONORS_PER_QUARTILE = 12
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
QUARTILE_COLORS = {
    "Q1": "#1f77b4",
    "Q2": "#ff7f0e",
    "Q3": "#2ca02c",
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
            "Add a separate age-quartile label with exactly 12 Step 08 "
            "LEC-positive donors per group; do not recluster or overwrite the "
            "original quartile label."
        )
    )
    parser.add_argument("--input_h5ad", default=DEFAULT_INPUT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--out_h5ad", default=DEFAULT_OUT_H5AD)
    parser.add_argument("--original_quartile_key", default=DEFAULT_ORIGINAL_KEY)
    parser.add_argument("--new_quartile_key", default=DEFAULT_NEW_KEY)
    parser.add_argument("--state_key", default=DEFAULT_STATE_KEY)
    parser.add_argument("--donor_key", default=DEFAULT_DONOR_KEY)
    parser.add_argument("--age_key", default=DEFAULT_AGE_KEY)
    parser.add_argument("--project_key", default=DEFAULT_PROJECT_KEY)
    parser.add_argument("--sex_key", default=DEFAULT_SEX_KEY)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the donor mapping and summaries without creating outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing Step 08.3 primary h5ad.",
    )
    return parser.parse_args()


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


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


def one_value(group: pd.DataFrame, column: str, donor: str) -> object:
    values = group[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"Donor {donor!r} has {len(values)} distinct values for {column!r}: "
            f"{values[:8].tolist()}"
        )
    return values[0]


def donor_table_from_obs(
    obs: pd.DataFrame,
    donor_key: str,
    age_key: str,
    original_key: str,
    state_key: str,
    project_key: str,
    sex_key: str,
) -> pd.DataFrame:
    required = {donor_key, age_key, original_key, state_key}
    missing = sorted(required.difference(obs.columns))
    if missing:
        raise KeyError(f"Input h5ad is missing required obs columns: {missing}")

    working = obs.copy()
    working[donor_key] = normalized_text(working[donor_key])
    working[original_key] = normalized_text(working[original_key])
    working[state_key] = normalized_text(working[state_key])
    if working[donor_key].eq("").any():
        raise ValueError("Blank donor_key values are not allowed.")
    working[age_key] = pd.to_numeric(working[age_key], errors="coerce")
    if working[age_key].isna().any():
        raise ValueError("Every Step 08 LEC must have a resolved numeric age.")

    rows: list[dict[str, object]] = []
    for donor, group in working.groupby(donor_key, sort=False, observed=True):
        donor_text = str(donor)
        row: dict[str, object] = {
            "donor_key": donor_text,
            "age_numeric_resolved": float(one_value(group, age_key, donor_text)),
            "original_age_quartile": str(
                one_value(group, original_key, donor_text)
            ),
            "n_LEC_cells": int(len(group)),
            "n_LEC_states": int(group[state_key].nunique()),
        }
        row["Project"] = (
            str(one_value(group, project_key, donor_text))
            if project_key in group.columns
            else "unknown"
        )
        row["Sex"] = (
            str(one_value(group, sex_key, donor_text))
            if sex_key in group.columns
            else "unknown"
        )
        rows.append(row)
    donors = pd.DataFrame(rows)
    if len(donors) != EXPECTED_DONORS:
        raise ValueError(
            f"Step 08.3 requires exactly {EXPECTED_DONORS} LEC-positive donors "
            f"to make four groups of {DONORS_PER_QUARTILE}; found {len(donors)}."
        )
    if set(donors["original_age_quartile"]) != set(QUARTILES):
        raise ValueError(
            "Original age quartiles must be Q1-Q4; found "
            f"{sorted(donors['original_age_quartile'].unique(), key=natural_key)}"
        )
    return donors


def allocations_for_split_tie(
    donor_keys: list[str], group_counts: dict[str, int]
) -> list[dict[str, str]]:
    """Enumerate unique donor-to-group allocations for one split-age tie."""
    groups = [group for group in QUARTILES if group_counts.get(group, 0) > 0]
    results: list[dict[str, str]] = []

    def recurse(
        remaining: tuple[str, ...], group_index: int, assigned: dict[str, str]
    ) -> None:
        group = groups[group_index]
        count = int(group_counts[group])
        if group_index == len(groups) - 1:
            if len(remaining) != count:
                raise RuntimeError("Internal split-tie allocation error.")
            final = dict(assigned)
            final.update({donor: group for donor in remaining})
            results.append(final)
            return
        for chosen in itertools.combinations(remaining, count):
            chosen_set = set(chosen)
            next_assigned = dict(assigned)
            next_assigned.update({donor: group for donor in chosen})
            next_remaining = tuple(
                donor for donor in remaining if donor not in chosen_set
            )
            recurse(next_remaining, group_index + 1, next_assigned)

    recurse(tuple(sorted(donor_keys, key=natural_key)), 0, {})
    return results


def assign_exact_quartiles(donors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the least-disruptive exact-12 age-ordered donor assignment."""
    ordered = donors.sort_values(
        ["age_numeric_resolved", "donor_key"], kind="mergesort"
    ).reset_index(drop=True)
    possibilities: list[list[dict[str, str]]] = []
    for _, age_group in ordered.groupby(
        "age_numeric_resolved", sort=True, observed=True
    ):
        positions = age_group.index.to_numpy(dtype=int)
        position_groups = [QUARTILES[position // DONORS_PER_QUARTILE] for position in positions]
        counts = pd.Series(position_groups).value_counts().to_dict()
        donor_keys = age_group["donor_key"].astype(str).tolist()
        if len(counts) == 1:
            group = next(iter(counts))
            possibilities.append([{donor: group for donor in donor_keys}])
        else:
            possibilities.append(allocations_for_split_tie(donor_keys, counts))

    target_cells = float(donors["n_LEC_cells"].sum() / N_QUARTILES)
    donor_index = donors.set_index("donor_key")
    best_mapping: dict[str, str] | None = None
    best_objective: tuple[object, ...] | None = None
    for parts in itertools.product(*possibilities):
        mapping: dict[str, str] = {}
        for part in parts:
            mapping.update(part)
        assigned = donors["donor_key"].map(mapping)
        changed = int(
            assigned.ne(donors["original_age_quartile"]).sum()
        )
        cell_totals = (
            donors.assign(_new=assigned)
            .groupby("_new", observed=True)["n_LEC_cells"]
            .sum()
            .reindex(QUARTILES, fill_value=0)
            .astype(float)
        )
        cell_imbalance = float(np.square(cell_totals - target_cells).sum())
        lexical = tuple(
            mapping[donor] for donor in sorted(mapping, key=natural_key)
        )
        objective: tuple[object, ...] = (changed, cell_imbalance, lexical)
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_mapping = mapping

    if best_mapping is None or best_objective is None:
        raise RuntimeError("No exact age-ordered quartile assignment was found.")

    result = donors.copy()
    result["new_age_quartile"] = result["donor_key"].map(best_mapping)
    group_counts = result["new_age_quartile"].value_counts().reindex(QUARTILES)
    if not group_counts.eq(DONORS_PER_QUARTILE).all():
        raise RuntimeError(f"Exact donor-count invariant failed: {group_counts.to_dict()}")

    result["quartile_changed"] = result["new_age_quartile"].ne(
        result["original_age_quartile"]
    )
    split_ages = (
        result.groupby("age_numeric_resolved", observed=True)["new_age_quartile"]
        .nunique()
        .loc[lambda x: x.gt(1)]
        .index.astype(float)
        .tolist()
    )
    result["age_tie_split_across_new_quartiles"] = result[
        "age_numeric_resolved"
    ].isin(split_ages)
    result["new_quartile_order"] = result["new_age_quartile"].map(
        {group: index for index, group in enumerate(QUARTILES)}
    )
    result = result.sort_values(
        ["age_numeric_resolved", "new_quartile_order", "donor_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    result.insert(0, "age_order_rank", np.arange(1, len(result) + 1))
    result.drop(columns="new_quartile_order", inplace=True)

    split_rows: list[dict[str, object]] = []
    for age in split_ages:
        group = result.loc[result["age_numeric_resolved"].eq(age)]
        counts = (
            group["new_age_quartile"].value_counts().reindex(QUARTILES, fill_value=0)
        )
        split_rows.append(
            {
                "split_age": age,
                "n_donors_at_age": int(len(group)),
                "new_quartiles": ";".join(
                    quartile for quartile in QUARTILES if counts[quartile] > 0
                ),
                "donors_per_new_quartile": ";".join(
                    f"{quartile}:{int(counts[quartile])}"
                    for quartile in QUARTILES
                    if counts[quartile] > 0
                ),
            }
        )
    split_table = pd.DataFrame(split_rows)
    result.attrs["selection_objective"] = {
        "n_original_quartile_changes": int(best_objective[0]),
        "LEC_cell_imbalance_sum_squared": float(best_objective[1]),
    }
    return result, split_table


def make_range_labels(mapping: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    for quartile in QUARTILES:
        group = mapping.loc[mapping["new_age_quartile"].eq(quartile)]
        age_min = float(group["age_numeric_resolved"].min())
        age_max = float(group["age_numeric_resolved"].max())
        labels[quartile] = (
            f"{quartile} equal-LEC-donor ({age_min:g}-{age_max:g} years; "
            f"n={len(group)} donors)"
        )
    return labels


def make_summary_tables(
    obs: pd.DataFrame,
    mapping: pd.DataFrame,
    new_key: str,
    state_key: str,
    donor_key: str,
) -> dict[str, pd.DataFrame]:
    range_labels = make_range_labels(mapping)
    summary_rows: list[dict[str, object]] = []
    for quartile in QUARTILES:
        group = mapping.loc[mapping["new_age_quartile"].eq(quartile)]
        summary_rows.append(
            {
                "new_age_quartile": quartile,
                "new_age_quartile_range_label": range_labels[quartile],
                "age_min": float(group["age_numeric_resolved"].min()),
                "age_max": float(group["age_numeric_resolved"].max()),
                "age_median": float(group["age_numeric_resolved"].median()),
                "n_donors": int(len(group)),
                "n_LEC_cells": int(group["n_LEC_cells"].sum()),
                "median_LEC_cells_per_donor": float(group["n_LEC_cells"].median()),
                "min_LEC_cells_per_donor": int(group["n_LEC_cells"].min()),
                "max_LEC_cells_per_donor": int(group["n_LEC_cells"].max()),
                "n_projects": int(group["Project"].nunique()),
                "n_female": int(group["Sex"].eq("F").sum()),
                "n_male": int(group["Sex"].eq("M").sum()),
                "n_original_quartile_matches": int(
                    group["new_age_quartile"].eq(
                        group["original_age_quartile"]
                    ).sum()
                ),
                "n_original_quartile_changes": int(group["quartile_changed"].sum()),
            }
        )
    quartile_summary = pd.DataFrame(summary_rows)

    transition = pd.crosstab(
        mapping["original_age_quartile"], mapping["new_age_quartile"]
    ).reindex(index=QUARTILES, columns=QUARTILES, fill_value=0)
    transition.index.name = "original_age_quartile"
    transition = transition.reset_index()

    project_counts = (
        mapping.groupby(["new_age_quartile", "Project"], observed=True)
        .size()
        .rename("n_donors")
        .reset_index()
    )
    sex_counts = (
        mapping.groupby(["new_age_quartile", "Sex"], observed=True)
        .size()
        .rename("n_donors")
        .reset_index()
    )

    working = obs.copy()
    working["donor_key"] = normalized_text(working[donor_key])
    working[new_key] = normalized_text(working[new_key])
    working[state_key] = normalized_text(working[state_key])
    state_order = sorted(working[state_key].unique(), key=natural_key)
    cell_counts = (
        working.groupby([new_key, state_key], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    cell_totals = cell_counts.groupby(new_key, observed=True)["n_cells"].transform("sum")
    cell_counts["fraction_within_new_quartile"] = np.where(
        cell_totals.gt(0), cell_counts["n_cells"] / cell_totals, np.nan
    )
    cell_counts["percent_within_new_quartile"] = (
        100.0 * cell_counts["fraction_within_new_quartile"]
    )

    donor_state = (
        working.groupby(["donor_key", new_key, state_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    donor_totals = (
        working.groupby("donor_key", observed=True)
        .size()
        .rename("n_donor_LEC_cells")
        .reset_index()
    )
    donor_state = donor_state.merge(donor_totals, on="donor_key", how="left")
    donor_state["fraction_of_donor_LECs"] = (
        donor_state["n_cells"] / donor_state["n_donor_LEC_cells"]
    )
    full_index = pd.MultiIndex.from_product(
        [mapping["donor_key"].astype(str), state_order],
        names=["donor_key", state_key],
    )
    donor_state_full = (
        donor_state.set_index(["donor_key", state_key])
        .reindex(full_index)
        .reset_index()
    )
    donor_group_map = mapping.set_index("donor_key")["new_age_quartile"]
    donor_total_map = mapping.set_index("donor_key")["n_LEC_cells"]
    donor_state_full[new_key] = donor_state_full["donor_key"].map(donor_group_map)
    donor_state_full["n_cells"] = donor_state_full["n_cells"].fillna(0).astype(int)
    donor_state_full["n_donor_LEC_cells"] = donor_state_full["donor_key"].map(
        donor_total_map
    )
    donor_state_full["fraction_of_donor_LECs"] = (
        donor_state_full["n_cells"] / donor_state_full["n_donor_LEC_cells"]
    )
    donor_state_summary = (
        donor_state_full.groupby([new_key, state_key], observed=True)
        .agg(
            mean_donor_fraction=("fraction_of_donor_LECs", "mean"),
            median_donor_fraction=("fraction_of_donor_LECs", "median"),
            q25_donor_fraction=("fraction_of_donor_LECs", lambda x: x.quantile(0.25)),
            q75_donor_fraction=("fraction_of_donor_LECs", lambda x: x.quantile(0.75)),
            n_donors=("donor_key", "size"),
            n_donors_with_state=("n_cells", lambda x: int(x.gt(0).sum())),
        )
        .reset_index()
    )
    donor_state_summary["percent_mean_donor_fraction"] = (
        100.0 * donor_state_summary["mean_donor_fraction"]
    )
    donor_presence = donor_state_summary[
        [new_key, state_key, "n_donors", "n_donors_with_state"]
    ].copy()
    donor_presence["fraction_donors_with_state"] = (
        donor_presence["n_donors_with_state"] / donor_presence["n_donors"]
    )
    donor_presence["percent_donors_with_state"] = (
        100.0 * donor_presence["fraction_donors_with_state"]
    )

    # Largest single-donor share of each pooled quartile-state cell count.
    # This directly diagnoses whether a state-composition segment is driven by
    # one highly sampled donor. Only donors with at least one cell in the state
    # are candidates; all state cells in the quartile form the denominator.
    donor_state_dominance = donor_state.copy()
    donor_state_dominance["n_quartile_state_cells"] = (
        donor_state_dominance.groupby([new_key, state_key], observed=True)[
            "n_cells"
        ].transform("sum")
    )
    donor_state_dominance["n_donors_with_state"] = (
        donor_state_dominance.groupby([new_key, state_key], observed=True)[
            "donor_key"
        ].transform("nunique")
    )
    donor_state_dominance["donor_fraction_of_quartile_state"] = (
        donor_state_dominance["n_cells"]
        / donor_state_dominance["n_quartile_state_cells"]
    )
    highest_donor_quartile_state = (
        donor_state_dominance.sort_values(
            [new_key, state_key, "n_cells", "donor_key"],
            ascending=[True, True, False, True],
        )
        .groupby([new_key, state_key], observed=True, sort=False)
        .head(1)
        .rename(
            columns={
                "donor_key": "top_donor_key",
                "n_cells": "top_donor_n_cells",
                "donor_fraction_of_quartile_state": (
                    "top_donor_fraction_of_quartile_state"
                ),
            }
        )[
            [
                new_key,
                state_key,
                "top_donor_key",
                "top_donor_n_cells",
                "n_quartile_state_cells",
                "n_donors_with_state",
                "top_donor_fraction_of_quartile_state",
            ]
        ]
        .copy()
    )
    highest_donor_quartile_state[
        "top_donor_percent_of_quartile_state"
    ] = 100.0 * highest_donor_quartile_state[
        "top_donor_fraction_of_quartile_state"
    ]
    highest_donor_quartile_state["top_donor_over_50_percent"] = (
        highest_donor_quartile_state[
            "top_donor_fraction_of_quartile_state"
        ].gt(0.5)
    )
    state_rank = {state: index for index, state in enumerate(state_order)}
    highest_donor_quartile_state["_quartile_rank"] = (
        highest_donor_quartile_state[new_key].map(
            {quartile: index for index, quartile in enumerate(QUARTILES)}
        )
    )
    highest_donor_quartile_state["_state_rank"] = (
        highest_donor_quartile_state[state_key].map(state_rank)
    )
    highest_donor_quartile_state = (
        highest_donor_quartile_state.sort_values(
            ["_quartile_rank", "_state_rank"]
        )
        .drop(columns=["_quartile_rank", "_state_rank"])
        .reset_index(drop=True)
    )

    # Also show the largest donor share of all LEC cells in each quartile so
    # readers can distinguish overall sampling imbalance from state-specific
    # donor dominance.
    quartile_donor_totals = mapping[
        ["donor_key", "new_age_quartile", "n_LEC_cells"]
    ].rename(
        columns={
            "new_age_quartile": new_key,
            "donor_key": "top_donor_key",
            "n_LEC_cells": "top_donor_n_LEC_cells",
        }
    )
    quartile_donor_totals["n_quartile_LEC_cells"] = (
        quartile_donor_totals.groupby(new_key, observed=True)[
            "top_donor_n_LEC_cells"
        ].transform("sum")
    )
    quartile_donor_totals["top_donor_fraction_of_quartile_LECs"] = (
        quartile_donor_totals["top_donor_n_LEC_cells"]
        / quartile_donor_totals["n_quartile_LEC_cells"]
    )
    highest_donor_quartile_overall = (
        quartile_donor_totals.sort_values(
            [new_key, "top_donor_n_LEC_cells", "top_donor_key"],
            ascending=[True, False, True],
        )
        .groupby(new_key, observed=True, sort=False)
        .head(1)
        .copy()
    )
    highest_donor_quartile_overall[
        "top_donor_percent_of_quartile_LECs"
    ] = 100.0 * highest_donor_quartile_overall[
        "top_donor_fraction_of_quartile_LECs"
    ]
    highest_donor_quartile_overall["top_donor_over_50_percent"] = (
        highest_donor_quartile_overall[
            "top_donor_fraction_of_quartile_LECs"
        ].gt(0.5)
    )
    highest_donor_quartile_overall["_quartile_rank"] = (
        highest_donor_quartile_overall[new_key].map(
            {quartile: index for index, quartile in enumerate(QUARTILES)}
        )
    )
    highest_donor_quartile_overall = (
        highest_donor_quartile_overall.sort_values("_quartile_rank")
        .drop(columns="_quartile_rank")
        .reset_index(drop=True)
    )

    return {
        "quartile_summary": quartile_summary,
        "transition": transition,
        "project_counts": project_counts,
        "sex_counts": sex_counts,
        "state_cell_composition": cell_counts,
        "donor_state_composition": donor_state_summary,
        "donor_state_presence": donor_presence,
        "donor_state_full": donor_state_full,
        "highest_donor_quartile_state": highest_donor_quartile_state,
        "highest_donor_quartile_overall": highest_donor_quartile_overall,
    }


def state_palette(adata: ad.AnnData, state_key: str, states: list[str]) -> dict[str, str]:
    explicit = adata.uns.get(f"{state_key}_color_map", {})
    if isinstance(explicit, dict) and all(state in explicit for state in states):
        return {state: str(explicit[state]) for state in states}
    colors = list(adata.uns.get(f"{state_key}_colors", []))
    if len(colors) >= len(states):
        return {state: str(colors[index]) for index, state in enumerate(states)}
    base = list(matplotlib.colormaps["tab20"].colors)
    return {
        state: matplotlib.colors.to_hex(base[index % len(base)])
        for index, state in enumerate(states)
    }


def save_donor_distribution(
    mapping: pd.DataFrame, transition: pd.DataFrame, path: Path, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
    group_position = {group: index for index, group in enumerate(QUARTILES)}
    for group in QUARTILES:
        frame = mapping.loc[mapping["new_age_quartile"].eq(group)].copy()
        frame = frame.sort_values(["age_numeric_resolved", "donor_key"])
        offsets = np.linspace(-0.18, 0.18, len(frame)) if len(frame) > 1 else np.array([0.0])
        y = group_position[group] + offsets
        changed = frame["quartile_changed"].to_numpy(bool)
        axes[0].scatter(
            frame["age_numeric_resolved"],
            y,
            s=np.where(changed, 90, 60),
            color=QUARTILE_COLORS[group],
            marker="o",
            edgecolor=np.where(changed, "black", "white"),
            linewidth=np.where(changed, 1.5, 0.5),
            alpha=0.9,
            label=group,
        )
    axes[0].set_yticks(range(len(QUARTILES)))
    axes[0].set_yticklabels(QUARTILES)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Resolved donor age")
    axes[0].set_ylabel("New equal-LEC-donor quartile")
    axes[0].set_title("Exactly 12 LEC-positive donors per new quartile")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].text(
        0.01,
        -0.16,
        "Black outline = donor changed from original quartile",
        transform=axes[0].transAxes,
        fontsize=9,
    )

    matrix = transition.set_index("original_age_quartile").reindex(
        index=QUARTILES, columns=QUARTILES
    )
    image = axes[1].imshow(matrix.to_numpy(), cmap="Blues", vmin=0)
    axes[1].set_xticks(range(len(QUARTILES)), QUARTILES)
    axes[1].set_yticks(range(len(QUARTILES)), QUARTILES)
    axes[1].set_xlabel("New quartile")
    axes[1].set_ylabel("Original quartile")
    axes[1].set_title("Original-to-new donor transition")
    for row in range(len(QUARTILES)):
        for column in range(len(QUARTILES)):
            value = int(matrix.iloc[row, column])
            axes[1].text(column, row, str(value), ha="center", va="center")
    fig.colorbar(image, ax=axes[1], label="Number of donors", fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_donor_cell_counts(
    mapping: pd.DataFrame, summary: pd.DataFrame, path: Path, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    frame = summary.set_index("new_age_quartile").reindex(QUARTILES)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = [QUARTILE_COLORS[group] for group in QUARTILES]
    axes[0].bar(QUARTILES, frame["n_donors"], color=colors)
    axes[0].axhline(DONORS_PER_QUARTILE, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("LEC-positive donors")
    axes[0].set_title("Donor count")
    axes[1].bar(QUARTILES, frame["n_LEC_cells"], color=colors)
    axes[1].set_ylabel("LEC cells")
    axes[1].set_title("Total LEC-cell count")
    values = [
        mapping.loc[mapping["new_age_quartile"].eq(group), "n_LEC_cells"].to_numpy()
        for group in QUARTILES
    ]
    axes[2].boxplot(values, tick_labels=QUARTILES, showfliers=False)
    for index, (group, group_values) in enumerate(zip(QUARTILES, values), start=1):
        jitter = np.linspace(-0.12, 0.12, len(group_values))
        axes[2].scatter(
            np.full(len(group_values), index) + jitter,
            group_values,
            color=QUARTILE_COLORS[group],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
            s=35,
        )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("LEC cells per donor (log scale)")
    axes[2].set_title("Unequal sampling remains")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("New quartiles: donor and LEC-cell support", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_state_composition(
    tables: dict[str, pd.DataFrame],
    new_key: str,
    state_key: str,
    palette: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    states = sorted(palette, key=natural_key)
    cell = tables["state_cell_composition"].pivot(
        index=new_key, columns=state_key, values="percent_within_new_quartile"
    ).reindex(index=QUARTILES, columns=states, fill_value=0).fillna(0)
    donor = tables["donor_state_composition"].pivot(
        index=new_key, columns=state_key, values="percent_mean_donor_fraction"
    ).reindex(index=QUARTILES, columns=states, fill_value=0).fillna(0)
    presence = tables["donor_state_presence"].pivot(
        index=new_key, columns=state_key, values="percent_donors_with_state"
    ).reindex(index=QUARTILES, columns=states, fill_value=0).fillna(0)
    colors = [palette[state] for state in states]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cell.plot(kind="bar", stacked=True, color=colors, ax=axes[0], width=0.78, legend=False)
    donor.plot(kind="bar", stacked=True, color=colors, ax=axes[1], width=0.78, legend=False)
    image = axes[2].imshow(presence.T.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=100)
    axes[0].set_title("Pooled-cell composition")
    axes[0].set_ylabel("Cells within new quartile (%)")
    axes[1].set_title("Donor-normalized composition")
    axes[1].set_ylabel("Mean donor LEC fraction (%)")
    for ax in axes[:2]:
        ax.set_xlabel("New equal-donor quartile")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.2)
    axes[2].set_title("Donor presence for each state")
    axes[2].set_xticks(range(len(QUARTILES)), QUARTILES)
    axes[2].set_yticks(range(len(states)), states)
    axes[2].set_xlabel("New equal-donor quartile")
    axes[2].set_ylabel("LEC state")
    fig.colorbar(image, ax=axes[2], label="Donors with state (%)", fraction=0.046)
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=palette[state],
                   markeredgecolor="none", markersize=9, label=state)
        for state in states
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(states)),
               bbox_to_anchor=(0.5, -0.04), title="LEC state")
    fig.suptitle("LEC-state composition across the new age quartiles", fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compact_donor_label(value: object) -> str:
    """Keep figure annotations readable while the CSV retains the full key."""
    text = str(value)
    return text.rsplit("::", maxsplit=1)[-1]


def save_highest_donor_contribution(
    tables: dict[str, pd.DataFrame],
    new_key: str,
    state_key: str,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    state_table = tables["highest_donor_quartile_state"]
    overall_table = tables["highest_donor_quartile_overall"]
    states = sorted(state_table[state_key].unique(), key=natural_key)
    percent = state_table.pivot(
        index=state_key,
        columns=new_key,
        values="top_donor_percent_of_quartile_state",
    ).reindex(index=states, columns=QUARTILES)
    donor = state_table.pivot(
        index=state_key,
        columns=new_key,
        values="top_donor_key",
    ).reindex(index=states, columns=QUARTILES)

    height = max(6.5, 0.62 * len(states) + 2.3)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, height),
        gridspec_kw={"width_ratios": [2.35, 1]},
    )
    cmap = matplotlib.colormaps["YlOrRd"].copy()
    cmap.set_bad("#eeeeee")
    image = axes[0].imshow(
        percent.to_numpy(float), aspect="auto", cmap=cmap, vmin=0, vmax=100
    )
    axes[0].set_xticks(range(len(QUARTILES)), QUARTILES)
    axes[0].set_yticks(range(len(states)), states)
    axes[0].set_xlabel("New equal-donor quartile")
    axes[0].set_ylabel("LEC state")
    axes[0].set_title("Largest donor share within each quartile-state pool")
    for row, state in enumerate(states):
        for column, quartile in enumerate(QUARTILES):
            value = percent.loc[state, quartile]
            donor_key = donor.loc[state, quartile]
            if pd.isna(value):
                axes[0].text(column, row, "no cells", ha="center", va="center", fontsize=7)
                continue
            axes[0].text(
                column,
                row,
                f"{value:.1f}%\n{compact_donor_label(donor_key)}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if value >= 52 else "black",
                weight="bold" if value > 50 else "normal",
            )
            if value > 50:
                axes[0].add_patch(
                    Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="black",
                        linewidth=2.2,
                    )
                )
    fig.colorbar(
        image,
        ax=axes[0],
        label="Largest single-donor contribution (%)",
        fraction=0.045,
        pad=0.03,
    )

    overall = overall_table.set_index(new_key).reindex(QUARTILES)
    overall_values = overall[
        "top_donor_percent_of_quartile_LECs"
    ].to_numpy(float)
    bars = axes[1].bar(
        QUARTILES,
        overall_values,
        color=[QUARTILE_COLORS[group] for group in QUARTILES],
        width=0.72,
    )
    axes[1].axhline(50, color="black", linestyle="--", linewidth=1, alpha=0.7)
    for bar, quartile, value in zip(bars, QUARTILES, overall_values):
        donor_key = overall.loc[quartile, "top_donor_key"]
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%\n{compact_donor_label(donor_key)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_ylim(0, 108)
    axes[1].set_xlabel("New equal-donor quartile")
    axes[1].set_ylabel("Largest donor contribution (%)")
    axes[1].set_title("Largest donor share of all LECs\nin each quartile")
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Highest single-donor contribution", fontsize=15)
    fig.text(
        0.5,
        0.012,
        "Left denominator: all cells in the same quartile and state. "
        "Black outline marks >50%. Right denominator: all LEC cells in the quartile.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_project_sex_balance(
    mapping: pd.DataFrame, path: Path, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    project = pd.crosstab(mapping["new_age_quartile"], mapping["Project"]).reindex(QUARTILES)
    sex = pd.crosstab(mapping["new_age_quartile"], mapping["Sex"]).reindex(QUARTILES)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    project.plot(kind="bar", stacked=True, ax=axes[0], width=0.78, colormap="tab10")
    sex.plot(kind="bar", stacked=True, ax=axes[1], width=0.78, color=["#e377c2", "#17becf", "#7f7f7f"][: len(sex.columns)])
    axes[0].set_title("Project distribution")
    axes[1].set_title("Sex distribution")
    for ax in axes:
        ax.set_xlabel("New equal-donor quartile")
        ax.set_ylabel("Number of donors")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Donor covariate balance in the new quartiles", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def categorical_scatter(
    ax,
    coordinates: np.ndarray,
    values: pd.Series,
    palette: dict[str, str],
    title: str,
    label_on_data: bool = False,
) -> None:
    categories = sorted(normalized_text(values).unique(), key=natural_key)
    text_values = normalized_text(values).to_numpy()
    for category in categories:
        mask = text_values == category
        ax.scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=9,
            color=palette.get(category, "#808080"), alpha=0.78,
            linewidth=0, label=category,
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
    original_key: str,
    new_key: str,
    state_key: str,
    age_key: str,
    project_key: str,
    sex_key: str,
    state_colors: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    if "X_umap" not in adata.obsm:
        raise KeyError("The Step 08 h5ad has no X_umap embedding.")
    coordinates = np.asarray(adata.obsm["X_umap"])
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    categorical_scatter(
        axes[0, 0], coordinates, adata.obs[new_key], QUARTILE_COLORS,
        "New quartile: exactly 12 LEC-positive donors",
    )
    categorical_scatter(
        axes[0, 1], coordinates, adata.obs[original_key], QUARTILE_COLORS,
        "Original quartile (preserved)",
    )
    categorical_scatter(
        axes[0, 2], coordinates, adata.obs[state_key], state_colors,
        "Unchanged resolution-0.3 LEC states", label_on_data=True,
    )
    age = pd.to_numeric(adata.obs[age_key], errors="coerce").to_numpy(float)
    age_plot = axes[1, 0].scatter(
        coordinates[:, 0], coordinates[:, 1], c=age, s=9,
        cmap="viridis", alpha=0.8, linewidth=0,
    )
    axes[1, 0].set_title("Resolved donor age")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[1, 0].set_frame_on(False)
    fig.colorbar(age_plot, ax=axes[1, 0], label="Age")
    project_values = normalized_text(adata.obs[project_key])
    project_categories = sorted(project_values.unique(), key=natural_key)
    project_cmap = list(matplotlib.colormaps["tab10"].colors)
    project_palette = {
        value: matplotlib.colors.to_hex(project_cmap[index % len(project_cmap)])
        for index, value in enumerate(project_categories)
    }
    categorical_scatter(
        axes[1, 1], coordinates, project_values, project_palette, "Project",
    )
    sex_palette = {"F": "#e377c2", "M": "#17becf", "unknown": "#7f7f7f"}
    categorical_scatter(
        axes[1, 2], coordinates, adata.obs[sex_key], sex_palette, "Sex",
    )
    fig.suptitle(
        "Step 08.3 metadata-only age regrouping; embedding and states unchanged",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    path: Path,
    input_h5ad: Path,
    output_h5ad: Path,
    original_key: str,
    new_key: str,
    summary: pd.DataFrame,
    split_table: pd.DataFrame,
    selection_objective: dict[str, object],
    step08_resolution: object,
) -> None:
    ranges = summary[
        ["new_age_quartile", "age_min", "age_max", "n_donors", "n_LEC_cells"]
    ].to_csv(index=False).strip()
    split_text = (
        split_table.to_csv(index=False).strip()
        if not split_table.empty
        else "No age tie was split across a new quartile boundary."
    )
    text = f"""Step 08.3 equal-LEC-donor age-quartile notes
================================================

Run identity
------------
Script version: {SCRIPT_VERSION}
Input h5ad: {input_h5ad}
Output h5ad: {output_h5ad}
Source Step 08 Leiden resolution: {step08_resolution}

What changed
------------
A new metadata column, {new_key}, was added using exactly 12 donors per
quartile. The original {original_key} column was preserved without modification.
Range-label and original-label-change columns were also added.

What did not change
-------------------
No cells or genes were removed. X, raw, PCA, Harmony coordinates, neighbors,
UMAP, Leiden clusters, LEC_state labels, marker rankings, and state colors were
not recalculated. Current state-based Step 09 GSEA and Step 11 SCENIC results
therefore remain valid and unchanged.

Assignment rule
---------------
Donors were ordered by resolved age. Exact 12-donor boundaries were enforced.
Among age-ordered assignments, the selected mapping minimized the number of
changes from the original quartile first, then minimized squared imbalance in
total LEC-cell counts, then used donor_key for reproducibility.
Original-quartile changes: {selection_objective['n_original_quartile_changes']}

New observed ranges
-------------------
{ranges}

Split-age ties
--------------
Exact equal donor counts require identical ages at some boundaries to be split,
so adjacent observed age ranges can overlap. This is deliberate and recorded.
{split_text}

Interpretation
--------------
These new quartiles are intended for descriptive visualization and donor-level
age sensitivity analyses. Equal donor counts do not equalize LEC-cell counts.
For statistical aging tests, use donor-level pseudobulk or donor-level models;
do not treat individual cells as independent biological replicates. Continuous
age should remain the preferred primary exposure when feasible.

How to read the composition outputs
-----------------------------------
Pooled-cell composition combines all LEC cells in a quartile before calculating
state percentages. Every cell has equal weight, so a donor with many LEC cells
has more influence than a donor with few LEC cells.

Donor-normalized composition first calculates the state percentages separately
within each donor, including zero for states absent from that donor, and then
averages those percentages across the 12 donors. Every donor therefore has equal
weight, although a donor represented by only a few LEC cells can have an imprecise
within-donor percentage.

The highest-single-donor table and figure report, for every quartile-state pair,
the largest donor cell count divided by all cells in that same quartile-state
pool. This is the diagnostic that directly identifies a state segment dominated
by one donor. Values above 50 percent are explicitly flagged. The figure also
shows the largest donor share of all LEC cells in each quartile for context.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_dir = find_project_dir()
    input_h5ad = resolve_path(args.input_h5ad, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    output_h5ad = output_root / args.out_h5ad
    if not input_h5ad.exists():
        raise FileNotFoundError(f"Step 08 input not found: {input_h5ad}")
    if output_h5ad.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(
            f"Primary Step 08.3 output exists: {output_h5ad}. Use --overwrite."
        )

    print(f"Reading Step 08 h5ad: {input_h5ad}")
    adata = ad.read_h5ad(input_h5ad)
    original_obs_columns = list(adata.obs.columns)
    required_columns = [
        args.original_quartile_key,
        args.state_key,
        args.donor_key,
        args.age_key,
        args.project_key,
        args.sex_key,
    ]
    missing = [column for column in required_columns if column not in adata.obs]
    if missing:
        raise KeyError(f"Input h5ad is missing columns: {missing}")
    new_columns = [
        args.new_quartile_key,
        f"{args.new_quartile_key}_range_label",
        f"{args.new_quartile_key}_changed_from_original",
    ]
    existing_new = [column for column in new_columns if column in adata.obs]
    if existing_new:
        raise ValueError(
            "Input already contains Step 08.3 columns; use the primary Step 08 "
            f"h5ad instead. Existing: {existing_new}"
        )

    original_quartile_before = normalized_text(
        adata.obs[args.original_quartile_key]
    ).copy()
    donor_mapping = donor_table_from_obs(
        adata.obs,
        args.donor_key,
        args.age_key,
        args.original_quartile_key,
        args.state_key,
        args.project_key,
        args.sex_key,
    )
    donor_mapping, split_table = assign_exact_quartiles(donor_mapping)
    selection_objective = dict(donor_mapping.attrs["selection_objective"])
    range_labels = make_range_labels(donor_mapping)

    print("\nProposed exact-12 donor mapping:")
    print(
        donor_mapping[
            [
                "age_order_rank", "donor_key", "age_numeric_resolved",
                "original_age_quartile", "new_age_quartile",
                "quartile_changed", "n_LEC_cells",
            ]
        ].to_string(index=False)
    )
    print("\nSplit-age boundaries:")
    print(split_table.to_string(index=False) if not split_table.empty else "none")

    donor_to_new = donor_mapping.set_index("donor_key")["new_age_quartile"]
    donor_to_changed = donor_mapping.set_index("donor_key")["quartile_changed"]
    donor_values = normalized_text(adata.obs[args.donor_key])
    cell_new = donor_values.map(donor_to_new)
    if cell_new.isna().any():
        raise RuntimeError("Some Step 08 cells did not map to a new donor quartile.")
    adata.obs[args.new_quartile_key] = pd.Categorical(
        cell_new, categories=QUARTILES, ordered=True
    )
    adata.obs[f"{args.new_quartile_key}_range_label"] = pd.Categorical(
        cell_new.map(range_labels),
        categories=[range_labels[group] for group in QUARTILES],
        ordered=True,
    )
    adata.obs[f"{args.new_quartile_key}_changed_from_original"] = (
        donor_values.map(donor_to_changed).astype(bool).to_numpy()
    )
    adata.uns[f"{args.new_quartile_key}_colors"] = [
        QUARTILE_COLORS[group] for group in QUARTILES
    ]
    adata.uns[f"{args.new_quartile_key}_color_map"] = dict(QUARTILE_COLORS)

    if not normalized_text(adata.obs[args.original_quartile_key]).equals(
        original_quartile_before
    ):
        raise RuntimeError("Original age_quartile changed unexpectedly.")
    if adata.n_obs != len(cell_new):
        raise RuntimeError("Cell-count invariant failed.")

    tables = make_summary_tables(
        adata.obs,
        donor_mapping,
        args.new_quartile_key,
        args.state_key,
        args.donor_key,
    )
    print("\nNew quartile summary:")
    print(tables["quartile_summary"].to_string(index=False))
    if args.dry_run:
        print("\nDry run complete. No files were written.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    if not args.skip_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    donor_mapping.to_csv(
        output_root / "08_3_donor_quartile_mapping.csv", index=False
    )
    split_table.to_csv(
        output_root / "08_3_split_age_ties.csv", index=False
    )
    tables["quartile_summary"].to_csv(
        output_root / "08_3_quartile_summary.csv", index=False
    )
    tables["transition"].to_csv(
        output_root / "08_3_original_to_new_quartile_transition.csv", index=False
    )
    tables["project_counts"].to_csv(
        output_root / "08_3_donor_counts_by_quartile_project.csv", index=False
    )
    tables["sex_counts"].to_csv(
        output_root / "08_3_donor_counts_by_quartile_sex.csv", index=False
    )
    tables["state_cell_composition"].to_csv(
        output_root / "08_3_state_composition_by_new_quartile.csv", index=False
    )
    tables["donor_state_composition"].to_csv(
        output_root / "08_3_donor_normalized_state_composition.csv", index=False
    )
    tables["donor_state_presence"].to_csv(
        output_root / "08_3_state_donor_presence_by_new_quartile.csv", index=False
    )
    tables["donor_state_full"].to_csv(
        output_root / "08_3_donor_state_fractions.csv.gz",
        index=False,
        compression="gzip",
    )
    tables["highest_donor_quartile_state"].to_csv(
        output_root / "08_3_highest_donor_contribution_by_quartile_state.csv",
        index=False,
    )
    tables["highest_donor_quartile_overall"].to_csv(
        output_root / "08_3_highest_donor_contribution_by_quartile.csv",
        index=False,
    )
    adata.obs.reset_index(names="combined_cell_id").to_csv(
        output_root / "08_3_combined_LEC_cell_metadata.csv.gz",
        index=False,
        compression="gzip",
    )

    states = sorted(normalized_text(adata.obs[args.state_key]).unique(), key=natural_key)
    palette = state_palette(adata, args.state_key, states)
    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        figure_specs = [
            (
                "08_3_donor_age_distribution_and_transition.png",
                "Donor ages and original-to-new quartile transition",
                lambda path: save_donor_distribution(
                    donor_mapping, tables["transition"], path, args.dpi
                ),
            ),
            (
                "08_3_quartile_donor_and_cell_counts.png",
                "Donor count, total LEC cells, and LEC cells per donor",
                lambda path: save_donor_cell_counts(
                    donor_mapping, tables["quartile_summary"], path, args.dpi
                ),
            ),
            (
                "08_3_state_composition_by_quartile.png",
                "Pooled-cell, donor-normalized, and donor-presence state composition",
                lambda path: save_state_composition(
                    tables, args.new_quartile_key, args.state_key, palette,
                    path, args.dpi,
                ),
            ),
            (
                "08_3_highest_single_donor_contribution.png",
                (
                    "Largest single-donor share of each quartile-state cell pool "
                    "and of each quartile overall"
                ),
                lambda path: save_highest_donor_contribution(
                    tables,
                    args.new_quartile_key,
                    args.state_key,
                    path,
                    args.dpi,
                ),
            ),
            (
                "08_3_quartile_project_sex_balance.png",
                "Project and sex distribution among donors",
                lambda path: save_project_sex_balance(
                    donor_mapping, path, args.dpi
                ),
            ),
            (
                "08_3_umap_summary.png",
                "Unchanged Step 08 UMAP colored by new and original metadata",
                lambda path: save_umap_summary(
                    adata,
                    args.original_quartile_key,
                    args.new_quartile_key,
                    args.state_key,
                    args.age_key,
                    args.project_key,
                    args.sex_key,
                    palette,
                    path,
                    args.dpi,
                ),
            ),
        ]
        for filename, description, function in figure_specs:
            path = figures_dir / filename
            function(path)
            figure_manifest.append(
                {"figure": str(path), "description": description, "status": "ok"}
            )
    pd.DataFrame(
        figure_manifest, columns=["figure", "description", "status"]
    ).to_csv(output_root / "08_3_figure_manifest.csv", index=False)

    step08_embedding = dict(adata.uns.get("combined_LEC_embedding", {}))
    step08_combine = dict(adata.uns.get("step08_combine_info", {}))
    step08_resolution = step08_embedding.get(
        "leiden_resolution", step08_combine.get("leiden_resolution", "unknown")
    )
    adata.uns["step08_3_equal_LEC_donor_age_quartiles"] = {
        "script_version": SCRIPT_VERSION,
        "source_h5ad": str(input_h5ad),
        "original_quartile_key_preserved": args.original_quartile_key,
        "new_quartile_key": args.new_quartile_key,
        "n_quartiles": N_QUARTILES,
        "donors_per_quartile": DONORS_PER_QUARTILE,
        "n_LEC_positive_donors": EXPECTED_DONORS,
        "assignment_rule": (
            "age_order; minimize original-label changes; then minimize LEC-cell "
            "imbalance; then donor_key"
        ),
        "n_original_quartile_changes": int(
            selection_objective["n_original_quartile_changes"]
        ),
        "split_ages": split_table["split_age"].astype(float).tolist(),
        "expression_embedding_states_unchanged": True,
    }

    run_summary = {
        "script_version": SCRIPT_VERSION,
        "status": "completed",
        "input_h5ad": str(input_h5ad),
        "output_h5ad": str(output_h5ad),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_LEC_positive_donors": int(donor_mapping["donor_key"].nunique()),
        "n_quartiles": N_QUARTILES,
        "donors_per_quartile": DONORS_PER_QUARTILE,
        "new_quartile_key": args.new_quartile_key,
        "original_quartile_key_preserved": args.original_quartile_key,
        "n_original_quartile_changes": int(
            selection_objective["n_original_quartile_changes"]
        ),
        "split_ages": ";".join(
            f"{value:g}" for value in split_table["split_age"].astype(float)
        ),
        "step08_leiden_resolution": step08_resolution,
        "n_states": int(len(states)),
        "state_labels": ";".join(states),
        "reclustered": False,
    }
    pd.DataFrame([run_summary]).to_csv(
        output_root / "08_3_run_summary.csv", index=False
    )
    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "input_h5ad": str(input_h5ad),
        "output_root": str(output_root),
        "output_h5ad": str(output_h5ad),
        "original_quartile_key": args.original_quartile_key,
        "new_quartile_key": args.new_quartile_key,
        "state_key": args.state_key,
        "donor_key": args.donor_key,
        "age_key": args.age_key,
        "project_key": args.project_key,
        "sex_key": args.sex_key,
        "n_quartiles": N_QUARTILES,
        "donors_per_quartile": DONORS_PER_QUARTILE,
        "assignment_selection_objective": selection_objective,
        "range_labels": range_labels,
        "figures_enabled": not args.skip_figures,
        "dpi": args.dpi,
    }
    (output_root / "08_3_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    write_notes(
        output_root / "08_3_method_and_interpretation_notes.txt",
        input_h5ad,
        output_h5ad,
        args.original_quartile_key,
        args.new_quartile_key,
        tables["quartile_summary"],
        split_table,
        selection_objective,
        step08_resolution,
    )

    if list(adata.obs.columns[: len(original_obs_columns)]) != original_obs_columns:
        raise RuntimeError("Original obs-column order changed unexpectedly.")
    print(f"\nWriting metadata-only Step 08.3 h5ad: {output_h5ad}")
    adata.write_h5ad(output_h5ad, compression="gzip")
    print("Step 08.3 complete.")
    print("Original quartile preserved; expression, embedding, and states unchanged.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
