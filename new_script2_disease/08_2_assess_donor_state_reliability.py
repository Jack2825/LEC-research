#!/usr/bin/env python3
r"""
Step 08.2: assess donor and LEC-state reliability in the disease pipeline.

This is a read-only analysis of the exact final Step 08 LEC clustering object.
It does not remove cells or donors, choose a minimum threshold automatically,
rerun normalization/clustering, or write an h5ad. The objectives are to:

1. describe final-LEC support for every donor and state;
2. quantify donor, dataset, and cohort dominance of each state;
3. show how candidate minimum-LEC thresholds change donor/cohort retention;
4. estimate composition error by repeatedly downsampling the same high-depth
   reference donors to several candidate cell counts; and
5. quantify leave-one-donor-out sensitivity of pooled state composition.

The primary resampling reference set is fixed across all tested N values. By
default, every reference donor has at least 25 final LECs, and each donor is
sampled independently without replacement 2,000 times at
N=3,5,8,10,12,15,20,25. This matches aging Step 08.6, keeps donor weights equal,
and avoids changing the donor mix across N values.

Default input
-------------
new_result2_disease/08_combined_LEC_recluster/
    08_combined_final_LEC_reclustered.h5ad

Optional descriptive QC input
-----------------------------
new_result2_disease/08_1_LEC_identity_doublet_QC/tables/
    08_1_LEC_cell_identity_doublet_QC.csv.gz

Default output
--------------
new_result2_disease/08_2_donor_state_reliability/

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\08_2_assess_donor_state_reliability.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


SCRIPT_VERSION = "08_2_disease_final_LEC_donor_state_reliability_2026_08_11"
DEFAULT_INPUT = (
    "new_result2_disease/08_combined_LEC_recluster/"
    "08_combined_final_LEC_reclustered.h5ad"
)
DEFAULT_STEP08_1_CELL_QC = (
    "new_result2_disease/08_1_LEC_identity_doublet_QC/tables/"
    "08_1_LEC_cell_identity_doublet_QC.csv.gz"
)
DEFAULT_OUTPUT = "new_result2_disease/08_2_donor_state_reliability"
DEFAULT_N_VALUES = [3, 5, 8, 10, 12, 15, 20, 25]
DEFAULT_RETENTION_THRESHOLDS = [1, 5, 10, 15, 20, 25, 30]
DEFAULT_RESAMPLES = 2000
DEFAULT_TARGET_QUANTILE = "p90"
DEFAULT_TARGET_TV_ERROR_PCT = 20.0
TOLERANCE_GRID = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
EXPECTED_DATASETS = ["dataset01", "dataset08", "dataset11"]
EXPECTED_CONDITIONS = ["NF", "DCM", "HCM", "DoxCM", "HFpEF"]
EXPECTED_REGIONS = {"dataset01": "LV", "dataset08": "LV", "dataset11": "S"}

CONDITION_COLORS = {
    "NF": "#0072B2",
    "DCM": "#D55E00",
    "HCM": "#CC79A7",
    "DoxCM": "#56B4E9",
    "HFpEF": "#E69F00",
}
DATASET_COLORS = {
    "dataset01": "#4C78A8",
    "dataset08": "#F58518",
    "dataset11": "#54A24B",
}


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def natural_key(value: object) -> list[object]:
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", str(value))]


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def true_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return normalized_text(values).str.casefold().isin({"true", "1", "yes", "y", "t"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess donor and state reliability using final Step 08 disease LECs."
    )
    parser.add_argument("--input_h5ad", default=DEFAULT_INPUT)
    parser.add_argument("--step08_1_cell_qc", default=DEFAULT_STEP08_1_CELL_QC)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--reference_min_LECs", type=int, default=25)
    parser.add_argument("--n_values", nargs="+", type=int, default=DEFAULT_N_VALUES)
    parser.add_argument(
        "--retention_thresholds",
        nargs="+",
        type=int,
        default=DEFAULT_RETENTION_THRESHOLDS,
    )
    parser.add_argument("--n_resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=8022026)
    parser.add_argument(
        "--target_quantile",
        choices=("p90", "p95"),
        default=DEFAULT_TARGET_QUANTILE,
        help="Error quantile used for the descriptive candidate cutoff.",
    )
    parser.add_argument(
        "--target_tv_error_pct",
        type=float,
        default=DEFAULT_TARGET_TV_ERROR_PCT,
        help="Maximum TV error at target_quantile for the candidate cutoff.",
    )
    parser.add_argument("--donor_dominance_threshold", type=float, default=0.50)
    parser.add_argument("--dataset_dominance_threshold", type=float, default=0.90)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def state_palette(adata: ad.AnnData, states: list[str]) -> dict[str, str]:
    stored = adata.uns.get("LEC_state_color_map")
    if isinstance(stored, dict) and all(state in stored for state in states):
        return {state: str(stored[state]) for state in states}
    colors = list(adata.uns.get("LEC_state_colors", []))
    column = adata.obs["LEC_state"]
    if isinstance(column.dtype, pd.CategoricalDtype):
        categories = list(map(str, column.cat.categories))
        mapped = dict(zip(categories, colors))
        if all(state in mapped for state in states):
            return {state: mapped[state] for state in states}
    fallback = list(plt.get_cmap("tab20").colors)
    return {state: fallback[index % len(fallback)] for index, state in enumerate(states)}


def color_state_ticks(ax: plt.Axes, palette: dict[str, str], axis: str = "x") -> None:
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for label in labels:
        state = label.get_text()
        if state in palette:
            label.set_color(palette[state])
            label.set_fontweight("bold")


def validate_obs(obs: pd.DataFrame) -> None:
    required = {
        "dataset_folder",
        "condition",
        "donor_key",
        "sample_key",
        "LEC_state",
        "region_standardized_analysis",
        "step03_final_QC_pass",
        "selected_as_LEC",
    }
    missing = sorted(required - set(obs.columns))
    if missing:
        raise KeyError(f"Step 08 obs is missing required columns: {missing}")
    if (~true_mask(obs["step03_final_QC_pass"])).any():
        raise ValueError("Step 08 contains cells that did not pass Step 03 QC.")
    if (~true_mask(obs["selected_as_LEC"])).any():
        raise ValueError("Step 08 contains cells not selected as final LECs.")
    datasets = sorted(normalized_text(obs["dataset_folder"]).unique())
    conditions = sorted(normalized_text(obs["condition"]).unique())
    if datasets != EXPECTED_DATASETS:
        raise ValueError(f"Unexpected datasets: {datasets}")
    if sorted(conditions) != sorted(EXPECTED_CONDITIONS):
        raise ValueError(f"Unexpected conditions: {conditions}")
    region_check = (
        obs.assign(
            dataset_folder=normalized_text(obs["dataset_folder"]),
            region_standardized_analysis=normalized_text(obs["region_standardized_analysis"]),
        )
        .groupby("dataset_folder", observed=True)["region_standardized_analysis"]
        .agg(lambda x: ";".join(sorted(x.unique())))
        .to_dict()
    )
    if region_check != EXPECTED_REGIONS:
        raise ValueError(f"Unexpected regions: {region_check}")


def attach_optional_step08_1_qc(
    obs: pd.DataFrame, path: Path
) -> tuple[pd.DataFrame, str]:
    result = obs.copy()
    default_columns = {
        "mixed_cardiac_endothelial_review_flag": False,
        "doublet_percentile_ge_99": False,
        "doublet_score_percentile_within_final_LECs": np.nan,
    }
    if not path.is_file():
        for column, default in default_columns.items():
            result[column] = default
        return result, "not_found_not_required"

    qc = pd.read_csv(path, low_memory=False)
    if "step08_cell_id" not in qc.columns:
        raise KeyError(f"Optional Step 08.1 table has no step08_cell_id: {path}")
    qc["step08_cell_id"] = qc["step08_cell_id"].astype(str)
    if qc["step08_cell_id"].duplicated().any():
        raise ValueError("Optional Step 08.1 table contains duplicate step08_cell_id values.")
    input_ids = pd.Index(result.index.astype(str))
    qc_ids = pd.Index(qc["step08_cell_id"])
    if set(input_ids) != set(qc_ids):
        raise ValueError(
            "Optional Step 08.1 cell table does not exactly match the current Step 08 cells."
        )
    available = [column for column in default_columns if column in qc.columns]
    qc = qc.set_index("step08_cell_id").reindex(input_ids)
    for column, default in default_columns.items():
        if column in available:
            if column.endswith("flag") or column.startswith("doublet_percentile_ge"):
                result[column] = true_mask(qc[column]).to_numpy()
            else:
                result[column] = pd.to_numeric(qc[column], errors="coerce").to_numpy()
        else:
            result[column] = default
    return result, "exact_cell_match_loaded"


def make_donor_tables(
    obs: pd.DataFrame, states: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_columns = [
        column
        for column in [
            "dataset_folder",
            "condition",
            "sex",
            "age",
            "lvef",
            "region_standardized_analysis",
        ]
        if column in obs.columns
    ]
    for column in ["dataset_folder", "condition", "region_standardized_analysis"]:
        conflicts = obs.groupby("donor_key", observed=True)[column].nunique(dropna=False)
        if (conflicts > 1).any():
            bad = conflicts[conflicts > 1].index.tolist()[:5]
            raise ValueError(f"Donor metadata conflict in {column}: {bad}")

    donor_meta = (
        obs.reset_index(names="step08_cell_id")
        .sort_values("step08_cell_id")
        .drop_duplicates("donor_key")[["donor_key", *metadata_columns]]
        .set_index("donor_key")
    )
    donor_meta["cohort"] = (
        donor_meta["dataset_folder"].astype(str)
        + " | "
        + donor_meta["condition"].astype(str)
    )

    counts = pd.crosstab(obs["donor_key"], obs["LEC_state"]).reindex(
        index=donor_meta.index, columns=states, fill_value=0
    )
    counts = counts.astype(int)
    totals = counts.sum(axis=1)
    proportions = counts.div(totals, axis=0)

    donor_summary = donor_meta.copy()
    donor_summary["n_final_LECs"] = totals
    donor_summary["n_LEC_states_detected"] = (counts > 0).sum(axis=1)
    donor_summary["dominant_LEC_state"] = counts.idxmax(axis=1)
    donor_summary["dominant_state_fraction"] = proportions.max(axis=1)
    donor_summary["fraction_relative_cardiac_review_flags"] = obs.groupby(
        "donor_key", observed=True
    )["mixed_cardiac_endothelial_review_flag"].mean()
    donor_summary["fraction_relative_doublet_q99_flags"] = obs.groupby(
        "donor_key", observed=True
    )["doublet_percentile_ge_99"].mean()
    donor_summary["n_numeric_doublet_percentiles"] = obs.groupby(
        "donor_key", observed=True
    )["doublet_score_percentile_within_final_LECs"].count()
    donor_summary = donor_summary.reset_index()

    wide_counts = donor_summary[["donor_key", *metadata_columns, "cohort", "n_final_LECs"]].merge(
        counts.reset_index(), on="donor_key", how="left", validate="one_to_one"
    )
    long = counts.reset_index().melt(
        id_vars="donor_key", var_name="LEC_state", value_name="n_state_LECs"
    )
    long = long.merge(
        donor_summary[["donor_key", "dataset_folder", "condition", "cohort", "n_final_LECs"]],
        on="donor_key",
        validate="many_to_one",
    )
    long["state_fraction_within_donor"] = long["n_state_LECs"] / long["n_final_LECs"]
    return donor_summary, wide_counts, long


def make_leave_one_donor_out(
    donor_state_long: pd.DataFrame, states: list[str]
) -> pd.DataFrame:
    counts = donor_state_long.pivot(index="donor_key", columns="LEC_state", values="n_state_LECs")
    counts = counts.reindex(columns=states, fill_value=0).astype(float)
    proportions = counts.div(counts.sum(axis=1), axis=0)
    total_counts = counts.sum(axis=0)
    total_cells = float(total_counts.sum())
    pooled = total_counts / total_cells
    donor_balanced = proportions.mean(axis=0)
    rows: list[dict[str, object]] = []
    metadata = donor_state_long.drop_duplicates("donor_key").set_index("donor_key")
    for donor in counts.index:
        remaining_counts = total_counts - counts.loc[donor]
        remaining_total = float(remaining_counts.sum())
        remaining_pooled = remaining_counts / remaining_total
        remaining_balanced = proportions.drop(index=donor).mean(axis=0)
        for state in states:
            rows.append(
                {
                    "donor_key": donor,
                    "dataset_folder": metadata.at[donor, "dataset_folder"],
                    "condition": metadata.at[donor, "condition"],
                    "cohort": metadata.at[donor, "cohort"],
                    "LEC_state": state,
                    "donor_state_LECs": int(counts.at[donor, state]),
                    "all_donor_cell_pooled_fraction": float(pooled[state]),
                    "leave_one_donor_out_cell_pooled_fraction": float(remaining_pooled[state]),
                    "absolute_cell_pooled_change": float(abs(remaining_pooled[state] - pooled[state])),
                    "absolute_cell_pooled_change_percentage_points": float(
                        100.0 * abs(remaining_pooled[state] - pooled[state])
                    ),
                    "all_donor_balanced_fraction": float(donor_balanced[state]),
                    "leave_one_donor_out_balanced_fraction": float(remaining_balanced[state]),
                    "absolute_donor_balanced_change": float(
                        abs(remaining_balanced[state] - donor_balanced[state])
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_state_support(
    obs: pd.DataFrame,
    donor_state_long: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    states: list[str],
    donor_threshold: float,
    dataset_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_rows: list[dict[str, object]] = []
    for (cohort, state), part in obs.groupby(["cohort", "LEC_state"], observed=True):
        cohort_total = int((obs["cohort"] == cohort).sum())
        state_total = int((obs["LEC_state"] == state).sum())
        cohort_rows.append(
            {
                "cohort": str(cohort),
                "dataset_folder": str(part["dataset_folder"].iloc[0]),
                "condition": str(part["condition"].iloc[0]),
                "LEC_state": str(state),
                "n_LEC_cells": int(len(part)),
                "n_donors_with_state": int(part["donor_key"].nunique()),
                "fraction_within_cohort": float(len(part) / cohort_total),
                "fraction_of_state": float(len(part) / state_total),
            }
        )
    cohort_support = pd.DataFrame(cohort_rows)

    all_donors = int(obs["donor_key"].nunique())
    rows: list[dict[str, object]] = []
    for state in states:
        part = obs.loc[obs["LEC_state"].eq(state)]
        donor_counts = part["donor_key"].value_counts()
        dataset_counts = part["dataset_folder"].value_counts()
        condition_counts = part["condition"].value_counts()
        cohort_counts = part["cohort"].value_counts()
        donor_fractions = donor_counts / donor_counts.sum()
        top_donor = str(donor_counts.index[0])
        top_dataset = str(dataset_counts.index[0])
        top_condition = str(condition_counts.index[0])
        top_cohort = str(cohort_counts.index[0])
        state_donor_rows = donor_state_long.loc[donor_state_long["LEC_state"].eq(state)]
        loo = leave_one_out.loc[leave_one_out["LEC_state"].eq(state)]
        influential = loo.sort_values(
            "absolute_cell_pooled_change", ascending=False
        ).iloc[0]
        flags: list[str] = []
        if float(donor_fractions.iloc[0]) >= donor_threshold:
            flags.append("donor_dominance")
        if float(dataset_counts.iloc[0] / len(part)) >= dataset_threshold:
            flags.append("dataset_dominance")
        if part["dataset_folder"].nunique() < len(EXPECTED_DATASETS):
            flags.append("not_present_in_all_datasets")
        rows.append(
            {
                "LEC_state": state,
                "n_LEC_cells": int(len(part)),
                "cell_pooled_fraction": float(len(part) / len(obs)),
                "n_donors_with_state": int(donor_counts.size),
                "fraction_all_donors_with_state": float(donor_counts.size / all_donors),
                "n_donors_with_at_least_5_state_cells": int((donor_counts >= 5).sum()),
                "n_donors_with_at_least_10_state_cells": int((donor_counts >= 10).sum()),
                "donor_balanced_mean_state_fraction": float(
                    state_donor_rows["state_fraction_within_donor"].mean()
                ),
                "effective_donor_number": float(1.0 / np.square(donor_fractions).sum()),
                "top_donor": top_donor,
                "top_donor_n_LECs": int(donor_counts.iloc[0]),
                "top_donor_fraction": float(donor_fractions.iloc[0]),
                "n_datasets": int(part["dataset_folder"].nunique()),
                "top_dataset": top_dataset,
                "top_dataset_fraction": float(dataset_counts.iloc[0] / len(part)),
                "n_conditions": int(part["condition"].nunique()),
                "top_condition": top_condition,
                "top_condition_fraction": float(condition_counts.iloc[0] / len(part)),
                "n_cohorts": int(part["cohort"].nunique()),
                "top_cohort": top_cohort,
                "top_cohort_fraction": float(cohort_counts.iloc[0] / len(part)),
                "most_influential_donor_cell_pooled": str(influential["donor_key"]),
                "max_leave_one_donor_out_change_percentage_points": float(
                    influential["absolute_cell_pooled_change_percentage_points"]
                ),
                "fraction_relative_cardiac_review_flags": float(
                    part["mixed_cardiac_endothelial_review_flag"].mean()
                ),
                "fraction_relative_doublet_q99_flags": float(
                    part["doublet_percentile_ge_99"].mean()
                ),
                "reliability_review_flags": ";".join(flags) if flags else "none",
            }
        )
    return pd.DataFrame(rows), cohort_support


def make_retention_table(
    donor_summary: pd.DataFrame, thresholds: list[int]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", donor_summary)]
    for cohort, part in donor_summary.groupby("cohort", observed=True, sort=True):
        scopes.append(("cohort", str(cohort), part))
    for dataset, part in donor_summary.groupby("dataset_folder", observed=True, sort=True):
        scopes.append(("dataset", str(dataset), part))
    for scope_type, scope, part in scopes:
        n_total = int(len(part))
        cells_total = int(part["n_final_LECs"].sum())
        for threshold in thresholds:
            kept = part.loc[part["n_final_LECs"].ge(threshold)]
            cells_kept = int(kept["n_final_LECs"].sum())
            rows.append(
                {
                    "scope_type": scope_type,
                    "scope": scope,
                    "minimum_LECs": int(threshold),
                    "n_donors_total": n_total,
                    "n_donors_retained": int(len(kept)),
                    "donor_fraction_retained": float(len(kept) / n_total),
                    "n_LEC_cells_total": cells_total,
                    "n_LEC_cells_retained": cells_kept,
                    "cell_fraction_retained": float(cells_kept / cells_total),
                }
            )
    return pd.DataFrame(rows)


def make_reference_coverage(
    donor_summary: pd.DataFrame, reference_min: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort, part in donor_summary.groupby("cohort", observed=True, sort=True):
        reference = part.loc[part["n_final_LECs"].ge(reference_min)]
        rows.append(
            {
                "cohort": str(cohort),
                "dataset_folder": str(part["dataset_folder"].iloc[0]),
                "condition": str(part["condition"].iloc[0]),
                "n_donors": int(len(part)),
                "n_reference_donors": int(len(reference)),
                "reference_donor_fraction": float(len(reference) / len(part)),
                "minimum_donor_LECs": int(part["n_final_LECs"].min()),
                "median_donor_LECs": float(part["n_final_LECs"].median()),
                "maximum_donor_LECs": int(part["n_final_LECs"].max()),
                "reference_min_LECs": int(reference_min),
                "direct_resampling_support": "yes" if len(reference) else "no",
            }
        )
    return pd.DataFrame(rows)


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def run_resampling(
    donor_summary: pd.DataFrame,
    donor_state_long: pd.DataFrame,
    states: list[str],
    reference_min: int,
    n_values: list[int],
    n_resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_meta = donor_summary.loc[
        donor_summary["n_final_LECs"].ge(reference_min)
    ].copy()
    if reference_meta.empty:
        raise ValueError(f"No donors meet reference_min_LECs={reference_min}.")
    count_matrix = (
        donor_state_long.pivot(index="donor_key", columns="LEC_state", values="n_state_LECs")
        .reindex(columns=states, fill_value=0)
        .astype(np.int64)
    )
    rng = np.random.default_rng(seed)
    run_parts: list[pd.DataFrame] = []
    donor_summary_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    state_errors: dict[int, list[np.ndarray]] = {n: [] for n in n_values}
    meta = reference_meta.set_index("donor_key")
    for donor_order, donor in enumerate(reference_meta["donor_key"].astype(str), start=1):
        full_counts = count_matrix.loc[donor, states].to_numpy(dtype=np.int64)
        full_total = int(full_counts.sum())
        full_proportions = full_counts / float(full_total)
        full_present = full_counts > 0
        full_top = np.flatnonzero(full_counts == full_counts.max())
        for n_cells in n_values:
            draws = rng.multivariate_hypergeometric(full_counts, n_cells, size=n_resamples)
            sampled_proportions = draws / float(n_cells)
            absolute = np.abs(sampled_proportions - full_proportions)
            absolute_pp = absolute * 100.0
            tv_pct = 0.5 * absolute_pp.sum(axis=1)
            rmse_pp = np.sqrt(np.mean(np.square(absolute_pp), axis=1))
            max_abs_pp = absolute_pp.max(axis=1)
            observed_state_recall = (
                ((draws > 0) & full_present).sum(axis=1) / int(full_present.sum())
            )
            row_max = draws.max(axis=1)
            top_state_agreement = (draws[:, full_top] == row_max[:, None]).any(axis=1)
            state_errors[n_cells].append(absolute_pp)
            run_parts.append(
                pd.DataFrame(
                    {
                        "donor_key": donor,
                        "donor_order": donor_order,
                        "dataset_folder": meta.at[donor, "dataset_folder"],
                        "condition": meta.at[donor, "condition"],
                        "cohort": meta.at[donor, "cohort"],
                        "full_donor_LECs": full_total,
                        "n_downsample": int(n_cells),
                        "resample_run": np.arange(1, n_resamples + 1, dtype=int),
                        "tv_error_pct": tv_pct,
                        "rmse_state_percentage_points": rmse_pp,
                        "max_absolute_state_error_percentage_points": max_abs_pp,
                        "observed_state_recall": observed_state_recall,
                        "top_state_agreement": top_state_agreement,
                    }
                )
            )
            donor_row: dict[str, object] = {
                "donor_key": donor,
                "dataset_folder": meta.at[donor, "dataset_folder"],
                "condition": meta.at[donor, "condition"],
                "cohort": meta.at[donor, "cohort"],
                "full_donor_LECs": full_total,
                "n_downsample": int(n_cells),
                "n_resamples": int(n_resamples),
            }
            for metric, values in (
                ("tv_error_pct", tv_pct),
                ("rmse_state_percentage_points", rmse_pp),
                ("max_absolute_state_error_percentage_points", max_abs_pp),
            ):
                donor_row[f"{metric}_median"] = quantile(values, 0.50)
                donor_row[f"{metric}_p90"] = quantile(values, 0.90)
                donor_row[f"{metric}_p95"] = quantile(values, 0.95)
            donor_row["observed_state_recall_mean"] = float(observed_state_recall.mean())
            donor_row["top_state_agreement_fraction"] = float(top_state_agreement.mean())
            donor_summary_rows.append(donor_row)

            for state_index, state in enumerate(states):
                if not full_present[state_index]:
                    continue
                detection_rows.append(
                    {
                        "donor_key": donor,
                        "dataset_folder": meta.at[donor, "dataset_folder"],
                        "condition": meta.at[donor, "condition"],
                        "cohort": meta.at[donor, "cohort"],
                        "LEC_state": state,
                        "full_state_LECs": int(full_counts[state_index]),
                        "full_state_fraction": float(full_proportions[state_index]),
                        "n_downsample": int(n_cells),
                        "n_resamples": int(n_resamples),
                        "state_detection_probability": float(
                            (draws[:, state_index] > 0).mean()
                        ),
                    }
                )

    runs = pd.concat(run_parts, ignore_index=True)
    detection = pd.DataFrame(detection_rows)
    by_donor = pd.DataFrame(donor_summary_rows)

    threshold_rows: list[dict[str, object]] = []
    for n_cells, part in runs.groupby("n_downsample", observed=True, sort=True):
        donor_frame = by_donor.loc[by_donor["n_downsample"].eq(n_cells)]
        row: dict[str, object] = {
            "n_downsample": int(n_cells),
            "n_reference_donors": int(part["donor_key"].nunique()),
            "n_resamples_per_donor": int(part.groupby("donor_key").size().iloc[0]),
            "n_donor_resample_errors": int(len(part)),
            "mean_observed_state_recall": float(part["observed_state_recall"].mean()),
            "median_observed_state_recall": float(part["observed_state_recall"].median()),
            "top_state_agreement_fraction": float(part["top_state_agreement"].mean()),
        }
        for metric in (
            "tv_error_pct",
            "rmse_state_percentage_points",
            "max_absolute_state_error_percentage_points",
        ):
            values = part[metric].to_numpy(float)
            for label, probability in (("median", 0.50), ("p90", 0.90), ("p95", 0.95)):
                row[f"{metric}_{label}"] = quantile(values, probability)
                row[f"median_across_donors_of_donor_{metric}_{label}"] = quantile(
                    donor_frame[f"{metric}_{label}"].to_numpy(float), 0.50
                )
        # Backward-compatible fractional aliases used by the first disease 08.2 draft.
        row["median_total_variation_distance"] = row["tv_error_pct_median"] / 100.0
        row["p90_total_variation_distance"] = row["tv_error_pct_p90"] / 100.0
        row["p95_total_variation_distance"] = row["tv_error_pct_p95"] / 100.0
        row["median_max_absolute_state_error"] = (
            row["max_absolute_state_error_percentage_points_median"] / 100.0
        )
        row["p90_max_absolute_state_error"] = (
            row["max_absolute_state_error_percentage_points_p90"] / 100.0
        )
        row["p95_max_absolute_state_error"] = (
            row["max_absolute_state_error_percentage_points_p95"] / 100.0
        )
        threshold_rows.append(row)
    by_threshold = pd.DataFrame(threshold_rows)

    state_rows: list[dict[str, object]] = []
    for n_cells in n_values:
        errors = np.concatenate(state_errors[n_cells], axis=0)
        for state_index, state in enumerate(states):
            values = errors[:, state_index]
            state_rows.append(
                {
                    "n_downsample": int(n_cells),
                    "LEC_state": state,
                    "n_reference_donor_runs": int(len(values)),
                    "absolute_error_percentage_points_median": quantile(values, 0.50),
                    "absolute_error_percentage_points_p90": quantile(values, 0.90),
                    "absolute_error_percentage_points_p95": quantile(values, 0.95),
                }
            )
    state_absolute_errors = pd.DataFrame(state_rows)
    return runs, by_donor, detection, by_threshold, state_absolute_errors


def make_state_detection_summary(detection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (state, n_cells), part in detection.groupby(
        ["LEC_state", "n_downsample"], observed=True, sort=True
    ):
        rows.append(
            {
                "LEC_state": str(state),
                "n_downsample": int(n_cells),
                "n_reference_donors_with_state": int(part["donor_key"].nunique()),
                "median_full_state_LECs": float(part["full_state_LECs"].median()),
                "median_full_state_fraction": float(part["full_state_fraction"].median()),
                "mean_state_detection_probability": float(
                    part["state_detection_probability"].mean()
                ),
                "median_state_detection_probability": float(
                    part["state_detection_probability"].median()
                ),
                "p10_state_detection_probability": float(
                    part["state_detection_probability"].quantile(0.10)
                ),
            }
        )
    return pd.DataFrame(rows)


def make_candidate_threshold_lookup(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for quantile_name in ("p90", "p95"):
        column = f"tv_error_pct_{quantile_name}"
        for tolerance in TOLERANCE_GRID:
            passing = summary.loc[summary[column].le(tolerance)].sort_values("n_downsample")
            rows.append(
                {
                    "tv_error_quantile": quantile_name,
                    "maximum_allowed_tv_error_pct": tolerance,
                    "minimum_tested_N_meeting_target": (
                        int(passing.iloc[0]["n_downsample"]) if not passing.empty else np.nan
                    ),
                    "target_reached_in_tested_grid": bool(not passing.empty),
                }
            )
    return pd.DataFrame(rows)


def get_candidate_minimum(
    summary: pd.DataFrame, target_quantile: str, target_error_pct: float
) -> int | None:
    column = f"tv_error_pct_{target_quantile}"
    passing = summary.loc[summary[column].le(target_error_pct)].sort_values("n_downsample")
    return int(passing.iloc[0]["n_downsample"]) if not passing.empty else None


def make_threshold_tradeoff(
    retention: pd.DataFrame, resampling_by_threshold: pd.DataFrame
) -> pd.DataFrame:
    overall = retention.loc[retention["scope_type"].eq("overall")].copy()
    result = resampling_by_threshold.merge(
        overall,
        left_on="n_downsample",
        right_on="minimum_LECs",
        how="left",
        validate="one_to_one",
    )
    result["interpretation"] = (
        "Descriptive tradeoff only; no threshold is selected automatically. "
        "Resampling estimates come from the fixed high-depth reference set."
    )
    return result


def save_donor_support_figure(
    donor_summary: pd.DataFrame,
    retention: pd.DataFrame,
    reference_coverage: pd.DataFrame,
    cohort_order: list[str],
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    rng = np.random.default_rng(8022026)
    labels = [value.replace("dataset", "D") for value in cohort_order]

    ax = axes[0, 0]
    for x, cohort in enumerate(cohort_order):
        part = donor_summary.loc[donor_summary["cohort"].eq(cohort)]
        values = part["n_final_LECs"].to_numpy(float)
        condition = str(part["condition"].iloc[0])
        ax.scatter(
            x + rng.uniform(-0.16, 0.16, len(values)),
            values,
            s=34,
            color=CONDITION_COLORS.get(condition, "#777777"),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
        )
    for threshold, style in [(5, "--"), (10, ":"), (25, "-.")]:
        ax.axhline(threshold, color="#444444", lw=1.1, ls=style, label=f"{threshold} LECs")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("Final LECs per donor (log scale)")
    ax.set_title("All 95 final-LEC-positive donors")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    cohort_retention = retention.loc[retention["scope_type"].eq("cohort")]
    matrix = (
        cohort_retention.pivot(index="scope", columns="minimum_LECs", values="donor_fraction_retained")
        .reindex(index=cohort_order)
    )
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Candidate minimum final LECs per donor")
    ax.set_title("Fraction of donors retained in each cohort")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            ax.text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.58 else "black")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Donor fraction retained")

    ax = axes[1, 0]
    overall = retention.loc[retention["scope_type"].eq("overall")].sort_values("minimum_LECs")
    ax.plot(
        overall["minimum_LECs"],
        overall["donor_fraction_retained"],
        marker="o",
        lw=2,
        color="#0072B2",
        label="Donors retained",
    )
    ax.plot(
        overall["minimum_LECs"],
        overall["cell_fraction_retained"],
        marker="s",
        lw=2,
        color="#D55E00",
        label="LEC cells retained",
    )
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Candidate minimum final LECs per donor")
    ax.set_ylabel("Fraction retained")
    ax.set_title("Overall donor-versus-cell retention tradeoff")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    coverage = reference_coverage.set_index("cohort").reindex(cohort_order)
    x = np.arange(len(cohort_order))
    bars = ax.bar(
        x,
        coverage["n_reference_donors"],
        color=[CONDITION_COLORS.get(str(value), "#777777") for value in coverage["condition"]],
    )
    for bar, total in zip(bars, coverage["n_donors"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}/{int(total)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Reference donors")
    reference_min = int(coverage["reference_min_LECs"].iloc[0])
    ax.set_title(f"Direct resampling references (at least {reference_min} LECs)")
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Disease Step 08.2: donor support before applying any threshold", fontsize=17)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_state_support_figure(
    state_summary: pd.DataFrame,
    cohort_support: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    cohort_order: list[str],
    path: Path,
    dpi: int,
) -> None:
    summary = state_summary.set_index("LEC_state").reindex(states)
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    x = np.arange(len(states))
    colors = [palette[state] for state in states]

    ax = axes[0, 0]
    bars = ax.bar(x, summary["n_LEC_cells"], color=colors)
    for bar, donors in zip(bars, summary["n_donors_with_state"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(donors)} donors",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, states, rotation=35, ha="right")
    color_state_ticks(ax, palette)
    ax.set_ylabel("Final LEC cells")
    ax.set_title("Cell count and donor support per state")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    ax.bar(x, summary["top_donor_fraction"], color=colors)
    ax.axhline(0.50, color="#D55E00", ls="--", label="50% from one donor")
    for index, donor in enumerate(summary["top_donor"]):
        ax.text(index, summary.iloc[index]["top_donor_fraction"], str(donor).split("::")[-1],
                ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_ylim(0, max(0.58, float(summary["top_donor_fraction"].max()) * 1.18))
    ax.set_xticks(x, states, rotation=35, ha="right")
    color_state_ticks(ax, palette)
    ax.set_ylabel("Largest donor fraction of state")
    ax.set_title("Donor dominance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    cohort_matrix = (
        cohort_support.pivot(index="cohort", columns="LEC_state", values="fraction_of_state")
        .reindex(index=cohort_order, columns=states)
        .fillna(0)
    )
    image = ax.imshow(cohort_matrix.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(states)), states, rotation=35, ha="right")
    color_state_ticks(ax, palette)
    ax.set_yticks(range(len(cohort_order)), [x.replace("dataset", "D") for x in cohort_order])
    ax.set_title("Fraction of each state contributed by each cohort")
    for row in range(cohort_matrix.shape[0]):
        for column in range(cohort_matrix.shape[1]):
            value = cohort_matrix.iat[row, column]
            if value >= 0.05:
                ax.text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if value > 0.55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Fraction of state cells")

    ax = axes[1, 1]
    ax.bar(x, summary["max_leave_one_donor_out_change_percentage_points"], color=colors)
    for index, donor in enumerate(summary["most_influential_donor_cell_pooled"]):
        value = summary.iloc[index]["max_leave_one_donor_out_change_percentage_points"]
        ax.text(index, value, str(donor).split("::")[-1], ha="center", va="bottom",
                fontsize=7, rotation=45)
    ax.set_xticks(x, states, rotation=35, ha="right")
    color_state_ticks(ax, palette)
    ax.set_ylabel("Maximum absolute change (percentage points)")
    ax.set_title("Pooled state composition after leaving out one donor")
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Disease Step 08.2: LEC-state support and dominance", fontsize=17)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_resampling_figure(
    by_threshold: pd.DataFrame,
    detection_summary: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    target_quantile: str,
    target_error_pct: float,
    candidate_minimum: int | None,
    path: Path,
    dpi: int,
) -> None:
    summary = by_threshold.sort_values("n_downsample")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    x = summary["n_downsample"].to_numpy()

    ax = axes[0, 0]
    for column, label, color, marker in [
        ("tv_error_pct_median", "Median", "#0072B2", "o"),
        ("tv_error_pct_p90", "P90", "#D55E00", "s"),
        ("tv_error_pct_p95", "P95", "#D62728", "^"),
    ]:
        ax.plot(x, summary[column], marker=marker, lw=2, color=color, label=label)
    ax.axhline(target_error_pct, color="#555555", ls="--", lw=1.3,
               label=f"target: {target_quantile.upper()} <= {target_error_pct:g}%")
    if candidate_minimum is not None:
        ax.axvline(candidate_minimum, color="#009E73", ls=":", lw=1.7)
        ax.annotate(
            f"candidate N={candidate_minimum}",
            xy=(candidate_minimum, target_error_pct),
            xytext=(5, 12),
            textcoords="offset points",
            color="#007A58",
        )
    ax.set_ylabel("Total-variation error (%)")
    ax.set_title("Whole-composition error quantiles (P95 is not a CI)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    for column, label, color, marker in [
        ("max_absolute_state_error_percentage_points_median", "Median", "#009E73", "o"),
        ("max_absolute_state_error_percentage_points_p90", "P90", "#CC79A7", "s"),
        ("max_absolute_state_error_percentage_points_p95", "P95", "#8C564B", "^"),
    ]:
        ax.plot(x, summary[column], marker=marker, lw=2, color=color, label=label)
    ax.set_ylabel("Largest state error (percentage points)")
    ax.set_title("Worst state-proportion error within a run")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.plot(x, summary["mean_observed_state_recall"], marker="o", lw=2,
            color="#56B4E9", label="Mean observed-state recall")
    ax.plot(x, summary["top_state_agreement_fraction"], marker="s", lw=2,
            color="#E69F00", label="Top-state agreement")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Final LECs sampled per reference donor")
    ax.set_ylabel("Fraction")
    ax.set_title("State recovery and dominant-state agreement")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    matrix = (
        detection_summary.pivot(
            index="LEC_state", columns="n_downsample", values="mean_state_detection_probability"
        )
        .reindex(index=states, columns=x)
    )
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(x)), x)
    ax.set_yticks(range(len(states)), states)
    color_state_ticks(ax, palette, axis="y")
    ax.set_xlabel("Final LECs sampled per reference donor")
    ax.set_title("Mean detection probability when state exists in donor")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if value < 0.45 else "black")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Detection probability")
    reference_n = int(summary["n_reference_donors"].iloc[0])
    fig.suptitle(
        f"Disease Step 08.2: {reference_n} fixed reference donors; "
        "median/P90/P95 match aging Step 08.6",
        fontsize=17,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_cohort_state_figure(
    cohort_support: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    cohort_order: list[str],
    donor_summary: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    cell_matrix = (
        cohort_support.pivot(index="cohort", columns="LEC_state", values="fraction_within_cohort")
        .reindex(index=cohort_order, columns=states)
        .fillna(0)
    )
    donor_matrix = (
        cohort_support.pivot(index="cohort", columns="LEC_state", values="n_donors_with_state")
        .reindex(index=cohort_order, columns=states)
        .fillna(0)
    )
    cohort_donors = donor_summary.groupby("cohort", observed=True).size().reindex(cohort_order)
    donor_fraction = donor_matrix.div(cohort_donors, axis=0)
    labels = [x.replace("dataset", "D") for x in cohort_order]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), constrained_layout=True)
    for ax, matrix, title, color_map in [
        (axes[0], cell_matrix, "Cell-pooled state composition within cohort", "Blues"),
        (axes[1], donor_fraction, "Fraction of cohort donors containing each state", "Greens"),
    ]:
        image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap=color_map, aspect="auto")
        ax.set_xticks(range(len(states)), states, rotation=35, ha="right")
        color_state_ticks(ax, palette)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iat[row, column]
                ax.text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if value > 0.55 else "black")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Disease Step 08.2: descriptive cohort-by-state support", fontsize=17)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    path: Path,
    input_h5ad: Path,
    optional_qc: Path,
    optional_qc_status: str,
    n_cells: int,
    n_donors: int,
    reference_min: int,
    n_reference: int,
    n_values: list[int],
    n_resamples: int,
    target_quantile: str,
    target_error_pct: float,
    candidate_minimum: int | None,
    reference_coverage: pd.DataFrame,
) -> None:
    unsupported = reference_coverage.loc[
        reference_coverage["n_reference_donors"].eq(0), "cohort"
    ].tolist()
    candidate_text = (
        str(candidate_minimum)
        if candidate_minimum is not None
        else "not reached within the tested N grid"
    )
    path.write_text(
        f"""Disease Step 08.2 donor/state reliability
===============================================

Input and scope
---------------
Input h5ad (read only): {input_h5ad}
Final Step 08 LEC cells: {n_cells}
Final-LEC-positive donors: {n_donors}
Non-LEC cells analyzed: 0
Cells/donors removed: 0
Clustering, states, expression, and embeddings changed: no

Optional Step 08.1 QC table: {optional_qc}
Optional QC status: {optional_qc_status}
The optional table contributes descriptive relative cardiac and doublet-tail
flags only. These flags never remove cells or define the reliability threshold.

Primary resampling design
-------------------------
Fixed reference criterion: at least {reference_min} final LECs
Fixed reference donors: {n_reference}
Tested N values: {', '.join(map(str, n_values))}
Independent samples per donor per N: {n_resamples}
Sampling: without replacement within each donor

For every reference donor, the donor's complete observed state composition is
treated as the internal reference. Downsampled composition is compared with it
using total-variation error, RMSE, maximum absolute state error, observed-state
recall, and dominant-state agreement. Every reference donor contributes the
same number of runs at every N.

Alignment with aging Step 08.6
------------------------------
The primary TV, RMSE, and maximum-error definitions, N grid, 2,000-repeat
default, equal-donor weighting, and median/P90/P95 summaries match aging Step
08.6. P95 is the 95th percentile of resampling error; it is NOT a 95%
confidence interval. A formal CI would require an additional donor bootstrap.

Descriptive candidate rule
--------------------------
Configured target: {target_quantile.upper()} TV <= {target_error_pct:g}%.
Smallest tested N meeting that target: {candidate_text}.
This is a transparent reliability tolerance matching aging Step 08.6, not a
biological law and not an automatic donor-exclusion instruction.

Important limitations
---------------------
- The donor's complete observed composition is not biological ground truth; it
  is only the best available internal reference.
- High-depth reference donors may differ from low-depth donors.
- Cohorts with no direct >= {reference_min}-LEC reference donors are:
  {('; '.join(unsupported)) if unsupported else 'none'}.
  Reliability should not be silently extrapolated to unsupported cohorts.
- Candidate threshold tables describe donor and cell retention. A candidate N
  may be reported, but no donor is removed automatically.
- Cells are technical observations. Disease comparisons must use donors as
  biological replicates.
- Condition is coupled to dataset. Subsequent comparisons should be performed
  within dataset (dataset01: NF/DCM/HCM; dataset08: NF/DoxCM; dataset11:
  NF/HFpEF), not as one pooled cross-dataset disease contrast.
- Relative doublet percentiles from Step 08.1 are ranks, not confirmed doublet
  calls. They must not be used as automatic exclusion criteria.
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    n_values = sorted(set(args.n_values))
    thresholds = sorted(set(args.retention_thresholds))
    if any(value < 2 for value in n_values):
        raise ValueError("Every --n_values entry must be at least 2.")
    if max(n_values) > args.reference_min_LECs:
        raise ValueError("max(n_values) must be <= reference_min_LECs.")
    if any(value < 1 for value in thresholds):
        raise ValueError("Every retention threshold must be at least 1.")
    if args.n_resamples < 100:
        raise ValueError("--n_resamples must be at least 100 for stable tail summaries.")
    if args.target_tv_error_pct <= 0:
        raise ValueError("--target_tv_error_pct must be positive.")
    if not 0 < args.donor_dominance_threshold <= 1:
        raise ValueError("--donor_dominance_threshold must be in (0, 1].")
    if not 0 < args.dataset_dominance_threshold <= 1:
        raise ValueError("--dataset_dominance_threshold must be in (0, 1].")

    project_dir = find_project_dir()
    input_h5ad = resolve_path(args.input_h5ad, project_dir)
    optional_qc = resolve_path(args.step08_1_cell_qc, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    if not input_h5ad.is_file():
        raise FileNotFoundError(f"Step 08 h5ad does not exist: {input_h5ad}")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {output_root}. Pass --overwrite.")
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    print(f"Reading final Step 08 LEC object: {input_h5ad}")
    adata = ad.read_h5ad(input_h5ad, backed="r")
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    validate_obs(obs)
    obs["dataset_folder"] = normalized_text(obs["dataset_folder"])
    obs["condition"] = normalized_text(obs["condition"])
    obs["donor_key"] = normalized_text(obs["donor_key"])
    obs["sample_key"] = normalized_text(obs["sample_key"])
    obs["LEC_state"] = normalized_text(obs["LEC_state"])
    obs["cohort"] = obs["dataset_folder"] + " | " + obs["condition"]
    if obs["donor_key"].eq("").any():
        raise ValueError("Every final LEC must have a donor_key.")
    states = sorted(obs["LEC_state"].unique(), key=natural_key)
    palette = state_palette(adata, states)
    adata.file.close()

    obs, optional_qc_status = attach_optional_step08_1_qc(obs, optional_qc)
    donor_summary, donor_counts, donor_state_long = make_donor_tables(obs, states)
    cohort_order = (
        donor_summary[["dataset_folder", "condition", "cohort"]]
        .drop_duplicates()
        .sort_values(["dataset_folder", "condition"])["cohort"]
        .tolist()
    )
    leave_one_out = make_leave_one_donor_out(donor_state_long, states)
    state_summary, cohort_support = make_state_support(
        obs,
        donor_state_long,
        leave_one_out,
        states,
        args.donor_dominance_threshold,
        args.dataset_dominance_threshold,
    )
    retention = make_retention_table(donor_summary, thresholds)
    reference_coverage = make_reference_coverage(donor_summary, args.reference_min_LECs)

    print(
        f"Resampling donors with >= {args.reference_min_LECs} final LECs "
        f"at N={n_values}; {args.n_resamples} runs per donor/N."
    )
    (
        runs,
        resampling_by_donor,
        detection,
        resampling_by_threshold,
        state_absolute_errors,
    ) = run_resampling(
        donor_summary,
        donor_state_long,
        states,
        args.reference_min_LECs,
        n_values,
        args.n_resamples,
        args.seed,
    )
    detection_summary = make_state_detection_summary(detection)
    candidate_lookup = make_candidate_threshold_lookup(resampling_by_threshold)
    candidate_minimum = get_candidate_minimum(
        resampling_by_threshold, args.target_quantile, args.target_tv_error_pct
    )
    threshold_tradeoff = make_threshold_tradeoff(retention, resampling_by_threshold)
    reference_donors = donor_summary.loc[
        donor_summary["n_final_LECs"].ge(args.reference_min_LECs)
    ].copy()

    donor_summary.to_csv(tables_dir / "08_2_donor_support_summary.csv", index=False)
    donor_counts.to_csv(tables_dir / "08_2_donor_state_counts.csv", index=False)
    donor_state_long.to_csv(tables_dir / "08_2_donor_state_composition_long.csv", index=False)
    state_summary.to_csv(tables_dir / "08_2_state_support_and_dominance.csv", index=False)
    cohort_support.to_csv(tables_dir / "08_2_state_support_by_dataset_condition.csv", index=False)
    leave_one_out.to_csv(
        tables_dir / "08_2_leave_one_donor_out_state_sensitivity.csv.gz",
        index=False,
        compression="gzip",
    )
    retention.to_csv(tables_dir / "08_2_candidate_threshold_retention.csv", index=False)
    reference_donors.to_csv(tables_dir / "08_2_resampling_reference_donors.csv", index=False)
    reference_coverage.to_csv(tables_dir / "08_2_resampling_reference_cohort_coverage.csv", index=False)
    runs.to_csv(
        tables_dir / "08_2_resampling_runs.csv.gz", index=False, compression="gzip"
    )
    resampling_by_donor.to_csv(
        tables_dir / "08_2_resampling_summary_by_donor_and_N.csv", index=False
    )
    resampling_by_threshold.to_csv(
        tables_dir / "08_2_resampling_summary_by_N.csv", index=False
    )
    # Aging-compatible names make the two reliability branches easy to compare.
    resampling_by_threshold.to_csv(
        tables_dir / "08_2_error_summary_equal_donor_pooled.csv", index=False
    )
    state_absolute_errors.to_csv(
        tables_dir / "08_2_state_absolute_error_summary.csv", index=False
    )
    candidate_lookup.to_csv(
        tables_dir / "08_2_candidate_minimum_threshold_lookup.csv", index=False
    )
    detection.to_csv(
        tables_dir / "08_2_state_detection_probability_by_donor.csv.gz",
        index=False,
        compression="gzip",
    )
    detection_summary.to_csv(
        tables_dir / "08_2_state_detection_summary_by_N.csv", index=False
    )
    threshold_tradeoff.to_csv(
        tables_dir / "08_2_threshold_reliability_retention_tradeoff.csv", index=False
    )
    pd.DataFrame(
        {"LEC_state": states, "color": [palette[state] for state in states]}
    ).to_csv(tables_dir / "08_2_state_color_mapping.csv", index=False)

    input_validation = pd.DataFrame(
        [
            {
                "input_h5ad": str(input_h5ad),
                "n_final_LEC_cells": int(len(obs)),
                "n_final_LEC_positive_donors": int(obs["donor_key"].nunique()),
                "n_zero_LEC_donors": 0,
                "n_non_LEC_cells": 0,
                "n_states": int(len(states)),
                "datasets": ";".join(sorted(obs["dataset_folder"].unique())),
                "conditions": ";".join(sorted(obs["condition"].unique())),
                "optional_step08_1_qc_status": optional_qc_status,
                "status": "validated_final_Step08_LECs_only",
            }
        ]
    )
    input_validation.to_csv(tables_dir / "08_2_input_validation.csv", index=False)

    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        jobs = [
            (
                "08_2_donor_support_and_threshold_retention.png",
                "Donor LEC support, cohort retention, and high-depth reference coverage",
                lambda path: save_donor_support_figure(
                    donor_summary, retention, reference_coverage, cohort_order, path, args.dpi
                ),
            ),
            (
                "08_2_state_support_and_dominance.png",
                "State cell/donor support, cohort contribution, and donor influence",
                lambda path: save_state_support_figure(
                    state_summary, cohort_support, states, palette, cohort_order, path, args.dpi
                ),
            ),
            (
                "08_2_resampling_reliability.png",
                "Matched-reference within-donor downsampling reliability",
                lambda path: save_resampling_figure(
                    resampling_by_threshold,
                    detection_summary,
                    states,
                    palette,
                    args.target_quantile,
                    args.target_tv_error_pct,
                    candidate_minimum,
                    path,
                    args.dpi,
                ),
            ),
            (
                "08_2_cohort_state_support.png",
                "Cell-pooled state composition and donor presence by cohort",
                lambda path: save_cohort_state_figure(
                    cohort_support, states, palette, cohort_order, donor_summary, path, args.dpi
                ),
            ),
        ]
        for filename, description, function in jobs:
            try:
                function(figures_dir / filename)
                figure_manifest.append(
                    {"figure": filename, "description": description, "status": "ok"}
                )
            except Exception as error:
                warnings.warn(f"Figure failed ({filename}): {type(error).__name__}: {error}")
                figure_manifest.append(
                    {
                        "figure": filename,
                        "description": description,
                        "status": f"failed: {type(error).__name__}: {error}",
                    }
                )
    pd.DataFrame(figure_manifest).to_csv(output_root / "08_2_figure_manifest.csv", index=False)

    write_notes(
        output_root / "08_2_method_and_interpretation_notes.txt",
        input_h5ad,
        optional_qc,
        optional_qc_status,
        len(obs),
        obs["donor_key"].nunique(),
        args.reference_min_LECs,
        len(reference_donors),
        n_values,
        args.n_resamples,
        args.target_quantile,
        args.target_tv_error_pct,
        candidate_minimum,
        reference_coverage,
    )

    run_config = {
        "script_version": SCRIPT_VERSION,
        "input_h5ad": str(input_h5ad),
        "optional_step08_1_cell_qc": str(optional_qc),
        "optional_step08_1_qc_status": optional_qc_status,
        "output_root": str(output_root),
        "reference_min_LECs": args.reference_min_LECs,
        "n_values": n_values,
        "retention_thresholds": thresholds,
        "n_resamples_per_reference_donor_per_N": args.n_resamples,
        "seed": args.seed,
        "target_quantile": args.target_quantile,
        "target_tv_error_pct": args.target_tv_error_pct,
        "descriptive_candidate_minimum_N": candidate_minimum,
        "p95_definition": "95th percentile of resampling error; not a confidence interval",
        "donor_dominance_threshold": args.donor_dominance_threshold,
        "dataset_dominance_threshold": args.dataset_dominance_threshold,
        "cell_scope": "exact_current_Step08_final_LECs_only",
        "automatic_donor_exclusion": False,
        "cells_or_donors_removed": False,
        "h5ad_written": False,
    }
    (output_root / "08_2_run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    high_donor_states = state_summary.loc[
        state_summary["top_donor_fraction"].ge(args.donor_dominance_threshold), "LEC_state"
    ].tolist()
    unsupported_cohorts = reference_coverage.loc[
        reference_coverage["n_reference_donors"].eq(0), "cohort"
    ].tolist()
    run_summary = pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "status": "completed",
                "cell_scope": "exact_current_Step08_final_LECs_only",
                "n_final_LEC_cells": int(len(obs)),
                "n_final_LEC_positive_donors": int(obs["donor_key"].nunique()),
                "n_states": int(len(states)),
                "n_reference_donors": int(len(reference_donors)),
                "reference_min_LECs": int(args.reference_min_LECs),
                "n_resampling_runs": int(len(runs)),
                "target_quantile": args.target_quantile,
                "target_tv_error_pct": float(args.target_tv_error_pct),
                "descriptive_candidate_minimum_N": candidate_minimum,
                "donor_dominant_states": ";".join(high_donor_states) if high_donor_states else "none",
                "cohorts_without_direct_resampling_reference": (
                    ";".join(unsupported_cohorts) if unsupported_cohorts else "none"
                ),
                "automatic_donor_exclusion": False,
                "cells_or_donors_removed": False,
                "output_root": str(output_root),
            }
        ]
    )
    run_summary.to_csv(output_root / "08_2_run_summary.csv", index=False)

    print("\nCompleted disease Step 08.2 donor/state reliability analysis.")
    print(run_summary.to_string(index=False))
    print("\nResampling summary:")
    print(resampling_by_threshold.to_string(index=False))
    print(
        f"\nDescriptive candidate: {args.target_quantile.upper()} TV <= "
        f"{args.target_tv_error_pct:g}% -> "
        f"{candidate_minimum if candidate_minimum is not None else 'not reached'}."
    )
    print("No cells or donors were removed automatically.")
    print(f"Output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
