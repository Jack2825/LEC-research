#!/usr/bin/env python3
r"""
Disease Step 08.3: descriptive sensitivity analysis after requiring at least
25 final Step 08 LECs per donor.

This is the disease counterpart of the aging minimum-25-LEC filtered branch.
It is deliberately read-only: the original Step 08 h5ad, cells, expression,
embedding, clusters, and LEC_state labels are not modified. A donor is retained
when the total number of final Step 08 LECs assigned to that donor is >=25.

The script exports exact cell/donor filter mappings, donor-level state
percentages including zero-count donor-state pairs, state/cohort support,
donor-dominance summaries, and within-dataset comparison readiness. It does not
perform disease-association inference or substitute cell-level tests when a
comparison loses donor support.

Default input
-------------
new_result2_disease/08_combined_LEC_recluster/
    08_combined_final_LEC_reclustered.h5ad

Threshold provenance
--------------------
new_result2_disease/08_2_donor_state_reliability/
    08_2_run_summary.csv

Default output
--------------
new_result2_disease/08_3_min25_LEC_filtered_analysis/
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SCRIPT_VERSION = "08_3_disease_min25_final_LEC_filtered_sensitivity_2026_08_12d"
DEFAULT_INPUT = (
    "new_result2_disease/08_combined_LEC_recluster/"
    "08_combined_final_LEC_reclustered.h5ad"
)
DEFAULT_THRESHOLD_SUMMARY = (
    "new_result2_disease/08_2_donor_state_reliability/08_2_run_summary.csv"
)
DEFAULT_OUTPUT = "new_result2_disease/08_3_min25_LEC_filtered_analysis"
DEFAULT_MINIMUM_LECS = 25
EXPECTED_DATASETS = ["dataset01", "dataset08", "dataset11"]
EXPECTED_CONDITIONS = ["NF", "DCM", "HCM", "DoxCM", "HFpEF"]
EXPECTED_REGIONS = {"dataset01": "LV", "dataset08": "LV", "dataset11": "S"}
EXPECTED_COMPARISONS = [
    ("dataset01", "NF", "DCM"),
    ("dataset01", "NF", "HCM"),
    ("dataset08", "NF", "DoxCM"),
    ("dataset11", "NF", "HFpEF"),
]
CONDITION_COLORS = {
    "NF": "#0072B2",
    "DCM": "#D55E00",
    "HCM": "#CC79A7",
    "DoxCM": "#56B4E9",
    "HFpEF": "#E69F00",
}


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def true_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return normalized_text(values).str.casefold().isin({"true", "1", "yes", "y", "t"})


def natural_key(value: object) -> list[object]:
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", str(value))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Disease donor/state sensitivity analysis after a >=25-final-LEC filter."
    )
    parser.add_argument("--input_h5ad", default=DEFAULT_INPUT)
    parser.add_argument("--threshold_summary", default=DEFAULT_THRESHOLD_SUMMARY)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum_LECs", type=int, default=DEFAULT_MINIMUM_LECS)
    parser.add_argument(
        "--minimum_donors_per_condition",
        type=int,
        default=2,
        help="Minimum retained donors in both arms for a comparison to be marked ready.",
    )
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
    check = (
        obs.assign(
            dataset_folder=normalized_text(obs["dataset_folder"]),
            region_standardized_analysis=normalized_text(obs["region_standardized_analysis"]),
        )
        .groupby("dataset_folder", observed=True)["region_standardized_analysis"]
        .agg(lambda x: ";".join(sorted(x.unique())))
        .to_dict()
    )
    if check != EXPECTED_REGIONS:
        raise ValueError(f"Unexpected regions: {check}")


def threshold_provenance(path: Path, minimum: int) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        frame = pd.DataFrame(
            [{"threshold_summary": str(path), "status": "not_found", "minimum_LECs_used": minimum}]
        )
        return frame, "threshold_summary_not_found"
    source = pd.read_csv(path)
    if len(source) != 1:
        raise ValueError(f"Expected one row in Step 08.2 run summary: {path}")
    candidate = pd.to_numeric(source.get("descriptive_candidate_minimum_N"), errors="coerce").iloc[0]
    target_quantile = str(source.get("target_quantile", pd.Series([""])).iloc[0])
    target_error = pd.to_numeric(source.get("target_tv_error_pct"), errors="coerce").iloc[0]
    matches = bool(np.isfinite(candidate) and int(candidate) == minimum)
    frame = pd.DataFrame(
        [
            {
                "threshold_summary": str(path),
                "step08_2_script_version": source.get("script_version", pd.Series([""])).iloc[0],
                "target_quantile": target_quantile,
                "target_tv_error_pct": target_error,
                "step08_2_descriptive_candidate_minimum_N": candidate,
                "minimum_LECs_used": minimum,
                "candidate_matches_filter": matches,
                "status": "validated" if matches else "mismatch_review_required",
            }
        ]
    )
    if not matches:
        raise ValueError(
            f"Requested minimum {minimum} does not match Step 08.2 candidate {candidate}."
        )
    return frame, "validated_against_step08_2"


def make_filter_tables(
    obs: pd.DataFrame, states: list[str], minimum: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = [
        column
        for column in [
            "dataset_folder",
            "condition",
            "donor_id",
            "donor_id_source",
            "source_id",
            "sex",
            "age",
            "lvef",
            "region_standardized_analysis",
        ]
        if column in obs.columns
    ]
    donor_meta = (
        obs.reset_index(names="step08_cell_id")
        .sort_values("step08_cell_id")
        .drop_duplicates("donor_key")[["donor_key", *metadata]]
    )
    donor_counts = obs.groupby("donor_key", observed=True).size().rename("n_final_LECs")
    donor_meta = donor_meta.merge(donor_counts, on="donor_key", validate="one_to_one")
    donor_meta["cohort"] = (
        donor_meta["dataset_folder"].astype(str) + " | " + donor_meta["condition"].astype(str)
    )
    donor_meta["minimum_LECs"] = int(minimum)
    donor_meta["retained_minimum_LECs"] = donor_meta["n_final_LECs"].ge(minimum)
    donor_meta["filter_status"] = np.where(
        donor_meta["retained_minimum_LECs"], "retained", "excluded_below_minimum_LECs"
    )
    donor_meta["LECs_below_or_above_threshold"] = donor_meta["n_final_LECs"] - minimum
    donor_meta = donor_meta.sort_values(
        ["dataset_folder", "condition", "n_final_LECs", "donor_key"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    retained_donors = set(
        donor_meta.loc[donor_meta["retained_minimum_LECs"], "donor_key"].astype(str)
    )
    cell_mapping = obs[
        ["dataset_folder", "condition", "cohort", "donor_key", "LEC_state"]
    ].reset_index(names="step08_cell_id")
    cell_mapping["donor_n_final_LECs"] = cell_mapping["donor_key"].map(donor_counts)
    cell_mapping["minimum_LECs"] = int(minimum)
    cell_mapping["retained_minimum_LECs"] = cell_mapping["donor_key"].isin(retained_donors)
    cell_mapping["filter_status"] = np.where(
        cell_mapping["retained_minimum_LECs"], "retained", "excluded_with_low_LEC_donor"
    )
    retained_obs = obs.loc[obs["donor_key"].isin(retained_donors)].copy()

    counts = pd.crosstab(retained_obs["donor_key"], retained_obs["LEC_state"]).reindex(
        index=donor_meta.loc[donor_meta["retained_minimum_LECs"], "donor_key"],
        columns=states,
        fill_value=0,
    )
    counts = counts.astype(int)
    totals = counts.sum(axis=1)
    donor_state = counts.reset_index().melt(
        id_vars="donor_key", var_name="LEC_state", value_name="n_state_LECs"
    )
    donor_state = donor_state.merge(
        donor_meta.loc[
            donor_meta["retained_minimum_LECs"],
            ["donor_key", "dataset_folder", "condition", "cohort", "n_final_LECs"],
        ],
        on="donor_key",
        validate="many_to_one",
    )
    donor_state["state_percentage_within_donor"] = (
        100.0 * donor_state["n_state_LECs"] / donor_state["n_final_LECs"]
    )
    donor_state["state_present"] = donor_state["n_state_LECs"].gt(0)
    donor_state["minimum_LECs"] = int(minimum)
    return donor_meta, cell_mapping, retained_obs, counts.reset_index(), donor_state


def make_cohort_summary(donor_meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort, part in donor_meta.groupby("cohort", observed=True, sort=True):
        retained = part.loc[part["retained_minimum_LECs"]]
        rows.append(
            {
                "cohort": str(cohort),
                "dataset_folder": str(part["dataset_folder"].iloc[0]),
                "condition": str(part["condition"].iloc[0]),
                "n_donors_before": int(len(part)),
                "n_donors_retained": int(len(retained)),
                "n_donors_excluded": int(len(part) - len(retained)),
                "donor_fraction_retained": float(len(retained) / len(part)),
                "n_LEC_cells_before": int(part["n_final_LECs"].sum()),
                "n_LEC_cells_retained": int(retained["n_final_LECs"].sum()),
                "n_LEC_cells_excluded": int(
                    part["n_final_LECs"].sum() - retained["n_final_LECs"].sum()
                ),
                "LEC_cell_fraction_retained": float(
                    retained["n_final_LECs"].sum() / part["n_final_LECs"].sum()
                ),
                "minimum_retained_donor_LECs": (
                    int(retained["n_final_LECs"].min()) if len(retained) else np.nan
                ),
                "median_retained_donor_LECs": (
                    float(retained["n_final_LECs"].median()) if len(retained) else np.nan
                ),
                "maximum_retained_donor_LECs": (
                    int(retained["n_final_LECs"].max()) if len(retained) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def make_comparison_readiness(
    donor_meta: pd.DataFrame, minimum_donors_per_condition: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, reference, disease in EXPECTED_COMPARISONS:
        subset = donor_meta.loc[donor_meta["dataset_folder"].eq(dataset)]
        reference_all = subset.loc[subset["condition"].eq(reference)]
        disease_all = subset.loc[subset["condition"].eq(disease)]
        reference_kept = reference_all.loc[reference_all["retained_minimum_LECs"]]
        disease_kept = disease_all.loc[disease_all["retained_minimum_LECs"]]
        ready = (
            len(reference_kept) >= minimum_donors_per_condition
            and len(disease_kept) >= minimum_donors_per_condition
        )
        if ready:
            reason = "ready_for_future_donor_level_within_dataset_comparison"
        elif len(reference_kept) == 0:
            reason = "not_testable_no_retained_reference_donors"
        elif len(disease_kept) == 0:
            reason = "not_testable_no_retained_disease_donors"
        else:
            reason = "not_testable_fewer_than_minimum_donors_in_one_or_both_arms"
        rows.append(
            {
                "dataset_folder": dataset,
                "reference_condition": reference,
                "disease_condition": disease,
                "comparison": f"{dataset}: {disease} vs {reference}",
                "n_reference_donors_before": int(len(reference_all)),
                "n_disease_donors_before": int(len(disease_all)),
                "n_reference_donors_retained": int(len(reference_kept)),
                "n_disease_donors_retained": int(len(disease_kept)),
                "minimum_donors_per_condition": int(minimum_donors_per_condition),
                "donor_level_comparison_ready": bool(ready),
                "status": reason,
                "cell_level_test_substituted": False,
            }
        )
    return pd.DataFrame(rows)


def make_nichenet_donor_manifest(
    obs: pd.DataFrame,
    donor_meta: pd.DataFrame,
    readiness: pd.DataFrame,
    minimum: int,
) -> pd.DataFrame:
    """Create an explicit retained-donor handoff table for downstream NicheNet."""

    def unique_values(part: pd.DataFrame, column: str) -> list[str]:
        if column not in part.columns:
            return []
        values = normalized_text(part[column])
        values = values.loc[~values.str.casefold().isin({"", "nan", "none", "na", "n/a"})]
        return sorted(values.unique().tolist(), key=natural_key)

    ready_rows = readiness.loc[readiness["donor_level_comparison_ready"]].copy()
    retained = donor_meta.loc[donor_meta["retained_minimum_LECs"]].copy()
    rows: list[dict[str, object]] = []
    for donor in retained.itertuples(index=False):
        donor_key = str(donor.donor_key)
        dataset = str(donor.dataset_folder)
        condition = str(donor.condition)
        part = obs.loc[obs["donor_key"].eq(donor_key)]
        source_ids = unique_values(part, "donor_id_source")
        donor_ids = unique_values(part, "donor_id")
        original_donor_id = (
            source_ids[0]
            if len(source_ids) == 1
            else donor_ids[0]
            if len(donor_ids) == 1
            else donor_key.split("::", maxsplit=1)[-1]
        )
        memberships: list[str] = []
        for comparison in ready_rows.itertuples(index=False):
            if dataset == str(comparison.dataset_folder) and condition in {
                str(comparison.reference_condition),
                str(comparison.disease_condition),
            }:
                memberships.append(str(comparison.comparison))
        used_in_primary_test = bool(memberships)
        if used_in_primary_test and condition == "NF":
            analysis_role = "primary_within_dataset_reference"
        elif used_in_primary_test:
            analysis_role = "primary_within_dataset_disease"
        else:
            analysis_role = "descriptive_only_no_retained_within_dataset_reference"

        sample_keys = unique_values(part, "sample_key")
        sample_ids = unique_values(part, "sample_id_source")
        biosample_ids = unique_values(part, "biosample_id")
        source_accessions = unique_values(part, "source_id")
        rows.append(
            {
                "analysis_donor_key": donor_key,
                "original_donor_id": original_donor_id,
                "donor_id_source": ";".join(source_ids),
                "donor_id_obs": ";".join(donor_ids),
                "dataset_folder": dataset,
                "source_accession": ";".join(source_accessions),
                "condition": condition,
                "cohort": f"{dataset} | {condition}",
                "region_standardized_analysis": getattr(
                    donor, "region_standardized_analysis", ""
                ),
                "sex": getattr(donor, "sex", ""),
                "age": getattr(donor, "age", ""),
                "n_final_LECs_used": int(donor.n_final_LECs),
                "minimum_final_LECs_required": int(minimum),
                "passed_minimum_LEC_filter": True,
                "used_in_step08_3_composition": True,
                "used_in_step08_4_donor_level_analysis": True,
                "used_in_primary_within_dataset_test": used_in_primary_test,
                "analysis_role": analysis_role,
                "primary_comparison_membership": ";".join(memberships),
                "included_in_combined_NF_descriptive_figure": condition == "NF",
                "n_final_LEC_samples_contributing": int(
                    len(sample_keys or sample_ids or biosample_ids)
                ),
                "analysis_sample_keys": ";".join(sample_keys),
                "source_sample_ids": ";".join(sample_ids),
                "biosample_ids": ";".join(biosample_ids),
            }
        )

    result = pd.DataFrame(rows)
    reused_ids = (
        result.groupby("original_donor_id", observed=True)["dataset_folder"]
        .nunique()
        .gt(1)
    )
    result.insert(0, "recommended_join_key", result["analysis_donor_key"])
    result["original_donor_id_reused_across_datasets"] = result[
        "original_donor_id"
    ].map(reused_ids).fillna(False)
    result["join_key_note"] = (
        "Use recommended_join_key (dataset::donor); do not join across datasets "
        "on original_donor_id alone."
    )
    cohort_order = ordered_dataset_cohorts(set(result["cohort"].astype(str)))
    cohort_rank = {cohort: index for index, cohort in enumerate(cohort_order)}
    result["_cohort_order"] = result["cohort"].map(cohort_rank)
    result = result.sort_values(
        ["_cohort_order", "original_donor_id", "analysis_donor_key"]
    ).drop(columns="_cohort_order")

    if result["analysis_donor_key"].duplicated().any():
        raise RuntimeError("NicheNet donor manifest contains duplicated analysis donor keys.")
    if set(result["analysis_donor_key"]) != set(retained["donor_key"].astype(str)):
        raise RuntimeError("NicheNet donor manifest does not exactly match retained donors.")
    if result["original_donor_id"].astype(str).str.strip().eq("").any():
        raise RuntimeError("NicheNet donor manifest contains a missing original donor ID.")
    return result.reset_index(drop=True)


def make_state_filter_impact(
    obs: pd.DataFrame, retained_obs: pd.DataFrame, states: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state in states:
        before = obs.loc[obs["LEC_state"].eq(state)]
        after = retained_obs.loc[retained_obs["LEC_state"].eq(state)]
        rows.append(
            {
                "LEC_state": state,
                "n_cells_before": int(len(before)),
                "n_cells_retained": int(len(after)),
                "n_cells_excluded": int(len(before) - len(after)),
                "cell_fraction_retained": float(len(after) / len(before)) if len(before) else np.nan,
                "n_donors_before": int(before["donor_key"].nunique()),
                "n_donors_retained": int(after["donor_key"].nunique()),
                "donor_fraction_retained": float(
                    after["donor_key"].nunique() / before["donor_key"].nunique()
                ) if before["donor_key"].nunique() else np.nan,
                "cell_pooled_percentage_before": float(100.0 * len(before) / len(obs)),
                "cell_pooled_percentage_retained": float(100.0 * len(after) / len(retained_obs)),
                "cell_pooled_percentage_point_change": float(
                    100.0 * len(after) / len(retained_obs) - 100.0 * len(before) / len(obs)
                ),
            }
        )
    return pd.DataFrame(rows)


def make_cohort_state_summary(
    retained_obs: pd.DataFrame,
    donor_state: pd.DataFrame,
    states: list[str],
    cohort_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    dominance_rows: list[dict[str, object]] = []
    for cohort in cohort_order:
        cells = retained_obs.loc[retained_obs["cohort"].eq(cohort)]
        donor_part = donor_state.loc[donor_state["cohort"].eq(cohort)]
        n_donors = int(donor_part["donor_key"].nunique())
        dataset_folder, condition = cohort.split(" | ", maxsplit=1)
        for state in states:
            state_cells = cells.loc[cells["LEC_state"].eq(state)]
            state_donors = donor_part.loc[donor_part["LEC_state"].eq(state)]
            percentages = state_donors["state_percentage_within_donor"]
            rows.append(
                {
                    "cohort": cohort,
                    "dataset_folder": dataset_folder,
                    "condition": condition,
                    "LEC_state": state,
                    "n_retained_LEC_cells": int(len(state_cells)),
                    "cell_pooled_percentage_within_cohort": (
                        float(100.0 * len(state_cells) / len(cells))
                        if len(cells)
                        else np.nan
                    ),
                    "n_retained_donors": n_donors,
                    "n_donors_with_state": int(state_donors["state_present"].sum()),
                    "percent_donors_with_state": (
                        float(100.0 * state_donors["state_present"].mean())
                        if n_donors
                        else np.nan
                    ),
                    "mean_within_donor_state_percentage": (
                        float(percentages.mean()) if n_donors else np.nan
                    ),
                    "median_within_donor_state_percentage": (
                        float(percentages.median()) if n_donors else np.nan
                    ),
                    "sd_within_donor_state_percentage": float(percentages.std(ddof=1)),
                    "sem_within_donor_state_percentage": float(
                        percentages.std(ddof=1) / np.sqrt(n_donors)
                    ) if n_donors > 1 else np.nan,
                }
            )
            counts = state_cells["donor_key"].value_counts()
            if len(counts):
                top_donor = str(counts.index[0])
                top_count = int(counts.iloc[0])
                top_fraction = float(top_count / counts.sum())
            else:
                top_donor, top_count, top_fraction = "none", 0, np.nan
            dominance_rows.append(
                {
                    "cohort": cohort,
                    "LEC_state": state,
                    "n_state_LEC_cells": int(len(state_cells)),
                    "n_donors_with_state": int(len(counts)),
                    "top_donor": top_donor,
                    "top_donor_n_state_LECs": top_count,
                    "top_donor_fraction_of_cohort_state": top_fraction,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(dominance_rows)


def ordered_dataset_cohorts(cohorts: set[str]) -> list[str]:
    """Put NF first within every dataset and keep the planned disease order."""
    desired: list[str] = []
    for dataset, reference, disease in EXPECTED_COMPARISONS:
        for condition in [reference, disease]:
            cohort = f"{dataset} | {condition}"
            if cohort not in desired:
                desired.append(cohort)
    extras = sorted(set(cohorts) - set(desired), key=natural_key)
    return [cohort for cohort in desired if cohort in cohorts] + extras


def display_cohort_label(value: str) -> str:
    if value == "combined | NF":
        return "Combined NF"
    return value.replace("dataset", "D")


def make_combined_nf_state_summary(
    retained_obs: pd.DataFrame,
    donor_state: pd.DataFrame,
    states: list[str],
) -> tuple[pd.DataFrame, list[str], str]:
    """Combine retained NF donors descriptively while leaving diseases dataset-specific."""
    combined_obs = retained_obs.copy()
    combined_donor_state = donor_state.copy()
    combined_obs.loc[combined_obs["condition"].eq("NF"), "cohort"] = "combined | NF"
    combined_donor_state.loc[
        combined_donor_state["condition"].eq("NF"), "cohort"
    ] = "combined | NF"
    disease_order = [
        "dataset01 | DCM",
        "dataset01 | HCM",
        "dataset08 | DoxCM",
        "dataset11 | HFpEF",
    ]
    combined_order = ["combined | NF"] + [
        cohort
        for cohort in disease_order
        if cohort in set(combined_obs["cohort"].astype(str))
    ]
    summary, _ = make_cohort_state_summary(
        combined_obs, combined_donor_state, states, combined_order
    )
    nf_datasets = sorted(
        combined_obs.loc[combined_obs["condition"].eq("NF"), "dataset_folder"].unique(),
        key=natural_key,
    )
    source_label = " + ".join(value.replace("dataset", "D") for value in nf_datasets)
    return summary, combined_order, source_label


def save_filter_support_figure(
    cohort_summary: pd.DataFrame,
    donor_meta: pd.DataFrame,
    cohort_order: list[str],
    minimum: int,
    path: Path,
    dpi: int,
) -> None:
    summary = cohort_summary.set_index("cohort").reindex(cohort_order)
    labels = [display_cohort_label(value) for value in cohort_order]
    x = np.arange(len(cohort_order))
    width = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)

    ax = axes[0, 0]
    ax.bar(x - width / 2, summary["n_donors_before"], width, color="#BDBDBD")
    bars = ax.bar(
        x + width / 2,
        summary["n_donors_retained"],
        width,
        color=[CONDITION_COLORS.get(str(value), "#777777") for value in summary["condition"]],
    )
    for bar, before in zip(bars, summary["n_donors_before"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(bar.get_height())}/{int(before)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Final-LEC-positive donors")
    ax.set_title("Donor retention")
    condition_handles = [Patch(facecolor="#BDBDBD", label="Before")]
    for condition in summary["condition"].astype(str).drop_duplicates():
        condition_handles.append(
            Patch(
                facecolor=CONDITION_COLORS.get(condition, "#777777"),
                label=f"Retained {condition}",
            )
        )
    ax.legend(handles=condition_handles, frameon=False, ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    ax.bar(x - width / 2, summary["n_LEC_cells_before"], width, color="#BDBDBD")
    ax.bar(
        x + width / 2,
        summary["n_LEC_cells_retained"],
        width,
        color=[CONDITION_COLORS.get(str(value), "#777777") for value in summary["condition"]],
    )
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Final Step 08 LEC cells")
    ax.set_title("Cell retention")
    ax.legend(handles=condition_handles, frameon=False, ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    ax.bar(x - width / 2, 100 * summary["donor_fraction_retained"], width,
           color="#0072B2", label="Donors")
    ax.bar(x + width / 2, 100 * summary["LEC_cell_fraction_retained"], width,
           color="#D55E00", label="LEC cells")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Retained (%)")
    ax.set_title("Donor loss can exceed cell loss")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    rng = np.random.default_rng(8032026)
    for position, cohort in enumerate(cohort_order):
        part = donor_meta.loc[donor_meta["cohort"].eq(cohort)]
        condition = str(part["condition"].iloc[0])
        retained = part["retained_minimum_LECs"]
        ax.scatter(
            position + rng.uniform(-0.16, 0.16, len(part)),
            part["n_final_LECs"],
            s=np.where(retained, 36, 26),
            color=np.where(retained, CONDITION_COLORS.get(condition, "#777777"), "#BDBDBD"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.axhline(minimum, color="#111111", ls="--", label=f"minimum = {minimum}")
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Final LECs per donor (log scale)")
    ax.set_title("Gray donors are excluded only from this branch")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Disease Step 08.3: donor-level minimum of {minimum} final LECs", fontsize=17)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_composition_figure(
    cohort_state: pd.DataFrame,
    cohort_order: list[str],
    states: list[str],
    palette: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    labels = [display_cohort_label(value) for value in cohort_order]
    panels = [
        ("cell_pooled_percentage_within_cohort", "Cell-pooled state composition (%)", "Blues", 100),
        ("mean_within_donor_state_percentage", "Mean donor-level state percentage", "Oranges", 100),
        ("percent_donors_with_state", "Retained donors containing state (%)", "Greens", 100),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    for ax, (column, title, cmap, vmax) in zip(axes, panels):
        matrix = (
            cohort_state.pivot(index="cohort", columns="LEC_state", values=column)
            .reindex(index=cohort_order, columns=states)
        )
        color_map = plt.get_cmap(cmap).copy()
        color_map.set_bad("#D9D9D9")
        values = np.ma.masked_invalid(matrix.to_numpy(float))
        image = ax.imshow(values, vmin=0, vmax=vmax, cmap=color_map, aspect="auto")
        ax.set_xticks(range(len(states)), states, rotation=35, ha="right")
        color_state_ticks(ax, palette)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix.iat[row, col]
                if np.isfinite(value):
                    label = f"{value:.0f}"
                    color = "white" if value > 58 else "black"
                else:
                    label = "NA"
                    color = "#555555"
                ax.text(col, row, label, ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Minimum-25 branch: descriptive state support among retained donors", fontsize=17)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_cell_vs_donor_composition_figure(
    cohort_state: pd.DataFrame,
    cohort_order: list[str],
    states: list[str],
    palette: dict[str, str],
    path: Path,
    dpi: int,
    figure_title: str = (
        "Minimum-25 branch: cell-weighted and equal-donor state composition after filtering"
    ),
    explanatory_note: str = (
        "Left: every retained LEC contributes equally. Right: every retained donor contributes "
        "equally; the donor-level view is appropriate for biological-replicate interpretation."
    ),
) -> None:
    """Show pooled-cell and equal-donor state composition with one state palette."""
    labels = [display_cohort_label(value) for value in cohort_order]
    pooled = (
        cohort_state.pivot(
            index="cohort",
            columns="LEC_state",
            values="cell_pooled_percentage_within_cohort",
        )
        .reindex(index=cohort_order, columns=states)
    )
    donor_normalized = (
        cohort_state.pivot(
            index="cohort",
            columns="LEC_state",
            values="mean_within_donor_state_percentage",
        )
        .reindex(index=cohort_order, columns=states)
    )

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=False)
    x = np.arange(len(cohort_order))
    for ax, matrix, title, ylabel in [
        (
            axes[0],
            pooled,
            "Cell-level pooled composition (descriptive)",
            "State percentage of pooled cohort LECs",
        ),
        (
            axes[1],
            donor_normalized,
            "Donor-level normalized composition",
            "Mean within-donor state percentage",
        ),
    ]:
        valid = matrix.notna().any(axis=1).to_numpy(bool)
        bottom = np.zeros(len(cohort_order), dtype=float)
        for state in states:
            values = matrix[state].fillna(0).to_numpy(float)
            ax.bar(
                x,
                values,
                bottom=bottom,
                width=0.72,
                color=palette[state],
                edgecolor="white",
                linewidth=0.45,
                label=state,
            )
            bottom += values
        for index in np.flatnonzero(~valid):
            ax.bar(
                index,
                100,
                width=0.72,
                color="#D9D9D9",
                edgecolor="#777777",
                linewidth=0.8,
                hatch="//",
            )
            ax.text(
                index,
                50,
                "NA\nno retained\ndonors",
                ha="center",
                va="center",
                fontsize=8,
                color="#444444",
            )
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", color="#E3E3E3", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [Patch(facecolor=palette[state], label=state) for state in states]
    handles.append(Patch(facecolor="#D9D9D9", edgecolor="#777777", hatch="//", label="No retained donors"))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.73, wspace=0.24)
    fig.legend(
        handles=handles,
        title="Original Step 08 LEC state",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=5,
    )
    fig.suptitle(
        figure_title,
        fontsize=17,
        y=0.985,
    )
    fig.text(
        0.5,
        0.02,
        explanatory_note,
        ha="center",
        fontsize=10,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_state_impact_figure(
    state_impact: pd.DataFrame,
    dominance: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    summary = state_impact.set_index("LEC_state").reindex(states)
    x = np.arange(len(states))
    colors = [palette[state] for state in states]
    max_dominance = dominance.groupby("LEC_state", observed=True)[
        "top_donor_fraction_of_cohort_state"
    ].max().reindex(states)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)

    axes[0, 0].bar(x, 100 * summary["cell_fraction_retained"], color=colors)
    axes[0, 0].set_ylabel("State cells retained (%)")
    axes[0, 0].set_title("Cell retention by state")
    axes[0, 0].set_ylim(0, 105)

    axes[0, 1].bar(x, 100 * summary["donor_fraction_retained"], color=colors)
    axes[0, 1].set_ylabel("State-positive donors retained (%)")
    axes[0, 1].set_title("Donor-support retention by state")
    axes[0, 1].set_ylim(0, 105)

    width = 0.38
    axes[1, 0].bar(x - width / 2, summary["cell_pooled_percentage_before"], width,
                   color="#BDBDBD", label="Before")
    axes[1, 0].bar(x + width / 2, summary["cell_pooled_percentage_retained"], width,
                   color=colors, label="Minimum-25 branch")
    axes[1, 0].set_ylabel("Cell-pooled composition (%)")
    axes[1, 0].set_title("State composition before and after filtering")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(x, 100 * max_dominance, color=colors)
    axes[1, 1].axhline(50, color="#D55E00", ls="--", label="50% from one donor")
    axes[1, 1].set_ylabel("Largest donor contribution (%)")
    axes[1, 1].set_title("Worst cohort-specific donor dominance")
    axes[1, 1].legend(frameon=False)

    for ax in axes.ravel():
        ax.set_xticks(x, states, rotation=35, ha="right")
        color_state_ticks(ax, palette)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Minimum-25 filter impact on existing Step 08 states", fontsize=17)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_umap_figure(
    umap: np.ndarray,
    obs: pd.DataFrame,
    retained_mask: np.ndarray,
    states: list[str],
    palette: dict[str, str],
    minimum: int,
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    axes[0].scatter(umap[retained_mask, 0], umap[retained_mask, 1], s=4,
                    color="#0072B2", alpha=0.7, linewidths=0, label="Retained")
    axes[0].scatter(umap[~retained_mask, 0], umap[~retained_mask, 1], s=5,
                    color="#BDBDBD", alpha=0.75, linewidths=0, label="Excluded donor")
    axes[0].set_title(f"Donor-level minimum {minimum}-LEC filter")
    axes[0].legend(frameon=False, markerscale=3)

    retained_states = obs.loc[retained_mask, "LEC_state"].to_numpy(str)
    for state in states:
        mask = retained_states == state
        axes[1].scatter(umap[retained_mask, 0][mask], umap[retained_mask, 1][mask],
                        s=5, color=palette[state], alpha=0.8, linewidths=0, label=state)
    axes[1].set_title("Retained final LECs keep original Step 08 states")
    axes[1].legend(frameon=False, ncol=3, markerscale=2.5)
    for ax in axes:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Disease Step 08.3 is a filtered sensitivity view, not reclustering", fontsize=17)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_donor_state_heatmap(
    donor_state: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    cohort_order: list[str],
    path: Path,
    dpi: int,
) -> None:
    donor_order: list[str] = []
    separators: list[int] = []
    cohort_midpoints: list[float] = []
    cohort_labels: list[str] = []
    for cohort in cohort_order:
        part = donor_state.loc[donor_state["cohort"].eq(cohort)]
        order = (
            part.drop_duplicates("donor_key")
            .sort_values(["n_final_LECs", "donor_key"], ascending=[False, True])["donor_key"]
            .tolist()
        )
        if not order:
            continue
        start = len(donor_order)
        donor_order.extend(order)
        cohort_midpoints.append(start + (len(order) - 1) / 2)
        cohort_labels.append(display_cohort_label(cohort))
        separators.append(len(donor_order))
    matrix = (
        donor_state.pivot(index="donor_key", columns="LEC_state", values="state_percentage_within_donor")
        .reindex(index=donor_order, columns=states)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(11, 15), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=100, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(states)), states, rotation=35, ha="right")
    color_state_ticks(ax, palette)
    ax.set_yticks(cohort_midpoints, cohort_labels)
    for separator in separators[:-1]:
        ax.axhline(separator - 0.5, color="white", lw=1.5)
    ax.set_title("Each retained donor remains one biological-replicate row")
    ax.set_xlabel("Original Step 08 LEC state")
    ax.set_ylabel("Cohort blocks; donors ordered by LEC count")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Within-donor state percentage")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.minimum_LECs < 1:
        raise ValueError("--minimum_LECs must be positive.")
    if args.minimum_donors_per_condition < 2:
        raise ValueError("--minimum_donors_per_condition must be at least 2.")
    project_dir = find_project_dir()
    input_h5ad = resolve_path(args.input_h5ad, project_dir)
    threshold_summary_path = resolve_path(args.threshold_summary, project_dir)
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

    provenance, provenance_status = threshold_provenance(
        threshold_summary_path, args.minimum_LECs
    )
    print(f"Reading final Step 08 LEC object: {input_h5ad}")
    adata = ad.read_h5ad(input_h5ad, backed="r")
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    validate_obs(obs)
    for column in ["dataset_folder", "condition", "donor_key", "LEC_state"]:
        obs[column] = normalized_text(obs[column])
    obs["cohort"] = obs["dataset_folder"] + " | " + obs["condition"]
    states = sorted(obs["LEC_state"].unique(), key=natural_key)
    palette = state_palette(adata, states)
    if "X_umap" not in adata.obsm:
        raise KeyError("Step 08 h5ad has no X_umap embedding.")
    umap = np.asarray(adata.obsm["X_umap"])
    adata.file.close()

    donor_meta, cell_mapping, retained_obs, donor_counts, donor_state = make_filter_tables(
        obs, states, args.minimum_LECs
    )
    retained_mask = cell_mapping["retained_minimum_LECs"].to_numpy(bool)
    retained_donors = donor_meta.loc[donor_meta["retained_minimum_LECs"]].copy()
    excluded_donors = donor_meta.loc[~donor_meta["retained_minimum_LECs"]].copy()
    cohort_summary = make_cohort_summary(donor_meta)
    cohort_order = ordered_dataset_cohorts(set(cohort_summary["cohort"].astype(str)))
    readiness = make_comparison_readiness(donor_meta, args.minimum_donors_per_condition)
    nichenet_donors = make_nichenet_donor_manifest(
        obs, donor_meta, readiness, args.minimum_LECs
    )
    state_impact = make_state_filter_impact(obs, retained_obs, states)
    cohort_state, dominance = make_cohort_state_summary(
        retained_obs, donor_state, states, cohort_order
    )
    combined_nf_state, combined_nf_order, combined_nf_sources = make_combined_nf_state_summary(
        retained_obs, donor_state, states
    )

    if len(obs) != len(cell_mapping):
        raise RuntimeError("Cell mapping does not cover every Step 08 LEC.")
    if int(donor_state["n_state_LECs"].sum()) != len(retained_obs):
        raise RuntimeError("Retained donor-state counts do not sum to retained cells.")
    if len(donor_state) != len(retained_donors) * len(states):
        raise RuntimeError("Zero-count donor-state combinations were not fully retained.")
    if retained_donors["n_final_LECs"].min() < args.minimum_LECs:
        raise RuntimeError("A retained donor is below the requested minimum.")
    if len(excluded_donors) and excluded_donors["n_final_LECs"].max() >= args.minimum_LECs:
        raise RuntimeError("An excluded donor meets the requested minimum.")

    provenance.to_csv(tables_dir / "08_3_threshold_provenance.csv", index=False)
    donor_meta.to_csv(tables_dir / "08_3_all_donor_filter_mapping.csv", index=False)
    retained_donors.to_csv(tables_dir / "08_3_retained_donors_min25.csv", index=False)
    excluded_donors.to_csv(tables_dir / "08_3_excluded_donors_below25.csv", index=False)
    cell_mapping.to_csv(
        tables_dir / "08_3_final_LEC_cell_filter_mapping.csv.gz",
        index=False,
        compression="gzip",
    )
    cell_mapping.loc[cell_mapping["retained_minimum_LECs"], ["step08_cell_id"]].to_csv(
        tables_dir / "08_3_retained_final_LEC_cell_ids.csv.gz",
        index=False,
        compression="gzip",
    )
    donor_counts.to_csv(tables_dir / "08_3_retained_donor_state_counts.csv", index=False)
    donor_state.to_csv(
        tables_dir / "08_3_retained_donor_state_counts_percentages.csv.gz",
        index=False,
        compression="gzip",
    )
    cohort_summary.to_csv(tables_dir / "08_3_cohort_filter_impact.csv", index=False)
    readiness.to_csv(tables_dir / "08_3_within_dataset_comparison_readiness.csv", index=False)
    nichenet_donors.to_csv(
        output_root / "08_3_NicheNet_retained_donor_manifest.csv", index=False
    )
    state_impact.to_csv(tables_dir / "08_3_state_filter_impact.csv", index=False)
    cohort_state.to_csv(tables_dir / "08_3_retained_cohort_state_summary.csv", index=False)
    combined_nf_state.to_csv(
        tables_dir / "08_3_combined_NF_state_summary.csv", index=False
    )
    dominance.to_csv(
        tables_dir / "08_3_retained_highest_donor_contribution_by_cohort_state.csv",
        index=False,
    )
    pd.DataFrame(
        {"LEC_state": states, "color": [palette[state] for state in states]}
    ).to_csv(tables_dir / "08_3_state_color_mapping.csv", index=False)

    input_validation = pd.DataFrame(
        [
            {
                "input_h5ad": str(input_h5ad),
                "threshold_provenance_status": provenance_status,
                "minimum_LECs": int(args.minimum_LECs),
                "n_final_LEC_cells_before": int(len(obs)),
                "n_final_LEC_positive_donors_before": int(obs["donor_key"].nunique()),
                "n_retained_LEC_cells": int(len(retained_obs)),
                "n_retained_donors": int(len(retained_donors)),
                "n_NicheNet_manifest_donors": int(len(nichenet_donors)),
                "n_excluded_LEC_cells": int(len(obs) - len(retained_obs)),
                "n_excluded_donors": int(len(excluded_donors)),
                "n_non_LEC_cells": 0,
                "n_states_before": int(len(states)),
                "n_states_retained": int(retained_obs["LEC_state"].nunique()),
                "source_h5ad_modified": False,
                "new_h5ad_written": False,
                "status": "validated_minimum_LEC_read_only_sensitivity_branch",
            }
        ]
    )
    input_validation.to_csv(tables_dir / "08_3_input_validation.csv", index=False)

    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        jobs = [
            (
                "08_3_donor_and_cell_filter_impact.png",
                "Donor and cell retention by cohort under the minimum-25 filter",
                lambda path: save_filter_support_figure(
                    cohort_summary, donor_meta, cohort_order, args.minimum_LECs, path, args.dpi
                ),
            ),
            (
                "08_3_retained_cohort_state_composition.png",
                "Cell-pooled, donor-normalized, and donor-presence state summaries",
                lambda path: save_composition_figure(
                    cohort_state, cohort_order, states, palette, path, args.dpi
                ),
            ),
            (
                "08_3_cell_level_vs_donor_level_state_composition.png",
                "Dataset-specific side-by-side state composition using pooled cells and equal donor weighting; NF appears first within each dataset",
                lambda path: save_cell_vs_donor_composition_figure(
                    cohort_state, cohort_order, states, palette, path, args.dpi
                ),
            ),
            (
                "08_3_combined_NF_cell_vs_donor_state_composition.png",
                "Descriptive combined-NF state composition beside dataset-specific disease cohorts",
                lambda path: save_cell_vs_donor_composition_figure(
                    combined_nf_state,
                    combined_nf_order,
                    states,
                    palette,
                    path,
                    args.dpi,
                    figure_title="Minimum-25 branch: combined retained NF composition and disease cohorts",
                    explanatory_note=(
                        f"Combined NF uses retained NF donors from {combined_nf_sources}. "
                        "Cell-pooled and equal-donor summaries are descriptive only; disease tests remain within dataset."
                    ),
                ),
            ),
            (
                "08_3_state_filter_impact_and_dominance.png",
                "State support before/after filtering and retained-donor dominance",
                lambda path: save_state_impact_figure(
                    state_impact, dominance, states, palette, path, args.dpi
                ),
            ),
            (
                "08_3_retained_LEC_UMAP.png",
                "Retained and excluded-donor LECs on the unchanged Step 08 UMAP",
                lambda path: save_umap_figure(
                    umap, obs, retained_mask, states, palette, args.minimum_LECs, path, args.dpi
                ),
            ),
            (
                "08_3_retained_donor_state_percentage_heatmap.png",
                "Within-donor state percentages for every retained donor",
                lambda path: save_donor_state_heatmap(
                    donor_state, states, palette, cohort_order, path, args.dpi
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
    pd.DataFrame(figure_manifest).to_csv(output_root / "08_3_figure_manifest.csv", index=False)

    unavailable = readiness.loc[~readiness["donor_level_comparison_ready"], "comparison"].tolist()
    notes = f"""Disease Step 08.3 minimum-{args.minimum_LECs}-LEC filtered analysis
==================================================================

Scope
-----
Input h5ad (read only): {input_h5ad}
Threshold source: {threshold_summary_path}
Threshold provenance: {provenance_status}
Final Step 08 LECs before filter: {len(obs):,}
LEC-positive donors before filter: {obs['donor_key'].nunique()}
Retained final LECs: {len(retained_obs):,}
Retained donors: {len(retained_donors)}
Excluded final LECs: {len(obs) - len(retained_obs):,}
Excluded donors: {len(excluded_donors)}

Method
------
A donor is retained when its total number of final Step 08 LECs is at least
{args.minimum_LECs}. This is a donor-total threshold, not a donor-state
threshold. Every retained donor-state combination is exported, including zero
counts. Each donor remains one biological-replicate entry.

The original Step 08 clustering, UMAP, expression, and LEC_state labels are
preserved. No h5ad is written, matching the read-only aging filtered branch.

NicheNet handoff
----------------
08_3_NicheNet_retained_donor_manifest.csv contains exactly the retained donor
set, original donor IDs, dataset/condition, contributing sample IDs, retained
final-LEC counts, and analysis roles. Use recommended_join_key (dataset::donor)
as the cross-file join key. The five dataset11 HFpEF donors are marked
descriptive-only because no dataset11 NF donor survives this filter.

Comparison safeguards
---------------------
Disease contrasts must remain within dataset. A future donor-level comparison
is marked ready only if both conditions retain at least
{args.minimum_donors_per_condition} donors. Comparisons not ready after the
filter: {('; '.join(unavailable)) if unavailable else 'none'}.

No cell-pooled hypothesis test is substituted for a failed donor-level
comparison. In particular, dataset11 NF loses all donors at the minimum-25
threshold, so HFpEF versus NF is not testable in this filtered branch.

Interpretation
--------------
This is a sensitivity branch, not a claim that donors below 25 LECs are
biologically invalid. The unfiltered analysis should be retained alongside it.
Step 08.2 showed that P90 total-variation error first fell below the configured
20% tolerance at N=25 among high-depth reference donors, but P95 remained above
20%, and dataset11 NF had no direct >=25-LEC reference donor.
"""
    (output_root / "08_3_method_and_interpretation_notes.txt").write_text(
        notes, encoding="utf-8"
    )

    run_config = {
        "script_version": SCRIPT_VERSION,
        "input_h5ad": str(input_h5ad),
        "threshold_summary": str(threshold_summary_path),
        "threshold_provenance_status": provenance_status,
        "output_root": str(output_root),
        "minimum_total_final_LECs_per_donor": int(args.minimum_LECs),
        "minimum_donors_per_condition_for_readiness": int(
            args.minimum_donors_per_condition
        ),
        "cell_scope": "exact_current_Step08_final_LECs_only",
        "filter_level": "donor_total_final_LECs",
        "clustering_or_state_labels_changed": False,
        "source_h5ad_modified": False,
        "new_h5ad_written": False,
        "inference_performed": False,
        "cell_level_test_substituted": False,
    }
    (output_root / "08_3_run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    run_summary = pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "status": "completed",
                "minimum_LECs": int(args.minimum_LECs),
                "n_final_LEC_cells_before": int(len(obs)),
                "n_final_LEC_positive_donors_before": int(obs["donor_key"].nunique()),
                "n_retained_LEC_cells": int(len(retained_obs)),
                "n_retained_donors": int(len(retained_donors)),
                "n_excluded_LEC_cells": int(len(obs) - len(retained_obs)),
                "n_excluded_donors": int(len(excluded_donors)),
                "n_states_retained": int(retained_obs["LEC_state"].nunique()),
                "ready_comparisons": int(readiness["donor_level_comparison_ready"].sum()),
                "not_ready_comparisons": int((~readiness["donor_level_comparison_ready"]).sum()),
                "source_h5ad_modified": False,
                "new_h5ad_written": False,
                "output_root": str(output_root),
            }
        ]
    )
    run_summary.to_csv(output_root / "08_3_run_summary.csv", index=False)

    print("\nCompleted disease Step 08.3 minimum-LEC sensitivity branch.")
    print(run_summary.to_string(index=False))
    print("\nCohort retention:")
    print(
        cohort_summary[
            ["cohort", "n_donors_before", "n_donors_retained", "n_LEC_cells_retained"]
        ].to_string(index=False)
    )
    print("\nComparison readiness:")
    print(readiness[["comparison", "n_reference_donors_retained", "n_disease_donors_retained", "status"]].to_string(index=False))
    print("\nNo source h5ad, clustering, state label, cell, or donor was modified.")
    print(f"Output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
