#!/usr/bin/env python3
r"""
Step 08.4: donor-level LEC-state comparisons after the >=25-LEC filter.

This is the disease counterpart of the donor-level state-composition analysis
in the aging branch, adapted to the disease design. It uses only final Step 08
LECs belonging to donors retained by Step 08.3. Each donor is one biological
replicate, and every donor-state pair is retained, including zero counts.

Primary comparisons are pre-specified and remain within dataset:

    dataset01: DCM versus NF
    dataset01: HCM versus NF
    dataset08: DoxCM versus NF
    dataset11: HFpEF versus NF (reported as not testable after filtering)

For each ready comparison and state, the script reports a two-sided
Mann-Whitney U test, disease-minus-NF mean and median differences, rank-biserial
effect size, and donor bootstrap 95% confidence intervals. Benjamini-Hochberg
FDR is supplied both within each comparison (primary multiplicity family) and
across every ready comparison-state test (more conservative sensitivity view).

The plots show individual donor percentages and group means with 95% bootstrap
confidence intervals. No cell-level hypothesis tests are used. The script does
not change clustering, state labels, expression, the source h5ad, or write a
new h5ad.

Default input
-------------
new_result2_disease/08_combined_LEC_recluster/
    08_combined_final_LEC_reclustered.h5ad
new_result2_disease/08_3_min25_LEC_filtered_analysis/tables/
    08_3_retained_donors_min25.csv
    08_3_within_dataset_comparison_readiness.csv
    08_3_state_color_mapping.csv

Default output
--------------
new_result2_disease/08_4_donor_level_LEC_state_disease_analysis/

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\08_4_donor_level_LEC_state_disease_analysis.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_VERSION = "08_4_disease_min25_donor_level_state_analysis_2026_08_12c"
DEFAULT_INPUT = (
    "new_result2_disease/08_combined_LEC_recluster/"
    "08_combined_final_LEC_reclustered.h5ad"
)
DEFAULT_FILTER_ROOT = "new_result2_disease/08_3_min25_LEC_filtered_analysis"
DEFAULT_OUTPUT = (
    "new_result2_disease/08_4_donor_level_LEC_state_disease_analysis"
)

PLANNED_COMPARISONS = [
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

RETAINED_COHORT_ORDER = [
    "dataset01 | NF",
    "dataset01 | DCM",
    "dataset01 | HCM",
    "dataset08 | NF",
    "dataset08 | DoxCM",
    "dataset11 | NF",
    "dataset11 | HFpEF",
]


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare donor-level final-LEC state percentages within disease datasets "
            "after retaining donors with at least 25 final LECs."
        )
    )
    parser.add_argument("--input_h5ad", default=DEFAULT_INPUT)
    parser.add_argument("--filter_root", default=DEFAULT_FILTER_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--donor_key", default="donor_key")
    parser.add_argument("--dataset_key", default="dataset_folder")
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--state_key", default="LEC_state")
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def natural_key(value: str) -> list[object]:
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", value)]


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def clean_string(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(values.str.lower().isin({"", "nan", "none", "na", "n/a", "unknown"}))


def stable_seed(base_seed: int, *parts: object) -> int:
    label = "||".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(label).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((int(base_seed) + offset) % (2**32 - 1))


def bootstrap_summary(
    reference: np.ndarray,
    disease: np.ndarray,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float)
    disease = np.asarray(disease, dtype=float)
    if len(reference) == 0 or len(disease) == 0:
        return {
            "reference_mean_ci_low": np.nan,
            "reference_mean_ci_high": np.nan,
            "disease_mean_ci_low": np.nan,
            "disease_mean_ci_high": np.nan,
            "mean_difference_ci_low": np.nan,
            "mean_difference_ci_high": np.nan,
            "median_difference_ci_low": np.nan,
            "median_difference_ci_high": np.nan,
        }

    rng = np.random.default_rng(seed)
    ref_idx = rng.integers(0, len(reference), size=(iterations, len(reference)))
    dis_idx = rng.integers(0, len(disease), size=(iterations, len(disease)))
    ref_draws = reference[ref_idx]
    dis_draws = disease[dis_idx]
    ref_means = ref_draws.mean(axis=1)
    dis_means = dis_draws.mean(axis=1)
    mean_diff = dis_means - ref_means
    median_diff = np.median(dis_draws, axis=1) - np.median(ref_draws, axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = 100.0 * tail, 100.0 * (1.0 - tail)

    def interval(values: np.ndarray) -> tuple[float, float]:
        result = np.percentile(values, [lower, upper])
        return float(result[0]), float(result[1])

    ref_low, ref_high = interval(ref_means)
    dis_low, dis_high = interval(dis_means)
    mean_low, mean_high = interval(mean_diff)
    median_low, median_high = interval(median_diff)
    return {
        "reference_mean_ci_low": ref_low,
        "reference_mean_ci_high": ref_high,
        "disease_mean_ci_low": dis_low,
        "disease_mean_ci_high": dis_high,
        "mean_difference_ci_low": mean_low,
        "mean_difference_ci_high": mean_high,
        "median_difference_ci_low": median_low,
        "median_difference_ci_high": median_high,
    }


def bh_adjust(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    adjusted = np.full(len(numeric), np.nan, dtype=float)
    finite = np.isfinite(numeric)
    if finite.any():
        adjusted[finite] = multipletests(numeric[finite], method="fdr_bh")[1]
    return adjusted


def load_inputs(
    input_h5ad: Path,
    filter_root: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    retained_path = filter_root / "tables" / "08_3_retained_donors_min25.csv"
    readiness_path = filter_root / "tables" / "08_3_within_dataset_comparison_readiness.csv"
    colors_path = filter_root / "tables" / "08_3_state_color_mapping.csv"
    required_paths = [input_h5ad, retained_path, readiness_path, colors_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))

    retained = pd.read_csv(retained_path)
    readiness = pd.read_csv(readiness_path)
    state_colors = pd.read_csv(colors_path)
    if "LEC_state" not in state_colors.columns or "color" not in state_colors.columns:
        raise ValueError("Step 08.3 state-color table lacks LEC_state/color columns.")
    states = sorted(state_colors["LEC_state"].astype(str).unique(), key=natural_key)

    required_obs = [args.donor_key, args.dataset_key, args.condition_key, args.state_key]
    adata = ad.read_h5ad(input_h5ad, backed="r")
    absent = [column for column in required_obs if column not in adata.obs.columns]
    if absent:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
        raise KeyError(f"Input h5ad is missing required obs columns: {absent}")
    obs_columns = required_obs + [
        column
        for column in ["sex", "age", "lvef", "region_standardized_analysis"]
        if column in adata.obs.columns and column not in required_obs
    ]
    obs = adata.obs[obs_columns].copy()
    obs.insert(0, "step08_cell_id", adata.obs_names.astype(str))
    if getattr(adata, "file", None) is not None:
        adata.file.close()

    for column in required_obs:
        obs[column] = obs[column].astype(str)
    retained[args.donor_key] = retained[args.donor_key].astype(str)
    retained_keys = set(retained[args.donor_key])
    filtered_obs = obs.loc[obs[args.donor_key].isin(retained_keys)].copy()

    observed_keys = set(filtered_obs[args.donor_key])
    if observed_keys != retained_keys:
        absent_donors = sorted(retained_keys - observed_keys)
        unexpected = sorted(observed_keys - retained_keys)
        raise RuntimeError(
            "Step 08.3 donor filter does not match Step 08 h5ad. "
            f"Absent retained donors={absent_donors}; unexpected={unexpected}"
        )
    if len(filtered_obs) != int(pd.to_numeric(retained["n_final_LECs"]).sum()):
        raise RuntimeError(
            "Filtered cell count does not equal the retained Step 08.3 donor total: "
            f"{len(filtered_obs)} versus {int(pd.to_numeric(retained['n_final_LECs']).sum())}."
        )
    unknown_states = sorted(set(filtered_obs[args.state_key]) - set(states), key=natural_key)
    if unknown_states:
        raise RuntimeError(f"Filtered cells contain states absent from color mapping: {unknown_states}")
    return filtered_obs, retained, readiness, state_colors, states


def donor_state_tables(
    obs: pd.DataFrame,
    retained: pd.DataFrame,
    states: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    donor_meta = retained.copy()
    donor_meta[args.donor_key] = donor_meta[args.donor_key].astype(str)
    donor_meta[args.dataset_key] = donor_meta[args.dataset_key].astype(str)
    donor_meta[args.condition_key] = donor_meta[args.condition_key].astype(str)
    donor_meta["cohort"] = donor_meta[args.dataset_key] + " | " + donor_meta[args.condition_key]
    donor_meta["n_final_LECs"] = pd.to_numeric(donor_meta["n_final_LECs"], errors="raise").astype(int)

    counts = pd.crosstab(obs[args.donor_key], obs[args.state_key]).reindex(
        index=donor_meta[args.donor_key], columns=states, fill_value=0
    )
    counts.index.name = args.donor_key
    wide = counts.reset_index().merge(
        donor_meta,
        on=args.donor_key,
        how="left",
        validate="one_to_one",
        suffixes=("", "_step08_3"),
    )
    counted = wide[states].sum(axis=1).astype(int)
    if not np.array_equal(counted.to_numpy(), wide["n_final_LECs"].to_numpy(int)):
        raise RuntimeError("Donor state counts do not sum to Step 08.3 retained LEC totals.")

    id_columns = [
        args.donor_key,
        args.dataset_key,
        args.condition_key,
        "cohort",
        "sex",
        "age",
        "lvef",
        "region_standardized_analysis",
        "n_final_LECs",
    ]
    id_columns = [column for column in id_columns if column in wide.columns]
    long = wide.melt(
        id_vars=id_columns,
        value_vars=states,
        var_name="LEC_state",
        value_name="n_state_LECs",
    )
    long["n_state_LECs"] = long["n_state_LECs"].astype(int)
    long["state_percentage_of_donor_LECs"] = (
        100.0 * long["n_state_LECs"] / long["n_final_LECs"]
    )
    long["state_present_in_donor"] = long["n_state_LECs"].gt(0)
    long["LEC_state"] = pd.Categorical(long["LEC_state"], categories=states, ordered=True)
    return donor_meta, long.sort_values([args.dataset_key, args.condition_key, args.donor_key, "LEC_state"])


def make_cohort_summary(donor_meta: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = [args.dataset_key, args.condition_key, "cohort"]
    for keys, part in donor_meta.groupby(group_columns, sort=False, dropna=False):
        age = pd.to_numeric(part.get("age", pd.Series(index=part.index, dtype=float)), errors="coerce")
        sex = clean_string(part.get("sex", pd.Series(index=part.index, dtype="string")))
        row: dict[str, object] = {
            args.dataset_key: keys[0],
            args.condition_key: keys[1],
            "cohort": keys[2],
            "n_donors": int(len(part)),
            "n_final_LECs": int(part["n_final_LECs"].sum()),
            "median_LECs_per_donor": float(part["n_final_LECs"].median()),
            "minimum_LECs_per_donor": int(part["n_final_LECs"].min()),
            "maximum_LECs_per_donor": int(part["n_final_LECs"].max()),
            "n_age_available": int(age.notna().sum()),
            "age_mean": float(age.mean()) if age.notna().any() else np.nan,
            "age_sd": float(age.std(ddof=1)) if age.notna().sum() > 1 else np.nan,
            "age_median": float(age.median()) if age.notna().any() else np.nan,
            "age_minimum": float(age.min()) if age.notna().any() else np.nan,
            "age_maximum": float(age.max()) if age.notna().any() else np.nan,
            "n_sex_available": int(sex.notna().sum()),
            "n_female": int(sex.str.lower().eq("female").sum()),
            "n_male": int(sex.str.lower().eq("male").sum()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def make_cohort_state_summary(donor_state: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["dataset_folder", "condition", "cohort", "LEC_state"]
    for keys, part in donor_state.groupby(group_columns, observed=True, sort=False):
        values = part["state_percentage_of_donor_LECs"].to_numpy(float)
        rows.append(
            {
                "dataset_folder": keys[0],
                "condition": keys[1],
                "cohort": keys[2],
                "LEC_state": str(keys[3]),
                "n_donors": int(len(part)),
                "n_donors_with_state": int(part["state_present_in_donor"].sum()),
                "donor_prevalence_percent": float(100.0 * part["state_present_in_donor"].mean()),
                "total_state_LECs": int(part["n_state_LECs"].sum()),
                "mean_donor_percentage": float(np.mean(values)),
                "sd_donor_percentage": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "sem_donor_percentage": (
                    float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan
                ),
                "median_donor_percentage": float(np.median(values)),
                "minimum_donor_percentage": float(np.min(values)),
                "maximum_donor_percentage": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def normalize_readiness(readiness: pd.DataFrame) -> pd.DataFrame:
    result = readiness.copy()
    if "donor_level_comparison_ready" not in result.columns:
        raise KeyError("Step 08.3 readiness table lacks donor_level_comparison_ready.")
    result["donor_level_comparison_ready"] = as_bool(result["donor_level_comparison_ready"])
    return result


def make_primary_comparisons(
    donor_state: pd.DataFrame,
    readiness: pd.DataFrame,
    states: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    readiness = normalize_readiness(readiness)
    rows: list[dict[str, object]] = []
    readiness_rows: list[dict[str, object]] = []
    for dataset, reference_condition, disease_condition in PLANNED_COMPARISONS:
        match = readiness.loc[
            readiness["dataset_folder"].astype(str).eq(dataset)
            & readiness["reference_condition"].astype(str).eq(reference_condition)
            & readiness["disease_condition"].astype(str).eq(disease_condition)
        ]
        if len(match) != 1:
            raise RuntimeError(
                f"Expected one Step 08.3 readiness row for {dataset}: "
                f"{disease_condition} vs {reference_condition}; found {len(match)}."
            )
        ready = bool(match.iloc[0]["donor_level_comparison_ready"])
        comparison = f"{dataset}: {disease_condition} vs {reference_condition}"
        subset = donor_state.loc[donor_state["dataset_folder"].astype(str).eq(dataset)]
        n_reference = subset.loc[
            subset["condition"].astype(str).eq(reference_condition), "donor_key"
        ].nunique()
        n_disease = subset.loc[
            subset["condition"].astype(str).eq(disease_condition), "donor_key"
        ].nunique()
        analysis_status = "tested" if ready and n_reference >= 2 and n_disease >= 2 else "not_testable"
        reason = (
            "ready_in_step08_3_and_both_arms_have_at_least_two_retained_donors"
            if analysis_status == "tested"
            else "one_or_both_arms_have_fewer_than_two_retained_donors"
        )
        readiness_rows.append(
            {
                "comparison": comparison,
                "dataset_folder": dataset,
                "reference_condition": reference_condition,
                "disease_condition": disease_condition,
                "n_reference_donors": int(n_reference),
                "n_disease_donors": int(n_disease),
                "analysis_status": analysis_status,
                "reason": reason,
            }
        )
        if analysis_status != "tested":
            continue

        for state in states:
            state_part = subset.loc[subset["LEC_state"].astype(str).eq(state)]
            reference = state_part.loc[
                state_part["condition"].astype(str).eq(reference_condition),
                "state_percentage_of_donor_LECs",
            ].to_numpy(float)
            disease = state_part.loc[
                state_part["condition"].astype(str).eq(disease_condition),
                "state_percentage_of_donor_LECs",
            ].to_numpy(float)
            if len(reference) != n_reference or len(disease) != n_disease:
                raise RuntimeError(f"Incomplete donor-state grid for {comparison}, {state}.")
            test = stats.mannwhitneyu(disease, reference, alternative="two-sided", method="asymptotic")
            u = float(test.statistic)
            rank_biserial = 2.0 * u / (len(disease) * len(reference)) - 1.0
            ci = bootstrap_summary(
                reference,
                disease,
                args.bootstrap_iterations,
                args.confidence_level,
                stable_seed(args.seed, comparison, state),
            )
            rows.append(
                {
                    "comparison": comparison,
                    "dataset_folder": dataset,
                    "reference_condition": reference_condition,
                    "disease_condition": disease_condition,
                    "LEC_state": state,
                    "n_reference_donors": int(len(reference)),
                    "n_disease_donors": int(len(disease)),
                    "reference_mean_percentage": float(np.mean(reference)),
                    "reference_median_percentage": float(np.median(reference)),
                    "reference_sd_percentage": float(np.std(reference, ddof=1)),
                    "disease_mean_percentage": float(np.mean(disease)),
                    "disease_median_percentage": float(np.median(disease)),
                    "disease_sd_percentage": float(np.std(disease, ddof=1)),
                    "mean_difference_percentage_points": float(np.mean(disease) - np.mean(reference)),
                    "median_difference_percentage_points": float(np.median(disease) - np.median(reference)),
                    "mann_whitney_U_disease_vs_reference": u,
                    "rank_biserial_disease_vs_reference": float(rank_biserial),
                    "mann_whitney_p": float(test.pvalue),
                    "bootstrap_iterations": int(args.bootstrap_iterations),
                    "confidence_level": float(args.confidence_level),
                    **ci,
                }
            )

    results = pd.DataFrame(rows)
    if len(results):
        results["mann_whitney_FDR_within_comparison"] = results.groupby(
            "comparison", sort=False
        )["mann_whitney_p"].transform(lambda series: bh_adjust(series))
        results["mann_whitney_FDR_global_all_comparisons_states"] = bh_adjust(
            results["mann_whitney_p"]
        )
        results["significant_within_comparison_FDR"] = results[
            "mann_whitney_FDR_within_comparison"
        ].lt(args.alpha)
        results["significant_global_FDR"] = results[
            "mann_whitney_FDR_global_all_comparisons_states"
        ].lt(args.alpha)
        results["mean_difference_CI_excludes_zero"] = (
            results["mean_difference_ci_low"].gt(0)
            | results["mean_difference_ci_high"].lt(0)
        )
    return results, pd.DataFrame(readiness_rows)


def make_dataset01_omnibus(
    donor_state: pd.DataFrame,
    states: list[str],
    alpha: float,
) -> pd.DataFrame:
    part = donor_state.loc[donor_state["dataset_folder"].astype(str).eq("dataset01")]
    groups = [condition for condition in ["NF", "DCM", "HCM"] if condition in set(part["condition"].astype(str))]
    rows: list[dict[str, object]] = []
    for state in states:
        state_part = part.loc[part["LEC_state"].astype(str).eq(state)]
        arrays = [
            state_part.loc[
                state_part["condition"].astype(str).eq(condition),
                "state_percentage_of_donor_LECs",
            ].to_numpy(float)
            for condition in groups
        ]
        if len(arrays) >= 2 and all(len(values) >= 2 for values in arrays):
            test = stats.kruskal(*arrays)
            statistic, p_value, status = float(test.statistic), float(test.pvalue), "tested"
        else:
            statistic, p_value, status = np.nan, np.nan, "not_testable"
        row: dict[str, object] = {
            "dataset_folder": "dataset01",
            "LEC_state": state,
            "conditions": "|".join(groups),
            "kruskal_H": statistic,
            "kruskal_p": p_value,
            "status": status,
        }
        for condition, values in zip(groups, arrays):
            row[f"n_{condition}_donors"] = int(len(values))
        rows.append(row)
    result = pd.DataFrame(rows)
    result["kruskal_FDR_across_states"] = bh_adjust(result["kruskal_p"])
    result["significant_FDR"] = result["kruskal_FDR_across_states"].lt(alpha)
    return result


def state_palette(state_colors: pd.DataFrame, states: list[str]) -> dict[str, str]:
    mapping = dict(zip(state_colors["LEC_state"].astype(str), state_colors["color"].astype(str)))
    return {state: mapping[state] for state in states}


def save_support_figure(
    cohort_summary: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    summary = cohort_summary.copy()
    order_lookup = {cohort: index for index, cohort in enumerate(RETAINED_COHORT_ORDER)}
    summary["_display_order"] = summary["cohort"].map(order_lookup).fillna(len(order_lookup))
    summary = summary.sort_values(["_display_order", "cohort"]).reset_index(drop=True)
    labels = [f"{row.dataset_folder}\n{row.condition}" for row in summary.itertuples()]
    colors = [CONDITION_COLORS.get(str(value), "#777777") for value in summary["condition"]]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), constrained_layout=False)
    axes[0].bar(x, summary["n_donors"], color=colors, edgecolor="#222222", linewidth=0.7)
    axes[0].set_ylabel("Retained donors")
    axes[0].set_title("Biological-replicate support")
    axes[1].bar(x, summary["n_final_LECs"], color=colors, edgecolor="#222222", linewidth=0.7)
    axes[1].set_ylabel("Retained final LECs")
    axes[1].set_title("Cell support (descriptive only)")
    for ax, values in zip(axes, [summary["n_donors"], summary["n_final_LECs"]]):
        ax.set_xticks(x, labels, rotation=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", lw=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        for index, value in enumerate(values):
            ax.text(index, value, f"{int(value):,}", ha="center", va="bottom", fontsize=9)
    handles = [
        Patch(facecolor=color, edgecolor="#222222", label=condition)
        for condition, color in CONDITION_COLORS.items()
        if condition in set(summary["condition"].astype(str))
    ]
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.77, wspace=0.26)
    fig.legend(
        handles=handles,
        title="Condition",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=len(handles),
    )
    fig.suptitle(
        "Disease Step 08.4 uses only donors retained at >=25 final LECs",
        fontsize=16,
        y=0.985,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_comparison_donor_figure(
    donor_state: pd.DataFrame,
    results: pd.DataFrame,
    comparison_row: pd.Series,
    states: list[str],
    palette: dict[str, str],
    confidence_level: float,
    output: Path,
    dpi: int,
    seed: int,
) -> None:
    dataset = str(comparison_row["dataset_folder"])
    reference = str(comparison_row["reference_condition"])
    disease = str(comparison_row["disease_condition"])
    comparison = str(comparison_row["comparison"])
    part = donor_state.loc[
        donor_state["dataset_folder"].astype(str).eq(dataset)
        & donor_state["condition"].astype(str).isin([reference, disease])
    ]
    n_ref = int(part.loc[part["condition"].astype(str).eq(reference), "donor_key"].nunique())
    n_dis = int(part.loc[part["condition"].astype(str).eq(disease), "donor_key"].nunique())

    fig, axes = plt.subplots(3, 3, figsize=(14.8, 12.5), constrained_layout=False)
    rng = np.random.default_rng(stable_seed(seed, comparison, "jitter"))
    for ax, state in zip(axes.flat, states):
        state_part = part.loc[part["LEC_state"].astype(str).eq(state)]
        result = results.loc[
            results["comparison"].eq(comparison) & results["LEC_state"].eq(state)
        ].iloc[0]
        for position, condition in enumerate([reference, disease]):
            values = state_part.loc[
                state_part["condition"].astype(str).eq(condition),
                "state_percentage_of_donor_LECs",
            ].to_numpy(float)
            jitter = rng.uniform(-0.09, 0.09, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=34,
                color=CONDITION_COLORS[condition],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
                zorder=3,
            )
            if condition == reference:
                mean = float(result["reference_mean_percentage"])
                low = float(result["reference_mean_ci_low"])
                high = float(result["reference_mean_ci_high"])
            else:
                mean = float(result["disease_mean_percentage"])
                low = float(result["disease_mean_ci_low"])
                high = float(result["disease_mean_ci_high"])
            ax.errorbar(
                position,
                mean,
                yerr=np.array([[mean - low], [high - mean]]),
                fmt="D",
                markersize=6.5,
                color="#111111",
                markerfacecolor=CONDITION_COLORS[condition],
                markeredgecolor="#111111",
                capsize=4,
                linewidth=1.7,
                zorder=5,
            )
        q_value = float(result["mann_whitney_FDR_within_comparison"])
        title = f"{state}   FDR={q_value:.3g}" if np.isfinite(q_value) else f"{state}   FDR=NA"
        ax.set_title(title, color=palette[state], fontweight="bold")
        ax.set_xticks([0, 1], [f"{reference}\n(n={n_ref})", f"{disease}\n(n={n_dis})"])
        ax.set_ylabel("State % of donor LECs")
        ax.set_ylim(bottom=-1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e3e3e3", lw=0.7, alpha=0.8)
        ax.set_axisbelow(True)
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CONDITION_COLORS[reference],
               markeredgecolor="white", markersize=8, label="Individual donor: " + reference),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CONDITION_COLORS[disease],
               markeredgecolor="white", markersize=8, label="Individual donor: " + disease),
        Line2D([0], [0], marker="D", color="#111111", markerfacecolor="white", markersize=7,
               label=f"Group mean and {confidence_level * 100:.0f}% donor-bootstrap CI"),
    ]
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.07, top=0.86, hspace=0.42, wspace=0.18)
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
    )
    fig.suptitle(
        f"{comparison}: donor-level final-LEC state percentages",
        fontsize=16,
        y=0.985,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_forest_figure(
    results: pd.DataFrame,
    readiness: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    confidence_level: float,
    output: Path,
    dpi: int,
) -> None:
    tested = readiness.loc[readiness["analysis_status"].eq("tested")].reset_index(drop=True)
    fig, axes = plt.subplots(1, len(tested), figsize=(5.5 * len(tested), 6.7), sharey=True, constrained_layout=True)
    if len(tested) == 1:
        axes = np.array([axes])
    y = np.arange(len(states))
    for ax, comparison_row in zip(axes, tested.itertuples(index=False)):
        comparison = str(comparison_row.comparison)
        part = results.loc[results["comparison"].eq(comparison)].set_index("LEC_state").reindex(states)
        effect = part["mean_difference_percentage_points"].to_numpy(float)
        low = part["mean_difference_ci_low"].to_numpy(float)
        high = part["mean_difference_ci_high"].to_numpy(float)
        for index, state in enumerate(states):
            ax.errorbar(
                effect[index],
                y[index],
                xerr=np.array([[effect[index] - low[index]], [high[index] - effect[index]]]),
                fmt="o",
                color=palette[state],
                markeredgecolor="#222222",
                markersize=7,
                capsize=3,
                linewidth=1.5,
            )
            q_value = float(part.loc[state, "mann_whitney_FDR_within_comparison"])
            if np.isfinite(q_value) and q_value < 0.05:
                ax.text(high[index], y[index], "  FDR<0.05", va="center", fontsize=8, fontweight="bold")
        ax.axvline(0, color="#222222", lw=1.0, ls="--")
        ax.set_title(comparison.replace(": ", "\n"))
        ax.set_xlabel(
            f"Disease - NF mean percentage points\n({confidence_level * 100:.0f}% donor-bootstrap CI)"
        )
        ax.grid(axis="x", color="#e3e3e3", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, states)
    for label, state in zip(axes[0].get_yticklabels(), states):
        label.set_color(palette[state])
        label.set_fontweight("bold")
    axes[0].invert_yaxis()
    fig.suptitle("Within-dataset donor-level state-composition effects", fontsize=16)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_statistical_heatmap(
    results: pd.DataFrame,
    readiness: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    output: Path,
    dpi: int,
) -> None:
    comparisons = readiness.loc[readiness["analysis_status"].eq("tested"), "comparison"].tolist()
    effect = results.pivot(index="LEC_state", columns="comparison", values="mean_difference_percentage_points").reindex(index=states, columns=comparisons)
    q_values = results.pivot(index="LEC_state", columns="comparison", values="mann_whitney_FDR_within_comparison").reindex(index=states, columns=comparisons)
    significance = -np.log10(q_values.clip(lower=np.finfo(float).tiny))
    max_abs = float(np.nanmax(np.abs(effect.to_numpy(float))))
    max_abs = max(max_abs, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 7.0), constrained_layout=True)
    first = axes[0].imshow(effect.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    second_max = max(2.0, float(np.nanmax(significance.to_numpy(float))))
    second = axes[1].imshow(significance.to_numpy(float), aspect="auto", cmap="magma", vmin=0, vmax=second_max)
    short_labels = [value.replace("dataset", "D").replace(": ", "\n") for value in comparisons]
    for ax in axes:
        ax.set_xticks(np.arange(len(comparisons)), short_labels)
        ax.set_yticks(np.arange(len(states)), states)
        for label, state in zip(ax.get_yticklabels(), states):
            label.set_color(palette[state])
            label.set_fontweight("bold")
    axes[0].set_title("Disease - NF mean difference\n(percentage points)")
    axes[1].set_title("Mann-Whitney evidence\n(shading: -log10 FDR; labels: FDR)")
    for row in range(len(states)):
        for column in range(len(comparisons)):
            value = effect.iloc[row, column]
            axes[0].text(column, row, f"{value:+.1f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(value) > max_abs * 0.45 else "black")
            q_value = q_values.iloc[row, column]
            label = "NA" if not np.isfinite(q_value) else f"{q_value:.2g}"
            axes[1].text(column, row, label, ha="center", va="center", fontsize=8,
                         color="white" if significance.iloc[row, column] > second_max * 0.45 else "black")
    fig.colorbar(first, ax=axes[0], shrink=0.78, label="Percentage points")
    fig.colorbar(second, ax=axes[1], shrink=0.78, label="-log10 FDR")
    fig.suptitle("Filtered donor-level disease comparisons; datasets are never pooled", fontsize=16)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_dataset11_limitation_figure(
    donor_state: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    output: Path,
    dpi: int,
) -> None:
    part = donor_state.loc[donor_state["dataset_folder"].astype(str).eq("dataset11")]
    hfpef = part.loc[part["condition"].astype(str).eq("HFpEF")]
    medians = hfpef.groupby("LEC_state", observed=True)["state_percentage_of_donor_LECs"].median().reindex(states)
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    if len(hfpef):
        for index, state in enumerate(states):
            values = hfpef.loc[hfpef["LEC_state"].astype(str).eq(state), "state_percentage_of_donor_LECs"].to_numpy(float)
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(np.full(len(values), index) + jitter, values, color=palette[state], edgecolor="white", s=38, zorder=3)
            ax.scatter(index, medians.loc[state], marker="D", color="#111111", s=40, zorder=4)
    ax.set_xticks(np.arange(len(states)), states, rotation=35, ha="right")
    for label, state in zip(ax.get_xticklabels(), states):
        label.set_color(palette[state])
        label.set_fontweight("bold")
    ax.set_ylabel("State % of donor LECs")
    ax.set_title("dataset11 retained HFpEF donors are descriptive only")
    ax.text(
        0.5,
        0.98,
        "No dataset11 NF donor retained >=25 final LECs; HFpEF vs NF is not testable.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox={"facecolor": "#f2f2f2", "edgecolor": "#777777", "boxstyle": "round,pad=0.4"},
    )
    ax.grid(axis="y", color="#e3e3e3", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 1000:
        raise ValueError("--bootstrap_iterations must be at least 1000.")
    if not 0.80 <= args.confidence_level < 1.0:
        raise ValueError("--confidence_level must be in [0.80, 1.0).")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1.")

    project_dir = find_project_dir()
    input_h5ad = resolve_path(args.input_h5ad, project_dir)
    filter_root = resolve_path(args.filter_root, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    sentinel = output_root / "08_4_run_summary.csv"
    if sentinel.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {sentinel}\nUse --force to replace Step 08.4 tables and figures."
        )
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_h5ad}")
    obs, retained, raw_readiness, state_colors, states = load_inputs(
        input_h5ad, filter_root, args
    )
    donor_meta, donor_state = donor_state_tables(obs, retained, states, args)
    nichenet_source = filter_root / "08_3_NicheNet_retained_donor_manifest.csv"
    if not nichenet_source.exists():
        raise FileNotFoundError(
            f"Required Step 08.3 NicheNet donor manifest does not exist: {nichenet_source}"
        )
    nichenet_donors = pd.read_csv(nichenet_source)
    if "analysis_donor_key" not in nichenet_donors.columns:
        raise KeyError("Step 08.3 NicheNet donor manifest lacks analysis_donor_key.")
    manifest_keys = set(nichenet_donors["analysis_donor_key"].astype(str))
    analysis_keys = set(donor_meta[args.donor_key].astype(str))
    if manifest_keys != analysis_keys:
        raise RuntimeError(
            "Step 08.3 NicheNet donor manifest does not exactly match Step 08.4 donors."
        )
    readiness = normalize_readiness(raw_readiness)
    primary, analysis_readiness = make_primary_comparisons(
        donor_state, readiness, states, args
    )
    omnibus = make_dataset01_omnibus(donor_state, states, args.alpha)
    cohort_summary = make_cohort_summary(donor_meta, args)
    cohort_state = make_cohort_state_summary(donor_state)
    palette = state_palette(state_colors, states)

    expected_rows = len(donor_meta) * len(states)
    if len(donor_state) != expected_rows:
        raise RuntimeError(
            f"Donor-state grid has {len(donor_state)} rows; expected {expected_rows}."
        )
    totals = donor_state.groupby("donor_key", observed=True)["state_percentage_of_donor_LECs"].sum()
    if not np.allclose(totals.to_numpy(float), 100.0, atol=1e-8):
        raise RuntimeError("Donor state percentages do not sum to 100%.")
    if donor_meta["n_final_LECs"].min() < 25:
        raise RuntimeError("A Step 08.4 donor has fewer than 25 final LECs.")

    donor_meta.to_csv(tables_dir / "08_4_retained_donor_metadata.csv", index=False)
    nichenet_donors.to_csv(output_root / "08_4_NicheNet_donors_used.csv", index=False)
    donor_state.to_csv(
        tables_dir / "08_4_donor_state_counts_percentages.csv.gz",
        index=False,
        compression="gzip",
    )
    cohort_summary.to_csv(tables_dir / "08_4_cohort_donor_cell_demographics.csv", index=False)
    cohort_state.to_csv(tables_dir / "08_4_cohort_state_descriptive_summary.csv", index=False)
    analysis_readiness.to_csv(tables_dir / "08_4_comparison_readiness.csv", index=False)
    primary.to_csv(tables_dir / "08_4_primary_within_dataset_state_comparisons.csv", index=False)
    omnibus.to_csv(tables_dir / "08_4_dataset01_three_condition_omnibus.csv", index=False)
    state_colors.assign(color_role="LEC_state").to_csv(
        tables_dir / "08_4_state_color_mapping.csv", index=False
    )
    pd.DataFrame(
        [{"condition": key, "color": value, "color_role": "condition"} for key, value in CONDITION_COLORS.items()]
    ).to_csv(tables_dir / "08_4_condition_color_mapping.csv", index=False)

    input_validation = pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "input_h5ad": str(input_h5ad),
                "step08_3_filter_root": str(filter_root),
                "n_final_step08_LECs_used": int(len(obs)),
                "n_retained_donors": int(len(donor_meta)),
                "n_NicheNet_manifest_donors": int(len(nichenet_donors)),
                "minimum_LECs_per_donor": int(donor_meta["n_final_LECs"].min()),
                "n_LEC_states": int(len(states)),
                "donor_state_rows": int(len(donor_state)),
                "expected_donor_state_rows": int(expected_rows),
                "zero_count_donor_state_rows": int(donor_state["n_state_LECs"].eq(0).sum()),
                "ready_comparisons_tested": int(analysis_readiness["analysis_status"].eq("tested").sum()),
                "not_testable_comparisons": int(analysis_readiness["analysis_status"].eq("not_testable").sum()),
                "source_h5ad_modified": False,
                "new_h5ad_written": False,
                "status": "validated_donor_level_min25_within_dataset_analysis",
            }
        ]
    )
    input_validation.to_csv(tables_dir / "08_4_input_validation.csv", index=False)

    figure_manifest: list[dict[str, object]] = []

    def render(filename: str, description: str, callback) -> None:
        path = figures_dir / filename
        try:
            callback(path)
            figure_manifest.append({"figure": filename, "description": description, "status": "created"})
            print(f"Created figure: {path}")
        except Exception as exc:  # pragma: no cover - preserve tabular outputs
            warnings.warn(f"Could not create {filename}: {exc}")
            figure_manifest.append(
                {"figure": filename, "description": description, "status": "failed", "error": str(exc)}
            )

    render(
        "08_4_retained_cohort_support.png",
        "Retained biological-replicate and final-LEC support by cohort",
        lambda path: save_support_figure(cohort_summary, path, args.dpi),
    )
    tested_rows = analysis_readiness.loc[analysis_readiness["analysis_status"].eq("tested")]
    for _, comparison_row in tested_rows.iterrows():
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", str(comparison_row["comparison"])).strip("_")
        render(
            f"08_4_{safe_name}_donor_state_percentages_95CI.png",
            f"Individual donor state percentages and group means with 95% CI for {comparison_row['comparison']}",
            lambda path, row=comparison_row: save_comparison_donor_figure(
                donor_state,
                primary,
                row,
                states,
                palette,
                args.confidence_level,
                path,
                args.dpi,
                args.seed,
            ),
        )
    render(
        "08_4_all_comparisons_effects_95CI.png",
        "Disease-minus-NF donor-level mean state differences with 95% bootstrap confidence intervals",
        lambda path: save_forest_figure(
            primary, analysis_readiness, states, palette, args.confidence_level, path, args.dpi
        ),
    )
    render(
        "08_4_effect_and_FDR_heatmap.png",
        "Effect-size and within-comparison FDR overview",
        lambda path: save_statistical_heatmap(
            primary, analysis_readiness, states, palette, path, args.dpi
        ),
    )
    render(
        "08_4_dataset11_HFpEF_descriptive_not_testable.png",
        "Descriptive retained HFpEF donors with explicit missing-NF limitation",
        lambda path: save_dataset11_limitation_figure(
            donor_state, states, palette, path, args.dpi
        ),
    )
    pd.DataFrame(figure_manifest).to_csv(output_root / "08_4_figure_manifest.csv", index=False)

    not_testable = analysis_readiness.loc[
        analysis_readiness["analysis_status"].eq("not_testable"), "comparison"
    ].tolist()
    notes = f"""Disease Step 08.4 donor-level LEC-state analysis
{'=' * 66}

Scope
-----
Input h5ad (read only): {input_h5ad}
Step 08.3 filter: {filter_root}
Final Step 08 LECs analyzed: {len(obs):,}
Retained donors analyzed: {len(donor_meta)}
LEC states: {len(states)}
Minimum retained donor depth: {int(donor_meta['n_final_LECs'].min())} final LECs

Biological replicate and outcome
--------------------------------
The donor is the biological replicate. For each retained donor and each LEC
state, the response is:

    100 * state LEC count / all final LECs from that donor

Zero-count donor-state pairs are retained. Cell counts are descriptive support;
cells are never treated as independent replicates in hypothesis tests.

Primary comparisons
-------------------
Disease versus NF comparisons remain within dataset. A two-sided Mann-Whitney
U test is calculated independently for each state. Benjamini-Hochberg FDR is
applied across the {len(states)} states within each pre-specified comparison.
A global FDR across all ready comparison-state tests is also provided as a
more conservative sensitivity result.

Effect estimates include disease-minus-NF mean and median percentage-point
differences and rank-biserial correlation. Group means and differences have
{args.confidence_level * 100:.0f}% percentile bootstrap confidence intervals
from {args.bootstrap_iterations:,} resamples of donors within each condition.
These confidence intervals describe donor sampling uncertainty; they do not
correct for observational confounding.

Not testable after the >=25 filter
---------------------------------
{', '.join(not_testable) if not_testable else 'None'}

No cell-pooled substitute is run for a comparison lacking reference donors.
dataset11 HFpEF remains descriptive in this filtered branch because zero
dataset11 NF donors retained at least 25 final LECs.

Interpretation safeguards
-------------------------
1. These results test LEC-state composition, not within-state gene expression.
2. Percentages across states sum to 100%, so states are compositionally linked.
3. Dataset08 has only four retained NF donors and therefore limited power and
   unstable interval estimation even though it is technically testable.
4. The >=25-LEC result is a pre-specified sensitivity branch. Filtering donors
   by recovered LEC abundance can change cohort composition, so retain and
   report the unfiltered donor-level analysis alongside it.
5. No cross-dataset disease-versus-NF comparison is performed. Harmony does not
   turn cells from different datasets into exchangeable biological replicates.
6. A bootstrap CI excluding zero is an estimation summary, not a replacement
   for the multiplicity-adjusted Mann-Whitney result.
"""
    (output_root / "08_4_method_and_interpretation_notes.txt").write_text(
        notes, encoding="utf-8"
    )

    config = {
        "script_version": SCRIPT_VERSION,
        "input_h5ad": str(input_h5ad),
        "filter_root": str(filter_root),
        "output_root": str(output_root),
        "minimum_total_final_LECs_per_donor": 25,
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "confidence_level": float(args.confidence_level),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "primary_test": "two-sided Mann-Whitney U within dataset",
        "primary_multiplicity": "Benjamini-Hochberg across states within each comparison",
        "global_multiplicity_sensitivity": "Benjamini-Hochberg across all ready comparison-state tests",
        "cell_level_inference": False,
        "source_h5ad_modified": False,
        "new_h5ad_written": False,
    }
    (output_root / "08_4_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    run_summary = pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "n_final_LECs": int(len(obs)),
                "n_retained_donors": int(len(donor_meta)),
                "n_states": int(len(states)),
                "n_ready_comparisons_tested": int(tested_rows.shape[0]),
                "n_not_testable_comparisons": int(len(not_testable)),
                "n_primary_state_tests": int(len(primary)),
                "n_significant_within_comparison_FDR": int(primary["significant_within_comparison_FDR"].sum()),
                "n_significant_global_FDR": int(primary["significant_global_FDR"].sum()),
                "bootstrap_iterations": int(args.bootstrap_iterations),
                "confidence_level": float(args.confidence_level),
                "figures_created": int(sum(row["status"] == "created" for row in figure_manifest)),
                "figures_failed": int(sum(row["status"] == "failed" for row in figure_manifest)),
                "status": "completed",
            }
        ]
    )
    run_summary.to_csv(sentinel, index=False)

    print("\nCompleted disease Step 08.4 donor-level state analysis.")
    print(run_summary.to_string(index=False))
    print("\nComparison readiness:")
    print(
        analysis_readiness[
            ["comparison", "n_reference_donors", "n_disease_donors", "analysis_status"]
        ].to_string(index=False)
    )
    if len(primary):
        significant = primary.loc[primary["significant_within_comparison_FDR"]]
        print(f"\nWithin-comparison FDR-significant state tests: {len(significant)}")
        if len(significant):
            print(
                significant[
                    [
                        "comparison",
                        "LEC_state",
                        "mean_difference_percentage_points",
                        "mean_difference_ci_low",
                        "mean_difference_ci_high",
                        "mann_whitney_FDR_within_comparison",
                    ]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()
