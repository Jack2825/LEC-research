#!/usr/bin/env python3
r"""
Step 08.6: learn a donor-level minimum-LEC reliability curve by downsampling.

This script is a read-only downstream analysis of the Step 08.3 LEC h5ad. It
uses donors with at least ``--reference_min_LECs`` cells only as reference
donors. For every reference donor, it treats the donor's full observed 11-state
composition as the reconstruction target and repeatedly downsamples N cells
without replacement.

The primary reconstruction error is total-variation (TV) distance:

    TV (%) = 50 * sum_s |p_downsampled,s - p_full,s|

TV is evaluated on the entire state vector. A TV error of 20% means that 20%
of the composition mass would need to be reassigned to reproduce the full
observed vector. RMSE and maximum absolute state error, both in percentage
points, are retained as secondary diagnostics.

Every reference donor receives the same number of resampling runs, so pooled
donor-run summaries weight reference donors equally rather than weighting them
by their original number of cells. Donor-specific summaries are also exported.

No h5ad, cell labels, state labels, expression values, or embeddings are
changed. The learned cutoff is a sampling-reliability diagnostic, not a new
label written to the source object.

Default input
-------------
new_result2_aging/08_3_equal_LEC_donor_age_quartiles/dataset13/
    08_3_combined_LEC_equal_donor_age_quartiles.h5ad

Default output
--------------
new_result2_aging/08_6_donor_LEC_minimum_reliability/dataset13/

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_aging\08_6_donor_LEC_minimum_reliability.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg", force=True)


SCRIPT_VERSION = "08_6_dataset13_donor_LEC_minimum_reliability_2026_08_10"
DEFAULT_INPUT = (
    "new_result2_aging/08_3_equal_LEC_donor_age_quartiles/dataset13/"
    "08_3_combined_LEC_equal_donor_age_quartiles.h5ad"
)
DEFAULT_OUTPUT_ROOT = (
    "new_result2_aging/08_6_donor_LEC_minimum_reliability/dataset13"
)
DEFAULT_DONOR_KEY = "donor_key"
DEFAULT_STATE_KEY = "LEC_state"
DEFAULT_GROUP_KEY = "age_quartile_equal_LEC_donors"
DEFAULT_N_VALUES = "3,5,8,10,12,15,20,25"
DEFAULT_REFERENCE_MIN = 25
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 8052026
DEFAULT_TARGET_QUANTILE = "p90"
DEFAULT_TARGET_TV_ERROR_PCT = 20.0
TOLERANCE_GRID = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)


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


def parse_n_values(value: str) -> list[int]:
    try:
        values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("N values must be comma-separated integers.") from exc
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("At least one positive N value is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Downsample high-cell reference donors to learn reconstruction error "
            "for the complete donor-level LEC-state vector."
        )
    )
    parser.add_argument("--input_h5ad", default=DEFAULT_INPUT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--donor_key", default=DEFAULT_DONOR_KEY)
    parser.add_argument("--state_key", default=DEFAULT_STATE_KEY)
    parser.add_argument("--group_key", default=DEFAULT_GROUP_KEY)
    parser.add_argument(
        "--reference_min_LECs",
        type=int,
        default=DEFAULT_REFERENCE_MIN,
        help="Minimum full-pool LEC count for learning the reliability curve only.",
    )
    parser.add_argument(
        "--n_values",
        type=parse_n_values,
        default=parse_n_values(DEFAULT_N_VALUES),
        help=f"Comma-separated downsample sizes (default: {DEFAULT_N_VALUES}).",
    )
    parser.add_argument(
        "--n_resamples",
        type=int,
        default=DEFAULT_RESAMPLES,
        help="Independent without-replacement samples per reference donor and N.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--target_quantile",
        choices=("p90", "p95"),
        default=DEFAULT_TARGET_QUANTILE,
        help="Error quantile used for the optional candidate cutoff.",
    )
    parser.add_argument(
        "--target_tv_error_pct",
        type=float,
        default=DEFAULT_TARGET_TV_ERROR_PCT,
        help="Maximum TV error at target_quantile for the candidate cutoff.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing Step 08.6 output files.",
    )
    return parser.parse_args()


def load_obs(input_h5ad: Path) -> pd.DataFrame:
    adata = ad.read_h5ad(input_h5ad, backed="r")
    try:
        return adata.obs.copy()
    finally:
        adata.file.close()


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def prepare_counts(
    obs: pd.DataFrame,
    donor_key: str,
    state_key: str,
    group_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    required = [donor_key, state_key]
    missing = [column for column in required if column not in obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}")

    frame = obs.copy()
    frame[donor_key] = clean_text(frame[donor_key])
    frame[state_key] = clean_text(frame[state_key])
    invalid = (
        frame[donor_key].isna()
        | frame[state_key].isna()
        | frame[donor_key].eq("")
        | frame[state_key].eq("")
    )
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} cells with missing donor/state labels.")

    states = sorted(frame[state_key].unique().tolist(), key=natural_key)
    if len(states) != 11:
        raise ValueError(
            f"Expected the full 11-state vector, but found {len(states)} states: {states}"
        )

    counts = pd.crosstab(frame[donor_key], frame[state_key]).reindex(columns=states, fill_value=0)
    counts.index.name = "donor_key"
    counts = counts.astype(int)

    metadata_candidates = [
        group_key,
        "age_numeric_resolved",
        "age_numeric",
        "Age",
        "Project",
        "Sex",
        "age_quartile",
    ]
    metadata_columns = [column for column in metadata_candidates if column in frame.columns]
    metadata_rows: list[dict[str, object]] = []
    for donor, donor_frame in frame.groupby(donor_key, sort=False, observed=True):
        row: dict[str, object] = {"donor_key": donor}
        for column in metadata_columns:
            values = donor_frame[column].dropna().astype("string").str.strip()
            values = values.loc[values.ne("")].unique().tolist()
            row[column] = values[0] if len(values) == 1 else " | ".join(sorted(values))
        metadata_rows.append(row)
    donor_meta = pd.DataFrame(metadata_rows).set_index("donor_key")
    donor_meta = donor_meta.reindex(counts.index)
    donor_meta["n_total_LEC_cells"] = counts.sum(axis=1).astype(int)
    donor_meta = donor_meta.reset_index()
    return donor_meta, counts, states


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def run_resampling(
    counts: pd.DataFrame,
    reference_donors: list[str],
    states: list[str],
    n_values: list[int],
    n_resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    run_parts: list[pd.DataFrame] = []
    donor_summary_rows: list[dict[str, object]] = []
    state_errors: dict[int, list[np.ndarray]] = {n: [] for n in n_values}

    for donor_index, donor in enumerate(reference_donors):
        full_counts = counts.loc[donor, states].to_numpy(dtype=np.int64)
        full_total = int(full_counts.sum())
        full_proportion = full_counts / full_total

        for n in n_values:
            draws = rng.multivariate_hypergeometric(full_counts, n, size=n_resamples)
            sampled_proportion = draws / float(n)
            absolute_pp = np.abs(sampled_proportion - full_proportion) * 100.0
            tv_pct = 0.5 * absolute_pp.sum(axis=1)
            rmse_pp = np.sqrt(np.mean(np.square(absolute_pp), axis=1))
            max_abs_pp = absolute_pp.max(axis=1)
            state_errors[n].append(absolute_pp)

            run_parts.append(
                pd.DataFrame(
                    {
                        "donor_key": donor,
                        "donor_order": donor_index + 1,
                        "full_donor_LECs": full_total,
                        "n_downsample": n,
                        "resample_run": np.arange(1, n_resamples + 1, dtype=int),
                        "tv_error_pct": tv_pct,
                        "rmse_state_percentage_points": rmse_pp,
                        "max_absolute_state_error_percentage_points": max_abs_pp,
                    }
                )
            )

            row: dict[str, object] = {
                "donor_key": donor,
                "full_donor_LECs": full_total,
                "n_downsample": n,
                "n_resamples": n_resamples,
            }
            for metric_name, values in (
                ("tv_error_pct", tv_pct),
                ("rmse_state_percentage_points", rmse_pp),
                ("max_absolute_state_error_percentage_points", max_abs_pp),
            ):
                row[f"{metric_name}_median"] = quantile(values, 0.50)
                row[f"{metric_name}_p90"] = quantile(values, 0.90)
                row[f"{metric_name}_p95"] = quantile(values, 0.95)
            donor_summary_rows.append(row)

    all_runs = pd.concat(run_parts, ignore_index=True)
    donor_summary = pd.DataFrame(donor_summary_rows)

    state_rows: list[dict[str, object]] = []
    for n in n_values:
        errors = np.concatenate(state_errors[n], axis=0)
        for state_index, state in enumerate(states):
            values = errors[:, state_index]
            state_rows.append(
                {
                    "n_downsample": n,
                    "LEC_state": state,
                    "n_reference_donor_runs": int(len(values)),
                    "absolute_error_percentage_points_median": quantile(values, 0.50),
                    "absolute_error_percentage_points_p90": quantile(values, 0.90),
                    "absolute_error_percentage_points_p95": quantile(values, 0.95),
                }
            )
    state_summary = pd.DataFrame(state_rows)
    return all_runs, donor_summary, state_summary


def make_pooled_summary(
    all_runs: pd.DataFrame,
    donor_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for n, frame in all_runs.groupby("n_downsample", sort=True):
        donor_frame = donor_summary.loc[donor_summary["n_downsample"].eq(n)]
        row: dict[str, object] = {
            "n_downsample": int(n),
            "n_reference_donors": int(frame["donor_key"].nunique()),
            "n_resamples_per_donor": int(frame.groupby("donor_key").size().iloc[0]),
            "n_donor_resample_errors": int(len(frame)),
        }
        for metric in (
            "tv_error_pct",
            "rmse_state_percentage_points",
            "max_absolute_state_error_percentage_points",
        ):
            values = frame[metric].to_numpy(float)
            row[f"{metric}_median"] = quantile(values, 0.50)
            row[f"{metric}_p90"] = quantile(values, 0.90)
            row[f"{metric}_p95"] = quantile(values, 0.95)
            for label in ("median", "p90", "p95"):
                donor_values = donor_frame[f"{metric}_{label}"].to_numpy(float)
                row[f"median_across_donors_of_donor_{metric}_{label}"] = quantile(
                    donor_values, 0.50
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_downsample").reset_index(drop=True)


def make_threshold_lookup(summary: pd.DataFrame) -> pd.DataFrame:
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


def make_reference_threshold_sensitivity(
    all_runs: pd.DataFrame,
    reference_min: int,
) -> pd.DataFrame:
    """Show the finite-population effect from requiring deeper reference pools."""
    observed_max = int(all_runs["full_donor_LECs"].max())
    candidate_thresholds = sorted(
        {
            reference_min,
            max(reference_min, 40),
            max(reference_min, 50),
        }
    )
    rows: list[dict[str, object]] = []
    for threshold in candidate_thresholds:
        if threshold > observed_max:
            continue
        selected = all_runs.loc[all_runs["full_donor_LECs"].ge(threshold)]
        n_donors = int(selected["donor_key"].nunique())
        if n_donors < 4:
            continue
        for n, frame in selected.groupby("n_downsample", sort=True):
            values = frame["tv_error_pct"].to_numpy(float)
            rows.append(
                {
                    "stricter_reference_min_LECs": threshold,
                    "n_reference_donors": n_donors,
                    "n_downsample": int(n),
                    "tv_error_pct_median": quantile(values, 0.50),
                    "tv_error_pct_p90": quantile(values, 0.90),
                    "tv_error_pct_p95": quantile(values, 0.95),
                }
            )
    return pd.DataFrame(rows)


def make_filter_impact(
    donor_meta: pd.DataFrame,
    n_values: list[int],
    group_key: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_donors = len(donor_meta)
    total_cells = int(donor_meta["n_total_LEC_cells"].sum())
    group_values: list[str] = []
    if group_key in donor_meta.columns:
        group_values = sorted(
            donor_meta[group_key].dropna().astype(str).unique().tolist(), key=natural_key
        )
    for n in n_values:
        retained = donor_meta.loc[donor_meta["n_total_LEC_cells"].ge(n)]
        retained_cells = int(retained["n_total_LEC_cells"].sum())
        row: dict[str, object] = {
            "minimum_LECs": n,
            "n_donors_retained": int(len(retained)),
            "percent_donors_retained": 100.0 * len(retained) / total_donors,
            "n_LEC_cells_retained": retained_cells,
            "percent_LEC_cells_retained": 100.0 * retained_cells / total_cells,
        }
        for group in group_values:
            safe_group = re.sub(r"[^A-Za-z0-9]+", "_", group).strip("_")
            row[f"n_donors_retained_{safe_group}"] = int(
                retained[group_key].astype(str).eq(group).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def get_recommendation(
    summary: pd.DataFrame,
    target_quantile: str,
    target_error: float,
) -> int | None:
    column = f"tv_error_pct_{target_quantile}"
    passing = summary.loc[summary[column].le(target_error)].sort_values("n_downsample")
    return None if passing.empty else int(passing.iloc[0]["n_downsample"])


def make_reference_composition(
    counts: pd.DataFrame,
    reference_donors: list[str],
    states: list[str],
) -> pd.DataFrame:
    selected = counts.loc[reference_donors, states].copy()
    long = selected.reset_index().melt(
        id_vars="donor_key", var_name="LEC_state", value_name="n_state_cells"
    )
    totals = selected.sum(axis=1).rename("n_total_LEC_cells")
    long = long.merge(totals, left_on="donor_key", right_index=True, how="left")
    long["state_percentage"] = 100.0 * long["n_state_cells"] / long["n_total_LEC_cells"]
    return long


def save_primary_curve(
    summary: pd.DataFrame,
    target_quantile: str,
    target_error: float,
    recommendation: int | None,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    x = summary["n_downsample"].to_numpy(float)
    colors = {"median": "#1f77b4", "p90": "#ff7f0e", "p95": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    for label in ("median", "p90", "p95"):
        axes[0].plot(
            x,
            summary[f"tv_error_pct_{label}"],
            marker="o",
            linewidth=2.2,
            color=colors[label],
            label=label.upper(),
        )
    axes[0].axhline(target_error, color="#555555", linestyle="--", linewidth=1.4)
    if recommendation is not None:
        axes[0].axvline(recommendation, color="#2ca02c", linestyle=":", linewidth=1.8)
        axes[0].annotate(
            f"candidate N = {recommendation}",
            xy=(recommendation, target_error),
            xytext=(6, 14),
            textcoords="offset points",
            color="#247a24",
        )
    axes[0].set_title("Full 11-state vector reconstruction")
    axes[0].set_xlabel("Downsampled LECs per donor (N)")
    axes[0].set_ylabel("Total-variation error (%)")
    axes[0].set_xticks(x)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(title="Across donor-runs")

    for metric, label, color in (
        ("rmse_state_percentage_points_p90", "RMSE P90", "#9467bd"),
        (
            "max_absolute_state_error_percentage_points_p90",
            "Maximum state error P90",
            "#8c564b",
        ),
    ):
        axes[1].plot(x, summary[metric], marker="o", linewidth=2.2, label=label, color=color)
    axes[1].set_title("Secondary state-level error diagnostics")
    axes[1].set_xlabel("Downsampled LECs per donor (N)")
    axes[1].set_ylabel("Error (percentage points)")
    axes[1].set_xticks(x)
    axes[1].set_ylim(bottom=0)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle(
        f"Step 08.6 donor downsampling reliability; target: {target_quantile.upper()} TV <= {target_error:g}%",
        fontsize=14,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_donor_curves(
    donor_summary: pd.DataFrame,
    pooled_summary: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    for donor, frame in donor_summary.groupby("donor_key", sort=False):
        ordered = frame.sort_values("n_downsample")
        axes[0].plot(
            ordered["n_downsample"],
            ordered["tv_error_pct_median"],
            color="#8c8c8c",
            linewidth=1,
            alpha=0.55,
        )
        axes[1].plot(
            ordered["n_downsample"],
            ordered["tv_error_pct_p90"],
            color="#8c8c8c",
            linewidth=1,
            alpha=0.55,
        )
    x = pooled_summary["n_downsample"]
    axes[0].plot(x, pooled_summary["tv_error_pct_median"], "o-", color="#1f77b4", linewidth=2.6)
    axes[1].plot(x, pooled_summary["tv_error_pct_p90"], "o-", color="#ff7f0e", linewidth=2.6)
    for ax, title in zip(axes, ("Median error per reference donor", "P90 error per reference donor")):
        ax.set_title(title)
        ax.set_xlabel("Downsampled LECs per donor (N)")
        ax.set_ylabel("Total-variation error (%)")
        ax.set_xticks(x)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Reference-donor heterogeneity (gray) and pooled equal-donor curve (color)")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_state_heatmap(
    state_summary: pd.DataFrame,
    states: list[str],
    n_values: list[int],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    matrix = state_summary.pivot(
        index="LEC_state",
        columns="n_downsample",
        values="absolute_error_percentage_points_p90",
    ).reindex(index=states, columns=n_values)
    fig, ax = plt.subplots(figsize=(10.5, 6.6), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(np.arange(len(n_values)), labels=n_values)
    ax.set_yticks(np.arange(len(states)), labels=states)
    ax.set_xlabel("Downsampled LECs per donor (N)")
    ax.set_ylabel("LEC state")
    ax.set_title("P90 absolute reconstruction error for each state")
    for row in range(len(states)):
        for column in range(len(n_values)):
            value = matrix.iloc[row, column]
            ax.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Absolute error (percentage points)", fraction=0.046)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_reference_threshold_sensitivity(
    sensitivity: pd.DataFrame,
    target_error: float,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.8, 5.5), constrained_layout=True)
    colors = plt.get_cmap("Dark2")(
        np.linspace(0, 1, sensitivity["stricter_reference_min_LECs"].nunique())
    )
    for (threshold, frame), color in zip(
        sensitivity.groupby("stricter_reference_min_LECs", sort=True), colors
    ):
        ordered = frame.sort_values("n_downsample")
        n_donors = int(ordered["n_reference_donors"].iloc[0])
        ax.plot(
            ordered["n_downsample"],
            ordered["tv_error_pct_p90"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"reference >= {int(threshold)} (n={n_donors})",
        )
    ax.axhline(target_error, color="#555555", linestyle="--", linewidth=1.4)
    ax.set_xlabel("Downsampled LECs per donor (N)")
    ax.set_ylabel("P90 total-variation error (%)")
    ax.set_title("Sensitivity to reference-donor depth")
    ax.set_xticks(sorted(sensitivity["n_downsample"].unique()))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_filter_impact(
    impact: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    x = impact["minimum_LECs"].to_numpy(int)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].plot(x, impact["n_donors_retained"], "o-", color="#1f77b4", linewidth=2.2)
    for n, value in zip(x, impact["n_donors_retained"]):
        axes[0].annotate(str(int(value)), (n, value), xytext=(0, 6), textcoords="offset points", ha="center")
    axes[0].set_xlabel("Candidate minimum LECs per donor")
    axes[0].set_ylabel("Donor entries retained")
    axes[0].set_title("Biological-replicate retention")
    axes[0].set_xticks(x)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(
        x,
        impact["percent_donors_retained"],
        "o-",
        color="#ff7f0e",
        linewidth=2.2,
        label="Donor entries",
    )
    axes[1].plot(
        x,
        impact["percent_LEC_cells_retained"],
        "o-",
        color="#2ca02c",
        linewidth=2.2,
        label="LEC cells",
    )
    axes[1].set_xlabel("Candidate minimum LECs per donor")
    axes[1].set_ylabel("Retained (%)")
    axes[1].set_title("Donor-versus-cell retention tradeoff")
    axes[1].set_xticks(x)
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_reference_diagnostics(
    donor_meta: pd.DataFrame,
    reference_composition: pd.DataFrame,
    reference_min: int,
    upper_quartile: float,
    states: list[str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    reference_keys = reference_composition["donor_key"].drop_duplicates().tolist()
    ref_meta = donor_meta.set_index("donor_key").loc[reference_keys].sort_values("n_total_LEC_cells")
    comp = reference_composition.pivot(
        index="donor_key", columns="LEC_state", values="state_percentage"
    ).reindex(index=ref_meta.index, columns=states, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    all_counts = donor_meta["n_total_LEC_cells"].to_numpy(int)
    bins = np.arange(0.5, all_counts.max() + 2.5, max(1, math.ceil(all_counts.max() / 25)))
    axes[0].hist(all_counts, bins=bins, color="#6baed6", edgecolor="white")
    axes[0].axvline(reference_min, color="#d62728", linestyle="--", label=f"Reference >= {reference_min}")
    axes[0].axvline(upper_quartile, color="#2ca02c", linestyle=":", label=f"Observed Q3 = {upper_quartile:.2f}")
    axes[0].set_xlabel("Observed LECs per donor entry")
    axes[0].set_ylabel("Number of donor entries")
    axes[0].set_title("All-donor sampling distribution")
    axes[0].legend()

    bottom = np.zeros(len(comp))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(states)))
    for state, color in zip(states, colors):
        values = comp[state].to_numpy(float)
        axes[1].bar(np.arange(len(comp)), values, bottom=bottom, label=state, color=color, width=0.82)
        bottom += values
    labels = [f"{key.split('::')[-1]}\n(n={int(ref_meta.loc[key, 'n_total_LEC_cells'])})" for key in comp.index]
    axes[1].set_xticks(np.arange(len(comp)), labels=labels, rotation=65, ha="right", fontsize=8)
    axes[1].set_ylabel("Full observed donor composition (%)")
    axes[1].set_title("Reference donors used to learn the curve")
    axes[1].set_ylim(0, 100)
    axes[1].legend(title="LEC state", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    path: Path,
    input_h5ad: Path,
    states: list[str],
    donor_meta: pd.DataFrame,
    reference_min: int,
    upper_quartile: float,
    reference_donors: list[str],
    n_values: list[int],
    n_resamples: int,
    seed: int,
    target_quantile: str,
    target_error: float,
    recommendation: int | None,
) -> None:
    recommendation_text = (
        str(recommendation)
        if recommendation is not None
        else "not reached within the tested N grid"
    )
    text = f"""Step 08.6 donor LEC minimum reliability analysis
=====================================================

Input (read only): {input_h5ad}
States: {', '.join(states)}
All donor entries: {len(donor_meta)}
Observed donor LEC-count upper quartile: {upper_quartile:.2f}
Reference criterion: at least {reference_min} LECs
Reference donors: {len(reference_donors)}
Tested N: {', '.join(map(str, n_values))}
Resampling runs per reference donor per N: {n_resamples}
Random seed: {seed}

Method
------
Each eligible reference donor was analyzed independently. For each N, N cells
were sampled without replacement from that donor's observed LEC pool. The
downsampled 11-state composition was compared with that same donor's full
observed 11-state composition. Reference donors received equal numbers of runs.

Primary error
-------------
TV (%) = 50 * sum over states of |downsampled proportion - full proportion|.
The reported median, P90, and P95 are calculated over the pooled donor-run
errors. Because every reference donor has exactly the same number of runs, this
is equal-donor weighted. Donor-specific quantiles are exported separately.

Candidate rule
--------------
Configured target: {target_quantile.upper()} TV <= {target_error:g}%.
Smallest tested N meeting that target: {recommendation_text}.
This target is a transparent user-configurable tolerance, not a biological law.
The threshold lookup table shows how the candidate N changes under other error
tolerances and under P90 versus P95 control.

Interpretation limits
---------------------
This estimates how well N cells reconstruct the observed state composition of
well-sampled donors. The full observed vector is a proxy target, not error-free
biological truth. The analysis does not establish that a low-cell donor is
biologically invalid, does not test age association, and does not correct batch
effects. Without-replacement errors can be optimistic when N approaches the
reference donor's full cell count. The stricter-reference sensitivity table and
figure therefore repeat the summaries using reference pools of at least 40 and
50 LECs where available. A chosen minimum should be pre-specified without examining age-group
p-values, followed by reporting retained donors by age group/project/sex and a
sensitivity analysis using both all donors and the filtered donor set.

No source file or label was changed.
"""
    path.write_text(text, encoding="utf-8")


def ensure_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Use --overwrite to replace 08.6 files."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    if args.reference_min_LECs < 2:
        raise ValueError("--reference_min_LECs must be at least 2.")
    if args.n_resamples < 100:
        raise ValueError("--n_resamples must be at least 100 for stable tail summaries.")
    if max(args.n_values) > args.reference_min_LECs:
        raise ValueError(
            "Every tested N must be <= --reference_min_LECs so the same reference donors "
            "contribute at every N."
        )
    if not (0 < args.target_tv_error_pct < 100):
        raise ValueError("--target_tv_error_pct must be between 0 and 100.")

    project_dir = find_project_dir()
    input_h5ad = resolve_path(args.input_h5ad, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    if not input_h5ad.is_file():
        raise FileNotFoundError(f"Input h5ad does not exist: {input_h5ad}")

    obs = load_obs(input_h5ad)
    donor_meta, counts, states = prepare_counts(
        obs, args.donor_key, args.state_key, args.group_key
    )
    upper_quartile = float(donor_meta["n_total_LEC_cells"].quantile(0.75))
    donor_meta["is_reference_donor"] = donor_meta["n_total_LEC_cells"].ge(
        args.reference_min_LECs
    )
    reference_meta = donor_meta.loc[donor_meta["is_reference_donor"]].sort_values(
        ["n_total_LEC_cells", "donor_key"]
    )
    reference_donors = reference_meta["donor_key"].tolist()
    if len(reference_donors) < 8:
        raise ValueError(
            f"Only {len(reference_donors)} reference donors meet >= {args.reference_min_LECs}; "
            "at least 8 are required for a useful donor-balanced curve."
        )

    print(f"Input cells: {len(obs):,}")
    print(f"Donor entries: {len(donor_meta)}")
    print(f"States ({len(states)}): {', '.join(states)}")
    print(f"Observed LEC-count Q3: {upper_quartile:.2f}")
    print(
        f"Reference donors >= {args.reference_min_LECs} LECs: {len(reference_donors)} "
        f"(range {reference_meta['n_total_LEC_cells'].min()}-"
        f"{reference_meta['n_total_LEC_cells'].max()})"
    )
    print(f"N grid: {args.n_values}; resamples/donor/N: {args.n_resamples}")
    if args.dry_run:
        print("Dry run complete; no files written.")
        return 0

    ensure_output(output_root, args.overwrite)
    all_runs, donor_summary, state_summary = run_resampling(
        counts,
        reference_donors,
        states,
        args.n_values,
        args.n_resamples,
        args.seed,
    )
    pooled_summary = make_pooled_summary(all_runs, donor_summary)
    threshold_lookup = make_threshold_lookup(pooled_summary)
    reference_sensitivity = make_reference_threshold_sensitivity(
        all_runs, args.reference_min_LECs
    )
    filter_impact = make_filter_impact(donor_meta, args.n_values, args.group_key)
    recommendation = get_recommendation(
        pooled_summary, args.target_quantile, args.target_tv_error_pct
    )
    reference_composition = make_reference_composition(counts, reference_donors, states)
    donor_meta["meets_candidate_minimum"] = (
        donor_meta["n_total_LEC_cells"].ge(recommendation)
        if recommendation is not None
        else False
    )

    donor_meta.to_csv(output_root / "08_6_all_donor_LEC_counts.csv", index=False)
    reference_meta.to_csv(output_root / "08_6_reference_donors.csv", index=False)
    reference_composition.to_csv(
        output_root / "08_6_reference_donor_state_composition.csv", index=False
    )
    all_runs.to_csv(
        output_root / "08_6_resampling_error_all_donor_runs.csv.gz",
        index=False,
        compression="gzip",
    )
    donor_summary.to_csv(
        output_root / "08_6_error_summary_by_reference_donor.csv", index=False
    )
    pooled_summary.to_csv(
        output_root / "08_6_error_summary_equal_donor_pooled.csv", index=False
    )
    state_summary.to_csv(
        output_root / "08_6_state_absolute_error_summary.csv", index=False
    )
    threshold_lookup.to_csv(
        output_root / "08_6_candidate_minimum_threshold_lookup.csv", index=False
    )
    reference_sensitivity.to_csv(
        output_root / "08_6_stricter_reference_depth_sensitivity.csv", index=False
    )
    filter_impact.to_csv(
        output_root / "08_6_candidate_filter_retention_impact.csv", index=False
    )

    figures_dir = output_root / "figures"
    manifest_rows: list[dict[str, str]] = []
    if not args.skip_figures:
        figures = [
            (
                "08_6_full_vector_reconstruction_error_curve.png",
                "Median, P90, and P95 full-vector TV error plus secondary errors",
                lambda path: save_primary_curve(
                    pooled_summary,
                    args.target_quantile,
                    args.target_tv_error_pct,
                    recommendation,
                    path,
                    args.dpi,
                ),
            ),
            (
                "08_6_reference_donor_specific_error_curves.png",
                "Reference-donor heterogeneity in median and P90 TV error",
                lambda path: save_donor_curves(
                    donor_summary, pooled_summary, path, args.dpi
                ),
            ),
            (
                "08_6_state_specific_P90_absolute_error_heatmap.png",
                "P90 absolute percentage-point error by state and N",
                lambda path: save_state_heatmap(
                    state_summary, states, args.n_values, path, args.dpi
                ),
            ),
            (
                "08_6_reference_donor_diagnostics.png",
                "All-donor LEC counts and full compositions of reference donors",
                lambda path: save_reference_diagnostics(
                    donor_meta,
                    reference_composition,
                    args.reference_min_LECs,
                    upper_quartile,
                    states,
                    path,
                    args.dpi,
                ),
            ),
            (
                "08_6_stricter_reference_depth_sensitivity.png",
                "P90 TV curve under reference thresholds of at least 25, 40, and 50 LECs",
                lambda path: save_reference_threshold_sensitivity(
                    reference_sensitivity,
                    args.target_tv_error_pct,
                    path,
                    args.dpi,
                ),
            ),
            (
                "08_6_candidate_filter_retention_impact.png",
                "Donor and LEC-cell retention under each candidate minimum",
                lambda path: save_filter_impact(filter_impact, path, args.dpi),
            ),
        ]
        for filename, description, function in figures:
            path = figures_dir / filename
            function(path)
            manifest_rows.append(
                {"figure": str(path), "description": description, "status": "ok"}
            )
    pd.DataFrame(manifest_rows, columns=["figure", "description", "status"]).to_csv(
        output_root / "08_6_figure_manifest.csv", index=False
    )

    write_notes(
        output_root / "08_6_method_and_interpretation_notes.txt",
        input_h5ad,
        states,
        donor_meta,
        args.reference_min_LECs,
        upper_quartile,
        reference_donors,
        args.n_values,
        args.n_resamples,
        args.seed,
        args.target_quantile,
        args.target_tv_error_pct,
        recommendation,
    )
    config = {
        "script_version": SCRIPT_VERSION,
        "input_h5ad": str(input_h5ad),
        "output_root": str(output_root),
        "donor_key": args.donor_key,
        "state_key": args.state_key,
        "group_key": args.group_key,
        "states": states,
        "reference_min_LECs": args.reference_min_LECs,
        "observed_LEC_count_upper_quartile": upper_quartile,
        "n_reference_donors": len(reference_donors),
        "n_values": args.n_values,
        "n_resamples_per_reference_donor_per_N": args.n_resamples,
        "seed": args.seed,
        "sampling": "without replacement independently within each reference donor",
        "primary_error": "total variation distance on full state vector, percent",
        "target_quantile": args.target_quantile,
        "target_tv_error_pct": args.target_tv_error_pct,
        "candidate_minimum_N": recommendation,
        "source_modified": False,
    }
    (output_root / "08_6_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    run_summary = pd.DataFrame(
        [
            {
                "n_cells": len(obs),
                "n_donor_entries": len(donor_meta),
                "n_states": len(states),
                "observed_LEC_count_upper_quartile": upper_quartile,
                "reference_min_LECs": args.reference_min_LECs,
                "n_reference_donors": len(reference_donors),
                "n_resamples_per_donor_per_N": args.n_resamples,
                "target_quantile": args.target_quantile,
                "target_tv_error_pct": args.target_tv_error_pct,
                "candidate_minimum_N": recommendation,
            }
        ]
    )
    run_summary.to_csv(output_root / "08_6_run_summary.csv", index=False)

    print("\nEqual-donor pooled TV reconstruction error:")
    print(
        pooled_summary[
            [
                "n_downsample",
                "tv_error_pct_median",
                "tv_error_pct_p90",
                "tv_error_pct_p95",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )
    print(
        f"\nCandidate minimum for {args.target_quantile.upper()} TV <= "
        f"{args.target_tv_error_pct:g}%: "
        f"{recommendation if recommendation is not None else 'not reached'}"
    )
    print(f"Outputs: {output_root}")
    print("Source h5ad was not modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
