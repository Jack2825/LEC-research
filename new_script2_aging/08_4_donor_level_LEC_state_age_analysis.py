#!/usr/bin/env python3
r"""
Step 08.4: donor-level LEC-state percentages and age-group statistics.

This is a read-only downstream analysis of the Step 08.3 h5ad. For every donor
and every LEC state (including donor-state combinations with zero cells), it
calculates:

    cell-state percentage = state cells / all LEC cells from donor * 100

The donor is the biological replicate. The script plots individual donor
percentages with group mean +/- SEM and performs a Kruskal-Wallis test across
the four equal-LEC-donor age groups for every state. Benjamini-Hochberg FDR is
applied across states. States passing the omnibus FDR threshold receive Dunn
pairwise tests with Holm correction.

Sensitivity outputs include Welch one-way ANOVA, continuous-age Spearman tests,
leave-the-largest-state-donor-out tests, a minimum-LEC-count donor filter, and a
Project/Sex-adjusted robust OLS age-group test. These are reported separately
and do not replace the pre-specified primary Kruskal-Wallis analysis.

No cells, expression values, embeddings, clusters, state labels, or h5ad files
are changed.

Default input
-------------
new_result2_aging/08_3_equal_LEC_donor_age_quartiles/dataset13/
    08_3_combined_LEC_equal_donor_age_quartiles.h5ad

Default output
--------------
new_result2_aging/08_4_donor_level_LEC_state_age_analysis/dataset13/

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_aging\08_4_donor_level_LEC_state_age_analysis.py
"""

from __future__ import annotations

import argparse
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
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway


matplotlib.use("Agg", force=True)


SCRIPT_VERSION = "08_4_dataset13_donor_level_LEC_state_age_analysis_2026_08_09"
DEFAULT_INPUT = (
    "new_result2_aging/08_3_equal_LEC_donor_age_quartiles/dataset13/"
    "08_3_combined_LEC_equal_donor_age_quartiles.h5ad"
)
DEFAULT_OUTPUT_ROOT = (
    "new_result2_aging/08_4_donor_level_LEC_state_age_analysis/dataset13"
)
DEFAULT_GROUP_KEY = "age_quartile_equal_LEC_donors"
DEFAULT_STATE_KEY = "LEC_state"
DEFAULT_DONOR_KEY = "donor_key"
DEFAULT_AGE_KEY = "age_numeric_resolved"
DEFAULT_PROJECT_KEY = "Project"
DEFAULT_SEX_KEY = "Sex"
DEFAULT_ORIGINAL_GROUP_KEY = "age_quartile"
GROUPS = ["Q1", "Q2", "Q3", "Q4"]
GROUP_COLORS = {
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
            "Calculate donor-level LEC-state percentages, plot individual donors "
            "with mean +/- SEM, and test four equal-donor age groups."
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
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--min_donor_LECs_sensitivity",
        type=int,
        default=5,
        help="Minimum donor LEC count used only in the low-count sensitivity test.",
    )
    parser.add_argument(
        "--expected_donors_per_group",
        type=int,
        default=12,
        help="Validation target for the Step 08.3 equal-donor groups.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing Step 08.4 output files.",
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


def format_p(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.2e}" if number < 0.001 else f"{number:.3f}"


def add_fdr(
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


def prepare_donor_state_table(
    obs: pd.DataFrame,
    donor_key: str,
    state_key: str,
    group_key: str,
    age_key: str,
    project_key: str,
    sex_key: str,
    original_group_key: str,
    expected_donors_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    required = [donor_key, state_key, group_key, age_key, project_key, sex_key]
    missing = [column for column in required if column not in obs]
    if missing:
        raise KeyError(f"Input h5ad is missing required obs columns: {missing}")

    working = pd.DataFrame(index=obs.index.copy())
    working["donor_key"] = normalized_text(obs[donor_key])
    working["LEC_state"] = normalized_text(obs[state_key])
    working["age_group"] = normalized_text(obs[group_key])
    working["age_numeric"] = pd.to_numeric(obs[age_key], errors="coerce")
    working["Project"] = normalized_text(obs[project_key])
    working["Sex"] = normalized_text(obs[sex_key])
    if original_group_key in obs:
        working["original_age_group"] = normalized_text(obs[original_group_key])
    else:
        working["original_age_group"] = "not_available"

    for column in ["donor_key", "LEC_state", "age_group", "Project", "Sex"]:
        if working[column].eq("").any():
            raise ValueError(f"Empty values found in required field {column!r}.")
    if working["age_numeric"].isna().any():
        raise ValueError("Non-numeric or missing donor ages were found.")

    observed_groups = sorted(working["age_group"].unique(), key=natural_key)
    if observed_groups != GROUPS:
        raise ValueError(
            f"Expected age groups {GROUPS}, but observed {observed_groups}. "
            "Check --group_key and the Step 08.3 input."
        )

    consistency_columns = [
        "age_group",
        "age_numeric",
        "Project",
        "Sex",
        "original_age_group",
    ]
    consistency = working.groupby("donor_key", observed=True)[
        consistency_columns
    ].nunique(dropna=False)
    inconsistent = consistency.gt(1).any(axis=1)
    if inconsistent.any():
        donors = consistency.index[inconsistent].tolist()
        raise ValueError(
            "Donor-level metadata are not constant within donor for: "
            f"{donors[:10]}"
        )

    donor_meta = (
        working.groupby("donor_key", observed=True)
        .agg(
            age_group=("age_group", "first"),
            age_numeric=("age_numeric", "first"),
            Project=("Project", "first"),
            Sex=("Sex", "first"),
            original_age_group=("original_age_group", "first"),
            n_total_LEC_cells=("LEC_state", "size"),
        )
        .reset_index()
    )
    donor_meta["age_group"] = pd.Categorical(
        donor_meta["age_group"], categories=GROUPS, ordered=True
    )
    group_counts = donor_meta.groupby("age_group", observed=False).size().reindex(GROUPS)
    if not group_counts.eq(expected_donors_per_group).all():
        raise ValueError(
            "The equal-donor validation failed. Observed donor counts: "
            f"{group_counts.to_dict()}; expected {expected_donors_per_group} each."
        )

    states = sorted(working["LEC_state"].unique(), key=natural_key)
    observed_counts = (
        working.groupby(["donor_key", "LEC_state"], observed=True)
        .size()
        .rename("n_state_cells")
    )
    full_index = pd.MultiIndex.from_product(
        [sorted(donor_meta["donor_key"]), states],
        names=["donor_key", "LEC_state"],
    )
    donor_state = observed_counts.reindex(full_index, fill_value=0).reset_index()
    donor_state = donor_state.merge(
        donor_meta,
        on="donor_key",
        how="left",
        validate="many_to_one",
    )
    donor_state["n_state_cells"] = donor_state["n_state_cells"].astype(int)
    donor_state["cell_state_fraction"] = (
        donor_state["n_state_cells"] / donor_state["n_total_LEC_cells"]
    )
    donor_state["cell_state_percentage"] = 100.0 * donor_state["cell_state_fraction"]
    donor_state["state_present"] = donor_state["n_state_cells"].gt(0)
    donor_state["age_group"] = pd.Categorical(
        donor_state["age_group"], categories=GROUPS, ordered=True
    )
    donor_state["LEC_state"] = pd.Categorical(
        donor_state["LEC_state"], categories=states, ordered=True
    )
    donor_state = donor_state.sort_values(
        ["LEC_state", "age_group", "age_numeric", "donor_key"]
    ).reset_index(drop=True)

    state_sums = donor_state.groupby("donor_key", observed=True)[
        "n_state_cells"
    ].sum()
    donor_totals = donor_meta.set_index("donor_key")["n_total_LEC_cells"]
    if not state_sums.eq(donor_totals).all():
        raise RuntimeError("Donor-state cell counts do not sum to donor LEC totals.")
    percentage_sums = donor_state.groupby("donor_key", observed=True)[
        "cell_state_percentage"
    ].sum()
    if not np.allclose(percentage_sums, 100.0, atol=1e-8):
        raise RuntimeError("Donor state percentages do not sum to 100%.")

    range_labels: dict[str, str] = {}
    for group in GROUPS:
        frame = donor_meta.loc[donor_meta["age_group"].eq(group)]
        range_labels[group] = (
            f"{group}\n{frame['age_numeric'].min():g}-{frame['age_numeric'].max():g} y"
            f"\nn={len(frame)}"
        )
    return donor_meta, donor_state, states, range_labels


def make_group_summary(donor_state: pd.DataFrame) -> pd.DataFrame:
    summary = (
        donor_state.groupby(["LEC_state", "age_group"], observed=False)
        .agg(
            n_donors=("donor_key", "size"),
            n_donors_with_state=("state_present", "sum"),
            total_state_cells=("n_state_cells", "sum"),
            mean_percentage=("cell_state_percentage", "mean"),
            sd_percentage=("cell_state_percentage", "std"),
            median_percentage=("cell_state_percentage", "median"),
            q25_percentage=("cell_state_percentage", lambda x: x.quantile(0.25)),
            q75_percentage=("cell_state_percentage", lambda x: x.quantile(0.75)),
            min_percentage=("cell_state_percentage", "min"),
            max_percentage=("cell_state_percentage", "max"),
        )
        .reset_index()
    )
    summary["sem_percentage"] = summary["sd_percentage"] / np.sqrt(
        summary["n_donors"]
    )
    summary["percent_donors_with_state"] = (
        100.0 * summary["n_donors_with_state"] / summary["n_donors"]
    )
    return summary


def kruskal_result(
    frame: pd.DataFrame,
    value_column: str = "cell_state_percentage",
) -> dict[str, object]:
    arrays = [
        frame.loc[frame["age_group"].eq(group), value_column].to_numpy(float)
        for group in GROUPS
    ]
    sizes = [len(values) for values in arrays]
    if min(sizes, default=0) < 2:
        return {
            "kruskal_H": np.nan,
            "kruskal_df": len(GROUPS) - 1,
            "kruskal_p": np.nan,
            "epsilon_squared": np.nan,
            "test_status": "not_testable_group_n_below_2",
        }
    all_values = np.concatenate(arrays)
    if np.unique(all_values).size < 2:
        return {
            "kruskal_H": 0.0,
            "kruskal_df": len(GROUPS) - 1,
            "kruskal_p": 1.0,
            "epsilon_squared": 0.0,
            "test_status": "all_donor_percentages_identical",
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test = stats.kruskal(*arrays, nan_policy="omit")
    n_total = len(all_values)
    k = len(arrays)
    epsilon = max(0.0, (float(test.statistic) - k + 1) / (n_total - k))
    return {
        "kruskal_H": float(test.statistic),
        "kruskal_df": k - 1,
        "kruskal_p": float(test.pvalue),
        "epsilon_squared": float(epsilon),
        "test_status": "ok",
    }


def welch_result(frame: pd.DataFrame) -> dict[str, object]:
    if frame["cell_state_percentage"].nunique() < 2:
        return {
            "welch_F": 0.0,
            "welch_df_num": len(GROUPS) - 1,
            "welch_df_denom": np.nan,
            "welch_p": 1.0,
            "welch_status": "all_donor_percentages_identical",
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = anova_oneway(
                frame["cell_state_percentage"].to_numpy(float),
                groups=frame["age_group"].astype(str).to_numpy(),
                use_var="unequal",
                welch_correction=True,
            )
        if not np.isfinite(float(result.pvalue)):
            raise ValueError("non-finite Welch result")
        return {
            "welch_F": float(result.statistic),
            "welch_df_num": float(result.df_num),
            "welch_df_denom": float(result.df_denom),
            "welch_p": float(result.pvalue),
            "welch_status": "ok",
        }
    except Exception as exc:  # sensitivity output must not stop the primary test
        return {
            "welch_F": np.nan,
            "welch_df_num": len(GROUPS) - 1,
            "welch_df_denom": np.nan,
            "welch_p": np.nan,
            "welch_status": f"not_testable: {exc}",
        }


def make_omnibus_table(
    donor_state: pd.DataFrame,
    states: list[str],
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state in states:
        frame = donor_state.loc[donor_state["LEC_state"].eq(state)].copy()
        top = frame.sort_values(
            ["n_state_cells", "donor_key"], ascending=[False, True]
        ).iloc[0]
        total_state_cells = int(frame["n_state_cells"].sum())
        row: dict[str, object] = {
            "LEC_state": state,
            "n_donors_total": int(frame["donor_key"].nunique()),
            "n_donors_with_state": int(frame["state_present"].sum()),
            "n_state_cells": total_state_cells,
            "largest_state_donor": str(top["donor_key"]),
            "largest_state_donor_n_cells": int(top["n_state_cells"]),
            "largest_state_donor_percent_of_state_cells": (
                100.0 * int(top["n_state_cells"]) / total_state_cells
                if total_state_cells > 0
                else np.nan
            ),
        }
        for group in GROUPS:
            group_frame = frame.loc[frame["age_group"].eq(group)]
            row[f"n_donors_{group}"] = int(len(group_frame))
            row[f"mean_percentage_{group}"] = float(
                group_frame["cell_state_percentage"].mean()
            )
        row.update(kruskal_result(frame))
        row.update(welch_result(frame))
        rows.append(row)
    table = pd.DataFrame(rows)
    table = add_fdr(table, "kruskal_p", "kruskal_fdr_bh", alpha)
    table = add_fdr(table, "welch_p", "welch_fdr_bh", alpha)
    return table


def dunn_posthoc(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    work = frame[["age_group", "cell_state_percentage"]].copy()
    work["age_group"] = work["age_group"].astype(str)
    work["rank"] = stats.rankdata(work["cell_state_percentage"].to_numpy(float))
    n_total = len(work)
    tie_counts = work["cell_state_percentage"].value_counts().to_numpy(float)
    tie_sum = np.sum(tie_counts**3 - tie_counts)
    rank_variance = (
        n_total * (n_total + 1) / 12.0
        - tie_sum / (12.0 * (n_total - 1))
    )
    rows: list[dict[str, object]] = []
    for group1, group2 in itertools.combinations(GROUPS, 2):
        values1 = work.loc[work["age_group"].eq(group1), "cell_state_percentage"]
        values2 = work.loc[work["age_group"].eq(group2), "cell_state_percentage"]
        ranks1 = work.loc[work["age_group"].eq(group1), "rank"]
        ranks2 = work.loc[work["age_group"].eq(group2), "rank"]
        standard_error = math.sqrt(
            rank_variance * (1.0 / len(values1) + 1.0 / len(values2))
        )
        if standard_error > 0:
            z_value = float((ranks1.mean() - ranks2.mean()) / standard_error)
            p_value = float(2.0 * stats.norm.sf(abs(z_value)))
        else:
            z_value = 0.0
            p_value = 1.0
        rows.append(
            {
                "group1": group1,
                "group2": group2,
                "n_donors_group1": int(len(values1)),
                "n_donors_group2": int(len(values2)),
                "mean_percentage_group1": float(values1.mean()),
                "mean_percentage_group2": float(values2.mean()),
                "mean_percentage_difference_group1_minus_group2": float(
                    values1.mean() - values2.mean()
                ),
                "mean_rank_group1": float(ranks1.mean()),
                "mean_rank_group2": float(ranks2.mean()),
                "dunn_z": z_value,
                "dunn_p": p_value,
            }
        )
    result = pd.DataFrame(rows)
    reject, adjusted, _, _ = multipletests(
        result["dunn_p"], alpha=alpha, method="holm"
    )
    result["dunn_holm_p"] = adjusted
    result["dunn_holm_significant"] = reject
    return result


def make_posthoc_table(
    donor_state: pd.DataFrame,
    omnibus: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    columns = [
        "LEC_state",
        "omnibus_kruskal_fdr_bh",
        "group1",
        "group2",
        "n_donors_group1",
        "n_donors_group2",
        "mean_percentage_group1",
        "mean_percentage_group2",
        "mean_percentage_difference_group1_minus_group2",
        "mean_rank_group1",
        "mean_rank_group2",
        "dunn_z",
        "dunn_p",
        "dunn_holm_p",
        "dunn_holm_significant",
    ]
    rows: list[pd.DataFrame] = []
    significant_states = omnibus.loc[
        omnibus["kruskal_fdr_bh_significant"], "LEC_state"
    ].astype(str)
    for state in significant_states:
        frame = donor_state.loc[donor_state["LEC_state"].eq(state)].copy()
        state_result = dunn_posthoc(frame, alpha)
        state_result.insert(0, "LEC_state", state)
        state_result.insert(
            1,
            "omnibus_kruskal_fdr_bh",
            float(
                omnibus.loc[
                    omnibus["LEC_state"].eq(state), "kruskal_fdr_bh"
                ].iloc[0]
            ),
        )
        rows.append(state_result)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)[columns]


def make_continuous_age_table(
    donor_state: pd.DataFrame,
    states: list[str],
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state in states:
        frame = donor_state.loc[donor_state["LEC_state"].eq(state)]
        if frame["cell_state_percentage"].nunique() < 2:
            rho, p_value, status = 0.0, 1.0, "all_percentages_identical"
        else:
            test = stats.spearmanr(
                frame["age_numeric"], frame["cell_state_percentage"],
                nan_policy="omit",
            )
            rho, p_value, status = float(test.statistic), float(test.pvalue), "ok"
        rows.append(
            {
                "LEC_state": state,
                "n_donors": int(len(frame)),
                "spearman_rho_age_vs_state_percentage": rho,
                "spearman_p": p_value,
                "test_status": status,
            }
        )
    return add_fdr(pd.DataFrame(rows), "spearman_p", "spearman_fdr_bh", alpha)


def make_leave_top_donor_out_table(
    donor_state: pd.DataFrame,
    omnibus: pd.DataFrame,
    states: list[str],
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state in states:
        frame = donor_state.loc[donor_state["LEC_state"].eq(state)].copy()
        top = frame.sort_values(
            ["n_state_cells", "donor_key"], ascending=[False, True]
        ).iloc[0]
        reduced = frame.loc[frame["donor_key"].ne(top["donor_key"])].copy()
        test = kruskal_result(reduced)
        original = omnibus.loc[omnibus["LEC_state"].eq(state)].iloc[0]
        row: dict[str, object] = {
            "LEC_state": state,
            "excluded_top_donor": str(top["donor_key"]),
            "excluded_donor_state_cells": int(top["n_state_cells"]),
            "excluded_donor_total_LECs": int(top["n_total_LEC_cells"]),
            "excluded_donor_state_percentage": float(top["cell_state_percentage"]),
            "original_kruskal_p": float(original["kruskal_p"]),
            "original_kruskal_fdr_bh": float(original["kruskal_fdr_bh"]),
        }
        for group in GROUPS:
            row[f"n_donors_after_exclusion_{group}"] = int(
                reduced["age_group"].eq(group).sum()
            )
        row.update(
            {
                "leaveout_kruskal_H": test["kruskal_H"],
                "leaveout_kruskal_p": test["kruskal_p"],
                "leaveout_epsilon_squared": test["epsilon_squared"],
                "leaveout_test_status": test["test_status"],
            }
        )
        rows.append(row)
    return add_fdr(
        pd.DataFrame(rows),
        "leaveout_kruskal_p",
        "leaveout_kruskal_fdr_bh",
        alpha,
    )


def make_minimum_cell_sensitivity(
    donor_state: pd.DataFrame,
    states: list[str],
    minimum_cells: int,
    alpha: float,
) -> pd.DataFrame:
    eligible = donor_state.loc[
        donor_state["n_total_LEC_cells"].ge(minimum_cells)
    ].copy()
    rows: list[dict[str, object]] = []
    for state in states:
        frame = eligible.loc[eligible["LEC_state"].eq(state)].copy()
        test = kruskal_result(frame)
        row: dict[str, object] = {
            "LEC_state": state,
            "minimum_donor_LEC_cells": minimum_cells,
            "n_donors_retained": int(frame["donor_key"].nunique()),
        }
        for group in GROUPS:
            row[f"n_donors_{group}"] = int(frame["age_group"].eq(group).sum())
        row.update(
            {
                "kruskal_H": test["kruskal_H"],
                "kruskal_p": test["kruskal_p"],
                "epsilon_squared": test["epsilon_squared"],
                "test_status": test["test_status"],
            }
        )
        rows.append(row)
    return add_fdr(pd.DataFrame(rows), "kruskal_p", "kruskal_fdr_bh", alpha)


def make_adjusted_sensitivity(
    donor_state: pd.DataFrame,
    states: list[str],
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    formula = "cell_state_percentage ~ C(age_group) + C(Project) + C(Sex)"
    for state in states:
        frame = donor_state.loc[donor_state["LEC_state"].eq(state)].copy()
        frame["age_group"] = frame["age_group"].astype(str)
        if frame["cell_state_percentage"].nunique() < 2:
            rows.append(
                {
                    "LEC_state": state,
                    "model": formula,
                    "age_group_F_HC3": 0.0,
                    "age_group_df": len(GROUPS) - 1,
                    "age_group_p_HC3": 1.0,
                    "model_r_squared": 0.0,
                    "model_status": "all_percentages_identical",
                }
            )
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ols(formula, data=frame).fit()
                table = anova_lm(model, typ=2, robust="hc3")
            age_row = table.loc["C(age_group)"]
            rows.append(
                {
                    "LEC_state": state,
                    "model": formula,
                    "age_group_F_HC3": float(age_row["F"]),
                    "age_group_df": float(age_row["df"]),
                    "age_group_p_HC3": float(age_row["PR(>F)"]),
                    "model_r_squared": float(model.rsquared),
                    "model_status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "LEC_state": state,
                    "model": formula,
                    "age_group_F_HC3": np.nan,
                    "age_group_df": len(GROUPS) - 1,
                    "age_group_p_HC3": np.nan,
                    "model_r_squared": np.nan,
                    "model_status": f"not_testable: {exc}",
                }
            )
    return add_fdr(
        pd.DataFrame(rows),
        "age_group_p_HC3",
        "age_group_fdr_bh_HC3",
        alpha,
    )


def state_color_map(adata: ad.AnnData, state_key: str, states: list[str]) -> dict[str, str]:
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


def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def draw_state_panel(
    ax,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    omnibus_row: pd.Series,
    range_labels: dict[str, str],
    posthoc: pd.DataFrame | None,
    show_ylabel: bool = True,
    compact: bool = False,
) -> None:
    for position, group in enumerate(GROUPS):
        points = frame.loc[frame["age_group"].eq(group)].sort_values("donor_key")
        jitter = np.linspace(-0.17, 0.17, len(points))
        ax.scatter(
            np.full(len(points), position) + jitter,
            points["cell_state_percentage"],
            s=28 if compact else 46,
            color=GROUP_COLORS[group],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        group_summary = summary.loc[summary["age_group"].eq(group)].iloc[0]
        ax.errorbar(
            position,
            group_summary["mean_percentage"],
            yerr=group_summary["sem_percentage"],
            fmt="D",
            color="black",
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5 if compact else 7,
            linewidth=1.5,
            capsize=4,
            zorder=5,
        )

    labels = [range_labels[group] if not compact else group for group in GROUPS]
    ax.set_xticks(range(len(GROUPS)), labels)
    ax.set_xlim(-0.45, len(GROUPS) - 0.55)
    ax.set_ylabel("Donor LECs in state (%)" if show_ylabel else "")
    ax.grid(axis="y", alpha=0.2)
    state = str(omnibus_row["LEC_state"])
    ax.set_title(
        f"{state}: KW p={format_p(omnibus_row['kruskal_p'])}, "
        f"FDR={format_p(omnibus_row['kruskal_fdr_bh'])}",
        fontsize=10 if compact else 12,
    )

    observed_max = float(frame["cell_state_percentage"].max())
    base_top = max(5.0, observed_max * 1.12 + 1.0)
    significant_pairs = pd.DataFrame()
    if posthoc is not None and not posthoc.empty:
        significant_pairs = posthoc.loc[posthoc["dunn_holm_significant"]].copy()
    if significant_pairs.empty or compact:
        ax.set_ylim(0, base_top)
        return

    step = max(2.0, base_top * 0.09)
    group_position = {group: index for index, group in enumerate(GROUPS)}
    for index, (_, comparison) in enumerate(significant_pairs.iterrows()):
        x1 = group_position[str(comparison["group1"])]
        x2 = group_position[str(comparison["group2"])]
        y = base_top + index * step
        ax.plot([x1, x1, x2, x2], [y, y + 0.25 * step, y + 0.25 * step, y],
                color="black", linewidth=1)
        ax.text(
            (x1 + x2) / 2,
            y + 0.3 * step,
            significance_label(float(comparison["dunn_holm_p"])),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, base_top + (len(significant_pairs) + 0.7) * step)


def save_per_state_figures(
    donor_state: pd.DataFrame,
    summary: pd.DataFrame,
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
        state_summary = summary.loc[summary["LEC_state"].eq(state)].copy()
        omnibus_row = omnibus.loc[omnibus["LEC_state"].eq(state)].iloc[0]
        state_posthoc = posthoc.loc[posthoc["LEC_state"].eq(state)].copy()
        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        draw_state_panel(
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
        fig.suptitle("Donor-level LEC-state percentage across age groups", fontsize=13)
        fig.tight_layout()
        path = output_dir / f"08_4_{state}_donor_percentage_by_age.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append(
            {
                "figure": str(path),
                "description": (
                    f"{state}: individual donor state percentages, mean, SEM, "
                    "and significant Dunn-Holm comparisons"
                ),
                "status": "ok",
            }
        )
    return manifest


def save_all_states_figure(
    donor_state: pd.DataFrame,
    summary: pd.DataFrame,
    omnibus: pd.DataFrame,
    states: list[str],
    range_labels: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    n_columns = 3
    n_rows = math.ceil(len(states) / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 4.1 * n_rows))
    flat_axes = np.atleast_1d(axes).ravel()
    for index, state in enumerate(states):
        frame = donor_state.loc[donor_state["LEC_state"].eq(state)].copy()
        state_summary = summary.loc[summary["LEC_state"].eq(state)].copy()
        omnibus_row = omnibus.loc[omnibus["LEC_state"].eq(state)].iloc[0]
        draw_state_panel(
            flat_axes[index],
            frame,
            state_summary,
            omnibus_row,
            range_labels,
            posthoc=None,
            show_ylabel=index % n_columns == 0,
            compact=True,
        )
    for ax in flat_axes[len(states):]:
        ax.set_visible(False)
    fig.suptitle(
        "Donor-level LEC-state percentages: points are donors; diamonds are mean +/- SEM",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_statistical_summary_figure(
    omnibus: pd.DataFrame,
    state_colors: dict[str, str],
    path: Path,
    dpi: int,
    alpha: float,
) -> None:
    import matplotlib.pyplot as plt

    frame = omnibus.copy().sort_values("epsilon_squared", ascending=True)
    states = frame["LEC_state"].astype(str).tolist()
    colors = [state_colors[state] for state in states]
    q_values = frame["kruskal_fdr_bh"].astype(float).clip(lower=np.finfo(float).tiny)
    score = -np.log10(q_values)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5.5, 0.42 * len(states))))
    axes[0].barh(states, frame["epsilon_squared"], color=colors)
    axes[0].set_xlabel("Kruskal-Wallis epsilon-squared")
    axes[0].set_title("Age-group effect size")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(states, score, color=colors)
    axes[1].axvline(-math.log10(alpha), color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("-log10(BH FDR)")
    axes[1].set_title("Primary omnibus evidence")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("Donor-level LEC-state association with the four age groups", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    path: Path,
    input_h5ad: Path,
    donor_meta: pd.DataFrame,
    states: list[str],
    omnibus: pd.DataFrame,
    posthoc: pd.DataFrame,
    continuous: pd.DataFrame,
    adjusted: pd.DataFrame,
    minimum_cells: int,
    alpha: float,
) -> None:
    group_lines = []
    for group in GROUPS:
        frame = donor_meta.loc[donor_meta["age_group"].eq(group)]
        group_lines.append(
            f"{group}: {len(frame)} donors, ages {frame['age_numeric'].min():g}-"
            f"{frame['age_numeric'].max():g}, {frame['n_total_LEC_cells'].sum()} LEC cells"
        )
    significant_primary = omnibus.loc[
        omnibus["kruskal_fdr_bh_significant"],
        ["LEC_state", "kruskal_p", "kruskal_fdr_bh", "epsilon_squared"],
    ]
    significant_continuous = continuous.loc[
        continuous["spearman_fdr_bh_significant"],
        [
            "LEC_state",
            "spearman_rho_age_vs_state_percentage",
            "spearman_p",
            "spearman_fdr_bh",
        ],
    ]
    significant_adjusted = adjusted.loc[
        adjusted["age_group_fdr_bh_HC3_significant"],
        ["LEC_state", "age_group_p_HC3", "age_group_fdr_bh_HC3"],
    ]
    primary_text = (
        significant_primary.to_csv(index=False).strip()
        if not significant_primary.empty
        else "No LEC state passed the primary BH-FDR threshold."
    )
    continuous_text = (
        significant_continuous.to_csv(index=False).strip()
        if not significant_continuous.empty
        else "No LEC state passed BH FDR for continuous-age Spearman association."
    )
    adjusted_text = (
        significant_adjusted.to_csv(index=False).strip()
        if not significant_adjusted.empty
        else "No LEC state passed BH FDR in the Project/Sex-adjusted sensitivity model."
    )
    posthoc_text = (
        f"{len(posthoc)} pairwise rows were generated for states passing omnibus FDR."
        if not posthoc.empty
        else "No post hoc tests were generated because no state passed omnibus FDR."
    )
    text = f"""Step 08.4 donor-level LEC-state age analysis
================================================

Run identity
------------
Script version: {SCRIPT_VERSION}
Input h5ad: {input_h5ad}
Cells, expression, embeddings, clusters, and states changed: no

Analysis unit and calculation
-----------------------------
The donor is the biological replicate. Every donor x state combination is
retained, including zero cells. For donor d and state s:

    percentage(d,s) = 100 * n_cells(d,s) / total_LEC_cells(d)

Group mean and SEM are calculated across donors, not across cells. SEM equals
the donor-level standard deviation divided by the square root of donor n.

Age groups
----------
{chr(10).join(group_lines)}

Primary inference
-----------------
For each of {len(states)} states, a Kruskal-Wallis test compares donor-level
percentages across Q1-Q4. Benjamini-Hochberg correction is applied across the
{len(states)} omnibus p-values at alpha={alpha:g}. Epsilon-squared is reported
as an omnibus effect size. Dunn pairwise tests with Holm correction are run only
for states passing the primary omnibus FDR threshold.

Primary significant states:
{primary_text}

Post hoc status:
{posthoc_text}

Sensitivity analyses
--------------------
1. Welch one-way ANOVA is reported as a parametric sensitivity analysis.
2. Spearman correlation tests donor-level state percentage against continuous age.
3. The largest cell-contributing donor for each state is removed and the omnibus
   test repeated.
4. The omnibus test is repeated using donors with at least {minimum_cells} LEC cells.
5. Robust-HC3 OLS tests the age-group term after adjustment for Project and Sex.
   This adjusted percentage model is a sensitivity analysis, not the primary test.

Continuous-age significant states:
{continuous_text}

Project/Sex-adjusted significant states:
{adjusted_text}

Interpretation safeguards
-------------------------
Percentages are bounded, zero-inflated, and compositional: all state percentages
sum to 100% within each donor. An increase in one state necessarily changes the
relative percentages of other states. Donors with very few LEC cells have
imprecise percentages but remain in the primary analysis as requested. The
minimum-cell analysis is explicitly secondary.

The unadjusted four-group test demonstrates age-group association, not a causal
age effect. Project is imbalanced across age groups, and Harmony does not equalize
cell counts or donor composition. Interpret a state as robust only when its donor
points, effect size, dominant-donor sensitivity, continuous-age result, and
Project/Sex-adjusted result tell a consistent story.
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
    sentinel = output_root / "08_4_run_summary.csv"
    if not input_h5ad.exists():
        raise FileNotFoundError(f"Step 08.3 input not found: {input_h5ad}")
    if sentinel.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(
            f"Step 08.4 output already exists: {sentinel}. Use --overwrite."
        )

    print(f"Reading Step 08.3 h5ad: {input_h5ad}")
    adata = ad.read_h5ad(input_h5ad)
    donor_meta, donor_state, states, range_labels = prepare_donor_state_table(
        adata.obs,
        args.donor_key,
        args.state_key,
        args.group_key,
        args.age_key,
        args.project_key,
        args.sex_key,
        args.original_group_key,
        args.expected_donors_per_group,
    )
    group_summary = make_group_summary(donor_state)
    omnibus = make_omnibus_table(donor_state, states, args.alpha)
    posthoc = make_posthoc_table(donor_state, omnibus, args.alpha)
    continuous = make_continuous_age_table(donor_state, states, args.alpha)
    leaveout = make_leave_top_donor_out_table(
        donor_state, omnibus, states, args.alpha
    )
    minimum_cell = make_minimum_cell_sensitivity(
        donor_state,
        states,
        args.min_donor_LECs_sensitivity,
        args.alpha,
    )
    adjusted = make_adjusted_sensitivity(donor_state, states, args.alpha)

    print("\nDonor counts by new age group:")
    print(
        donor_meta.groupby("age_group", observed=False)
        .agg(n_donors=("donor_key", "size"), n_LECs=("n_total_LEC_cells", "sum"))
        .to_string()
    )
    print("\nPrimary state-level tests:")
    print(
        omnibus[
            [
                "LEC_state",
                "kruskal_H",
                "kruskal_p",
                "kruskal_fdr_bh",
                "epsilon_squared",
                "kruskal_fdr_bh_significant",
            ]
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

    donor_meta.to_csv(output_root / "08_4_donor_metadata.csv", index=False)
    donor_state.to_csv(
        output_root / "08_4_donor_state_counts_percentages.csv.gz",
        index=False,
        compression="gzip",
    )
    group_summary.to_csv(
        output_root / "08_4_state_age_group_summary_mean_sem.csv", index=False
    )
    omnibus.to_csv(output_root / "08_4_state_omnibus_tests.csv", index=False)
    posthoc.to_csv(
        output_root / "08_4_state_posthoc_Dunn_Holm.csv", index=False
    )
    continuous.to_csv(
        output_root / "08_4_continuous_age_spearman.csv", index=False
    )
    leaveout.to_csv(
        output_root / "08_4_leave_top_donor_out_sensitivity.csv", index=False
    )
    minimum_cell.to_csv(
        output_root
        / f"08_4_min{args.min_donor_LECs_sensitivity}_LEC_donor_sensitivity.csv",
        index=False,
    )
    adjusted.to_csv(
        output_root / "08_4_project_sex_adjusted_sensitivity.csv", index=False
    )

    palette = state_color_map(adata, args.state_key, states)
    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        figure_manifest.extend(
            save_per_state_figures(
                donor_state,
                group_summary,
                omnibus,
                posthoc,
                states,
                range_labels,
                per_state_dir,
                args.dpi,
            )
        )
        combined_path = figures_dir / "08_4_all_states_donor_percentage_by_age.png"
        save_all_states_figure(
            donor_state,
            group_summary,
            omnibus,
            states,
            range_labels,
            combined_path,
            args.dpi,
        )
        figure_manifest.append(
            {
                "figure": str(combined_path),
                "description": "All states: individual donors and mean +/- SEM by age group",
                "status": "ok",
            }
        )
        statistics_path = figures_dir / "08_4_state_age_association_summary.png"
        save_statistical_summary_figure(
            omnibus, palette, statistics_path, args.dpi, args.alpha
        )
        figure_manifest.append(
            {
                "figure": str(statistics_path),
                "description": "Kruskal-Wallis effect sizes and BH-FDR evidence by state",
                "status": "ok",
            }
        )
    pd.DataFrame(
        figure_manifest, columns=["figure", "description", "status"]
    ).to_csv(output_root / "08_4_figure_manifest.csv", index=False)

    run_summary = {
        "script_version": SCRIPT_VERSION,
        "status": "completed",
        "input_h5ad": str(input_h5ad),
        "output_root": str(output_root),
        "n_LEC_cells": int(adata.n_obs),
        "n_donors": int(donor_meta["donor_key"].nunique()),
        "n_age_groups": len(GROUPS),
        "donors_per_group": args.expected_donors_per_group,
        "n_states": len(states),
        "state_labels": ";".join(states),
        "primary_test": "Kruskal-Wallis on donor-level percentages",
        "multiple_testing": "Benjamini-Hochberg across states",
        "alpha": args.alpha,
        "n_primary_FDR_significant_states": int(
            omnibus["kruskal_fdr_bh_significant"].sum()
        ),
        "primary_FDR_significant_states": ";".join(
            omnibus.loc[
                omnibus["kruskal_fdr_bh_significant"], "LEC_state"
            ].astype(str)
        ),
        "n_posthoc_rows": len(posthoc),
        "min_donor_LECs_sensitivity": args.min_donor_LECs_sensitivity,
        "h5ad_modified": False,
    }
    pd.DataFrame([run_summary]).to_csv(sentinel, index=False)
    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "input_h5ad": str(input_h5ad),
        "output_root": str(output_root),
        "group_key": args.group_key,
        "state_key": args.state_key,
        "donor_key": args.donor_key,
        "age_key": args.age_key,
        "project_key": args.project_key,
        "sex_key": args.sex_key,
        "original_group_key": args.original_group_key,
        "group_order": GROUPS,
        "group_range_labels": range_labels,
        "alpha": args.alpha,
        "primary_test": "Kruskal-Wallis",
        "omnibus_adjustment": "Benjamini-Hochberg FDR across LEC states",
        "posthoc": "Dunn pairwise with Holm correction; omnibus-FDR gated",
        "min_donor_LECs_sensitivity": args.min_donor_LECs_sensitivity,
        "figures_enabled": not args.skip_figures,
        "dpi": args.dpi,
    }
    (output_root / "08_4_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    write_notes(
        output_root / "08_4_method_and_interpretation_notes.txt",
        input_h5ad,
        donor_meta,
        states,
        omnibus,
        posthoc,
        continuous,
        adjusted,
        args.min_donor_LECs_sensitivity,
        args.alpha,
    )
    print(f"\nStep 08.4 outputs written to: {output_root}")
    print("Input h5ad was read only; clustering and metadata were not changed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
