#!/usr/bin/env python
r"""
Step 07: describe final LEC abundance in the selected disease cohorts.

This standalone descriptive step uses:

* Step 03 QC-passed cells as the all-cell denominator;
* Step 06 independently reclustered EC candidates as the EC denominator;
* Step 06 ``selected_as_LEC`` cells as the LEC numerator.

Every retained donor is kept, including donors with zero EC or zero LEC cells.
The script reports pooled cell percentages and donor-level distributions. These
are descriptive abundance summaries; they are not cross-dataset inferential
tests because capture depth and dataset effects differ.

Color rules
-----------
The same semantic name always receives the same color across Step 07 figures.
LEC remains Step 06 red (#D62728), and other_EC_candidate remains light gray.
Conditions use a fixed colorblind-friendly palette, black outlines, and fixed
hatches so multi-condition figures remain clear without relying only on color.
The full mapping is saved to ``07_color_key.csv``.

Default inputs
--------------
new_result2_disease/03/
new_result2_disease/06_EC_recluster/

Default outputs
---------------
new_result2_disease/07_LEC_description/

PowerShell
----------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\07_describe_LEC_selected_disease.py

Read-only validation
--------------------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\07_describe_LEC_selected_disease.py `
      --validate_only
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_VERSION = "07_selected_disease_exclude_dataset02_dataset06_dataset12_fixed_colors_2026_08_11"
DEFAULT_STEP03_ROOT = "new_result2_disease/03"
DEFAULT_STEP06_ROOT = "new_result2_disease/06_EC_recluster"
DEFAULT_OUTPUT_ROOT = "new_result2_disease/07_LEC_description"

EXPECTED_COHORTS = (
    ("dataset01", "DCM"),
    ("dataset01", "HCM"),
    ("dataset01", "NF"),
    ("dataset08", "DoxCM"),
    ("dataset08", "NF"),
    ("dataset11", "HFpEF"),
    ("dataset11", "NF"),
)
FORBIDDEN_DATASETS = {"dataset02", "dataset03", "dataset06", "dataset10", "dataset12"}
CONDITION_ORDER = ["NF", "DCM", "HCM", "DoxCM", "HFpEF"]

# Okabe-Ito colors, chosen to remain distinct under common color-vision changes.
CONDITION_COLORS = {
    "NF": "#0072B2",
    "DCM": "#D55E00",
    "HCM": "#CC79A7",
    "DoxCM": "#56B4E9",
    "HFpEF": "#E69F00",
}
CONDITION_HATCHES = {
    "NF": "",
    "DCM": "//",
    "HCM": "\\\\",
    "DoxCM": "++",
    "HFpEF": "..",
}
DATASET_COLORS = {
    "dataset01": "#332288",
    "dataset08": "#117733",
    "dataset11": "#DDCC77",
}
DATASET_MARKERS = {
    "dataset01": "o",
    "dataset08": "v",
    "dataset11": "D",
}
STAGE_COLORS = {
    "All QC-passed cells": "#6A3D9A",
    "EC candidates": "#4C78A8",
    "LEC": "#D62728",
    "other_EC_candidate": "#BDBDBD",
}
DONOR_STATUS_COLORS = {
    "At least one LEC": "#D62728",
    "No LEC detected": "#D9D9D9",
}
SEX_COLORS = {
    "Female": "#CC79A7",
    "Male": "#0072B2",
    "Unknown": "#7F7F7F",
}


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def true_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return normalized_text(values).str.casefold().isin(
        {"true", "1", "yes", "y", "lec"}
    )


def safe_divide(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0 else np.nan


def safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value).strip())
    return re.sub(r"_+", "_", text).strip("_") or "NA"


def cohort_name(dataset: str, condition: str) -> str:
    return f"{dataset} | {condition}"


def cohort_prefix(dataset: str, condition: str) -> str:
    return f"{dataset}_{safe_name(condition)}"


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in {source}: {missing}")


def parse_cohort_subset(value: str) -> set[tuple[str, str]] | None:
    if str(value).strip().casefold() in {"", "all"}:
        return None
    selected: set[tuple[str, str]] = set()
    for token in re.split(r"[,;]", str(value)):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid cohort {token!r}; use dataset:condition, e.g. dataset08:NF."
            )
        dataset, condition = [part.strip() for part in token.split(":", 1)]
        selected.add((dataset, condition))
    invalid = selected - set(EXPECTED_COHORTS)
    if invalid:
        raise ValueError(f"Unapproved cohort subset: {sorted(invalid)}")
    return selected


def first_nonempty(values: pd.Series) -> str:
    text = normalized_text(values)
    text = text.loc[text.ne("")]
    return str(text.iloc[0]) if len(text) else ""


def first_numeric(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.iloc[0]) if len(numeric) else np.nan


def read_step03_master(step03_root: Path) -> pd.DataFrame:
    path = step03_root / "03_master_run_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Step 03 master summary not found: {path}")
    table = pd.read_csv(path)
    require_columns(
        table,
        [
            "dataset_folder", "condition", "status", "n_cells_after_all_QC",
            "n_donors_after_QC", "output_h5ad",
        ],
        path,
    )
    table["dataset_folder"] = normalized_text(table["dataset_folder"])
    table["condition"] = normalized_text(table["condition"])
    if table["dataset_folder"].isin(FORBIDDEN_DATASETS).any():
        raise ValueError(
            "An excluded dataset (dataset02, dataset03, dataset06, dataset10, "
            "or dataset12) "
            "is present in Step 03."
        )
    observed = set(zip(table["dataset_folder"], table["condition"]))
    expected = set(EXPECTED_COHORTS)
    if observed != expected:
        raise ValueError(
            f"Step 03 cohort mismatch. Missing={sorted(expected-observed)}; "
            f"unexpected={sorted(observed-expected)}"
        )
    if table.duplicated(["dataset_folder", "condition"]).any():
        raise ValueError("Step 03 master summary contains duplicate cohort rows.")
    statuses = normalized_text(table["status"]).str.casefold()
    valid = (
        statuses.str.startswith("completed")
        | statuses.str.startswith("reused")
        | statuses.str.startswith("existing")
    )
    if not valid.all():
        raise ValueError(
            "Incomplete Step 03 cohorts:\n"
            + table.loc[~valid, ["dataset_folder", "condition", "status"]].to_string(index=False)
        )
    order = {pair: index for index, pair in enumerate(EXPECTED_COHORTS)}
    table["cohort_order"] = [
        order[(dataset, condition)]
        for dataset, condition in zip(table["dataset_folder"], table["condition"])
    ]
    table["cohort"] = [
        cohort_name(dataset, condition)
        for dataset, condition in zip(table["dataset_folder"], table["condition"])
    ]
    return table.sort_values("cohort_order").reset_index(drop=True)


def read_step06_summary(step06_root: Path) -> pd.DataFrame:
    path = step06_root / "06_EC_recluster_run_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Step 06 run summary not found: {path}")
    table = pd.read_csv(path)
    require_columns(
        table,
        [
            "dataset_folder", "condition", "status", "n_EC_cells",
            "n_LEC_cells", "output_h5ad",
        ],
        path,
    )
    table["dataset_folder"] = normalized_text(table["dataset_folder"])
    table["condition"] = normalized_text(table["condition"])
    if table["dataset_folder"].isin(FORBIDDEN_DATASETS).any():
        raise ValueError(
            "An excluded dataset (dataset02, dataset03, dataset06, dataset10, "
            "or dataset12) "
            "is present in Step 06."
        )
    observed = set(zip(table["dataset_folder"], table["condition"]))
    expected = set(EXPECTED_COHORTS)
    if observed != expected:
        raise ValueError(
            f"Step 06 cohort mismatch. Missing={sorted(expected-observed)}; "
            f"unexpected={sorted(observed-expected)}"
        )
    if table.duplicated(["dataset_folder", "condition"]).any():
        raise ValueError("Step 06 run summary contains duplicate cohort rows.")
    statuses = normalized_text(table["status"]).str.casefold()
    if not statuses.str.startswith(("completed", "reused")).all():
        raise ValueError(
            "Step 06 is incomplete:\n"
            + table.loc[
                ~statuses.str.startswith(("completed", "reused")),
                ["dataset_folder", "condition", "status"],
            ].to_string(index=False)
        )
    return table


def join_pipeline_summaries(step03: pd.DataFrame, step06: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_folder", "condition", "status", "n_EC_cells", "n_LEC_cells",
        "n_EC_only_leiden_clusters", "n_LEC_clusters", "output_h5ad",
    ]
    columns = [column for column in columns if column in step06.columns]
    joined = step03.merge(
        step06[columns],
        on=["dataset_folder", "condition"],
        how="left",
        validate="one_to_one",
        suffixes=("_step03", "_step06"),
    )
    if joined["output_h5ad_step06"].isna().any():
        raise ValueError("Step 03 and Step 06 cohort summaries do not align.")
    return joined.sort_values("cohort_order").reset_index(drop=True)


def donor_key_from_obs(obs: pd.DataFrame) -> pd.Series:
    if "donor_key" in obs.columns:
        key = normalized_text(obs["donor_key"])
        if key.ne("").all():
            return key
    if "dataset_folder" in obs.columns and "donor_id_source" in obs.columns:
        return normalized_text(obs["dataset_folder"]) + "::" + normalized_text(obs["donor_id_source"])
    raise KeyError("A complete namespaced donor_key is required in Step 03 and Step 06.")


def normalize_sex(values: pd.Series) -> pd.Series:
    text = normalized_text(values).str.casefold()
    mapping = {
        "f": "Female", "female": "Female", "woman": "Female",
        "m": "Male", "male": "Male", "man": "Male",
    }
    return text.map(mapping).fillna("Unknown")


def read_obs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs.copy()
        obs.index = obs.index.astype(str)
        if obs.index.duplicated().any():
            raise ValueError(f"Duplicate cell IDs in {path}.")
        return obs
    finally:
        adata.file.close()


def validate_cohort(
    all_obs: pd.DataFrame,
    ec_obs: pd.DataFrame,
    dataset: str,
    condition: str,
    all_path: Path,
    ec_path: Path,
    expected_all: int,
    expected_ec: int,
    expected_lec: int,
) -> dict[str, object]:
    require_columns(
        all_obs,
        [
            "dataset_folder", "disease", "donor_key",
            "region_standardized_analysis", "step03_final_QC_pass",
        ],
        all_path,
    )
    require_columns(
        ec_obs,
        [
            "dataset_folder", "condition", "disease", "donor_key",
            "region_standardized_analysis", "selected_as_LEC",
            "celltypist_majority_vote", "leiden", "allcell_leiden",
        ],
        ec_path,
    )
    if len(all_obs) != int(expected_all) or len(ec_obs) != int(expected_ec):
        raise ValueError(
            f"{cohort_name(dataset, condition)} count mismatch: "
            f"all {len(all_obs):,}/{expected_all:,}, EC {len(ec_obs):,}/{expected_ec:,}."
        )
    for name, obs in [("Step 03", all_obs), ("Step 06", ec_obs)]:
        datasets = set(normalized_text(obs["dataset_folder"]).unique())
        diseases = set(normalized_text(obs["disease"]).unique())
        if datasets != {dataset} or diseases != {condition}:
            raise ValueError(
                f"{cohort_name(dataset, condition)} {name} identity mismatch: "
                f"datasets={sorted(datasets)}, diseases={sorted(diseases)}."
            )
    if not true_mask(all_obs["step03_final_QC_pass"]).all():
        raise ValueError(f"{cohort_name(dataset, condition)} contains non-QC Step 03 cells.")
    regions_all = set(normalized_text(all_obs["region_standardized_analysis"]).unique())
    regions_ec = set(normalized_text(ec_obs["region_standardized_analysis"]).unique())
    allowed = {"S"} if dataset == "dataset11" else {"LV", "AP"}
    if not regions_all.issubset(allowed) or not regions_ec.issubset(allowed):
        raise ValueError(
            f"{cohort_name(dataset, condition)} violates region rule: "
            f"all={sorted(regions_all)}, EC={sorted(regions_ec)}, allowed={sorted(allowed)}."
        )
    missing_ec = ec_obs.index.difference(all_obs.index)
    if len(missing_ec):
        raise ValueError(
            f"{cohort_name(dataset, condition)}: {len(missing_ec):,} EC cells are absent from Step 03."
        )
    lec_flag = true_mask(ec_obs["selected_as_LEC"])
    n_lec = int(lec_flag.sum())
    if n_lec != int(expected_lec):
        raise ValueError(
            f"{cohort_name(dataset, condition)}: LEC count {n_lec:,} differs from Step 06 {expected_lec:,}."
        )
    lec_votes = set(normalized_text(ec_obs.loc[lec_flag, "celltypist_majority_vote"]).unique())
    invalid_votes = {
        label for label in lec_votes
        if not (label.casefold() == "ec8_ln" or "lymph" in label.casefold())
    }
    if invalid_votes:
        raise ValueError(
            f"{cohort_name(dataset, condition)} LEC cells have non-lymphatic votes: {sorted(invalid_votes)}"
        )
    all_donors = donor_key_from_obs(all_obs)
    ec_donors = donor_key_from_obs(ec_obs)
    absent_donors = set(ec_donors.unique()) - set(all_donors.unique())
    if absent_donors:
        raise ValueError(
            f"{cohort_name(dataset, condition)} EC donors absent from Step 03: {sorted(absent_donors)}"
        )
    return {
        "dataset_folder": dataset,
        "condition": condition,
        "cohort": cohort_name(dataset, condition),
        "status": "validated",
        "step03_allcell_h5ad": str(all_path),
        "step06_EC_h5ad": str(ec_path),
        "regions": ";".join(sorted(regions_all)),
        "n_all_QC_cells": int(len(all_obs)),
        "n_EC_cells": int(len(ec_obs)),
        "n_LEC_cells": n_lec,
        "n_all_donors": int(all_donors.nunique()),
        "n_EC_donors": int(ec_donors.nunique()),
        "n_EC_cells_absent_from_allcell": 0,
        "LEC_vote_labels": ";".join(sorted(lec_votes)),
        "region_rule": "dataset11_septum_allowed_else_LV_AP_only",
    }


def donor_table(
    all_obs: pd.DataFrame,
    ec_obs: pd.DataFrame,
    dataset: str,
    condition: str,
) -> pd.DataFrame:
    all_meta = all_obs.copy()
    ec_meta = ec_obs.copy()
    all_meta["donor_key"] = donor_key_from_obs(all_meta)
    ec_meta["donor_key"] = donor_key_from_obs(ec_meta)
    lec_meta = ec_meta.loc[true_mask(ec_meta["selected_as_LEC"])].copy()
    all_counts = all_meta.groupby("donor_key", observed=True).size().rename("n_all_QC_cells")
    ec_counts = ec_meta.groupby("donor_key", observed=True).size().rename("n_EC_cells")
    lec_counts = lec_meta.groupby("donor_key", observed=True).size().rename("n_LEC_cells")
    counts = pd.concat([all_counts, ec_counts, lec_counts], axis=1).fillna(0)
    for column in ["n_all_QC_cells", "n_EC_cells", "n_LEC_cells"]:
        counts[column] = counts[column].astype(int)

    metadata_candidates = [
        "donor_id", "donor_id_source", "biosample_id", "source_id",
        "sex", "age", "lvef", "region_standardized_analysis",
        "comparison_limitation", "has_within_dataset_NF",
    ]
    metadata_candidates = [column for column in metadata_candidates if column in all_meta]
    metadata_rows: list[dict[str, object]] = []
    for donor, group in all_meta.groupby("donor_key", observed=True, sort=True):
        item: dict[str, object] = {"donor_key": donor}
        for column in metadata_candidates:
            if column == "lvef":
                item[column] = first_numeric(group[column])
            else:
                item[column] = first_nonempty(group[column])
        metadata_rows.append(item)
    metadata = pd.DataFrame(metadata_rows).set_index("donor_key")
    table = metadata.join(counts, how="outer").reset_index()
    table.insert(0, "cohort", cohort_name(dataset, condition))
    table.insert(0, "condition", condition)
    table.insert(0, "dataset_folder", dataset)
    if "sex" in table:
        table["sex_standardized"] = normalize_sex(table["sex"])
    else:
        table["sex_standardized"] = "Unknown"
    table["EC_fraction_among_all_QC"] = table["n_EC_cells"] / table["n_all_QC_cells"]
    table["EC_percent_among_all_QC"] = 100.0 * table["EC_fraction_among_all_QC"]
    table["LEC_fraction_among_all_QC"] = table["n_LEC_cells"] / table["n_all_QC_cells"]
    table["LEC_percent_among_all_QC"] = 100.0 * table["LEC_fraction_among_all_QC"]
    table["LEC_fraction_among_EC"] = np.where(
        table["n_EC_cells"].gt(0), table["n_LEC_cells"] / table["n_EC_cells"], np.nan
    )
    table["LEC_percent_among_EC"] = 100.0 * table["LEC_fraction_among_EC"]
    table["has_at_least_one_EC"] = table["n_EC_cells"].gt(0)
    table["has_at_least_one_LEC"] = table["n_LEC_cells"].gt(0)
    return table.sort_values("donor_key").reset_index(drop=True)


def metric_summary(values: pd.Series, prefix: str) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return {
        f"{prefix}_mean": float(numeric.mean()) if len(numeric) else np.nan,
        f"{prefix}_median": float(numeric.median()) if len(numeric) else np.nan,
        f"{prefix}_Q1": float(numeric.quantile(0.25)) if len(numeric) else np.nan,
        f"{prefix}_Q3": float(numeric.quantile(0.75)) if len(numeric) else np.nan,
        f"{prefix}_min": float(numeric.min()) if len(numeric) else np.nan,
        f"{prefix}_max": float(numeric.max()) if len(numeric) else np.nan,
    }


def summarize_cohort(
    all_obs: pd.DataFrame,
    ec_obs: pd.DataFrame,
    donors: pd.DataFrame,
    dataset: str,
    condition: str,
) -> dict[str, object]:
    n_all = int(len(all_obs))
    n_ec = int(len(ec_obs))
    n_lec = int(true_mask(ec_obs["selected_as_LEC"]).sum())
    comparison_limitation = (
        first_nonempty(all_obs["comparison_limitation"])
        if "comparison_limitation" in all_obs else ""
    )
    within_nf = (
        true_mask(all_obs["has_within_dataset_NF"]).all()
        if "has_within_dataset_NF" in all_obs else condition == "NF"
    )
    result: dict[str, object] = {
        "dataset_folder": dataset,
        "condition": condition,
        "cohort": cohort_name(dataset, condition),
        "n_donors_all": int(donors["donor_key"].nunique()),
        "n_donors_with_EC": int(donors["has_at_least_one_EC"].sum()),
        "n_donors_with_LEC": int(donors["has_at_least_one_LEC"].sum()),
        "donor_coverage_with_EC_percent": 100.0 * float(donors["has_at_least_one_EC"].mean()),
        "donor_coverage_with_LEC_percent": 100.0 * float(donors["has_at_least_one_LEC"].mean()),
        "n_all_QC_cells": n_all,
        "n_EC_cells": n_ec,
        "n_LEC_cells": n_lec,
        "EC_fraction_among_all_QC_pooled": safe_divide(n_ec, n_all),
        "EC_percent_among_all_QC_pooled": 100.0 * safe_divide(n_ec, n_all),
        "LEC_fraction_among_all_QC_pooled": safe_divide(n_lec, n_all),
        "LEC_percent_among_all_QC_pooled": 100.0 * safe_divide(n_lec, n_all),
        "LEC_fraction_among_EC_pooled": safe_divide(n_lec, n_ec),
        "LEC_percent_among_EC_pooled": 100.0 * safe_divide(n_lec, n_ec),
        "has_within_dataset_NF": bool(within_nf),
        "comparison_limitation": comparison_limitation,
        "LEC_definition": "fresh EC-only exact cluster majority vote EC8_ln_or_lymphatic",
        "all_cell_denominator": "Step 03 QC-passed cells within the same dataset-condition cohort",
        "EC_denominator": "Step 06 selected EC candidates within the same dataset-condition cohort",
        "interpretation": "descriptive_not_cross_dataset_inference",
    }
    result.update(metric_summary(
        donors["EC_percent_among_all_QC"], "donor_EC_percent_among_all_QC"
    ))
    result.update(metric_summary(
        donors["LEC_percent_among_all_QC"], "donor_LEC_percent_among_all_QC"
    ))
    result.update(metric_summary(
        donors["LEC_percent_among_EC"], "donor_LEC_percent_among_EC_among_donors_with_EC"
    ))
    return result


def compact_lec_metadata(
    ec_obs: pd.DataFrame,
    dataset: str,
    condition: str,
) -> pd.DataFrame:
    lec = ec_obs.loc[true_mask(ec_obs["selected_as_LEC"])].copy()
    lec["donor_key"] = donor_key_from_obs(lec)
    columns = [
        "donor_key", "donor_id", "donor_id_source", "biosample_id", "source_id",
        "sex", "age", "lvef", "disease", "region_standardized_analysis",
        "allcell_leiden", "leiden", "ec_recluster_id",
        "celltypist_raw_label", "celltypist_majority_vote",
        "celltypist_max_probability", "final_LEC_by_majority_vote",
        "step05_celltypist_raw_label", "step05_celltypist_majority_vote",
        "qc_n_genes", "qc_total_counts", "qc_pct_mt",
    ]
    columns = [column for column in columns if column in lec]
    result = lec[columns].copy()
    result.insert(0, "cohort", cohort_name(dataset, condition))
    result.insert(0, "condition_output", condition)
    result.insert(0, "dataset_folder_output", dataset)
    result.insert(0, "cell_id", result.index.astype(str))
    return result.reset_index(drop=True)


def grouped_counts(
    all_obs: pd.DataFrame,
    ec_obs: pd.DataFrame,
    dataset: str,
    condition: str,
    group_column: str,
) -> pd.DataFrame:
    if group_column not in all_obs or group_column not in ec_obs:
        return pd.DataFrame()
    all_meta = all_obs.copy()
    ec_meta = ec_obs.copy()
    if group_column == "sex":
        all_meta["group_value"] = normalize_sex(all_meta[group_column])
        ec_meta["group_value"] = normalize_sex(ec_meta[group_column])
    else:
        all_meta["group_value"] = normalized_text(all_meta[group_column]).replace("", "Unknown")
        ec_meta["group_value"] = normalized_text(ec_meta[group_column]).replace("", "Unknown")
    all_meta["donor_key"] = donor_key_from_obs(all_meta)
    ec_meta["donor_key"] = donor_key_from_obs(ec_meta)
    lec_meta = ec_meta.loc[true_mask(ec_meta["selected_as_LEC"])]
    all_table = all_meta.groupby("group_value", observed=True).agg(
        n_all_QC_cells=("donor_key", "size"), n_donors_all=("donor_key", "nunique")
    )
    ec_table = ec_meta.groupby("group_value", observed=True).agg(
        n_EC_cells=("donor_key", "size"), n_donors_with_EC=("donor_key", "nunique")
    )
    lec_table = lec_meta.groupby("group_value", observed=True).agg(
        n_LEC_cells=("donor_key", "size"), n_donors_with_LEC=("donor_key", "nunique")
    )
    table = pd.concat([all_table, ec_table, lec_table], axis=1).fillna(0).reset_index()
    count_columns = [column for column in table if column.startswith("n_")]
    table[count_columns] = table[count_columns].astype(int)
    table["EC_percent_among_all_QC"] = np.where(
        table["n_all_QC_cells"].gt(0), 100.0 * table["n_EC_cells"] / table["n_all_QC_cells"], np.nan
    )
    table["LEC_percent_among_all_QC"] = np.where(
        table["n_all_QC_cells"].gt(0), 100.0 * table["n_LEC_cells"] / table["n_all_QC_cells"], np.nan
    )
    table["LEC_percent_among_EC"] = np.where(
        table["n_EC_cells"].gt(0), 100.0 * table["n_LEC_cells"] / table["n_EC_cells"], np.nan
    )
    table.insert(0, "grouping_variable", group_column)
    table.insert(0, "cohort", cohort_name(dataset, condition))
    table.insert(0, "condition", condition)
    table.insert(0, "dataset_folder", dataset)
    return table


def build_color_key(step06_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for name, color in CONDITION_COLORS.items():
        rows.append({
            "category_type": "condition", "name": name, "color_hex": color,
            "hatch": CONDITION_HATCHES[name], "source": "Step07_fixed_Okabe_Ito",
        })
    for name, color in DATASET_COLORS.items():
        rows.append({
            "category_type": "dataset", "name": name, "color_hex": color,
            "hatch": "", "source": "Step07_fixed_Paul_Tol",
        })
    for name, color in STAGE_COLORS.items():
        rows.append({
            "category_type": "cell_stage_or_LEC_call", "name": name,
            "color_hex": color, "hatch": "", "source": "Step06_preserved_or_Step07_fixed",
        })
    for name, color in DONOR_STATUS_COLORS.items():
        rows.append({
            "category_type": "donor_LEC_status", "name": name,
            "color_hex": color, "hatch": "", "source": "Step07_fixed",
        })
    for name, color in SEX_COLORS.items():
        rows.append({
            "category_type": "sex", "name": name, "color_hex": color,
            "hatch": "", "source": "Step07_fixed",
        })
    prior_path = step06_root / "06_celltypist_color_key.csv"
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        if {"celltypist_label", "color"}.issubset(prior.columns):
            for _, item in prior.iterrows():
                rows.append({
                    "category_type": "CellTypist_label",
                    "name": str(item["celltypist_label"]),
                    "color_hex": str(item["color"]),
                    "hatch": "",
                    "source": "Step06_preserved",
                })
    return pd.DataFrame(rows)


def cohort_tick_labels(summary: pd.DataFrame) -> list[str]:
    return [
        f"{row.dataset_folder}\n{row.condition}"
        for row in summary.itertuples()
    ]


def donor_display_label(row: pd.Series) -> str:
    for column in ["donor_id_source", "donor_id", "biosample_id", "source_id"]:
        if column in row.index:
            value = str(row[column]).strip()
            if value and value.casefold() not in {"nan", "none"}:
                return value
    value = str(row.get("donor_key", ""))
    return value.split("::", 1)[-1]


def condition_legend(conditions: list[str]) -> list[Patch]:
    ordered = [condition for condition in CONDITION_ORDER if condition in conditions]
    return [
        Patch(
            facecolor=CONDITION_COLORS[condition], edgecolor="black",
            hatch=CONDITION_HATCHES[condition], label=condition,
        )
        for condition in ordered
    ]


def label_percent_bars(ax, bars, values: np.ndarray) -> None:
    labels = []
    for value in values:
        if not np.isfinite(value):
            labels.append("NA")
        elif value == 0:
            labels.append("0")
        elif value < 0.1:
            labels.append(f"{value:.3f}%")
        elif value < 1:
            labels.append(f"{value:.2f}%")
        else:
            labels.append(f"{value:.1f}%")
    ax.bar_label(bars, labels=labels, padding=3, fontsize=7.5, rotation=90)


def plot_pooled_percentages(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    specifications = [
        ("EC_percent_among_all_QC_pooled", "EC candidates among all QC cells", "EC / all QC (%)"),
        ("LEC_percent_among_all_QC_pooled", "LEC among all QC cells", "LEC / all QC (%)"),
        ("LEC_percent_among_EC_pooled", "LEC among EC candidates", "LEC / EC (%)"),
    ]
    x = np.arange(len(summary))
    colors = [CONDITION_COLORS[str(value)] for value in summary["condition"]]
    hatches = [CONDITION_HATCHES[str(value)] for value in summary["condition"]]
    figure_width = max(21.5, 1.9 * len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(figure_width, 7.1))
    for ax, (column, title, ylabel) in zip(axes, specifications):
        values = pd.to_numeric(summary[column], errors="coerce").to_numpy(float)
        bars = ax.bar(x, values, width=0.72, color=colors, edgecolor="black", linewidth=0.7)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_xticks(x, cohort_tick_labels(summary), rotation=38, ha="right")
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        finite = values[np.isfinite(values)]
        upper = max(float(finite.max()) * 1.28 if len(finite) else 1.0, 0.15)
        ax.set_ylim(0, upper)
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        label_percent_bars(ax, bars, values)
    axes[-1].legend(
        handles=condition_legend(summary["condition"].astype(str).tolist()),
        title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
    )
    fig.suptitle("Pooled cell-level EC and LEC abundance (descriptive)", y=0.985, fontsize=14)
    fig.tight_layout(rect=(0.01, 0.07, 0.94, 0.94), w_pad=2.6)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cell_counts(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    specifications = [
        ("n_all_QC_cells", "All QC-passed cells", STAGE_COLORS["All QC-passed cells"]),
        ("n_EC_cells", "EC candidates", STAGE_COLORS["EC candidates"]),
        ("n_LEC_cells", "LEC", STAGE_COLORS["LEC"]),
    ]
    x = np.arange(len(summary))
    figure_width = max(21.5, 1.9 * len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(figure_width, 6.8))
    for ax, (column, label, color) in zip(axes, specifications):
        values = pd.to_numeric(summary[column], errors="coerce").fillna(0).to_numpy(int)
        bars = ax.bar(x, values, width=0.72, color=color, edgecolor="black", linewidth=0.7)
        ax.set_xticks(x, cohort_tick_labels(summary), rotation=38, ha="right")
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel("Cells")
        ax.set_title(label)
        ax.set_ylim(0, max(values.max() * 1.22, 1))
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.bar_label(bars, labels=[f"{value:,}" for value in values], padding=3, fontsize=7.5, rotation=90)
    fig.suptitle("Cell counts at each stage (separate y-axis per panel)", y=0.985, fontsize=14)
    fig.tight_layout(rect=(0.01, 0.07, 0.99, 0.94), w_pad=2.6)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_donor_percentages(
    donors: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
    dpi: int,
    seed: int,
) -> None:
    specifications = [
        ("EC_percent_among_all_QC", "EC / all QC per donor"),
        ("LEC_percent_among_all_QC", "LEC / all QC per donor"),
        ("LEC_percent_among_EC", "LEC / EC per donor with EC"),
    ]
    cohorts = summary["cohort"].astype(str).tolist()
    x = np.arange(len(cohorts))
    rng = np.random.default_rng(seed)
    figure_width = max(22.0, 1.9 * len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(figure_width, 7.2))
    for ax, (column, title) in zip(axes, specifications):
        for index, cohort in enumerate(cohorts):
            subset = donors.loc[donors["cohort"].eq(cohort)]
            values = pd.to_numeric(subset[column], errors="coerce").dropna().to_numpy(float)
            condition = str(summary.loc[summary["cohort"].eq(cohort), "condition"].iloc[0])
            jitter = rng.uniform(-0.18, 0.18, len(values))
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=28,
                color=CONDITION_COLORS[condition],
                alpha=0.78,
                edgecolor="black",
                linewidth=0.35,
                zorder=3,
            )
            if len(values):
                q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
                ax.vlines(index, q1, q3, color="black", linewidth=5.5, zorder=4)
                ax.vlines(index, q1, q3, color="white", linewidth=2.6, zorder=5)
                ax.hlines(median, index - 0.22, index + 0.22, color="black", linewidth=2.2, zorder=6)
        ax.set_xticks(x, [label.replace(" | ", "\n") for label in cohorts], rotation=38, ha="right")
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel("Percentage per donor (%)")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(
        handles=condition_legend(summary["condition"].astype(str).tolist()),
        title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
    )
    fig.suptitle("Donor-level abundance (points; black/white IQR and black median)", y=0.985, fontsize=14)
    fig.tight_layout(rect=(0.01, 0.07, 0.94, 0.94), w_pad=2.6)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_donor_coverage(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    x = np.arange(len(summary))
    positive = pd.to_numeric(summary["n_donors_with_LEC"], errors="coerce").fillna(0).to_numpy(int)
    total = pd.to_numeric(summary["n_donors_all"], errors="coerce").fillna(0).to_numpy(int)
    negative = total - positive
    figure_width = max(14.5, 1.25 * len(summary))
    fig, ax = plt.subplots(figsize=(figure_width, 6.8))
    ax.bar(
        x, positive, width=0.72, color=DONOR_STATUS_COLORS["At least one LEC"],
        edgecolor="black", linewidth=0.7, label="At least one LEC",
    )
    ax.bar(
        x, negative, width=0.72, bottom=positive,
        color=DONOR_STATUS_COLORS["No LEC detected"], edgecolor="black",
        linewidth=0.7, label="No LEC detected",
    )
    ax.set_xticks(x, cohort_tick_labels(summary), rotation=35, ha="right")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylabel("Donors")
    ax.set_title("Donor coverage of final LEC calls")
    ax.set_ylim(0, max(total.max() + 4, 1))
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for index, (numerator, denominator) in enumerate(zip(positive, total)):
        ax.text(index, denominator + 0.35, f"{numerator}/{denominator}", ha="center", fontsize=8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout(rect=(0.02, 0.06, 0.91, 0.97))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_within_dataset_donor_comparisons(
    donors: pd.DataFrame,
    path: Path,
    dpi: int,
    seed: int,
) -> None:
    datasets = [
        dataset for dataset in DATASET_COLORS
        if donors.loc[donors["dataset_folder"].eq(dataset), "condition"].nunique() > 1
    ]
    columns = 2
    rows = math.ceil(len(datasets) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 5.2 * rows), squeeze=False)
    rng = np.random.default_rng(seed)
    for ax, dataset in zip(axes.flat, datasets):
        subset = donors.loc[donors["dataset_folder"].eq(dataset)]
        conditions = [condition for condition in CONDITION_ORDER if condition in set(subset["condition"])]
        for index, condition in enumerate(conditions):
            values = pd.to_numeric(
                subset.loc[subset["condition"].eq(condition), "LEC_percent_among_all_QC"],
                errors="coerce",
            ).dropna().to_numpy(float)
            jitter = rng.uniform(-0.14, 0.14, len(values))
            ax.scatter(
                np.full(len(values), index) + jitter, values, s=35,
                color=CONDITION_COLORS[condition], edgecolor="black",
                linewidth=0.4, alpha=0.8, zorder=3,
            )
            if len(values):
                q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
                ax.vlines(index, q1, q3, color="black", linewidth=4, zorder=4)
                ax.hlines(median, index - 0.18, index + 0.18, color="black", linewidth=2, zorder=5)
        ax.set_xticks(np.arange(len(conditions)), conditions)
        ax.set_ylabel("LEC / all QC cells per donor (%)")
        ax.set_title(dataset, fontsize=12, pad=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(datasets):]:
        ax.axis("off")
    handles = condition_legend(donors["condition"].astype(str).tolist())
    fig.legend(
        handles=handles, title="Condition", loc="upper center",
        bbox_to_anchor=(0.5, 0.947), ncol=len(handles), frameon=False,
    )
    fig.suptitle(
        "Within-dataset donor distributions only (descriptive; no cross-dataset pooling)",
        y=0.992, fontsize=14,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.89), h_pad=2.4, w_pad=2.2)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_LEC_counts_by_condition_overview(
    donors: pd.DataFrame,
    path: Path,
    dpi: int,
    seed: int,
    low_threshold: int,
) -> None:
    """Show every donor's absolute LEC count, grouped by condition."""
    conditions = [
        condition for condition in CONDITION_ORDER
        if condition in set(donors["condition"].astype(str))
    ]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    for index, condition in enumerate(conditions):
        condition_table = donors.loc[donors["condition"].eq(condition)]
        for dataset in DATASET_COLORS:
            subset = condition_table.loc[condition_table["dataset_folder"].eq(dataset)]
            if subset.empty:
                continue
            values = pd.to_numeric(subset["n_LEC_cells"], errors="coerce").fillna(0).to_numpy(float)
            jitter = rng.uniform(-0.22, 0.22, len(values))
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=42,
                marker=DATASET_MARKERS[dataset],
                color=CONDITION_COLORS[condition],
                edgecolor="black",
                linewidth=0.45,
                alpha=0.82,
                zorder=3,
            )
    maximum = int(pd.to_numeric(donors["n_LEC_cells"], errors="coerce").fillna(0).max())
    linthresh = max(1, int(low_threshold))
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.2, base=10)
    tick_candidates = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ticks = [value for value in tick_candidates if value <= max(maximum * 1.15, low_threshold + 1)]
    if maximum not in ticks and maximum > 0:
        ticks.append(maximum)
    ax.set_yticks(sorted(set(ticks)))
    ax.set_yticklabels([f"{value:,}" for value in sorted(set(ticks))])
    ax.axhline(
        low_threshold, color=STAGE_COLORS["LEC"], linestyle="--",
        linewidth=1.3, alpha=0.8, label=f"Low-count threshold ({low_threshold})",
    )
    ax.set_xticks(np.arange(len(conditions)), conditions)
    ax.set_xlabel("Condition")
    ax.set_ylabel("LEC cells per donor (symlog scale)")
    ax.set_title(
        "Absolute LEC cells per donor by condition\n"
        "Linear detail from 0 to the red threshold; compressed above the threshold"
    )
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    condition_handles = condition_legend(conditions)
    condition_legend_artist = ax.legend(
        handles=condition_handles, title="Condition color",
        bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
    )
    ax.add_artist(condition_legend_artist)
    dataset_handles = [
        Line2D(
            [0], [0], marker=DATASET_MARKERS[dataset], linestyle="none",
            markerfacecolor="white", markeredgecolor="black", markersize=7,
            label=dataset,
        )
        for dataset in DATASET_COLORS
        if dataset in set(donors["dataset_folder"])
    ]
    ax.legend(
        handles=dataset_handles, title="Dataset marker",
        bbox_to_anchor=(1.02, 0.53), loc="upper left", frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.82, 0.98))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cohort_donor_LEC_counts(
    donors: pd.DataFrame,
    dataset: str,
    condition: str,
    path: Path,
    dpi: int,
    low_threshold: int,
) -> None:
    """One readable, exact-value donor chart for a dataset-condition cohort."""
    table = donors.loc[
        donors["dataset_folder"].eq(dataset) & donors["condition"].eq(condition)
    ].copy()
    table["n_LEC_cells"] = pd.to_numeric(table["n_LEC_cells"], errors="coerce").fillna(0).astype(int)
    table["donor_label"] = table.apply(donor_display_label, axis=1)
    table = table.sort_values(["n_LEC_cells", "donor_label"], ascending=[True, True]).reset_index(drop=True)
    height = max(4.8, 2.2 + 0.32 * len(table))
    fig, ax = plt.subplots(figsize=(10.5, height))
    y = np.arange(len(table))
    bars = ax.barh(
        y,
        table["n_LEC_cells"],
        color=CONDITION_COLORS[condition],
        edgecolor="black",
        linewidth=0.7,
        hatch=CONDITION_HATCHES[condition],
        alpha=0.9,
    )
    maximum = int(table["n_LEC_cells"].max()) if len(table) else 0
    label_offset = max(maximum * 0.012, 0.35)
    for bar, value in zip(bars, table["n_LEC_cells"]):
        if value == 0:
            bar.set_edgecolor("#7F7F7F")
            ax.scatter(0, bar.get_y() + bar.get_height() / 2, s=18, color="#7F7F7F", zorder=4)
            text_color = "#6B6B6B"
        elif value <= low_threshold:
            bar.set_edgecolor(STAGE_COLORS["LEC"])
            bar.set_linewidth(2.0)
            text_color = STAGE_COLORS["LEC"]
        else:
            text_color = "black"
        ax.text(
            float(value) + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            va="center",
            ha="left",
            fontsize=8,
            color=text_color,
            fontweight="bold" if 0 < value <= low_threshold else "normal",
        )
    ax.set_yticks(y, table["donor_label"].astype(str))
    ax.invert_yaxis()
    for tick, value in zip(ax.get_yticklabels(), table["n_LEC_cells"]):
        if value == 0:
            tick.set_color("#6B6B6B")
        elif value <= low_threshold:
            tick.set_color(STAGE_COLORS["LEC"])
            tick.set_fontweight("bold")
    if maximum > low_threshold:
        ax.axvline(
            low_threshold, color=STAGE_COLORS["LEC"], linestyle="--",
            linewidth=1.2, alpha=0.8,
        )
    ax.set_xlim(0, max(maximum + 5 * label_offset, low_threshold * 1.25, 1))
    ax.set_xlabel("LEC cells")
    ax.set_ylabel("Donor")
    ax.set_title(
        f"{dataset} | {condition}: absolute LEC count per donor\n"
        f"Red labels = 1-{low_threshold} LEC; gray labels = 0 LEC"
    )
    ax.grid(axis="x", alpha=0.2, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe disease-pipeline LEC percentages among all QC cells and EC candidates."
    )
    parser.add_argument("--step03_root", default=DEFAULT_STEP03_ROOT)
    parser.add_argument("--step06_root", default=DEFAULT_STEP06_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cohorts", default="all", help="all or comma-separated dataset:condition values")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument(
        "--low_lec_threshold",
        type=int,
        default=25,
        help="Flag donors with 1 to this many LECs; zero-LEC donors are flagged separately.",
    )
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dpi < 72:
        raise ValueError("dpi must be at least 72.")
    if args.low_lec_threshold < 1:
        raise ValueError("low_lec_threshold must be at least 1.")
    project_dir = find_project_dir()
    step03_root = resolve_path(project_dir, args.step03_root)
    step06_root = resolve_path(project_dir, args.step06_root)
    output_root = resolve_path(project_dir, args.output_root)
    step03 = read_step03_master(step03_root)
    step06 = read_step06_summary(step06_root)
    pipeline = join_pipeline_summaries(step03, step06)
    subset = parse_cohort_subset(args.cohorts)
    if subset is not None:
        pipeline = pipeline.loc[
            [(dataset, condition) in subset for dataset, condition in zip(
                pipeline["dataset_folder"], pipeline["condition"]
            )]
        ].copy()

    print("=" * 100)
    print("Selected disease Step 07: LEC abundance description")
    print(f"Step 03 root: {step03_root}")
    print(f"Step 06 root: {step06_root}")
    print(f"Output root: {output_root}")
    print(f"Cohorts: {', '.join(pipeline['cohort'])}")
    print("Denominators: Step 03 all QC cells; Step 06 EC candidates")
    print("Donor table: all retained donors, including zero-EC and zero-LEC donors")
    print("Colors: fixed by semantic name; condition hatches and black outlines enabled")
    print("=" * 100)

    validations: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    donor_tables: list[pd.DataFrame] = []
    lec_tables: list[pd.DataFrame] = []
    sex_tables: list[pd.DataFrame] = []
    region_tables: list[pd.DataFrame] = []
    for row in pipeline.itertuples(index=False):
        dataset = str(row.dataset_folder)
        condition = str(row.condition)
        all_path = resolve_path(project_dir, str(row.output_h5ad_step03))
        ec_path = resolve_path(project_dir, str(row.output_h5ad_step06))
        all_obs = read_obs(all_path)
        ec_obs = read_obs(ec_path)
        validation = validate_cohort(
            all_obs, ec_obs, dataset, condition, all_path, ec_path,
            int(row.n_cells_after_all_QC), int(row.n_EC_cells), int(row.n_LEC_cells),
        )
        donors = donor_table(all_obs, ec_obs, dataset, condition)
        validation["n_donors_with_EC"] = int(donors["has_at_least_one_EC"].sum())
        validation["n_donors_with_LEC"] = int(donors["has_at_least_one_LEC"].sum())
        validations.append(validation)
        summaries.append(summarize_cohort(all_obs, ec_obs, donors, dataset, condition))
        donor_tables.append(donors)
        lec_tables.append(compact_lec_metadata(ec_obs, dataset, condition))
        sex = grouped_counts(all_obs, ec_obs, dataset, condition, "sex")
        if not sex.empty:
            sex_tables.append(sex)
        region = grouped_counts(
            all_obs, ec_obs, dataset, condition, "region_standardized_analysis"
        )
        if not region.empty:
            region_tables.append(region)

    validation_table = pd.DataFrame(validations)
    summary_table = pd.DataFrame(summaries)
    order = {pair: index for index, pair in enumerate(EXPECTED_COHORTS)}
    summary_table["_order"] = [
        order[(dataset, condition)]
        for dataset, condition in zip(summary_table["dataset_folder"], summary_table["condition"])
    ]
    summary_table = summary_table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    donor_result = pd.concat(donor_tables, ignore_index=True)
    donor_result["zero_LEC_count"] = donor_result["n_LEC_cells"].eq(0)
    donor_result["small_positive_LEC_count"] = donor_result["n_LEC_cells"].between(
        1, args.low_lec_threshold, inclusive="both"
    )
    donor_result["low_LEC_count_flag"] = (
        donor_result["zero_LEC_count"] | donor_result["small_positive_LEC_count"]
    )
    donor_result["low_LEC_threshold"] = int(args.low_lec_threshold)
    low_lec_donors = donor_result.loc[donor_result["low_LEC_count_flag"]].copy()
    nonempty_lec_tables = [table for table in lec_tables if not table.empty]
    lec_result = (
        pd.concat(nonempty_lec_tables, ignore_index=True)
        if nonempty_lec_tables else pd.DataFrame()
    )
    color_key = build_color_key(step06_root)

    print(summary_table[[
        "cohort", "n_donors_all", "n_donors_with_LEC", "n_all_QC_cells",
        "n_EC_cells", "n_LEC_cells", "LEC_percent_among_all_QC_pooled",
        "LEC_percent_among_EC_pooled",
    ]].to_string(index=False))
    if args.validate_only:
        print("Validation complete; no Step 07 outputs were created.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "step03_root": str(step03_root),
        "step06_root": str(step06_root),
        "output_root": str(output_root),
        "cohorts": pipeline["cohort"].tolist(),
        "LEC_definition": "fresh EC-only exact cluster majority vote EC8_ln_or_lymphatic",
        "all_cell_denominator": "Step 03 QC-passed cells within cohort",
        "EC_denominator": "Step 06 selected EC candidates within cohort",
        "donor_policy": "retain_all_donors_including_zero_EC_and_zero_LEC",
        "interpretation": "descriptive_not_cross_dataset_inference",
        "color_policy": "same_semantic_name_same_color_plus_condition_hatches",
        "condition_colors": CONDITION_COLORS,
        "condition_hatches": CONDITION_HATCHES,
        "stage_colors": STAGE_COLORS,
        "dpi": args.dpi,
        "random_seed": args.random_seed,
        "low_LEC_threshold": args.low_lec_threshold,
        "low_LEC_flag_definition": "zero_or_1_through_threshold_absolute_LEC_cells_per_donor",
        "figures_enabled": not args.skip_figures,
    }
    with (output_root / "07_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    validation_table.to_csv(output_root / "07_input_validation_summary.csv", index=False)
    summary_table.to_csv(output_root / "07_LEC_percentage_summary_by_cohort.csv", index=False)
    donor_result.to_csv(output_root / "07_LEC_percentage_by_donor.csv", index=False)
    low_lec_donors.to_csv(
        output_root / "07_donors_with_low_LEC_counts.csv", index=False
    )
    lec_result.to_csv(
        output_root / "07_final_LEC_cell_metadata.csv.gz", index=False, compression="gzip"
    )
    color_key.to_csv(output_root / "07_color_key.csv", index=False)
    if sex_tables:
        pd.concat(sex_tables, ignore_index=True).to_csv(
            output_root / "07_LEC_counts_by_sex_and_cohort.csv", index=False
        )
    if region_tables:
        pd.concat(region_tables, ignore_index=True).to_csv(
            output_root / "07_LEC_counts_by_region_and_cohort.csv", index=False
        )

    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        figure_dir = output_root / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        specifications = [
            (
                "07_pooled_EC_LEC_percentages_by_cohort.png",
                "Pooled EC and LEC percentages by cohort",
                lambda path: plot_pooled_percentages(summary_table, path, args.dpi),
            ),
            (
                "07_cell_counts_all_EC_LEC_by_cohort.png",
                "All, EC and LEC cell counts by cohort",
                lambda path: plot_cell_counts(summary_table, path, args.dpi),
            ),
            (
                "07_donor_level_EC_LEC_percentages_by_cohort.png",
                "Donor-level EC and LEC percentages",
                lambda path: plot_donor_percentages(
                    donor_result, summary_table, path, args.dpi, args.random_seed
                ),
            ),
            (
                "07_LEC_donor_coverage_by_cohort.png",
                "Donor coverage of LEC calls",
                lambda path: plot_donor_coverage(summary_table, path, args.dpi),
            ),
            (
                "07_within_dataset_donor_LEC_comparisons.png",
                "Within-dataset donor LEC comparisons",
                lambda path: plot_within_dataset_donor_comparisons(
                    donor_result, path, args.dpi, args.random_seed
                ),
            ),
            (
                "07_LEC_cells_per_donor_by_condition_overview.png",
                "Absolute LEC cell counts per donor grouped by condition",
                lambda path: plot_LEC_counts_by_condition_overview(
                    donor_result, path, args.dpi, args.random_seed,
                    args.low_lec_threshold,
                ),
            ),
        ]
        for filename, description, function in specifications:
            path = figure_dir / filename
            function(path)
            figure_manifest.append({
                "figure": str(path),
                "description": description,
                "color_policy": "same_semantic_name_same_color",
            })
        donor_figure_dir = figure_dir / "LEC_counts_per_donor_by_cohort"
        donor_figure_dir.mkdir(parents=True, exist_ok=True)
        for row in summary_table.itertuples(index=False):
            filename = (
                f"07_{safe_name(row.dataset_folder)}_{safe_name(row.condition)}_"
                "LEC_cells_per_donor.png"
            )
            path = donor_figure_dir / filename
            plot_cohort_donor_LEC_counts(
                donor_result,
                str(row.dataset_folder),
                str(row.condition),
                path,
                args.dpi,
                args.low_lec_threshold,
            )
            figure_manifest.append({
                "figure": str(path),
                "description": (
                    f"Absolute LEC counts for every donor in {row.dataset_folder} | "
                    f"{row.condition}; low threshold={args.low_lec_threshold}"
                ),
                "color_policy": "condition_color_preserved_low_counts_red_outline",
            })
        pd.DataFrame(figure_manifest).to_csv(
            output_root / "07_figure_manifest.csv", index=False
        )

    print(f"Outputs: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
