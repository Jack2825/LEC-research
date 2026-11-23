#!/usr/bin/env python3
r"""
Step 07: describe final NF LEC abundance across donor-age quartiles.

This standalone script uses:

* Step 03 all QC-passed cells as the all-cell denominator.
* Step 06 EC-reclustered cells as the EC denominator.
* Step 06 ``selected_as_LEC`` cells, defined by the fresh EC-only CellTypist
  majority label EC8_ln, as the LEC numerator.

It validates that LEC is a subset of EC and EC is a subset of all QC cells,
then reports both pooled cell-level percentages and donor-level percentages.
The latter are important for paper-level interpretation because donors have
different numbers of captured nuclei.

Default output
--------------
new_result2_aging/07_LEC_description/dataset13/

Main outputs
------------
07_LEC_percentage_summary_by_quartile.csv
07_LEC_percentage_by_donor.csv
07_LEC_counts_by_sex_and_quartile.csv
07_LEC_counts_by_exact_age_and_quartile.csv
07_final_LEC_cell_metadata.csv.gz
07_input_validation_summary.csv
figures/*.png

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_aging\07_describe_LEC_percentages_age_quartiles.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "07_dataset13_NF_LEC_description_2026_08_06"
DATASET = "dataset13"
REQUIRED_DISEASE = "NF"
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]

DEFAULT_ALLCELL_ROOT = "new_result2_aging/03/dataset13"
DEFAULT_EC_ROOT = "new_result2_aging/06_EC_recluster/dataset13"
DEFAULT_OUTPUT_ROOT = "new_result2_aging/07_LEC_description/dataset13"


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Describe final majority-vote LEC percentages among all QC cells "
            "and among EC cells for dataset13 NF age quartiles."
        )
    )
    parser.add_argument("--allcell_root", default=DEFAULT_ALLCELL_ROOT)
    parser.add_argument("--ec_root", default=DEFAULT_EC_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--quartiles",
        nargs="+",
        choices=QUARTILES,
        default=QUARTILES,
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--skip_figures",
        action="store_true",
        help="Write description CSVs but skip figures.",
    )
    return parser.parse_args()


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def true_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return normalized_text(values).str.lower().isin({"true", "1", "yes", "y", "lec"})


def safe_divide(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0 else np.nan


def first_nonempty(values: pd.Series):
    text = normalized_text(values)
    nonempty = text.loc[text.ne("")]
    return nonempty.iloc[0] if len(nonempty) else ""


def first_numeric(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.iloc[0]) if len(numeric) else np.nan


def donor_key_from_obs(obs: pd.DataFrame) -> pd.Series:
    if "donor_id_quartile_unique" in obs.columns:
        key = normalized_text(obs["donor_id_quartile_unique"])
        if key.ne("").all():
            return key
    if "Project" in obs.columns and "donor_id" in obs.columns:
        return normalized_text(obs["Project"]) + "::" + normalized_text(obs["donor_id"])
    raise KeyError(
        "Cannot construct a namespaced donor identifier. Expected "
        "donor_id_quartile_unique or Project + donor_id."
    )


def validate_group(
    all_obs: pd.DataFrame,
    ec_obs: pd.DataFrame,
    quartile: str,
    all_path: Path,
    ec_path: Path,
) -> dict[str, object]:
    required_all = ["disease", "age_quartile"]
    missing_all = [column for column in required_all if column not in all_obs.columns]
    if missing_all:
        raise KeyError(f"Missing {missing_all} from Step 03 obs: {all_path}")
    required_ec = [
        "disease",
        "age_quartile",
        "selected_as_LEC",
        "celltypist_majority_vote",
        "leiden",
    ]
    missing_ec = [column for column in required_ec if column not in ec_obs.columns]
    if missing_ec:
        raise KeyError(f"Missing {missing_ec} from Step 06 obs: {ec_path}")

    all_disease = set(normalized_text(all_obs["disease"]).str.upper().unique())
    ec_disease = set(normalized_text(ec_obs["disease"]).str.upper().unique())
    if all_disease != {REQUIRED_DISEASE} or ec_disease != {REQUIRED_DISEASE}:
        raise ValueError(
            f"Expected NF only for {quartile}; Step03={all_disease}, Step06={ec_disease}"
        )
    all_q = set(normalized_text(all_obs["age_quartile"]).unique())
    ec_q = set(normalized_text(ec_obs["age_quartile"]).unique())
    if all_q != {quartile} or ec_q != {quartile}:
        raise ValueError(
            f"Quartile mismatch for {quartile}; Step03={all_q}, Step06={ec_q}"
        )

    all_ids = set(all_obs.index.astype(str))
    ec_ids = set(ec_obs.index.astype(str))
    lec_flag = true_mask(ec_obs["selected_as_LEC"])
    lec_ids = set(ec_obs.index[lec_flag].astype(str))
    if not ec_ids.issubset(all_ids):
        raise ValueError(
            f"{len(ec_ids - all_ids):,} Step 06 EC cells are absent from Step 03 for {quartile}."
        )
    if not lec_ids.issubset(ec_ids):
        raise ValueError(f"LEC is not a subset of EC for {quartile}.")

    selected_votes = set(
        normalized_text(ec_obs.loc[lec_flag, "celltypist_majority_vote"]).unique()
    )
    if selected_votes != {"EC8_ln"}:
        raise ValueError(
            f"Final LEC cells for {quartile} do not have only EC8_ln majority votes: "
            f"{sorted(selected_votes)}"
        )
    return {
        "dataset": DATASET,
        "disease": REQUIRED_DISEASE,
        "age_quartile": quartile,
        "status": "validated",
        "step03_allcell_h5ad": str(all_path),
        "step06_EC_h5ad": str(ec_path),
        "n_all_QC_cells": int(len(all_obs)),
        "n_EC_cells": int(len(ec_obs)),
        "n_LEC_cells": int(lec_flag.sum()),
        "n_EC_cells_absent_from_allcell": 0,
        "n_LEC_cells_absent_from_EC": 0,
        "final_LEC_majority_label": "EC8_ln",
    }


def compact_cell_metadata(ec_obs: pd.DataFrame, quartile: str) -> pd.DataFrame:
    lec = ec_obs.loc[true_mask(ec_obs["selected_as_LEC"])].copy()
    lec["donor_key"] = donor_key_from_obs(lec)
    useful = [
        "donor_key",
        "donor_id_quartile_unique",
        "donor_id",
        "Donor",
        "Project",
        "age_numeric_resolved",
        "age_source",
        "age_quartile",
        "age_quartile_range_label",
        "sex",
        "Sex",
        "disease",
        "allcell_leiden",
        "leiden",
        "celltypist_raw_label",
        "celltypist_majority_vote",
        "celltypist_max_probability",
        "final_LEC_by_majority_vote",
        "step05_celltypist_raw_label",
        "step05_celltypist_majority_vote",
        "qc_n_genes",
        "qc_total_counts",
        "qc_pct_mt",
    ]
    useful = list(dict.fromkeys(column for column in useful if column in lec.columns))
    result = lec[useful].copy()
    result.insert(0, "age_quartile_output", quartile)
    result.insert(0, "disease_output", REQUIRED_DISEASE)
    result.insert(0, "dataset", DATASET)
    result.insert(0, "cell_id", result.index.astype(str))
    return result.reset_index(drop=True)


def make_donor_table(
    all_obs: pd.DataFrame,
    ec_obs: pd.DataFrame,
    quartile: str,
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

    metadata_columns = [
        column
        for column in [
            "donor_id",
            "Donor",
            "Project",
            "source_id",
            "sex",
            "Sex",
            "age_numeric_resolved",
            "age_source",
            "age_quartile_range_label",
        ]
        if column in all_meta.columns
    ]
    metadata_rows: list[dict[str, object]] = []
    for donor_key, group in all_meta.groupby("donor_key", observed=True, sort=True):
        row: dict[str, object] = {"donor_key": donor_key}
        for column in metadata_columns:
            if column == "age_numeric_resolved":
                row[column] = first_numeric(group[column])
            else:
                row[column] = first_nonempty(group[column])
        metadata_rows.append(row)
    metadata = pd.DataFrame(metadata_rows).set_index("donor_key")
    donor = metadata.join(counts, how="outer").reset_index()
    donor.insert(0, "age_quartile", quartile)
    donor.insert(0, "disease", REQUIRED_DISEASE)
    donor.insert(0, "dataset", DATASET)
    donor["EC_fraction_among_all_QC"] = donor["n_EC_cells"] / donor["n_all_QC_cells"]
    donor["EC_percent_among_all_QC"] = 100.0 * donor["EC_fraction_among_all_QC"]
    donor["LEC_fraction_among_all_QC"] = donor["n_LEC_cells"] / donor["n_all_QC_cells"]
    donor["LEC_percent_among_all_QC"] = 100.0 * donor["LEC_fraction_among_all_QC"]
    donor["LEC_fraction_among_EC"] = np.where(
        donor["n_EC_cells"].gt(0), donor["n_LEC_cells"] / donor["n_EC_cells"], np.nan
    )
    donor["LEC_percent_among_EC"] = 100.0 * donor["LEC_fraction_among_EC"]
    donor["has_at_least_one_LEC"] = donor["n_LEC_cells"].gt(0)
    return donor.sort_values(
        ["age_numeric_resolved", "donor_key"], na_position="last"
    ).reset_index(drop=True)


def summarize_quartile(
    all_obs: pd.DataFrame,
    ec_obs: pd.DataFrame,
    donor: pd.DataFrame,
    quartile: str,
) -> dict[str, object]:
    n_all = int(len(all_obs))
    n_ec = int(len(ec_obs))
    n_lec = int(true_mask(ec_obs["selected_as_LEC"]).sum())
    age = pd.to_numeric(donor.get("age_numeric_resolved"), errors="coerce")
    all_pct = donor["LEC_percent_among_all_QC"]
    ec_pct = donor["LEC_percent_among_EC"]
    range_label = (
        first_nonempty(all_obs["age_quartile_range_label"])
        if "age_quartile_range_label" in all_obs.columns
        else quartile
    )
    return {
        "dataset": DATASET,
        "disease": REQUIRED_DISEASE,
        "age_quartile": quartile,
        "age_quartile_range_label": range_label,
        "age_min_resolved": float(age.min()) if age.notna().any() else np.nan,
        "age_max_resolved": float(age.max()) if age.notna().any() else np.nan,
        "n_donors_all": int(donor["donor_key"].nunique()),
        "n_donors_with_EC": int(donor.loc[donor["n_EC_cells"].gt(0), "donor_key"].nunique()),
        "n_donors_with_LEC": int(donor.loc[donor["n_LEC_cells"].gt(0), "donor_key"].nunique()),
        "donor_coverage_with_LEC_percent": 100.0 * donor["has_at_least_one_LEC"].mean(),
        "n_all_QC_cells": n_all,
        "n_EC_cells": n_ec,
        "n_LEC_cells": n_lec,
        "EC_fraction_among_all_QC_pooled": safe_divide(n_ec, n_all),
        "EC_percent_among_all_QC_pooled": 100.0 * safe_divide(n_ec, n_all),
        "LEC_fraction_among_all_QC_pooled": safe_divide(n_lec, n_all),
        "LEC_percent_among_all_QC_pooled": 100.0 * safe_divide(n_lec, n_all),
        "LEC_fraction_among_EC_pooled": safe_divide(n_lec, n_ec),
        "LEC_percent_among_EC_pooled": 100.0 * safe_divide(n_lec, n_ec),
        "donor_LEC_percent_among_all_QC_mean": float(all_pct.mean()),
        "donor_LEC_percent_among_all_QC_median": float(all_pct.median()),
        "donor_LEC_percent_among_all_QC_Q1": float(all_pct.quantile(0.25)),
        "donor_LEC_percent_among_all_QC_Q3": float(all_pct.quantile(0.75)),
        "donor_LEC_percent_among_EC_mean": float(ec_pct.mean()),
        "donor_LEC_percent_among_EC_median": float(ec_pct.median()),
        "donor_LEC_percent_among_EC_Q1": float(ec_pct.quantile(0.25)),
        "donor_LEC_percent_among_EC_Q3": float(ec_pct.quantile(0.75)),
        "LEC_definition": "fresh EC-only CellTypist majority vote EC8_ln",
        "all_cell_denominator": "Step 03 NF QC-passed cells",
        "EC_denominator": "Step 06 selected EC-related source clusters after EC-only reclustering",
    }


def grouped_counts(
    all_obs: pd.DataFrame,
    ec_obs: pd.DataFrame,
    quartile: str,
    group_column: str,
) -> pd.DataFrame:
    all_meta = all_obs.copy()
    ec_meta = ec_obs.copy()
    if group_column not in all_meta.columns or group_column not in ec_meta.columns:
        return pd.DataFrame()
    if group_column == "age_numeric_resolved":
        all_meta["group_value"] = pd.to_numeric(all_meta[group_column], errors="coerce")
        ec_meta["group_value"] = pd.to_numeric(ec_meta[group_column], errors="coerce")
    else:
        all_meta["group_value"] = normalized_text(all_meta[group_column]).replace("", "unknown")
        ec_meta["group_value"] = normalized_text(ec_meta[group_column]).replace("", "unknown")
    all_meta["donor_key"] = donor_key_from_obs(all_meta)
    ec_meta["donor_key"] = donor_key_from_obs(ec_meta)
    lec_meta = ec_meta.loc[true_mask(ec_meta["selected_as_LEC"])]

    all_table = all_meta.groupby("group_value", dropna=False).agg(
        n_all_QC_cells=("donor_key", "size"), n_donors_all=("donor_key", "nunique")
    )
    ec_table = ec_meta.groupby("group_value", dropna=False).agg(
        n_EC_cells=("donor_key", "size"), n_donors_with_EC=("donor_key", "nunique")
    )
    lec_table = lec_meta.groupby("group_value", dropna=False).agg(
        n_LEC_cells=("donor_key", "size"), n_donors_with_LEC=("donor_key", "nunique")
    )
    table = pd.concat([all_table, ec_table, lec_table], axis=1).fillna(0).reset_index()
    count_columns = [column for column in table.columns if column.startswith("n_")]
    table[count_columns] = table[count_columns].astype(int)
    table["LEC_percent_among_all_QC"] = np.where(
        table["n_all_QC_cells"].gt(0), 100 * table["n_LEC_cells"] / table["n_all_QC_cells"], np.nan
    )
    table["LEC_percent_among_EC"] = np.where(
        table["n_EC_cells"].gt(0), 100 * table["n_LEC_cells"] / table["n_EC_cells"], np.nan
    )
    table.insert(0, "grouping_variable", group_column)
    table.insert(0, "age_quartile", quartile)
    table.insert(0, "disease", REQUIRED_DISEASE)
    table.insert(0, "dataset", DATASET)
    return table


def x_labels(summary: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for _, row in summary.iterrows():
        minimum = row["age_min_resolved"]
        maximum = row["age_max_resolved"]
        def fmt(value: float) -> str:
            return str(int(value)) if float(value).is_integer() else f"{value:g}"
        labels.append(
            f"{row['age_quartile']}\n{fmt(minimum)}–{fmt(maximum)} y\n"
            f"{int(row['n_donors_all'])} donors"
        )
    return labels


def save_bar_percentage(
    summary: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    path: Path,
    color: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(summary))
    values = summary[column].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(x, values, width=0.62, color=color)
    ax.set_xticks(x, x_labels(summary))
    ax.set_xlabel("NF donor-age quartile")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.55, len(summary) - 0.45)
    upper = max(values.max() * 1.25, 0.1)
    ax.set_ylim(0, upper)
    ax.grid(axis="y", alpha=0.2)
    labels = [f"{value:.3f}%" if value < 1 else f"{value:.2f}%" for value in values]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_count_panels(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(summary))
    specs = [
        ("n_all_QC_cells", "All QC-passed cells", "#4C78A8"),
        ("n_EC_cells", "Selected EC cells", "#2A9D8F"),
        ("n_LEC_cells", "Final LEC cells", "#E76F51"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, (column, title, color) in zip(axes, specs):
        values = summary[column].to_numpy(dtype=int)
        bars = ax.bar(x, values, width=0.62, color=color)
        ax.set_xticks(x, [row.split("\n")[0] for row in x_labels(summary)])
        ax.set_xlabel("Quartile")
        ax.set_ylabel("Number of cells")
        ax.set_title(title)
        ax.set_xlim(-0.55, len(summary) - 0.45)
        ax.set_ylim(0, max(values) * 1.22)
        ax.grid(axis="y", alpha=0.2)
        ax.bar_label(bars, labels=[f"{value:,}" for value in values], padding=3, fontsize=8)
    fig.suptitle("Cell counts at each stage (separate y-axis per panel)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_donor_percentage_figure(
    donor: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    specs = [
        ("LEC_percent_among_all_QC", "LEC among all QC cells per donor"),
        ("LEC_percent_among_EC", "LEC among EC cells per donor"),
    ]
    quartile_order = summary["age_quartile"].astype(str).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    for ax, (column, title) in zip(axes, specs):
        for index, quartile in enumerate(quartile_order):
            values = pd.to_numeric(
                donor.loc[donor["age_quartile"].eq(quartile), column], errors="coerce"
            ).dropna().to_numpy()
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=30,
                alpha=0.75,
                color="#264653",
                edgecolor="white",
                linewidth=0.35,
            )
            if len(values):
                q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
                ax.vlines(index, q1, q3, color="#E76F51", linewidth=5, zorder=4)
                ax.hlines(median, index - 0.18, index + 0.18, color="black", linewidth=2, zorder=5)
        ax.set_xticks(np.arange(len(summary)), x_labels(summary))
        ax.set_xlabel("NF donor-age quartile")
        ax.set_ylabel("LEC percentage per donor (%)")
        ax.set_title(title)
        ax.set_xlim(-0.55, len(summary) - 0.45)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Donor-level LEC abundance (points; median and IQR in black/orange)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_donor_coverage(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(summary))
    with_lec = summary["n_donors_with_LEC"].to_numpy(dtype=int)
    total = summary["n_donors_all"].to_numpy(dtype=int)
    without_lec = total - with_lec
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(x, with_lec, width=0.62, color="#2A9D8F", label="At least one LEC")
    ax.bar(x, without_lec, width=0.62, bottom=with_lec, color="#D9D9D9", label="No LEC detected")
    ax.set_xticks(x, x_labels(summary))
    ax.set_xlabel("NF donor-age quartile")
    ax.set_ylabel("Number of donors")
    ax.set_title("Donor coverage of final LEC calls")
    ax.set_xlim(-0.55, len(summary) - 0.45)
    ax.set_ylim(0, max(total) + 2)
    ax.legend(frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0))
    for i, (positive, denominator) in enumerate(zip(with_lec, total)):
        ax.text(i, denominator + 0.25, f"{positive}/{denominator}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    project_dir = find_project_dir()
    allcell_root = resolve_path(args.allcell_root, project_dir)
    ec_root = resolve_path(args.ec_root, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        import anndata as ad
        if not args.skip_figures:
            import matplotlib
            matplotlib.use("Agg", force=True)
    except Exception as error:
        print(f"ERROR: Required analysis packages are unavailable: {error}", file=sys.stderr)
        return 2

    config = {
        "script_version": SCRIPT_VERSION,
        "dataset": DATASET,
        "disease": REQUIRED_DISEASE,
        "quartiles": args.quartiles,
        "allcell_root": str(allcell_root),
        "ec_root": str(ec_root),
        "output_root": str(output_root),
        "LEC_definition": "fresh EC-only CellTypist majority vote EC8_ln",
        "all_cell_denominator": "Step 03 NF QC-passed cells",
        "EC_denominator": "Step 06 EC-reclustered cells",
        "figures_enabled": not args.skip_figures,
    }
    with (output_root / "07_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    summaries: list[dict[str, object]] = []
    donors: list[pd.DataFrame] = []
    lec_cells: list[pd.DataFrame] = []
    validations: list[dict[str, object]] = []
    sex_tables: list[pd.DataFrame] = []
    age_tables: list[pd.DataFrame] = []

    for quartile in args.quartiles:
        prefix = f"dataset13_{quartile}"
        all_path = allcell_root / quartile / f"{prefix}_clustered.h5ad"
        ec_path = ec_root / quartile / f"{prefix}_EC_reclustered.h5ad"
        if not all_path.exists():
            raise FileNotFoundError(f"Step 03 all-cell h5ad not found: {all_path}")
        if not ec_path.exists():
            raise FileNotFoundError(f"Step 06 EC h5ad not found: {ec_path}")

        all_adata = ad.read_h5ad(all_path, backed="r")
        ec_adata = ad.read_h5ad(ec_path, backed="r")
        try:
            all_obs = all_adata.obs.copy()
            ec_obs = ec_adata.obs.copy()
        finally:
            all_adata.file.close()
            ec_adata.file.close()

        validation = validate_group(all_obs, ec_obs, quartile, all_path, ec_path)
        donor = make_donor_table(all_obs, ec_obs, quartile)
        summary = summarize_quartile(all_obs, ec_obs, donor, quartile)
        validation["n_donors_all"] = int(donor["donor_key"].nunique())
        validation["n_donors_with_LEC"] = int(donor["has_at_least_one_LEC"].sum())
        validations.append(validation)
        donors.append(donor)
        summaries.append(summary)
        lec_cells.append(compact_cell_metadata(ec_obs, quartile))

        sex_column = "sex" if "sex" in all_obs.columns and "sex" in ec_obs.columns else "Sex"
        sex_table = grouped_counts(all_obs, ec_obs, quartile, sex_column)
        if not sex_table.empty:
            sex_tables.append(sex_table)
        age_table = grouped_counts(all_obs, ec_obs, quartile, "age_numeric_resolved")
        if not age_table.empty:
            age_tables.append(age_table)

    summary_table = pd.DataFrame(summaries)
    summary_table["_order"] = summary_table["age_quartile"].map(
        {quartile: index for index, quartile in enumerate(QUARTILES)}
    )
    summary_table = summary_table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    donor_table = pd.concat(donors, ignore_index=True)
    validation_table = pd.DataFrame(validations)
    lec_cell_table = pd.concat(lec_cells, ignore_index=True)

    summary_table.to_csv(
        output_root / "07_LEC_percentage_summary_by_quartile.csv", index=False
    )
    donor_table.to_csv(output_root / "07_LEC_percentage_by_donor.csv", index=False)
    validation_table.to_csv(output_root / "07_input_validation_summary.csv", index=False)
    lec_cell_table.to_csv(
        output_root / "07_final_LEC_cell_metadata.csv.gz",
        index=False,
        compression="gzip",
    )
    if sex_tables:
        pd.concat(sex_tables, ignore_index=True).to_csv(
            output_root / "07_LEC_counts_by_sex_and_quartile.csv", index=False
        )
    if age_tables:
        pd.concat(age_tables, ignore_index=True).to_csv(
            output_root / "07_LEC_counts_by_exact_age_and_quartile.csv", index=False
        )

    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        figure_dir = output_root / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        figure_specs = [
            (
                "07_LEC_percent_among_all_QC_by_quartile.png",
                "LEC percentage among all NF QC-passed cells",
            ),
            (
                "07_LEC_percent_among_EC_by_quartile.png",
                "LEC percentage among selected EC cells",
            ),
            ("07_cell_counts_all_EC_LEC_by_quartile.png", "All, EC, and LEC cell counts"),
            ("07_donor_level_LEC_percentages_by_quartile.png", "Donor-level LEC percentages"),
            ("07_LEC_donor_coverage_by_quartile.png", "Donor coverage of LEC calls"),
        ]
        save_bar_percentage(
            summary_table,
            "LEC_percent_among_all_QC_pooled",
            "Final LEC abundance among all NF QC-passed cells",
            "LEC among all QC cells (%)",
            figure_dir / figure_specs[0][0],
            "#E76F51",
            args.dpi,
        )
        save_bar_percentage(
            summary_table,
            "LEC_percent_among_EC_pooled",
            "Final LEC abundance among selected EC cells",
            "LEC among EC cells (%)",
            figure_dir / figure_specs[1][0],
            "#2A9D8F",
            args.dpi,
        )
        save_count_panels(summary_table, figure_dir / figure_specs[2][0], args.dpi)
        save_donor_percentage_figure(
            donor_table, summary_table, figure_dir / figure_specs[3][0], args.dpi
        )
        save_donor_coverage(summary_table, figure_dir / figure_specs[4][0], args.dpi)
        for filename, description in figure_specs:
            figure_manifest.append(
                {"figure": str(figure_dir / filename), "description": description}
            )
        pd.DataFrame(figure_manifest).to_csv(
            output_root / "07_figure_manifest.csv", index=False
        )

    print("Dataset13 NF aging Step 07 completed.")
    print(
        summary_table[
            [
                "age_quartile",
                "age_quartile_range_label",
                "n_donors_all",
                "n_donors_with_LEC",
                "n_all_QC_cells",
                "n_EC_cells",
                "n_LEC_cells",
                "LEC_percent_among_all_QC_pooled",
                "LEC_percent_among_EC_pooled",
            ]
        ].to_string(index=False)
    )
    print(f"Outputs: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
