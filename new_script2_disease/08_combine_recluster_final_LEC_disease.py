#!/usr/bin/env python
r"""
Step 08: combine and recluster final LECs from every active disease cohort.

This is the disease counterpart of aging Step 08. It merges all final LECs
selected in disease Step 06 across dataset01 DCM/HCM/NF, dataset08 DoxCM/NF,
and dataset11 HFpEF/NF. It does not merge only NF.

LEC identities come from Step 06 ``selected_as_LEC``. Full log-normalized,
unscaled expression is recovered from the matching Step 03 ``raw`` objects in
backed mode, so the large all-cell H5AD files are never loaded completely.
Cohorts are concatenated with an outer gene join. Scaling, HVG selection, PCA,
and Harmony are performed only in a temporary embedding object. Harmony uses
``dataset_folder`` as the technical batch key; condition and donor are not used
as Harmony covariates. Leiden states are named from ``state0`` upward.

Default parameters mirror the completed aging combined-LEC analysis:
2,000 HVGs, 30 PCs, 15 neighbors, Leiden resolution 0.6 (matching the executed run), and top 25 markers.
State markers are ranked with Wilcoxon on the retained unscaled log-normalized
expression and are descriptive cell-level markers, not donor-level tests.

Default inputs
--------------
new_result2_disease/03/03_master_run_summary.csv
new_result2_disease/06_EC_recluster/06_EC_recluster_run_summary.csv

Default outputs
---------------
new_result2_disease/08_combined_LEC_recluster/

Read-only validation
--------------------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\08_combine_recluster_final_LEC_disease.py `
      --validate_only

Full run
--------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\08_combine_recluster_final_LEC_disease.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# Recent Python/Windows installations may not provide the retired WMIC tool.
# Giving joblib an explicit logical-core fallback avoids a misleading traceback.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


SCRIPT_VERSION = "08_disease_all_conditions_dataset_harmony_res0p3_h5ad_fix_2026_08_11"
DEFAULT_STEP03_ROOT = "new_result2_disease/03"
DEFAULT_STEP06_ROOT = "new_result2_disease/06_EC_recluster"
DEFAULT_OUTPUT_ROOT = "new_result2_disease/08_combined_LEC_recluster"
DEFAULT_OUTPUT_H5AD = "08_combined_final_LEC_reclustered.h5ad"
DEFAULT_RESOLUTION = 0.6

EXPECTED_COHORTS = (
    ("dataset01", "DCM"),
    ("dataset01", "HCM"),
    ("dataset01", "NF"),
    ("dataset08", "DoxCM"),
    ("dataset08", "NF"),
    ("dataset11", "HFpEF"),
    ("dataset11", "NF"),
)
DATASET_ORDER = ["dataset01", "dataset08", "dataset11"]
CONDITION_ORDER = ["NF", "DCM", "HCM", "DoxCM", "HFpEF"]
DATASET_COLORS = {
    "dataset01": "#332288",
    "dataset08": "#117733",
    "dataset11": "#DDCC77",
}
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


def natural_key(value: object) -> tuple[tuple[int, object], ...]:
    pieces = re.split(r"(\d+(?:\.\d+)?)", str(value))
    out: list[tuple[int, object]] = []
    for piece in pieces:
        if not piece:
            continue
        try:
            out.append((0, float(piece)))
        except ValueError:
            out.append((1, piece.casefold()))
    return tuple(out)


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def close_backed(adata: ad.AnnData) -> None:
    try:
        adata.file.close()
    except Exception:
        pass


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in {source}: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine final disease LECs across every active condition, correct "
            "dataset-level batch differences with Harmony, cluster, and rank markers."
        )
    )
    parser.add_argument("--step03_root", default=DEFAULT_STEP03_ROOT)
    parser.add_argument("--step06_root", default=DEFAULT_STEP06_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--out_h5ad", default=DEFAULT_OUTPUT_H5AD)
    parser.add_argument("--n_top_hvg", type=int, default=2000)
    parser.add_argument("--n_pcs", type=int, default=30)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument(
        "--harmony_batch_key",
        default="dataset_folder",
        choices=["dataset_folder"],
        help="Fixed technical batch key. Disease condition is never used as batch.",
    )
    parser.add_argument("--no_harmony", action="store_true")
    parser.add_argument("--top_n_markers", type=int, default=25)
    parser.add_argument("--marker_method", default="wilcoxon", choices=["wilcoxon", "t-test", "t-test_overestim_var", "logreg"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_markers", action="store_true")
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_top_hvg < 50:
        raise ValueError("--n_top_hvg must be at least 50.")
    if args.n_pcs < 2 or args.n_neighbors < 2:
        raise ValueError("--n_pcs and --n_neighbors must be at least 2.")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive.")
    if args.top_n_markers < 1:
        raise ValueError("--top_n_markers must be positive.")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")


def read_pipeline_summaries(
    step03_root: Path, step06_root: Path
) -> pd.DataFrame:
    step03_path = step03_root / "03_master_run_summary.csv"
    step06_path = step06_root / "06_EC_recluster_run_summary.csv"
    if not step03_path.exists():
        raise FileNotFoundError(f"Missing Step 03 summary: {step03_path}")
    if not step06_path.exists():
        raise FileNotFoundError(f"Missing Step 06 summary: {step06_path}")
    step03 = pd.read_csv(step03_path)
    step06 = pd.read_csv(step06_path)
    require_columns(
        step03,
        ["dataset_folder", "condition", "status", "n_cells_after_all_QC", "output_h5ad"],
        step03_path,
    )
    require_columns(
        step06,
        ["dataset_folder", "condition", "status", "n_EC_cells", "n_LEC_cells", "output_h5ad"],
        step06_path,
    )
    for frame in [step03, step06]:
        frame["dataset_folder"] = normalized_text(frame["dataset_folder"])
        frame["condition"] = normalized_text(frame["condition"])
        if frame.duplicated(["dataset_folder", "condition"]).any():
            raise ValueError("A pipeline summary contains duplicate cohort rows.")
    expected = set(EXPECTED_COHORTS)
    for name, frame in [("Step 03", step03), ("Step 06", step06)]:
        observed = set(zip(frame["dataset_folder"], frame["condition"]))
        if observed != expected:
            raise ValueError(
                f"{name} cohort mismatch. Missing={sorted(expected-observed)}; "
                f"unexpected={sorted(observed-expected)}"
            )
        statuses = normalized_text(frame["status"]).str.casefold()
        valid = statuses.str.startswith("completed") | statuses.str.startswith("reused")
        if not valid.all():
            raise ValueError(
                f"{name} contains incomplete rows:\n"
                + frame.loc[~valid, ["dataset_folder", "condition", "status"]].to_string(index=False)
            )
    pipeline = step03.merge(
        step06,
        on=["dataset_folder", "condition"],
        suffixes=("_step03", "_step06"),
        validate="one_to_one",
    )
    order = {pair: index for index, pair in enumerate(EXPECTED_COHORTS)}
    pipeline["cohort_order"] = [
        order[(dataset, condition)]
        for dataset, condition in zip(pipeline["dataset_folder"], pipeline["condition"])
    ]
    pipeline["cohort"] = (
        pipeline["dataset_folder"].astype(str) + " | " + pipeline["condition"].astype(str)
    )
    return pipeline.sort_values("cohort_order").reset_index(drop=True)


def validate_cohort_sources(
    row: pd.Series, project_dir: Path
) -> dict[str, object]:
    dataset = str(row["dataset_folder"])
    condition = str(row["condition"])
    cohort = str(row["cohort"])
    all_path = resolve_path(project_dir, str(row["output_h5ad_step03"]))
    ec_path = resolve_path(project_dir, str(row["output_h5ad_step06"]))
    if not all_path.exists() or not ec_path.exists():
        raise FileNotFoundError(f"Missing Step 03 or Step 06 H5AD for {cohort}.")

    all_data = ad.read_h5ad(all_path, backed="r")
    ec_data = ad.read_h5ad(ec_path, backed="r")
    try:
        if all_data.n_obs != int(row["n_cells_after_all_QC"]):
            raise ValueError(f"{cohort}: Step 03 cell count does not match its summary.")
        if ec_data.n_obs != int(row["n_EC_cells"]):
            raise ValueError(f"{cohort}: Step 06 EC count does not match its summary.")
        require_columns(
            ec_data.obs,
            ["selected_as_LEC", "donor_key", "dataset_folder", "condition"],
            ec_path,
        )
        lec_mask = true_mask(ec_data.obs["selected_as_LEC"])
        lec_ids = pd.Index(ec_data.obs_names[lec_mask].astype(str))
        if len(lec_ids) != int(row["n_LEC_cells"]):
            raise ValueError(f"{cohort}: Step 06 LEC count does not match its summary.")
        if len(lec_ids) == 0:
            raise ValueError(f"{cohort}: no final LECs are available for combination.")
        missing = lec_ids.difference(pd.Index(all_data.obs_names.astype(str)))
        if len(missing):
            raise ValueError(f"{cohort}: {len(missing)} Step 06 LEC IDs are absent from Step 03.")
        if all_data.raw is None:
            raise ValueError(f"{cohort}: Step 03 raw expression is absent.")
        datasets = set(normalized_text(ec_data.obs["dataset_folder"]).unique())
        conditions = set(normalized_text(ec_data.obs["condition"]).unique())
        if datasets != {dataset} or conditions != {condition}:
            raise ValueError(
                f"{cohort}: metadata mismatch: datasets={datasets}, conditions={conditions}."
            )
        labels = set()
        if "celltypist_majority_vote" in ec_data.obs.columns:
            labels = set(
                normalized_text(ec_data.obs.loc[lec_mask, "celltypist_majority_vote"]).unique()
            )
            if labels != {"EC8_ln"}:
                raise ValueError(f"{cohort}: selected LEC labels are {sorted(labels)}, not EC8_ln only.")
        return {
            "dataset_folder": dataset,
            "condition": condition,
            "cohort": cohort,
            "n_step03_cells": int(all_data.n_obs),
            "n_step06_EC_cells": int(ec_data.n_obs),
            "n_final_LEC_cells": int(len(lec_ids)),
            "n_final_LEC_donors": int(normalized_text(ec_data.obs.loc[lec_mask, "donor_key"]).nunique()),
            "n_genes_step03_raw": int(all_data.raw.n_vars),
            "LEC_majority_labels": ";".join(sorted(labels)),
            "step03_h5ad": str(all_path),
            "step06_h5ad": str(ec_path),
            "status": "validated",
        }
    finally:
        close_backed(all_data)
        close_backed(ec_data)


def load_cohort_lecs(
    row: pd.Series, project_dir: Path
) -> tuple[ad.AnnData, dict[str, object], set[str]]:
    dataset = str(row["dataset_folder"])
    condition = str(row["condition"])
    cohort = str(row["cohort"])
    all_path = resolve_path(project_dir, str(row["output_h5ad_step03"]))
    ec_path = resolve_path(project_dir, str(row["output_h5ad_step06"]))

    print(f"[{cohort}] reading Step 06 LEC identities")
    ec_data = ad.read_h5ad(ec_path, backed="r")
    try:
        lec_mask = true_mask(ec_data.obs["selected_as_LEC"])
        lec_meta = ec_data.obs.loc[lec_mask].copy()
    finally:
        close_backed(ec_data)
    if lec_meta.empty:
        raise ValueError(f"{cohort}: no final Step 06 LEC cells.")

    print(f"[{cohort}] backed-loading only matching Step 03 rows")
    all_data = ad.read_h5ad(all_path, backed="r")
    try:
        selected_ids = pd.Index(lec_meta.index.astype(str))
        positions = pd.Index(all_data.obs_names.astype(str)).get_indexer(selected_ids)
        if (positions < 0).any():
            raise ValueError(f"{cohort}: some Step 06 LEC IDs are absent from Step 03.")
        sort_order = np.argsort(positions)
        positions = positions[sort_order]
        selected_ids = selected_ids[sort_order]
        lec_meta = lec_meta.reindex(selected_ids)
        selected = all_data[positions].to_memory()
    finally:
        close_backed(all_data)
    if list(selected.obs_names.astype(str)) != list(selected_ids.astype(str)):
        raise ValueError(f"{cohort}: backed Step 03 row order is inconsistent.")
    if selected.raw is None:
        raise ValueError(f"{cohort}: Step 03 raw is absent.")

    expression = selected.raw.to_adata()
    expression.obs = selected.obs.copy()
    del selected
    gc.collect()

    carry_columns = [
        "allcell_leiden",
        "step05_celltypist_raw_label",
        "step05_celltypist_majority_vote",
        "celltypist_raw_label",
        "celltypist_majority_vote",
        "celltypist_max_probability",
        "celltypist_probability_best_label",
        "selected_as_LEC",
        "final_LEC_by_majority_vote",
        "ec_recluster_id",
    ]
    for column in carry_columns:
        if column in lec_meta.columns:
            expression.obs[column] = lec_meta[column].to_numpy()
    if "leiden" in lec_meta.columns:
        expression.obs["step06_ec_leiden"] = normalized_text(lec_meta["leiden"]).to_numpy()

    expression.var_names = expression.var_names.astype(str)
    expression.var_names_make_unique()
    original_ids = expression.obs_names.astype(str)
    expression.obs["source_cell_id"] = original_ids
    expression.obs["source_step03_h5ad"] = str(all_path)
    expression.obs["source_step06_h5ad"] = str(ec_path)
    expression.obs["dataset_folder"] = dataset
    expression.obs["condition"] = condition
    expression.obs["cohort"] = cohort
    expression.obs["donor_key"] = normalized_text(expression.obs["donor_key"]).to_numpy()
    expression.obs_names = pd.Index(
        [f"{dataset}|{condition}|{cell_id}" for cell_id in original_ids],
        dtype="object",
    )
    genes = set(expression.var_names.astype(str))
    summary = {
        "dataset_folder": dataset,
        "condition": condition,
        "cohort": cohort,
        "n_final_LEC_cells": int(expression.n_obs),
        "n_final_LEC_donors": int(expression.obs["donor_key"].nunique()),
        "n_genes_recovered": int(expression.n_vars),
        "expression_source": "Step03_raw_log_normalized",
        "step03_h5ad": str(all_path),
        "step06_h5ad": str(ec_path),
    }
    print(
        f"[{cohort}] retained {expression.n_obs:,} LECs from "
        f"{summary['n_final_LEC_donors']} donors and {expression.n_vars:,} genes"
    )
    return expression, summary, genes


def gene_presence_table(
    union_genes: Iterable[str], genes_by_cohort: dict[str, set[str]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cohort_names = [f"{dataset} | {condition}" for dataset, condition in EXPECTED_COHORTS]
    for gene in sorted(set(union_genes), key=natural_key):
        row: dict[str, object] = {"gene": gene}
        present = 0
        for cohort in cohort_names:
            column = "present_" + safe_name(cohort.replace(" | ", "_"))
            value = gene in genes_by_cohort.get(cohort, set())
            row[column] = value
            present += int(value)
        row["n_cohorts_present"] = present
        row["present_all_cohorts"] = present == len(cohort_names)
        rows.append(row)
    return pd.DataFrame(rows)


def state_palette(n_states: int) -> list[str]:
    colors: list[tuple[float, float, float]] = []
    for name in ["tab20", "tab20b", "tab20c"]:
        colors.extend(list(matplotlib.colormaps[name].colors))
    return [matplotlib.colors.to_hex(colors[index % len(colors)]) for index in range(n_states)]


def transfer_embedding(combined: ad.AnnData, work: ad.AnnData) -> None:
    for key in ["X_pca", "X_pca_harmony", "X_umap"]:
        if key in work.obsm:
            combined.obsm[key] = work.obsm[key].copy()
    for key in ["connectivities", "distances"]:
        if key in work.obsp:
            combined.obsp[key] = work.obsp[key].copy()
    for key in ["pca", "neighbors", "umap", "combined_leiden"]:
        if key in work.uns:
            combined.uns[key] = work.uns[key]


def compute_embedding_and_states(
    combined: ad.AnnData, args: argparse.Namespace
) -> tuple[ad.AnnData, dict[str, object], pd.DataFrame]:
    import scanpy as sc

    work = combined.copy()
    n_top = min(int(args.n_top_hvg), work.n_vars)
    sc.pp.highly_variable_genes(
        work,
        n_top_genes=n_top,
        flavor="seurat",
        batch_key="dataset_folder",
        subset=False,
    )
    hvg = work.var["highly_variable"].fillna(False).to_numpy(dtype=bool)
    if int(hvg.sum()) < 50:
        raise ValueError(f"Only {int(hvg.sum())} combined HVGs were selected.")
    combined.var["highly_variable_for_combined_embedding"] = hvg
    work = work[:, hvg].copy()
    sc.pp.scale(work, max_value=10)
    n_pcs = min(int(args.n_pcs), work.n_obs - 1, work.n_vars - 1)
    if n_pcs < 2:
        raise ValueError("Too few dimensions for PCA.")
    sc.tl.pca(work, n_comps=n_pcs, svd_solver="arpack", random_state=args.random_state)

    n_neighbors = min(int(args.n_neighbors), work.n_obs - 1)
    uncorrected = work.copy()
    sc.pp.neighbors(
        uncorrected,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep="X_pca",
        random_state=args.random_state,
    )
    sc.tl.umap(uncorrected, random_state=args.random_state)
    combined.obsm["X_umap_uncorrected"] = uncorrected.obsm["X_umap"].copy()
    del uncorrected
    gc.collect()

    use_rep = "X_pca"
    harmony_status = "disabled"
    if not args.no_harmony:
        batch = normalized_text(work.obs[args.harmony_batch_key]).replace("", "unknown")
        work.obs[args.harmony_batch_key] = pd.Categorical(
            batch.astype(str), categories=DATASET_ORDER, ordered=True
        )
        if work.obs[args.harmony_batch_key].nunique() < 2:
            harmony_status = "skipped_only_one_dataset"
        else:
            try:
                import scanpy.external as sce

                sce.pp.harmony_integrate(
                    work,
                    key=args.harmony_batch_key,
                    basis="X_pca",
                    adjusted_basis="X_pca_harmony",
                )
                use_rep = "X_pca_harmony"
                harmony_status = "ok"
            except Exception as error:
                harmony_status = f"failed_{type(error).__name__}: {str(error)[:300]}"
                warnings.warn(f"Harmony failed; ordinary PCA will be used: {error}")

    sc.pp.neighbors(
        work,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        random_state=args.random_state,
    )
    sc.tl.umap(work, random_state=args.random_state)
    sc.tl.leiden(
        work,
        resolution=float(args.resolution),
        key_added="combined_leiden",
        random_state=args.random_state,
    )
    transfer_embedding(combined, work)
    combined.obs["combined_leiden"] = normalized_text(work.obs["combined_leiden"]).to_numpy()
    clusters = sorted(combined.obs["combined_leiden"].unique(), key=natural_key)
    colors = state_palette(len(clusters))
    mapping = pd.DataFrame(
        {
            "combined_leiden": clusters,
            "LEC_state": [f"state{index}" for index in range(len(clusters))],
            "color_hex": colors,
        }
    )
    cluster_to_state = dict(zip(mapping["combined_leiden"], mapping["LEC_state"]))
    state_order = mapping["LEC_state"].tolist()
    combined.obs["combined_leiden"] = pd.Categorical(
        combined.obs["combined_leiden"], categories=clusters, ordered=True
    )
    combined.obs["LEC_state"] = pd.Categorical(
        combined.obs["combined_leiden"].astype(str).map(cluster_to_state),
        categories=state_order,
        ordered=True,
    )
    combined.uns["LEC_state_colors"] = colors
    combined.uns["combined_leiden_colors"] = colors
    combined.uns["LEC_state_color_map"] = dict(zip(state_order, colors))
    combined.uns["dataset_folder_colors"] = [DATASET_COLORS[name] for name in DATASET_ORDER]
    combined.uns["condition_colors"] = [CONDITION_COLORS[name] for name in CONDITION_ORDER]
    params = {
        "method": "Harmony_on_PCA" if use_rep == "X_pca_harmony" else "PCA",
        "harmony_batch_key": args.harmony_batch_key,
        "harmony_status": harmony_status,
        "neighbors_representation": use_rep,
        "n_hvg_used": int(hvg.sum()),
        "n_pcs_used": int(n_pcs),
        "n_neighbors_used": int(n_neighbors),
        "leiden_resolution": float(args.resolution),
        "random_state": int(args.random_state),
        "state_name_start": 0,
    }
    combined.uns["combined_LEC_embedding"] = params
    return combined, params, mapping


def composition_table(
    obs: pd.DataFrame, group_column: str, group_order: list[str]
) -> pd.DataFrame:
    state_order = list(obs["LEC_state"].cat.categories)
    frame = obs[[group_column, "LEC_state"]].copy()
    frame[group_column] = pd.Categorical(
        normalized_text(frame[group_column]), categories=group_order, ordered=True
    )
    table = (
        frame.groupby([group_column, "LEC_state"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = table.groupby(group_column, observed=True)["n_cells"].transform("sum")
    table["fraction_within_group"] = np.where(totals.gt(0), table["n_cells"] / totals, np.nan)
    table["percent_within_group"] = 100.0 * table["fraction_within_group"]
    table["LEC_state"] = pd.Categorical(
        table["LEC_state"], categories=state_order, ordered=True
    )
    return table


def build_summary_tables(combined: ad.AnnData) -> dict[str, pd.DataFrame]:
    obs = combined.obs.copy()
    state_order = list(obs["LEC_state"].cat.categories)
    counts = (
        obs.groupby("LEC_state", observed=True)
        .size()
        .reindex(state_order, fill_value=0)
        .rename("n_cells")
        .reset_index()
    )
    counts["fraction_of_all_LECs"] = counts["n_cells"] / len(obs)
    counts["percent_of_all_LECs"] = 100.0 * counts["fraction_of_all_LECs"]

    donor_state = (
        obs.groupby(["dataset_folder", "condition", "donor_key", "LEC_state"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    donor_totals = (
        obs.groupby(["dataset_folder", "condition", "donor_key"], observed=True)
        .agg(n_LEC_cells=("LEC_state", "size"), n_LEC_states=("LEC_state", "nunique"))
        .reset_index()
    )
    donor_state = donor_state.merge(
        donor_totals,
        on=["dataset_folder", "condition", "donor_key"],
        how="left",
    )
    donor_state["fraction_of_donor_LECs"] = donor_state["n_cells"] / donor_state["n_LEC_cells"]

    qc_rows: list[dict[str, object]] = []
    for state in state_order:
        group = obs.loc[obs["LEC_state"].astype(str).eq(state)]
        donor_vc = normalized_text(group["donor_key"]).value_counts()
        dataset_vc = normalized_text(group["dataset_folder"]).value_counts()
        condition_vc = normalized_text(group["condition"]).value_counts()
        n_cells = len(group)
        qc_rows.append(
            {
                "LEC_state": state,
                "combined_leiden": str(group["combined_leiden"].iloc[0]),
                "n_cells": int(n_cells),
                "n_donors": int(group["donor_key"].nunique()),
                "n_datasets": int(group["dataset_folder"].nunique()),
                "n_conditions": int(group["condition"].nunique()),
                "top_donor": donor_vc.index[0] if len(donor_vc) else "",
                "top_donor_n_cells": int(donor_vc.iloc[0]) if len(donor_vc) else 0,
                "top_donor_fraction": float(donor_vc.iloc[0] / n_cells) if n_cells and len(donor_vc) else np.nan,
                "donor_dominance_flag_over_50pct": bool(donor_vc.iloc[0] / n_cells > 0.5) if n_cells and len(donor_vc) else False,
                "top_dataset": dataset_vc.index[0] if len(dataset_vc) else "",
                "top_dataset_fraction": float(dataset_vc.iloc[0] / n_cells) if n_cells and len(dataset_vc) else np.nan,
                "dataset_dominance_flag_over_80pct": bool(dataset_vc.iloc[0] / n_cells > 0.8) if n_cells and len(dataset_vc) else False,
                "top_condition": condition_vc.index[0] if len(condition_vc) else "",
                "top_condition_fraction": float(condition_vc.iloc[0] / n_cells) if n_cells and len(condition_vc) else np.nan,
            }
        )
    return {
        "state_counts": counts,
        "state_by_condition": composition_table(obs, "condition", CONDITION_ORDER),
        "state_by_dataset": composition_table(obs, "dataset_folder", DATASET_ORDER),
        "donor_state_counts": donor_state,
        "donor_totals": donor_totals,
        "state_qc": pd.DataFrame(qc_rows),
    }


def rank_state_markers(
    combined: ad.AnnData,
    top_n: int,
    method: str,
    gene_presence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import scanpy as sc

    state_order = list(combined.obs["LEC_state"].cat.categories)
    sc.tl.rank_genes_groups(
        combined,
        groupby="LEC_state",
        method=method,
        use_raw=True,
        pts=True,
        key_added="rank_genes_groups_LEC_state",
    )
    frames: list[pd.DataFrame] = []
    for state in state_order:
        frame = sc.get.rank_genes_groups_df(
            combined, group=state, key="rank_genes_groups_LEC_state"
        )
        frame.insert(0, "rank", np.arange(1, len(frame) + 1))
        frame.insert(0, "LEC_state", state)
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True)
    full = full.merge(gene_presence.rename(columns={"gene": "names"}), on="names", how="left")
    top = full.groupby("LEC_state", sort=False, observed=True).head(top_n).copy()
    wide = top.pivot(index="LEC_state", columns="rank", values="names").reindex(state_order).reset_index()
    wide.columns = [
        "LEC_state" if column == "LEC_state" else f"marker_{int(column)}"
        for column in wide.columns
    ]
    return full, top, wide


def save_umap_summary(combined: ad.AnnData, path: Path, dpi: int, resolution: float) -> None:
    import scanpy as sc

    specs = [
        ("LEC_state", "Combined LEC state"),
        ("dataset_folder", "Dataset"),
        ("condition", "Condition"),
        ("sex", "Sex"),
        ("region_standardized_analysis", "Heart region"),
        ("combined_leiden", f"Combined Leiden (resolution {resolution:g})"),
    ]
    specs = [(key, title) for key, title in specs if key in combined.obs.columns]
    ncols = 3
    nrows = math.ceil(len(specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 5.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, title) in zip(axes, specs):
        legend = "on data" if key in {"LEC_state", "combined_leiden"} else "right margin"
        sc.pl.umap(
            combined,
            color=key,
            ax=ax,
            show=False,
            frameon=False,
            size=18,
            alpha=0.82,
            title=title,
            legend_loc=legend,
        )
    for ax in axes[len(specs):]:
        ax.axis("off")
    fig.suptitle("All active disease LECs combined across datasets and conditions", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_batch_comparison(combined: ad.AnnData, path: Path, dpi: int) -> None:
    import scanpy as sc

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    sc.pl.embedding(
        combined,
        basis="umap_uncorrected",
        color="dataset_folder",
        ax=axes[0],
        show=False,
        frameon=False,
        size=18,
        title="Before Harmony: dataset",
    )
    sc.pl.umap(
        combined,
        color="dataset_folder",
        ax=axes[1],
        show=False,
        frameon=False,
        size=18,
        title="After dataset-key Harmony",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_composition(
    table: pd.DataFrame,
    group_column: str,
    group_order: list[str],
    state_colors: dict[str, str],
    title: str,
    path: Path,
    dpi: int,
) -> None:
    pivot = table.pivot(
        index=group_column, columns="LEC_state", values="percent_within_group"
    ).fillna(0)
    pivot = pivot.reindex(group_order).dropna(how="all")
    pivot = pivot.reindex(sorted(pivot.columns, key=natural_key), axis=1)
    colors = [state_colors[str(state)] for state in pivot.columns]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    pivot.plot(kind="bar", stacked=True, color=colors, width=0.78, ax=ax, edgecolor="white", linewidth=0.3)
    ax.set_xlabel(group_column.replace("_", " ").title())
    ax.set_ylabel("Cells within group (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(title="LEC state", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_state_support(
    state_qc: pd.DataFrame, state_colors: dict[str, str], path: Path, dpi: int
) -> None:
    states = state_qc["LEC_state"].astype(str).tolist()
    colors = [state_colors[state] for state in states]
    x = np.arange(len(states))
    specs = [
        ("n_cells", "Cells", None),
        ("n_donors", "Donors", None),
        ("n_datasets", "Datasets", None),
        ("top_donor_fraction", "Largest donor contribution (%)", 100.0),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    for ax, (column, title, multiplier) in zip(axes, specs):
        values = pd.to_numeric(state_qc[column], errors="coerce").fillna(0).to_numpy(float)
        if multiplier:
            values *= multiplier
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        if column == "top_donor_fraction":
            ax.axhline(50, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x, states, rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Combined disease LEC-state support and dominance QC", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_marker_dotplot(combined: ad.AnnData, top_markers: pd.DataFrame, path: Path) -> None:
    import scanpy as sc

    marker_dict = {
        str(state): group["names"].astype(str).head(3).tolist()
        for state, group in top_markers.groupby("LEC_state", sort=False, observed=True)
    }
    if not marker_dict:
        return
    plot = sc.pl.dotplot(
        combined,
        var_names=marker_dict,
        groupby="LEC_state",
        use_raw=True,
        standard_scale="var",
        show=False,
        return_fig=True,
        title="Top combined disease LEC-state markers",
    )
    plot.savefig(path)
    plt.close("all")


def make_obs_h5ad_safe(adata: ad.AnnData) -> None:
    for column in adata.obs.columns:
        values = adata.obs[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            continue
        if not (
            pd.api.types.is_object_dtype(values.dtype)
            or pd.api.types.is_string_dtype(values.dtype)
        ):
            continue
        non_null = values.dropna()
        if non_null.empty:
            adata.obs[column] = values.fillna("").astype(str).astype(object)
            continue
        if non_null.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            adata.obs[column] = values.map(
                lambda value: False if pd.isna(value) else bool(value)
            ).astype(bool)
            continue
        age_like = str(column).casefold() in {
            "age", "age_numeric", "age_numeric_resolved", "donor_age"
        }
        if age_like:
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.notna().sum() == values.notna().sum():
                adata.obs[column] = numeric.astype(float)
                continue
        adata.obs[column] = values.map(
            lambda value: "" if value is None or (np.isscalar(value) and pd.isna(value)) else str(value)
        ).astype(object)


def make_var_h5ad_safe(adata: ad.AnnData) -> None:
    """Normalize concat-created mixed var dtypes before copying into raw."""
    for column in adata.var.columns:
        values = adata.var[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_string_dtype(values.dtype) and not pd.api.types.is_object_dtype(values.dtype):
            adata.var[column] = values.fillna("").astype(str)
            continue
        if not pd.api.types.is_object_dtype(values.dtype):
            continue
        non_null = values.dropna()
        if non_null.empty:
            adata.var[column] = values.fillna("").astype(str)
            continue
        if non_null.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            adata.var[column] = values.map(
                lambda value: False if pd.isna(value) else bool(value)
            ).astype(bool)
            continue
        if non_null.map(
            lambda value: isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
        ).all():
            adata.var[column] = pd.to_numeric(values, errors="coerce")
            continue
        adata.var[column] = values.map(
            lambda value: "" if value is None or (np.isscalar(value) and pd.isna(value)) else str(value)
        ).astype(str)


def write_h5ad_atomic(adata: ad.AnnData, output_h5ad: Path) -> None:
    temporary = output_h5ad.with_name(
        f".{output_h5ad.stem}.partial.{os.getpid()}{output_h5ad.suffix}"
    )
    try:
        adata.write_h5ad(temporary, compression="gzip")
        os.replace(temporary, output_h5ad)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def write_notes(
    path: Path,
    args: argparse.Namespace,
    run_summary: dict[str, object],
    state_qc: pd.DataFrame,
) -> None:
    dominant_donor = state_qc.loc[state_qc["donor_dominance_flag_over_50pct"]]
    dominant_dataset = state_qc.loc[state_qc["dataset_dominance_flag_over_80pct"]]
    text = f"""Disease Step 08 combined LEC-state method and interpretation notes
================================================================

Merge scope
-----------
All final Step 06 LECs from all seven active dataset-condition cohorts were
combined. The merge is not restricted to NF.

Expression and clustering
-------------------------
Expression source: Step 03 raw (log-normalized, unscaled, all genes).
Gene join: outer union across cohort inputs.
HVGs used only for embedding: {run_summary['n_hvg_used_for_embedding']}
Harmony batch key: {run_summary['harmony_batch_key']}
Harmony status: {run_summary['harmony_status']}
Leiden resolution: {args.resolution:g}
State labels: {run_summary['state_labels']}

Interpretation safeguards
-------------------------
Harmony uses dataset_folder, not condition or donor. Dataset and disease are
partly confounded because several diseases occur in only one dataset; Harmony
cannot prove that a remaining difference is biological or that a removed
difference was technical. Inspect the before/after Harmony plot and the state
dataset-support table. States dominated by one donor or one dataset require
sensitivity analysis before biological interpretation.

Marker rankings use cells as observations and are descriptive annotations.
Disease comparisons require donors as biological replicates, preferably
state-specific pseudobulk differential expression/GSEA within datasets.

QC flags
--------
Donor-dominated states (>50%): {len(dominant_donor)}
Dataset-dominated states (>80%): {len(dominant_dataset)}
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)
    project_dir = find_project_dir()
    step03_root = resolve_path(project_dir, args.step03_root)
    step06_root = resolve_path(project_dir, args.step06_root)
    output_root = resolve_path(project_dir, args.output_root)
    output_h5ad = output_root / args.out_h5ad
    pipeline = read_pipeline_summaries(step03_root, step06_root)

    print("=" * 100)
    print("Disease Step 08: combine all final LECs, dataset Harmony, states, and markers")
    print("Merge scope: ALL active dataset-condition cohorts (not NF only)")
    print("Cohorts:", ", ".join(pipeline["cohort"]))
    print("Harmony batch key: dataset_folder")
    print(f"Leiden resolution: {args.resolution:g}; state names start at state0")
    print("Gene policy: Step 03 raw expression with OUTER union")
    print("=" * 100)

    validation_rows = [
        validate_cohort_sources(row, project_dir)
        for _, row in pipeline.iterrows()
    ]
    validation = pd.DataFrame(validation_rows)
    print(
        validation[[
            "dataset_folder", "condition", "n_step03_cells", "n_step06_EC_cells",
            "n_final_LEC_cells", "n_final_LEC_donors", "n_genes_step03_raw",
            "LEC_majority_labels",
        ]].to_string(index=False)
    )
    print(
        f"Validated {len(validation)} cohorts, {validation['n_final_LEC_cells'].sum():,} "
        f"final LECs, and {validation['n_final_LEC_donors'].sum():,} cohort-level donor counts."
    )
    if args.validate_only:
        print("Validation complete; no Step 08 outputs were created.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    run_summary_path = output_root / "08_run_summary.csv"
    if output_h5ad.exists() and not args.overwrite:
        if run_summary_path.exists():
            raise FileExistsError(
                f"Completed output already exists: {output_h5ad}. Use --overwrite to replace it."
            )
        warnings.warn(
            "An H5AD exists without 08_run_summary.csv, indicating an incomplete "
            "prior run. It will be replaced only after a new temporary H5AD is "
            "written successfully."
        )
    figure_dir = output_root / "figures"
    if not args.skip_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    subsets: list[ad.AnnData] = []
    input_summaries: list[dict[str, object]] = []
    genes_by_cohort: dict[str, set[str]] = {}
    for _, row in pipeline.iterrows():
        subset, summary, genes = load_cohort_lecs(row, project_dir)
        subsets.append(subset)
        input_summaries.append(summary)
        genes_by_cohort[str(row["cohort"])] = genes

    print(f"Combining {len(subsets)} cohort subsets with an outer gene join")
    combined = ad.concat(
        subsets,
        axis=0,
        join="outer",
        merge="first",
        uns_merge=None,
        label=None,
        index_unique=None,
        fill_value=0,
    )
    del subsets
    gc.collect()
    if not combined.obs_names.is_unique:
        raise ValueError("Combined cell identifiers are not unique.")
    if set(normalized_text(combined.obs["condition"]).unique()) != set(CONDITION_ORDER):
        raise ValueError("The combined object does not contain every expected condition.")
    if set(normalized_text(combined.obs["dataset_folder"]).unique()) != set(DATASET_ORDER):
        raise ValueError("The combined object does not contain every expected dataset.")

    combined.obs["dataset_folder"] = pd.Categorical(
        normalized_text(combined.obs["dataset_folder"]), categories=DATASET_ORDER, ordered=True
    )
    combined.obs["condition"] = pd.Categorical(
        normalized_text(combined.obs["condition"]), categories=CONDITION_ORDER, ordered=True
    )
    gene_presence = gene_presence_table(combined.var_names.astype(str), genes_by_cohort)
    coverage = gene_presence.set_index("gene").reindex(combined.var_names.astype(str))
    for column in coverage.columns:
        combined.var[column] = coverage[column].to_numpy()
    combined.var["gene_symbol"] = combined.var_names.astype(str)
    combined.var["outer_union_retained"] = True
    make_var_h5ad_safe(combined)
    combined.raw = combined.copy()

    combined, embedding, state_mapping = compute_embedding_and_states(combined, args)
    state_colors = dict(zip(state_mapping["LEC_state"], state_mapping["color_hex"]))
    tables = build_summary_tables(combined)

    marker_status = "skipped"
    marker_message = ""
    full_markers = pd.DataFrame()
    top_markers = pd.DataFrame()
    wide_markers = pd.DataFrame()
    if not args.skip_markers:
        try:
            print("Ranking combined-state marker genes")
            full_markers, top_markers, wide_markers = rank_state_markers(
                combined, args.top_n_markers, args.marker_method, gene_presence
            )
            marker_status = "completed"
        except Exception as error:
            marker_status = "failed"
            marker_message = f"{type(error).__name__}: {error}"
            warnings.warn(f"Marker ranking failed: {error}")

    validation.to_csv(output_root / "08_input_validation.csv", index=False)
    pd.DataFrame(input_summaries).to_csv(output_root / "08_cohort_input_summary.csv", index=False)
    gene_presence.to_csv(output_root / "08_gene_presence_by_cohort.csv", index=False)
    state_mapping.to_csv(output_root / "08_state_name_mapping.csv", index=False)
    state_mapping[["LEC_state", "color_hex"]].to_csv(output_root / "08_state_color_mapping.csv", index=False)
    pd.DataFrame(
        [{"category_type": "condition", "name": name, "color_hex": color} for name, color in CONDITION_COLORS.items()]
        + [{"category_type": "dataset", "name": name, "color_hex": color} for name, color in DATASET_COLORS.items()]
        + [{"category_type": "LEC_state", "name": state, "color_hex": color} for state, color in state_colors.items()]
    ).to_csv(output_root / "08_color_key.csv", index=False)
    tables["state_counts"].to_csv(output_root / "08_state_counts.csv", index=False)
    tables["state_by_condition"].to_csv(output_root / "08_state_by_condition.csv", index=False)
    tables["state_by_dataset"].to_csv(output_root / "08_state_by_dataset.csv", index=False)
    tables["donor_state_counts"].to_csv(output_root / "08_state_counts_by_donor.csv", index=False)
    tables["donor_totals"].to_csv(output_root / "08_donor_LEC_counts.csv", index=False)
    tables["state_qc"].to_csv(output_root / "08_state_dataset_donor_QC.csv", index=False)
    if marker_status == "completed":
        full_markers.to_csv(output_root / "08_state_marker_ranking_all_genes.csv.gz", index=False, compression="gzip")
        top_markers.to_csv(output_root / f"08_state_top{args.top_n_markers}_markers_long.csv", index=False)
        wide_markers.to_csv(output_root / f"08_state_top{args.top_n_markers}_markers_wide.csv", index=False)

    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        jobs = [
            (
                "08_combined_LEC_umap_summary.png",
                "UMAP by state, dataset, condition, sex, region, and Leiden",
                lambda path: save_umap_summary(combined, path, args.dpi, args.resolution),
            ),
            (
                "08_dataset_batch_correction_comparison.png",
                "Uncorrected versus dataset-Harmony UMAP",
                lambda path: save_batch_comparison(combined, path, args.dpi),
            ),
            (
                "08_LEC_state_composition_by_condition.png",
                "LEC-state composition within condition",
                lambda path: save_composition(
                    tables["state_by_condition"], "condition", CONDITION_ORDER,
                    state_colors, "Combined LEC-state composition by condition", path, args.dpi,
                ),
            ),
            (
                "08_LEC_state_composition_by_dataset.png",
                "LEC-state composition within dataset",
                lambda path: save_composition(
                    tables["state_by_dataset"], "dataset_folder", DATASET_ORDER,
                    state_colors, "Combined LEC-state composition by dataset", path, args.dpi,
                ),
            ),
            (
                "08_LEC_state_support_QC.png",
                "State cells, donors, datasets, and largest-donor contribution",
                lambda path: save_state_support(tables["state_qc"], state_colors, path, args.dpi),
            ),
        ]
        if marker_status == "completed" and not top_markers.empty:
            jobs.append(
                (
                    "08_combined_LEC_top_state_marker_dotplot.png",
                    "Top three marker genes per combined LEC state",
                    lambda path: save_marker_dotplot(combined, top_markers, path),
                )
            )
        for filename, description, function in jobs:
            path = figure_dir / filename
            try:
                function(path)
                figure_manifest.append({"figure": str(path), "description": description, "status": "completed"})
            except Exception as error:
                warnings.warn(f"Figure failed ({filename}): {error}")
                figure_manifest.append({"figure": str(path), "description": description, "status": f"failed: {type(error).__name__}: {error}"})
        pd.DataFrame(figure_manifest).to_csv(output_root / "08_figure_manifest.csv", index=False)

    combined.uns["step08_combine_info"] = {
        "script_version": SCRIPT_VERSION,
        "merge_scope": "all_final_LECs_all_active_disease_conditions",
        "cohorts": pipeline["cohort"].tolist(),
        "LEC_definition": "Step06 fresh EC-only CellTypist majority vote EC8_ln",
        "expression_source": "Step03_raw_log_normalized",
        "gene_join": "outer",
        "harmony_batch_key": "dataset_folder",
        "state_name_start": 0,
        "leiden_resolution": float(args.resolution),
        "n_cells": int(combined.n_obs),
        "n_genes": int(combined.n_vars),
        "n_states": int(combined.obs["LEC_state"].nunique()),
    }
    make_obs_h5ad_safe(combined)
    combined.obs.reset_index(names="combined_cell_id").to_csv(
        output_root / "08_combined_LEC_cell_metadata.csv.gz", index=False, compression="gzip"
    )
    print(f"Writing combined H5AD: {output_h5ad}")
    write_h5ad_atomic(combined, output_h5ad)

    state_qc = tables["state_qc"]
    run_summary = {
        "script_version": SCRIPT_VERSION,
        "status": "completed",
        "merge_scope": "all_final_LECs_all_active_disease_conditions",
        "n_cohorts": int(len(pipeline)),
        "n_datasets": int(combined.obs["dataset_folder"].nunique()),
        "n_conditions": int(combined.obs["condition"].nunique()),
        "n_final_LEC_cells": int(combined.n_obs),
        "n_final_LEC_donors": int(combined.obs["donor_key"].nunique()),
        "n_genes_outer_union": int(combined.n_vars),
        "n_genes_present_all_cohorts": int(gene_presence["present_all_cohorts"].sum()),
        "n_states": int(combined.obs["LEC_state"].nunique()),
        "state_labels": ";".join(map(str, combined.obs["LEC_state"].cat.categories)),
        "leiden_resolution": float(args.resolution),
        "harmony_batch_key": embedding["harmony_batch_key"],
        "harmony_status": embedding["harmony_status"],
        "neighbors_representation": embedding["neighbors_representation"],
        "n_hvg_used_for_embedding": embedding["n_hvg_used"],
        "n_pcs_used": embedding["n_pcs_used"],
        "n_neighbors_used": embedding["n_neighbors_used"],
        "n_states_donor_dominated_over_50pct": int(state_qc["donor_dominance_flag_over_50pct"].sum()),
        "n_states_dataset_dominated_over_80pct": int(state_qc["dataset_dominance_flag_over_80pct"].sum()),
        "marker_method": args.marker_method,
        "marker_status": marker_status,
        "marker_message": marker_message,
        "output_h5ad": str(output_h5ad),
    }
    pd.DataFrame([run_summary]).to_csv(run_summary_path, index=False)
    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "step03_root": str(step03_root),
        "step06_root": str(step06_root),
        "output_root": str(output_root),
        "expected_cohorts": [list(pair) for pair in EXPECTED_COHORTS],
        "merge_scope": "all_final_LECs_all_active_disease_conditions",
        "expression_source": "Step03_raw_log_normalized",
        "gene_join": "outer",
        "n_top_hvg": args.n_top_hvg,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
        "resolution": args.resolution,
        "random_state": args.random_state,
        "harmony_enabled": not args.no_harmony,
        "harmony_batch_key": args.harmony_batch_key,
        "top_n_markers": args.top_n_markers,
        "marker_method": args.marker_method,
        "figures_enabled": not args.skip_figures,
    }
    with (output_root / "08_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    write_notes(output_root / "08_method_and_interpretation_notes.txt", args, run_summary, state_qc)

    print("Done")
    print(f"Combined shape: {combined.n_obs:,} LECs x {combined.n_vars:,} genes")
    print(f"States: {run_summary['state_labels']}")
    print(f"Harmony: {run_summary['harmony_status']}")
    print(f"Marker ranking: {marker_status}")
    print(f"Output: {output_h5ad}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
