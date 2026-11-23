#!/usr/bin/env python
r"""
Step 06: independently refine endothelial cells in the selected disease cohorts.

This standalone script uses the completed disease Steps 03 and 05. For each of
the exact seven retained dataset-condition cohorts it:

1. selects every all-cell Leiden cluster flagged as endothelial by Step 05;
2. backed-reads only those cells from the Step 03 h5ad (all genes retained);
3. carries the Step 05 raw and all-cell-cluster majority CellTypist labels;
4. recomputes EC-subset HVGs, PCA, donor Harmony, neighbors, Leiden and UMAP;
5. reruns Healthy Adult Heart CellTypist on every EC-candidate cell in chunks;
6. computes an exact majority vote within each new EC-only Leiden cluster;
7. calls clusters with a fresh EC8_ln/lymphatic majority label as LEC;
8. exports annotated EC h5ad files, full CSV summaries and brief figures.

The active cohort allowlist is enforced. Dataset02 and dataset06 are excluded
after the completed LEC-usability screen. Dataset03 (ICM), RV-only dataset10,
dataset12 COVID, and dataset08 NICM are also excluded. Region/QC validation
requires LV/AP for all ordinary cohorts and allows septum only for dataset11
HFpEF/NF, as established in disease Step 03.

Default inputs
--------------
new_result2_disease/03/
new_result2_disease/05_marker_celltypist/

Default outputs
---------------
new_result2_disease/06_EC_recluster/

PowerShell
----------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\06_recluster_EC_selected_disease.py

Read-only validation
--------------------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\06_recluster_EC_selected_disease.py `
      --validate_only
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import traceback
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Patch

try:
    import scanpy as sc
except ImportError as error:
    raise ImportError("scanpy is required; use the project .venv.") from error

try:
    import celltypist
except ImportError as error:
    raise ImportError("celltypist is required; use the project .venv.") from error


SCRIPT_VERSION = "06_selected_disease_exclude_dataset02_dataset06_dataset12_2026_08_11"
DEFAULT_STEP03_ROOT = "new_result2_disease/03"
DEFAULT_STEP05_ROOT = "new_result2_disease/05_marker_celltypist"
DEFAULT_STEP05_SUMMARY = (
    "new_result2_disease/05_marker_celltypist/"
    "05_all_cohorts_cluster_annotation_summary.csv"
)
DEFAULT_OUTPUT_ROOT = "new_result2_disease/06_EC_recluster"
DEFAULT_MODEL = "resources/celltypist/Healthy_Adult_Heart.pkl"
DEFAULT_CLUSTER_KEY = "leiden"

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

ENDOTHELIAL_MARKERS = [
    "PECAM1", "CDH5", "VWF", "CLDN5", "KDR", "FLT1", "EMCN", "RAMP2", "ESAM",
]
LEC_MARKERS = [
    "PROX1", "FLT4", "PDPN", "LYVE1", "CCL21", "MMRN1", "NRP2", "MRC1",
]
EC_SUBTYPE_MARKERS = [
    "GJA5", "EFNB2", "HEY1", "SOX17", "ACKR1", "SELE", "NRG1", "NFATC1",
    "NPR3", "CA4", "RGCC",
]
NEGATIVE_LINEAGE_MARKERS = [
    "DCN", "LUM", "COL1A1", "COL1A2", "COL3A1", "PDGFRA", "POSTN",
    "PTPRC", "LST1", "CD68", "C1QA", "LYZ",
    "RGS5", "PDGFRB", "ABCC9", "KCNJ8", "ACTA2", "TAGLN", "MYH11",
    "TNNT2", "TNNI3", "MYH6", "MYH7", "TTN",
]
MARKER_GENES = list(dict.fromkeys(
    ENDOTHELIAL_MARKERS + LEC_MARKERS + EC_SUBTYPE_MARKERS + NEGATIVE_LINEAGE_MARKERS
))


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def true_mask(values: pd.Series) -> pd.Series:
    return normalized_text(values).str.casefold().isin({"true", "1", "yes", "y"})


def sort_labels(values) -> list[str]:
    def key(value: object) -> tuple[int, object]:
        text = str(value)
        try:
            return (0, float(text))
        except ValueError:
            return (1, text.casefold())
    return sorted({str(value) for value in values}, key=key)


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
    requested: set[tuple[str, str]] = set()
    for token in re.split(r"[,;]", str(value)):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid cohort {token!r}; use dataset:condition, e.g. dataset08:NF."
            )
        dataset, condition = [item.strip() for item in token.split(":", 1)]
        requested.add((dataset, condition))
    invalid = requested - set(EXPECTED_COHORTS)
    if invalid:
        raise ValueError(f"Unapproved cohort subset: {sorted(invalid)}")
    return requested


def read_master(step03_root: Path) -> pd.DataFrame:
    source = step03_root / "03_master_run_summary.csv"
    if not source.exists():
        raise FileNotFoundError(f"Step 03 master summary not found: {source}")
    table = pd.read_csv(source)
    require_columns(
        table,
        [
            "dataset_folder", "condition", "status", "n_cells_after_all_QC",
            "n_donors_after_QC", "n_clusters", "output_h5ad",
        ],
        source,
    )
    table["dataset_folder"] = normalized_text(table["dataset_folder"])
    table["condition"] = normalized_text(table["condition"])
    forbidden = table["dataset_folder"].isin(FORBIDDEN_DATASETS)
    if forbidden.any():
        raise ValueError(
            "Forbidden datasets reappeared in Step 03: "
            f"{sorted(table.loc[forbidden, 'dataset_folder'].unique())}"
        )
    observed = set(zip(table["dataset_folder"], table["condition"]))
    expected = set(EXPECTED_COHORTS)
    if observed != expected:
        raise ValueError(
            f"Step 03 cohort mismatch. Missing={sorted(expected-observed)}; "
            f"unexpected={sorted(observed-expected)}"
        )
    if table.duplicated(["dataset_folder", "condition"]).any():
        raise ValueError("Step 03 master summary contains duplicate cohorts.")
    statuses = normalized_text(table["status"]).str.casefold()
    failed = ~(
        statuses.str.startswith("completed")
        | statuses.str.startswith("reused")
        | statuses.str.startswith("existing")
    )
    if failed.any():
        raise ValueError(
            "Step 03 has incomplete cohorts:\n"
            + table.loc[failed, ["dataset_folder", "condition", "status"]].to_string(index=False)
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


def read_selection(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Step 05 cluster summary not found: {summary_path}")
    table = pd.read_csv(summary_path, dtype={"cluster": str})
    require_columns(
        table,
        [
            "dataset_folder", "condition", "cluster", "n_cells",
            "cluster_majority_cell_type", "celltypist_label_is_endothelial",
            "celltypist_label_is_lymphatic",
        ],
        summary_path,
    )
    for column in ["dataset_folder", "condition", "cluster", "cluster_majority_cell_type"]:
        table[column] = normalized_text(table[column])
    if table["dataset_folder"].isin(FORBIDDEN_DATASETS).any():
        raise ValueError(
            "Step 05 summary contains an excluded dataset: dataset02, dataset03, "
            "dataset06, dataset10, or dataset12."
        )
    observed = set(zip(table["dataset_folder"], table["condition"]))
    expected = set(EXPECTED_COHORTS)
    if observed != expected:
        raise ValueError(
            f"Step 05 cohort mismatch. Missing={sorted(expected-observed)}; "
            f"unexpected={sorted(observed-expected)}"
        )
    selected = table.loc[true_mask(table["celltypist_label_is_endothelial"])].copy()
    if selected.empty:
        raise ValueError("Step 05 did not flag any endothelial-related clusters.")
    selected["n_cells"] = pd.to_numeric(selected["n_cells"], errors="raise").astype(int)
    selected["selection_rule"] = "step05_celltypist_label_is_endothelial_true"
    selected["cohort"] = [
        cohort_name(dataset, condition)
        for dataset, condition in zip(selected["dataset_folder"], selected["condition"])
    ]
    missing = expected - set(zip(selected["dataset_folder"], selected["condition"]))
    if missing:
        raise ValueError(f"No endothelial source cluster was selected for: {sorted(missing)}")
    order = {pair: index for index, pair in enumerate(EXPECTED_COHORTS)}
    selected["cohort_order"] = [
        order[(dataset, condition)]
        for dataset, condition in zip(selected["dataset_folder"], selected["condition"])
    ]
    selected["cluster_order"] = pd.Categorical(
        selected["cluster"], categories=sort_labels(selected["cluster"]), ordered=True
    )
    return selected.sort_values(["cohort_order", "cluster_order"]).drop(
        columns="cluster_order"
    ).reset_index(drop=True)


def resolve_h5ad(project_dir: Path, row: pd.Series) -> Path:
    path = resolve_path(project_dir, str(row["output_h5ad"]))
    if not path.exists():
        raise FileNotFoundError(f"Step 03 h5ad not found: {path}")
    return path


def step05_prediction_path(
    step05_root: Path, dataset: str, condition: str
) -> Path:
    prefix = cohort_prefix(dataset, condition)
    path = step05_root / dataset / condition / f"{prefix}_celltypist_cell_predictions.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Step 05 cell predictions not found: {path}")
    return path


def validate_source(
    source,
    path: Path,
    dataset: str,
    condition: str,
    expected_cells: int,
    selected_clusters: list[str],
    expected_selected_cells: int,
) -> dict[str, object]:
    require_columns(
        source.obs,
        [
            DEFAULT_CLUSTER_KEY, "donor_key", "disease",
            "region_standardized_analysis", "step03_final_QC_pass",
        ],
        path,
    )
    if int(source.n_obs) != int(expected_cells):
        raise ValueError(
            f"{dataset}/{condition}: Step 03 cell count is {source.n_obs:,}; "
            f"master summary says {expected_cells:,}."
        )
    diseases = set(normalized_text(source.obs["disease"]).unique())
    if diseases != {condition}:
        raise ValueError(
            f"{dataset}/{condition}: disease values are {sorted(diseases)}."
        )
    qc = true_mask(source.obs["step03_final_QC_pass"])
    if not qc.all():
        raise ValueError(f"{dataset}/{condition}: non-QC-passed cells remain in Step 03.")
    regions = set(normalized_text(source.obs["region_standardized_analysis"]).unique())
    allowed_regions = {"S"} if dataset == "dataset11" else {"LV", "AP"}
    if not regions or not regions.issubset(allowed_regions):
        raise ValueError(
            f"{dataset}/{condition}: regions {sorted(regions)} violate allowed "
            f"regions {sorted(allowed_regions)}."
        )
    cluster_values = normalized_text(source.obs[DEFAULT_CLUSTER_KEY])
    absent = set(selected_clusters) - set(cluster_values.unique())
    if absent:
        raise ValueError(
            f"{dataset}/{condition}: selected Step 05 clusters absent from h5ad: {sorted(absent)}"
        )
    n_selected = int(cluster_values.isin(selected_clusters).sum())
    if n_selected != int(expected_selected_cells):
        raise ValueError(
            f"{dataset}/{condition}: selected EC cells={n_selected:,}, but Step 05 "
            f"cluster summary totals {expected_selected_cells:,}."
        )
    if source.n_vars < 500:
        raise ValueError(f"{dataset}/{condition}: only {source.n_vars} genes are available.")
    return {
        "dataset_folder": dataset,
        "condition": condition,
        "cohort": cohort_name(dataset, condition),
        "status": "validated",
        "source_h5ad": str(path),
        "n_source_cells": int(source.n_obs),
        "n_source_genes": int(source.n_vars),
        "n_source_donors": int(normalized_text(source.obs["donor_key"]).nunique()),
        "regions": ";".join(sort_labels(regions)),
        "selected_source_clusters": ";".join(sort_labels(selected_clusters)),
        "n_selected_source_clusters": int(len(selected_clusters)),
        "n_selected_EC_cells": n_selected,
    }


def rows_to_memory(source, row_indices: np.ndarray) -> ad.AnnData:
    """Backed-read selected X rows while avoiding the duplicate source .raw."""
    matrix = source.X[row_indices, :]
    matrix = matrix.tocsr() if sp.issparse(matrix) else np.asarray(matrix)
    result = ad.AnnData(
        X=matrix,
        obs=source.obs.iloc[row_indices].copy(),
        var=source.var.copy(),
    )
    result.obs_names = source.obs_names[row_indices].astype(str)
    result.var_names = source.var_names.astype(str)
    result.var_names_make_unique()
    return result


def read_selected_step05_predictions(
    path: Path,
    selected_clusters: set[str],
    expected_cell_ids: pd.Index,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    required = [
        "cell_id", "cluster", "celltypist_raw_label", "celltypist_majority_vote",
        "celltypist_max_probability", "celltypist_probability_best_label",
    ]
    header = pd.read_csv(path, nrows=0)
    require_columns(header, required, path)
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype={"cell_id": str, "cluster": str}):
        chunk["cluster"] = normalized_text(chunk["cluster"])
        keep = chunk["cluster"].isin(selected_clusters)
        if keep.any():
            pieces.append(chunk.loc[keep, required].copy())
    if not pieces:
        raise ValueError(f"No selected EC predictions were found in {path}.")
    table = pd.concat(pieces, ignore_index=True)
    table["cell_id"] = normalized_text(table["cell_id"])
    if table["cell_id"].duplicated().any():
        raise ValueError(f"Duplicate Step 05 cell IDs in {path}.")
    table = table.set_index("cell_id", drop=False)
    expected = pd.Index(expected_cell_ids.astype(str))
    missing = expected.difference(table.index)
    extra = table.index.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            f"Step 05 selected-cell coverage mismatch in {path}: "
            f"missing={len(missing):,}, extra={len(extra):,}."
        )
    return table.reindex(expected)


def sample_lognorm_diagnostics(adata: ad.AnnData, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    take = min(256, adata.n_obs)
    rows = np.sort(rng.choice(adata.n_obs, size=take, replace=False))
    block = adata.X[rows, :]
    if sp.issparse(block):
        block = block.tocsr().copy()
        values = block.data
        if values.size and (not np.isfinite(values).all() or values.min() < 0):
            raise ValueError("Selected Step 03 X contains negative or non-finite values.")
        block.data = np.expm1(block.data)
        totals = np.asarray(block.sum(axis=1)).ravel()
    else:
        array = np.asarray(block, dtype=float)
        if not np.isfinite(array).all() or np.nanmin(array) < 0:
            raise ValueError("Selected Step 03 X contains negative or non-finite values.")
        totals = np.expm1(array).sum(axis=1)
    positive = totals[np.isfinite(totals) & (totals > 0)]
    if not len(positive):
        raise ValueError("Selected Step 03 expression has no positive sampled totals.")
    near = float(np.mean(np.isclose(positive, 1e4, rtol=1e-3, atol=1.0)))
    if near < 0.95:
        raise ValueError(
            "Step 03 X does not look like log1p expression normalized to 10,000: "
            f"only {near:.1%} of sampled cells reconstruct near 10,000."
        )
    return {
        "expression_source": "Step03_X_log1p_normalized_10000",
        "sampled_cells_for_normalization_check": int(len(positive)),
        "fraction_sampled_cells_reconstructing_to_10000": near,
        "median_reconstructed_total": float(np.median(positive)),
    }


def recompute_ec_embedding(
    adata: ad.AnnData,
    args: argparse.Namespace,
) -> tuple[ad.AnnData, dict[str, object]]:
    adata = adata.copy()
    adata.var_names_make_unique()
    batch_key = "donor_key" if adata.obs["donor_key"].nunique() > 1 else None
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(args.n_top_hvg, adata.n_vars),
        flavor="seurat",
        batch_key=batch_key,
    )
    n_hvg = int(adata.var["highly_variable"].sum())
    if n_hvg < 50:
        raise ValueError(f"Only {n_hvg} EC-subset HVGs were found.")
    n_pcs = int(min(args.n_pcs, adata.n_obs - 1, n_hvg - 1))
    if n_pcs < 2:
        raise ValueError("Too few cells/HVGs for EC-subset PCA.")
    sc.tl.pca(
        adata,
        n_comps=n_pcs,
        mask_var="highly_variable",
        svd_solver="arpack",
        zero_center=False,
        random_state=args.random_seed,
    )
    representation = "X_pca"
    harmony_used = False
    harmony_error = ""
    if args.use_harmony and batch_key:
        try:
            import scanpy.external as sce
            sce.pp.harmony_integrate(
                adata,
                key=batch_key,
                basis="X_pca",
                adjusted_basis="X_pca_harmony",
            )
            representation = "X_pca_harmony"
            harmony_used = True
        except Exception as error:
            harmony_error = str(error)
            warnings.warn(f"Harmony failed; using unadjusted EC PCA: {error}")
    n_neighbors = int(min(args.n_neighbors, max(2, adata.n_obs - 1)))
    n_pcs_neighbors = int(min(args.n_pcs_neighbors, n_pcs))
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs_neighbors,
        use_rep=representation,
        metric=args.neighbor_metric,
        random_state=args.random_seed,
    )
    sc.tl.umap(
        adata,
        min_dist=args.umap_min_dist,
        random_state=args.random_seed,
    )
    try:
        sc.tl.leiden(
            adata,
            resolution=args.resolution,
            key_added="leiden",
            flavor="igraph",
            n_iterations=2,
            directed=False,
            random_state=args.random_seed,
        )
    except Exception:
        sc.tl.leiden(
            adata,
            resolution=args.resolution,
            key_added="leiden",
            random_state=args.random_seed,
        )
    adata.obs["leiden"] = normalized_text(adata.obs["leiden"]).astype("category")
    params = {
        "script_version": SCRIPT_VERSION,
        "expression_was_renormalized": False,
        "all_genes_retained_in_X": True,
        "hvg_flavor": "seurat_on_existing_log1p_expression",
        "n_hvg": n_hvg,
        "n_pcs": n_pcs,
        "n_pcs_neighbors": n_pcs_neighbors,
        "n_neighbors": n_neighbors,
        "neighbor_metric": args.neighbor_metric,
        "representation": representation,
        "harmony_requested": bool(args.use_harmony),
        "harmony_used": harmony_used,
        "harmony_key": batch_key or "",
        "harmony_error": harmony_error,
        "umap_min_dist": float(args.umap_min_dist),
        "leiden_resolution": float(args.resolution),
        "random_seed": int(args.random_seed),
    }
    adata.uns["06_EC_recluster_params"] = params
    return adata, params


def model_overlap(adata: ad.AnnData, model_features: pd.Index) -> dict[str, object]:
    query = pd.Index(adata.var_names.astype(str))
    model_upper = set(model_features.astype(str).str.upper())
    overlap = sorted(set(query.str.upper()) & model_upper)
    return {
        "n_query_genes": int(adata.n_vars),
        "n_model_genes": int(len(model_features)),
        "n_model_genes_present": int(len(overlap)),
        "model_gene_overlap_fraction": float(len(overlap) / len(model_features)),
        "overlap_gene_examples": "; ".join(overlap[:50]),
    }


def prediction_frame(prediction, query: ad.AnnData) -> pd.DataFrame:
    labels = prediction.predicted_labels.copy()
    if isinstance(labels, pd.Series):
        labels = labels.to_frame("predicted_labels")
    labels.index = labels.index.astype(str)
    raw_column = "predicted_labels" if "predicted_labels" in labels else labels.columns[0]
    table = pd.DataFrame({
        "cell_id": query.obs_names.astype(str),
        "celltypist_raw_label": normalized_text(
            labels[raw_column].reindex(query.obs_names.astype(str))
        ).to_numpy(),
    })
    probability = prediction.probability_matrix
    if isinstance(probability, pd.DataFrame):
        probability.index = probability.index.astype(str)
        probability = probability.reindex(query.obs_names.astype(str))
        table["celltypist_max_probability"] = probability.max(axis=1).to_numpy()
        table["celltypist_probability_best_label"] = probability.idxmax(axis=1).astype(str).to_numpy()
    else:
        table["celltypist_max_probability"] = np.nan
        table["celltypist_probability_best_label"] = ""
    return table


def run_celltypist_chunked(
    adata: ad.AnnData,
    model,
    chunk_size: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for start in range(0, adata.n_obs, chunk_size):
        stop = min(start + chunk_size, adata.n_obs)
        query = ad.AnnData(
            X=adata.X[start:stop, :].copy(),
            obs=adata.obs.iloc[start:stop].copy(),
            var=adata.var.copy(),
        )
        query.obs_names = adata.obs_names[start:stop].astype(str)
        query.var_names = adata.var_names.astype(str)
        prediction = celltypist.annotate(query, model=model, majority_voting=False)
        table = prediction_frame(prediction, query)
        table["ec_leiden"] = normalized_text(query.obs["leiden"]).to_numpy()
        table["donor_key"] = normalized_text(query.obs["donor_key"]).to_numpy()
        parts.append(table)
        print(f"    CellTypist {stop:,}/{adata.n_obs:,} cells")
        del query, prediction, table
        gc.collect()
    result = pd.concat(parts, ignore_index=True)
    if len(result) != adata.n_obs or result["cell_id"].duplicated().any():
        raise ValueError("Fresh chunked CellTypist failed full-cell coverage validation.")
    return result


def label_is_endothelial(label: object) -> bool:
    text = str(label).strip().casefold()
    return bool(re.search(r"(^ec\d|endothelial|endocardial)", text))


def label_is_lymphatic(label: object) -> bool:
    text = str(label).strip().casefold()
    return text == "ec8_ln" or "lymph" in text


def exact_majority_vote(
    cells: pd.DataFrame,
    dataset: str,
    condition: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distribution = (
        cells.groupby(["ec_leiden", "celltypist_raw_label"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    sizes = cells["ec_leiden"].value_counts().rename("cluster_n_cells")
    distribution["fraction_within_cluster"] = distribution["n_cells"] / distribution[
        "ec_leiden"
    ].map(sizes)
    distribution = distribution.sort_values(
        ["ec_leiden", "n_cells", "celltypist_raw_label"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    winner_map: dict[str, str] = {}
    for cluster in sort_labels(distribution["ec_leiden"]):
        group = distribution.loc[distribution["ec_leiden"].astype(str).eq(cluster)].reset_index(drop=True)
        winner = group.iloc[0]
        second = group.iloc[1] if len(group) > 1 else None
        label = str(winner["celltypist_raw_label"])
        winner_map[cluster] = label
        subset = cells.loc[cells["ec_leiden"].astype(str).eq(cluster)]
        confidence = pd.to_numeric(subset["celltypist_max_probability"], errors="coerce")
        rows.append({
            "dataset_folder": dataset,
            "condition": condition,
            "cohort": cohort_name(dataset, condition),
            "ec_leiden": cluster,
            "n_cells": int(len(subset)),
            "cluster_majority_cell_type": label,
            "majority_vote_support_fraction": float(winner["fraction_within_cluster"]),
            "raw_prediction_majority_label": label,
            "raw_prediction_majority_fraction": float(winner["fraction_within_cluster"]),
            "second_raw_prediction_label": (
                str(second["celltypist_raw_label"]) if second is not None else ""
            ),
            "second_raw_prediction_fraction": (
                float(second["fraction_within_cluster"]) if second is not None else 0.0
            ),
            "mean_celltypist_confidence": float(confidence.mean()),
            "median_celltypist_confidence": float(confidence.median()),
            "n_distinct_raw_cell_types": int(subset["celltypist_raw_label"].nunique()),
            "celltypist_label_is_endothelial": label_is_endothelial(label),
            "celltypist_label_is_lymphatic": label_is_lymphatic(label),
            "selected_as_LEC": label_is_lymphatic(label),
            "LEC_selection_rule": "fresh_EC_only_exact_majority_vote_EC8_ln_or_lymphatic",
        })
    cluster_summary = pd.DataFrame(rows)
    cells["celltypist_majority_vote"] = cells["ec_leiden"].astype(str).map(winner_map)
    lec_map = cluster_summary.set_index("ec_leiden")["selected_as_LEC"]
    cells["selected_as_LEC"] = cells["ec_leiden"].astype(str).map(lec_map).fillna(False).astype(bool)
    cells["final_LEC_by_majority_vote"] = np.where(
        cells["selected_as_LEC"], "LEC", "other_EC_candidate"
    )
    distribution.insert(0, "cohort", cohort_name(dataset, condition))
    distribution.insert(0, "condition", condition)
    distribution.insert(0, "dataset_folder", dataset)
    return cells, cluster_summary, distribution


def marker_expression_summary(
    adata: ad.AnnData,
    dataset: str,
    condition: str,
) -> pd.DataFrame:
    names = pd.Index(adata.var_names.astype(str))
    lookup = {name.upper(): name for name in names}
    present = [gene for gene in MARKER_GENES if gene in lookup]
    matched = [lookup[gene] for gene in present]
    clusters = normalized_text(adata.obs["leiden"]).to_numpy()
    rows: list[dict[str, object]] = []
    matrix = adata[:, matched].X if matched else None
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
    for cluster in sort_labels(pd.unique(clusters)):
        mask = clusters == cluster
        row: dict[str, object] = {
            "dataset_folder": dataset,
            "condition": condition,
            "cohort": cohort_name(dataset, condition),
            "ec_leiden": cluster,
            "n_cells_marker_summary": int(mask.sum()),
        }
        if matched:
            block = matrix[mask]
            if sp.issparse(block):
                means = np.asarray(block.mean(axis=0)).ravel()
                detected = np.asarray((block > 0).mean(axis=0)).ravel()
            else:
                array = np.asarray(block)
                means = np.mean(array, axis=0)
                detected = np.mean(array > 0, axis=0)
            for gene, mean, fraction in zip(present, means, detected):
                row[f"mean_expr_{gene}"] = float(mean)
                row[f"pct_expr_{gene}"] = float(fraction)
        for name, genes in [
            ("endothelial", ENDOTHELIAL_MARKERS),
            ("LEC", LEC_MARKERS),
            ("negative_check", NEGATIVE_LINEAGE_MARKERS),
        ]:
            available = [gene for gene in genes if gene in present]
            row[f"{name}_markers_present"] = "; ".join(available)
            row[f"n_{name}_markers_present"] = len(available)
            row[f"{name}_marker_pct_expr_mean"] = (
                float(np.mean([row[f"pct_expr_{gene}"] for gene in available]))
                if available else np.nan
            )
            row[f"{name}_marker_score_mean"] = (
                float(np.mean([row[f"mean_expr_{gene}"] for gene in available]))
                if available else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def rank_markers(
    adata: ad.AnnData,
    dataset: str,
    condition: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if adata.obs["leiden"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame(), "only one EC cluster"
    n_rank = adata.n_vars if args.marker_n_genes <= 0 else min(args.marker_n_genes, adata.n_vars)
    n_rank = max(args.top_n_markers, n_rank)
    key = "rank_genes_groups_EC_leiden"
    kwargs = dict(
        groupby="leiden",
        reference="rest",
        method=args.marker_method,
        n_genes=n_rank,
        use_raw=False,
        pts=True,
        key_added=key,
    )
    try:
        sc.tl.rank_genes_groups(adata, **kwargs)
    except TypeError:
        kwargs.pop("pts", None)
        sc.tl.rank_genes_groups(adata, **kwargs)
    tables: list[pd.DataFrame] = []
    for cluster in sort_labels(adata.obs["leiden"].astype(str).unique()):
        table = sc.get.rank_genes_groups_df(adata, group=cluster, key=key)
        table = table.rename(columns={
            "names": "gene", "logfoldchanges": "logfoldchange",
            "pct_nz_group": "pct_in", "pct_nz_reference": "pct_out",
        })
        table["gene"] = normalized_text(table["gene"])
        table = table.loc[~table["gene"].str.upper().isin({"", "NAN", "NONE"})]
        if "scores" in table:
            table = table.loc[pd.to_numeric(table["scores"], errors="coerce") > 0]
        table = table.head(args.top_n_markers).copy()
        table.insert(0, "rank", np.arange(1, len(table) + 1))
        table.insert(0, "ec_leiden", cluster)
        table.insert(0, "cohort", cohort_name(dataset, condition))
        table.insert(0, "condition", condition)
        table.insert(0, "dataset_folder", dataset)
        tables.append(table)
    long = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    wide_rows: list[dict[str, object]] = []
    for cluster in sort_labels(long["ec_leiden"].unique()) if not long.empty else []:
        genes = long.loc[long["ec_leiden"].eq(cluster)].sort_values("rank")["gene"].tolist()
        row: dict[str, object] = {
            "dataset_folder": dataset,
            "condition": condition,
            "cohort": cohort_name(dataset, condition),
            "ec_leiden": cluster,
            "top_marker_genes_semicolon": "; ".join(genes),
        }
        for index, gene in enumerate(genes, start=1):
            row[f"marker_{index}"] = gene
        wide_rows.append(row)
    return long, pd.DataFrame(wide_rows), "completed"


def stable_label_colors(labels: list[str]) -> dict[str, str]:
    values = sorted(set(map(str, labels)), key=str.casefold)
    cmap = plt.get_cmap("gist_ncar")
    return {
        label: to_hex(cmap((index + 0.5) / max(len(values), 1)))
        for index, label in enumerate(values)
    }


def categorical_colors(labels) -> dict[str, str]:
    values = sort_labels(labels)
    cmap = plt.get_cmap("tab20")
    return {label: to_hex(cmap(index % 20)) for index, label in enumerate(values)}


def plot_indices(labels: pd.Series, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or len(labels) <= maximum:
        return np.arange(len(labels), dtype=int)
    values = normalized_text(labels).to_numpy()
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    minimum = max(20, maximum // max(10 * len(pd.unique(values)), 1))
    for label in sort_labels(pd.unique(values)):
        available = np.flatnonzero(values == label)
        take = min(minimum, len(available))
        chosen.extend(rng.choice(available, size=take, replace=False).tolist())
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) > maximum:
        chosen = rng.choice(chosen, size=maximum, replace=False).tolist()
    remaining = np.setdiff1d(np.arange(len(values)), np.asarray(chosen, dtype=int))
    extra = min(maximum - len(chosen), len(remaining))
    if extra:
        chosen.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    return np.asarray(chosen, dtype=int)


def save_umap(
    adata: ad.AnnData,
    column: str,
    title: str,
    path: Path,
    colors: dict[str, str],
    maximum: int,
    seed: int,
    dpi: int,
) -> None:
    index = plot_indices(adata.obs[column], maximum, seed)
    xy = np.asarray(adata.obsm["X_umap"])[index]
    labels = normalized_text(adata.obs.iloc[index][column]).to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for label in sort_labels(pd.unique(labels)):
        mask = labels == label
        ax.scatter(
            xy[mask, 0], xy[mask, 1], s=4, alpha=0.75,
            color=colors.get(label, "#777777"), linewidths=0, rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(color=colors.get(label, "#777777"), label=label) for label in sort_labels(pd.unique(labels))]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_per_cohort_figures(
    adata: ad.AnnData,
    cluster_summary: pd.DataFrame,
    dataset: str,
    condition: str,
    figure_dir: Path,
    celltype_colors: dict[str, str],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    prefix = cohort_prefix(dataset, condition)
    cohort = cohort_name(dataset, condition)
    manifest: list[dict[str, str]] = []
    cluster_colors = categorical_colors(adata.obs["leiden"])
    source_colors = categorical_colors(adata.obs["allcell_leiden"])
    specifications = [
        ("allcell_leiden", "Selected parent all-cell Leiden clusters", "source_allcell_leiden", source_colors),
        ("leiden", "New EC-only Leiden clusters", "EC_leiden", cluster_colors),
        ("celltypist_raw_label", "Fresh EC-only raw CellTypist labels", "EC_celltypist_raw", celltype_colors),
        ("celltypist_majority_vote", "Fresh EC-cluster CellTypist majority vote", "EC_celltypist_majority", celltype_colors),
        (
            "final_LEC_by_majority_vote", "Final LEC selection", "final_LEC_selection",
            {"LEC": "#D62728", "other_EC_candidate": "#BDBDBD"},
        ),
    ]
    for offset, (column, subtitle, suffix, colors) in enumerate(specifications):
        path = figure_dir / f"{prefix}_{suffix}_UMAP.png"
        save_umap(
            adata, column, f"{cohort}: {subtitle}", path, colors,
            args.max_umap_cells, args.random_seed + offset, args.dpi,
        )
        manifest.append({
            "dataset_folder": dataset, "condition": condition, "cohort": cohort,
            "figure": str(path), "description": subtitle,
        })
    ordered = cluster_summary.copy()
    ordered["_order"] = pd.Categorical(
        ordered["ec_leiden"].astype(str), categories=sort_labels(ordered["ec_leiden"]), ordered=True
    )
    ordered = ordered.sort_values("_order")
    path = figure_dir / f"{prefix}_EC_cluster_sizes_and_labels.png"
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(ordered)), 4.8))
    bar_colors = [
        celltype_colors.get(str(label), "#777777")
        for label in ordered["cluster_majority_cell_type"]
    ]
    ax.bar(ordered["ec_leiden"].astype(str), ordered["n_cells"], color=bar_colors)
    ax.set_xlabel("EC-only Leiden cluster")
    ax.set_ylabel("Cells")
    ax.set_title(f"{cohort}: refined EC clusters (color = majority CellTypist label)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest.append({
        "dataset_folder": dataset, "condition": condition, "cohort": cohort,
        "figure": str(path), "description": "Refined EC cluster sizes and majority labels",
    })
    return manifest


def save_combined_figures(
    run_summary: pd.DataFrame,
    marker_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    output_root: Path,
    celltype_colors: dict[str, str],
    dpi: int,
) -> list[dict[str, str]]:
    figure_dir = output_root / "figures_summary"
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    completed = run_summary.loc[run_summary["status"].isin(["completed", "reused"])].copy()
    if completed.empty:
        return manifest
    labels = completed["cohort"].astype(str).tolist()
    x = np.arange(len(labels))

    path = figure_dir / "06_EC_and_LEC_cell_counts_by_cohort.png"
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(labels)), 5.2))
    width = 0.38
    ax.bar(x - width / 2, completed["n_EC_cells"], width, label="EC candidates", color="#4C78A8")
    ax.bar(x + width / 2, completed["n_LEC_cells"], width, label="LEC", color="#D62728")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Cells")
    ax.set_title("Refined endothelial candidates and majority-vote LEC cells")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest.append({"cohort": "combined", "figure": str(path), "description": "EC and LEC counts by cohort"})

    path = figure_dir / "06_LEC_percentage_among_EC_by_cohort.png"
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(labels)), 5.2))
    ax.bar(x, completed["percent_LEC_among_EC"], color="#D62728")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("LEC among selected EC candidates (%)")
    ax.set_title("Fresh EC-cluster majority-vote LEC percentage")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest.append({"cohort": "combined", "figure": str(path), "description": "LEC percentage among EC candidates"})

    if not cluster_summary.empty:
        composition = cluster_summary.pivot_table(
            index="cohort", columns="cluster_majority_cell_type", values="n_cells",
            aggfunc="sum", fill_value=0,
        ).reindex(labels).fillna(0)
        fractions = composition.div(composition.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        path = figure_dir / "06_EC_CellTypist_composition_by_cohort.png"
        fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(labels)), 6.0))
        bottom = np.zeros(len(fractions))
        for label in sorted(fractions.columns, key=str.casefold):
            values = fractions[label].to_numpy()
            ax.bar(x, values, bottom=bottom, label=label, color=celltype_colors.get(label, "#777777"))
            bottom += values
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of selected EC cells")
        ax.set_title("Fresh majority-vote CellTypist composition")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        manifest.append({"cohort": "combined", "figure": str(path), "description": "Fresh CellTypist composition by cohort"})

    lec_rows = cluster_summary.loc[true_mask(cluster_summary["selected_as_LEC"])].copy()
    genes = [gene for gene in LEC_MARKERS if f"pct_expr_{gene}" in marker_summary.columns]
    if not lec_rows.empty and genes:
        keys = lec_rows[["dataset_folder", "condition", "ec_leiden"]]
        heat = keys.merge(marker_summary, on=["dataset_folder", "condition", "ec_leiden"], how="left")
        row_labels = [
            f"{cohort_name(d, c)} | EC{cluster}"
            for d, c, cluster in zip(heat["dataset_folder"], heat["condition"], heat["ec_leiden"])
        ]
        matrix = heat[[f"pct_expr_{gene}" for gene in genes]].to_numpy(float)
        path = figure_dir / "06_LEC_candidate_marker_detection_heatmap.png"
        fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(genes)), max(4, 0.35 * len(row_labels))))
        image = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0, vmax=max(0.01, np.nanmax(matrix)))
        ax.set_xticks(np.arange(len(genes)), genes, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        ax.set_title("LEC candidate clusters: marker detection fractions")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        colorbar.set_label("Fraction expressing")
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        manifest.append({"cohort": "combined", "figure": str(path), "description": "LEC marker detection heatmap"})
    return manifest


def cohort_paths(
    output_root: Path,
    dataset: str,
    condition: str,
    top_n_markers: int,
) -> dict[str, Path]:
    folder = output_root / dataset / condition
    prefix = cohort_prefix(dataset, condition)
    return {
        "folder": folder,
        "h5ad": folder / f"{prefix}_EC_reclustered.h5ad",
        "selected": folder / f"{prefix}_selected_EC_cells_before_reclustering.csv.gz",
        "cells": folder / f"{prefix}_EC_celltypist_cell_predictions.csv.gz",
        "distribution": folder / f"{prefix}_EC_celltypist_label_distribution_by_cluster.csv",
        "clusters": folder / f"{prefix}_EC_celltypist_cluster_summary.csv",
        "counts": folder / f"{prefix}_EC_recluster_counts.csv",
        "marker_summary": folder / f"{prefix}_EC_recluster_marker_summary.csv",
        "marker_long": folder / f"{prefix}_EC_top{top_n_markers}_markers_long.csv",
        "marker_wide": folder / f"{prefix}_EC_top{top_n_markers}_markers_wide.csv",
        "lec_clusters": folder / f"{prefix}_final_LEC_candidate_clusters.csv",
        "lec_cells": folder / f"{prefix}_final_LEC_cells_by_majority_vote.csv.gz",
        "overlap": folder / f"{prefix}_EC_celltypist_model_gene_overlap.csv",
        "run": folder / f"{prefix}_06_run_summary.csv",
        "figures": folder / "figures",
    }


def load_existing(paths: dict[str, Path]) -> dict[str, object] | None:
    core = ["h5ad", "cells", "distribution", "clusters", "counts", "marker_summary", "lec_clusters", "run"]
    if not all(paths[key].exists() for key in core):
        return None
    run = pd.read_csv(paths["run"]).iloc[0].to_dict()
    run["status"] = "reused"
    return {
        "summary": run,
        "counts": pd.read_csv(paths["counts"], dtype={"ec_leiden": str, "source_allcell_leiden": str}),
        "marker_summary": pd.read_csv(paths["marker_summary"], dtype={"ec_leiden": str}),
        "marker_long": pd.read_csv(paths["marker_long"], dtype={"ec_leiden": str}) if paths["marker_long"].exists() else pd.DataFrame(),
        "marker_wide": pd.read_csv(paths["marker_wide"], dtype={"ec_leiden": str}) if paths["marker_wide"].exists() else pd.DataFrame(),
        "clusters": pd.read_csv(paths["clusters"], dtype={"ec_leiden": str}),
        "distribution": pd.read_csv(paths["distribution"], dtype={"ec_leiden": str}),
        "lec_clusters": pd.read_csv(paths["lec_clusters"], dtype={"ec_leiden": str}),
        "figures": [],
    }


def process_cohort(
    row: pd.Series,
    cohort_selection: pd.DataFrame,
    project_dir: Path,
    step05_root: Path,
    output_root: Path,
    model,
    model_features: pd.Index,
    celltype_colors: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, object]:
    dataset = str(row["dataset_folder"])
    condition = str(row["condition"])
    cohort = cohort_name(dataset, condition)
    paths = cohort_paths(output_root, dataset, condition, args.top_n_markers)
    if args.resume and not args.force:
        existing = load_existing(paths)
        if existing is not None:
            print(f"[{cohort}] reusing completed Step 06 outputs")
            return existing
    if paths["h5ad"].exists() and not args.force:
        raise FileExistsError(
            f"Partial/existing output found: {paths['h5ad']}. Use --force to replace it."
        )
    paths["folder"].mkdir(parents=True, exist_ok=True)
    source_path = resolve_h5ad(project_dir, row)
    selected_clusters = sort_labels(cohort_selection["cluster"])
    expected_selected = int(cohort_selection["n_cells"].sum())
    print(f"\n[{cohort}] parent EC clusters: {selected_clusters}")
    source = ad.read_h5ad(source_path, backed="r")
    try:
        cluster_values = normalized_text(source.obs[DEFAULT_CLUSTER_KEY])
        selected_rows = np.flatnonzero(cluster_values.isin(selected_clusters).to_numpy())
        ec = rows_to_memory(source, selected_rows)
    finally:
        source.file.close()
    if ec.n_obs != expected_selected:
        raise ValueError(f"{cohort}: backed selection returned the wrong number of cells.")
    ec.obs["allcell_leiden"] = normalized_text(ec.obs[DEFAULT_CLUSTER_KEY]).to_numpy()
    ec.obs["step05_EC_candidate"] = "yes"
    ec.obs["dataset_folder"] = dataset
    ec.obs["condition"] = condition
    ec.obs["cohort"] = cohort

    prediction_path = step05_prediction_path(step05_root, dataset, condition)
    carried = read_selected_step05_predictions(
        prediction_path, set(selected_clusters), pd.Index(ec.obs_names.astype(str))
    )
    for column in [
        "celltypist_raw_label", "celltypist_majority_vote",
        "celltypist_max_probability", "celltypist_probability_best_label",
    ]:
        values = carried[column]
        output = f"step05_{column}"
        if column == "celltypist_max_probability":
            ec.obs[output] = pd.to_numeric(values, errors="coerce").to_numpy()
        else:
            ec.obs[output] = normalized_text(values).to_numpy()
    selected_columns = [
        "allcell_leiden", "donor_key", "donor_id", "donor_id_source",
        "source_cell_id", "dataset_folder", "condition", "disease",
        "region_standardized_analysis", "sex", "age",
        "step05_celltypist_raw_label", "step05_celltypist_majority_vote",
        "step05_celltypist_max_probability",
    ]
    selected_columns = [column for column in selected_columns if column in ec.obs]
    selected_table = ec.obs[selected_columns].copy()
    selected_table.insert(0, "cell_id", selected_table.index.astype(str))
    selected_table.to_csv(paths["selected"], index=False, compression="gzip")

    normalization = sample_lognorm_diagnostics(ec, args.random_seed)
    ec, embedding = recompute_ec_embedding(ec, args)
    ec.obs["ec_recluster_id"] = (
        dataset + "|" + condition + "|" + ec.obs["leiden"].astype(str)
    )
    overlap = model_overlap(ec, model_features)
    if overlap["n_model_genes_present"] < args.min_model_genes_present:
        raise ValueError(
            f"{cohort}: only {overlap['n_model_genes_present']:,} model genes are present."
        )
    print(f"[{cohort}] fresh CellTypist over {ec.n_obs:,} EC candidates")
    fresh_cells = run_celltypist_chunked(ec, model, args.celltypist_chunk_size)
    fresh_cells, cluster_summary, distribution = exact_majority_vote(
        fresh_cells, dataset, condition
    )
    indexed = fresh_cells.set_index("cell_id")
    for column in [
        "celltypist_raw_label", "celltypist_majority_vote",
        "celltypist_max_probability", "celltypist_probability_best_label",
        "selected_as_LEC", "final_LEC_by_majority_vote",
    ]:
        values = indexed.reindex(ec.obs_names.astype(str))[column]
        if column == "celltypist_max_probability":
            ec.obs[column] = pd.to_numeric(values, errors="coerce").to_numpy()
        elif column == "selected_as_LEC":
            ec.obs[column] = values.fillna(False).astype(bool).to_numpy()
        else:
            ec.obs[column] = normalized_text(values).to_numpy()

    counts = (
        ec.obs.groupby(["leiden", "allcell_leiden"], observed=True)
        .size().reset_index(name="n_cells")
        .rename(columns={"leiden": "ec_leiden", "allcell_leiden": "source_allcell_leiden"})
    )
    counts.insert(0, "cohort", cohort)
    counts.insert(0, "condition", condition)
    counts.insert(0, "dataset_folder", dataset)
    marker_summary = marker_expression_summary(ec, dataset, condition)
    marker_long = pd.DataFrame()
    marker_wide = pd.DataFrame()
    marker_status = "skipped"
    marker_message = "disabled by --skip_markers"
    if not args.skip_markers:
        marker_long, marker_wide, marker_message = rank_markers(
            ec, dataset, condition, args
        )
        marker_status = "completed" if not marker_long.empty else "empty"
    cluster_annotation = cluster_summary.merge(
        marker_summary,
        on=["dataset_folder", "condition", "cohort", "ec_leiden"],
        how="left",
    )
    if not marker_wide.empty:
        cluster_annotation = cluster_annotation.merge(
            marker_wide[["dataset_folder", "condition", "ec_leiden", "top_marker_genes_semicolon"]],
            on=["dataset_folder", "condition", "ec_leiden"], how="left",
        )
    lec_clusters = cluster_annotation.loc[true_mask(cluster_annotation["selected_as_LEC"])].copy()
    lec_cluster_ids = set(lec_clusters["ec_leiden"].astype(str))
    ec.obs["selected_as_LEC"] = ec.obs["leiden"].astype(str).isin(lec_cluster_ids).to_numpy()
    ec.obs["final_LEC_by_majority_vote"] = np.where(
        ec.obs["selected_as_LEC"], "LEC", "other_EC_candidate"
    )
    lec_columns = [
        "leiden", "allcell_leiden", "donor_key", "donor_id", "donor_id_source",
        "source_cell_id", "dataset_folder", "condition", "disease",
        "region_standardized_analysis", "sex", "age",
        "celltypist_raw_label", "celltypist_majority_vote",
        "celltypist_max_probability", "step05_celltypist_raw_label",
        "step05_celltypist_majority_vote",
    ]
    lec_columns = [column for column in lec_columns if column in ec.obs]
    lec_cells = ec.obs.loc[ec.obs["selected_as_LEC"], lec_columns].copy()
    lec_cells.insert(0, "cell_id", lec_cells.index.astype(str))

    ec.uns["06_EC_source_selection"] = {
        "script_version": SCRIPT_VERSION,
        "selection_rule": "step05_celltypist_label_is_endothelial_true",
        "source_step03_h5ad": str(source_path),
        "source_step05_predictions": str(prediction_path),
        "selected_source_clusters": selected_clusters,
        "dataset_folder": dataset,
        "condition": condition,
        "region_rule": "dataset11_septum_allowed_else_LV_AP_only",
    }
    ec.uns["06_fresh_celltypist"] = {
        "model": str(args.model_resolved),
        "prediction_scope": "all_EC_candidate_cells_chunked",
        "majority_vote_scope": "all_raw_predictions_within_new_EC_leiden_cluster",
        "LEC_selection_rule": "fresh_EC_only_exact_majority_vote_EC8_ln_or_lymphatic",
        **overlap,
    }

    fresh_cells.insert(0, "cohort", cohort)
    fresh_cells.insert(0, "condition", condition)
    fresh_cells.insert(0, "dataset_folder", dataset)
    fresh_cells.to_csv(paths["cells"], index=False, compression="gzip")
    distribution.to_csv(paths["distribution"], index=False)
    cluster_annotation.to_csv(paths["clusters"], index=False)
    counts.to_csv(paths["counts"], index=False)
    marker_summary.to_csv(paths["marker_summary"], index=False)
    if not marker_long.empty:
        marker_long.to_csv(paths["marker_long"], index=False)
    if not marker_wide.empty:
        marker_wide.to_csv(paths["marker_wide"], index=False)
    lec_clusters.to_csv(paths["lec_clusters"], index=False)
    lec_cells.to_csv(paths["lec_cells"], index=False, compression="gzip")
    pd.DataFrame([{**overlap, **normalization}]).to_csv(paths["overlap"], index=False)

    # Save scientific outputs before optional plotting.
    ec.write_h5ad(paths["h5ad"], compression="lzf")
    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        try:
            figure_manifest = save_per_cohort_figures(
                ec, cluster_annotation, dataset, condition, paths["figures"],
                celltype_colors, args,
            )
        except Exception as error:
            warnings.warn(f"{cohort}: figures failed after scientific outputs were saved: {error}")

    n_lec_cells = int(ec.obs["selected_as_LEC"].sum())
    summary = {
        "script_version": SCRIPT_VERSION,
        "dataset_folder": dataset,
        "condition": condition,
        "cohort": cohort,
        "status": "completed",
        "selected_source_allcell_clusters": ";".join(selected_clusters),
        "n_source_all_cells": int(row["n_cells_after_all_QC"]),
        "n_selected_source_clusters": int(len(selected_clusters)),
        "n_EC_cells": int(ec.n_obs),
        "n_EC_genes_retained": int(ec.n_vars),
        "n_donors_in_EC_subset": int(normalized_text(ec.obs["donor_key"]).nunique()),
        "n_EC_only_leiden_clusters": int(ec.obs["leiden"].nunique()),
        "selected_LEC_clusters": ";".join(sort_labels(lec_cluster_ids)),
        "n_LEC_clusters": int(len(lec_cluster_ids)),
        "n_LEC_cells": n_lec_cells,
        "percent_LEC_among_EC": 100.0 * n_lec_cells / ec.n_obs,
        "marker_status": marker_status,
        "marker_message": marker_message,
        "harmony_used": embedding["harmony_used"],
        "harmony_key": embedding["harmony_key"],
        "n_hvg": embedding["n_hvg"],
        "n_pcs": embedding["n_pcs"],
        "leiden_resolution": embedding["leiden_resolution"],
        "n_model_genes_present": overlap["n_model_genes_present"],
        "model_gene_overlap_fraction": overlap["model_gene_overlap_fraction"],
        "output_h5ad": str(paths["h5ad"]),
    }
    pd.DataFrame([summary]).to_csv(paths["run"], index=False)
    del ec, fresh_cells, indexed, carried
    gc.collect()
    return {
        "summary": summary,
        "counts": counts,
        "marker_summary": marker_summary,
        "marker_long": marker_long,
        "marker_wide": marker_wide,
        "clusters": cluster_annotation,
        "distribution": distribution,
        "lec_clusters": lec_clusters,
        "figures": figure_manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Step 05 EC clusters, recluster per disease cohort, and find LEC by fresh majority vote."
    )
    parser.add_argument("--step03_root", default=DEFAULT_STEP03_ROOT)
    parser.add_argument("--step05_root", default=DEFAULT_STEP05_ROOT)
    parser.add_argument("--step05_summary", default=DEFAULT_STEP05_SUMMARY)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cohorts", default="all", help="all or comma-separated dataset:condition values")
    parser.add_argument("--n_top_hvg", type=int, default=1500)
    parser.add_argument("--n_pcs", type=int, default=30)
    parser.add_argument("--n_pcs_neighbors", type=int, default=30)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--neighbor_metric", default="cosine")
    parser.add_argument("--umap_min_dist", type=float, default=0.2)
    parser.add_argument("--resolution", type=float, default=0.4)
    parser.add_argument("--no_harmony", action="store_false", dest="use_harmony")
    parser.set_defaults(use_harmony=True)
    parser.add_argument(
        "--marker_method",
        choices=["t-test", "t-test_overestim_var", "wilcoxon", "logreg"],
        default="t-test_overestim_var",
    )
    parser.add_argument("--marker_n_genes", type=int, default=500)
    parser.add_argument("--top_n_markers", type=int, default=25)
    parser.add_argument("--celltypist_chunk_size", type=int, default=10000)
    parser.add_argument("--min_model_genes_present", type=int, default=500)
    parser.add_argument("--max_umap_cells", type=int, default=30000)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_markers", action="store_true")
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_top_hvg < 50 or args.n_pcs < 2 or args.n_pcs_neighbors < 2:
        raise ValueError("n_top_hvg must be >=50 and PCA dimensions must be >=2.")
    if args.n_neighbors < 2 or args.resolution <= 0:
        raise ValueError("n_neighbors must be >=2 and resolution must be positive.")
    if args.marker_n_genes < 0 or args.top_n_markers < 1:
        raise ValueError("marker_n_genes cannot be negative and top_n_markers must be positive.")
    if args.celltypist_chunk_size < 100 or args.min_model_genes_present < 1:
        raise ValueError("CellTypist chunk size must be >=100 and model-gene minimum positive.")
    if args.max_umap_cells < 100:
        raise ValueError("max_umap_cells must be >=100.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    project_dir = find_project_dir()
    step03_root = resolve_path(project_dir, args.step03_root)
    step05_root = resolve_path(project_dir, args.step05_root)
    step05_summary = resolve_path(project_dir, args.step05_summary)
    output_root = resolve_path(project_dir, args.output_root)
    model_path = resolve_path(project_dir, args.model)
    args.model_resolved = model_path
    if not model_path.exists():
        raise FileNotFoundError(f"CellTypist model not found: {model_path}")
    master = read_master(step03_root)
    selection = read_selection(step05_summary)
    subset = parse_cohort_subset(args.cohorts)
    if subset is not None:
        master = master.loc[
            [(dataset, condition) in subset for dataset, condition in zip(master["dataset_folder"], master["condition"])]
        ].copy()
        selection = selection.loc[
            [(dataset, condition) in subset for dataset, condition in zip(selection["dataset_folder"], selection["condition"])]
        ].copy()
    model = celltypist.models.Model.load(str(model_path))
    model_features = pd.Index([str(feature) for feature in model.features])
    model_labels = [str(label) for label in model.cell_types]
    celltype_colors = stable_label_colors(model_labels)

    print("=" * 100)
    print("Selected disease Step 06: independent EC reclustering + fresh LEC majority vote")
    print(f"Step 03 root: {step03_root}")
    print(f"Step 05 summary: {step05_summary}")
    print(f"Output root: {output_root}")
    print(f"Cohorts: {', '.join(master['cohort'])}")
    print("Region rule: LV/AP; dataset11 septum explicitly allowed")
    print("Excluded: dataset02, dataset06, dataset03, RV-only dataset10, dataset12 COVID, and dataset08 NICM")
    print(f"Clustering: {args.n_top_hvg} HVGs, {args.n_pcs} PCs, Harmony={args.use_harmony}, resolution={args.resolution}")
    print(f"CellTypist: all EC-candidate cells, chunks of {args.celltypist_chunk_size}")
    print("=" * 100)

    validation_rows: list[dict[str, object]] = []
    for _, row in master.iterrows():
        dataset = str(row["dataset_folder"])
        condition = str(row["condition"])
        cohort_selection = selection.loc[
            selection["dataset_folder"].eq(dataset) & selection["condition"].eq(condition)
        ]
        selected_clusters = sort_labels(cohort_selection["cluster"])
        source_path = resolve_h5ad(project_dir, row)
        prediction_path = step05_prediction_path(step05_root, dataset, condition)
        source = ad.read_h5ad(source_path, backed="r")
        try:
            validation = validate_source(
                source, source_path, dataset, condition,
                int(row["n_cells_after_all_QC"]), selected_clusters,
                int(cohort_selection["n_cells"].sum()),
            )
            overlap = model_overlap(source, model_features)
            if overlap["n_model_genes_present"] < args.min_model_genes_present:
                raise ValueError(
                    f"{cohort_name(dataset, condition)}: only "
                    f"{overlap['n_model_genes_present']} model genes overlap."
                )
            header = pd.read_csv(prediction_path, nrows=0)
            require_columns(
                header,
                ["cell_id", "cluster", "celltypist_raw_label", "celltypist_majority_vote"],
                prediction_path,
            )
            validation.update(overlap)
            validation["step05_predictions"] = str(prediction_path)
            validation_rows.append(validation)
        finally:
            source.file.close()
    validation_table = pd.DataFrame(validation_rows)
    print(validation_table[[
        "dataset_folder", "condition", "n_source_cells", "n_source_donors",
        "regions", "n_selected_source_clusters", "n_selected_EC_cells",
        "n_model_genes_present",
    ]].to_string(index=False))
    print(
        f"Selected total: {validation_table['n_selected_EC_cells'].sum():,} cells "
        f"from {validation_table['n_selected_source_clusters'].sum():,} parent clusters."
    )
    if args.validate_only:
        print("Validation complete; no Step 06 outputs were created.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    validation_table.to_csv(output_root / "06_input_validation.csv", index=False)
    selection.to_csv(output_root / "06_EC_source_cluster_selection.csv", index=False)
    pd.DataFrame(
        [{"celltypist_label": label, "color": color} for label, color in celltype_colors.items()]
    ).to_csv(output_root / "06_celltypist_color_key.csv", index=False)
    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "step03_root": str(step03_root),
        "step05_root": str(step05_root),
        "step05_summary": str(step05_summary),
        "output_root": str(output_root),
        "model": str(model_path),
        "expected_cohorts": [list(item) for item in EXPECTED_COHORTS],
        "selected_cohorts": master["cohort"].tolist(),
        "forbidden_datasets": sorted(FORBIDDEN_DATASETS),
        "region_rule": "LV_AP_except_dataset11_septum_allowed",
        "selection_rule": "step05_celltypist_label_is_endothelial_true",
        "LEC_selection_rule": "fresh_EC_only_exact_majority_vote_EC8_ln_or_lymphatic",
        "all_genes_retained_in_X": True,
        **{
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }
    with (output_root / "06_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    outputs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for _, row in master.iterrows():
        dataset = str(row["dataset_folder"])
        condition = str(row["condition"])
        cohort_selection = selection.loc[
            selection["dataset_folder"].eq(dataset) & selection["condition"].eq(condition)
        ].copy()
        try:
            outputs.append(process_cohort(
                row, cohort_selection, project_dir, step05_root, output_root,
                model, model_features, celltype_colors, args,
            ))
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            print(f"[{cohort_name(dataset, condition)}] ERROR: {message}")
            traceback.print_exc()
            errors.append({
                "script_version": SCRIPT_VERSION,
                "dataset_folder": dataset,
                "condition": condition,
                "cohort": cohort_name(dataset, condition),
                "status": "error",
                "message": message,
            })
            gc.collect()

    summaries = [item["summary"] for item in outputs] + errors
    run_summary = pd.DataFrame(summaries)
    run_summary.to_csv(output_root / "06_EC_recluster_run_summary.csv", index=False)
    combined: dict[str, tuple[str, bool]] = {
        "counts": ("06_EC_recluster_counts_all_cohorts.csv", False),
        "marker_summary": ("06_EC_recluster_marker_summary_all_cohorts.csv", True),
        "marker_long": (f"06_EC_recluster_top{args.top_n_markers}_markers_long.csv", True),
        "marker_wide": (f"06_EC_recluster_top{args.top_n_markers}_markers_wide.csv", True),
        "clusters": ("06_EC_celltypist_cluster_summary_all_cohorts.csv", True),
        "distribution": ("06_EC_celltypist_label_distribution_all_cohorts.csv", False),
        "lec_clusters": ("06_final_LEC_candidate_clusters_all_cohorts.csv", True),
    }
    combined_tables: dict[str, pd.DataFrame] = {}
    for key, (filename, sort_columns) in combined.items():
        frames = [item[key] for item in outputs if not item[key].empty]
        table = pd.concat(frames, ignore_index=True, sort=sort_columns) if frames else pd.DataFrame()
        combined_tables[key] = table
        if not table.empty:
            table.to_csv(output_root / filename, index=False)

    figure_manifest = [figure for item in outputs for figure in item["figures"]]
    if not args.skip_figures and outputs:
        try:
            figure_manifest.extend(save_combined_figures(
                run_summary,
                combined_tables["marker_summary"],
                combined_tables["clusters"],
                output_root,
                celltype_colors,
                args.dpi,
            ))
        except Exception as error:
            warnings.warn(f"Combined figures failed after scientific outputs were saved: {error}")
    if figure_manifest:
        pd.DataFrame(figure_manifest).to_csv(output_root / "06_figure_manifest.csv", index=False)

    display = [
        column for column in [
            "cohort", "status", "n_EC_cells", "n_EC_only_leiden_clusters",
            "selected_LEC_clusters", "n_LEC_cells", "percent_LEC_among_EC", "message",
        ] if column in run_summary
    ]
    print("\nStep 06 summary:")
    print(run_summary[display].to_string(index=False))
    print(f"Outputs: {output_root}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
