#!/usr/bin/env python
r"""
Step 05: cluster markers and CellTypist annotation for selected disease cohorts.

The workflow follows aging Step 05 while remaining practical for the larger
disease datasets. For every approved Step 03 dataset-condition cohort it:

1. validates the exact 7-cohort allowlist (dataset02, dataset06, dataset03,
   RV-only dataset10, dataset12 COVID, and dataset08 NICM are excluded);
2. calculates positive Leiden-cluster markers on a reproducible,
   cluster-stratified sample (50,000 cells per cohort by default);
3. runs the local Healthy Adult Heart CellTypist model on every QC-passed cell
   in memory-safe chunks;
4. assigns one CellTypist majority-vote label to each Leiden cluster using all
   cell-level predictions, not only the marker sample;
5. combines CellTypist votes, confidence, top markers, and endothelial/LEC
   marker expression in full CSV summaries;
6. creates per-cohort UMAPs and brief combined figures.

The Step 03 h5ad files are read-only. Annotated h5ad copies are not written;
per-cell predictions are saved as compressed CSV files.

Default input
-------------
new_result2_disease/03/03_master_run_summary.csv

Default output
--------------
new_result2_disease/05_marker_celltypist/

PowerShell
----------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\05_find_markers_and_celltypist_selected_disease.py

Read-only validation
--------------------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\05_find_markers_and_celltypist_selected_disease.py `
      --validate_only
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import traceback
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
    raise ImportError(
        "scanpy is required. Run Step 05 with the project .venv."
    ) from error

try:
    import celltypist
except ImportError as error:
    raise ImportError(
        "celltypist is required. Run Step 05 with the project .venv."
    ) from error


SCRIPT_VERSION = "05_selected_disease_exclude_dataset02_dataset06_dataset12_2026_08_11"
DEFAULT_STEP03_ROOT = "new_result2_disease/03"
DEFAULT_OUTPUT_ROOT = "new_result2_disease/05_marker_celltypist"
DEFAULT_MODEL = "resources/celltypist/Healthy_Adult_Heart.pkl"
DEFAULT_CLUSTER_KEY = "leiden"
DEFAULT_TOP_N = 25
DEFAULT_MARKER_METHOD = "wilcoxon"
DEFAULT_MAX_MARKER_CELLS = 50000
DEFAULT_CELLTYPIST_CHUNK_SIZE = 10000
DEFAULT_MAX_UMAP_CELLS = 30000

EXPECTED_COHORTS = (
    ("dataset01", "DCM"),
    ("dataset01", "HCM"),
    ("dataset01", "NF"),
    ("dataset08", "DoxCM"),
    ("dataset08", "NF"),
    ("dataset11", "HFpEF"),
    ("dataset11", "NF"),
)

ENDOTHELIAL_MARKERS = [
    "PECAM1", "CDH5", "VWF", "CLDN5", "KDR", "FLT1", "EMCN", "RAMP2", "ESAM"
]
LEC_MARKERS = [
    "PROX1", "FLT4", "PDPN", "LYVE1", "CCL21", "MMRN1", "NRP2", "MRC1"
]
NON_ENDOTHELIAL_MARKERS = [
    "TNNT2", "TNNI3", "ACTC1", "MYH6", "MYH7", "TTN", "MYBPC3",
    "DCN", "LUM", "COL1A1", "COL1A2", "PDGFRA",
    "RGS5", "PDGFRB", "ABCC9", "KCNJ8", "CSPG4",
    "PTPRC", "LST1", "CD68", "CD3D", "NKG7",
]
MARKER_EXPRESSION_GENES = list(
    dict.fromkeys(ENDOTHELIAL_MARKERS + LEC_MARKERS + NON_ENDOTHELIAL_MARKERS)
)


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value))
    return re.sub(r"_+", "_", text).strip("_") or "NA"


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("Missing").str.strip()


def sort_labels(values) -> list[str]:
    def key(value: object) -> tuple[int, object]:
        text = str(value)
        return (0, int(text)) if text.isdigit() else (1, text.casefold())
    return sorted({str(value) for value in values}, key=key)


def cohort_name(dataset: str, condition: str) -> str:
    return f"{dataset} | {condition}"


def cohort_prefix(dataset: str, condition: str) -> str:
    return f"{dataset}_{safe_name(condition)}"


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in {source}: {missing}")


def parse_cohort_subset(text: str) -> set[tuple[str, str]] | None:
    if str(text).strip().casefold() in {"", "all"}:
        return None
    requested: set[tuple[str, str]] = set()
    for token in re.split(r"[,;]", str(text)):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid cohort {token!r}; use dataset:condition, for example dataset08:NF."
            )
        dataset, condition = [part.strip() for part in token.split(":", 1)]
        requested.add((dataset, condition))
    invalid = requested - set(EXPECTED_COHORTS)
    if invalid:
        raise ValueError(f"Unapproved cohort subset: {sorted(invalid)}")
    return requested


def validate_master(master: pd.DataFrame, source: Path) -> pd.DataFrame:
    require_columns(
        master,
        [
            "dataset_folder", "dataset_id", "condition", "status",
            "n_cells_after_all_QC", "n_donors_after_QC", "n_clusters",
            "output_h5ad",
        ],
        source,
    )
    master = master.copy()
    master["dataset_folder"] = normalized_text(master["dataset_folder"])
    master["condition"] = normalized_text(master["condition"])
    forbidden = master["dataset_folder"].isin(
        {"dataset02", "dataset03", "dataset06", "dataset10", "dataset12"}
    )
    if forbidden.any():
        raise ValueError(
            "Excluded datasets are present in Step 03: "
            f"{sorted(master.loc[forbidden, 'dataset_folder'].unique())}"
        )
    if master["status"].astype(str).str.casefold().eq("error").any():
        failed = master.loc[
            master["status"].astype(str).str.casefold().eq("error"),
            ["dataset_folder", "condition"],
        ]
        raise ValueError("Step 03 has failed cohorts:\n" + failed.to_string(index=False))
    observed = set(zip(master["dataset_folder"], master["condition"]))
    expected = set(EXPECTED_COHORTS)
    if observed != expected:
        raise ValueError(
            f"Step 03 cohort mismatch. Missing={sorted(expected-observed)}; "
            f"unexpected={sorted(observed-expected)}"
        )
    if master.duplicated(["dataset_folder", "condition"]).any():
        raise ValueError("Step 03 master summary has duplicate cohort rows.")
    order = {pair: index for index, pair in enumerate(EXPECTED_COHORTS)}
    master["cohort_order"] = [
        order[(dataset, condition)]
        for dataset, condition in zip(master["dataset_folder"], master["condition"])
    ]
    master["cohort"] = [
        cohort_name(dataset, condition)
        for dataset, condition in zip(master["dataset_folder"], master["condition"])
    ]
    return master.sort_values("cohort_order").reset_index(drop=True)


def resolve_h5ad(project_dir: Path, row: pd.Series) -> Path:
    path = resolve_path(project_dir, str(row["output_h5ad"]))
    if not path.exists():
        raise FileNotFoundError(f"Missing Step 03 h5ad: {path}")
    return path


def validate_h5ad(
    source,
    path: Path,
    dataset: str,
    condition: str,
    expected_cells: int,
    expected_clusters: int,
) -> str:
    if int(source.n_obs) != int(expected_cells):
        raise ValueError(
            f"Cell-count mismatch for {dataset}/{condition}: expected "
            f"{expected_cells:,}, found {source.n_obs:,}."
        )
    cluster_key = DEFAULT_CLUSTER_KEY
    if cluster_key not in source.obs.columns:
        raise KeyError(f"{cluster_key!r} is absent from {path}")
    require_columns(source.obs, ["donor_key", "disease", cluster_key], path)
    observed_conditions = set(normalized_text(source.obs["disease"]).unique())
    if observed_conditions != {condition}:
        raise ValueError(
            f"Condition mismatch in {path}: expected {condition}, "
            f"found {sorted(observed_conditions)}"
        )
    n_clusters = normalized_text(source.obs[cluster_key]).nunique()
    if int(n_clusters) != int(expected_clusters):
        raise ValueError(
            f"Cluster-count mismatch for {dataset}/{condition}: master has "
            f"{expected_clusters}, h5ad has {n_clusters}."
        )
    if "X_umap" not in source.obsm:
        raise KeyError(f"X_umap is absent from {path}")
    return cluster_key


def stratified_indices(
    clusters: pd.Series,
    maximum: int,
    seed: int,
    minimum_per_cluster: int = 25,
) -> np.ndarray:
    labels = normalized_text(clusters).to_numpy()
    if maximum <= 0 or len(labels) <= maximum:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sort_labels(pd.unique(labels)):
        indices = np.flatnonzero(labels == label)
        take = min(minimum_per_cluster, len(indices))
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    if len(selected) > maximum:
        raise ValueError(
            f"Maximum {maximum} is too small to retain {minimum_per_cluster} "
            "cells per cluster."
        )
    remaining = np.setdiff1d(
        np.arange(len(labels), dtype=np.int64),
        np.asarray(selected, dtype=np.int64),
        assume_unique=False,
    )
    extra = min(maximum - len(selected), len(remaining))
    if extra:
        selected.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def rows_to_memory(source, row_indices: np.ndarray) -> ad.AnnData:
    """Read selected X rows without loading the duplicate .raw matrix."""
    matrix = source.X[row_indices, :]
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
    else:
        matrix = np.asarray(matrix)
    result = ad.AnnData(
        X=matrix,
        obs=source.obs.iloc[row_indices].copy(),
        var=source.var.copy(),
    )
    result.obs_names = source.obs_names[row_indices].astype(str)
    result.var_names = source.var_names.astype(str)
    result.var_names_make_unique()
    return result


def model_overlap(source, model_features: pd.Index) -> dict[str, object]:
    query = pd.Index(source.var_names.astype(str))
    model_upper = set(model_features.str.upper())
    overlap = sorted(set(query.str.upper()) & model_upper)
    return {
        "n_query_genes": int(source.n_vars),
        "n_model_genes": int(len(model_features)),
        "n_model_genes_present": int(len(overlap)),
        "model_gene_overlap_fraction": float(len(overlap) / len(model_features)),
        "overlap_gene_examples": "; ".join(overlap[:50]),
    }


def make_cluster_counts(
    source,
    dataset: str,
    condition: str,
    cluster_key: str,
) -> pd.DataFrame:
    counts = (
        normalized_text(source.obs[cluster_key])
        .value_counts()
        .rename_axis("cluster")
        .reset_index(name="n_cells")
    )
    order = {label: index for index, label in enumerate(sort_labels(counts["cluster"]))}
    counts["_order"] = counts["cluster"].map(order)
    counts = counts.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    counts.insert(0, "condition", condition)
    counts.insert(0, "dataset_folder", dataset)
    counts.insert(2, "cohort", cohort_name(dataset, condition))
    return counts


def rank_markers(
    marker_data: ad.AnnData,
    cluster_key: str,
    method: str,
    n_rank_genes: int,
) -> pd.DataFrame:
    marker_data.obs[cluster_key] = normalized_text(
        marker_data.obs[cluster_key]
    ).astype("category")
    available = int(marker_data.n_vars)
    n_genes = available if n_rank_genes <= 0 else min(n_rank_genes, available)
    sc.tl.rank_genes_groups(
        marker_data,
        groupby=cluster_key,
        reference="rest",
        method=method,
        n_genes=n_genes,
        use_raw=False,
        pts=True,
        rankby_abs=False,
    )
    tables: list[pd.DataFrame] = []
    groups = list(marker_data.uns["rank_genes_groups"]["names"].dtype.names)
    for group in groups:
        table = sc.get.rank_genes_groups_df(marker_data, group=group)
        table.insert(0, "cluster", str(group))
        tables.append(table)
    ranked = pd.concat(tables, ignore_index=True).rename(
        columns={"names": "gene", "logfoldchanges": "logfoldchange"}
    )
    return ranked


def select_top_markers(
    ranked: pd.DataFrame,
    dataset: str,
    condition: str,
    top_n: int,
    positive_only: bool,
) -> pd.DataFrame:
    markers = ranked.copy()
    markers["gene"] = normalized_text(markers["gene"])
    markers = markers.loc[
        ~markers["gene"].str.upper().isin({"", "NAN", "NONE", "UNKNOWN", "MISSING"})
    ].copy()
    if positive_only and "logfoldchange" in markers.columns:
        markers = markers.loc[
            markers["logfoldchange"].isna() | markers["logfoldchange"].gt(0)
        ].copy()
    sort_columns = [column for column in ["cluster", "scores", "pvals_adj"] if column in markers]
    ascending = [column in {"cluster", "pvals_adj"} for column in sort_columns]
    markers = markers.sort_values(sort_columns, ascending=ascending)
    markers = markers.groupby("cluster", group_keys=False).head(top_n).copy()
    markers["marker_rank"] = markers.groupby("cluster").cumcount() + 1
    gene_upper = markers["gene"].str.upper()
    markers["is_endothelial_marker"] = gene_upper.isin(ENDOTHELIAL_MARKERS)
    markers["is_LEC_marker"] = gene_upper.isin(LEC_MARKERS)
    markers["is_common_non_endothelial_marker"] = gene_upper.isin(
        NON_ENDOTHELIAL_MARKERS
    )
    markers.insert(0, "cohort", cohort_name(dataset, condition))
    markers.insert(0, "condition", condition)
    markers.insert(0, "dataset_folder", dataset)
    front = [
        "dataset_folder", "condition", "cohort", "cluster", "marker_rank", "gene"
    ]
    return markers[front + [column for column in markers if column not in front]]


def marker_wide(top: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster in sort_labels(top["cluster"].unique()):
        subset = top[top["cluster"].astype(str).eq(cluster)].sort_values("marker_rank")
        genes = subset["gene"].astype(str).tolist()
        row: dict[str, object] = {
            "dataset_folder": str(subset["dataset_folder"].iloc[0]),
            "condition": str(subset["condition"].iloc[0]),
            "cohort": str(subset["cohort"].iloc[0]),
            "cluster": cluster,
            "top_marker_genes_semicolon": "; ".join(genes),
        }
        for index in range(top_n):
            row[f"marker_{index + 1}"] = genes[index] if index < len(genes) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def marker_expression_summary(
    marker_data: ad.AnnData,
    cluster_key: str,
) -> pd.DataFrame:
    names = pd.Index(marker_data.var_names.astype(str))
    lookup = {gene.upper(): index for index, gene in enumerate(names)}
    present = [gene for gene in MARKER_EXPRESSION_GENES if gene in lookup]
    clusters = normalized_text(marker_data.obs[cluster_key]).to_numpy()
    rows: list[dict[str, object]] = []
    if not present:
        return pd.DataFrame({"cluster": sort_labels(pd.unique(clusters))})
    positions = [lookup[gene] for gene in present]
    expression = marker_data.X[:, positions]
    if sp.issparse(expression):
        expression = expression.tocsr()
    for cluster in sort_labels(pd.unique(clusters)):
        mask = clusters == cluster
        block = expression[mask]
        if sp.issparse(block):
            means = np.asarray(block.mean(axis=0)).ravel()
            percentages = np.asarray((block > 0).mean(axis=0)).ravel()
        else:
            block = np.asarray(block)
            means = block.mean(axis=0)
            percentages = (block > 0).mean(axis=0)
        row: dict[str, object] = {
            "cluster": cluster,
            "n_cells_marker_summary": int(mask.sum()),
        }
        for gene, mean_value, pct_value in zip(present, means, percentages):
            row[f"mean_expr_{gene}"] = float(mean_value)
            row[f"pct_expr_{gene}"] = float(pct_value)
        for set_name, genes in [
            ("endothelial", ENDOTHELIAL_MARKERS),
            ("LEC", LEC_MARKERS),
            ("negative_check", NON_ENDOTHELIAL_MARKERS),
        ]:
            available = [gene for gene in genes if gene in present]
            row[f"{set_name}_markers_present"] = "; ".join(available)
            row[f"n_{set_name}_markers_present"] = len(available)
            row[f"{set_name}_marker_pct_expr_mean"] = (
                float(np.mean([row[f"pct_expr_{gene}"] for gene in available]))
                if available else np.nan
            )
            row[f"{set_name}_marker_score_mean"] = (
                float(np.mean([row[f"mean_expr_{gene}"] for gene in available]))
                if available else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_frame(prediction, query: ad.AnnData) -> pd.DataFrame:
    labels = prediction.predicted_labels.copy()
    if isinstance(labels, pd.Series):
        labels = labels.to_frame("predicted_labels")
    labels.index = labels.index.astype(str)
    raw_column = "predicted_labels" if "predicted_labels" in labels else labels.columns[0]
    result = pd.DataFrame(index=query.obs_names.astype(str))
    result["celltypist_raw_label"] = normalized_text(
        labels[raw_column].reindex(result.index)
    ).to_numpy()
    probability = prediction.probability_matrix
    if isinstance(probability, pd.DataFrame):
        probability.index = probability.index.astype(str)
        result["celltypist_max_probability"] = (
            probability.max(axis=1).reindex(result.index).to_numpy()
        )
        result["celltypist_probability_best_label"] = normalized_text(
            probability.idxmax(axis=1).reindex(result.index)
        ).to_numpy()
    return result


def predict_all_cells_chunked(
    source,
    dataset: str,
    condition: str,
    cluster_key: str,
    model,
    chunk_size: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for start in range(0, source.n_obs, chunk_size):
        stop = min(start + chunk_size, source.n_obs)
        print(f"  CellTypist cells {start + 1:,}-{stop:,} / {source.n_obs:,}")
        rows = np.arange(start, stop, dtype=np.int64)
        query = rows_to_memory(source, rows)
        prediction = celltypist.annotate(
            query,
            model=model,
            majority_voting=False,
        )
        table = prediction_frame(prediction, query)
        table.insert(0, "cell_id", table.index)
        table.insert(1, "source_row_index", rows)
        table.insert(2, "dataset_folder", dataset)
        table.insert(3, "condition", condition)
        table.insert(4, "cohort", cohort_name(dataset, condition))
        table.insert(
            5,
            "cluster",
            normalized_text(source.obs.iloc[rows][cluster_key]).to_numpy(),
        )
        table.insert(
            6,
            "donor_key",
            normalized_text(source.obs.iloc[rows]["donor_key"]).to_numpy(),
        )
        parts.append(table.reset_index(drop=True))
        del prediction, query, table
        gc.collect()
    result = pd.concat(parts, ignore_index=True)
    if len(result) != source.n_obs or result["cell_id"].duplicated().any():
        raise ValueError("Chunked CellTypist predictions failed cell coverage validation.")
    return result


def majority_vote_tables(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distribution = (
        cells.groupby(["cluster", "celltypist_raw_label"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    cluster_sizes = cells["cluster"].value_counts().rename("cluster_n_cells")
    distribution["fraction_within_cluster"] = distribution["n_cells"] / distribution[
        "cluster"
    ].map(cluster_sizes)
    distribution = distribution.sort_values(
        ["cluster", "n_cells", "celltypist_raw_label"],
        ascending=[True, False, True],
    )
    winners = distribution.groupby("cluster", observed=True).first().reset_index()
    second = distribution.groupby("cluster", observed=True).nth(1).reset_index()
    winner_map = winners.set_index("cluster")["celltypist_raw_label"]
    cells["celltypist_majority_vote"] = cells["cluster"].map(winner_map)

    cluster_rows: list[dict[str, object]] = []
    second_map = second.set_index("cluster") if not second.empty else pd.DataFrame()
    for _, winner in winners.iterrows():
        cluster = str(winner["cluster"])
        subset = cells[cells["cluster"].astype(str).eq(cluster)]
        confidence = pd.to_numeric(
            subset.get("celltypist_max_probability"), errors="coerce"
        )
        if not second.empty and cluster in second_map.index:
            second_row = second_map.loc[cluster]
            second_label = str(second_row["celltypist_raw_label"])
            second_fraction = float(second_row["fraction_within_cluster"])
        else:
            second_label = ""
            second_fraction = 0.0
        cluster_rows.append(
            {
                "cluster": cluster,
                "n_cells": int(len(subset)),
                "cluster_majority_cell_type": str(winner["celltypist_raw_label"]),
                "majority_vote_support_fraction": float(winner["fraction_within_cluster"]),
                "raw_prediction_majority_label": str(winner["celltypist_raw_label"]),
                "raw_prediction_majority_fraction": float(winner["fraction_within_cluster"]),
                "second_raw_prediction_label": second_label,
                "second_raw_prediction_fraction": second_fraction,
                "mean_celltypist_confidence": float(confidence.mean()),
                "median_celltypist_confidence": float(confidence.median()),
                "n_distinct_raw_cell_types": int(
                    subset["celltypist_raw_label"].nunique()
                ),
            }
        )
    cluster_predictions = pd.DataFrame(cluster_rows)
    celltype_counts = (
        cells["celltypist_majority_vote"]
        .value_counts()
        .rename_axis("cell_type")
        .reset_index(name="n_cells")
    )
    celltype_counts["percent_cells"] = 100.0 * celltype_counts["n_cells"] / len(cells)
    return cluster_predictions, distribution, celltype_counts


def merge_cluster_evidence(
    dataset: str,
    condition: str,
    predictions: pd.DataFrame,
    wide: pd.DataFrame,
    expression: pd.DataFrame,
) -> pd.DataFrame:
    summary = predictions.merge(
        wide[["cluster", "top_marker_genes_semicolon"]], on="cluster", how="left"
    ).merge(expression, on="cluster", how="left")
    summary.insert(0, "cohort", cohort_name(dataset, condition))
    summary.insert(0, "condition", condition)
    summary.insert(0, "dataset_folder", dataset)
    labels = summary["cluster_majority_cell_type"].astype(str)
    summary["celltypist_label_is_endothelial"] = labels.str.contains(
        "EC|endothelial|endocardial", case=False, regex=True
    )
    summary["celltypist_label_is_lymphatic"] = labels.str.contains(
        "lymph|EC8_ln", case=False, regex=True
    )
    order = {label: index for index, label in enumerate(sort_labels(summary["cluster"]))}
    summary["_order"] = summary["cluster"].map(order)
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def stable_label_colors(model_labels: list[str]) -> dict[str, str]:
    labels = sorted(set(map(str, model_labels)), key=str.casefold)
    cmap = plt.get_cmap("gist_ncar")
    colors = {
        label: to_hex(cmap(index / max(len(labels), 1)))
        for index, label in enumerate(labels)
    }
    colors["Other"] = "#B7B7B7"
    colors["Missing"] = "#B7B7B7"
    return colors


def save_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_cohort_umap(
    source,
    cells: pd.DataFrame,
    dataset: str,
    condition: str,
    label_colors: dict[str, str],
    maximum: int,
    seed: int,
    path: Path,
    dpi: int,
) -> None:
    indices = stratified_indices(cells["cluster"], maximum, seed, minimum_per_cluster=50)
    coordinates = np.asarray(source.obsm["X_umap"])[indices]
    labels = cells.iloc[indices]["celltypist_majority_vote"].astype(str).to_numpy()
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    present = sorted(pd.unique(labels), key=str.casefold)
    for label in present:
        mask = labels == label
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=2.0,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
            color=label_colors.get(label, "#B7B7B7"),
            label=label,
        )
    ax.set_title(
        f"{dataset} | {condition}: CellTypist Leiden-cluster majority vote\n"
        f"{len(indices):,} displayed of {source.n_obs:,} cells"
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
    save_figure(path, dpi)


def plot_celltype_composition(
    counts: pd.DataFrame,
    cohort_order: list[str],
    label_colors: dict[str, str],
    path: Path,
    dpi: int,
    top_labels: int = 18,
) -> None:
    totals = counts.groupby("cell_type", observed=True)["n_cells"].sum().sort_values(
        ascending=False
    )
    keep = set(totals.head(top_labels).index.astype(str))
    plot = counts.copy()
    plot["plot_label"] = plot["cell_type"].astype(str).where(
        plot["cell_type"].astype(str).isin(keep), "Other"
    )
    grouped = (
        plot.groupby(["cohort", "plot_label"], observed=True)["n_cells"]
        .sum()
        .unstack(fill_value=0)
        .reindex(cohort_order, fill_value=0)
    )
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100.0
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    bottom = np.zeros(len(percentages), dtype=float)
    columns = sorted(percentages.columns, key=str.casefold)
    for label in columns:
        values = percentages[label].to_numpy(float)
        ax.bar(
            np.arange(len(percentages)),
            values,
            bottom=bottom,
            color=label_colors.get(str(label), "#B7B7B7"),
            label=str(label),
        )
        bottom += values
    ax.set_xticks(np.arange(len(percentages)), percentages.index, rotation=45, ha="right")
    ax.set_ylabel("Cells (%)")
    ax.set_title("CellTypist majority-vote cell-type composition by disease cohort")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
    save_figure(path, dpi)


def plot_vote_heatmap(
    distribution: pd.DataFrame,
    path: Path,
    dpi: int,
    top_labels: int = 25,
) -> None:
    distribution = distribution.copy()
    distribution["row"] = (
        distribution["cohort"].astype(str)
        + " | cluster "
        + distribution["cluster"].astype(str)
    )
    top = (
        distribution.groupby("celltypist_raw_label", observed=True)["n_cells"]
        .sum()
        .nlargest(top_labels)
        .index
    )
    subset = distribution[distribution["celltypist_raw_label"].isin(top)]
    matrix = subset.pivot_table(
        index="row",
        columns="celltypist_raw_label",
        values="fraction_within_cluster",
        aggfunc="sum",
        fill_value=0,
    )
    height = min(34, max(8, 0.22 * len(matrix)))
    fig, ax = plt.subplots(figsize=(14, height))
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index, fontsize=6)
    ax.set_title("Raw CellTypist prediction fractions within each Leiden cluster")
    fig.colorbar(image, ax=ax, label="Fraction within cluster", shrink=0.35)
    save_figure(path, dpi)


def plot_marker_heatmap(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    columns = [
        f"pct_expr_{gene}"
        for gene in ENDOTHELIAL_MARKERS + LEC_MARKERS
        if f"pct_expr_{gene}" in summary.columns
    ]
    if not columns:
        return
    rows = (
        summary["cohort"].astype(str)
        + " | cluster "
        + summary["cluster"].astype(str)
    )
    values = summary[columns].fillna(0).to_numpy(float)
    height = min(34, max(8, 0.22 * len(summary)))
    fig, ax = plt.subplots(figsize=(12.5, height))
    image = ax.imshow(values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(
        np.arange(len(columns)),
        [column.replace("pct_expr_", "") for column in columns],
        rotation=90,
    )
    ax.set_yticks(np.arange(len(rows)), rows, fontsize=6)
    ax.set_title("Endothelial and LEC marker detection by Leiden cluster")
    fig.colorbar(image, ax=ax, label="Fraction expressing", shrink=0.35)
    save_figure(path, dpi)


def add_context_columns(
    table: pd.DataFrame,
    dataset: str,
    condition: str,
) -> pd.DataFrame:
    table = table.copy()
    table.insert(0, "cohort", cohort_name(dataset, condition))
    table.insert(0, "condition", condition)
    table.insert(0, "dataset_folder", dataset)
    return table


def process_cohort(
    row: pd.Series,
    project_dir: Path,
    output_root: Path,
    model,
    model_features: pd.Index,
    label_colors: dict[str, str],
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, pd.DataFrame], list[dict[str, str]]]:
    dataset = str(row["dataset_folder"])
    condition = str(row["condition"])
    cohort = cohort_name(dataset, condition)
    prefix = cohort_prefix(dataset, condition)
    outdir = output_root / dataset / safe_name(condition)
    figures_dir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    error_path = outdir / f"{prefix}_step05_ERROR.txt"
    source_path = resolve_h5ad(project_dir, row)

    print("\n" + "=" * 100)
    print(f"Step 05 {cohort}: {source_path}")
    print("Output:", outdir)
    print("=" * 100)
    source = sc.read_h5ad(source_path, backed="r")
    try:
        cluster_key = validate_h5ad(
            source,
            source_path,
            dataset,
            condition,
            int(row["n_cells_after_all_QC"]),
            int(row["n_clusters"]),
        )
        overlap = model_overlap(source, model_features)
        if overlap["n_model_genes_present"] < args.min_model_genes_present:
            raise ValueError(
                f"Only {overlap['n_model_genes_present']} model genes are present; "
                f"minimum is {args.min_model_genes_present}."
            )
        print(
            "Model-gene overlap: "
            f"{overlap['n_model_genes_present']:,}/{overlap['n_model_genes']:,} "
            f"({overlap['model_gene_overlap_fraction']:.1%})"
        )

        full_cluster_counts = make_cluster_counts(
            source, dataset, condition, cluster_key
        )
        full_cluster_counts.to_csv(
            outdir / f"{prefix}_cluster_counts.csv", index=False
        )

        marker_indices = stratified_indices(
            source.obs[cluster_key],
            args.max_marker_cells,
            args.random_seed + int(row["cohort_order"]),
        )
        print(
            f"Marker ranking sample: {len(marker_indices):,}/{source.n_obs:,} cells"
        )
        marker_data = rows_to_memory(source, marker_indices)
        expression = marker_expression_summary(marker_data, cluster_key)
        ranked = rank_markers(
            marker_data,
            cluster_key,
            args.method,
            args.n_rank_genes,
        )
        if args.save_all_ranked:
            add_context_columns(ranked, dataset, condition).to_csv(
                outdir / f"{prefix}_ALL_ranked_genes.csv.gz",
                index=False,
                compression="gzip",
            )
        top = select_top_markers(
            ranked,
            dataset,
            condition,
            args.top_n,
            args.positive_only,
        )
        wide = marker_wide(top, args.top_n)
        top_path = outdir / f"{prefix}_top{args.top_n}_markers_long.csv"
        wide_path = outdir / f"{prefix}_top{args.top_n}_markers_wide.csv"
        top.to_csv(top_path, index=False)
        wide.to_csv(wide_path, index=False)
        del ranked, marker_data
        gc.collect()

        cells = predict_all_cells_chunked(
            source,
            dataset,
            condition,
            cluster_key,
            model,
            args.celltypist_chunk_size,
        )
        cluster_predictions, distribution, celltype_counts = majority_vote_tables(cells)
        distribution = add_context_columns(distribution, dataset, condition)
        celltype_counts = add_context_columns(celltype_counts, dataset, condition)
        cluster_summary = merge_cluster_evidence(
            dataset,
            condition,
            cluster_predictions,
            wide,
            expression,
        )

        cell_path = outdir / f"{prefix}_celltypist_cell_predictions.csv.gz"
        if not args.skip_cell_predictions:
            cells.to_csv(cell_path, index=False, compression="gzip")
        cluster_summary_path = outdir / f"{prefix}_cluster_annotation_summary.csv"
        cluster_summary.to_csv(cluster_summary_path, index=False)
        distribution.to_csv(
            outdir / f"{prefix}_celltypist_label_distribution_by_cluster.csv",
            index=False,
        )
        celltype_counts.to_csv(
            outdir / f"{prefix}_celltypist_celltype_counts.csv", index=False
        )
        overlap_frame = pd.DataFrame(
            [{
                "dataset_folder": dataset,
                "condition": condition,
                "cohort": cohort,
                **overlap,
            }]
        )
        overlap_frame.to_csv(
            outdir / f"{prefix}_celltypist_model_gene_overlap.csv", index=False
        )

        umap_path = figures_dir / f"{prefix}_umap_celltypist_majority_vote.png"
        plot_cohort_umap(
            source,
            cells,
            dataset,
            condition,
            label_colors,
            args.max_umap_cells,
            args.random_seed + 5000 + int(row["cohort_order"]),
            umap_path,
            args.dpi,
        )
        figure_rows = [
            {
                "scope": cohort,
                "figure": str(umap_path),
                "description": "UMAP colored by full-cohort CellTypist Leiden-cluster majority vote",
            }
        ]

        summary = {
            "script_version": SCRIPT_VERSION,
            "dataset_folder": dataset,
            "condition": condition,
            "cohort": cohort,
            "status": "completed",
            "source_h5ad": str(source_path),
            "model_path": str(args.model_resolved),
            "cluster_key": cluster_key,
            "marker_method": args.method,
            "top_n": int(args.top_n),
            "n_cells_input": int(source.n_obs),
            "n_cells_celltypist": int(len(cells)),
            "n_cells_marker_ranking": int(len(marker_indices)),
            "marker_analysis_is_subsample": bool(len(marker_indices) < source.n_obs),
            "n_genes_query": int(source.n_vars),
            "n_clusters": int(len(full_cluster_counts)),
            "n_model_genes": overlap["n_model_genes"],
            "n_model_genes_present": overlap["n_model_genes_present"],
            "model_gene_overlap_fraction": overlap["model_gene_overlap_fraction"],
            "marker_long_csv": str(top_path),
            "marker_wide_csv": str(wide_path),
            "cluster_annotation_summary_csv": str(cluster_summary_path),
            "cell_predictions_csv": "" if args.skip_cell_predictions else str(cell_path),
        }
        pd.DataFrame([summary]).to_csv(
            outdir / f"{prefix}_step05_run_summary.csv", index=False
        )
        if error_path.exists():
            error_path.unlink()
        tables = {
            "top": top,
            "wide": wide,
            "cluster_summary": cluster_summary,
            "distribution": distribution,
            "celltype_counts": celltype_counts,
            "overlap": overlap_frame,
        }
        del cells
        gc.collect()
        return summary, tables, figure_rows
    finally:
        source.file.close()


def existing_cohort_outputs(
    row: pd.Series,
    output_root: Path,
    top_n: int,
) -> tuple[dict[str, object], dict[str, pd.DataFrame], list[dict[str, str]]] | None:
    dataset = str(row["dataset_folder"])
    condition = str(row["condition"])
    prefix = cohort_prefix(dataset, condition)
    outdir = output_root / dataset / safe_name(condition)
    paths = {
        "summary": outdir / f"{prefix}_step05_run_summary.csv",
        "top": outdir / f"{prefix}_top{top_n}_markers_long.csv",
        "wide": outdir / f"{prefix}_top{top_n}_markers_wide.csv",
        "cluster_summary": outdir / f"{prefix}_cluster_annotation_summary.csv",
        "distribution": outdir / f"{prefix}_celltypist_label_distribution_by_cluster.csv",
        "celltype_counts": outdir / f"{prefix}_celltypist_celltype_counts.csv",
        "overlap": outdir / f"{prefix}_celltypist_model_gene_overlap.csv",
    }
    if not all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
        return None
    summary = pd.read_csv(paths["summary"]).iloc[0].to_dict()
    if str(summary.get("status", "")).casefold() != "completed":
        return None
    summary["status"] = "reused_existing_output"
    tables = {
        name: pd.read_csv(path)
        for name, path in paths.items()
        if name != "summary"
    }
    umap_path = (
        outdir / "figures" / f"{prefix}_umap_celltypist_majority_vote.png"
    )
    figures = []
    if umap_path.exists():
        figures.append(
            {
                "scope": cohort_name(dataset, condition),
                "figure": str(umap_path),
                "description": "UMAP colored by CellTypist Leiden-cluster majority vote",
            }
        )
    return summary, tables, figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster markers and chunked Healthy Heart CellTypist annotation."
    )
    parser.add_argument("--step03_root", default=DEFAULT_STEP03_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--cohorts",
        default="all",
        help="all or comma-separated dataset:condition values, e.g. dataset08:NF",
    )
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--method",
        default=DEFAULT_MARKER_METHOD,
        choices=["wilcoxon", "t-test", "t-test_overestim_var", "logreg"],
    )
    parser.add_argument(
        "--n_rank_genes",
        type=int,
        default=0,
        help="Genes ranked per cluster; 0 ranks every available gene.",
    )
    parser.add_argument(
        "--positive_only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_all_ranked", action="store_true")
    parser.add_argument(
        "--max_marker_cells",
        type=int,
        default=DEFAULT_MAX_MARKER_CELLS,
        help="Cluster-stratified cells used for marker ranking; 0 uses all cells.",
    )
    parser.add_argument(
        "--celltypist_chunk_size",
        type=int,
        default=DEFAULT_CELLTYPIST_CHUNK_SIZE,
    )
    parser.add_argument("--max_umap_cells", type=int, default=DEFAULT_MAX_UMAP_CELLS)
    parser.add_argument("--min_model_genes_present", type=int, default=500)
    parser.add_argument("--skip_cell_predictions", action="store_true")
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_n < 1 or args.max_marker_cells < 0:
        raise ValueError("top_n must be positive and max_marker_cells cannot be negative.")
    if args.celltypist_chunk_size < 100 or args.max_umap_cells < 100:
        raise ValueError("CellTypist chunk and UMAP sample sizes must be at least 100.")
    if args.dpi < 72:
        raise ValueError("dpi must be at least 72.")

    project_dir = find_project_dir()
    step03_root = resolve_path(project_dir, args.step03_root)
    output_root = resolve_path(project_dir, args.output_root)
    model_path = resolve_path(project_dir, args.model)
    args.model_resolved = model_path
    master_path = step03_root / "03_master_run_summary.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing Step 03 master summary: {master_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing CellTypist model: {model_path}")
    master = validate_master(pd.read_csv(master_path), master_path)
    subset = parse_cohort_subset(args.cohorts)
    selected = master[
        [
            subset is None or (dataset, condition) in subset
            for dataset, condition in zip(master["dataset_folder"], master["condition"])
        ]
    ].copy()
    if selected.empty:
        raise ValueError("No cohorts selected.")

    model = celltypist.models.Model.load(str(model_path))
    model_features = pd.Index([str(feature) for feature in model.features])
    label_colors = stable_label_colors([str(label) for label in model.cell_types])

    print("=" * 100)
    print("Selected disease Step 05: markers + chunked CellTypist")
    print("Step 03:", step03_root)
    print("Output:", output_root)
    print("Model:", model_path)
    print("Selected cohorts:", ", ".join(selected["cohort"]))
    print("Marker cells per cohort:", args.max_marker_cells or "all")
    print("CellTypist cells: all, chunk size", args.celltypist_chunk_size)
    print("Excluded: dataset02, dataset06, dataset03, RV-only dataset10, dataset12 COVID, and dataset08 NICM")
    print("=" * 100)

    validation_rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        path = resolve_h5ad(project_dir, row)
        source = sc.read_h5ad(path, backed="r")
        try:
            cluster_key = validate_h5ad(
                source,
                path,
                str(row["dataset_folder"]),
                str(row["condition"]),
                int(row["n_cells_after_all_QC"]),
                int(row["n_clusters"]),
            )
            overlap = model_overlap(source, model_features)
            validation_rows.append(
                {
                    "dataset_folder": row["dataset_folder"],
                    "condition": row["condition"],
                    "cohort": row["cohort"],
                    "status": "validated",
                    "source_h5ad": str(path),
                    "n_cells": int(source.n_obs),
                    "n_genes": int(source.n_vars),
                    "n_clusters": int(
                        normalized_text(source.obs[cluster_key]).nunique()
                    ),
                    **overlap,
                }
            )
        finally:
            source.file.close()
    validation = pd.DataFrame(validation_rows)
    if (validation["n_model_genes_present"] < args.min_model_genes_present).any():
        raise ValueError("At least one cohort has insufficient CellTypist model genes.")
    if args.validate_only:
        print(validation.to_string(index=False))
        print("Validation complete; no Step 05 outputs were created.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_root / "05_input_validation.csv", index=False)
    summaries: list[dict[str, object]] = []
    combined: dict[str, list[pd.DataFrame]] = {
        "top": [],
        "wide": [],
        "cluster_summary": [],
        "distribution": [],
        "celltype_counts": [],
        "overlap": [],
    }
    figure_rows: list[dict[str, str]] = []

    for _, row in selected.iterrows():
        dataset = str(row["dataset_folder"])
        condition = str(row["condition"])
        try:
            reused = None
            if args.resume and not args.force:
                reused = existing_cohort_outputs(row, output_root, args.top_n)
            if reused is None:
                summary, tables, figures = process_cohort(
                    row,
                    project_dir,
                    output_root,
                    model,
                    model_features,
                    label_colors,
                    args,
                )
            else:
                print(f"Reusing completed Step 05 output: {dataset}/{condition}")
                summary, tables, figures = reused
            summaries.append(summary)
            for name, table in tables.items():
                combined[name].append(table)
            figure_rows.extend(figures)
        except Exception as error:
            traceback.print_exc()
            outdir = output_root / dataset / safe_name(condition)
            outdir.mkdir(parents=True, exist_ok=True)
            error_path = outdir / f"{cohort_prefix(dataset, condition)}_step05_ERROR.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            summaries.append(
                {
                    "script_version": SCRIPT_VERSION,
                    "dataset_folder": dataset,
                    "condition": condition,
                    "cohort": cohort_name(dataset, condition),
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            gc.collect()
        pd.DataFrame(summaries).to_csv(
            output_root / "05_run_summary_all_cohorts.csv", index=False
        )

    combined_paths = {
        "top": output_root / f"05_all_cohorts_top{args.top_n}_markers_long.csv",
        "wide": output_root / f"05_all_cohorts_top{args.top_n}_markers_wide.csv",
        "cluster_summary": output_root / "05_all_cohorts_cluster_annotation_summary.csv",
        "distribution": output_root / "05_all_cohorts_celltypist_label_distribution.csv",
        "celltype_counts": output_root / "05_celltypist_celltype_counts_by_cohort.csv",
        "overlap": output_root / "05_celltypist_model_gene_overlap_by_cohort.csv",
    }
    combined_frames: dict[str, pd.DataFrame] = {}
    for name, parts in combined.items():
        if not parts:
            continue
        frame = pd.concat(parts, ignore_index=True)
        frame.to_csv(combined_paths[name], index=False)
        combined_frames[name] = frame

    summary_figures = output_root / "figures_summary"
    summary_figures.mkdir(parents=True, exist_ok=True)
    cohort_order = selected["cohort"].astype(str).tolist()
    if "celltype_counts" in combined_frames:
        path = summary_figures / "05_celltypist_celltype_proportions_by_cohort.png"
        plot_celltype_composition(
            combined_frames["celltype_counts"], cohort_order, label_colors, path, args.dpi
        )
        figure_rows.append(
            {"scope": "combined", "figure": str(path), "description": "CellTypist majority-vote cell-type proportions"}
        )
    if "distribution" in combined_frames:
        path = summary_figures / "05_celltypist_cluster_vote_heatmap.png"
        plot_vote_heatmap(combined_frames["distribution"], path, args.dpi)
        figure_rows.append(
            {"scope": "combined", "figure": str(path), "description": "Raw CellTypist label fractions by cluster"}
        )
    if "cluster_summary" in combined_frames:
        path = summary_figures / "05_endothelial_LEC_marker_heatmap.png"
        plot_marker_heatmap(combined_frames["cluster_summary"], path, args.dpi)
        figure_rows.append(
            {"scope": "combined", "figure": str(path), "description": "Endothelial and LEC marker detection by cluster"}
        )

    pd.DataFrame(figure_rows).to_csv(
        output_root / "05_figure_manifest.csv", index=False
    )
    pd.DataFrame(
        [
            {"celltypist_label": label, "color": color}
            for label, color in sorted(label_colors.items(), key=lambda item: item[0].casefold())
        ]
    ).to_csv(output_root / "05_celltypist_color_key.csv", index=False)
    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "step03_root": str(step03_root),
        "output_root": str(output_root),
        "model": str(model_path),
        "expected_cohorts": [list(pair) for pair in EXPECTED_COHORTS],
        "selected_cohorts": selected["cohort"].tolist(),
        "celltypist_scope": "all_cells_chunked",
        "majority_vote_scope": "all_cell_predictions_within_each_Leiden_cluster",
        "marker_ranking_scope": (
            "all_cells" if args.max_marker_cells == 0 else "cluster_stratified_sample"
        ),
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output_root / "05_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    summary_frame = pd.DataFrame(summaries)
    errors = int(summary_frame["status"].eq("error").sum())
    print("\nStep 05 finished.")
    print("Run summary:", output_root / "05_run_summary_all_cohorts.csv")
    print("Errors:", errors)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
