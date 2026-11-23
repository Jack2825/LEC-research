#!/usr/bin/env python
r"""
Step 05 for the dataset13 NF-only aging-quartile pipeline.

For each completed Step 03 quartile (Q1-Q4), this standalone script:

1. Validates that the input contains only NF cells and the expected quartile.
2. Finds positive marker genes for every Leiden cluster versus the rest of
   that quartile and exports the top 25 by default.
3. Runs the local Healthy Adult Heart CellTypist model on individual cells.
4. Uses Leiden clusters for CellTypist majority voting.
5. Combines CellTypist votes, confidence, top marker genes, and expression of
   endothelial/LEC marker sets in a cluster-level annotation summary.
6. Writes per-cell predictions, per-cluster summaries, combined CSV files,
   and a small set of explanatory figures.

The Step 03 h5ad files are read but never changed. This script does not write
annotated h5ad copies; cell-level labels are saved as compressed CSV files so
they can be joined by cell ID without duplicating several large h5ad files.

Default input
-------------
new_result2_aging/03/dataset13/Q1/dataset13_Q1_clustered.h5ad
... Q2, Q3, Q4

Default output
--------------
new_result2_aging/05_marker_celltypist/dataset13/

Full run
--------
PowerShell:

    .\.venv\Scripts\python.exe `
      .\new_script2_aging\05_find_markers_and_celltypist_age_quartiles.py

Small test
----------

    .\.venv\Scripts\python.exe `
      .\new_script2_aging\05_find_markers_and_celltypist_age_quartiles.py `
      --quartiles Q1 --max_cells 2000 --method t-test_overestim_var
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scanpy as sc
except ImportError as error:
    raise ImportError(
        "scanpy is required. Activate the project .venv before Step 05."
    ) from error

try:
    import celltypist
except ImportError as error:
    raise ImportError(
        "celltypist is required. Install it in the project .venv first."
    ) from error


SCRIPT_VERSION = "05_dataset13_NF_age_markers_celltypist_2026_08_06"
DEFAULT_INPUT_ROOT = "new_result2_aging/03/dataset13"
DEFAULT_OUTPUT_ROOT = "new_result2_aging/05_marker_celltypist/dataset13"
DEFAULT_MODEL = "resources/celltypist/Healthy_Adult_Heart.pkl"
DEFAULT_CLUSTER_KEY = "leiden"
DEFAULT_TOP_N = 25
DEFAULT_METHOD = "wilcoxon"
REQUIRED_DISEASE = "NF"
QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]

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


def safe_name(value: str) -> str:
    text = str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "NA"


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("Missing").str.strip()


def sort_cluster_labels(labels) -> list[str]:
    def key(label: str):
        text = str(label)
        try:
            return 0, int(text)
        except ValueError:
            return 1, text

    return sorted([str(label) for label in labels], key=key)


def parse_quartiles(text: str) -> list[str]:
    requested = [part.strip().upper() for part in text.replace(";", ",").split(",")]
    requested = [part for part in requested if part]
    invalid = sorted(set(requested) - set(QUARTILE_LABELS))
    if invalid:
        raise ValueError(f"Invalid quartiles: {invalid}. Expected Q1,Q2,Q3,Q4.")
    if not requested:
        raise ValueError("At least one quartile must be requested.")
    return [label for label in QUARTILE_LABELS if label in set(requested)]


def ensure_cluster_key(adata, requested: str) -> str:
    if requested in adata.obs.columns:
        return requested
    for candidate in ["leiden", "clusters", "cluster", "SubCluster"]:
        if candidate in adata.obs.columns:
            print(f"Cluster key {requested!r} absent; using {candidate!r}.")
            return candidate
    raise KeyError(
        f"No cluster column was found. Available obs columns: {adata.obs.columns.tolist()[:60]}"
    )


def validate_nf_quartile(adata, quartile: str, source: Path) -> None:
    disease_column = "disease" if "disease" in adata.obs.columns else None
    if disease_column is None:
        raise KeyError(f"Cannot verify NF-only input; 'disease' is absent from {source}.")
    diseases = normalized_text(adata.obs[disease_column])
    if not diseases.str.casefold().eq(REQUIRED_DISEASE.casefold()).all():
        raise ValueError(
            f"Mixed diseases found in {source}: {diseases.value_counts().to_dict()}"
        )
    if "age_quartile" not in adata.obs.columns:
        raise KeyError(f"'age_quartile' is absent from {source}.")
    observed = set(normalized_text(adata.obs["age_quartile"]).unique())
    if observed != {quartile}:
        raise ValueError(
            f"Expected only {quartile} in {source}, but found {sorted(observed)}."
        )


def make_cluster_counts(adata, cluster_key: str) -> pd.DataFrame:
    counts = (
        normalized_text(adata.obs[cluster_key])
        .value_counts()
        .rename_axis("cluster")
        .reset_index(name="n_cells")
    )
    order = {label: index for index, label in enumerate(sort_cluster_labels(counts["cluster"]))}
    counts["_order"] = counts["cluster"].map(order)
    return counts.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def stratified_subsample(adata, cluster_key: str, max_cells: int, seed: int):
    """Return all cells or a reproducible cluster-stratified test subset."""
    if max_cells <= 0 or adata.n_obs <= max_cells:
        return adata
    rng = np.random.default_rng(seed)
    clusters = normalized_text(adata.obs[cluster_key]).to_numpy()
    selected: list[int] = []
    unique_clusters = sort_cluster_labels(pd.unique(clusters))
    minimum_counts = {
        cluster: min(10, int(np.sum(clusters == cluster)))
        for cluster in unique_clusters
    }
    minimum_total = sum(minimum_counts.values())
    if max_cells < minimum_total:
        raise ValueError(
            f"--max_cells={max_cells} is too small for a stable stratified test. "
            f"Use at least {minimum_total} cells to retain up to 10 per cluster."
        )
    for cluster in unique_clusters:
        indices = np.flatnonzero(clusters == cluster)
        take = minimum_counts[cluster]
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    if len(selected) < max_cells:
        remaining = np.setdiff1d(np.arange(adata.n_obs), np.asarray(selected), assume_unique=False)
        extra_n = min(max_cells - len(selected), len(remaining))
        if extra_n > 0:
            selected.extend(rng.choice(remaining, size=extra_n, replace=False).tolist())
    selected = sorted(selected)
    return adata[selected].copy()


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"CellTypist model not found: {model_path}")
    model = celltypist.models.Model.load(str(model_path))
    features = pd.Index([str(feature) for feature in model.features])
    return model, features


def prepare_gene_names_for_model(adata, model_features: pd.Index) -> tuple[object, dict]:
    """Prefer gene symbols only when they improve model-feature matching."""
    model_upper = set(model_features.str.upper())
    current_names = pd.Index(adata.var_names.astype(str))
    current_overlap = len(set(current_names.str.upper()) & model_upper)
    chosen_source = "var_names"
    query = adata

    if "gene_symbol" in adata.var.columns:
        symbols = adata.var["gene_symbol"].astype("string").fillna("").str.strip()
        valid = symbols.ne("")
        symbol_overlap = len(set(symbols.loc[valid].str.upper()) & model_upper)
        if valid.sum() >= 1000 and symbol_overlap > current_overlap:
            query = adata[:, valid.to_numpy()].copy()
            query.var_names = symbols.loc[valid].astype(str).to_numpy()
            query.var_names_make_unique()
            current_names = pd.Index(query.var_names.astype(str))
            current_overlap = len(set(current_names.str.upper()) & model_upper)
            chosen_source = "var_gene_symbol"

    overlap_genes = sorted(set(current_names.str.upper()) & model_upper)
    info = {
        "gene_name_source": chosen_source,
        "n_query_genes": int(query.n_vars),
        "n_model_genes": int(len(model_features)),
        "n_model_genes_present": int(len(overlap_genes)),
        "model_gene_overlap_fraction": float(len(overlap_genes) / len(model_features)),
        "query_gene_overlap_fraction": float(len(overlap_genes) / query.n_vars),
        "overlap_gene_examples": "; ".join(overlap_genes[:50]),
    }
    return query, info


def rank_all_genes(
    adata,
    cluster_key: str,
    method: str,
    n_rank_genes: int,
) -> pd.DataFrame:
    use_raw = adata.raw is not None
    available_genes = adata.raw.n_vars if use_raw else adata.n_vars
    n_genes = available_genes if n_rank_genes <= 0 else min(n_rank_genes, available_genes)
    adata.obs[cluster_key] = normalized_text(adata.obs[cluster_key]).astype("category")
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        reference="rest",
        method=method,
        n_genes=n_genes,
        use_raw=use_raw,
        pts=True,
        rankby_abs=False,
    )
    groups = list(adata.uns["rank_genes_groups"]["names"].dtype.names)
    tables: list[pd.DataFrame] = []
    for group in groups:
        table = sc.get.rank_genes_groups_df(adata, group=group)
        table.insert(0, "cluster", str(group))
        tables.append(table)
    ranked = pd.concat(tables, ignore_index=True)
    ranked = ranked.rename(
        columns={"names": "gene", "logfoldchanges": "logfoldchange"}
    )
    for column in [
        "scores", "logfoldchange", "pvals", "pvals_adj",
        "pct_nz_group", "pct_nz_reference",
    ]:
        if column in ranked.columns:
            ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    return ranked


def select_top_markers(
    ranked: pd.DataFrame,
    top_n: int,
    positive_only: bool,
) -> pd.DataFrame:
    markers = ranked.copy()
    markers["gene"] = normalized_text(markers["gene"])
    markers = markers.loc[
        ~markers["gene"].str.upper().isin(["", "NAN", "NONE", "UNKNOWN", "MISSING"])
    ].copy()
    if positive_only and "logfoldchange" in markers.columns:
        markers = markers.loc[
            markers["logfoldchange"].isna() | markers["logfoldchange"].gt(0)
        ].copy()
    sort_columns = ["cluster", "scores", "pvals_adj"]
    sort_columns = [column for column in sort_columns if column in markers.columns]
    ascending = [True if column == "cluster" else column == "pvals_adj" for column in sort_columns]
    markers = markers.sort_values(sort_columns, ascending=ascending)
    markers = markers.groupby("cluster", group_keys=False).head(top_n).copy()
    markers["marker_rank"] = markers.groupby("cluster").cumcount() + 1
    gene_upper = markers["gene"].str.upper()
    markers["is_endothelial_marker"] = gene_upper.isin(ENDOTHELIAL_MARKERS)
    markers["is_LEC_marker"] = gene_upper.isin(LEC_MARKERS)
    markers["is_common_non_endothelial_marker"] = gene_upper.isin(
        NON_ENDOTHELIAL_MARKERS
    )
    front = ["cluster", "marker_rank", "gene"]
    return markers[front + [column for column in markers.columns if column not in front]]


def make_marker_wide(top: pd.DataFrame, quartile: str, top_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    for cluster in sort_cluster_labels(top["cluster"].unique()):
        subset = top.loc[top["cluster"].astype(str).eq(cluster)].sort_values("marker_rank")
        genes = subset["gene"].astype(str).tolist()
        row = {
            "dataset": "dataset13",
            "disease": REQUIRED_DISEASE,
            "age_quartile": quartile,
            "cluster": cluster,
            "top_marker_genes_semicolon": "; ".join(genes),
        }
        for index in range(top_n):
            row[f"marker_{index + 1}"] = genes[index] if index < len(genes) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def marker_expression_summary(adata, cluster_key: str) -> pd.DataFrame:
    """Summarize expression of endothelial, LEC, and negative-check genes."""
    matrix_adata = adata.raw if adata.raw is not None else adata
    gene_names = pd.Index(matrix_adata.var_names.astype(str))
    lookup = {gene.upper(): gene for gene in gene_names}
    present = [gene for gene in MARKER_EXPRESSION_GENES if gene in lookup]
    matched = [lookup[gene] for gene in present]
    clusters = normalized_text(adata.obs[cluster_key]).to_numpy()
    cluster_labels = sort_cluster_labels(pd.unique(clusters))
    rows: list[dict] = []
    if not present:
        return pd.DataFrame({"cluster": cluster_labels})

    expression = matrix_adata[:, matched].X
    if sp.issparse(expression):
        expression = expression.tocsr()
    for cluster in cluster_labels:
        mask = clusters == cluster
        block = expression[mask]
        if sp.issparse(block):
            mean_values = np.asarray(block.mean(axis=0)).ravel()
            pct_values = np.asarray((block > 0).mean(axis=0)).ravel()
        else:
            array = np.asarray(block)
            mean_values = np.mean(array, axis=0)
            pct_values = np.mean(array > 0, axis=0)
        row: dict[str, object] = {"cluster": cluster, "n_cells_marker_summary": int(mask.sum())}
        for gene, mean_value, pct_value in zip(present, mean_values, pct_values):
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


def run_celltypist(adata, model_path: Path, cluster_key: str):
    predictions = celltypist.annotate(
        adata,
        model=str(model_path),
        majority_voting=True,
        over_clustering=cluster_key,
    )
    labels = predictions.predicted_labels.copy()
    if isinstance(labels, pd.Series):
        labels = labels.to_frame("predicted_labels")
    labels.index = labels.index.astype(str)
    probability = predictions.probability_matrix
    if isinstance(probability, pd.DataFrame):
        probability.index = probability.index.astype(str)
        labels["max_probability"] = probability.max(axis=1).reindex(labels.index)
        labels["probability_best_label"] = probability.idxmax(axis=1).reindex(labels.index)
    return predictions, labels


def choose_prediction_columns(labels: pd.DataFrame) -> tuple[str, str]:
    raw_column = "predicted_labels" if "predicted_labels" in labels.columns else labels.columns[0]
    vote_column = "majority_voting" if "majority_voting" in labels.columns else raw_column
    return raw_column, vote_column


def summarize_predictions(
    adata,
    labels: pd.DataFrame,
    cluster_key: str,
    quartile: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_column, vote_column = choose_prediction_columns(labels)
    cell_table = pd.DataFrame(index=adata.obs_names.astype(str))
    cell_table["cell_id"] = cell_table.index
    cell_table["dataset"] = "dataset13"
    cell_table["disease"] = REQUIRED_DISEASE
    cell_table["age_quartile"] = quartile
    cell_table["cluster"] = normalized_text(adata.obs[cluster_key]).to_numpy()
    cell_table["celltypist_raw_label"] = normalized_text(
        labels[raw_column].reindex(cell_table.index)
    ).to_numpy()
    cell_table["celltypist_majority_vote"] = normalized_text(
        labels[vote_column].reindex(cell_table.index)
    ).to_numpy()
    if "max_probability" in labels.columns:
        cell_table["celltypist_max_probability"] = pd.to_numeric(
            labels["max_probability"].reindex(cell_table.index), errors="coerce"
        ).to_numpy()
    if "probability_best_label" in labels.columns:
        cell_table["celltypist_probability_best_label"] = normalized_text(
            labels["probability_best_label"].reindex(cell_table.index)
        ).to_numpy()

    cluster_rows: list[dict] = []
    distribution_rows: list[dict] = []
    for cluster in sort_cluster_labels(cell_table["cluster"].unique()):
        subset = cell_table.loc[cell_table["cluster"].eq(cluster)].copy()
        n_cells = int(subset.shape[0])
        raw_counts = subset["celltypist_raw_label"].value_counts()
        vote_counts = subset["celltypist_majority_vote"].value_counts()
        raw_majority = str(raw_counts.index[0]) if len(raw_counts) else "unknown"
        raw_majority_fraction = float(raw_counts.iloc[0] / n_cells) if n_cells else np.nan
        second_label = str(raw_counts.index[1]) if len(raw_counts) > 1 else ""
        second_fraction = float(raw_counts.iloc[1] / n_cells) if len(raw_counts) > 1 else 0.0
        vote_label = str(vote_counts.index[0]) if len(vote_counts) else raw_majority
        vote_output_consistency = float(vote_counts.iloc[0] / n_cells) if n_cells else np.nan
        vote_support_fraction = (
            float(subset["celltypist_raw_label"].eq(vote_label).mean())
            if n_cells else np.nan
        )
        confidence = (
            pd.to_numeric(subset.get("celltypist_max_probability"), errors="coerce")
            if "celltypist_max_probability" in subset.columns else pd.Series(dtype=float)
        )
        cluster_rows.append(
            {
                "dataset": "dataset13",
                "disease": REQUIRED_DISEASE,
                "age_quartile": quartile,
                "cluster": cluster,
                "n_cells": n_cells,
                "cluster_majority_cell_type": vote_label,
                "majority_vote_support_fraction": vote_support_fraction,
                "celltypist_vote_output_consistency_fraction": vote_output_consistency,
                "raw_prediction_majority_label": raw_majority,
                "raw_prediction_majority_fraction": raw_majority_fraction,
                "second_raw_prediction_label": second_label,
                "second_raw_prediction_fraction": second_fraction,
                "mean_celltypist_confidence": float(confidence.mean()) if len(confidence) else np.nan,
                "median_celltypist_confidence": float(confidence.median()) if len(confidence) else np.nan,
                "n_distinct_raw_cell_types": int(raw_counts.shape[0]),
            }
        )
        for cell_type, count in raw_counts.items():
            distribution_rows.append(
                {
                    "dataset": "dataset13",
                    "disease": REQUIRED_DISEASE,
                    "age_quartile": quartile,
                    "cluster": cluster,
                    "cell_type": str(cell_type),
                    "n_cells": int(count),
                    "fraction_within_cluster": float(count / n_cells),
                }
            )

    cluster_table = pd.DataFrame(cluster_rows)
    distribution = pd.DataFrame(distribution_rows)
    celltype_counts = (
        cell_table["celltypist_majority_vote"]
        .value_counts()
        .rename_axis("cell_type")
        .reset_index(name="n_cells")
    )
    celltype_counts.insert(0, "age_quartile", quartile)
    celltype_counts.insert(0, "disease", REQUIRED_DISEASE)
    celltype_counts.insert(0, "dataset", "dataset13")
    celltype_counts["percent_cells"] = 100.0 * celltype_counts["n_cells"] / len(cell_table)
    return cell_table.reset_index(drop=True), cluster_table, distribution, celltype_counts


def merge_cluster_evidence(
    cluster_predictions: pd.DataFrame,
    marker_wide: pd.DataFrame,
    marker_expression: pd.DataFrame,
) -> pd.DataFrame:
    summary = cluster_predictions.merge(
        marker_wide[["cluster", "top_marker_genes_semicolon"]],
        on="cluster",
        how="left",
    )
    summary = summary.merge(marker_expression, on="cluster", how="left")
    summary["celltypist_label_is_endothelial"] = summary[
        "cluster_majority_cell_type"
    ].astype(str).str.contains("EC|endothelial|endocardial", case=False, regex=True)
    summary["celltypist_label_is_lymphatic"] = summary[
        "cluster_majority_cell_type"
    ].astype(str).str.contains("lymph|EC8_ln", case=False, regex=True)
    return summary.sort_values(
        ["age_quartile", "cluster"],
        key=lambda series: series.map(
            {label: index for index, label in enumerate(QUARTILE_LABELS)}
        ) if series.name == "age_quartile" else series.map(
            lambda value: int(value) if str(value).isdigit() else 10**9
        ),
    ).reset_index(drop=True)


def save_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_quartile_umap(adata, cell_table: pd.DataFrame, quartile: str, path: Path, dpi: int) -> bool:
    if "X_umap" not in adata.obsm:
        return False
    label_map = cell_table.set_index("cell_id")["celltypist_majority_vote"]
    adata.obs["celltypist_majority_vote"] = adata.obs_names.astype(str).map(label_map)
    adata.obs["celltypist_majority_vote"] = adata.obs[
        "celltypist_majority_vote"
    ].astype("category")
    try:
        sc.pl.umap(
            adata,
            color="celltypist_majority_vote",
            title=f"{quartile} NF: CellTypist majority-vote cell type",
            legend_loc="right margin",
            frameon=False,
            show=False,
        )
        save_figure(path, dpi)
        return True
    except Exception as error:
        print(f"Could not create {quartile} CellTypist UMAP: {error}")
        plt.close()
        return False


def plot_celltype_proportions(counts: pd.DataFrame, path: Path, dpi: int) -> bool:
    if counts.empty:
        return False
    top_types = (
        counts.groupby("cell_type", observed=True)["n_cells"].sum()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    plot_data = counts.copy()
    plot_data["plot_cell_type"] = plot_data["cell_type"].where(
        plot_data["cell_type"].isin(top_types), "Other"
    )
    grouped = (
        plot_data.groupby(["age_quartile", "plot_cell_type"], observed=True)["n_cells"]
        .sum()
        .reset_index()
    )
    grouped["percent"] = 100.0 * grouped["n_cells"] / grouped.groupby(
        "age_quartile", observed=True
    )["n_cells"].transform("sum")
    pivot = grouped.pivot(
        index="age_quartile", columns="plot_cell_type", values="percent"
    ).fillna(0)
    observed_quartiles = [
        label for label in QUARTILE_LABELS if label in set(grouped["age_quartile"])
    ]
    pivot = pivot.reindex(observed_quartiles)
    ordered_columns = [label for label in top_types if label in pivot.columns]
    if "Other" in pivot.columns:
        ordered_columns.append("Other")
    pivot = pivot[ordered_columns]
    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 5.6),
        colormap="tab20",
        width=0.72,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_title("NF CellTypist majority-vote cell-type composition")
    ax.set_xlabel("Age quartile")
    ax.set_ylabel("Cells (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Cell type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_figure(path, dpi)
    return True


def plot_vote_heatmap(distribution: pd.DataFrame, path: Path, dpi: int) -> bool:
    if distribution.empty:
        return False
    top_types = (
        distribution.groupby("cell_type", observed=True)["n_cells"].sum()
        .sort_values(ascending=False)
        .head(11)
        .index.tolist()
    )
    all_distribution = distribution.copy()
    all_distribution["quartile_cluster"] = (
        all_distribution["age_quartile"] + ":C" + all_distribution["cluster"].astype(str)
    )
    all_distribution["plot_cell_type"] = all_distribution["cell_type"].where(
        all_distribution["cell_type"].isin(top_types), "Other"
    )
    subset = (
        all_distribution.groupby(
            ["quartile_cluster", "age_quartile", "cluster", "plot_cell_type"],
            observed=True,
            as_index=False,
        )[["n_cells", "fraction_within_cluster"]]
        .sum()
    )
    row_order = []
    for quartile in QUARTILE_LABELS:
        labels = all_distribution.loc[
            all_distribution["age_quartile"].eq(quartile), "cluster"
        ].unique()
        row_order.extend([f"{quartile}:C{cluster}" for cluster in sort_cluster_labels(labels)])
    pivot = subset.pivot_table(
        index="quartile_cluster",
        columns="plot_cell_type",
        values="fraction_within_cluster",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=row_order, columns=top_types + ["Other"], fill_value=0)
    height = max(6.0, min(14.0, 0.22 * len(pivot.index) + 2.5))
    fig, ax = plt.subplots(figsize=(11, height))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index, fontsize=7)
    ax.set_title("Raw CellTypist label fractions within each Leiden cluster")
    ax.set_xlabel("CellTypist cell type")
    ax.set_ylabel("Age quartile and cluster")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Fraction of cluster")
    fig.tight_layout()
    save_figure(path, dpi)
    return True


def plot_marker_heatmap(cluster_summary: pd.DataFrame, path: Path, dpi: int) -> bool:
    marker_columns = [
        f"pct_expr_{gene}" for gene in ENDOTHELIAL_MARKERS + LEC_MARKERS
        if f"pct_expr_{gene}" in cluster_summary.columns
    ]
    if not marker_columns or cluster_summary.empty:
        return False
    ordered = cluster_summary.copy()
    ordered["quartile_order"] = ordered["age_quartile"].map(
        {label: index for index, label in enumerate(QUARTILE_LABELS)}
    )
    ordered["cluster_order"] = ordered["cluster"].map(
        lambda value: int(value) if str(value).isdigit() else 10**9
    )
    ordered = ordered.sort_values(["quartile_order", "cluster_order"])
    row_labels = ordered["age_quartile"] + ":C" + ordered["cluster"].astype(str)
    values = ordered[marker_columns].fillna(0).to_numpy(dtype=float)
    height = max(6.0, min(14.0, 0.22 * len(ordered) + 2.5))
    fig, ax = plt.subplots(figsize=(11, height))
    image = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(
        np.arange(len(marker_columns)),
        [column.replace("pct_expr_", "") for column in marker_columns],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(row_labels)), row_labels, fontsize=7)
    ax.set_title("Endothelial and LEC marker detection by Leiden cluster")
    ax.set_xlabel("Marker gene")
    ax.set_ylabel("Age quartile and cluster")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Fraction of cells expressing gene")
    fig.tight_layout()
    save_figure(path, dpi)
    return True


def process_quartile(
    quartile: str,
    input_path: Path,
    output_root: Path,
    model_path: Path,
    model_features: pd.Index,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, pd.DataFrame], list[dict]]:
    outdir = output_root / quartile
    figure_dir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"dataset13_{quartile}"

    print("\n" + "=" * 96)
    print(f"Step 05 {quartile}: {input_path}")
    print("Output:", outdir)
    print("=" * 96)

    adata_full = sc.read_h5ad(input_path)
    adata_full.obs_names = adata_full.obs_names.astype(str)
    adata_full.var_names_make_unique()
    validate_nf_quartile(adata_full, quartile, input_path)
    cluster_key = ensure_cluster_key(adata_full, args.cluster_key)
    adata_full.obs[cluster_key] = normalized_text(adata_full.obs[cluster_key]).astype("category")
    n_cells_input = int(adata_full.n_obs)

    adata = stratified_subsample(
        adata_full,
        cluster_key=cluster_key,
        max_cells=args.max_cells,
        seed=args.random_seed + QUARTILE_LABELS.index(quartile),
    )
    if adata is not adata_full:
        del adata_full
        gc.collect()
    adata.obs[cluster_key] = normalized_text(adata.obs[cluster_key]).astype("category")
    cluster_counts = make_cluster_counts(adata, cluster_key)
    cluster_counts.insert(0, "age_quartile", quartile)
    cluster_counts.insert(0, "disease", REQUIRED_DISEASE)
    cluster_counts.insert(0, "dataset", "dataset13")
    cluster_counts.to_csv(outdir / f"{prefix}_cluster_counts.csv", index=False)
    if cluster_counts.shape[0] < 2:
        raise ValueError(f"Need at least two clusters in {quartile}; found {cluster_counts.shape[0]}.")

    marker_expression = marker_expression_summary(adata, cluster_key)
    ranked = rank_all_genes(
        adata,
        cluster_key=cluster_key,
        method=args.method,
        n_rank_genes=args.n_rank_genes,
    )
    if args.save_all_ranked:
        ranked.to_csv(
            outdir / f"{prefix}_ALL_ranked_genes.csv.gz",
            index=False,
            compression="gzip",
        )
    top = select_top_markers(ranked, args.top_n, args.positive_only)
    top.insert(0, "source_h5ad", str(input_path))
    top.insert(0, "age_quartile", quartile)
    top.insert(0, "disease", REQUIRED_DISEASE)
    top.insert(0, "dataset", "dataset13")
    marker_long_path = outdir / f"{prefix}_top{args.top_n}_markers_long.csv"
    top.to_csv(marker_long_path, index=False)
    marker_wide = make_marker_wide(top, quartile, args.top_n)
    marker_wide_path = outdir / f"{prefix}_top{args.top_n}_markers_wide.csv"
    marker_wide.to_csv(marker_wide_path, index=False)

    query, overlap_info = prepare_gene_names_for_model(adata, model_features)
    if overlap_info["n_model_genes_present"] < args.min_model_genes_present:
        raise ValueError(
            f"Only {overlap_info['n_model_genes_present']} model genes are present; "
            f"minimum required is {args.min_model_genes_present}."
        )
    print(
        "CellTypist model-gene overlap: "
        f"{overlap_info['n_model_genes_present']:,}/{overlap_info['n_model_genes']:,} "
        f"({overlap_info['model_gene_overlap_fraction']:.1%})"
    )
    predictions, predicted_labels = run_celltypist(query, model_path, cluster_key)
    cell_predictions, cluster_predictions, label_distribution, celltype_counts = (
        summarize_predictions(query, predicted_labels, cluster_key, quartile)
    )
    cluster_summary = merge_cluster_evidence(
        cluster_predictions,
        marker_wide,
        marker_expression,
    )

    cell_path = outdir / f"{prefix}_celltypist_cell_predictions.csv.gz"
    if not args.skip_cell_predictions:
        cell_predictions.to_csv(cell_path, index=False, compression="gzip")
    cluster_summary_path = outdir / f"{prefix}_cluster_annotation_summary.csv"
    cluster_summary.to_csv(cluster_summary_path, index=False)
    label_distribution.to_csv(
        outdir / f"{prefix}_celltypist_label_distribution_by_cluster.csv",
        index=False,
    )
    celltype_counts.to_csv(
        outdir / f"{prefix}_celltypist_celltype_counts.csv",
        index=False,
    )
    pd.DataFrame([{"age_quartile": quartile, **overlap_info}]).to_csv(
        outdir / f"{prefix}_celltypist_model_gene_overlap.csv",
        index=False,
    )

    figure_rows: list[dict] = []
    umap_path = figure_dir / f"{prefix}_umap_celltypist_majority_vote.png"
    if plot_quartile_umap(query, cell_predictions, quartile, umap_path, args.dpi):
        figure_rows.append(
            {
                "scope": quartile,
                "figure": str(umap_path),
                "description": "UMAP colored by CellTypist Leiden-cluster majority vote",
            }
        )

    summary = {
        "script_version": SCRIPT_VERSION,
        "dataset": "dataset13",
        "disease": REQUIRED_DISEASE,
        "age_quartile": quartile,
        "status": "completed",
        "source_h5ad": str(input_path),
        "model_path": str(model_path),
        "cluster_key": cluster_key,
        "marker_method": args.method,
        "top_n": int(args.top_n),
        "n_cells_input": n_cells_input,
        "n_cells_analyzed": int(query.n_obs),
        "analysis_is_subsample": bool(query.n_obs < n_cells_input),
        "n_genes_query": int(query.n_vars),
        "n_clusters": int(cluster_counts.shape[0]),
        "n_model_genes": overlap_info["n_model_genes"],
        "n_model_genes_present": overlap_info["n_model_genes_present"],
        "model_gene_overlap_fraction": overlap_info["model_gene_overlap_fraction"],
        "marker_long_csv": str(marker_long_path),
        "marker_wide_csv": str(marker_wide_path),
        "cluster_annotation_summary_csv": str(cluster_summary_path),
        "cell_predictions_csv": "" if args.skip_cell_predictions else str(cell_path),
    }
    pd.DataFrame([summary]).to_csv(
        outdir / f"{prefix}_step05_run_summary.csv", index=False
    )

    tables = {
        "top": top,
        "wide": marker_wide,
        "cluster_summary": cluster_summary,
        "label_distribution": label_distribution,
        "celltype_counts": celltype_counts,
        "overlap": pd.DataFrame([{"age_quartile": quartile, **overlap_info}]),
    }
    del predictions, predicted_labels, ranked
    if query is not adata:
        del query
    del adata
    gc.collect()
    return summary, tables, figure_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find cluster markers and run Heart CellTypist majority-vote annotation "
            "for NF-only dataset13 age quartiles."
        )
    )
    parser.add_argument("--input_root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quartiles", default="Q1,Q2,Q3,Q4")
    parser.add_argument("--cluster_key", default=DEFAULT_CLUSTER_KEY)
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=["wilcoxon", "t-test", "t-test_overestim_var", "logreg"],
    )
    parser.add_argument(
        "--n_rank_genes",
        type=int,
        default=0,
        help="Genes ranked per cluster; 0 means every available gene.",
    )
    parser.add_argument(
        "--positive_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save_all_ranked", action="store_true")
    parser.add_argument(
        "--max_cells",
        type=int,
        default=0,
        help="Cluster-stratified test subset per quartile; 0 analyzes every cell.",
    )
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--min_model_genes_present", type=int, default=500)
    parser.add_argument("--skip_cell_predictions", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top_n must be at least 1.")
    if args.max_cells < 0:
        raise ValueError("--max_cells cannot be negative.")

    project_dir = find_project_dir()
    input_root = resolve_path(project_dir, args.input_root)
    output_root = resolve_path(project_dir, args.output_root)
    model_path = resolve_path(project_dir, args.model)
    quartiles = parse_quartiles(args.quartiles)
    output_root.mkdir(parents=True, exist_ok=True)

    model, model_features = load_model(model_path)
    print("=" * 100)
    print("Dataset13 NF-only aging Step 05: markers + CellTypist")
    print("Input root:", input_root)
    print("Output root:", output_root)
    print("Model:", model_path)
    print("Model genes:", len(model_features))
    print("Quartiles:", ", ".join(quartiles))
    print("Marker method:", args.method)
    print("Maximum cells per quartile:", args.max_cells or "all")
    print("=" * 100)

    summaries: list[dict] = []
    combined: dict[str, list[pd.DataFrame]] = {
        "top": [],
        "wide": [],
        "cluster_summary": [],
        "label_distribution": [],
        "celltype_counts": [],
        "overlap": [],
    }
    figure_rows: list[dict] = []

    for quartile in quartiles:
        input_path = input_root / quartile / f"dataset13_{quartile}_clustered.h5ad"
        try:
            if not input_path.exists():
                raise FileNotFoundError(f"Missing Step 03 input: {input_path}")
            summary, tables, figures = process_quartile(
                quartile,
                input_path,
                output_root,
                model_path,
                model_features,
                args,
            )
            summaries.append(summary)
            for name, table in tables.items():
                combined[name].append(table)
            figure_rows.extend(figures)
        except Exception as error:
            print(f"ERROR in {quartile}: {error}")
            traceback.print_exc()
            error_dir = output_root / quartile
            error_dir.mkdir(parents=True, exist_ok=True)
            with open(
                error_dir / f"dataset13_{quartile}_step05_ERROR.txt",
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(str(error) + "\n\n" + traceback.format_exc())
            summaries.append(
                {
                    "script_version": SCRIPT_VERSION,
                    "dataset": "dataset13",
                    "disease": REQUIRED_DISEASE,
                    "age_quartile": quartile,
                    "status": "error",
                    "source_h5ad": str(input_path),
                    "error": str(error),
                }
            )
            gc.collect()

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_root / "05_run_summary_all_quartiles.csv", index=False)

    combined_paths = {
        "top": output_root / f"05_all_quartiles_top{args.top_n}_markers_long.csv",
        "wide": output_root / f"05_all_quartiles_top{args.top_n}_markers_wide.csv",
        "cluster_summary": output_root / "05_all_quartiles_cluster_annotation_summary.csv",
        "label_distribution": output_root / "05_all_quartiles_celltypist_label_distribution.csv",
        "celltype_counts": output_root / "05_celltypist_celltype_counts_by_quartile.csv",
        "overlap": output_root / "05_celltypist_model_gene_overlap_by_quartile.csv",
    }
    combined_frames: dict[str, pd.DataFrame] = {}
    for name, parts in combined.items():
        if not parts:
            continue
        frame = pd.concat(parts, ignore_index=True)
        frame.to_csv(combined_paths[name], index=False)
        combined_frames[name] = frame

    summary_figure_dir = output_root / "figures_summary"
    summary_figure_dir.mkdir(parents=True, exist_ok=True)
    if "celltype_counts" in combined_frames:
        path = summary_figure_dir / "05_celltypist_celltype_proportions_by_quartile.png"
        if plot_celltype_proportions(combined_frames["celltype_counts"], path, args.dpi):
            figure_rows.append(
                {"scope": "combined", "figure": str(path), "description": "Cell-type proportions by age quartile"}
            )
    if "label_distribution" in combined_frames:
        path = summary_figure_dir / "05_celltypist_cluster_vote_heatmap.png"
        if plot_vote_heatmap(combined_frames["label_distribution"], path, args.dpi):
            figure_rows.append(
                {"scope": "combined", "figure": str(path), "description": "Raw CellTypist label fractions by cluster"}
            )
    if "cluster_summary" in combined_frames:
        path = summary_figure_dir / "05_endothelial_LEC_marker_heatmap.png"
        if plot_marker_heatmap(combined_frames["cluster_summary"], path, args.dpi):
            figure_rows.append(
                {"scope": "combined", "figure": str(path), "description": "Endothelial and LEC marker detection by cluster"}
            )
    pd.DataFrame(figure_rows).to_csv(
        output_root / "05_figure_manifest.csv", index=False
    )

    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "model": str(model_path),
        "model_cell_types": [str(value) for value in model.cell_types],
        **vars(args),
    }
    with open(output_root / "05_run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    completed = int(summary_frame["status"].eq("completed").sum())
    errors = int(summary_frame["status"].eq("error").sum())
    print("\nDone.")
    print(f"Completed quartiles: {completed}; errors: {errors}")
    print("Run summary:", output_root / "05_run_summary_all_quartiles.csv")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
