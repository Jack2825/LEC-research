#!/usr/bin/env python
r"""
Step 03 for the dataset13 adult-aging quartile pipeline.

This is a standalone aging Step 03. Its QC, normalization, Harmony, UMAP,
Leiden, plotting, and reporting behavior follows the original new_scripts Step
03, but it does not import or depend on that file.

The aging-specific grouping steps are:

1. Load the existing full dataset13 count h5ad.
2. Retain only NF (non-failing/normal) cells before constructing the cohort.
3. Retain only standardized cardiac regions LV and AP. All RV, septum, LA,
   RA, missing, and other region labels are removed before donor grouping/QC.
4. Resolve one age per biological donor.
5. Retain donors with resolved age >= 30 years (default).
6. Build a project-namespaced donor ID because donor labels such as D1 and H2
   occur in more than one source project.
7. Sort biological donors by age and assign approximately equal-count donor
   quartiles Q1-Q4. With a donor count not divisible by four, group sizes differ
   by at most one donor.
8. Propagate the donor quartile to every cell.
9. Run the same Step 03 hard QC and condition-specific clustering separately
   for Q1, Q2, Q3, and Q4.

Quartiles are calculated from donors, never from cells. A deterministic donor
key breaks ties between donors with the same exact or inferred age. Boundary
ties are reported in the quartile summary so this choice remains auditable.

Default input
-------------
Data/converted/dataset_13_Age_full_30plus_age10yr_standardized.h5ad

Default output
--------------
new_result2_aging/03/dataset13/Q1/
new_result2_aging/03/dataset13/Q2/
new_result2_aging/03/dataset13/Q3/
new_result2_aging/03/dataset13/Q4/

The existing decade-based inputs and results are not overwritten.

Example
-------
PowerShell:

    .\.venv\Scripts\python.exe `
      .\new_script2_aging\03_cluster_dataset13_age_quartiles_meanIQR_QC.py
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Iterable

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
        "scanpy is required. Activate the project .venv before running Step 03."
    ) from error


SCRIPT_VERSION = "03_dataset13_NF_LV_AP_only_age_quartiles_2026_08_06"
DEFAULT_H5AD = (
    "Data/converted/"
    "dataset_13_Age_full_30plus_age10yr_standardized.h5ad"
)
DEFAULT_OUTPUT_ROOT = "new_result2_aging/03"
DATASET_ID = "dataset_13_Age"
DATASET_FOLDER = "dataset13"
QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]
UNIQUE_DONOR_COL = "donor_id_quartile_unique"
REQUIRED_DISEASE = "NF"
REQUIRED_STANDARDIZED_REGIONS = ["LV", "AP"]


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def first_existing_column(frame: pd.DataFrame, requested: str, candidates: list[str]) -> str:
    if requested:
        if requested not in frame.columns:
            raise KeyError(
                f"Requested column '{requested}' is absent. "
                f"Available columns include: {list(frame.columns[:40])}"
            )
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(
        "None of the required columns were found. Tried: " + ", ".join(candidates)
    )


# =============================================================================
# Standalone Step 03 QC and clustering functions
# =============================================================================

def optional_obs_column(
    obs: pd.DataFrame,
    requested: str,
    candidates: Iterable[str],
) -> str | None:
    if requested and requested in obs.columns:
        return requested
    for candidate in candidates:
        if candidate and candidate in obs.columns:
            return candidate
    return None


def choose_count_matrix(adata, args: argparse.Namespace) -> str:
    if args.raw_count_layer:
        if args.raw_count_layer not in adata.layers:
            raise KeyError(
                f"Requested raw-count layer '{args.raw_count_layer}' is absent."
            )
        adata.X = adata.layers[args.raw_count_layer].copy()
        return f"layer:{args.raw_count_layer}"

    for layer in ["counts", "cellranger_raw", "raw_counts", "count", "RNA_counts"]:
        if layer in adata.layers:
            adata.X = adata.layers[layer].copy()
            return f"layer:{layer}"
    return "X_current"


def remove_large_unused_layers(adata, keep_layers: set[str] | None = None) -> None:
    keep_layers = keep_layers or set()
    for layer in list(adata.layers.keys()):
        if layer not in keep_layers:
            del adata.layers[layer]


def to_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def mito_to_percent(values: pd.Series) -> pd.Series:
    numeric = to_numeric(values)
    if numeric.notna().any() and numeric.dropna().max() <= 1.5:
        return numeric * 100.0
    return numeric


def sparse_row_entropy(matrix) -> np.ndarray:
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()

    output = np.zeros(matrix.shape[0], dtype=float)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    for row_index in range(matrix.shape[0]):
        row_sum = row_sums[row_index]
        if row_sum <= 0:
            continue
        data = matrix.data[matrix.indptr[row_index]:matrix.indptr[row_index + 1]]
        if data.size == 0:
            continue
        probabilities = data / row_sum
        probabilities = probabilities[probabilities > 0]
        output[row_index] = float(
            -np.sum(probabilities * np.log(probabilities))
        )
    return output


def compute_qc_metrics(adata, args: argparse.Namespace) -> dict:
    """Compute the same standardized QC columns used by the original Step 03."""
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.astype(str).str.upper().str.startswith("MT-")
    try:
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt"],
            inplace=True,
            percent_top=None,
            log1p=False,
        )
    except Exception:
        matrix = adata.X
        if sp.issparse(matrix):
            total_counts = np.asarray(matrix.sum(axis=1)).ravel()
            n_genes = np.asarray((matrix > 0).sum(axis=1)).ravel()
            mt_mask = adata.var["mt"].to_numpy()
            mt_counts = (
                np.asarray(matrix[:, mt_mask].sum(axis=1)).ravel()
                if mt_mask.any()
                else np.zeros(adata.n_obs)
            )
        else:
            dense = np.asarray(matrix)
            total_counts = dense.sum(axis=1)
            n_genes = (dense > 0).sum(axis=1)
            mt_mask = adata.var["mt"].to_numpy()
            mt_counts = (
                dense[:, mt_mask].sum(axis=1)
                if mt_mask.any()
                else np.zeros(adata.n_obs)
            )
        adata.obs["total_counts"] = total_counts
        adata.obs["n_genes_by_counts"] = n_genes
        adata.obs["pct_counts_mt"] = np.where(
            total_counts > 0,
            mt_counts / total_counts * 100.0,
            0.0,
        )

    genes_col = optional_obs_column(
        adata.obs,
        args.gene_count_col,
        ["n_genes", "n_genes_by_counts", "nFeature_RNA", "cellbender_ngenes"],
    ) or "n_genes_by_counts"
    counts_col = optional_obs_column(
        adata.obs,
        args.count_col,
        ["n_counts", "total_counts", "nCount_RNA", "cellbender_ncount"],
    ) or "total_counts"
    mito_col = optional_obs_column(
        adata.obs,
        args.mito_col,
        ["percent_mito", "pct_counts_mt", "percent.mito", "pct_mito"],
    ) or "pct_counts_mt"
    doublet_col = optional_obs_column(
        adata.obs,
        args.doublet_score_col,
        ["doublet_score", "doublet_scores", "scrublet_score"],
    )
    exon_col = optional_obs_column(
        adata.obs,
        args.exon_prop_col,
        ["exon_prop", "exonic_fraction", "exon_fraction"],
    )
    entropy_col = optional_obs_column(
        adata.obs,
        args.entropy_col,
        ["entropy", "cell_entropy", "RNA_entropy"],
    )

    adata.obs["qc_n_genes"] = to_numeric(adata.obs[genes_col]).to_numpy()
    adata.obs["qc_total_counts"] = to_numeric(adata.obs[counts_col]).to_numpy()
    adata.obs["qc_pct_mt"] = mito_to_percent(adata.obs[mito_col]).to_numpy()
    adata.obs["qc_doublet_score"] = (
        to_numeric(adata.obs[doublet_col]).to_numpy()
        if doublet_col
        else np.nan
    )
    adata.obs["qc_exon_fraction"] = (
        to_numeric(adata.obs[exon_col]).to_numpy()
        if exon_col
        else np.nan
    )
    if entropy_col:
        adata.obs["qc_entropy"] = to_numeric(adata.obs[entropy_col]).to_numpy()
    elif args.compute_entropy:
        print("Computing per-cell entropy because no entropy column was found.")
        adata.obs["qc_entropy"] = sparse_row_entropy(adata.X)
    else:
        adata.obs["qc_entropy"] = np.nan

    return {
        "genes_col": genes_col,
        "counts_col": counts_col,
        "mito_col": mito_col,
        "doublet_col": doublet_col,
        "exon_col": exon_col,
        "entropy_col": entropy_col,
    }


def apply_hard_qc(adata, args: argparse.Namespace) -> tuple[pd.Series, pd.DataFrame]:
    obs = adata.obs
    n_genes = to_numeric(obs["qc_n_genes"])
    total_counts = to_numeric(obs["qc_total_counts"])
    pct_mt = to_numeric(obs["qc_pct_mt"])
    keep = pd.Series(True, index=adata.obs_names)
    keep &= n_genes.ge(args.min_genes) & n_genes.notna()
    keep &= total_counts.ge(args.min_counts) & total_counts.notna()
    keep &= pct_mt.le(args.max_mito_percent) & pct_mt.notna()

    reasons = pd.DataFrame(index=adata.obs_names)
    reasons["remove_hard_low_genes"] = n_genes.lt(args.min_genes)
    reasons["remove_hard_low_counts"] = total_counts.lt(args.min_counts)
    reasons["remove_hard_high_mito"] = pct_mt.gt(args.max_mito_percent)
    reasons["keep_after_hard_QC"] = keep
    return keep, reasons


def mean_iqr_bounds(values: pd.Series, multiplier: float) -> tuple[float, ...]:
    numeric = to_numeric(values).dropna()
    if numeric.shape[0] < 4:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    mean = float(numeric.mean())
    q1 = float(np.percentile(numeric, 25))
    q3 = float(np.percentile(numeric, 75))
    iqr = q3 - q1
    return mean, q1, q3, mean - multiplier * iqr, mean + multiplier * iqr


def apply_mean_iqr_qc(
    adata,
    args: argparse.Namespace,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    obs = adata.obs
    keep = pd.Series(True, index=adata.obs_names)
    reasons = pd.DataFrame(index=adata.obs_names)
    if args.two_sided_all_qc_outliers:
        directions = {
            metric: "two_sided"
            for metric in [
                "qc_n_genes",
                "qc_total_counts",
                "qc_pct_mt",
                "qc_doublet_score",
                "qc_exon_fraction",
                "qc_entropy",
            ]
        }
    else:
        directions = {
            "qc_n_genes": "two_sided",
            "qc_total_counts": "two_sided",
            "qc_pct_mt": "high",
            "qc_doublet_score": "high",
            "qc_exon_fraction": "high",
            "qc_entropy": "low",
        }

    threshold_rows: list[dict] = []
    for metric, direction in directions.items():
        if metric not in obs.columns:
            continue
        values = to_numeric(obs[metric])
        n_non_na = int(values.notna().sum())
        if n_non_na < 4:
            threshold_rows.append(
                {
                    "metric": metric,
                    "status": "skipped_too_few_nonNA_values",
                    "direction": direction,
                    "n_nonNA": n_non_na,
                }
            )
            continue
        if (
            metric in {"qc_doublet_score", "qc_exon_fraction", "qc_entropy"}
            and n_non_na < max(4, int(0.1 * len(values)))
        ):
            threshold_rows.append(
                {
                    "metric": metric,
                    "status": "skipped_optional_metric_mostly_NA",
                    "direction": direction,
                    "n_nonNA": n_non_na,
                }
            )
            continue

        mean, q1, q3, lower, upper = mean_iqr_bounds(
            values, args.iqr_multiplier
        )
        remove_low = pd.Series(False, index=adata.obs_names)
        remove_high = pd.Series(False, index=adata.obs_names)
        if direction in {"low", "two_sided"}:
            remove_low = values.lt(lower)
        if direction in {"high", "two_sided"}:
            remove_high = values.gt(upper)
        remove_metric = remove_low.fillna(False) | remove_high.fillna(False)
        keep &= ~remove_metric
        reasons[f"remove_meanIQR_{metric}_low"] = remove_low.fillna(False)
        reasons[f"remove_meanIQR_{metric}_high"] = remove_high.fillna(False)
        threshold_rows.append(
            {
                "metric": metric,
                "status": "used",
                "direction": direction,
                "n_nonNA": n_non_na,
                "mean": mean,
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "iqr_multiplier": args.iqr_multiplier,
                "lower_threshold_mean_minus_multiplier_IQR": lower,
                "upper_threshold_mean_plus_multiplier_IQR": upper,
                "n_removed_low": int(remove_low.fillna(False).sum()),
                "n_removed_high": int(remove_high.fillna(False).sum()),
                "n_removed_total_metric": int(remove_metric.sum()),
            }
        )
    reasons["keep_after_meanIQR_QC"] = keep
    return keep, reasons, pd.DataFrame(threshold_rows)


def preprocess_and_cluster(
    adata_counts,
    outdir: Path,
    prefix: str,
    donor_col: str | None,
    args: argparse.Namespace,
):
    """HVG, log-normalization, PCA, optional Harmony, UMAP, and Leiden."""
    adata = adata_counts.copy()
    adata.var_names_make_unique()
    if args.keep_counts_layer:
        adata.layers["counts_before_norm"] = adata.X.copy()

    hvg_flavor_used = "seurat_v3"
    batch_key = (
        donor_col
        if donor_col
        and donor_col in adata.obs.columns
        and adata.obs[donor_col].nunique(dropna=True) > 1
        else None
    )
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=args.n_top_hvg,
            flavor="seurat_v3",
            batch_key=batch_key,
        )
    except Exception as error:
        print(f"[{prefix}] seurat_v3 HVG failed; using seurat. Reason: {error}")
        hvg_flavor_used = "seurat"
        temporary = adata.copy()
        sc.pp.normalize_total(temporary, target_sum=1e4)
        sc.pp.log1p(temporary)
        sc.pp.highly_variable_genes(
            temporary,
            n_top_genes=args.n_top_hvg,
            flavor="seurat",
            batch_key=batch_key,
        )
        adata.var["highly_variable"] = (
            temporary.var["highly_variable"]
            .reindex(adata.var_names)
            .fillna(False)
            .to_numpy()
        )
        del temporary

    n_hvg = int(adata.var["highly_variable"].sum())
    if n_hvg < args.min_hvg_for_clustering:
        if args.allow_all_genes_if_few_hvg:
            print(
                f"[{prefix}] Only {n_hvg} HVGs found; using all "
                f"{adata.n_vars} genes for dataset13 clustering."
            )
            adata.var["highly_variable"] = True
            n_hvg = int(adata.n_vars)
            hvg_flavor_used += "_too_few_use_all_genes"
        else:
            raise ValueError(
                f"Only {n_hvg} HVGs were found. Use "
                "--allow_all_genes_if_few_hvg for the 5000-feature matrix."
            )

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    if args.scale_hvg:
        try:
            sc.pp.scale(adata, max_value=10, zero_center=False)
        except TypeError:
            sc.pp.scale(adata, max_value=10)

    n_components = int(min(args.n_pcs, adata.n_obs - 1, n_hvg - 1))
    if n_components < 2:
        raise ValueError(f"Too few cells/HVGs for PCA: n_components={n_components}")
    sc.tl.pca(
        adata,
        n_comps=n_components,
        mask_var="highly_variable",
        svd_solver="arpack",
        zero_center=False,
    )

    representation = "X_pca"
    harmony_used = False
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
            print(f"[{prefix}] Harmony unavailable/failed; using PCA. Reason: {error}")

    n_pcs_neighbors = int(min(args.n_pcs_neighbors, n_components))
    sc.pp.neighbors(
        adata,
        n_neighbors=min(args.n_neighbors, max(2, adata.n_obs - 1)),
        n_pcs=n_pcs_neighbors,
        use_rep=representation,
        metric=args.neighbor_metric,
    )
    sc.tl.umap(adata, min_dist=args.umap_min_dist)
    try:
        sc.tl.leiden(
            adata,
            resolution=args.leiden_resolution,
            key_added="leiden",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    except Exception:
        sc.tl.leiden(
            adata,
            resolution=args.leiden_resolution,
            key_added="leiden",
        )

    adata.uns["03_age_quartile_cluster_params"] = {
        "hvg_flavor_used": hvg_flavor_used,
        "n_hvg": n_hvg,
        "n_pcs": n_components,
        "n_pcs_neighbors": n_pcs_neighbors,
        "use_rep": representation,
        "harmony_used": harmony_used,
        "harmony_key": batch_key,
        "n_neighbors": args.n_neighbors,
        "neighbor_metric": args.neighbor_metric,
        "umap_min_dist": args.umap_min_dist,
        "leiden_resolution": args.leiden_resolution,
    }

    figure_dir = outdir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    try:
        sc.pl.umap(adata, color=["leiden"], show=False)
        plt.savefig(
            figure_dir / f"{prefix}_umap_leiden.png",
            dpi=220,
            bbox_inches="tight",
        )
    finally:
        plt.close()
    try:
        sc.pl.pca_variance_ratio(adata, log=True, show=False)
        plt.savefig(
            figure_dir / f"{prefix}_pca_variance_ratio.png",
            dpi=220,
            bbox_inches="tight",
        )
    except Exception:
        pass
    finally:
        plt.close()
    return adata


def run_one_condition(
    adata_condition,
    dataset_compact: str,
    condition_label: str,
    condition_outdir: Path,
    donor_col: str | None,
    sample_col: str | None,
    disease_col: str | None,
    author_celltype_col: str | None,
    args: argparse.Namespace,
) -> dict:
    """Run the standalone Step 03 workflow for one age quartile."""
    condition_outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_compact}_{condition_label}"
    qc_cols = [
        "qc_n_genes",
        "qc_total_counts",
        "qc_pct_mt",
        "qc_doublet_score",
        "qc_exon_fraction",
        "qc_entropy",
    ]
    adata_condition.obs[qc_cols].describe().to_csv(
        condition_outdir / f"{prefix}_QC_metric_summary_before_filter.csv"
    )

    hard_keep, hard_reasons = apply_hard_qc(adata_condition, args)
    hard_reasons.to_csv(
        condition_outdir / f"{prefix}_hard_QC_cell_flags.csv"
    )
    adata_hard = adata_condition[hard_keep.to_numpy()].copy()
    if adata_hard.n_obs < args.min_cells_per_condition:
        raise ValueError(f"Too few cells after hard QC: {adata_hard.n_obs}")

    if args.use_mean_iqr_qc:
        mean_iqr_keep, mean_iqr_reasons, thresholds = apply_mean_iqr_qc(
            adata_hard, args
        )
    else:
        mean_iqr_keep = pd.Series(True, index=adata_hard.obs_names)
        mean_iqr_reasons = pd.DataFrame(
            {"keep_after_meanIQR_QC": True}, index=adata_hard.obs_names
        )
        thresholds = pd.DataFrame(
            [{"status": "meanIQR_QC_disabled_by_default"}]
        )
    mean_iqr_reasons.to_csv(
        condition_outdir / f"{prefix}_meanIQR_QC_cell_flags.csv"
    )
    thresholds.to_csv(
        condition_outdir / f"{prefix}_meanIQR_QC_thresholds.csv", index=False
    )
    adata_qc = adata_hard[mean_iqr_keep.to_numpy()].copy()

    removed = pd.concat(
        [
            pd.DataFrame(
                {
                    "cell_id": pd.Index(adata_condition.obs_names)[
                        ~hard_keep.to_numpy()
                    ].astype(str),
                    "remove_stage": "hard_QC",
                }
            ),
            pd.DataFrame(
                {
                    "cell_id": pd.Index(adata_hard.obs_names)[
                        ~mean_iqr_keep.to_numpy()
                    ].astype(str),
                    "remove_stage": "meanIQR_QC",
                }
            ),
        ],
        ignore_index=True,
    )
    removed.to_csv(
        condition_outdir / f"{prefix}_cells_removed_by_QC.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "step": "condition_start",
                "n_cells": int(adata_condition.n_obs),
                "percent_of_start": 100.0,
            },
            {
                "step": "after_hard_QC",
                "n_cells": int(adata_hard.n_obs),
                "percent_of_start": 100.0 * adata_hard.n_obs / adata_condition.n_obs,
            },
            {
                "step": "after_hard_QC_plus_meanIQR_QC",
                "n_cells": int(adata_qc.n_obs),
                "percent_of_start": 100.0 * adata_qc.n_obs / adata_condition.n_obs,
            },
        ]
    ).to_csv(condition_outdir / f"{prefix}_QC_stage_counts.csv", index=False)

    if adata_qc.n_obs < args.min_cells_per_condition:
        raise ValueError(f"Too few cells after mean-IQR QC: {adata_qc.n_obs}")
    clustered = preprocess_and_cluster(
        adata_qc,
        outdir=condition_outdir,
        prefix=prefix,
        donor_col=donor_col,
        args=args,
    )

    clustered.obs["leiden"].value_counts().sort_index().rename_axis(
        "leiden"
    ).reset_index(name="n_cells").to_csv(
        condition_outdir / f"{prefix}_cluster_counts.csv", index=False
    )
    if donor_col and donor_col in clustered.obs.columns:
        pd.crosstab(clustered.obs["leiden"], clustered.obs[donor_col]).to_csv(
            condition_outdir / f"{prefix}_clusters_vs_donor.csv"
        )
    if sample_col and sample_col in clustered.obs.columns:
        pd.crosstab(clustered.obs["leiden"], clustered.obs[sample_col]).to_csv(
            condition_outdir / f"{prefix}_clusters_vs_sample.csv"
        )
    if author_celltype_col and author_celltype_col in clustered.obs.columns:
        pd.crosstab(
            clustered.obs["leiden"], clustered.obs[author_celltype_col]
        ).to_csv(
            condition_outdir / f"{prefix}_clusters_vs_author_celltype.csv"
        )

    plot_columns = ["leiden"]
    for column in [disease_col, donor_col, sample_col, author_celltype_col]:
        if not column or column not in clustered.obs.columns or column in plot_columns:
            continue
        if (
            column in {donor_col, sample_col}
            and clustered.obs[column].nunique(dropna=True)
            > args.max_categories_to_plot
        ):
            continue
        plot_columns.append(column)
    try:
        sc.pl.umap(clustered, color=plot_columns, show=False, wspace=0.35)
        plt.savefig(
            condition_outdir / "figures" / f"{prefix}_umap_summary.png",
            dpi=220,
            bbox_inches="tight",
        )
    except Exception:
        pass
    finally:
        plt.close()

    if not args.keep_all_layers:
        keep_layers = {"counts_before_norm"} if args.keep_counts_layer else set()
        remove_large_unused_layers(clustered, keep_layers)
    write_gene_inventory(
        clustered,
        condition_outdir / f"{prefix}_gene_inventory.csv",
    )
    output_h5ad = condition_outdir / f"{prefix}_clustered.h5ad"
    clustered.write_h5ad(output_h5ad)

    summary = {
        "dataset": dataset_compact,
        "condition": condition_label,
        "n_cells_start": int(adata_condition.n_obs),
        "n_cells_after_hard_QC": int(adata_hard.n_obs),
        "n_cells_after_meanIQR_QC": int(adata_qc.n_obs),
        "n_removed_total": int(adata_condition.n_obs - adata_qc.n_obs),
        "percent_final_kept": 100.0 * adata_qc.n_obs / adata_condition.n_obs,
        "n_clusters": int(clustered.obs["leiden"].nunique()),
        "leiden_resolution": args.leiden_resolution,
        "output_h5ad": str(output_h5ad),
    }
    pd.DataFrame([summary]).to_csv(
        condition_outdir / f"{prefix}_run_summary.csv", index=False
    )
    with open(
        condition_outdir / f"{prefix}_run_summary.txt", "w", encoding="utf-8"
    ) as handle:
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")
    return summary


def normalize_identifier(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip()


def valid_identifier_mask(values: pd.Series) -> pd.Series:
    normalized = normalize_identifier(values).str.lower()
    return ~normalized.isin({"", "nan", "none", "null", "na", "n/a", "unknown"})


def parse_range_midpoint(values: pd.Series) -> pd.Series:
    """Parse values such as 50-55 or 60-69 into numeric midpoints."""
    text = values.astype(str).str.strip()
    extracted = text.str.extract(
        r"(?P<lower>\d+(?:\.\d+)?)\s*[-\u2013\u2014]\s*(?P<upper>\d+(?:\.\d+)?)"
    )
    lower = pd.to_numeric(extracted["lower"], errors="coerce")
    upper = pd.to_numeric(extracted["upper"], errors="coerce")
    midpoint = (lower + upper) / 2.0
    midpoint.loc[upper < lower] = np.nan
    return midpoint


def create_unique_donor_key(obs: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, dict]:
    """
    Namespace donor IDs by source project.

    The inspected dataset13 input has donor labels reused across projects, so
    donor_id alone is not a globally unique biological-replicate identifier.
    """
    donor_col = first_existing_column(
        obs,
        args.donor_col,
        ["donor_id", "Donor", "donor", "patient_id", "patient", "subject"],
    )
    donor = normalize_identifier(obs[donor_col])
    donor_valid = valid_identifier_mask(obs[donor_col])

    namespace_col: str | None = None
    namespace = pd.Series("dataset13", index=obs.index, dtype="object")
    if args.donor_namespace_col:
        if args.donor_namespace_col not in obs.columns:
            raise KeyError(
                f"Requested donor namespace column '{args.donor_namespace_col}' is absent."
            )
        namespace_col = args.donor_namespace_col
    else:
        for candidate in ["Project", "project", "source_id", "rds_object_name"]:
            if candidate in obs.columns:
                namespace_col = candidate
                break

    if namespace_col is not None:
        namespace = normalize_identifier(obs[namespace_col])
        namespace_valid = valid_identifier_mask(obs[namespace_col])
        namespace = namespace.where(namespace_valid, "UNKNOWN_SOURCE")

    unique_key = namespace + "::" + donor
    unique_key = unique_key.where(donor_valid, np.nan)

    donor_source_pairs = pd.DataFrame(
        {"donor_id_original": donor, "donor_source": namespace}
    ).loc[donor_valid].drop_duplicates()
    sources_per_raw_id = donor_source_pairs.groupby("donor_id_original")[
        "donor_source"
    ].nunique()

    diagnostics = {
        "donor_column": donor_col,
        "donor_namespace_column": namespace_col or "constant:dataset13",
        "n_raw_donor_labels": int(donor.loc[donor_valid].nunique()),
        "n_unique_namespaced_donors": int(unique_key.dropna().nunique()),
        "n_raw_donor_labels_reused_across_sources": int((sources_per_raw_id > 1).sum()),
    }
    return unique_key, diagnostics


def resolve_donor_ages(
    obs: pd.DataFrame,
    donor_key: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    """Resolve one exact or range-midpoint age for every namespaced donor."""
    exact_col = first_existing_column(
        obs,
        args.age_exact_col,
        ["age", "Age", "age_years", "Age_years"],
    )
    exact_age = pd.to_numeric(obs[exact_col], errors="coerce")

    if args.age_range_cols:
        range_candidates = [
            value.strip() for value in args.age_range_cols.split(",") if value.strip()
        ]
    else:
        range_candidates = [
            "Age_range",
            "age_range",
            "Age_range_normalized",
            "age_10yr_30plus",
        ]

    existing_range_cols = [column for column in range_candidates if column in obs.columns]
    range_midpoint = pd.Series(np.nan, index=obs.index, dtype=float)
    range_source = pd.Series("", index=obs.index, dtype="object")
    for column in existing_range_cols:
        parsed = parse_range_midpoint(obs[column])
        take = range_midpoint.isna() & parsed.notna()
        range_midpoint.loc[take] = parsed.loc[take]
        range_source.loc[take] = column

    work = pd.DataFrame(
        {
            UNIQUE_DONOR_COL: donor_key,
            "age_exact": exact_age,
            "age_range_midpoint": range_midpoint,
            "age_range_source": range_source,
        },
        index=obs.index,
    ).dropna(subset=[UNIQUE_DONOR_COL])

    grouped = work.groupby(UNIQUE_DONOR_COL, observed=True)
    exact_stats = grouped["age_exact"].agg(
        exact_min="min", exact_max="max", exact_n="count"
    )
    range_stats = grouped["age_range_midpoint"].agg(
        range_min="min", range_max="max", range_n="count"
    )
    donor_table = exact_stats.join(range_stats, how="outer")

    exact_conflict = (
        donor_table["exact_n"].gt(0)
        & (donor_table["exact_max"] - donor_table["exact_min"]).abs().gt(1e-8)
    )
    range_conflict = (
        donor_table["exact_n"].eq(0)
        & donor_table["range_n"].gt(0)
        & (donor_table["range_max"] - donor_table["range_min"]).abs().gt(1e-8)
    )
    conflicts = donor_table.loc[exact_conflict | range_conflict]
    if not conflicts.empty:
        preview = conflicts.head(20).to_string()
        raise ValueError(
            "At least one biological donor has conflicting age records. "
            "Resolve these records before calculating quartiles.\n" + preview
        )

    donor_table["age_numeric_resolved"] = donor_table["exact_min"].where(
        donor_table["exact_n"].gt(0), donor_table["range_min"]
    )
    donor_table["age_source"] = np.where(
        donor_table["exact_n"].gt(0), "exact", "reported_range_midpoint"
    )
    donor_table.loc[
        donor_table["age_numeric_resolved"].isna(), "age_source"
    ] = "missing"

    first_metadata_columns = [
        column
        for column in [
            "donor_id",
            "Donor",
            "Project",
            "source_id",
            "disease",
            "Disease",
            "sex",
            "Sex",
            "Age_range",
            "Age_range_normalized",
            "age_10yr_30plus",
        ]
        if column in obs.columns
    ]
    metadata = obs[first_metadata_columns].copy()
    metadata[UNIQUE_DONOR_COL] = donor_key
    metadata = metadata.dropna(subset=[UNIQUE_DONOR_COL])
    metadata_first = metadata.groupby(UNIQUE_DONOR_COL, observed=True).first()
    donor_table = donor_table.join(metadata_first, how="left")

    cell_counts = donor_key.value_counts(dropna=True)
    donor_table["n_cells_before_QC"] = donor_table.index.map(cell_counts).astype(int)
    donor_table = donor_table.reset_index()

    diagnostics = {
        "age_exact_column": exact_col,
        "age_range_columns_checked": existing_range_cols,
        "n_donors_with_exact_age": int((donor_table["age_source"] == "exact").sum()),
        "n_donors_with_range_midpoint_age": int(
            (donor_table["age_source"] == "reported_range_midpoint").sum()
        ),
        "n_donors_with_missing_age": int(
            donor_table["age_numeric_resolved"].isna().sum()
        ),
    }
    return donor_table, diagnostics


def assign_equal_count_quartiles(
    donor_table: pd.DataFrame,
    min_age: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign Q1-Q4 after sorting donors by age and deterministic donor key."""
    eligible = donor_table.loc[
        donor_table["age_numeric_resolved"].notna()
        & donor_table["age_numeric_resolved"].ge(min_age)
    ].copy()
    if eligible.shape[0] < 4:
        raise ValueError(
            f"Only {eligible.shape[0]} donors have resolved age >= {min_age}; "
            "four quartiles cannot be constructed."
        )

    eligible = eligible.sort_values(
        ["age_numeric_resolved", UNIQUE_DONOR_COL], kind="mergesort"
    ).reset_index(drop=True)
    deterministic_rank = pd.Series(
        np.arange(1, eligible.shape[0] + 1), index=eligible.index, dtype=float
    )
    eligible["age_quartile"] = pd.qcut(
        deterministic_rank,
        q=4,
        labels=QUARTILE_LABELS,
    ).astype(str)
    eligible["age_quartile_rank_tiebreak"] = deterministic_rank.astype(int)
    eligible["quartile_assignment_method"] = (
        "equal_count_rank_by_age_then_namespaced_donor_id"
    )

    summary = (
        eligible.groupby("age_quartile", observed=True)
        .agg(
            n_donors=(UNIQUE_DONOR_COL, "size"),
            n_cells_before_QC=("n_cells_before_QC", "sum"),
            age_min=("age_numeric_resolved", "min"),
            age_max=("age_numeric_resolved", "max"),
            n_exact_age=("age_source", lambda values: int((values == "exact").sum())),
            n_range_midpoint_age=(
                "age_source",
                lambda values: int((values == "reported_range_midpoint").sum()),
            ),
        )
        .reindex(QUARTILE_LABELS)
        .reset_index()
    )
    summary["next_quartile"] = summary["age_quartile"].shift(-1)
    summary["next_quartile_age_min"] = summary["age_min"].shift(-1)
    summary["boundary_tie_split_with_next_quartile"] = np.isclose(
        summary["age_max"],
        summary["next_quartile_age_min"],
        equal_nan=False,
    )
    summary["boundary_age"] = np.where(
        summary["boundary_tie_split_with_next_quartile"],
        summary["age_max"],
        np.nan,
    )
    summary["n_donors_at_boundary_age_in_this_quartile"] = 0
    summary["n_donors_at_boundary_age_in_next_quartile"] = 0
    for row_index in range(summary.shape[0] - 1):
        if not bool(
            summary.loc[row_index, "boundary_tie_split_with_next_quartile"]
        ):
            continue
        boundary_age = float(summary.loc[row_index, "boundary_age"])
        this_label = str(summary.loc[row_index, "age_quartile"])
        next_label = str(summary.loc[row_index, "next_quartile"])
        at_boundary = np.isclose(
            eligible["age_numeric_resolved"].astype(float), boundary_age
        )
        summary.loc[
            row_index, "n_donors_at_boundary_age_in_this_quartile"
        ] = int(
            (at_boundary & eligible["age_quartile"].eq(this_label)).sum()
        )
        summary.loc[
            row_index, "n_donors_at_boundary_age_in_next_quartile"
        ] = int(
            (at_boundary & eligible["age_quartile"].eq(next_label)).sum()
        )

    summary["age_quartile_range_label"] = summary.apply(
        lambda row: (
            f"{row['age_quartile']} "
            f"({_format_age_value(row['age_min'])}-{_format_age_value(row['age_max'])} "
            f"years; n={int(row['n_donors'])} donors)"
        ),
        axis=1,
    )
    range_label_map = summary.set_index("age_quartile")[
        "age_quartile_range_label"
    ]
    eligible["age_quartile_range_label"] = eligible["age_quartile"].map(
        range_label_map
    )
    return eligible, summary


def _format_age_value(value: float) -> str:
    """Format inferred ages without hiding meaningful half-year midpoints."""
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def write_gene_inventory(adata, output_csv: Path) -> None:
    """Export feature names and unambiguous gene/HVG metadata."""
    inventory = pd.DataFrame(
        {"gene_name": pd.Index(adata.var_names).astype(str).to_numpy()}
    )
    preferred_columns = [
        "gene_symbol",
        "gene_original",
        "mt",
        "highly_variable",
        "highly_variable_rank",
        "means",
        "variances",
        "variances_norm",
        "highly_variable_nbatches",
    ]
    for column in preferred_columns:
        if column not in adata.var.columns:
            continue
        output_column = str(column)
        if output_column == "gene_name":
            output_column = "gene_name_from_var_metadata"
        inventory[output_column] = adata.var[column].to_numpy()
    inventory.to_csv(output_csv, index=False)


def write_quartile_audit_tables(
    donor_assignments: pd.DataFrame,
    quartile_summary: pd.DataFrame,
    dataset_outdir: Path,
) -> None:
    donor_assignments.to_csv(
        dataset_outdir / "dataset13_age_quartile_donor_assignments.csv", index=False
    )
    quartile_summary.to_csv(
        dataset_outdir / "dataset13_age_quartile_summary_before_QC.csv", index=False
    )

    for column, output_name in [
        ("disease", "dataset13_age_quartile_donor_counts_by_disease.csv"),
        ("Disease", "dataset13_age_quartile_donor_counts_by_Disease_raw.csv"),
        ("sex", "dataset13_age_quartile_donor_counts_by_sex.csv"),
        ("Sex", "dataset13_age_quartile_donor_counts_by_Sex_raw.csv"),
        ("Project", "dataset13_age_quartile_donor_counts_by_project.csv"),
    ]:
        if column not in donor_assignments.columns:
            continue
        table = pd.crosstab(
            donor_assignments["age_quartile"],
            donor_assignments[column].astype(str),
            dropna=False,
        ).reindex(QUARTILE_LABELS, fill_value=0)
        table.to_csv(dataset_outdir / output_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone dataset13 NF, LV/AP-only, age>=30 donor-quartile "
            "Step 03 with mean-IQR QC and clustering."
        )
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--condition_labels",
        default="",
        help="Optional comma-separated subset of Q1,Q2,Q3,Q4. Empty runs all.",
    )

    # Cohort restriction. Filtering happens before donor quartile assignment.
    parser.add_argument(
        "--disease_col",
        default="disease",
        help="Disease column used to retain the NF cohort.",
    )
    parser.add_argument(
        "--region_col",
        default="Standardized_region",
        help=(
            "Standardized tissue/region column. Only LV and AP are retained, "
            "and filtering occurs before donor quartile assignment and QC."
        ),
    )

    # Donor-level age grouping.
    parser.add_argument("--min_age", type=float, default=30.0)
    parser.add_argument("--donor_col", default="donor_id")
    parser.add_argument(
        "--donor_namespace_col",
        default="Project",
        help=(
            "Column used to namespace reused donor labels. Default: Project. "
            "Use an empty string only after proving donor_id is globally unique."
        ),
    )
    parser.add_argument(
        "--age_exact_col",
        default="age",
        help="Exact/numeric age column. Non-numeric values fall back to age ranges.",
    )
    parser.add_argument(
        "--age_range_cols",
        default="Age_range,Age_range_normalized,age_10yr_30plus",
        help="Comma-separated age-range columns, checked from left to right.",
    )

    # Dataset13 metadata used by the shared QC functions.
    parser.add_argument("--sample_col", default="biosample_id")
    parser.add_argument("--author_celltype_col", default="cell_type_leiden0.6")
    parser.add_argument("--gene_count_col", default="n_genes")
    parser.add_argument("--count_col", default="n_counts")
    parser.add_argument("--mito_col", default="percent_mito")
    parser.add_argument("--doublet_score_col", default="")
    parser.add_argument("--exon_prop_col", default="")
    parser.add_argument("--entropy_col", default="")
    parser.add_argument("--raw_count_layer", default="")

    # Same hard QC defaults as the existing Step 03.
    parser.add_argument("--min_genes", type=float, default=150)
    parser.add_argument("--min_counts", type=float, default=150)
    parser.add_argument("--max_mito_percent", type=float, default=5.0)
    parser.add_argument("--min_cells_per_condition", type=int, default=200)

    # Same optional mean-IQR QC. It remains disabled by default.
    parser.add_argument("--iqr_multiplier", type=float, default=1.5)
    parser.add_argument("--use_mean_iqr_qc", action="store_true", default=False)
    parser.add_argument("--two_sided_all_qc_outliers", action="store_true")
    parser.add_argument("--compute_entropy", action="store_true")

    # Same clustering parameters as the existing aging Step 03.
    parser.add_argument("--n_top_hvg", type=int, default=2000)
    parser.add_argument("--min_hvg_for_clustering", type=int, default=20)
    parser.add_argument(
        "--allow_all_genes_if_few_hvg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default true for the limited 5000-feature dataset13 matrix.",
    )
    parser.add_argument("--n_pcs", type=int, default=100)
    parser.add_argument("--n_pcs_neighbors", type=int, default=100)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--neighbor_metric", default="cosine")
    parser.add_argument("--umap_min_dist", type=float, default=0.2)
    parser.add_argument("--leiden_resolution", type=float, default=0.6)
    parser.add_argument("--scale_hvg", action="store_true")
    parser.add_argument("--no_harmony", action="store_false", dest="use_harmony")
    parser.set_defaults(use_harmony=True)

    parser.add_argument("--keep_counts_layer", action="store_true")
    parser.add_argument("--keep_all_layers", action="store_true")
    parser.add_argument("--max_categories_to_plot", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = find_project_dir()

    h5ad_path = resolve_project_path(project_dir, args.h5ad)
    output_root = resolve_project_path(project_dir, args.output_root)
    dataset_outdir = output_root / DATASET_FOLDER
    dataset_outdir.mkdir(parents=True, exist_ok=True)

    if not h5ad_path.exists():
        raise FileNotFoundError(f"Input h5ad was not found: {h5ad_path}")

    print("=" * 90)
    print("Dataset13 NF, LV/AP-only adult-aging donor-quartile Step 03")
    print("Input:", h5ad_path)
    print("Output:", dataset_outdir)
    print("Minimum age:", args.min_age)
    print("Required disease:", REQUIRED_DISEASE)
    print("Required standardized regions:", ", ".join(REQUIRED_STANDARDIZED_REGIONS))
    print("=" * 90)

    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()

    n_cells_input_before_disease_filter = int(adata.n_obs)
    disease_filter_col = first_existing_column(
        adata.obs,
        args.disease_col,
        ["disease", "Disease"],
    )
    requested_disease = REQUIRED_DISEASE
    disease_values = normalize_identifier(adata.obs[disease_filter_col])
    disease_keep_mask = disease_values.str.casefold().eq(
        requested_disease.casefold()
    )
    n_cells_kept_by_disease = int(disease_keep_mask.sum())
    if n_cells_kept_by_disease == 0:
        available = disease_values.value_counts().head(20).to_dict()
        raise ValueError(
            f"No cells matched {disease_filter_col}={requested_disease!r}. "
            f"Observed values and cell counts: {available}"
        )

    disease_filter_counts = (
        disease_values.value_counts(dropna=False)
        .rename_axis("disease_value")
        .reset_index(name="n_cells_before_filter")
    )
    disease_filter_counts["kept_for_aging_analysis"] = disease_filter_counts[
        "disease_value"
    ].str.casefold().eq(requested_disease.casefold())
    disease_filter_counts.to_csv(
        dataset_outdir / "dataset13_disease_filter_cell_counts.csv", index=False
    )

    # Tissue restriction is deliberately applied before donor construction,
    # quartile assignment, QC thresholds, HVG selection, and clustering.
    # "Transmural LV Apex" is already represented as AP in Standardized_region.
    n_cells_after_disease_before_region_filter = n_cells_kept_by_disease
    region_filter_col = first_existing_column(
        adata.obs,
        args.region_col,
        ["Standardized_region", "standardized_region"],
    )
    region_values_all = normalize_identifier(
        adata.obs[region_filter_col]
    ).str.upper()
    allowed_regions = set(REQUIRED_STANDARDIZED_REGIONS)
    region_keep_mask = region_values_all.isin(allowed_regions)
    combined_cohort_mask = disease_keep_mask & region_keep_mask
    n_cells_kept_by_region = int(combined_cohort_mask.sum())
    if n_cells_kept_by_region == 0:
        available = (
            region_values_all.loc[disease_keep_mask]
            .value_counts(dropna=False)
            .head(30)
            .to_dict()
        )
        raise ValueError(
            f"No cells matched {region_filter_col} in "
            f"{REQUIRED_STANDARDIZED_REGIONS}. Observed values/counts: {available}"
        )

    region_filter_counts = (
        region_values_all.loc[disease_keep_mask]
        .value_counts(dropna=False)
        .rename_axis("standardized_region_value")
        .reset_index(name="n_NF_cells_before_region_filter")
    )
    region_filter_counts["kept_for_aging_analysis"] = region_filter_counts[
        "standardized_region_value"
    ].isin(allowed_regions)
    region_filter_counts.to_csv(
        dataset_outdir / "dataset13_region_filter_cell_counts.csv", index=False
    )

    if "X_umap" in adata.obsm:
        adata.obsm["X_umap_author_ref"] = adata.obsm["X_umap"].copy()
        del adata.obsm["X_umap"]

    # Subset the large AnnData only once to avoid a second expensive NF copy.
    adata = adata[combined_cohort_mask.to_numpy()].copy()
    adata.obs["aging_disease_filter"] = requested_disease
    adata.obs["aging_region_filter"] = "LV_or_AP"
    adata.obs["aging_standardized_region"] = (
        normalize_identifier(adata.obs[region_filter_col]).str.upper().values
    )
    n_cells_excluded_non_requested_disease = int(
        n_cells_input_before_disease_filter - n_cells_kept_by_disease
    )
    n_cells_excluded_non_LV_AP_region = int(
        n_cells_after_disease_before_region_filter - adata.n_obs
    )
    print(
        f"Disease filter {disease_filter_col}={requested_disease}: "
        f"kept {n_cells_kept_by_disease:,}/{n_cells_input_before_disease_filter:,} cells"
    )
    print(
        f"Region filter {region_filter_col} in {REQUIRED_STANDARDIZED_REGIONS}: "
        f"kept {adata.n_obs:,}/{n_cells_after_disease_before_region_filter:,} NF cells"
    )
    matrix_source = choose_count_matrix(adata, args)

    unique_donor_key, donor_diagnostics = create_unique_donor_key(adata.obs, args)
    donor_table, age_diagnostics = resolve_donor_ages(
        adata.obs, unique_donor_key, args
    )
    donor_assignments, quartile_summary = assign_equal_count_quartiles(
        donor_table, min_age=args.min_age
    )
    write_quartile_audit_tables(
        donor_assignments, quartile_summary, dataset_outdir
    )
    quartile_map = donor_assignments.set_index(UNIQUE_DONOR_COL)["age_quartile"]
    quartile_range_label_map = donor_assignments.set_index(UNIQUE_DONOR_COL)[
        "age_quartile_range_label"
    ]
    age_map = donor_assignments.set_index(UNIQUE_DONOR_COL)["age_numeric_resolved"]
    age_source_map = donor_assignments.set_index(UNIQUE_DONOR_COL)["age_source"]

    eligible_cell_mask = unique_donor_key.isin(quartile_map.index).to_numpy()
    n_cells_before_age_filter = int(adata.n_obs)
    n_cells_excluded_by_age_or_metadata = int((~eligible_cell_mask).sum())
    if not eligible_cell_mask.all():
        adata = adata[eligible_cell_mask].copy()
        unique_donor_key = unique_donor_key.loc[adata.obs_names]

    adata.obs[UNIQUE_DONOR_COL] = unique_donor_key.astype(str).values
    adata.obs["age_numeric_resolved"] = (
        adata.obs[UNIQUE_DONOR_COL].map(age_map).astype(float).values
    )
    adata.obs["age_source"] = (
        adata.obs[UNIQUE_DONOR_COL].map(age_source_map).astype(str).values
    )
    quartiles = adata.obs[UNIQUE_DONOR_COL].map(quartile_map).astype(str)
    adata.obs["age_quartile"] = pd.Categorical(
        quartiles, categories=QUARTILE_LABELS, ordered=True
    )
    adata.obs["age_quartile_range_label"] = adata.obs[
        UNIQUE_DONOR_COL
    ].map(quartile_range_label_map).astype(str).values

    if "age_10yr_30plus" in adata.obs.columns:
        adata.obs["age_decade_original"] = adata.obs[
            "age_10yr_30plus"
        ].astype(str).values
    if "condition" in adata.obs.columns:
        adata.obs["condition_before_age_quartile"] = adata.obs[
            "condition"
        ].astype(str).values
    adata.obs["condition"] = adata.obs["age_quartile"].astype(str).values

    qc_info = compute_qc_metrics(adata, args)

    dataset_info = {
        "script_version": SCRIPT_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_folder": DATASET_FOLDER,
        "input_h5ad": str(h5ad_path),
        "matrix_source": matrix_source,
        "disease_filter_column": disease_filter_col,
        "disease_filter_value": requested_disease,
        "region_filter_column": region_filter_col,
        "region_filter_values": ";".join(REQUIRED_STANDARDIZED_REGIONS),
        "n_cells_input": n_cells_input_before_disease_filter,
        "n_cells_excluded_non_requested_disease": n_cells_excluded_non_requested_disease,
        "n_cells_after_disease_filter_before_region_filter": n_cells_after_disease_before_region_filter,
        "n_cells_excluded_non_LV_AP_region": n_cells_excluded_non_LV_AP_region,
        "n_cells_after_region_filter_before_age_filter": n_cells_before_age_filter,
        "n_cells_excluded_by_age_or_missing_donor_age": n_cells_excluded_by_age_or_metadata,
        "n_cells_age_eligible": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "condition_col": "age_quartile",
        "min_age": args.min_age,
        "n_quartile_eligible_donors": int(donor_assignments.shape[0]),
        **donor_diagnostics,
        **age_diagnostics,
        **qc_info,
    }
    pd.DataFrame([dataset_info]).to_csv(
        dataset_outdir / "dataset13_dataset_input_summary.csv", index=False
    )

    qc_cols = [
        "qc_n_genes",
        "qc_total_counts",
        "qc_pct_mt",
        "qc_doublet_score",
        "qc_exon_fraction",
        "qc_entropy",
    ]
    adata.obs[qc_cols].describe().to_csv(
        dataset_outdir / "dataset13_QC_metric_summary_before_quartile_split.csv"
    )

    cell_counts = (
        adata.obs["age_quartile"]
        .value_counts(sort=False)
        .reindex(QUARTILE_LABELS, fill_value=0)
    )
    donor_counts = (
        adata.obs[["age_quartile", UNIQUE_DONOR_COL]]
        .drop_duplicates()["age_quartile"]
        .value_counts(sort=False)
        .reindex(QUARTILE_LABELS, fill_value=0)
    )
    pd.DataFrame(
        {
            "condition": QUARTILE_LABELS,
            "n_cells_before_QC": [int(cell_counts[label]) for label in QUARTILE_LABELS],
            "n_donors_before_QC": [int(donor_counts[label]) for label in QUARTILE_LABELS],
        }
    ).to_csv(dataset_outdir / "dataset13_condition_counts_before.csv", index=False)
    pd.crosstab(
        adata.obs["age_quartile"],
        adata.obs["aging_standardized_region"],
        dropna=False,
    ).reindex(QUARTILE_LABELS, fill_value=0).to_csv(
        dataset_outdir / "dataset13_age_quartile_cell_counts_by_region.csv"
    )

    if args.condition_labels.strip():
        wanted = {
            value.strip()
            for value in args.condition_labels.replace(";", ",").split(",")
            if value.strip()
        }
        invalid = sorted(wanted - set(QUARTILE_LABELS))
        if invalid:
            raise ValueError(
                f"Unknown quartile labels: {invalid}. Expected Q1,Q2,Q3,Q4."
            )
        conditions = [label for label in QUARTILE_LABELS if label in wanted]
    else:
        conditions = QUARTILE_LABELS

    summaries: list[dict] = []
    for condition in conditions:
        condition_mask = (
            adata.obs["age_quartile"].astype(str).to_numpy() == condition
        )
        n_cells = int(condition_mask.sum())
        n_donors = int(
            adata.obs.loc[condition_mask, UNIQUE_DONOR_COL].nunique(dropna=True)
        )
        condition_outdir = dataset_outdir / condition

        print("\n" + "-" * 90)
        print(
            f"Quartile {condition}: {n_donors:,} donors, "
            f"{n_cells:,} cells before QC"
        )

        if n_cells < args.min_cells_per_condition:
            summaries.append(
                {
                    "dataset_id": DATASET_ID,
                    "dataset_folder": DATASET_FOLDER,
                    "condition": condition,
                    "status": "skipped_too_few_cells_before_QC",
                    "n_cells_before_qc": n_cells,
                    "n_donors_before_qc": n_donors,
                }
            )
            continue

        try:
            summary = run_one_condition(
                adata[condition_mask].copy(),
                dataset_compact=DATASET_FOLDER,
                condition_label=condition,
                condition_outdir=condition_outdir,
                donor_col=UNIQUE_DONOR_COL,
                sample_col=args.sample_col if args.sample_col in adata.obs.columns else None,
                disease_col="age_quartile",
                author_celltype_col=(
                    args.author_celltype_col
                    if args.author_celltype_col in adata.obs.columns
                    else None
                ),
                args=args,
            )
            summary.update(
                {
                    "dataset_id": DATASET_ID,
                    "dataset_folder": DATASET_FOLDER,
                    "condition": condition,
                    "n_donors_before_qc": n_donors,
                    "status": "completed",
                }
            )
            summaries.append(summary)
        except Exception as error:
            print(f"ERROR in dataset13 {condition}: {error}")
            traceback.print_exc()
            condition_outdir.mkdir(parents=True, exist_ok=True)
            with open(
                condition_outdir / f"dataset13_{condition}_ERROR.txt",
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(str(error) + "\n\n" + traceback.format_exc())
            summaries.append(
                {
                    "dataset_id": DATASET_ID,
                    "dataset_folder": DATASET_FOLDER,
                    "condition": condition,
                    "status": "error",
                    "error": str(error),
                    "n_cells_before_qc": n_cells,
                    "n_donors_before_qc": n_donors,
                }
            )

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(
        dataset_outdir / "dataset13_dataset_run_summary.csv", index=False
    )
    summary_frame.to_csv(
        output_root / "03_master_run_summary_age_quartile_QC.csv", index=False
    )

    config = {
        "script_version": SCRIPT_VERSION,
        "implementation": "standalone_no_base_script_dependency",
        "project_dir": str(project_dir),
        "required_standardized_regions": REQUIRED_STANDARDIZED_REGIONS,
        **vars(args),
        **donor_diagnostics,
        **age_diagnostics,
    }
    with open(
        output_root / "03_age_quartile_run_config.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, indent=2, default=str)

    print("\nDone.")
    print("Quartile donor assignments:")
    print(dataset_outdir / "dataset13_age_quartile_donor_assignments.csv")
    print("Master summary:")
    print(output_root / "03_master_run_summary_age_quartile_QC.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
