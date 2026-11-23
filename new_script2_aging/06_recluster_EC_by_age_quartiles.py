#!/usr/bin/env python3
r"""
Step 06: select and independently recluster endothelial-cell candidates in each
dataset13 NF age quartile.

This is a standalone script. It does not import or execute any earlier script.

Default selection rule
----------------------
The completed Step 05 cluster annotation table is read, and every source
all-cell Leiden cluster with ``celltypist_label_is_endothelial == True`` is
selected. This includes lymphatic endothelial (EC8_ln) clusters and, crucially,
the Q2 EC7_endocardial-majority cluster that contains the raw EC8_ln minority.

Each quartile is then handled separately:

1. Validate dataset13, NF-only status, quartile identity, and Step 05 coverage.
2. Subset all cells in the selected endothelial-related source clusters.
3. Preserve the source all-cell Leiden and both raw and majority-vote Step 05
   CellTypist labels.
4. Recompute HVGs, PCA, neighbors, Leiden clusters, and UMAP within the EC subset.
5. Rerun the Healthy Adult Heart CellTypist model on the EC-only expression,
   using the new EC Leiden clusters for fresh majority voting.
6. Mark EC-only clusters with majority label EC8_ln as final LEC candidates.
7. Export the annotated EC-only h5ad, LEC-cell manifest, cluster counts,
   marker summaries, top markers, and brief figures.

Default inputs
--------------
new_result2_aging/03/dataset13/Q*/dataset13_Q*_clustered.h5ad
new_result2_aging/05_marker_celltypist/dataset13/

Default outputs
---------------
new_result2_aging/06_EC_recluster/dataset13/

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_aging\06_recluster_EC_by_age_quartiles.py

Use ``--overwrite`` only when intentionally replacing existing Step 06 h5ad
outputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "06_dataset13_NF_age_quartile_EC_recluster_majority_2026_08_06"
DATASET = "dataset13"
REQUIRED_DISEASE = "NF"
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]

DEFAULT_INPUT_ROOT = "new_result2_aging/03/dataset13"
DEFAULT_STEP05_ROOT = "new_result2_aging/05_marker_celltypist/dataset13"
DEFAULT_STEP05_SUMMARY = (
    "new_result2_aging/05_marker_celltypist/dataset13/"
    "05_all_quartiles_cluster_annotation_summary.csv"
)
DEFAULT_OUTPUT_ROOT = "new_result2_aging/06_EC_recluster/dataset13"
DEFAULT_CELLTYPIST_MODEL = "resources/celltypist/Healthy_Adult_Heart.pkl"

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
MARKER_GENES = list(
    dict.fromkeys(
        ENDOTHELIAL_MARKERS
        + LEC_MARKERS
        + EC_SUBTYPE_MARKERS
        + NEGATIVE_LINEAGE_MARKERS
    )
)


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select Step 05 endothelial-related all-cell clusters and recluster "
            "their cells independently within NF age quartiles Q1-Q4."
        )
    )
    parser.add_argument("--input_root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--step05_root", default=DEFAULT_STEP05_ROOT)
    parser.add_argument("--step05_summary", default=DEFAULT_STEP05_SUMMARY)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--celltypist_model",
        default=DEFAULT_CELLTYPIST_MODEL,
        help="Heart CellTypist model used for the fresh EC-only majority vote.",
    )
    parser.add_argument(
        "--min_model_genes_present",
        type=int,
        default=500,
        help="Stop if fewer model genes are available for CellTypist.",
    )
    parser.add_argument(
        "--quartiles",
        nargs="+",
        choices=QUARTILES,
        default=QUARTILES,
        help="Quartiles to process; default: Q1 Q2 Q3 Q4.",
    )
    parser.add_argument(
        "--cluster_key",
        default="leiden",
        help="Source all-cell cluster column; default: leiden.",
    )
    parser.add_argument(
        "--expression_source",
        choices=["auto", "raw", "X"],
        default="auto",
        help="Expression source for EC reclustering; auto prefers adata.raw.",
    )
    parser.add_argument(
        "--preprocess_mode",
        choices=["auto", "counts", "already_log"],
        default="auto",
        help="auto detects whether normalization/log1p is required.",
    )
    parser.add_argument("--n_top_hvg", type=int, default=1500)
    parser.add_argument("--n_pcs", type=int, default=30)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--resolution", type=float, default=0.4)
    parser.add_argument("--min_selected_cells", type=int, default=50)
    parser.add_argument("--min_genes_per_cell", type=int, default=50)
    parser.add_argument("--min_cells_per_gene", type=int, default=3)
    parser.add_argument(
        "--marker_method",
        choices=["t-test", "t-test_overestim_var", "wilcoxon", "logreg"],
        default="t-test_overestim_var",
    )
    parser.add_argument("--marker_n_genes", type=int, default=500)
    parser.add_argument("--top_n_markers", type=int, default=25)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--skip_markers",
        action="store_true",
        help="Skip differential marker ranking; canonical marker summaries remain enabled.",
    )
    parser.add_argument(
        "--skip_figures",
        action="store_true",
        help="Skip all per-quartile and combined figures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-quartile Step 06 h5ad outputs.",
    )
    return parser.parse_args()


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def true_mask(values: pd.Series) -> pd.Series:
    return normalized_text(values).str.lower().isin({"true", "1", "yes", "y"})


def cluster_sort_key(value: object) -> tuple[int, float | str]:
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def safe_filename(value: object) -> str:
    text = re.sub(r"[\\/\s]+", "_", str(value).strip())
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "group"


def read_ec_selection(summary_path: Path, quartiles: list[str]) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Step 05 cluster summary not found: {summary_path}")
    table = pd.read_csv(summary_path, low_memory=False)
    required = [
        "dataset",
        "disease",
        "age_quartile",
        "cluster",
        "n_cells",
        "cluster_majority_cell_type",
        "celltypist_label_is_endothelial",
    ]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(
            f"Step 05 cluster summary is missing {missing}. Found: {list(table.columns)}"
        )

    dataset = normalized_text(table["dataset"]).str.lower()
    disease = normalized_text(table["disease"]).str.upper()
    quartile = normalized_text(table["age_quartile"])
    selected = table.loc[
        dataset.eq(DATASET.lower())
        & disease.eq(REQUIRED_DISEASE)
        & quartile.isin(quartiles)
        & true_mask(table["celltypist_label_is_endothelial"])
    ].copy()
    if selected.empty:
        raise ValueError(
            "No EC source clusters satisfy celltypist_label_is_endothelial == True."
        )

    selected["dataset"] = DATASET
    selected["disease"] = REQUIRED_DISEASE
    selected["age_quartile"] = normalized_text(selected["age_quartile"])
    selected["cluster"] = normalized_text(selected["cluster"])
    selected["selected_for_EC_recluster"] = True
    selected["EC_selection_rule"] = "step05_celltypist_label_is_endothelial_true"
    selected["source_step05_summary"] = str(summary_path)

    missing_quartiles = sorted(set(quartiles) - set(selected["age_quartile"]))
    if missing_quartiles:
        raise ValueError(f"No selected EC clusters for quartiles: {missing_quartiles}")
    return selected.sort_values(
        ["age_quartile", "cluster"],
        key=lambda series: (
            series.map({q: i for i, q in enumerate(QUARTILES)})
            if series.name == "age_quartile"
            else series.map(cluster_sort_key)
        ),
    ).reset_index(drop=True)


def validate_source_adata(adata, source: Path, quartile: str, cluster_key: str) -> None:
    if cluster_key not in adata.obs.columns:
        raise KeyError(
            f"Source cluster column '{cluster_key}' is absent from {source}."
        )
    if "disease" not in adata.obs.columns:
        raise KeyError(f"Cannot verify NF-only input; 'disease' is absent from {source}.")
    diseases = set(normalized_text(adata.obs["disease"]).str.upper().unique())
    if diseases != {REQUIRED_DISEASE}:
        raise ValueError(f"Expected only NF in {source}; found {sorted(diseases)}")
    if "age_quartile" not in adata.obs.columns:
        raise KeyError(f"'age_quartile' is absent from {source}.")
    observed_quartiles = set(normalized_text(adata.obs["age_quartile"]).unique())
    if observed_quartiles != {quartile}:
        raise ValueError(
            f"Expected only {quartile} in {source}; found {sorted(observed_quartiles)}"
        )


def read_step05_predictions(path: Path, expected_ids: pd.Index) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Step 05 cell prediction file not found: {path}")
    predictions = pd.read_csv(path, low_memory=False)
    required = [
        "cell_id",
        "celltypist_raw_label",
        "celltypist_majority_vote",
        "celltypist_max_probability",
    ]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Step 05 prediction file is missing columns: {missing}")
    predictions["cell_id"] = normalized_text(predictions["cell_id"])
    if predictions["cell_id"].duplicated().any():
        raise ValueError(f"Duplicate cell_id values in {path}")
    predictions = predictions.set_index("cell_id")
    missing_ids = expected_ids.difference(predictions.index)
    if len(missing_ids):
        examples = "; ".join(missing_ids[:5].astype(str))
        raise ValueError(
            f"Step 05 predictions lack {len(missing_ids):,} source cells. Examples: {examples}"
        )
    return predictions


def sparse_sample_values(matrix, max_values: int = 20000) -> np.ndarray:
    try:
        import scipy.sparse as sp

        if sp.issparse(matrix):
            values = matrix.data
            if values.size == 0:
                return np.array([], dtype=float)
            if values.size > max_values:
                rng = np.random.default_rng(0)
                values = values[rng.choice(values.size, size=max_values, replace=False)]
            return np.asarray(values, dtype=float)
    except Exception:
        pass

    array = np.asarray(matrix)
    if array.size == 0:
        return np.array([], dtype=float)
    values = array.ravel()
    values = values[np.isfinite(values) & (values != 0)]
    if values.size > max_values:
        rng = np.random.default_rng(0)
        values = values[rng.choice(values.size, size=max_values, replace=False)]
    return np.asarray(values, dtype=float)


def detect_preprocess_mode(adata) -> str:
    values = sparse_sample_values(adata.X)
    if values.size == 0:
        return "already_log"
    integer_like = float(np.mean(np.isclose(values, np.round(values)))) > 0.98
    if float(np.nanmin(values)) >= 0 and integer_like and float(np.nanmax(values)) > 20:
        return "counts"
    return "already_log"


def choose_expression_source(adata_subset, requested: str):
    if requested in {"auto", "raw"} and adata_subset.raw is not None:
        expression = adata_subset.raw.to_adata()
        expression.obs = adata_subset.obs.copy()
        expression.uns = dict(adata_subset.uns)
        return expression, "raw"
    if requested == "raw" and adata_subset.raw is None:
        warnings.warn("adata.raw is absent; falling back to X.")
    return adata_subset.copy(), "X"


def prepare_celltypist_query(adata, model_features: pd.Index):
    """Use unscaled EC-only expression and verify model-feature overlap."""
    import scanpy as sc
    import scipy.sparse as sp

    if adata.raw is None:
        raise ValueError(
            "Fresh EC-only CellTypist annotation requires adata.raw, but it is absent."
        )
    query = adata.raw.to_adata()
    query.obs = adata.obs.copy()
    if "X_umap" in adata.obsm:
        query.obsm["X_umap"] = adata.obsm["X_umap"].copy()
    query.var_names_make_unique()

    # EC-only gene filtering can reduce the total represented by expression
    # that was previously log1p-normalized to 10,000. CellTypist validates that
    # expm1(X) sums to 10,000 per cell. Reconstruct the nonnegative expression,
    # renormalize over the genes retained in this query, and log1p again when
    # required. This uses unscaled expression and avoids CellTypist's invalid-
    # matrix warning.
    if sp.issparse(query.X):
        reconstructed = query.X.copy().tocsr()
        reconstructed.data = np.expm1(reconstructed.data)
        totals_before = np.asarray(reconstructed.sum(axis=1)).ravel()
    else:
        reconstructed = np.expm1(np.asarray(query.X, dtype=float))
        totals_before = np.asarray(reconstructed.sum(axis=1)).ravel()
    finite_positive = np.isfinite(totals_before) & (totals_before > 0)
    normalized_fraction_before = float(
        np.mean(np.isclose(totals_before[finite_positive], 1e4, rtol=1e-3, atol=1.0))
    ) if finite_positive.any() else 0.0
    query_was_renormalized = normalized_fraction_before < 0.99
    if query_was_renormalized:
        query.X = reconstructed
        query.uns.pop("log1p", None)
        sc.pp.normalize_total(query, target_sum=1e4)
        sc.pp.log1p(query)

    model_upper = set(model_features.astype(str).str.upper())
    current_names = pd.Index(query.var_names.astype(str))
    current_overlap = len(set(current_names.str.upper()) & model_upper)
    gene_name_source = "raw_var_names"

    if "gene_symbol" in query.var.columns:
        symbols = query.var["gene_symbol"].astype("string").fillna("").str.strip()
        valid = symbols.ne("")
        symbol_overlap = len(set(symbols.loc[valid].str.upper()) & model_upper)
        if valid.sum() >= 1000 and symbol_overlap > current_overlap:
            query = query[:, valid.to_numpy()].copy()
            query.var_names = symbols.loc[valid].astype(str).to_numpy()
            query.var_names_make_unique()
            current_names = pd.Index(query.var_names.astype(str))
            current_overlap = len(set(current_names.str.upper()) & model_upper)
            gene_name_source = "raw_var_gene_symbol"

    overlap_genes = sorted(set(current_names.str.upper()) & model_upper)
    overlap = {
        "gene_name_source": gene_name_source,
        "n_query_genes": int(query.n_vars),
        "n_model_genes": int(len(model_features)),
        "n_model_genes_present": int(len(overlap_genes)),
        "model_gene_overlap_fraction": float(
            len(overlap_genes) / len(model_features)
        ),
        "celltypist_query_renormalized_to_10000": bool(query_was_renormalized),
        "fraction_cells_at_10000_before_renormalization": normalized_fraction_before,
        "median_reconstructed_total_before_renormalization": (
            float(np.median(totals_before[finite_positive]))
            if finite_positive.any()
            else np.nan
        ),
        "overlap_gene_examples": "; ".join(overlap_genes[:50]),
    }
    return query, overlap


def label_is_endothelial(label: object) -> bool:
    text = str(label).strip().lower()
    return bool(re.search(r"(^ec\d|endothelial|endocardial)", text))


def label_is_lymphatic(label: object) -> bool:
    text = str(label).strip().lower()
    return text == "ec8_ln" or "lymph" in text


def run_fresh_celltypist_majority(
    adata,
    quartile: str,
    model_path: Path,
    model_features: pd.Index,
    min_model_genes_present: int,
):
    """Rerun CellTypist and majority-vote over the new EC-only Leiden clusters."""
    import celltypist

    query, overlap = prepare_celltypist_query(adata, model_features)
    if overlap["n_model_genes_present"] < min_model_genes_present:
        raise ValueError(
            "Only "
            f"{overlap['n_model_genes_present']:,} CellTypist model genes are present; "
            f"minimum required is {min_model_genes_present:,}."
        )
    result = celltypist.annotate(
        query,
        model=str(model_path),
        majority_voting=True,
        over_clustering="leiden",
    )
    labels = result.predicted_labels.copy()
    if isinstance(labels, pd.Series):
        labels = labels.to_frame("predicted_labels")
    labels.index = labels.index.astype(str)
    raw_column = (
        "predicted_labels" if "predicted_labels" in labels.columns else labels.columns[0]
    )
    vote_column = "majority_voting" if "majority_voting" in labels.columns else raw_column

    probability = result.probability_matrix
    if isinstance(probability, pd.DataFrame):
        probability.index = probability.index.astype(str)
        labels["max_probability"] = probability.max(axis=1).reindex(labels.index)
        labels["probability_best_label"] = probability.idxmax(axis=1).reindex(labels.index)

    cell_table = pd.DataFrame(index=query.obs_names.astype(str))
    cell_table["cell_id"] = cell_table.index
    cell_table["dataset"] = DATASET
    cell_table["disease"] = REQUIRED_DISEASE
    cell_table["age_quartile"] = quartile
    cell_table["ec_leiden"] = normalized_text(query.obs["leiden"]).to_numpy()
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
    else:
        cell_table["celltypist_max_probability"] = np.nan
    if "probability_best_label" in labels.columns:
        cell_table["celltypist_probability_best_label"] = normalized_text(
            labels["probability_best_label"].reindex(cell_table.index)
        ).to_numpy()
    else:
        cell_table["celltypist_probability_best_label"] = ""

    cluster_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    for cluster in sorted(cell_table["ec_leiden"].unique(), key=cluster_sort_key):
        subset = cell_table.loc[cell_table["ec_leiden"].eq(cluster)]
        n_cells = int(len(subset))
        raw_counts = subset["celltypist_raw_label"].value_counts()
        vote_counts = subset["celltypist_majority_vote"].value_counts()
        raw_label = str(raw_counts.index[0]) if len(raw_counts) else "unknown"
        vote_label = str(vote_counts.index[0]) if len(vote_counts) else raw_label
        second_label = str(raw_counts.index[1]) if len(raw_counts) > 1 else ""
        confidence = pd.to_numeric(
            subset["celltypist_max_probability"], errors="coerce"
        )
        cluster_rows.append(
            {
                "dataset": DATASET,
                "disease": REQUIRED_DISEASE,
                "age_quartile": quartile,
                "ec_leiden": str(cluster),
                "n_cells": n_cells,
                "cluster_majority_cell_type": vote_label,
                "majority_vote_support_fraction": (
                    float(raw_counts.get(vote_label, 0) / n_cells) if n_cells else np.nan
                ),
                "raw_prediction_majority_label": raw_label,
                "raw_prediction_majority_fraction": (
                    float(raw_counts.iloc[0] / n_cells) if n_cells and len(raw_counts) else np.nan
                ),
                "second_raw_prediction_label": second_label,
                "second_raw_prediction_fraction": (
                    float(raw_counts.iloc[1] / n_cells) if n_cells and len(raw_counts) > 1 else 0.0
                ),
                "mean_celltypist_confidence": (
                    float(confidence.mean()) if confidence.notna().any() else np.nan
                ),
                "median_celltypist_confidence": (
                    float(confidence.median()) if confidence.notna().any() else np.nan
                ),
                "celltypist_label_is_endothelial": label_is_endothelial(vote_label),
                "celltypist_label_is_lymphatic": label_is_lymphatic(vote_label),
                "selected_as_LEC": label_is_lymphatic(vote_label),
                "LEC_selection_rule": "fresh_EC_only_CellTypist_majority_vote_EC8_ln",
            }
        )
        for label, count in raw_counts.items():
            distribution_rows.append(
                {
                    "dataset": DATASET,
                    "disease": REQUIRED_DISEASE,
                    "age_quartile": quartile,
                    "ec_leiden": str(cluster),
                    "cell_type": str(label),
                    "n_cells": int(count),
                    "fraction_within_cluster": float(count / n_cells),
                }
            )

    cell_table["selected_as_LEC"] = cell_table["celltypist_majority_vote"].map(
        label_is_lymphatic
    )
    cell_table["final_LEC_by_majority_vote"] = np.where(
        cell_table["selected_as_LEC"], "LEC", "other_EC_candidate"
    )
    return (
        cell_table.reset_index(drop=True),
        pd.DataFrame(cluster_rows),
        pd.DataFrame(distribution_rows),
        overlap,
    )


def recluster_ec_subset(adata, args: argparse.Namespace):
    import scanpy as sc

    adata = adata.copy()
    adata.var_names_make_unique()
    n_before_filter = int(adata.n_obs)

    if args.min_genes_per_cell > 0:
        sc.pp.filter_cells(adata, min_genes=args.min_genes_per_cell)
    if args.min_cells_per_gene > 0:
        sc.pp.filter_genes(adata, min_cells=args.min_cells_per_gene)
    if adata.n_obs < 2 or adata.n_vars < 2:
        raise ValueError(
            f"Too few cells/genes after EC filtering: {adata.n_obs} cells, {adata.n_vars} genes"
        )

    mode = (
        detect_preprocess_mode(adata)
        if args.preprocess_mode == "auto"
        else args.preprocess_mode
    )
    if mode == "counts":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Preserve full, unscaled expression for marker analysis and the next step.
    adata.raw = adata.copy()
    if 0 < args.n_top_hvg < adata.n_vars:
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=args.n_top_hvg,
                flavor="seurat",
            )
            n_hvg = (
                int(adata.var["highly_variable"].sum())
                if "highly_variable" in adata.var.columns
                else 0
            )
            if n_hvg >= 50:
                adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
        except Exception as error:
            warnings.warn(f"HVG selection failed; retaining all genes: {error}")

    n_pcs = min(args.n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    if n_pcs < 2:
        raise ValueError(
            f"Too few dimensions for PCA: {adata.n_obs} cells, {adata.n_vars} genes"
        )
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(
        adata,
        n_comps=n_pcs,
        svd_solver="arpack",
        random_state=args.random_state,
    )
    n_neighbors = max(2, min(args.n_neighbors, adata.n_obs - 1))
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        random_state=args.random_state,
    )
    sc.tl.leiden(
        adata,
        resolution=args.resolution,
        key_added="leiden",
        random_state=args.random_state,
    )
    sc.tl.umap(adata, random_state=args.random_state)
    adata.uns["ec_recluster_params"] = {
        "script_version": SCRIPT_VERSION,
        "preprocess_mode_used": mode,
        "n_cells_before_internal_filter": n_before_filter,
        "n_cells_after_internal_filter": int(adata.n_obs),
        "n_top_hvg_requested": int(args.n_top_hvg),
        "n_pcs_used": int(n_pcs),
        "n_neighbors_used": int(n_neighbors),
        "leiden_resolution": float(args.resolution),
        "random_state": int(args.random_state),
    }
    return adata, mode, n_pcs, n_neighbors


def marker_expression_summary(adata, cluster_key: str, quartile: str) -> pd.DataFrame:
    import scipy.sparse as sp

    matrix_adata = adata.raw if adata.raw is not None else adata
    gene_names = pd.Index(matrix_adata.var_names.astype(str))
    lookup = {gene.upper(): gene for gene in gene_names}
    present = [gene for gene in MARKER_GENES if gene in lookup]
    matched = [lookup[gene] for gene in present]
    clusters = normalized_text(adata.obs[cluster_key]).to_numpy()
    labels = sorted(pd.unique(clusters), key=cluster_sort_key)
    if not matched:
        return pd.DataFrame(
            {"dataset": DATASET, "disease": REQUIRED_DISEASE, "age_quartile": quartile, "ec_leiden": labels}
        )

    expression = matrix_adata[:, matched].X
    if sp.issparse(expression):
        expression = expression.tocsr()
    rows: list[dict[str, object]] = []
    for cluster in labels:
        mask = clusters == cluster
        block = expression[mask]
        if sp.issparse(block):
            means = np.asarray(block.mean(axis=0)).ravel()
            detected = np.asarray((block > 0).mean(axis=0)).ravel()
        else:
            array = np.asarray(block)
            means = np.mean(array, axis=0)
            detected = np.mean(array > 0, axis=0)
        row: dict[str, object] = {
            "dataset": DATASET,
            "disease": REQUIRED_DISEASE,
            "age_quartile": quartile,
            "ec_leiden": cluster,
            "n_cells": int(mask.sum()),
        }
        for gene, mean, pct in zip(present, means, detected):
            row[f"mean_expr_{gene}"] = float(mean)
            row[f"pct_expr_{gene}"] = float(pct)
        for set_name, genes in [
            ("endothelial", ENDOTHELIAL_MARKERS),
            ("LEC", LEC_MARKERS),
            ("negative_check", NEGATIVE_LINEAGE_MARKERS),
        ]:
            available = [gene for gene in genes if gene in present]
            row[f"{set_name}_markers_present"] = "; ".join(available)
            row[f"{set_name}_marker_pct_expr_mean"] = (
                float(np.mean([row[f"pct_expr_{gene}"] for gene in available]))
                if available
                else np.nan
            )
            row[f"{set_name}_marker_score_mean"] = (
                float(np.mean([row[f"mean_expr_{gene}"] for gene in available]))
                if available
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def find_top_markers(adata, quartile: str, args: argparse.Namespace):
    import scanpy as sc

    cluster_key = "leiden"
    adata.obs[cluster_key] = normalized_text(adata.obs[cluster_key]).astype("category")
    n_clusters = adata.obs[cluster_key].nunique()
    if n_clusters < 2:
        return pd.DataFrame(), pd.DataFrame(), "only one EC cluster"
    available = adata.raw.n_vars if adata.raw is not None else adata.n_vars
    n_rank = available if args.marker_n_genes <= 0 else min(args.marker_n_genes, available)
    n_rank = min(available, max(args.top_n_markers, n_rank))
    key = "rank_genes_groups_ec_leiden"
    kwargs = dict(
        groupby=cluster_key,
        reference="rest",
        method=args.marker_method,
        n_genes=n_rank,
        use_raw=adata.raw is not None,
        key_added=key,
    )
    try:
        sc.tl.rank_genes_groups(adata, pts=True, **kwargs)
    except TypeError:
        sc.tl.rank_genes_groups(adata, **kwargs)

    rows: list[pd.DataFrame] = []
    groups = sorted(adata.obs[cluster_key].astype(str).unique(), key=cluster_sort_key)
    for group in groups:
        table = sc.get.rank_genes_groups_df(adata, group=group, key=key)
        table = table.rename(
            columns={
                "names": "gene",
                "logfoldchanges": "logfoldchange",
                "pct_nz_group": "pct_in",
                "pct_nz_reference": "pct_out",
            }
        )
        table["gene"] = normalized_text(table["gene"])
        table = table.loc[~table["gene"].str.upper().isin({"", "NAN", "NONE"})]
        if "logfoldchange" in table.columns:
            logfc = pd.to_numeric(table["logfoldchange"], errors="coerce")
            table = table.loc[logfc.isna() | logfc.gt(0)]
        table = table.head(args.top_n_markers).copy()
        table.insert(0, "marker_rank", np.arange(1, len(table) + 1))
        table.insert(0, "ec_leiden", str(group))
        table.insert(0, "age_quartile", quartile)
        table.insert(0, "disease", REQUIRED_DISEASE)
        table.insert(0, "dataset", DATASET)
        table["n_cells_in_ec_leiden"] = int(
            adata.obs[cluster_key].astype(str).eq(str(group)).sum()
        )
        rows.append(table)
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    wide_rows: list[dict[str, object]] = []
    for group in groups:
        subset = long.loc[long["ec_leiden"].astype(str).eq(str(group))]
        genes = subset["gene"].astype(str).tolist()
        row: dict[str, object] = {
            "dataset": DATASET,
            "disease": REQUIRED_DISEASE,
            "age_quartile": quartile,
            "ec_leiden": str(group),
            "n_cells_in_ec_leiden": int(
                adata.obs[cluster_key].astype(str).eq(str(group)).sum()
            ),
            "top_marker_genes_semicolon": "; ".join(genes),
            "marker_method": args.marker_method,
        }
        for index in range(args.top_n_markers):
            row[f"marker_{index + 1}"] = genes[index] if index < len(genes) else ""
        wide_rows.append(row)
    return long, pd.DataFrame(wide_rows), "completed"


def save_per_quartile_figures(
    adata,
    counts: pd.DataFrame,
    quartile: str,
    figure_dir: Path,
    dpi: int,
) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import scanpy as sc

    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    prefix = f"dataset13_{quartile}"
    plot_specs = [
        ("leiden", "EC-only Leiden cluster", "umap_ec_leiden"),
        ("allcell_leiden", "Source all-cell Leiden cluster", "umap_source_allcell_leiden"),
        ("step05_celltypist_raw_label", "Carried Step 05 raw CellTypist label", "umap_step05_raw_celltypist"),
        ("celltypist_raw_label", "Fresh EC-only raw CellTypist label", "umap_EC_celltypist_raw"),
        ("celltypist_majority_vote", "Fresh EC-only CellTypist majority vote", "umap_EC_celltypist_majority_vote"),
        ("final_LEC_by_majority_vote", "Final LEC selection", "umap_final_LEC_selection"),
    ]
    for color, title, suffix in plot_specs:
        if color not in adata.obs.columns:
            continue
        sc.pl.umap(
            adata,
            color=color,
            show=False,
            frameon=False,
            title=f"{quartile} NF EC candidates: {title}",
            legend_loc="right margin",
        )
        path = figure_dir / f"{prefix}_{suffix}.png"
        plt.gcf().savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(plt.gcf())
        manifest.append(
            {"age_quartile": quartile, "figure": str(path), "description": title}
        )

    cluster_totals = (
        counts.groupby("ec_leiden", observed=True)["n_cells"].sum().reset_index()
    )
    # Cast categorical Leiden labels before mapping tuple sort keys. Mapping a
    # Categorical directly can create a MultiIndex and trigger pandas' 
    # "isna is not defined for MultiIndex" error.
    cluster_totals["_order"] = (
        cluster_totals["ec_leiden"].astype(str).map(cluster_sort_key)
    )
    cluster_totals = cluster_totals.sort_values("_order")
    fig, ax = plt.subplots(figsize=(max(6, 0.65 * len(cluster_totals)), 4.5))
    ax.bar(cluster_totals["ec_leiden"].astype(str), cluster_totals["n_cells"], color="#2A9D8F")
    ax.set_xlabel("EC-only Leiden cluster")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"{quartile} NF: EC-only cluster sizes")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = figure_dir / f"{prefix}_EC_cluster_cell_counts.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest.append(
        {"age_quartile": quartile, "figure": str(path), "description": "EC-only cluster cell counts"}
    )
    return manifest


def save_combined_figures(
    run_summary: pd.DataFrame,
    marker_summary: pd.DataFrame,
    output_root: Path,
    dpi: int,
) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_dir = output_root / "figures_summary"
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    completed = run_summary.loc[run_summary["status"].eq("completed")].copy()
    if not completed.empty:
        completed["age_quartile"] = pd.Categorical(
            completed["age_quartile"], categories=QUARTILES, ordered=True
        )
        completed = completed.sort_values("age_quartile")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(
            completed["age_quartile"].astype(str),
            completed["n_EC_cells_after_filter"],
            color="#2878B5",
        )
        ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
        ax.set_xlabel("NF donor-age quartile")
        ax.set_ylabel("Selected EC-candidate cells")
        ax.set_title("EC candidates retained for quartile-specific reclustering")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        path = figure_dir / "06_selected_EC_cells_by_quartile.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        manifest.append(
            {"age_quartile": "combined", "figure": str(path), "description": "Selected EC-cell counts by quartile"}
        )

    heatmap_genes = ENDOTHELIAL_MARKERS + LEC_MARKERS
    pct_columns = [f"pct_expr_{gene}" for gene in heatmap_genes]
    available = [column for column in pct_columns if column in marker_summary.columns]
    if not marker_summary.empty and available:
        heat = marker_summary.copy()
        heat["row_label"] = heat["age_quartile"].astype(str) + " | EC" + heat["ec_leiden"].astype(str)
        heat["_q_order"] = heat["age_quartile"].map({q: i for i, q in enumerate(QUARTILES)})
        heat["_c_order"] = heat["ec_leiden"].astype(str).map(cluster_sort_key)
        heat = heat.sort_values(["_q_order", "_c_order"])
        matrix = heat[available].to_numpy(dtype=float)
        fig_height = max(6, 0.32 * len(heat) + 2)
        fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(available)), fig_height))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(available)))
        ax.set_xticklabels([column.removeprefix("pct_expr_") for column in available], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(heat)))
        ax.set_yticklabels(heat["row_label"].tolist())
        ax.set_title("Endothelial and LEC marker detection after EC-only reclustering")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        colorbar.set_label("Fraction of cells with detected expression")
        fig.tight_layout()
        path = figure_dir / "06_EC_LEC_marker_detection_heatmap.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        manifest.append(
            {"age_quartile": "combined", "figure": str(path), "description": "EC and LEC marker detection heatmap"}
        )
    return manifest


def process_quartile(
    quartile: str,
    selection: pd.DataFrame,
    input_root: Path,
    step05_root: Path,
    output_root: Path,
    model_path: Path,
    model_features: pd.Index,
    args: argparse.Namespace,
):
    import scanpy as sc

    quartile_dir = output_root / quartile
    quartile_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = quartile_dir / "figures"
    prefix = f"dataset13_{quartile}"
    source_h5ad = input_root / quartile / f"{prefix}_clustered.h5ad"
    prediction_csv = (
        step05_root / quartile / f"{prefix}_celltypist_cell_predictions.csv.gz"
    )
    output_h5ad = quartile_dir / f"{prefix}_EC_reclustered.h5ad"
    if output_h5ad.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_h5ad}. Use --overwrite to replace it."
        )
    if not source_h5ad.exists():
        raise FileNotFoundError(f"Step 03 source h5ad not found: {source_h5ad}")

    selected_clusters = sorted(
        selection.loc[selection["age_quartile"].eq(quartile), "cluster"].astype(str).unique(),
        key=cluster_sort_key,
    )
    print(f"\n[{quartile}] selected source all-cell clusters: {selected_clusters}")
    print(f"[{quartile}] reading {source_h5ad}")
    adata_all = sc.read_h5ad(source_h5ad)
    validate_source_adata(adata_all, source_h5ad, quartile, args.cluster_key)
    predictions = read_step05_predictions(prediction_csv, pd.Index(adata_all.obs_names.astype(str)))

    source_clusters = normalized_text(adata_all.obs[args.cluster_key])
    mask = source_clusters.isin(selected_clusters).to_numpy()
    n_selected = int(mask.sum())
    if n_selected < args.min_selected_cells:
        raise ValueError(
            f"Only {n_selected} EC candidates selected, below --min_selected_cells={args.min_selected_cells}."
        )
    adata_selected = adata_all[mask].copy()
    adata_selected.obs["allcell_leiden"] = source_clusters.loc[mask].astype(str).to_numpy()
    adata_selected.obs["step05_EC_candidate"] = "yes"
    adata_selected.obs["manual_EC_candidate"] = "yes"

    selected_predictions = predictions.reindex(adata_selected.obs_names.astype(str))
    prediction_columns = [
        "celltypist_raw_label",
        "celltypist_majority_vote",
        "celltypist_max_probability",
        "celltypist_probability_best_label",
    ]
    for column in prediction_columns:
        if column not in selected_predictions.columns:
            continue
        values = selected_predictions[column]
        output_column = f"step05_{column}"
        if column == "celltypist_max_probability":
            adata_selected.obs[output_column] = pd.to_numeric(
                values, errors="coerce"
            ).to_numpy()
        else:
            adata_selected.obs[output_column] = normalized_text(values).to_numpy()

    step05_raw_ec8_before = int(
        normalized_text(adata_selected.obs["step05_celltypist_raw_label"])
        .eq("EC8_ln")
        .sum()
    )
    donor_col = (
        "donor_id_quartile_unique"
        if "donor_id_quartile_unique" in adata_selected.obs.columns
        else "donor_id"
    )
    n_donors = int(normalized_text(adata_selected.obs[donor_col]).nunique())
    adata_selected.uns["step06_ec_source_selection"] = {
        "script_version": SCRIPT_VERSION,
        "selection_rule": "step05_celltypist_label_is_endothelial_true",
        "source_step03_h5ad": str(source_h5ad),
        "source_step05_predictions": str(prediction_csv),
        "source_cluster_key": args.cluster_key,
        "selected_source_clusters": selected_clusters,
        "dataset": DATASET,
        "disease": REQUIRED_DISEASE,
        "age_quartile": quartile,
    }

    selected_columns = [
        "allcell_leiden",
        "step05_EC_candidate",
        donor_col,
        "donor_id",
        "Donor",
        "Project",
        "age_numeric_resolved",
        "age_quartile",
        "age_quartile_range_label",
        "sex",
        "Sex",
        "disease",
        "step05_celltypist_raw_label",
        "step05_celltypist_majority_vote",
        "step05_celltypist_max_probability",
    ]
    selected_columns = list(dict.fromkeys(c for c in selected_columns if c in adata_selected.obs.columns))
    selected_cells = adata_selected.obs[selected_columns].copy()
    selected_cells.insert(0, "cell_id", selected_cells.index.astype(str))
    selected_cells.to_csv(
        quartile_dir / f"{prefix}_selected_EC_cells_before_reclustering.csv",
        index=False,
    )

    expression, expression_source = choose_expression_source(
        adata_selected, args.expression_source
    )
    expression.uns["step06_ec_source_selection"] = dict(
        adata_selected.uns["step06_ec_source_selection"]
    )
    expression.uns["step06_ec_source_selection"]["expression_source_used"] = expression_source
    del adata_all, adata_selected
    gc.collect()

    adata_ec, preprocess_mode, n_pcs, n_neighbors = recluster_ec_subset(expression, args)
    del expression
    gc.collect()
    adata_ec.obs["dataset_folder"] = DATASET
    adata_ec.obs["condition"] = quartile
    adata_ec.obs["disease"] = REQUIRED_DISEASE
    adata_ec.obs["age_quartile"] = quartile
    adata_ec.obs["ec_recluster_id"] = (
        DATASET + "|" + quartile + "|" + adata_ec.obs["leiden"].astype(str)
    )

    print(
        f"[{quartile}] running fresh CellTypist majority voting over EC-only Leiden clusters"
    )
    (
        celltypist_cells,
        celltypist_clusters,
        celltypist_distribution,
        model_overlap,
    ) = run_fresh_celltypist_majority(
        adata_ec,
        quartile,
        model_path,
        model_features,
        args.min_model_genes_present,
    )
    celltypist_indexed = celltypist_cells.set_index("cell_id")
    fresh_columns = [
        "celltypist_raw_label",
        "celltypist_majority_vote",
        "celltypist_max_probability",
        "celltypist_probability_best_label",
        "selected_as_LEC",
        "final_LEC_by_majority_vote",
    ]
    for column in fresh_columns:
        values = celltypist_indexed.reindex(adata_ec.obs_names.astype(str))[column]
        if column == "celltypist_max_probability":
            adata_ec.obs[column] = pd.to_numeric(values, errors="coerce").to_numpy()
        elif column == "selected_as_LEC":
            adata_ec.obs[column] = values.fillna(False).astype(bool).to_numpy()
        else:
            adata_ec.obs[column] = normalized_text(values).to_numpy()
    adata_ec.uns["fresh_ec_only_celltypist"] = {
        "model": str(model_path),
        "majority_voting": True,
        "over_clustering": "leiden",
        "LEC_selection_rule": "fresh_EC_only_CellTypist_majority_vote_EC8_ln",
        **model_overlap,
    }
    celltypist_cells.to_csv(
        quartile_dir / f"{prefix}_EC_celltypist_cell_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    celltypist_distribution.to_csv(
        quartile_dir / f"{prefix}_EC_celltypist_label_distribution_by_cluster.csv",
        index=False,
    )
    pd.DataFrame([{"age_quartile": quartile, **model_overlap}]).to_csv(
        quartile_dir / f"{prefix}_EC_celltypist_model_gene_overlap.csv",
        index=False,
    )

    counts = (
        adata_ec.obs.groupby(["leiden", "allcell_leiden"], observed=True)
        .size()
        .reset_index(name="n_cells")
        .rename(columns={"leiden": "ec_leiden", "allcell_leiden": "source_allcell_leiden"})
    )
    counts.insert(0, "age_quartile", quartile)
    counts.insert(0, "disease", REQUIRED_DISEASE)
    counts.insert(0, "dataset", DATASET)
    counts.to_csv(quartile_dir / f"{prefix}_EC_recluster_counts.csv", index=False)

    marker_summary = marker_expression_summary(adata_ec, "leiden", quartile)
    marker_summary.to_csv(
        quartile_dir / f"{prefix}_EC_recluster_marker_summary.csv", index=False
    )

    marker_status = "skipped"
    marker_message = "disabled by --skip_markers"
    marker_long = pd.DataFrame()
    marker_wide = pd.DataFrame()
    if not args.skip_markers:
        marker_long, marker_wide, marker_message = find_top_markers(
            adata_ec, quartile, args
        )
        marker_status = "completed" if not marker_long.empty else "empty"
        marker_long.to_csv(
            quartile_dir / f"{prefix}_EC_top{args.top_n_markers}_markers_long.csv",
            index=False,
        )
        marker_wide.to_csv(
            quartile_dir / f"{prefix}_EC_top{args.top_n_markers}_markers_wide.csv",
            index=False,
        )

    cluster_annotation = celltypist_clusters.merge(
        marker_summary,
        on=["dataset", "disease", "age_quartile", "ec_leiden", "n_cells"],
        how="left",
    )
    if not marker_wide.empty and "top_marker_genes_semicolon" in marker_wide.columns:
        cluster_annotation = cluster_annotation.merge(
            marker_wide[["age_quartile", "ec_leiden", "top_marker_genes_semicolon"]],
            on=["age_quartile", "ec_leiden"],
            how="left",
        )
    cluster_annotation.to_csv(
        quartile_dir / f"{prefix}_EC_celltypist_cluster_summary.csv",
        index=False,
    )
    lec_candidates = cluster_annotation.loc[
        cluster_annotation["selected_as_LEC"].fillna(False).astype(bool)
    ].copy()
    lec_candidates.to_csv(
        quartile_dir / f"{prefix}_final_LEC_candidate_clusters.csv",
        index=False,
    )
    lec_clusters = set(lec_candidates["ec_leiden"].astype(str))
    adata_ec.obs["selected_as_LEC"] = (
        adata_ec.obs["leiden"].astype(str).isin(lec_clusters).to_numpy()
    )
    adata_ec.obs["final_LEC_by_majority_vote"] = np.where(
        adata_ec.obs["selected_as_LEC"], "LEC", "other_EC_candidate"
    )
    lec_columns = [
        "leiden",
        "allcell_leiden",
        donor_col,
        "donor_id",
        "Donor",
        "Project",
        "age_numeric_resolved",
        "age_quartile",
        "age_quartile_range_label",
        "sex",
        "Sex",
        "disease",
        "celltypist_raw_label",
        "celltypist_majority_vote",
        "celltypist_max_probability",
        "step05_celltypist_raw_label",
        "step05_celltypist_majority_vote",
    ]
    lec_columns = list(dict.fromkeys(c for c in lec_columns if c in adata_ec.obs.columns))
    lec_cells = adata_ec.obs.loc[adata_ec.obs["selected_as_LEC"], lec_columns].copy()
    lec_cells.insert(0, "cell_id", lec_cells.index.astype(str))
    lec_cells.to_csv(
        quartile_dir / f"{prefix}_final_LEC_cells_by_majority_vote.csv",
        index=False,
    )

    # Save the complete scientific result before optional plotting. A figure
    # failure must not discard clustering or CellTypist majority-vote output.
    adata_ec.write_h5ad(output_h5ad, compression="gzip")

    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        try:
            figure_manifest = save_per_quartile_figures(
                adata_ec, counts, quartile, figure_dir, args.dpi
            )
        except Exception as error:
            warnings.warn(
                f"Figures failed for {quartile}, but the annotated h5ad was saved: {error}"
            )

    fresh_raw_ec8_after = int(
        normalized_text(adata_ec.obs["celltypist_raw_label"]).eq("EC8_ln").sum()
    )
    n_lec_cells = int(adata_ec.obs["selected_as_LEC"].sum())
    summary = {
        "script_version": SCRIPT_VERSION,
        "dataset": DATASET,
        "disease": REQUIRED_DISEASE,
        "age_quartile": quartile,
        "status": "completed",
        "selection_rule": "step05_celltypist_label_is_endothelial_true",
        "selected_source_allcell_clusters": ";".join(selected_clusters),
        "n_source_all_cells": int(mask.size),
        "n_selected_EC_cells_before_filter": n_selected,
        "n_EC_cells_after_filter": int(adata_ec.n_obs),
        "n_donors_in_EC_subset": n_donors,
        "n_step05_raw_EC8_ln_in_selected_source_clusters": step05_raw_ec8_before,
        "n_fresh_raw_EC8_ln_after_EC_recluster": fresh_raw_ec8_after,
        "fresh_CellTypist_majority_voting": True,
        "fresh_CellTypist_over_clustering": "leiden",
        "n_fresh_CellTypist_model_genes_present": model_overlap[
            "n_model_genes_present"
        ],
        "fresh_CellTypist_model_gene_overlap_fraction": model_overlap[
            "model_gene_overlap_fraction"
        ],
        "LEC_selection_rule": "fresh_EC_only_CellTypist_majority_vote_EC8_ln",
        "selected_LEC_clusters": ";".join(
            sorted(lec_clusters, key=cluster_sort_key)
        ),
        "n_selected_LEC_clusters": int(len(lec_clusters)),
        "n_selected_LEC_cells": n_lec_cells,
        "n_EC_only_leiden_clusters": int(adata_ec.obs["leiden"].nunique()),
        "expression_source_used": expression_source,
        "preprocess_mode_used": preprocess_mode,
        "n_genes_after_HVG": int(adata_ec.n_vars),
        "n_genes_in_raw": int(adata_ec.raw.n_vars if adata_ec.raw is not None else adata_ec.n_vars),
        "n_pcs_used": int(n_pcs),
        "n_neighbors_used": int(n_neighbors),
        "leiden_resolution": float(args.resolution),
        "marker_status": marker_status,
        "marker_message": marker_message,
        "source_h5ad": str(source_h5ad),
        "source_step05_predictions": str(prediction_csv),
        "output_h5ad": str(output_h5ad),
    }
    pd.DataFrame([summary]).to_csv(
        quartile_dir / f"{prefix}_step06_run_summary.csv", index=False
    )
    print(
        f"[{quartile}] saved {output_h5ad} | {adata_ec.n_obs:,} EC cells | "
        f"{adata_ec.obs['leiden'].nunique()} EC-only clusters | "
        f"fresh majority-vote LEC clusters: {sorted(lec_clusters, key=cluster_sort_key)} | "
        f"LEC cells: {n_lec_cells:,}"
    )
    del adata_ec
    gc.collect()
    return (
        summary,
        counts,
        marker_summary,
        marker_long,
        marker_wide,
        cluster_annotation,
        celltypist_distribution,
        lec_candidates,
        figure_manifest,
    )


def main() -> int:
    args = parse_args()
    project_dir = find_project_dir()
    input_root = resolve_path(args.input_root, project_dir)
    step05_root = resolve_path(args.step05_root, project_dir)
    step05_summary = resolve_path(args.step05_summary, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    model_path = resolve_path(args.celltypist_model, project_dir)

    try:
        import scanpy  # noqa: F401
        import leidenalg  # noqa: F401
        import celltypist
    except Exception as error:
        print(
            "ERROR: Scanpy, leidenalg, and CellTypist are required in the project environment. "
            f"Details: {error}",
            file=sys.stderr,
        )
        return 2
    if not model_path.exists():
        print(f"ERROR: CellTypist model not found: {model_path}", file=sys.stderr)
        return 2
    model = celltypist.models.Model.load(str(model_path))
    model_features = pd.Index([str(feature) for feature in model.features])

    output_root.mkdir(parents=True, exist_ok=True)
    print("Dataset13 NF aging Step 06: quartile-specific EC reclustering")
    print(f"Project directory: {project_dir}")
    print(f"Step 03 input root: {input_root}")
    print(f"Step 05 cluster summary: {step05_summary}")
    print(f"Fresh EC-only CellTypist model: {model_path}")
    print(f"Output root: {output_root}")

    selection = read_ec_selection(step05_summary, args.quartiles)
    selection_path = output_root / "06_EC_source_cluster_selection.csv"
    selection.to_csv(selection_path, index=False)
    print("Source clusters selected by Step 05 endothelial flag:")
    print(
        selection[
            ["age_quartile", "cluster", "n_cells", "cluster_majority_cell_type"]
        ].to_string(index=False)
    )

    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "input_root": str(input_root),
        "step05_root": str(step05_root),
        "step05_summary": str(step05_summary),
        "output_root": str(output_root),
        "celltypist_model": str(model_path),
        "fresh_celltypist_majority_voting": True,
        "fresh_celltypist_over_clustering": "leiden",
        "min_model_genes_present": args.min_model_genes_present,
        "quartiles": args.quartiles,
        "selection_rule": "step05_celltypist_label_is_endothelial_true",
        "expression_source": args.expression_source,
        "preprocess_mode": args.preprocess_mode,
        "n_top_hvg": args.n_top_hvg,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
        "resolution": args.resolution,
        "marker_method": args.marker_method,
        "marker_n_genes": args.marker_n_genes,
        "top_n_markers": args.top_n_markers,
        "random_state": args.random_state,
        "skip_markers": args.skip_markers,
        "skip_figures": args.skip_figures,
    }
    with (output_root / "06_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    summaries: list[dict[str, object]] = []
    all_counts: list[pd.DataFrame] = []
    all_marker_summaries: list[pd.DataFrame] = []
    all_marker_long: list[pd.DataFrame] = []
    all_marker_wide: list[pd.DataFrame] = []
    all_celltypist_clusters: list[pd.DataFrame] = []
    all_celltypist_distributions: list[pd.DataFrame] = []
    all_lec_candidates: list[pd.DataFrame] = []
    figure_manifest: list[dict[str, str]] = []
    any_error = False

    for quartile in args.quartiles:
        try:
            result = process_quartile(
                quartile,
                selection,
                input_root,
                step05_root,
                output_root,
                model_path,
                model_features,
                args,
            )
            (
                summary,
                counts,
                marker_summary,
                marker_long,
                marker_wide,
                celltypist_clusters,
                celltypist_distribution,
                lec_candidates,
                figures,
            ) = result
            summaries.append(summary)
            all_counts.append(counts)
            all_marker_summaries.append(marker_summary)
            if not marker_long.empty:
                all_marker_long.append(marker_long)
            if not marker_wide.empty:
                all_marker_wide.append(marker_wide)
            all_celltypist_clusters.append(celltypist_clusters)
            all_celltypist_distributions.append(celltypist_distribution)
            if not lec_candidates.empty:
                all_lec_candidates.append(lec_candidates)
            figure_manifest.extend(figures)
        except Exception as error:
            any_error = True
            message = f"{type(error).__name__}: {error}"
            print(f"[{quartile}] ERROR: {message}", file=sys.stderr)
            summaries.append(
                {
                    "script_version": SCRIPT_VERSION,
                    "dataset": DATASET,
                    "disease": REQUIRED_DISEASE,
                    "age_quartile": quartile,
                    "status": "error",
                    "selection_rule": "step05_celltypist_label_is_endothelial_true",
                    "message": message,
                }
            )
            gc.collect()

    run_summary = pd.DataFrame(summaries)
    run_summary.to_csv(output_root / "06_EC_recluster_run_summary.csv", index=False)
    if all_counts:
        pd.concat(all_counts, ignore_index=True).to_csv(
            output_root / "06_EC_recluster_counts_all_quartiles.csv", index=False
        )
    combined_marker_summary = (
        pd.concat(all_marker_summaries, ignore_index=True)
        if all_marker_summaries
        else pd.DataFrame()
    )
    if not combined_marker_summary.empty:
        combined_marker_summary.to_csv(
            output_root / "06_EC_recluster_marker_summary_all_quartiles.csv",
            index=False,
        )
    if all_marker_long:
        pd.concat(all_marker_long, ignore_index=True).to_csv(
            output_root / f"06_EC_recluster_top{args.top_n_markers}_markers_long.csv",
            index=False,
        )
    if all_marker_wide:
        pd.concat(all_marker_wide, ignore_index=True, sort=False).to_csv(
            output_root / f"06_EC_recluster_top{args.top_n_markers}_markers_wide.csv",
            index=False,
        )
    if all_celltypist_clusters:
        pd.concat(all_celltypist_clusters, ignore_index=True, sort=False).to_csv(
            output_root / "06_EC_celltypist_cluster_summary_all_quartiles.csv",
            index=False,
        )
    if all_celltypist_distributions:
        pd.concat(all_celltypist_distributions, ignore_index=True).to_csv(
            output_root / "06_EC_celltypist_label_distribution_all_quartiles.csv",
            index=False,
        )
    if all_lec_candidates:
        pd.concat(all_lec_candidates, ignore_index=True, sort=False).to_csv(
            output_root / "06_final_LEC_candidate_clusters_all_quartiles.csv",
            index=False,
        )

    if not args.skip_figures:
        try:
            figure_manifest.extend(
                save_combined_figures(
                    run_summary, combined_marker_summary, output_root, args.dpi
                )
            )
        except Exception as error:
            warnings.warn(
                f"Combined Step 06 figures failed, but scientific outputs were saved: {error}"
            )
    if figure_manifest:
        pd.DataFrame(figure_manifest).to_csv(
            output_root / "06_figure_manifest.csv", index=False
        )

    print("\nStep 06 run summary:")
    display_columns = [
        column
        for column in [
            "age_quartile",
            "status",
            "selected_source_allcell_clusters",
            "n_EC_cells_after_filter",
            "n_EC_only_leiden_clusters",
            "selected_LEC_clusters",
            "n_selected_LEC_cells",
            "message",
        ]
        if column in run_summary.columns
    ]
    print(run_summary[display_columns].to_string(index=False))
    print(f"Outputs: {output_root}")
    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
