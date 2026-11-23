#!/usr/bin/env python3
r"""
Step 08.1: validate LEC identity, doublets, and cardiac-expression contamination
for one donor/sample.

This is a donor-targeted QC analysis, but it deliberately uses the complete
10x sample as the reference. Scrublet is run on all nuclei from the sample,
never on the already-selected LEC subset. The script also compares the target
LECs with author-annotated lymphatic ECs, other endothelial cells,
cardiomyocytes, and the multi-donor Step 08 LEC collection.

The analysis is descriptive. It does not delete cells, rewrite Step 08, or
claim formal differential expression from a single donor.

Default target
--------------
donor_key: GSE183852::TWCM-13-80
sample: inferred from Step 08 metadata (currently GSM5572789)
focus state: state with the largest target-donor fraction, currently state3

Default inputs
--------------
Data/converted/dataset_13_Age_full_30plus_age10yr_standardized.h5ad
new_result2_aging/08_combined_LEC_recluster/dataset13/
    08_combined_LEC_cell_metadata.csv.gz
    08_combined_final_LEC_reclustered.h5ad
new_result2_aging/03/dataset13/Q*/dataset13_Q*_clustered.h5ad

Default output
--------------
new_result2_aging/08_1_donor_LEC_validation/dataset13/<donor>/

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_aging\08_1_validate_specific_donor_LEC_identity_doublets.py

For a quick smoke test without Scrublet:

    .\.venv\Scripts\python.exe `
      .\new_script2_aging\08_1_validate_specific_donor_LEC_identity_doublets.py `
      --skip_doublets --skip_figures
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import fisher_exact, mannwhitneyu


matplotlib.use("Agg", force=True)


SCRIPT_VERSION = "08_1_dataset13_specific_donor_LEC_validation_resolution_aware_2026_08_07"
DATASET = "dataset13"
DEFAULT_DONOR_KEY = "GSE183852::TWCM-13-80"
DEFAULT_FULL_COUNTS = (
    "Data/converted/dataset_13_Age_full_30plus_age10yr_standardized.h5ad"
)
DEFAULT_STEP08_METADATA = (
    "new_result2_aging/08_combined_LEC_recluster/dataset13/"
    "08_combined_LEC_cell_metadata.csv.gz"
)
DEFAULT_STEP08_H5AD = (
    "new_result2_aging/08_combined_LEC_recluster/dataset13/"
    "08_combined_final_LEC_reclustered.h5ad"
)
DEFAULT_STEP03_ROOT = "new_result2_aging/03/dataset13"
DEFAULT_OUTPUT_ROOT = "new_result2_aging/08_1_donor_LEC_validation/dataset13"


MARKER_PANELS: dict[str, list[str]] = {
    "LEC_specific": [
        "PROX1",
        "PDPN",
        "FLT4",
        "CCL21",
        "LYVE1",
        "NRP2",
        "RELN",
    ],
    "pan_endothelial": [
        "PECAM1",
        "CDH5",
        "CLDN5",
        "VWF",
        "KDR",
        "ERG",
        "ESAM",
        "EMCN",
    ],
    "cardiomyocyte": [
        "MYH6",
        "MYH7",
        "MYL2",
        "MYL3",
        "TNNI3",
        "TNNT2",
        "ACTC1",
        "CKM",
        "RYR2",
        "TTN",
        "TCAP",
    ],
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "COL3A1"],
    "pericyte_smc": ["RGS5", "PDGFRB", "CSPG4", "ACTA2", "MCAM"],
    "immune": ["PTPRC", "LST1", "TYROBP", "FCER1G", "CD3D"],
}


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def sanitize_token(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._") or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate LEC identity, doublets, and cardiac-expression "
            "contamination for one Step 08 donor."
        )
    )
    parser.add_argument("--donor_key", default=DEFAULT_DONOR_KEY)
    parser.add_argument(
        "--sample",
        default="",
        help="Sample ID. Default: infer the unique sample from Step 08 metadata.",
    )
    parser.add_argument(
        "--focus_state",
        default="",
        help=(
            "LEC state to examine separately. Default: state with the largest "
            "fraction contributed by the target donor."
        ),
    )
    parser.add_argument("--full_counts_h5ad", default=DEFAULT_FULL_COUNTS)
    parser.add_argument("--step08_metadata", default=DEFAULT_STEP08_METADATA)
    parser.add_argument("--step08_h5ad", default=DEFAULT_STEP08_H5AD)
    parser.add_argument("--step03_root", default=DEFAULT_STEP03_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--scrublet_seeds", nargs="+", type=int, default=[0, 1, 2]
    )
    parser.add_argument("--expected_doublet_rate", type=float, default=0.06)
    parser.add_argument("--scrublet_n_pcs", type=int, default=30)
    parser.add_argument(
        "--scrublet_high_threshold",
        type=float,
        default=0.15,
        help="Sensitive score threshold reported alongside rank-based calls.",
    )
    parser.add_argument(
        "--scrublet_strict_threshold",
        type=float,
        default=0.25,
        help=(
            "Strict score threshold also passed to Scanpy so scikit-image is "
            "not required."
        ),
    )
    parser.add_argument(
        "--scrublet_top_fraction",
        type=float,
        default=0.06,
        help="Liberal within-sample rank fraction used for sensitivity QC.",
    )
    parser.add_argument("--neighbor_n", type=int, default=15)
    parser.add_argument("--top_genes_figure", type=int, default=25)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_doublets", action="store_true")
    parser.add_argument("--skip_neighborhood", action="store_true")
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of this script's named outputs.",
    )
    return parser.parse_args()


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def natural_key(value: object) -> tuple[object, ...]:
    pieces = re.split(r"(\d+(?:\.\d+)?)", str(value))
    return tuple(
        (0, float(piece)) if piece and piece.replace(".", "", 1).isdigit()
        else (1, piece.lower())
        for piece in pieces
        if piece
    )


def first_existing(columns: Iterable[str], frame: pd.DataFrame) -> str | None:
    return next((column for column in columns if column in frame.columns), None)


def dense_vector(matrix: object) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.toarray()).ravel()
    return np.asarray(matrix).ravel()


def vector_sum_axis0(matrix: object) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.sum(axis=0)).ravel()
    return np.asarray(matrix).sum(axis=0).ravel()


def validate_raw_counts(adata: ad.AnnData) -> dict[str, object]:
    if adata.n_obs < 3 or adata.n_vars < 50:
        raise ValueError(
            f"Too few cells/genes for donor validation: {adata.shape}."
        )
    sample = adata.X[: min(500, adata.n_obs), : min(500, adata.n_vars)]
    values = sample.data if sp.issparse(sample) else np.asarray(sample).ravel()
    if values.size and float(np.nanmin(values)) < 0:
        raise ValueError("The full-sample matrix contains negative values.")
    integer_fraction = (
        float(np.mean(np.isclose(values, np.rint(values))))
        if values.size
        else 1.0
    )
    if integer_fraction < 0.99:
        raise ValueError(
            "Scrublet requires raw integer-like counts, but the selected "
            f"matrix has integer_fraction={integer_fraction:.3f}."
        )
    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "integer_fraction_checked_subset": integer_fraction,
        "matrix_sparse": bool(sp.issparse(adata.X)),
    }


def normalize_log1p(counts: object, target_sum: float = 1e4) -> sp.csr_matrix:
    matrix = sp.csr_matrix(counts, dtype=np.float64)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    factors = np.divide(
        target_sum,
        totals,
        out=np.zeros_like(totals, dtype=float),
        where=totals > 0,
    )
    normalized = sp.diags(factors).dot(matrix).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def marker_availability(var_names: pd.Index) -> pd.DataFrame:
    available = set(map(str, var_names))
    rows = []
    for panel, genes in MARKER_PANELS.items():
        for gene in genes:
            rows.append(
                {"panel": panel, "gene": gene, "available": gene in available}
            )
    return pd.DataFrame(rows)


def add_signature_scores(
    obs: pd.DataFrame,
    log_matrix: sp.csr_matrix,
    var_names: pd.Index,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    result = obs.copy()
    name_to_index = {str(name): index for index, name in enumerate(var_names)}
    used: dict[str, list[str]] = {}
    for panel, requested in MARKER_PANELS.items():
        genes = [gene for gene in requested if gene in name_to_index]
        if not genes:
            raise ValueError(f"No genes from marker panel {panel!r} are present.")
        used[panel] = genes
        indices = [name_to_index[gene] for gene in genes]
        values = log_matrix[:, indices].toarray()
        means = values.mean(axis=0)
        std = values.std(axis=0)
        std[std == 0] = 1.0
        z = (values - means) / std
        prefix = panel.lower()
        result[f"{prefix}_zscore"] = z.mean(axis=1)
        result[f"{prefix}_mean_logexpr"] = values.mean(axis=1)
        result[f"{prefix}_genes_detected"] = (values > 0).sum(axis=1)
        result[f"{prefix}_fraction_detected"] = (values > 0).mean(axis=1)
    return result, used


def expression_rank_table(
    group_name: str,
    mask: np.ndarray,
    counts: sp.csr_matrix,
    log_matrix: sp.csr_matrix,
    var_names: pd.Index,
    reference_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    n_cells = int(mask.sum())
    if n_cells == 0:
        raise ValueError(f"Expression group {group_name!r} has zero cells.")
    count_sum = vector_sum_axis0(counts[mask])
    total = float(count_sum.sum())
    cpm = count_sum / total * 1e6 if total > 0 else np.zeros_like(count_sum)
    mean_log = np.asarray(log_matrix[mask].mean(axis=0)).ravel()
    pct_detected = (
        np.asarray((counts[mask] > 0).sum(axis=0)).ravel() / n_cells * 100
    )
    frame = pd.DataFrame(
        {
            "group": group_name,
            "gene": var_names.astype(str),
            "n_cells": n_cells,
            "count_sum": count_sum,
            "cpm": cpm,
            "mean_logexpr": mean_log,
            "pct_detected": pct_detected,
        }
    )
    frame["absolute_expression_rank"] = (
        frame["cpm"].rank(method="min", ascending=False).astype(int)
    )
    for reference_name, reference_mask in reference_masks.items():
        reference_n = int(reference_mask.sum())
        if reference_n == 0:
            continue
        ref_sum = vector_sum_axis0(counts[reference_mask])
        ref_total = float(ref_sum.sum())
        ref_cpm = (
            ref_sum / ref_total * 1e6
            if ref_total > 0
            else np.zeros_like(ref_sum)
        )
        safe = sanitize_token(reference_name).lower()
        frame[f"reference_{safe}_cpm"] = ref_cpm
        frame[f"log2fc_vs_{safe}"] = np.log2((cpm + 1.0) / (ref_cpm + 1.0))
        frame[f"specificity_rank_vs_{safe}"] = (
            frame[f"log2fc_vs_{safe}"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
    return frame.sort_values("absolute_expression_rank").reset_index(drop=True)


def summarize_marker_expression(
    rank_tables: dict[str, pd.DataFrame], used_panels: dict[str, list[str]]
) -> pd.DataFrame:
    gene_to_panel = {
        gene: panel for panel, genes in used_panels.items() for gene in genes
    }
    genes = set(gene_to_panel)
    frames = []
    for group, table in rank_tables.items():
        part = table.loc[table["gene"].isin(genes)].copy()
        part.insert(1, "panel", part["gene"].map(gene_to_panel))
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def group_masks_from_sample(
    obs: pd.DataFrame, focus_state: str
) -> dict[str, np.ndarray]:
    target = obs["is_target_LEC"].fillna(False).to_numpy(dtype=bool)
    state = normalized_text(obs["LEC_state"])
    cell_type = normalized_text(obs["cell.type"])
    broad = normalized_text(obs["broad.cell.type"])
    author_lec = cell_type.str.casefold().eq("lymphatic ecs").to_numpy()
    endothelial = broad.str.casefold().eq("endothelial").to_numpy()
    cardiac = broad.str.casefold().eq("cardiac").to_numpy()
    return {
        "target_all_LECs": target,
        f"target_{focus_state}": target & state.eq(focus_state).to_numpy(),
        "target_other_LECs": target & ~state.eq(focus_state).to_numpy(),
        "author_Lymphatic_ECs": author_lec,
        "author_other_endothelial": endothelial & ~author_lec,
        "author_cardiomyocytes": cardiac,
        "sample_non_target": ~target,
        "author_non_cardiac": ~cardiac,
        "author_non_endothelial": ~endothelial,
    }


def run_scrublet(
    sample: ad.AnnData,
    seeds: list[int],
    expected_rate: float,
    n_pcs: int,
    high_threshold: float,
    strict_threshold: float,
    top_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import scanpy as sc

    if not seeds:
        raise ValueError("At least one Scrublet seed is required.")
    result = pd.DataFrame(index=sample.obs_names.astype(str))
    summaries = []
    score_columns = []
    high_columns = []
    strict_columns = []
    top_columns = []
    for seed in seeds:
        print(f"Running Scrublet on full sample: seed={seed}")
        work = sample.copy()
        sc.pp.scrublet(
            work,
            expected_doublet_rate=float(expected_rate),
            n_prin_comps=min(int(n_pcs), max(2, work.n_vars - 1)),
            random_state=int(seed),
            # A supplied threshold avoids an optional scikit-image dependency.
            threshold=float(strict_threshold),
            verbose=False,
        )
        scores = pd.to_numeric(work.obs["doublet_score"], errors="coerce")
        top_cutoff = float(scores.quantile(1.0 - top_fraction))
        score_column = f"scrublet_score_seed{seed}"
        high_column = f"scrublet_high_seed{seed}"
        strict_column = f"scrublet_strict_seed{seed}"
        top_column = f"scrublet_top_fraction_seed{seed}"
        result[score_column] = scores.to_numpy()
        result[high_column] = scores.ge(high_threshold).to_numpy()
        result[strict_column] = scores.ge(strict_threshold).to_numpy()
        result[top_column] = scores.ge(top_cutoff).to_numpy()
        score_columns.append(score_column)
        high_columns.append(high_column)
        strict_columns.append(strict_column)
        top_columns.append(top_column)
        simulated = np.asarray(
            work.uns.get("scrublet", {}).get("doublet_scores_sim", [])
        )
        summaries.append(
            {
                "seed": int(seed),
                "n_sample_cells": int(work.n_obs),
                "expected_doublet_rate": float(expected_rate),
                "high_threshold": float(high_threshold),
                "strict_threshold": float(strict_threshold),
                "top_fraction": float(top_fraction),
                "top_fraction_score_cutoff": top_cutoff,
                "observed_score_median": float(scores.median()),
                "observed_score_q95": float(scores.quantile(0.95)),
                "observed_score_max": float(scores.max()),
                "n_high": int(scores.ge(high_threshold).sum()),
                "n_strict": int(scores.ge(strict_threshold).sum()),
                "n_top_fraction": int(scores.ge(top_cutoff).sum()),
                "simulated_n": int(simulated.size),
                "simulated_fraction_above_high": (
                    float(np.mean(simulated >= high_threshold))
                    if simulated.size
                    else np.nan
                ),
                "simulated_fraction_above_strict": (
                    float(np.mean(simulated >= strict_threshold))
                    if simulated.size
                    else np.nan
                ),
            }
        )
        del work
        gc.collect()
    majority = math.ceil(len(seeds) / 2)
    result["scrublet_score_mean"] = result[score_columns].mean(axis=1)
    result["scrublet_score_median"] = result[score_columns].median(axis=1)
    result["scrublet_score_max"] = result[score_columns].max(axis=1)
    result["scrublet_high_n_seeds"] = result[high_columns].sum(axis=1)
    result["scrublet_strict_n_seeds"] = result[strict_columns].sum(axis=1)
    result["scrublet_top_fraction_n_seeds"] = result[top_columns].sum(axis=1)
    result["scrublet_consensus_high"] = (
        result["scrublet_high_n_seeds"] >= majority
    )
    result["scrublet_consensus_strict"] = (
        result["scrublet_strict_n_seeds"] >= majority
    )
    result["scrublet_consensus_top_fraction"] = (
        result["scrublet_top_fraction_n_seeds"] >= majority
    )
    return result, pd.DataFrame(summaries)


def scrublet_group_summary(
    obs: pd.DataFrame,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for group, frame in obs.groupby(group_column, observed=True):
        rows.append(
            {
                "group": str(group),
                "n_cells": int(len(frame)),
                "score_median": float(frame["scrublet_score_mean"].median()),
                "score_mean": float(frame["scrublet_score_mean"].mean()),
                "score_q90": float(frame["scrublet_score_mean"].quantile(0.90)),
                "score_max": float(frame["scrublet_score_mean"].max()),
                "n_consensus_high": int(frame["scrublet_consensus_high"].sum()),
                "fraction_consensus_high": float(
                    frame["scrublet_consensus_high"].mean()
                ),
                "n_consensus_strict": int(
                    frame["scrublet_consensus_strict"].sum()
                ),
                "fraction_consensus_strict": float(
                    frame["scrublet_consensus_strict"].mean()
                ),
                "n_consensus_top_fraction": int(
                    frame["scrublet_consensus_top_fraction"].sum()
                ),
                "fraction_consensus_top_fraction": float(
                    frame["scrublet_consensus_top_fraction"].mean()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    tests = []
    groups = list(obs[group_column].dropna().astype(str).unique())
    target_groups = [group for group in groups if group != "sample_other_nuclei"]
    if len(target_groups) >= 2:
        first, second = target_groups[:2]
        a = obs.loc[obs[group_column].astype(str).eq(first)]
        b = obs.loc[obs[group_column].astype(str).eq(second)]
        if len(a) and len(b):
            u_result = mannwhitneyu(
                a["scrublet_score_mean"],
                b["scrublet_score_mean"],
                alternative="two-sided",
            )
            table = [
                [
                    int(a["scrublet_consensus_strict"].sum()),
                    int((~a["scrublet_consensus_strict"]).sum()),
                ],
                [
                    int(b["scrublet_consensus_strict"].sum()),
                    int((~b["scrublet_consensus_strict"]).sum()),
                ],
            ]
            odds, fisher_p = fisher_exact(table, alternative="two-sided")
            tests.append(
                {
                    "group_a": first,
                    "group_b": second,
                    "mannwhitney_u": float(u_result.statistic),
                    "mannwhitney_two_sided_p": float(u_result.pvalue),
                    "strict_doublet_odds_ratio": float(odds),
                    "strict_doublet_fisher_two_sided_p": float(fisher_p),
                }
            )
    return summary, pd.DataFrame(tests)


def neighborhood_purity(
    step03_path: Path,
    target_ids: list[str],
    neighbor_n: int,
) -> pd.DataFrame:
    print(f"Reading original all-cell graph: {step03_path}")
    source = ad.read_h5ad(step03_path, backed="r")
    missing_columns = [
        column
        for column in ["cell.type", "broad.cell.type"]
        if column not in source.obs.columns
    ]
    if missing_columns:
        source.file.close()
        raise KeyError(f"Step 03 is missing columns: {missing_columns}")
    indexer = source.obs_names.astype(str).get_indexer(target_ids)
    missing = [target_ids[i] for i, value in enumerate(indexer) if value < 0]
    if missing:
        source.file.close()
        raise KeyError(
            f"{len(missing)} target cells are missing from Step 03; first={missing[:3]}"
        )
    graph = source.obsp["connectivities"].tocsr()
    cell_type = normalized_text(source.obs["cell.type"]).to_numpy(dtype=str)
    broad = normalized_text(source.obs["broad.cell.type"]).to_numpy(dtype=str)
    donor_column = first_existing(
        ["donor_id_quartile_unique", "donor_key", "Donor", "donor_id"],
        source.obs,
    )
    donor = (
        normalized_text(source.obs[donor_column]).to_numpy(dtype=str)
        if donor_column
        else np.repeat("", source.n_obs)
    )
    rows = []
    for cell_id, row_index in zip(target_ids, indexer):
        start, end = graph.indptr[row_index], graph.indptr[row_index + 1]
        neighbors = graph.indices[start:end]
        weights = graph.data[start:end]
        keep = neighbors != row_index
        neighbors, weights = neighbors[keep], weights[keep]
        if len(neighbors) > neighbor_n:
            selected = np.argpartition(weights, -neighbor_n)[-neighbor_n:]
            neighbors = neighbors[selected]
            weights = weights[selected]
        neighbor_cell_type = pd.Series(cell_type[neighbors])
        neighbor_broad = pd.Series(broad[neighbors])
        focal_donor = donor[row_index]
        rows.append(
            {
                "source_cell_id": cell_id,
                "n_neighbors_evaluated": int(len(neighbors)),
                "neighbor_author_LEC_fraction": float(
                    neighbor_cell_type.str.casefold().eq("lymphatic ecs").mean()
                ),
                "neighbor_endothelial_fraction": float(
                    neighbor_broad.str.casefold().eq("endothelial").mean()
                ),
                "neighbor_cardiac_fraction": float(
                    neighbor_broad.str.casefold().eq("cardiac").mean()
                ),
                "neighbor_same_donor_fraction": (
                    float(np.mean(donor[neighbors] == focal_donor))
                    if focal_donor
                    else np.nan
                ),
                "mean_neighbor_connectivity": (
                    float(np.mean(weights)) if len(weights) else np.nan
                ),
            }
        )
    source.file.close()
    return pd.DataFrame(rows).set_index("source_cell_id")


def step08_donor_cardiac_comparison(
    h5ad_path: Path,
    cardiac_genes: list[str],
    lec_genes: list[str],
    target_donor: str,
    focus_state: str,
) -> pd.DataFrame:
    combined = ad.read_h5ad(h5ad_path)
    matrix_source = combined.raw if combined.raw is not None else combined
    var_names = matrix_source.var_names.astype(str)
    cardiac = [gene for gene in cardiac_genes if gene in var_names]
    lec = [gene for gene in lec_genes if gene in var_names]
    if not cardiac or not lec:
        raise ValueError("Step 08 lacks available cardiac or LEC marker genes.")
    cardiac_values = matrix_source[:, cardiac].X
    lec_values = matrix_source[:, lec].X
    cardiac_score = (
        np.asarray(cardiac_values.mean(axis=1)).ravel()
        if sp.issparse(cardiac_values)
        else np.asarray(cardiac_values).mean(axis=1)
    )
    lec_score = (
        np.asarray(lec_values.mean(axis=1)).ravel()
        if sp.issparse(lec_values)
        else np.asarray(lec_values).mean(axis=1)
    )
    obs = combined.obs.copy()
    obs["cardiac_mean_logexpr"] = cardiac_score
    obs["LEC_mean_logexpr"] = lec_score
    obs["is_focus_state"] = normalized_text(obs["LEC_state"]).eq(focus_state)
    table = (
        obs.groupby("donor_key", observed=True)
        .agg(
            n_LECs=("LEC_state", "size"),
            n_focus_state=("is_focus_state", "sum"),
            focus_state_fraction=("is_focus_state", "mean"),
            cardiac_mean_logexpr=("cardiac_mean_logexpr", "mean"),
            cardiac_median_logexpr=("cardiac_mean_logexpr", "median"),
            LEC_mean_logexpr=("LEC_mean_logexpr", "mean"),
            Project=("Project", "first"),
            age_quartile=("age_quartile", "first"),
            sample=("sample", "first"),
        )
        .reset_index()
    )
    table["cardiac_score_rank_descending"] = (
        table["cardiac_mean_logexpr"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    table["is_target_donor"] = table["donor_key"].astype(str).eq(target_donor)
    del combined
    return table.sort_values("cardiac_score_rank_descending")


def summarize_author_celltypes(obs: pd.DataFrame) -> pd.DataFrame:
    score_columns = [
        "lec_specific_zscore",
        "pan_endothelial_zscore",
        "cardiomyocyte_zscore",
        "fibroblast_zscore",
        "pericyte_smc_zscore",
        "immune_zscore",
    ]
    aggregations: dict[str, tuple[str, object]] = {
        "n_cells": ("cell.type", "size")
    }
    for column in score_columns:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_median"] = (column, "median")
    return (
        obs.groupby(["broad.cell.type", "cell.type"], observed=True)
        .agg(**aggregations)
        .reset_index()
        .sort_values("n_cells", ascending=False)
    )


def validation_summary(
    obs: pd.DataFrame,
    group_masks: dict[str, np.ndarray],
    focus_state: str,
    has_doublets: bool,
    has_neighbors: bool,
    donor_comparison: pd.DataFrame,
    target_donor: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    cell_type = normalized_text(obs["cell.type"])
    broad = normalized_text(obs["broad.cell.type"])
    author_lec = cell_type.str.casefold().eq("lymphatic ecs").to_numpy()
    author_endothelial = broad.str.casefold().eq("endothelial").to_numpy()
    non_cardiac = ~broad.str.casefold().eq("cardiac").to_numpy()

    def q(values: pd.Series, mask: np.ndarray, quantile: float) -> float:
        selected = pd.to_numeric(values[mask], errors="coerce").dropna()
        return float(selected.quantile(quantile)) if len(selected) else np.nan

    thresholds = {
        "LEC_specific_zscore_author_LEC_q25": q(
            obs["lec_specific_zscore"], author_lec, 0.25
        ),
        # Lymphatic ECs can express some pan-endothelial genes less strongly
        # than blood ECs. Calibrate this supporting score to author LECs rather
        # than requiring the lower quartile of all blood/endothelial cells.
        "pan_endothelial_zscore_author_LEC_q25": q(
            obs["pan_endothelial_zscore"], author_lec, 0.25
        ),
        "cardiomyocyte_zscore_non_cardiac_q95": q(
            obs["cardiomyocyte_zscore"], non_cardiac, 0.95
        ),
    }
    donor_row = donor_comparison.loc[
        donor_comparison["donor_key"].astype(str).eq(target_donor)
    ]
    if donor_row.empty:
        donor_cardiac_rank_fraction = np.nan
    else:
        donor_cardiac_rank_fraction = float(
            donor_row["cardiac_score_rank_descending"].iloc[0]
            / len(donor_comparison)
        )
    thresholds["target_donor_cardiac_rank_fraction"] = (
        donor_cardiac_rank_fraction
    )
    thresholds["donor_cardiac_rank_elevated_cutoff"] = 0.10
    target_group_names = [
        "target_all_LECs",
        f"target_{focus_state}",
        "target_other_LECs",
    ]
    rows = []
    for group in target_group_names:
        mask = group_masks[group]
        frame = obs.loc[mask]
        if frame.empty:
            continue
        row: dict[str, object] = {
            "group": group,
            "n_cells": int(len(frame)),
            "author_LEC_fraction": float(
                normalized_text(frame["cell.type"])
                .str.casefold()
                .eq("lymphatic ecs")
                .mean()
            ),
            "median_nCount_RNA": float(
                pd.to_numeric(frame["nCount_RNA"], errors="coerce").median()
            ),
            "median_nFeature_RNA": float(
                pd.to_numeric(frame["nFeature_RNA"], errors="coerce").median()
            ),
        }
        for panel in MARKER_PANELS:
            column = f"{panel.lower()}_zscore"
            row[f"median_{column}"] = float(frame[column].median())
        if has_doublets:
            row["median_scrublet_score"] = float(
                frame["scrublet_score_mean"].median()
            )
            row["fraction_scrublet_consensus_high"] = float(
                frame["scrublet_consensus_high"].mean()
            )
            row["fraction_scrublet_consensus_strict"] = float(
                frame["scrublet_consensus_strict"].mean()
            )
        else:
            row["median_scrublet_score"] = np.nan
            row["fraction_scrublet_consensus_high"] = np.nan
            row["fraction_scrublet_consensus_strict"] = np.nan
        if has_neighbors:
            row["median_neighbor_LEC_fraction"] = float(
                frame["neighbor_author_LEC_fraction"].median()
            )
            row["median_neighbor_endothelial_fraction"] = float(
                frame["neighbor_endothelial_fraction"].median()
            )
            row["median_neighbor_cardiac_fraction"] = float(
                frame["neighbor_cardiac_fraction"].median()
            )
        else:
            row["median_neighbor_LEC_fraction"] = np.nan
            row["median_neighbor_endothelial_fraction"] = np.nan
            row["median_neighbor_cardiac_fraction"] = np.nan

        lec_supported = (
            row["median_lec_specific_zscore"]
            >= thresholds["LEC_specific_zscore_author_LEC_q25"]
        )
        endothelial_supported = (
            row["median_pan_endothelial_zscore"]
            >= thresholds["pan_endothelial_zscore_author_LEC_q25"]
        )
        neighbor_supported = (
            not has_neighbors
            or row["median_neighbor_endothelial_fraction"] >= 0.5
        )
        cell_level_cardiac_elevated = (
            row["median_cardiomyocyte_zscore"]
            >= thresholds["cardiomyocyte_zscore_non_cardiac_q95"]
        )
        donor_level_cardiac_elevated = bool(
            np.isfinite(donor_cardiac_rank_fraction)
            and donor_cardiac_rank_fraction
            <= thresholds["donor_cardiac_rank_elevated_cutoff"]
        )
        cardiac_elevated = bool(
            cell_level_cardiac_elevated or donor_level_cardiac_elevated
        )
        doublet_burden = bool(
            has_doublets
            and (
                row["fraction_scrublet_consensus_strict"] >= 0.10
                or (
                    row["fraction_scrublet_consensus_high"] >= 0.20
                    and row["median_scrublet_score"] >= 0.10
                )
            )
        )
        row["LEC_expression_supported"] = bool(lec_supported)
        row["pan_endothelial_expression_supported"] = bool(
            endothelial_supported
        )
        row["endothelial_neighborhood_supported"] = bool(neighbor_supported)
        row["cell_level_cardiac_signal_elevated"] = bool(
            cell_level_cardiac_elevated
        )
        row["target_donor_cardiac_rank_fraction"] = (
            donor_cardiac_rank_fraction
        )
        row["donor_level_cardiac_signal_elevated"] = bool(
            donor_level_cardiac_elevated
        )
        row["cardiac_signal_elevated"] = bool(cardiac_elevated)
        row["doublet_burden_flag"] = doublet_burden
        if lec_supported and endothelial_supported and neighbor_supported:
            if doublet_burden and cardiac_elevated:
                conclusion = "possible_LEC_cardiomyocyte_doublet_mixture"
            elif cardiac_elevated:
                conclusion = (
                    "LEC_with_cardiac_signal_not_explained_by_doublets"
                    if has_doublets
                    else "LEC_with_cardiac_signal_doublets_not_evaluated"
                )
            else:
                conclusion = "confirmed_LEC"
        elif lec_supported or endothelial_supported:
            conclusion = "uncertain_LEC_identity_requires_review"
        else:
            conclusion = "LEC_identity_not_supported"
        row["automated_QC_conclusion"] = conclusion
        rows.append(row)
    return pd.DataFrame(rows), thresholds


def save_identity_scatter(
    obs: pd.DataFrame,
    focus_state: str,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6.5))
    background = ~obs["is_target_LEC"].fillna(False)
    ax.scatter(
        obs.loc[background, "lec_specific_zscore"],
        obs.loc[background, "cardiomyocyte_zscore"],
        s=8,
        c="#bdbdbd",
        alpha=0.25,
        linewidths=0,
        label="Other sample nuclei",
    )
    focus = obs["is_target_LEC"].fillna(False) & normalized_text(
        obs["LEC_state"]
    ).eq(focus_state)
    other = obs["is_target_LEC"].fillna(False) & ~focus
    ax.scatter(
        obs.loc[other, "lec_specific_zscore"],
        obs.loc[other, "cardiomyocyte_zscore"],
        s=35,
        c="#1f77b4",
        alpha=0.85,
        label="Target donor: other LECs",
    )
    ax.scatter(
        obs.loc[focus, "lec_specific_zscore"],
        obs.loc[focus, "cardiomyocyte_zscore"],
        s=40,
        c="#ff7f0e",
        alpha=0.9,
        label=f"Target donor: {focus_state}",
    )
    if "scrublet_consensus_strict" in obs.columns:
        strict = obs["scrublet_consensus_strict"].fillna(False)
        ax.scatter(
            obs.loc[strict, "lec_specific_zscore"],
            obs.loc[strict, "cardiomyocyte_zscore"],
            s=70,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            label="Strict Scrublet consensus",
        )
    ax.axhline(0, color="#999999", lw=0.6)
    ax.axvline(0, color="#999999", lw=0.6)
    ax.set_xlabel("LEC-specific signature z-score")
    ax.set_ylabel("Cardiomyocyte signature z-score")
    ax.set_title("Target-donor identity and cardiac-expression signal")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_marker_dotplot(
    marker_table: pd.DataFrame,
    group_order: list[str],
    used_panels: dict[str, list[str]],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    genes = [gene for genes in used_panels.values() for gene in genes]
    table = marker_table.loc[
        marker_table["group"].isin(group_order) & marker_table["gene"].isin(genes)
    ].copy()
    table["group"] = pd.Categorical(
        table["group"], categories=group_order, ordered=True
    )
    table["gene"] = pd.Categorical(table["gene"], categories=genes, ordered=True)
    table = table.dropna(subset=["group", "gene"])
    x = table["gene"].cat.codes.to_numpy()
    y = table["group"].cat.codes.to_numpy()
    sizes = 12 + 150 * table["pct_detected"].to_numpy() / 100.0
    fig_width = max(12, len(genes) * 0.48)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        c=table["mean_logexpr"],
        cmap="Reds",
        edgecolors="#666666",
        linewidths=0.25,
    )
    ax.set_xticks(range(len(genes)), genes, rotation=90)
    ax.set_yticks(range(len(group_order)), group_order)
    ax.invert_yaxis()
    ax.set_title("Marker expression: size = detected %, color = mean log expression")
    fig.colorbar(scatter, ax=ax, label="Mean log-normalized expression")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_scrublet_qc(
    obs: pd.DataFrame,
    focus_state: str,
    high_threshold: float,
    strict_threshold: float,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    group_order = [f"target_{focus_state}", "target_other_LECs", "sample_other_nuclei"]
    data = [
        obs.loc[obs["target_group"].astype(str).eq(group), "scrublet_score_mean"]
        .dropna()
        .to_numpy()
        for group in group_order
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot(data, tick_labels=group_order, showfliers=True)
    axes[0].axhline(high_threshold, color="#ff7f0e", ls="--", lw=1, label="high")
    axes[0].axhline(strict_threshold, color="#d62728", ls="--", lw=1, label="strict")
    axes[0].set_ylabel("Mean Scrublet score across seeds")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(frameon=False)

    background = obs["target_group"].astype(str).eq("sample_other_nuclei")
    axes[1].scatter(
        obs.loc[background, "nCount_RNA"],
        obs.loc[background, "scrublet_score_mean"],
        s=8,
        c="#bdbdbd",
        alpha=0.25,
        linewidths=0,
    )
    colors = {f"target_{focus_state}": "#ff7f0e", "target_other_LECs": "#1f77b4"}
    for group, color in colors.items():
        mask = obs["target_group"].astype(str).eq(group)
        axes[1].scatter(
            obs.loc[mask, "nCount_RNA"],
            obs.loc[mask, "scrublet_score_mean"],
            s=35,
            c=color,
            alpha=0.85,
            label=group,
        )
    axes[1].axhline(high_threshold, color="#ff7f0e", ls="--", lw=1)
    axes[1].axhline(strict_threshold, color="#d62728", ls="--", lw=1)
    axes[1].set_xlabel("Raw UMI count")
    axes[1].set_ylabel("Mean Scrublet score")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Full-sample doublet assessment")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_neighbor_purity(
    obs: pd.DataFrame, focus_state: str, path: Path, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    groups = [f"target_{focus_state}", "target_other_LECs"]
    metrics = [
        "neighbor_author_LEC_fraction",
        "neighbor_endothelial_fraction",
        "neighbor_cardiac_fraction",
    ]
    labels = ["Author LEC", "Endothelial", "Cardiac"]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, group in enumerate(groups):
        frame = obs.loc[obs["target_group"].astype(str).eq(group)]
        medians = [float(frame[metric].median()) for metric in metrics]
        ax.bar(
            x + (index - 0.5) * width,
            medians,
            width,
            label=group,
            color=["#ff7f0e", "#1f77b4"][index],
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Median fraction among original graph neighbors")
    ax.set_title("Original all-cell neighborhood identity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_donor_cardiac_comparison(
    table: pd.DataFrame, target_donor: str, path: Path, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    ordered = table.sort_values("cardiac_mean_logexpr", ascending=True).reset_index(
        drop=True
    )
    colors = np.where(ordered["donor_key"].astype(str).eq(target_donor), "#d62728", "#9ecae1")
    fig, ax = plt.subplots(figsize=(9, max(5, len(ordered) * 0.16)))
    ax.barh(np.arange(len(ordered)), ordered["cardiac_mean_logexpr"], color=colors)
    ax.set_yticks(np.arange(len(ordered)), ordered["donor_key"], fontsize=6)
    ax.set_xlabel("Mean cardiomyocyte-panel log expression in Step 08 LECs")
    ax.set_title("Cardiac-expression signal across LEC donors")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_top_gene_bars(
    rank_tables: dict[str, pd.DataFrame],
    focus_state: str,
    top_n: int,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    groups = ["target_all_LECs", f"target_{focus_state}"]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, top_n * 0.25)))
    for ax, group in zip(axes, groups):
        table = rank_tables[group].head(top_n).sort_values("cpm")
        ax.barh(table["gene"], table["cpm"], color="#4c78a8")
        ax.set_xlabel("Pseudobulk CPM")
        ax.set_title(group)
    fig.suptitle("Most abundant genes in target donor LECs")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_interpretation_notes(
    path: Path,
    donor_key: str,
    sample_id: str,
    focus_state: str,
    summary: pd.DataFrame,
    thresholds: dict[str, float],
    target_n: int,
    sample_n: int,
    doublets_run: bool,
    neighbors_run: bool,
    step08_info: dict[str, object],
) -> None:
    lines = [
        "STEP 08.1 SPECIFIC-DONOR LEC VALIDATION",
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Source Step 08 script: {step08_info['script_version']}",
        f"Source Step 08 Leiden resolution: {step08_info['leiden_resolution']}",
        f"Source Step 08 states: {';'.join(step08_info['states'])}",
        f"Target donor: {donor_key}",
        f"Target sample: {sample_id}",
        f"Focus state: {focus_state}",
        f"Target Step 08 LECs: {target_n}",
        f"Complete sample nuclei used as reference: {sample_n}",
        f"Scrublet performed on complete sample: {doublets_run}",
        f"Original Step 03 neighborhood evaluated: {neighbors_run}",
        "",
        "AUTOMATED QC RESULTS",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"- {row['group']}: {row['automated_QC_conclusion']} "
            f"(n={row['n_cells']})"
        )
    lines.extend(
        [
            "",
            "DATA-ADAPTIVE REFERENCE THRESHOLDS",
            *[f"- {key}: {value:.6g}" for key, value in thresholds.items()],
            "",
            "INTERPRETATION RULES",
            "- LEC and pan-endothelial scores are calibrated against author-annotated",
            "  cells in the same 10x sample, not against arbitrary universal cutoffs.",
            "- Cardiac signal is called elevated either relative to the 95th",
            "  percentile of author-annotated non-cardiac nuclei in the same",
            "  sample or when the donor is in the top 10% of Step 08 donors.",
            "- A doublet-burden flag requires both elevated Scrublet evidence and",
            "  a material fraction of affected cells; one borderline cell cannot",
            "  explain an entire state.",
            "- Original-graph neighborhood identity is used as evidence independent",
            "  of the Step 08 LEC-state clustering.",
            "",
            "LIMITATIONS",
            "- Absolute gene-expression ranks are descriptive and can be dominated",
            "  by housekeeping genes or ambient RNA; specificity and detection",
            "  fractions must be considered together.",
            "- Scrublet can miss homotypic doublets and is not proof of singlet status.",
            "- The local count matrix contains 5,000 genes and already-called cells,",
            "  not unfiltered empty droplets. This script can identify a cardiac",
            "  expression signal but cannot formally estimate ambient molecules as",
            "  CellBender or a complete SoupX input would.",
            "- A single donor is not a biological replicate for inferential DE.",
            "",
            "No Step 03, Step 06, or Step 08 input is modified by this script.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not 0 < args.expected_doublet_rate < 1:
        raise ValueError("--expected_doublet_rate must be between 0 and 1.")
    if not 0 < args.scrublet_top_fraction < 1:
        raise ValueError("--scrublet_top_fraction must be between 0 and 1.")
    if not 0 < args.scrublet_high_threshold < args.scrublet_strict_threshold:
        raise ValueError(
            "Require 0 < high Scrublet threshold < strict Scrublet threshold."
        )
    if args.neighbor_n < 1:
        raise ValueError("--neighbor_n must be positive.")

    project_dir = find_project_dir()
    full_counts_path = resolve_path(args.full_counts_h5ad, project_dir)
    step08_metadata_path = resolve_path(args.step08_metadata, project_dir)
    step08_h5ad_path = resolve_path(args.step08_h5ad, project_dir)
    step03_root = resolve_path(args.step03_root, project_dir)
    base_output = resolve_path(args.output_root, project_dir)

    for path in [full_counts_path, step08_metadata_path, step08_h5ad_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    donor_key = str(args.donor_key).strip()
    metadata = pd.read_csv(step08_metadata_path)
    required = {"source_cell_id", "donor_key", "LEC_state", "age_quartile"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise KeyError(f"Step 08 metadata is missing columns: {missing}")
    step08_backed = ad.read_h5ad(step08_h5ad_path, backed="r")
    try:
        combine_info = dict(step08_backed.uns.get("step08_combine_info", {}))
        embedding_info = dict(
            step08_backed.uns.get("combined_LEC_embedding", {})
        )
        step08_states = sorted(
            normalized_text(step08_backed.obs["LEC_state"]).unique(),
            key=natural_key,
        )
        step08_info = {
            "script_version": str(combine_info.get("script_version", "unknown")),
            "leiden_resolution": float(
                embedding_info.get(
                    "leiden_resolution",
                    combine_info.get("leiden_resolution", np.nan),
                )
            ),
            "states": step08_states,
        }
    finally:
        step08_backed.file.close()
    target_metadata = metadata.loc[
        normalized_text(metadata["donor_key"]).eq(donor_key)
    ].copy()
    if target_metadata.empty:
        raise ValueError(f"Target donor {donor_key!r} is absent from Step 08.")

    sample_column = first_existing(["sample", "Sample"], target_metadata)
    if not sample_column:
        raise KeyError("Step 08 metadata lacks sample/Sample.")
    samples = normalized_text(target_metadata[sample_column]).unique().tolist()
    sample_id = str(args.sample).strip() if args.sample else ""
    if sample_id:
        target_metadata = target_metadata.loc[
            normalized_text(target_metadata[sample_column]).eq(sample_id)
        ].copy()
        if target_metadata.empty:
            raise ValueError(
                f"No Step 08 cells match donor={donor_key!r}, sample={sample_id!r}."
            )
    elif len(samples) == 1:
        sample_id = samples[0]
    else:
        raise ValueError(
            f"Donor {donor_key!r} maps to multiple samples {samples}; pass --sample."
        )

    state_totals = metadata.groupby("LEC_state", observed=True).size()
    donor_state = target_metadata.groupby("LEC_state", observed=True).size()
    donor_fraction = (donor_state / state_totals.reindex(donor_state.index)).sort_values(
        ascending=False
    )
    focus_state = str(args.focus_state).strip() or str(donor_fraction.index[0])
    if focus_state not in set(normalized_text(target_metadata["LEC_state"])):
        raise ValueError(
            f"Focus state {focus_state!r} has no cells from donor {donor_key!r}."
        )
    quartiles = normalized_text(target_metadata["age_quartile"]).unique().tolist()
    if len(quartiles) != 1:
        raise ValueError(
            f"Target donor/sample spans multiple quartiles {quartiles}; expected one."
        )
    quartile = quartiles[0]

    output_dir = base_output / sanitize_token(donor_key)
    figure_dir = output_dir / "figures"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {output_dir}. Pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target donor: {donor_key}")
    print(f"Target sample: {sample_id}")
    print(f"Focus state: {focus_state}")
    print(f"Target Step 08 LECs: {len(target_metadata)}")
    print(f"Reading full counts backed: {full_counts_path}")
    full = ad.read_h5ad(full_counts_path, backed="r")
    full_sample_column = first_existing(["sample", "Sample"], full.obs)
    if not full_sample_column:
        full.file.close()
        raise KeyError("Full-count object lacks sample/Sample.")
    sample_mask = normalized_text(full.obs[full_sample_column]).eq(sample_id).to_numpy()
    if not sample_mask.any():
        full.file.close()
        raise ValueError(f"Sample {sample_id!r} is absent from full counts.")
    sample = full[sample_mask].to_memory()
    full.file.close()
    count_qc = validate_raw_counts(sample)
    sample.obs_names = sample.obs_names.astype(str)
    if not sample.obs_names.is_unique:
        raise ValueError("Full-sample cell identifiers are not unique.")

    target_metadata["source_cell_id"] = normalized_text(
        target_metadata["source_cell_id"]
    )
    target_metadata = target_metadata.drop_duplicates("source_cell_id")
    target_ids = target_metadata["source_cell_id"].tolist()
    missing_ids = sorted(set(target_ids) - set(sample.obs_names))
    if missing_ids:
        raise KeyError(
            f"{len(missing_ids)} target cells do not map to the full sample; "
            f"first={missing_ids[:3]}"
        )
    target_lookup = target_metadata.set_index("source_cell_id")
    columns_to_join = [
        column
        for column in [
            "LEC_state",
            "combined_leiden",
            "donor_key",
            "celltypist_raw_label",
            "celltypist_majority_vote",
            "celltypist_max_probability",
            "step06_ec_leiden",
            "ec_recluster_id",
        ]
        if column in target_lookup.columns
    ]
    candidate = target_lookup[columns_to_join].copy()
    candidate = candidate.rename(
        columns={
            "donor_key": "target_donor_key",
            "LEC_state": "LEC_state",
        }
    )
    sample.obs = sample.obs.join(candidate, how="left")
    sample.obs["is_target_LEC"] = sample.obs.index.isin(target_ids)
    state_text = normalized_text(sample.obs["LEC_state"])
    sample.obs["target_group"] = np.where(
        sample.obs["is_target_LEC"] & state_text.eq(focus_state),
        f"target_{focus_state}",
        np.where(
            sample.obs["is_target_LEC"],
            "target_other_LECs",
            "sample_other_nuclei",
        ),
    )

    counts = sp.csr_matrix(sample.X, dtype=np.float64)
    log_matrix = normalize_log1p(counts)
    availability = marker_availability(sample.var_names)
    sample.obs, used_panels = add_signature_scores(
        sample.obs, log_matrix, sample.var_names
    )
    masks = group_masks_from_sample(sample.obs, focus_state)

    reference_masks = {
        "sample_non_target": masks["sample_non_target"],
        "author_other_endothelial": masks["author_other_endothelial"],
        "author_cardiomyocytes": masks["author_cardiomyocytes"],
    }
    rank_group_names = [
        "target_all_LECs",
        f"target_{focus_state}",
        "target_other_LECs",
        "author_Lymphatic_ECs",
        "author_other_endothelial",
        "author_cardiomyocytes",
    ]
    rank_tables: dict[str, pd.DataFrame] = {}
    for group in rank_group_names:
        if int(masks[group].sum()) == 0:
            warnings.warn(f"Skipping empty expression group: {group}")
            continue
        print(f"Ranking absolute expression: {group} (n={masks[group].sum()})")
        rank_tables[group] = expression_rank_table(
            group,
            masks[group],
            counts,
            log_matrix,
            sample.var_names,
            reference_masks,
        )

    marker_table = summarize_marker_expression(rank_tables, used_panels)
    author_celltype_summary = summarize_author_celltypes(sample.obs)

    scrublet_run_summary = pd.DataFrame()
    scrublet_summary = pd.DataFrame()
    scrublet_tests = pd.DataFrame()
    if not args.skip_doublets:
        scrublet, scrublet_run_summary = run_scrublet(
            sample,
            list(dict.fromkeys(args.scrublet_seeds)),
            args.expected_doublet_rate,
            args.scrublet_n_pcs,
            args.scrublet_high_threshold,
            args.scrublet_strict_threshold,
            args.scrublet_top_fraction,
        )
        sample.obs = sample.obs.join(scrublet, how="left")
        scrublet_summary, scrublet_tests = scrublet_group_summary(
            sample.obs, "target_group"
        )

    neighborhood = pd.DataFrame()
    step03_path = (
        step03_root / quartile / f"{DATASET}_{quartile}_clustered.h5ad"
    )
    if not args.skip_neighborhood:
        if not step03_path.exists():
            raise FileNotFoundError(step03_path)
        neighborhood = neighborhood_purity(
            step03_path, target_ids, int(args.neighbor_n)
        )
        sample.obs = sample.obs.join(neighborhood, how="left")

    donor_comparison = step08_donor_cardiac_comparison(
        step08_h5ad_path,
        used_panels["cardiomyocyte"],
        used_panels["LEC_specific"],
        donor_key,
        focus_state,
    )
    validation, thresholds = validation_summary(
        sample.obs,
        masks,
        focus_state,
        has_doublets=not args.skip_doublets,
        has_neighbors=not args.skip_neighborhood,
        donor_comparison=donor_comparison,
        target_donor=donor_key,
    )

    # Write tables only after every primary computation succeeds.
    availability.to_csv(
        output_dir / "08_1_signature_gene_availability.csv", index=False
    )
    marker_table.to_csv(
        output_dir / "08_1_marker_expression_by_group.csv", index=False
    )
    author_celltype_summary.to_csv(
        output_dir / "08_1_author_celltype_signature_summary.csv", index=False
    )
    donor_comparison.to_csv(
        output_dir / "08_1_step08_donor_cardiac_signature_comparison.csv",
        index=False,
    )
    validation.to_csv(
        output_dir / "08_1_group_validation_summary.csv", index=False
    )
    pd.DataFrame(
        [{"threshold": key, "value": value} for key, value in thresholds.items()]
    ).to_csv(output_dir / "08_1_validation_thresholds.csv", index=False)
    for group, table in rank_tables.items():
        table.to_csv(
            output_dir
            / f"08_1_gene_absolute_expression_rank_{sanitize_token(group)}.csv.gz",
            index=False,
            compression="gzip",
        )
    pd.concat(
        [table.head(100) for table in rank_tables.values()], ignore_index=True
    ).to_csv(output_dir / "08_1_top100_genes_each_group.csv", index=False)
    if not args.skip_doublets:
        scrublet_run_summary.to_csv(
            output_dir / "08_1_scrublet_run_summary.csv", index=False
        )
        scrublet_summary.to_csv(
            output_dir / "08_1_scrublet_group_summary.csv", index=False
        )
        scrublet_tests.to_csv(
            output_dir / "08_1_scrublet_group_tests.csv", index=False
        )
    if not neighborhood.empty:
        neighborhood.reset_index().to_csv(
            output_dir / "08_1_original_graph_neighbor_purity.csv", index=False
        )

    cell_output_columns = [
        column
        for column in [
            "sample",
            "Sample",
            "Donor",
            "Project",
            "cell.type",
            "broad.cell.type",
            "LEC_state",
            "combined_leiden",
            "target_donor_key",
            "is_target_LEC",
            "target_group",
            "nCount_RNA",
            "nFeature_RNA",
            "celltypist_raw_label",
            "celltypist_majority_vote",
            "celltypist_max_probability",
        ]
        if column in sample.obs.columns
    ]
    score_columns = [
        column
        for column in sample.obs.columns
        if column.endswith("_zscore")
        or column.endswith("_mean_logexpr")
        or column.endswith("_genes_detected")
        or column.endswith("_fraction_detected")
        or column.startswith("scrublet_")
        or column.startswith("neighbor_")
        or column == "n_neighbors_evaluated"
    ]
    cell_output = sample.obs[
        list(dict.fromkeys(cell_output_columns + score_columns))
    ].copy()
    cell_output.index.name = "source_cell_id"
    cell_output.reset_index().to_csv(
        output_dir / "08_1_full_sample_cell_QC.csv.gz",
        index=False,
        compression="gzip",
    )
    cell_output.loc[cell_output["is_target_LEC"].fillna(False)].reset_index().to_csv(
        output_dir / "08_1_target_donor_LEC_cell_QC.csv",
        index=False,
    )

    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        jobs = [
            (
                "08_1_identity_vs_cardiac_signature.png",
                lambda path: save_identity_scatter(
                    sample.obs, focus_state, path, args.dpi
                ),
            ),
            (
                "08_1_marker_expression_dotplot.png",
                lambda path: save_marker_dotplot(
                    marker_table,
                    [
                        f"target_{focus_state}",
                        "target_other_LECs",
                        "author_Lymphatic_ECs",
                        "author_other_endothelial",
                        "author_cardiomyocytes",
                    ],
                    used_panels,
                    path,
                    args.dpi,
                ),
            ),
            (
                "08_1_step08_donor_cardiac_signature.png",
                lambda path: save_donor_cardiac_comparison(
                    donor_comparison, donor_key, path, args.dpi
                ),
            ),
            (
                "08_1_top_absolute_expression_genes.png",
                lambda path: save_top_gene_bars(
                    rank_tables,
                    focus_state,
                    int(args.top_genes_figure),
                    path,
                    args.dpi,
                ),
            ),
        ]
        if not args.skip_doublets:
            jobs.append(
                (
                    "08_1_scrublet_QC.png",
                    lambda path: save_scrublet_qc(
                        sample.obs,
                        focus_state,
                        args.scrublet_high_threshold,
                        args.scrublet_strict_threshold,
                        path,
                        args.dpi,
                    ),
                )
            )
        if not args.skip_neighborhood:
            jobs.append(
                (
                    "08_1_original_graph_neighbor_purity.png",
                    lambda path: save_neighbor_purity(
                        sample.obs, focus_state, path, args.dpi
                    ),
                )
            )
        for filename, function in jobs:
            path = figure_dir / filename
            try:
                function(path)
                figure_manifest.append(
                    {"figure": str(path), "status": "ok"}
                )
            except Exception as error:
                warnings.warn(f"Figure failed ({filename}): {error}")
                figure_manifest.append(
                    {
                        "figure": str(path),
                        "status": f"failed: {type(error).__name__}: {error}",
                    }
                )
        pd.DataFrame(figure_manifest).to_csv(
            output_dir / "08_1_figure_manifest.csv", index=False
        )

    write_interpretation_notes(
        output_dir / "08_1_interpretation_notes.txt",
        donor_key,
        sample_id,
        focus_state,
        validation,
        thresholds,
        int(sample.obs["is_target_LEC"].sum()),
        int(sample.n_obs),
        not args.skip_doublets,
        not args.skip_neighborhood,
        step08_info,
    )

    run_config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "donor_key": donor_key,
        "sample": sample_id,
        "focus_state": focus_state,
        "age_quartile": quartile,
        "full_counts_h5ad": str(full_counts_path),
        "step08_metadata": str(step08_metadata_path),
        "step08_h5ad": str(step08_h5ad_path),
        "step08_source_info": step08_info,
        "step03_h5ad": str(step03_path),
        "output_dir": str(output_dir),
        "count_matrix_QC": count_qc,
        "marker_panels_requested": MARKER_PANELS,
        "marker_panels_used": used_panels,
        "scrublet_seeds": list(dict.fromkeys(args.scrublet_seeds)),
        "expected_doublet_rate": args.expected_doublet_rate,
        "scrublet_n_pcs": args.scrublet_n_pcs,
        "scrublet_high_threshold": args.scrublet_high_threshold,
        "scrublet_strict_threshold": args.scrublet_strict_threshold,
        "scrublet_top_fraction": args.scrublet_top_fraction,
        "neighbor_n": args.neighbor_n,
        "doublets_run": not args.skip_doublets,
        "neighborhood_run": not args.skip_neighborhood,
        "figures_run": not args.skip_figures,
    }
    with (output_dir / "08_1_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    run_summary = {
        "script_version": SCRIPT_VERSION,
        "status": "completed",
        "donor_key": donor_key,
        "sample": sample_id,
        "focus_state": focus_state,
        "step08_leiden_resolution": step08_info["leiden_resolution"],
        "step08_state_labels": ";".join(step08_info["states"]),
        "n_sample_cells": int(sample.n_obs),
        "n_target_LECs": int(sample.obs["is_target_LEC"].sum()),
        "n_focus_state_cells": int(
            (
                sample.obs["is_target_LEC"]
                & normalized_text(sample.obs["LEC_state"]).eq(focus_state)
            ).sum()
        ),
        "doublets_run": not args.skip_doublets,
        "neighborhood_run": not args.skip_neighborhood,
        "output_dir": str(output_dir),
    }
    pd.DataFrame([run_summary]).to_csv(
        output_dir / "08_1_run_summary.csv", index=False
    )

    print("\nDone.")
    print(pd.DataFrame(validation).to_string(index=False))
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
