#!/usr/bin/env python3
r"""
Step 10A: prepare integer-count inputs for disease-pipeline pySCENIC.

The current Step 08 object supplies the final LEC cells, state labels, donors,
datasets, conditions, and UMAP, but its expression is log-normalized. This
script uses each cell's original source-row index to recover the integer count
rows used in Step 03. Harmony, PCA, scaled expression, and GSEA rankings are
never used for SCENIC expression.

Two matrices are exported:

* ``10_expression_network_balanced_raw_counts.tsv`` is donor-, state-, and
  dataset-capped and is used for GRNBoost2 and cisTarget pruning.
* ``10_expression_all_LECs_raw_counts.tsv`` contains every Step 08 LEC and is
  used for AUCell, so downstream activity is available for every cell.

Genes must be detected in at least 100 cells of the balanced network subset and
occur in both local cisTarget databases. All passing TFs are retained, then the
most variable passing target genes are retained up to 10,000 total genes by
default. Every inclusion and exclusion is recorded in the gene-QC table.

Default output: ``new_result2_disease/10_SCENIC/input``

Run:
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\10_prepare_SCENIC_combined_LEC_states.py

Planning/validation without writing output:
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\10_prepare_SCENIC_combined_LEC_states.py `
      --validate_only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


SCRIPT_VERSION = "10A_disease_combined_LEC_SCENIC_integer_counts_balanced_2026_08_11"
DEFAULT_STEP08 = (
    "new_result2_disease/08_combined_LEC_recluster/"
    "08_combined_final_LEC_reclustered.h5ad"
)
DEFAULT_RESOURCES = "resources/scenic"
DEFAULT_OUTPUT = "new_result2_disease/10_SCENIC/input"

TF_FILENAME = "allTFs_hg38.txt"
DB10_FILENAME = (
    "hg38_10kbp_up_10kbp_down_full_tx_v10_clust."
    "genes_vs_motifs.rankings.feather"
)
DB500_FILENAME = (
    "hg38_500bp_up_100bp_down_full_tx_v10_clust."
    "genes_vs_motifs.rankings.feather"
)
MOTIF_FILENAME = "motifs-v10-nr.hgnc-m0.00001-o0.0.tbl"


@dataclass(frozen=True)
class CountSource:
    dataset: str
    h5ad: str
    h5_group: str
    transform: str = "none"
    total_column: str = ""
    normalization_target: float = 10000.0


COUNT_SOURCES = {
    "dataset01": CountSource(
        "dataset01",
        "Data/converted/human_dcm_hcm_scportal_03.17.2022.h5ad",
        "/layers/cellranger_raw",
    ),
    "dataset08": CountSource(
        "dataset08",
        "Data/converted/dataset_08_DoxCMP_standardized.h5ad",
        "/X",
    ),
    "dataset11": CountSource(
        "dataset11",
        "Data/converted/dataset_11_HFpEF_SCP3342_standardized.h5ad",
        "/X",
        transform="inverse_log1p_normalization_using_cell_total",
        total_column="n_counts",
    ),
}

EXPECTED_CONDITIONS = {"DCM", "DOXCM", "HCM", "HFPEF", "NF"}
ALLOWED_REGIONS = {
    "dataset01": {"AP", "LV"},
    "dataset08": {"AP", "LV"},
    "dataset11": {"S", "SEPTUM"},
}


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover integer counts and prepare donor/dataset-capped disease "
            "LEC inputs for pySCENIC."
        )
    )
    parser.add_argument("--step08_h5ad", default=DEFAULT_STEP08)
    parser.add_argument("--resources_dir", default=DEFAULT_RESOURCES)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--state_key", default="LEC_state")
    parser.add_argument("--donor_key", default="donor_key")
    parser.add_argument("--dataset_key", default="dataset_folder")
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--source_row_key", default="source_row_index")
    parser.add_argument("--source_cell_key", default="source_cell_id")
    parser.add_argument("--min_cells_gene", type=int, default=100)
    parser.add_argument("--max_genes", type=int, default=10000)
    parser.add_argument("--min_selected_tfs", type=int, default=500)
    parser.add_argument("--max_cells_per_donor_state", type=int, default=20)
    parser.add_argument("--max_cells_per_state_dataset", type=int, default=150)
    parser.add_argument("--max_cells_per_state", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--h5_max_gap_rows", type=int, default=256)
    parser.add_argument("--h5_max_window_rows", type=int, default=4096)
    parser.add_argument("--inverse_count_integer_tolerance", type=float, default=0.01)
    parser.add_argument("--export_chunk_size", type=int, default=50)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def natural_key(value: object) -> tuple:
    pieces = re.split(r"(\d+(?:\.\d+)?)", str(value))
    result = []
    for piece in pieces:
        if not piece:
            continue
        try:
            result.append((0, float(piece)))
        except ValueError:
            result.append((1, piece.lower()))
    return tuple(result)


def load_database_genes(path: Path) -> set[str]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as error:
        raise ImportError("pyarrow is required for cisTarget coverage checks.") from error
    with pa.memory_map(str(path), "r") as source:
        names = ipc.open_file(source).schema.names
    return {
        str(name) for name in names
        if str(name).lower() not in {"motifs", "tracks"}
    }


def validate_step08(obs: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    required = {
        args.state_key, args.donor_key, args.dataset_key, args.condition_key,
        args.source_row_key, args.source_cell_key,
        "region_standardized_analysis",
    }
    missing = required.difference(obs.columns)
    if missing:
        raise KeyError(f"Step 08 obs is missing required columns: {sorted(missing)}")
    if not obs.index.is_unique:
        raise ValueError("Step 08 cell IDs are not unique.")
    if normalized_text(obs[args.donor_key]).eq("").any():
        raise ValueError("Blank donor keys are present.")
    if normalized_text(obs[args.source_cell_key]).eq("").any():
        raise ValueError("Blank source-cell IDs are present.")

    datasets = set(normalized_text(obs[args.dataset_key]).str.lower().unique())
    if datasets != set(COUNT_SOURCES):
        raise ValueError(
            f"Expected datasets {sorted(COUNT_SOURCES)}, found {sorted(datasets)}."
        )
    conditions = set(normalized_text(obs[args.condition_key]).str.upper().unique())
    if conditions != EXPECTED_CONDITIONS:
        raise ValueError(
            f"Expected conditions {sorted(EXPECTED_CONDITIONS)}, found {sorted(conditions)}."
        )
    dataset_values = normalized_text(obs[args.dataset_key]).str.lower()
    region_values = normalized_text(obs["region_standardized_analysis"]).str.upper()
    region_report = {}
    for dataset in sorted(datasets):
        observed = set(region_values.loc[dataset_values.eq(dataset)].unique())
        unexpected = observed.difference(ALLOWED_REGIONS[dataset])
        if unexpected:
            raise ValueError(f"{dataset} has disallowed regions: {sorted(unexpected)}")
        region_report[dataset] = sorted(observed)
    return {
        "datasets": sorted(datasets),
        "conditions": sorted(conditions),
        "regions_by_dataset": region_report,
    }


def sparse_detection_and_variance(matrix) -> tuple[np.ndarray, np.ndarray]:
    if sp.issparse(matrix):
        csr = matrix.tocsr()
        detected = np.asarray(csr.getnnz(axis=0)).ravel().astype(np.int64)
        sums = np.asarray(csr.sum(axis=0)).ravel().astype(np.float64)
        squared = csr.copy().astype(np.float64)
        squared.data **= 2
        sums_sq = np.asarray(squared.sum(axis=0)).ravel()
        variance = np.maximum(sums_sq / csr.shape[0] - (sums / csr.shape[0]) ** 2, 0)
    else:
        values = np.asarray(matrix, dtype=float)
        detected = np.count_nonzero(values, axis=0).astype(np.int64)
        variance = values.var(axis=0)
    return detected, variance


def select_genes(
    raw,
    tf_names: set[str],
    db10_genes: set[str],
    db500_genes: set[str],
    min_cells: int,
    max_genes: int,
    min_tfs: int,
) -> tuple[pd.DataFrame, list[str]]:
    genes = pd.Index(raw.var_names.astype(str))
    if not genes.is_unique:
        raise ValueError("Step 08 raw gene names are not unique.")
    detected, variance = sparse_detection_and_variance(raw.X)
    qc = pd.DataFrame(
        {
            "gene": genes,
            "n_cells_detected": detected,
            "percent_cells_detected": 100.0 * detected / raw.n_obs,
            "log_expression_variance": variance,
            "is_candidate_TF": genes.isin(tf_names),
            "present_cisTarget_10kb": genes.isin(db10_genes),
            "present_cisTarget_500bp": genes.isin(db500_genes),
        }
    )
    qc["present_both_cisTarget_databases"] = (
        qc["present_cisTarget_10kb"] & qc["present_cisTarget_500bp"]
    )
    qc["passes_min_cells"] = qc["n_cells_detected"].ge(min_cells)
    qc["eligible_for_pyscenic"] = (
        qc["passes_min_cells"] & qc["present_both_cisTarget_databases"]
    )
    eligible = qc.loc[qc["eligible_for_pyscenic"]].copy()
    eligible_tfs = eligible.loc[eligible["is_candidate_TF"]]
    if len(eligible_tfs) < min_tfs:
        raise ValueError(
            f"Only {len(eligible_tfs)} eligible TFs; minimum required is {min_tfs}."
        )
    if max_genes < len(eligible_tfs):
        raise ValueError("max_genes is smaller than the number of eligible TFs.")

    selected = set(eligible_tfs["gene"])
    target_slots = max_genes - len(selected)
    targets = (
        eligible.loc[~eligible["gene"].isin(selected)]
        .sort_values(
            ["log_expression_variance", "n_cells_detected", "gene"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(target_slots)
    )
    selected.update(targets["gene"])
    qc["selected_for_pyscenic"] = qc["gene"].isin(selected)

    def status(row: pd.Series) -> str:
        if row["selected_for_pyscenic"]:
            return "selected_TF" if row["is_candidate_TF"] else "selected_target"
        if not row["passes_min_cells"]:
            return "excluded_detected_in_too_few_cells"
        if not row["present_both_cisTarget_databases"]:
            return "excluded_not_in_both_cisTarget_databases"
        return "excluded_by_max_genes_variance_rank"

    qc["selection_status"] = qc.apply(status, axis=1)
    selected_genes = genes[genes.isin(selected)].tolist()
    return qc, selected_genes


def select_network_cells(
    obs: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.random_seed)
    state_values = normalized_text(obs[args.state_key])
    donor_values = normalized_text(obs[args.donor_key])
    dataset_values = normalized_text(obs[args.dataset_key]).str.lower()
    selected: list[int] = []

    for state in sorted(state_values.unique(), key=natural_key):
        state_pool: list[int] = []
        datasets = sorted(dataset_values.loc[state_values.eq(state)].unique())
        for dataset in datasets:
            pair_pool: list[int] = []
            pair_mask = state_values.eq(state) & dataset_values.eq(dataset)
            for donor in sorted(donor_values.loc[pair_mask].unique()):
                positions = np.flatnonzero(
                    (pair_mask & donor_values.eq(donor)).to_numpy()
                )
                cap = min(len(positions), args.max_cells_per_donor_state)
                chosen = rng.choice(positions, size=cap, replace=False)
                pair_pool.extend(chosen.tolist())
            if len(pair_pool) > args.max_cells_per_state_dataset:
                pair_pool = rng.choice(
                    np.asarray(pair_pool),
                    size=args.max_cells_per_state_dataset,
                    replace=False,
                ).tolist()
            state_pool.extend(pair_pool)
        if len(state_pool) > args.max_cells_per_state:
            state_pool = rng.choice(
                np.asarray(state_pool), size=args.max_cells_per_state, replace=False
            ).tolist()
        selected.extend(state_pool)

    selected_array = np.asarray(sorted(set(selected)), dtype=np.int64)
    if len(selected_array) < 500:
        raise ValueError(f"Only {len(selected_array)} balanced network cells selected.")
    flag = np.zeros(len(obs), dtype=bool)
    flag[selected_array] = True
    audit = pd.DataFrame(
        {
            "cell_id": obs.index.astype(str),
            "selected_for_network_inference": flag,
            "LEC_state": state_values.to_numpy(),
            "donor_key": donor_values.to_numpy(),
            "dataset_folder": dataset_values.to_numpy(),
        }
    )
    summary = (
        audit.groupby(["LEC_state", "dataset_folder"], observed=True)
        .agg(
            n_all_cells=("cell_id", "size"),
            n_network_cells=("selected_for_network_inference", "sum"),
            n_all_donors=("donor_key", "nunique"),
            n_network_donors=(
                "donor_key",
                lambda values: values.loc[audit.loc[values.index, "selected_for_network_inference"]].nunique(),
            ),
        )
        .reset_index()
    )
    return selected_array, audit, summary


def _read_csr_window(
    node: h5py.Group, row_start: int, row_stop: int, n_vars: int
) -> sp.csr_matrix:
    pointers = np.asarray(node["indptr"][row_start : row_stop + 1], dtype=np.int64)
    data_start, data_stop = int(pointers[0]), int(pointers[-1])
    data = np.asarray(node["data"][data_start:data_stop])
    indices = np.asarray(node["indices"][data_start:data_stop], dtype=np.int32)
    return sp.csr_matrix(
        (data, indices, pointers - data_start), shape=(row_stop - row_start, n_vars)
    )


def read_selected_csr_rows(
    path: Path,
    group_path: str,
    positions: np.ndarray,
    max_gap_rows: int,
    max_window_rows: int,
) -> sp.csr_matrix:
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) == 0 or np.any(np.diff(positions) <= 0):
        raise ValueError("Selected source rows must be nonempty, sorted, and unique.")
    with h5py.File(path, "r") as handle:
        node = handle[group_path]
        if not isinstance(node, h5py.Group):
            blocks = []
            for start in range(0, len(positions), max_window_rows):
                chosen = positions[start : start + max_window_rows]
                blocks.append(sp.csr_matrix(np.asarray(node[chosen, :])))
            return sp.vstack(blocks, format="csr")
        encoding = str(node.attrs.get("encoding-type", ""))
        if encoding != "csr_matrix":
            raise ValueError(f"Expected CSR storage at {path}::{group_path}; got {encoding}.")
        n_rows, n_vars = (int(value) for value in node.attrs["shape"])
        if positions[0] < 0 or positions[-1] >= n_rows:
            raise IndexError("A requested source row is outside the matrix.")
        blocks = []
        start = 0
        windows = 0
        while start < len(positions):
            stop = start + 1
            window_start = int(positions[start])
            last = window_start
            while stop < len(positions):
                next_row = int(positions[stop])
                if (
                    next_row - last - 1 > max_gap_rows
                    or next_row - window_start + 1 > max_window_rows
                ):
                    break
                last = next_row
                stop += 1
            selected = positions[start:stop]
            full = _read_csr_window(node, window_start, int(selected[-1]) + 1, n_vars)
            blocks.append(full[selected - window_start, :].tocsr())
            start = stop
            windows += 1
            if windows == 1 or windows % 25 == 0 or stop == len(positions):
                print(
                    f"    windows={windows:,}; selected rows={stop:,}/{len(positions):,}"
                )
    return sp.vstack(blocks, format="csr")


def transform_counts(
    matrix: sp.csr_matrix,
    source_obs: pd.DataFrame,
    spec: CountSource,
    tolerance: float,
) -> tuple[sp.csr_matrix, dict[str, object]]:
    if spec.transform == "none":
        transformed = matrix.tocsr()
        report = {"count_transform": "none"}
    elif spec.transform == "inverse_log1p_normalization_using_cell_total":
        totals = pd.to_numeric(source_obs[spec.total_column], errors="coerce").to_numpy(float)
        if not np.isfinite(totals).all() or np.any(totals <= 0):
            raise ValueError(f"{spec.dataset}: invalid {spec.total_column} values.")
        transformed = matrix.astype(np.float64, copy=True).tocsr()
        transformed.data = np.expm1(transformed.data)
        transformed = sp.diags(
            totals / spec.normalization_target, format="csr"
        ).dot(transformed).tocsr()
        deviations = np.abs(transformed.data - np.rint(transformed.data))
        maximum_deviation = float(deviations.max()) if deviations.size else 0.0
        if maximum_deviation > tolerance:
            raise ValueError(
                f"{spec.dataset}: inverse-count deviation {maximum_deviation:g} "
                f"exceeds tolerance {tolerance:g}."
            )
        transformed.data = np.rint(transformed.data).astype(np.int32)
        transformed.eliminate_zeros()
        row_difference = np.max(
            np.abs(np.asarray(transformed.sum(axis=1)).ravel() - totals)
        )
        if row_difference > 0:
            raise ValueError(
                f"{spec.dataset}: reconstructed row-total difference={row_difference}."
            )
        report = {
            "count_transform": spec.transform,
            "inverse_transform_maximum_integer_deviation": maximum_deviation,
            "inverse_transform_maximum_row_sum_difference": float(row_difference),
        }
    else:
        raise ValueError(f"Unsupported count transform: {spec.transform}")
    if transformed.data.size == 0 or not np.isfinite(transformed.data).all():
        raise ValueError(f"{spec.dataset}: invalid or empty recovered count matrix.")
    if transformed.data.min() < 0 or not np.allclose(
        transformed.data, np.rint(transformed.data), rtol=0, atol=1e-8
    ):
        raise ValueError(f"{spec.dataset}: recovered values are not nonnegative integers.")
    transformed.data = np.rint(transformed.data).astype(np.int32)
    return transformed, {
        **report,
        "count_min_nonzero": int(transformed.data.min()),
        "count_max": int(transformed.data.max()),
        "count_n_nonzero": int(transformed.nnz),
        "count_integer_like": True,
    }


def align_gene_columns(
    matrix: sp.csr_matrix,
    source_genes: pd.Index,
    selected_genes: list[str],
) -> tuple[sp.csr_matrix, int]:
    locations = source_genes.get_indexer(selected_genes)
    present = locations >= 0
    if present.sum() < 1000:
        raise ValueError("Too few selected genes are present in a source dataset.")
    sub = matrix[:, locations[present]].tocoo()
    output_columns = np.flatnonzero(present)[sub.col]
    aligned = sp.csr_matrix(
        (sub.data, (sub.row, output_columns)),
        shape=(matrix.shape[0], len(selected_genes)),
        dtype=np.int32,
    )
    return aligned, int((~present).sum())


def recover_all_counts(
    obs: pd.DataFrame,
    selected_genes: list[str],
    project: Path,
    args: argparse.Namespace,
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    dataset_values = normalized_text(obs[args.dataset_key]).str.lower()
    blocks = []
    global_rows = []
    reports = []
    for dataset, spec in COUNT_SOURCES.items():
        source_path = resolve_path(spec.h5ad, project)
        global_index = np.flatnonzero(dataset_values.eq(dataset).to_numpy())
        source_positions = pd.to_numeric(
            obs.iloc[global_index][args.source_row_key], errors="raise"
        ).to_numpy(np.int64)
        if len(np.unique(source_positions)) != len(source_positions):
            raise ValueError(f"{dataset}: duplicated source-row indexes.")
        sort_order = np.argsort(source_positions)
        sorted_positions = source_positions[sort_order]
        source = ad.read_h5ad(source_path, backed="r")
        try:
            source_genes = pd.Index(source.var_names.astype(str))
            if not source_genes.is_unique:
                raise ValueError(f"{dataset}: source gene names are not unique.")
            source_obs = source.obs.iloc[sorted_positions].copy()
            expected_ids_with_step03_prefix = normalized_text(
                obs.iloc[global_index[sort_order]][args.source_cell_key]
            ).to_numpy()
            expected_ids = np.asarray(
                [
                    value.split("|", 2)[2]
                    if value.startswith(dataset + "|") and value.count("|") >= 2
                    else value
                    for value in expected_ids_with_step03_prefix
                ]
            )
            observed_ids = source.obs_names[sorted_positions].astype(str).to_numpy()
            if not np.array_equal(expected_ids, observed_ids):
                raise ValueError(f"{dataset}: source-row and source-cell IDs do not agree.")
        finally:
            try:
                source.file.close()
            except Exception:
                pass
        print(f"Recovering {len(sorted_positions):,} integer-count rows for {dataset}...")
        matrix = read_selected_csr_rows(
            source_path,
            spec.h5_group,
            sorted_positions,
            args.h5_max_gap_rows,
            args.h5_max_window_rows,
        )
        matrix, report = transform_counts(
            matrix, source_obs, spec, args.inverse_count_integer_tolerance
        )
        matrix, missing_genes = align_gene_columns(
            matrix, source_genes, selected_genes
        )
        inverse_order = np.argsort(sort_order)
        matrix = matrix[inverse_order, :].tocsr()
        blocks.append(matrix)
        global_rows.append(global_index)
        reports.append(
            {
                "dataset_folder": dataset,
                "source_h5ad": str(source_path),
                "source_h5_group": spec.h5_group,
                "n_cells": int(len(global_index)),
                "n_source_genes": int(len(source_genes)),
                "n_selected_genes_absent_from_source": missing_genes,
                **report,
            }
        )
    stacked = sp.vstack(blocks, format="csr")
    row_index = np.concatenate(global_rows)
    if set(row_index.tolist()) != set(range(len(obs))):
        raise RuntimeError("Recovered rows do not cover every Step 08 cell exactly once.")
    stacked = stacked[np.argsort(row_index), :].tocsr()
    return stacked, pd.DataFrame(reports)


def export_tsv(
    matrix: sp.csr_matrix,
    cells: list[str],
    genes: list[str],
    path: Path,
    chunk_size: int,
) -> None:
    if matrix.shape != (len(cells), len(genes)):
        raise ValueError("Expression matrix, cell IDs, and gene IDs do not align.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["Cell", *genes])
        for start in range(0, len(cells), chunk_size):
            stop = min(start + chunk_size, len(cells))
            block = matrix[start:stop].toarray()
            for offset, row in enumerate(block):
                writer.writerow([cells[start + offset], *row.astype(np.int64)])
            if start == 0 or stop == len(cells) or (start // chunk_size) % 20 == 0:
                print(f"  exported cells {start + 1:,}-{stop:,}/{len(cells):,}")


def main() -> int:
    args = parse_args()
    positive = [
        args.min_cells_gene, args.max_genes, args.min_selected_tfs,
        args.max_cells_per_donor_state, args.max_cells_per_state_dataset,
        args.max_cells_per_state, args.export_chunk_size,
    ]
    if any(value < 1 for value in positive):
        raise ValueError("All filtering, sampling, and chunk parameters must be positive.")
    if not 0 < args.inverse_count_integer_tolerance < 0.5:
        raise ValueError("inverse_count_integer_tolerance must be in (0, 0.5).")

    project = find_project_dir()
    step08_path = resolve_path(args.step08_h5ad, project)
    resources = resolve_path(args.resources_dir, project)
    output = resolve_path(args.output_root, project)
    all_expression = output / "10_expression_all_LECs_raw_counts.tsv"
    network_expression = output / "10_expression_network_balanced_raw_counts.tsv"
    if not step08_path.is_file():
        raise FileNotFoundError(f"Missing Step 08 h5ad: {step08_path}")
    required_resources = {
        "TF_list": resources / TF_FILENAME,
        "database_10kb": resources / DB10_FILENAME,
        "database_500bp": resources / DB500_FILENAME,
        "motif_annotations": resources / MOTIF_FILENAME,
    }
    for label, path in required_resources.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {label}: {path}")
    for dataset, spec in COUNT_SOURCES.items():
        path = resolve_path(spec.h5ad, project)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing source h5ad for {dataset}: {path}")

    print(f"Reading Step 08: {step08_path}")
    step08 = ad.read_h5ad(step08_path)
    if step08.raw is None:
        raise ValueError("Step 08 adata.raw is required for gene filtering.")
    obs = step08.obs.copy()
    validation = validate_step08(obs, args)
    step08_info = dict(step08.uns.get("step08_combine_info", {}))
    embedding_info = dict(step08.uns.get("combined_LEC_embedding", {}))

    network_rows, sampling_audit, sampling_summary = select_network_cells(obs, args)
    print("Reading SCENIC reference coverage and selecting genes...")
    tf_names = {
        value.strip()
        for value in required_resources["TF_list"].read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    db10 = load_database_genes(required_resources["database_10kb"])
    db500 = load_database_genes(required_resources["database_500bp"])
    gene_qc, selected_genes = select_genes(
        step08.raw[network_rows, :],
        tf_names,
        db10,
        db500,
        args.min_cells_gene,
        args.max_genes,
        args.min_selected_tfs,
    )
    n_selected_tfs = int(
        gene_qc.loc[gene_qc["selected_for_pyscenic"], "is_candidate_TF"].sum()
    )
    states = sorted(normalized_text(obs[args.state_key]).unique(), key=natural_key)
    print(
        f"Plan: {len(obs):,} AUCell cells; {len(network_rows):,} balanced GRN cells; "
        f"{len(selected_genes):,} genes; {n_selected_tfs:,} TFs; {len(states)} states."
    )
    print(sampling_summary.to_string(index=False))
    if args.validate_only:
        print("Validation-only mode complete; no files were written.")
        return 0

    if (all_expression.exists() or network_expression.exists()) and not args.overwrite:
        raise FileExistsError(
            "A primary Step 10A matrix already exists. Use --overwrite only when "
            "replacement is intended."
        )
    output.mkdir(parents=True, exist_ok=True)
    counts, count_provenance = recover_all_counts(
        obs, selected_genes, project, args
    )
    if counts.shape != (len(obs), len(selected_genes)):
        raise RuntimeError("Recovered count matrix shape is inconsistent.")

    cells = obs.index.astype(str).tolist()
    print("Exporting all-cell AUCell expression matrix...")
    export_tsv(
        counts, cells, selected_genes, all_expression, args.export_chunk_size
    )
    print("Exporting balanced GRN/CTX expression matrix...")
    export_tsv(
        counts[network_rows, :],
        obs.index[network_rows].astype(str).tolist(),
        selected_genes,
        network_expression,
        args.export_chunk_size,
    )

    metadata_columns = [
        args.state_key, args.donor_key, args.dataset_key, args.condition_key,
        "sex", "age", "cohort", "region_standardized_analysis",
        "ec_recluster_id", "celltypist_raw_label", "celltypist_majority_vote",
        args.source_row_key, args.source_cell_key,
    ]
    metadata_columns = [column for column in metadata_columns if column in obs]
    metadata = obs[metadata_columns].copy()
    metadata.insert(0, "cell_id", obs.index.astype(str))
    metadata["selected_for_network_inference"] = sampling_audit[
        "selected_for_network_inference"
    ].to_numpy()
    metadata.to_csv(output / "10_cell_metadata_for_pyscenic.csv", index=False)
    gene_qc.to_csv(output / "10_gene_filter_and_reference_QC.csv", index=False)
    gene_qc.loc[gene_qc["is_candidate_TF"]].to_csv(
        output / "10_TF_coverage.csv", index=False
    )
    sampling_audit.to_csv(output / "10_network_cell_sampling_audit.csv", index=False)
    sampling_summary.to_csv(output / "10_network_sampling_state_dataset_QC.csv", index=False)
    count_provenance.to_csv(output / "10_count_recovery_by_dataset.csv", index=False)

    donor_state_counts = (
        obs.assign(LEC_state=normalized_text(obs[args.state_key]))
        .groupby(["LEC_state", args.dataset_key, args.donor_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    donor_state_counts.to_csv(output / "10_donor_state_cell_counts.csv", index=False)

    summary = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project),
        "step08_h5ad": str(step08_path),
        "step08_script_version": str(step08_info.get("script_version", "unknown")),
        "step08_leiden_resolution": float(
            embedding_info.get("leiden_resolution", step08_info.get("leiden_resolution", np.nan))
        ),
        "n_all_LEC_cells": int(len(obs)),
        "n_network_balanced_cells": int(len(network_rows)),
        "n_states": int(len(states)),
        "states": states,
        "n_step08_raw_genes": int(step08.raw.n_vars),
        "gene_filter_cell_scope": "donor_dataset_capped_network_cells",
        "n_genes_selected": int(len(selected_genes)),
        "n_candidate_TFs_reference": int(len(tf_names)),
        "n_candidate_TFs_selected": n_selected_tfs,
        "min_cells_gene": args.min_cells_gene,
        "max_genes": args.max_genes,
        "max_cells_per_donor_state": args.max_cells_per_donor_state,
        "max_cells_per_state_dataset": args.max_cells_per_state_dataset,
        "max_cells_per_state": args.max_cells_per_state,
        "random_seed": args.random_seed,
        "expression_source": "original_integer_counts_recovered_by_source_row_index",
        "network_expression_tsv": str(network_expression),
        "all_cell_expression_tsv": str(all_expression),
        "network_use": "GRNBoost2_and_CTX",
        "all_cell_use": "AUCell",
        "batch_correction": "none_expression_not_Harmony_corrected",
        "validation": validation,
        "resources": {key: str(path) for key, path in required_resources.items()},
    }
    pd.DataFrame(
        [
            {
                "item": key,
                "value": json.dumps(value) if isinstance(value, (list, dict)) else value,
            }
            for key, value in summary.items()
        ]
    ).to_csv(output / "10_SCENIC_preparation_summary.csv", index=False)
    (output / "10_SCENIC_preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nStep 10A preparation complete.")
    print(f"All-cell AUCell matrix: {all_expression}")
    print(f"Balanced GRN/CTX matrix: {network_expression}")
    print(
        "Next run: new_script2_disease/10_run_pyscenic_combined_LEC_states.ps1"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
