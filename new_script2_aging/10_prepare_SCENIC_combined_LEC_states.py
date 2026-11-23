#!/usr/bin/env python3
r"""
Step 10A: prepare raw-count input for pySCENIC on combined aging LEC states.

The Step 08 object contains the final 1,119 NF AP/LV LECs and their combined
state assignments, but its X/raw values are log-normalized. This script uses
``source_cell_id`` to recover the matching INTEGER count rows from the original
dataset13 h5ad. It never uses Harmony, PCA, scaled values, or GSEA rankings.

The available dataset13 source has 5,000 genes. That is the maximum feature
space available to this pipeline. Genes are retained when detected in at least
three selected LECs by default. cisTarget and TF-list coverage are reported for
every gene; genes are not silently removed merely because a reference database
does not contain them.

Default inputs
--------------
new_result2_aging/08_combined_LEC_recluster/dataset13/
    08_combined_final_LEC_reclustered.h5ad
Data/converted/dataset_13_Age_full_30plus_age10yr_standardized.h5ad
resources/scenic/*

Default output
--------------
new_result2_aging/10_SCENIC/dataset13/input/

Main outputs
------------
10_expression_raw_counts_for_pyscenic.tsv   cells x genes
10_cell_metadata_for_pyscenic.csv
10_gene_filter_and_reference_QC.csv
10_TF_coverage.csv
10_SCENIC_preparation_summary.csv/json

Run
---
    .\.venv\Scripts\python.exe `
      .\new_script2_aging\10_prepare_SCENIC_combined_LEC_states.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SCRIPT_VERSION = "10A_dataset13_NF_combined_LEC_SCENIC_prepare_resolution_aware_2026_08_07"
DEFAULT_STEP08_H5AD = (
    "new_result2_aging/08_combined_LEC_recluster/dataset13/"
    "08_combined_final_LEC_reclustered.h5ad"
)
DEFAULT_SOURCE_COUNTS = (
    "Data/converted/dataset_13_Age_full_30plus_age10yr_standardized.h5ad"
)
DEFAULT_RESOURCES = "resources/scenic"
DEFAULT_OUTPUT = "new_result2_aging/10_SCENIC/dataset13/input"

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


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover integer counts for the combined Step 08 aging LECs and "
            "export a validated pySCENIC expression matrix."
        )
    )
    parser.add_argument("--step08_h5ad", default=DEFAULT_STEP08_H5AD)
    parser.add_argument("--source_counts_h5ad", default=DEFAULT_SOURCE_COUNTS)
    parser.add_argument("--resources_dir", default=DEFAULT_RESOURCES)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--state_key", default="LEC_state")
    parser.add_argument("--source_cell_key", default="source_cell_id")
    parser.add_argument("--donor_key", default="donor_key")
    parser.add_argument("--min_cells_gene", type=int, default=3)
    parser.add_argument("--min_total_counts_gene", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing primary expression TSV.",
    )
    return parser.parse_args()


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def first_existing(columns: Iterable[str], frame: pd.DataFrame) -> Optional[str]:
    return next((column for column in columns if column in frame.columns), None)


def natural_key(value: object) -> tuple:
    import re

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
        raise ImportError(
            "pyarrow is required to validate cisTarget database coverage."
        ) from error
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        names = reader.schema.names
    return {str(name) for name in names if str(name).lower() not in {"motifs", "tracks"}}


def validate_step08(obs: pd.DataFrame, state_key: str, source_cell_key: str) -> None:
    for column in [state_key, source_cell_key]:
        if column not in obs.columns:
            raise KeyError(f"Required Step 08 obs column is absent: {column}")
    source_ids = normalized_text(obs[source_cell_key])
    if source_ids.eq("").any():
        raise ValueError(f"Blank values are present in {source_cell_key}.")
    if source_ids.duplicated().any():
        examples = source_ids.loc[source_ids.duplicated(keep=False)].head().tolist()
        raise ValueError(f"{source_cell_key} is not unique. Examples: {examples}")

    disease_col = first_existing(["disease", "aging_disease_filter"], obs)
    if disease_col:
        found = set(normalized_text(obs[disease_col]).str.upper().unique())
        if found != {"NF"}:
            raise ValueError(f"Expected NF only; {disease_col} contains {sorted(found)}")
    region_col = first_existing(
        ["aging_standardized_region", "Standardized_region"], obs
    )
    if region_col:
        found = set(normalized_text(obs[region_col]).str.upper().unique())
        unexpected = found.difference({"AP", "LV"})
        if unexpected:
            raise ValueError(
                f"Expected AP/LV only; {region_col} also contains {sorted(unexpected)}"
            )


def count_statistics(matrix) -> tuple[np.ndarray, np.ndarray]:
    if sp.issparse(matrix):
        csr = matrix.tocsr()
        detected = np.asarray(csr.getnnz(axis=0)).ravel().astype(np.int64)
        totals = np.asarray(csr.sum(axis=0)).ravel().astype(np.float64)
    else:
        values = np.asarray(matrix)
        detected = np.count_nonzero(values, axis=0).astype(np.int64)
        totals = values.sum(axis=0, dtype=np.float64)
    return detected, totals


def validate_count_matrix(matrix) -> dict[str, object]:
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
    if values.size == 0:
        raise ValueError("Recovered count matrix contains no nonzero values.")
    if not np.isfinite(values).all():
        raise ValueError("Recovered count matrix contains non-finite values.")
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < 0:
        raise ValueError(f"Recovered count matrix contains negative values: min={minimum}")
    integer_like = bool(np.allclose(values, np.rint(values), rtol=0, atol=1e-8))
    if not integer_like:
        raise ValueError(
            "Recovered source matrix is not integer-like. Refusing to label it raw counts."
        )
    return {
        "count_minimum_nonzero": minimum,
        "count_maximum": maximum,
        "integer_like": integer_like,
        "n_nonzero": int(values.size if sp.issparse(matrix) else np.count_nonzero(values)),
    }


def exclusion_reason(row: pd.Series) -> str:
    reasons = []
    if not row["passes_min_cells"]:
        reasons.append("detected_in_too_few_cells")
    if not row["passes_min_total_counts"]:
        reasons.append("total_counts_below_threshold")
    return ";".join(reasons) if reasons else "retained"


def export_tsv(matrix, cells: list[str], genes: list[str], path: Path, chunk: int) -> None:
    if chunk < 1:
        raise ValueError("chunk_size must be at least 1.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["Cell", *genes])
        for start in range(0, len(cells), chunk):
            end = min(start + chunk, len(cells))
            block = matrix[start:end]
            if sp.issparse(block):
                block = block.toarray()
            block = np.asarray(block)
            for offset, row in enumerate(block):
                writer.writerow([cells[start + offset], *np.rint(row).astype(np.int64)])
            print(f"  exported cells {start + 1:,}-{end:,} / {len(cells):,}")


def main() -> int:
    args = parse_args()
    if args.min_cells_gene < 1 or args.min_total_counts_gene < 1:
        raise ValueError("Gene thresholds must be at least 1.")

    project_dir = find_project_dir()
    step08_path = resolve_path(args.step08_h5ad, project_dir)
    source_path = resolve_path(args.source_counts_h5ad, project_dir)
    resources_dir = resolve_path(args.resources_dir, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    expression_path = output_root / "10_expression_raw_counts_for_pyscenic.tsv"
    if expression_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Primary output already exists: {expression_path}\n"
            "Use --overwrite only when replacement is intended."
        )

    required_resources = {
        "TF_list": resources_dir / TF_FILENAME,
        "database_10kb": resources_dir / DB10_FILENAME,
        "database_500bp": resources_dir / DB500_FILENAME,
        "motif_annotations": resources_dir / MOTIF_FILENAME,
    }
    for label, path in {
        "Step 08 h5ad": step08_path,
        "source count h5ad": source_path,
        **required_resources,
    }.items():
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {label}: {path}")
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Reading final LEC metadata: {step08_path}")
    step08 = ad.read_h5ad(step08_path, backed="r")
    try:
        obs = step08.obs.copy()
        step08_cells = step08.obs_names.astype(str).tolist()
        n_step08_genes = int(step08.n_vars)
        step08_combine_info = dict(step08.uns.get("step08_combine_info", {}))
        step08_embedding_info = dict(
            step08.uns.get("combined_LEC_embedding", {})
        )
    finally:
        try:
            step08.file.close()
        except Exception:
            pass
    validate_step08(obs, args.state_key, args.source_cell_key)
    source_cell_ids = normalized_text(obs[args.source_cell_key])

    print(f"Recovering integer counts: {source_path}")
    source = ad.read_h5ad(source_path, backed="r")
    try:
        if not source.obs_names.is_unique:
            raise ValueError("Source count h5ad obs_names are not unique.")
        if not source.var_names.is_unique:
            raise ValueError("Source count h5ad var_names are not unique.")
        source_index = pd.Series(
            np.arange(source.n_obs, dtype=np.int64),
            index=source.obs_names.astype(str),
        )
        missing_cells = source_cell_ids.loc[~source_cell_ids.isin(source_index.index)]
        if len(missing_cells):
            raise ValueError(
                f"{len(missing_cells)} Step 08 source cell IDs are absent from the "
                f"original counts. Examples: {missing_cells.head().tolist()}"
            )
        positions = source_index.loc[source_cell_ids].to_numpy(dtype=np.int64)
        recovered = source[positions, :].to_memory()
        source_shape = [int(source.n_obs), int(source.n_vars)]
    finally:
        try:
            source.file.close()
        except Exception:
            pass

    recovered.obs_names = pd.Index(step08_cells)
    recovered.var_names = recovered.var_names.astype(str)
    count_info = validate_count_matrix(recovered.X)
    detected, totals = count_statistics(recovered.X)

    print("Reading SCENIC reference coverage...")
    tf_names = {
        value.strip()
        for value in required_resources["TF_list"].read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    db10_genes = load_database_genes(required_resources["database_10kb"])
    db500_genes = load_database_genes(required_resources["database_500bp"])
    genes = pd.Index(recovered.var_names.astype(str))
    gene_qc = pd.DataFrame(
        {
            "gene": genes,
            "n_cells_detected": detected,
            "percent_cells_detected": 100.0 * detected / recovered.n_obs,
            "total_raw_counts": totals,
            "is_candidate_TF": genes.isin(tf_names),
            "present_cisTarget_10kb": genes.isin(db10_genes),
            "present_cisTarget_500bp": genes.isin(db500_genes),
        }
    )
    gene_qc["present_both_cisTarget_databases"] = (
        gene_qc["present_cisTarget_10kb"] & gene_qc["present_cisTarget_500bp"]
    )
    gene_qc["passes_min_cells"] = gene_qc["n_cells_detected"].ge(
        args.min_cells_gene
    )
    gene_qc["passes_min_total_counts"] = gene_qc["total_raw_counts"].ge(
        args.min_total_counts_gene
    )
    gene_qc["selected_for_pyscenic"] = (
        gene_qc["passes_min_cells"] & gene_qc["passes_min_total_counts"]
    )
    gene_qc["filter_status"] = gene_qc.apply(exclusion_reason, axis=1)
    selected_mask = gene_qc["selected_for_pyscenic"].to_numpy(bool)
    if selected_mask.sum() < 1000:
        raise ValueError(
            f"Only {selected_mask.sum()} genes pass filtering; this is unexpectedly low."
        )
    n_selected_tfs = int(
        gene_qc.loc[selected_mask, "is_candidate_TF"].sum()
    )
    if n_selected_tfs < 100:
        raise ValueError(
            f"Only {n_selected_tfs} candidate TFs pass filtering; refusing to proceed."
        )

    selected_genes = genes[selected_mask].tolist()
    selected_matrix = recovered.X[:, selected_mask]
    print(
        f"Exporting {recovered.n_obs:,} cells x {len(selected_genes):,} genes "
        f"({n_selected_tfs:,} candidate TFs)."
    )
    export_tsv(
        selected_matrix,
        step08_cells,
        selected_genes,
        expression_path,
        args.chunk_size,
    )

    metadata_columns = [
        args.state_key,
        args.donor_key,
        "combined_leiden",
        "age_quartile",
        "age_quartile_range_label",
        "age_numeric_resolved",
        "Project",
        "Sex",
        "sex",
        "aging_standardized_region",
        "Standardized_region",
        "disease",
        args.source_cell_key,
    ]
    metadata_columns = list(dict.fromkeys(c for c in metadata_columns if c in obs.columns))
    metadata = obs[metadata_columns].copy()
    metadata.insert(0, "cell_id", step08_cells)
    metadata.to_csv(output_root / "10_cell_metadata_for_pyscenic.csv", index=False)

    state_order = sorted(normalized_text(obs[args.state_key]).unique(), key=natural_key)
    state_rows = []
    for state in state_order:
        group = obs.loc[normalized_text(obs[args.state_key]).eq(state)]
        donor_counts = (
            normalized_text(group[args.donor_key]).value_counts()
            if args.donor_key in group.columns
            else pd.Series(dtype=int)
        )
        state_rows.append(
            {
                "LEC_state": state,
                "n_cells": int(len(group)),
                "n_donors": (
                    int(group[args.donor_key].nunique())
                    if args.donor_key in group.columns
                    else np.nan
                ),
                "top_donor": donor_counts.index[0] if len(donor_counts) else "",
                "top_donor_fraction": (
                    float(donor_counts.iloc[0] / len(group))
                    if len(group) and len(donor_counts)
                    else np.nan
                ),
            }
        )
    pd.DataFrame(state_rows).to_csv(
        output_root / "10_state_cell_donor_QC.csv", index=False
    )
    gene_qc.to_csv(
        output_root / "10_gene_filter_and_reference_QC.csv", index=False
    )
    gene_qc.loc[gene_qc["is_candidate_TF"]].to_csv(
        output_root / "10_TF_coverage.csv", index=False
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "step08_h5ad": str(step08_path),
        "source_counts_h5ad": str(source_path),
        "expression_tsv": str(expression_path),
        "n_cells": int(recovered.n_obs),
        "n_states": int(len(state_order)),
        "states": state_order,
        "step08_script_version": str(
            step08_combine_info.get("script_version", "unknown")
        ),
        "step08_leiden_resolution": float(
            step08_embedding_info.get(
                "leiden_resolution",
                step08_combine_info.get("leiden_resolution", np.nan),
            )
        ),
        "step08_excluded_donors": list(
            step08_combine_info.get("excluded_donors", [])
        ),
        "source_h5ad_shape": source_shape,
        "n_genes_in_available_source": int(recovered.n_vars),
        "n_genes_in_step08": n_step08_genes,
        "n_genes_selected": int(selected_mask.sum()),
        "n_candidate_TFs_in_reference": int(len(tf_names)),
        "n_candidate_TFs_in_available_source": int(gene_qc["is_candidate_TF"].sum()),
        "n_candidate_TFs_selected": n_selected_tfs,
        "n_selected_genes_in_cisTarget_10kb": int(
            gene_qc.loc[selected_mask, "present_cisTarget_10kb"].sum()
        ),
        "n_selected_genes_in_cisTarget_500bp": int(
            gene_qc.loc[selected_mask, "present_cisTarget_500bp"].sum()
        ),
        "min_cells_gene": args.min_cells_gene,
        "min_total_counts_gene": args.min_total_counts_gene,
        "expression_source": "original_dataset13_integer_counts_X",
        "feature_space_limitation": "available source contains 5000 genes",
        **count_info,
        "resources": {key: str(value) for key, value in required_resources.items()},
    }
    pd.DataFrame(
        [{"item": key, "value": json.dumps(value) if isinstance(value, (list, dict)) else value}
         for key, value in summary.items()]
    ).to_csv(output_root / "10_SCENIC_preparation_summary.csv", index=False)
    (output_root / "10_SCENIC_preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nStep 10A preparation complete.")
    print(f"Expression matrix: {expression_path}")
    print(
        "Next run: new_script2_aging/10_run_pyscenic_combined_LEC_states.ps1"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
