#!/usr/bin/env python
r"""
Step 03 for the selected human cardiac disease pipeline.

This is a standalone, memory-conscious counterpart of the aging Step 03. It
does not import the older disease scripts and it never combines datasets at
this stage. Every dataset-condition is filtered, quality-controlled, and
clustered independently so that EC and LEC discovery can also be performed
independently in later steps.

Approved cohort
---------------
dataset01: DCM, HCM, NF                  documented LV
dataset08: DoxCM, NF                     documented LV; NICM excluded
dataset11: HFpEF, NF                     approved septum exception

Dataset02 and dataset06 are excluded from the active LEC pipeline after the
completed screening run: dataset02 produced no final LEC calls in either DCM
or NF, while dataset06 produced final LEC calls only in NF and therefore cannot
support a within-dataset DCM-versus-NF LEC comparison. This is a downstream
LEC-usability exclusion, not a claim that their Step 03 QC failed. Dataset10 is
excluded because all seven sample identifiers end in -RV and no LV/AP cells are
present. Dataset12 COVID is also excluded because it is no longer needed in the
current disease analysis.

Dataset11 retains only a 10,000-total log1p-normalized X matrix. Integer counts
are recovered using each cell's n_counts value for count-based QC and
seurat_v3 HVG selection. Recovery allows up to 0.01 count of floating-point
error before rounding and still requires every reconstructed per-cell total to
match n_counts exactly.

Default QC and clustering parameters match the actual aging Step 03 run:
  * raw integer counts;
  * >=150 genes, >=150 counts, <=5% mitochondrial counts;
  * mean +/- 1.5*IQR filtering available but OFF by default;
  * 2,000 HVGs, up to 100 PCs, donor Harmony, 15 cosine neighbors;
  * UMAP min_dist 0.2 and Leiden resolution 0.6.

All genes are retained in the normalized object. HVGs are used only for
dimensional reduction. UMAP coordinates are computed, but figures are left to
Step 04.

Large-file behavior
-------------------
The source H5AD files are 0.3-33 GB. AnnData backed mode may materialize an
entire sparse layer when it is accessed. To avoid this, this script reads only
the selected raw-count CSR rows directly from HDF5, in bounded row windows.
It processes one dataset-condition at a time and does not create a combined
million-cell matrix.

Default output
--------------
new_result2_disease/03/<dataset_folder>/<condition>/

Read-only cohort check
----------------------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\03_cluster_selected_disease_datasets_QC.py `
      --validate_only

Full run
--------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\03_cluster_selected_disease_datasets_QC.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import scanpy as sc
except ImportError as error:
    raise ImportError(
        "scanpy is required. Run this script with the project .venv."
    ) from error


SCRIPT_VERSION = "03_selected_disease_exclude_dataset02_dataset06_dataset12_2026_08_11"
DEFAULT_OUTPUT_ROOT = "new_result2_disease/03"
DEFAULT_RANDOM_SEED = 0
DEFAULT_INVERSE_COUNT_INTEGER_TOLERANCE = 0.01


@dataclass(frozen=True)
class DatasetSpec:
    dataset_folder: str
    dataset_id: str
    input_h5ad: str
    count_h5_path: str
    condition_col: str
    condition_map: dict[str, str]
    condition_order: tuple[str, ...]
    donor_col: str = "donor_id"
    sample_col: str = "biosample_id"
    region_col: str = ""
    region_map: Optional[dict[str, str]] = None
    documented_region: str = ""
    region_filter_basis: str = "dataset_level_documentation"
    region_policy: str = "AP_LV_standard"
    has_within_dataset_nf: bool = True
    comparison_limitation: str = ""
    count_transform: str = "none"
    count_total_col: str = ""
    count_normalization_target: float = 10000.0


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_folder="dataset01",
        dataset_id="dataset_01_DCM_HCM",
        input_h5ad="Data/converted/human_dcm_hcm_scportal_03.17.2022.h5ad",
        count_h5_path="/layers/cellranger_raw",
        condition_col="disease",
        condition_map={"DCM": "DCM", "HCM": "HCM", "NF": "NF"},
        condition_order=("DCM", "HCM", "NF"),
        documented_region="LV",
        region_filter_basis="dataset_level_documented_left_ventricle",
    ),
    DatasetSpec(
        dataset_folder="dataset08",
        dataset_id="dataset_08_DoxCMP",
        input_h5ad="Data/converted/dataset_08_DoxCMP_standardized.h5ad",
        count_h5_path="/X",
        condition_col="disease",
        condition_map={"DoxCM": "DoxCM", "NF": "NF"},
        condition_order=("DoxCM", "NF"),
        documented_region="LV",
        region_filter_basis="dataset_level_documented_left_ventricular_myocardium",
        comparison_limitation=(
            "NICM is intentionally excluded; retain only DoxCM and its "
            "within-dataset NF controls."
        ),
    ),
    DatasetSpec(
        dataset_folder="dataset11",
        dataset_id="dataset_11_HFpEF_SCP3342",
        input_h5ad="Data/converted/dataset_11_HFpEF_SCP3342_standardized.h5ad",
        count_h5_path="/X",
        condition_col="disease",
        condition_map={"HFpEF": "HFpEF", "NF": "NF"},
        condition_order=("HFpEF", "NF"),
        documented_region="S",
        region_filter_basis="cell_level_organ_label_interventricular_septum",
        region_policy="approved_HFpEF_septum_exception",
        count_transform="inverse_log1p_normalization_using_cell_total",
        count_total_col="n_counts",
    ),
)


EXCLUDED_DATASET_POLICY = (
    {
        "dataset_folder": "dataset02",
        "status": "excluded_from_LEC_pipeline",
        "reason": (
            "Completed screening retained QC-passed EC candidates but produced "
            "zero final LEC calls in both DCM and NF; excluded from downstream "
            "LEC analysis, not because Step 03 QC failed"
        ),
    },
    {
        "dataset_folder": "dataset06",
        "status": "excluded_from_LEC_pipeline",
        "reason": (
            "Completed screening produced final LEC calls only in NF (126 cells) "
            "and none in DCM, so it cannot support the intended paired "
            "within-dataset disease-versus-NF LEC comparison"
        ),
    },
    {
        "dataset_folder": "dataset12",
        "status": "excluded_from_disease_pipeline",
        "reason": "COVID cohort is no longer needed in the current analysis",
    },
    {
        "dataset_folder": "dataset03",
        "status": "excluded",
        "reason": (
            "ICM dataset row was not yellow on the workbook Datasets sheet; "
            "the disease-list highlight is not used as dataset-selection evidence"
        ),
    },
    {
        "dataset_folder": "dataset10",
        "status": "excluded",
        "reason": (
            "All seven sample/donor identifiers end in -RV; no LV or AP cells "
            "are present in the input h5ad"
        ),
    },
    {
        "dataset_folder": "dataset04",
        "status": "excluded",
        "reason": "AS is not an approved target condition and tissue is septum",
    },
    {
        "dataset_folder": "dataset05",
        "status": "excluded",
        "reason": "NF-only source is not paired with an approved target disease",
    },
    {
        "dataset_folder": "dataset07",
        "status": "excluded",
        "reason": "AF is not selected and tissue is left atrium",
    },
    {
        "dataset_folder": "dataset09",
        "status": "excluded",
        "reason": "Workbook instruction explicitly says do not use this dataset",
    },
    {
        "dataset_folder": "dataset13",
        "status": "excluded_from_disease_pipeline",
        "reason": (
            "Reserved for the aging pipeline and not treated as an independent "
            "disease cohort"
        ),
    },
)


DOUBLEt_SCORE_CANDIDATES = (
    "cellranger_doublet_scores",
    "doublet_score",
    "doublet_scores",
    "scrublet_score",
    "scrublet_score_z",
    "predicted_doublet_score",
)
COUNT_CANDIDATES = (
    "total_counts",
    "n_counts",
    "cellbender_ncount",
    "nCount_RNA",
)
GENE_COUNT_CANDIDATES = (
    "n_genes_by_counts",
    "n_genes",
    "cellbender_ngenes",
    "nFeature_RNA",
)
MITO_PERCENT_CANDIDATES = (
    "pct_counts_mt",
    "percent_mito",
    "cellranger_percent_mito",
    "percent.mito",
)
EXON_CANDIDATES = (
    "exon_prop",
    "exonic_fraction",
    "exon_fraction",
    "pct_exonic",
    "percent_exonic",
)
ENTROPY_CANDIDATES = ("entropy", "cell_entropy", "RNA_entropy")


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def natural_key(value: object) -> tuple:
    pieces = re.split(r"(\d+(?:\.\d+)?)", str(value))
    key = []
    for piece in pieces:
        if not piece:
            continue
        try:
            key.append((0, float(piece)))
        except ValueError:
            key.append((1, piece.casefold()))
    return tuple(key)


def clean_label(value: object) -> str:
    text = re.sub(r"\s+", "_", str(value).strip())
    text = text.replace("/", "_").replace("\\", "_").replace(":", "_")
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "NA"


def parse_csv_set(value: str) -> Optional[set[str]]:
    if not value.strip() or value.strip().casefold() == "all":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    return next((column for column in candidates if column in frame.columns), None)


def h5_sparse_shape(path: Path, group_path: str) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        if group_path not in handle:
            raise KeyError(f"Count matrix path is absent: {path}::{group_path}")
        node = handle[group_path]
        if not isinstance(node, h5py.Group):
            if len(node.shape) != 2:
                raise ValueError(f"Expected a 2D matrix at {path}::{group_path}")
            return int(node.shape[0]), int(node.shape[1])
        encoding = str(node.attrs.get("encoding-type", ""))
        if encoding not in {"csr_matrix", "csc_matrix"}:
            raise ValueError(
                f"Unsupported H5AD matrix encoding '{encoding}' at {path}::{group_path}"
            )
        shape = tuple(int(value) for value in node.attrs["shape"])
        return shape[0], shape[1]


def _read_csr_window(
    node: h5py.Group,
    row_start: int,
    row_stop: int,
    n_vars: int,
) -> sp.csr_matrix:
    pointers = np.asarray(node["indptr"][row_start : row_stop + 1], dtype=np.int64)
    data_start = int(pointers[0])
    data_stop = int(pointers[-1])
    data = np.asarray(node["data"][data_start:data_stop])
    indices = np.asarray(node["indices"][data_start:data_stop], dtype=np.int32)
    pointers = pointers - data_start
    return sp.csr_matrix(
        (data, indices, pointers), shape=(row_stop - row_start, n_vars)
    )


def read_selected_csr_rows(
    path: Path,
    group_path: str,
    positions: np.ndarray,
    max_gap_rows: int,
    max_window_rows: int,
) -> sp.csr_matrix:
    """Read selected rows without materializing the complete backed layer."""
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1 or len(positions) == 0:
        raise ValueError("At least one selected row position is required.")
    if np.any(np.diff(positions) <= 0):
        raise ValueError("Selected row positions must be strictly increasing and unique.")
    if max_gap_rows < 0 or max_window_rows < 1:
        raise ValueError("HDF5 row-window settings are invalid.")

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
            raise ValueError(
                f"Expected CSR row storage at {path}::{group_path}; found {encoding}."
            )
        n_rows, n_vars = (int(value) for value in node.attrs["shape"])
        if positions[0] < 0 or positions[-1] >= n_rows:
            raise IndexError("A selected row position is outside the source matrix.")

        blocks: list[sp.csr_matrix] = []
        selection_start = 0
        window_number = 0
        while selection_start < len(positions):
            selection_stop = selection_start + 1
            window_row_start = int(positions[selection_start])
            last_selected = window_row_start
            while selection_stop < len(positions):
                next_selected = int(positions[selection_stop])
                gap = next_selected - last_selected - 1
                proposed_span = next_selected - window_row_start + 1
                if gap > max_gap_rows or proposed_span > max_window_rows:
                    break
                last_selected = next_selected
                selection_stop += 1

            selected = positions[selection_start:selection_stop]
            window_row_stop = int(selected[-1]) + 1
            full_window = _read_csr_window(
                node, window_row_start, window_row_stop, n_vars
            )
            relative = selected - window_row_start
            blocks.append(full_window[relative, :].tocsr())
            selection_start = selection_stop
            window_number += 1
            if window_number == 1 or window_number % 25 == 0 or selection_stop == len(positions):
                print(
                    f"    count windows read: {window_number:,}; "
                    f"selected rows loaded: {selection_stop:,}/{len(positions):,}"
                )

    matrix = sp.vstack(blocks, format="csr")
    if matrix.shape[0] != len(positions):
        raise RuntimeError("Selected count row order/length was not preserved.")
    return matrix


def merged_conditions(obs: pd.DataFrame, spec: DatasetSpec) -> pd.Series:
    if spec.condition_col not in obs.columns:
        raise KeyError(
            f"{spec.dataset_folder}: condition column '{spec.condition_col}' is absent."
        )
    lookup = {str(key).strip().casefold(): value for key, value in spec.condition_map.items()}
    original = normalized_text(obs[spec.condition_col])
    return original.str.casefold().map(lookup).fillna("")


def standardized_regions(obs: pd.DataFrame, spec: DatasetSpec) -> pd.Series:
    if spec.region_col:
        if spec.region_col not in obs.columns:
            raise KeyError(
                f"{spec.dataset_folder}: region column '{spec.region_col}' is absent."
            )
        assert spec.region_map is not None
        lookup = {
            str(key).strip().casefold(): value for key, value in spec.region_map.items()
        }
        return normalized_text(obs[spec.region_col]).str.casefold().map(lookup).fillna("")
    if not spec.documented_region:
        raise ValueError(f"{spec.dataset_folder}: no region rule was configured.")
    return pd.Series(spec.documented_region, index=obs.index, dtype="string")


def cohort_mask(
    obs: pd.DataFrame,
    spec: DatasetSpec,
    condition: Optional[str] = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    condition_values = merged_conditions(obs, spec)
    region_values = standardized_regions(obs, spec)
    keep = condition_values.ne("") & region_values.ne("")
    if condition is not None:
        keep &= condition_values.eq(condition)
    return keep, condition_values, region_values


def validate_count_values(matrix: sp.csr_matrix, label: str) -> dict[str, object]:
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{label}: count matrix has an empty dimension.")
    data = matrix.data
    if data.size == 0:
        raise ValueError(f"{label}: count matrix has no nonzero entries.")
    if not np.isfinite(data).all():
        raise ValueError(f"{label}: count matrix contains non-finite values.")
    if float(data.min()) < 0:
        raise ValueError(f"{label}: count matrix contains negative values.")
    integer_like = bool(np.allclose(data, np.rint(data), rtol=0, atol=1e-8))
    if not integer_like:
        raise ValueError(f"{label}: selected count source is not integer-like.")
    return {
        "count_min_nonzero": float(data.min()),
        "count_max": float(data.max()),
        "count_n_nonzero": int(data.size),
        "count_integer_like": integer_like,
    }


def apply_configured_count_transform(
    matrix: sp.csr_matrix,
    obs: pd.DataFrame,
    spec: DatasetSpec,
    integer_tolerance: float,
) -> tuple[sp.csr_matrix, dict[str, object]]:
    """Recover integer counts when a source retained only log1p-normalized X."""
    if spec.count_transform == "none":
        return matrix, {"count_transform": "none"}
    if spec.count_transform != "inverse_log1p_normalization_using_cell_total":
        raise ValueError(
            f"{spec.dataset_folder}: unsupported count transform {spec.count_transform}"
        )
    if not spec.count_total_col or spec.count_total_col not in obs.columns:
        raise KeyError(
            f"{spec.dataset_folder}: count reconstruction requires obs column "
            f"'{spec.count_total_col}'."
        )
    totals = pd.to_numeric(obs[spec.count_total_col], errors="coerce").to_numpy(float)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError(
            f"{spec.dataset_folder}: invalid per-cell totals for count reconstruction."
        )
    transformed = matrix.astype(np.float64, copy=True).tocsr()
    transformed.data = np.expm1(transformed.data)
    transformed = sp.diags(
        totals / float(spec.count_normalization_target), format="csr"
    ).dot(transformed).tocsr()
    deviations = np.abs(transformed.data - np.rint(transformed.data))
    maximum_deviation = float(deviations.max()) if deviations.size else 0.0
    if maximum_deviation > integer_tolerance:
        raise ValueError(
            f"{spec.dataset_folder}: inverse-normalized values are not sufficiently "
            f"integer-like (maximum deviation {maximum_deviation:.6g}; "
            f"allowed tolerance {integer_tolerance:.6g})."
        )
    transformed.data = np.rint(transformed.data).astype(np.int32)
    transformed.eliminate_zeros()
    reconstructed_totals = np.asarray(transformed.sum(axis=1)).ravel()
    maximum_row_sum_difference = float(
        np.max(np.abs(reconstructed_totals - totals))
    )
    if maximum_row_sum_difference > 0:
        raise ValueError(
            f"{spec.dataset_folder}: reconstructed row totals do not exactly match "
            f"{spec.count_total_col}; maximum difference={maximum_row_sum_difference}."
        )
    return transformed, {
        "count_transform": spec.count_transform,
        "count_total_col": spec.count_total_col,
        "count_normalization_target": spec.count_normalization_target,
        "inverse_transform_integer_tolerance": integer_tolerance,
        "inverse_transform_maximum_integer_deviation": maximum_deviation,
        "inverse_transform_maximum_row_sum_difference": maximum_row_sum_difference,
    }


def inspect_dataset(
    spec: DatasetSpec,
    project_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    path = resolve_project_path(project_dir, spec.input_h5ad)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty input H5AD: {path}")

    source = ad.read_h5ad(path, backed="r")
    try:
        obs = source.obs.copy()
        shape = (int(source.n_obs), int(source.n_vars))
        if not source.obs_names.is_unique:
            raise ValueError(f"{spec.dataset_folder}: source obs_names are not unique.")
        if spec.donor_col not in obs.columns:
            raise KeyError(
                f"{spec.dataset_folder}: donor column '{spec.donor_col}' is absent."
            )
        count_shape = h5_sparse_shape(path, spec.count_h5_path)
        if count_shape != shape:
            raise ValueError(
                f"{spec.dataset_folder}: count matrix {count_shape} does not match "
                f"AnnData shape {shape}."
            )
        keep, merged, regions = cohort_mask(obs, spec)
        donors = normalized_text(obs[spec.donor_col])
        if donors.loc[keep].eq("").any():
            raise ValueError(f"{spec.dataset_folder}: selected cells have blank donor IDs.")

        rows: list[dict[str, object]] = []
        for condition in spec.condition_order:
            selected = keep & merged.eq(condition)
            rows.append(
                {
                    "dataset_folder": spec.dataset_folder,
                    "dataset_id": spec.dataset_id,
                    "status": "included",
                    "condition": condition,
                    "n_cells_source": shape[0],
                    "n_cells_after_condition_region_policy_before_QC": int(selected.sum()),
                    "n_donors_before_QC": int(donors.loc[selected].nunique()),
                    "regions_retained": ";".join(
                        sorted(regions.loc[selected].unique(), key=natural_key)
                    ),
                    "region_filter_basis": spec.region_filter_basis,
                    "region_policy": spec.region_policy,
                    "has_within_dataset_NF": spec.has_within_dataset_nf,
                    "comparison_limitation": spec.comparison_limitation,
                    "input_h5ad": str(path),
                    "count_h5_path": spec.count_h5_path,
                    "n_genes_source": shape[1],
                }
            )

        original = normalized_text(obs[spec.condition_col])
        mapping_frame = pd.DataFrame(
            {
                "condition_original": original,
                "condition_merged": merged,
                "region_standardized_analysis": regions,
                "retained_by_cohort_policy": keep,
            }
        )
        mapping_rows = (
            mapping_frame.groupby(
                [
                    "condition_original",
                    "condition_merged",
                    "region_standardized_analysis",
                    "retained_by_cohort_policy",
                ],
                dropna=False,
                observed=True,
            )
            .size()
            .rename("n_cells")
            .reset_index()
        )
        mapping_rows.insert(0, "dataset_folder", spec.dataset_folder)
        return rows, mapping_rows.to_dict("records")
    finally:
        try:
            source.file.close()
        except Exception:
            pass
        del source
        gc.collect()


def load_condition_counts(
    spec: DatasetSpec,
    condition: str,
    project_dir: Path,
    args: argparse.Namespace,
) -> tuple[ad.AnnData, dict[str, object]]:
    path = resolve_project_path(project_dir, spec.input_h5ad)
    source = ad.read_h5ad(path, backed="r")
    try:
        obs_all = source.obs
        keep, merged_all, regions_all = cohort_mask(obs_all, spec, condition)
        positions = np.flatnonzero(keep.to_numpy()).astype(np.int64)
        if len(positions) < args.min_cells_per_condition:
            raise ValueError(
                f"{spec.dataset_folder} {condition}: only {len(positions):,} cells "
                "remain after condition/region policy before QC."
            )
        obs = obs_all.iloc[positions].copy()
        var = source.var.copy()
        source_cells = source.obs_names[positions].astype(str)
        merged = merged_all.iloc[positions].astype(str).to_numpy()
        regions = regions_all.iloc[positions].astype(str).to_numpy()
    finally:
        try:
            source.file.close()
        except Exception:
            pass
        del source

    print(
        f"  Reading raw counts for {spec.dataset_folder}/{condition}: "
        f"{len(positions):,} selected rows"
    )
    matrix = read_selected_csr_rows(
        path,
        spec.count_h5_path,
        positions,
        max_gap_rows=args.h5_max_gap_rows,
        max_window_rows=args.h5_max_window_rows,
    )
    matrix, transform_report = apply_configured_count_transform(
        matrix,
        obs,
        spec,
        integer_tolerance=args.inverse_count_integer_tolerance,
    )
    count_report = validate_count_values(
        matrix, f"{spec.dataset_folder}/{condition}"
    )

    original_var_names = pd.Index(var.index.astype(str))
    var = var.copy()
    var["source_var_name"] = original_var_names.to_numpy()
    obs = obs.copy()
    obs["source_cell_id"] = source_cells
    obs["source_row_index"] = positions
    obs["dataset_id"] = spec.dataset_id
    obs["dataset_folder"] = spec.dataset_folder
    obs["condition_original"] = normalized_text(obs[spec.condition_col]).to_numpy()
    obs["condition_merged"] = merged
    obs["disease"] = merged
    obs["region_standardized_analysis"] = regions
    obs["region_filter_basis"] = spec.region_filter_basis
    obs["region_policy"] = spec.region_policy
    obs["has_within_dataset_NF"] = spec.has_within_dataset_nf
    obs["comparison_limitation"] = spec.comparison_limitation

    donor = normalized_text(obs[spec.donor_col])
    if donor.eq("").any():
        raise ValueError(f"{spec.dataset_folder}/{condition}: blank donor IDs found.")
    obs["donor_id_source"] = donor.to_numpy()
    obs["donor_key"] = spec.dataset_folder + "::" + donor.to_numpy()
    sample = (
        normalized_text(obs[spec.sample_col])
        if spec.sample_col in obs.columns
        else donor
    )
    obs["sample_id_source"] = sample.to_numpy()
    obs["sample_key"] = spec.dataset_folder + "::" + sample.to_numpy()

    output_cells = pd.Index(
        [
            f"{spec.dataset_folder}|{condition}|{str(cell).replace('|', '_')}"
            for cell in source_cells
        ]
    )
    if not output_cells.is_unique:
        raise ValueError(f"{spec.dataset_folder}/{condition}: output cell IDs are not unique.")
    obs.index = output_cells
    var.index = original_var_names

    result = ad.AnnData(X=matrix, obs=obs, var=var)
    result.var_names_make_unique()
    result.var["gene_name"] = result.var_names.astype(str)
    result.uns["03_count_source"] = {
        "source_h5ad": str(path),
        "h5_group": spec.count_h5_path,
        **transform_report,
        **count_report,
    }
    return result, {
        "source_h5ad": str(path),
        "count_h5_path": spec.count_h5_path,
        "n_cells_after_condition_region_policy_before_QC": int(result.n_obs),
        "n_donors_before_QC": int(result.obs["donor_key"].nunique()),
        "n_genes_source": int(result.n_vars),
        **transform_report,
        **count_report,
    }


def compute_qc_metrics(adata: ad.AnnData, args: argparse.Namespace) -> dict[str, object]:
    matrix = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    genes = pd.Index(adata.var_names.astype(str))
    mt_mask = genes.str.upper().str.startswith("MT-")
    n_mt_genes = int(mt_mask.sum())
    if n_mt_genes >= 5:
        total_counts = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
        n_genes = np.asarray(matrix.getnnz(axis=1)).ravel().astype(float)
        mt_counts = np.asarray(matrix[:, mt_mask].sum(axis=1)).ravel().astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_mt = np.divide(
                100.0 * mt_counts,
                total_counts,
                out=np.full_like(total_counts, np.nan),
                where=total_counts > 0,
            )
        qc_metric_source = "recalculated_from_selected_integer_count_matrix"
        source_count_col = ""
        source_gene_col = ""
        source_mito_col = ""
    else:
        source_count_col = first_existing(adata.obs, COUNT_CANDIDATES)
        source_gene_col = first_existing(adata.obs, GENE_COUNT_CANDIDATES)
        source_mito_col = first_existing(adata.obs, MITO_PERCENT_CANDIDATES)
        if not source_count_col or not source_gene_col or not source_mito_col:
            raise ValueError(
                f"Only {n_mt_genes} mitochondrial genes were retained and complete "
                "source QC columns are unavailable."
            )
        total_counts = pd.to_numeric(
            adata.obs[source_count_col], errors="coerce"
        ).to_numpy(float)
        n_genes = pd.to_numeric(
            adata.obs[source_gene_col], errors="coerce"
        ).to_numpy(float)
        pct_mt = pd.to_numeric(
            adata.obs[source_mito_col], errors="coerce"
        ).to_numpy(float)
        finite_mito = pct_mt[np.isfinite(pct_mt)]
        if finite_mito.size and float(np.nanmax(finite_mito)) <= 1.5:
            pct_mt = pct_mt * 100.0
        if (
            not np.isfinite(total_counts).all()
            or not np.isfinite(n_genes).all()
            or not np.isfinite(pct_mt).all()
        ):
            raise ValueError("Source QC columns contain missing/non-finite values.")
        qc_metric_source = "source_obs_columns_because_mito_genes_absent_from_matrix"
    adata.obs["qc_total_counts"] = total_counts
    adata.obs["qc_n_genes"] = n_genes
    adata.obs["qc_pct_mt"] = pct_mt

    doublet_col = first_existing(adata.obs, DOUBLEt_SCORE_CANDIDATES)
    exon_col = first_existing(adata.obs, EXON_CANDIDATES)
    entropy_col = first_existing(adata.obs, ENTROPY_CANDIDATES)
    adata.obs["qc_doublet_score"] = (
        pd.to_numeric(adata.obs[doublet_col], errors="coerce").to_numpy()
        if doublet_col
        else np.nan
    )
    adata.obs["qc_exon_fraction"] = (
        pd.to_numeric(adata.obs[exon_col], errors="coerce").to_numpy()
        if exon_col
        else np.nan
    )
    if entropy_col:
        adata.obs["qc_entropy"] = pd.to_numeric(
            adata.obs[entropy_col], errors="coerce"
        ).to_numpy()
    elif args.compute_entropy:
        row_sums = total_counts
        entropy = np.zeros(adata.n_obs, dtype=float)
        for row_index in range(adata.n_obs):
            start, stop = matrix.indptr[row_index : row_index + 2]
            values = matrix.data[start:stop]
            if row_sums[row_index] > 0 and len(values):
                probabilities = values / row_sums[row_index]
                probabilities = probabilities[probabilities > 0]
                entropy[row_index] = -np.sum(probabilities * np.log(probabilities))
        adata.obs["qc_entropy"] = entropy
    else:
        adata.obs["qc_entropy"] = np.nan

    return {
        "n_mitochondrial_genes": n_mt_genes,
        "qc_metric_source": qc_metric_source,
        "source_count_column": source_count_col,
        "source_gene_count_column": source_gene_col,
        "source_mito_percent_column": source_mito_col,
        "doublet_score_column": doublet_col or "",
        "exon_fraction_column": exon_col or "",
        "entropy_column": entropy_col or "",
        "entropy_computed": bool(not entropy_col and args.compute_entropy),
    }


def hard_qc_flags(
    adata: ad.AnnData, args: argparse.Namespace
) -> tuple[pd.Series, pd.DataFrame]:
    obs = adata.obs
    genes = pd.to_numeric(obs["qc_n_genes"], errors="coerce")
    counts = pd.to_numeric(obs["qc_total_counts"], errors="coerce")
    mito = pd.to_numeric(obs["qc_pct_mt"], errors="coerce")
    remove_low_genes = genes.lt(args.min_genes) | genes.isna()
    remove_low_counts = counts.lt(args.min_counts) | counts.isna()
    remove_high_mito = mito.gt(args.max_mito_percent) | mito.isna()
    keep = ~(remove_low_genes | remove_low_counts | remove_high_mito)
    flags = pd.DataFrame(
        {
            "remove_hard_low_genes": remove_low_genes,
            "remove_hard_low_counts": remove_low_counts,
            "remove_hard_high_mito": remove_high_mito,
            "keep_after_hard_QC": keep,
        },
        index=adata.obs_names,
    )
    return keep, flags


def mean_iqr_bounds(values: pd.Series, multiplier: float) -> tuple[float, ...]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 4:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    mean = float(numeric.mean())
    q1 = float(np.percentile(numeric, 25))
    q3 = float(np.percentile(numeric, 75))
    iqr = q3 - q1
    return mean, q1, q3, mean - multiplier * iqr, mean + multiplier * iqr


def mean_iqr_qc(
    adata: ad.AnnData, args: argparse.Namespace
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    metrics = {
        "qc_n_genes": "two_sided",
        "qc_total_counts": "two_sided",
        "qc_pct_mt": "high",
        "qc_doublet_score": "high",
        "qc_exon_fraction": "high",
        "qc_entropy": "low",
    }
    if args.two_sided_all_qc_outliers:
        metrics = {key: "two_sided" for key in metrics}
    keep = pd.Series(True, index=adata.obs_names)
    flags = pd.DataFrame(index=adata.obs_names)
    thresholds = []
    for metric, direction in metrics.items():
        values = pd.to_numeric(adata.obs[metric], errors="coerce")
        n_non_na = int(values.notna().sum())
        if n_non_na < 4 or (
            metric in {"qc_doublet_score", "qc_exon_fraction", "qc_entropy"}
            and n_non_na < max(4, int(0.1 * len(values)))
        ):
            thresholds.append(
                {
                    "metric": metric,
                    "status": "skipped_insufficient_nonmissing_values",
                    "direction": direction,
                    "n_nonNA": n_non_na,
                }
            )
            continue
        mean, q1, q3, lower, upper = mean_iqr_bounds(values, args.iqr_multiplier)
        remove_low = values.lt(lower) if direction in {"low", "two_sided"} else pd.Series(False, index=values.index)
        remove_high = values.gt(upper) if direction in {"high", "two_sided"} else pd.Series(False, index=values.index)
        remove = remove_low.fillna(False) | remove_high.fillna(False)
        keep &= ~remove
        flags[f"remove_meanIQR_{metric}_low"] = remove_low.fillna(False)
        flags[f"remove_meanIQR_{metric}_high"] = remove_high.fillna(False)
        thresholds.append(
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
                "lower_threshold": lower,
                "upper_threshold": upper,
                "n_removed_low": int(remove_low.fillna(False).sum()),
                "n_removed_high": int(remove_high.fillna(False).sum()),
            }
        )
    flags["keep_after_meanIQR_QC"] = keep
    return keep, flags, pd.DataFrame(thresholds)


def make_obs_h5ad_safe(adata: ad.AnnData) -> None:
    for column in list(adata.obs.columns):
        values = adata.obs[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            adata.obs[column] = values.astype("string").fillna("").astype(str)
        elif pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            adata.obs[column] = values.map(
                lambda value: "" if value is None or (np.isscalar(value) and pd.isna(value)) else str(value)
            ).astype(str)


def preprocess_and_cluster(
    adata: ad.AnnData,
    args: argparse.Namespace,
) -> tuple[ad.AnnData, dict[str, object]]:
    batch_key = "donor_key" if adata.obs["donor_key"].nunique() > 1 else None
    normalized = False
    hvg_flavor = "seurat_v3"
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(args.n_top_hvg, adata.n_vars),
            flavor="seurat_v3",
            batch_key=batch_key,
        )
    except Exception as error:
        print(f"  seurat_v3 HVG failed; using seurat after log normalization: {error}")
        hvg_flavor = "seurat"
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        normalized = True
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(args.n_top_hvg, adata.n_vars),
            flavor="seurat",
            batch_key=batch_key,
        )

    n_hvg = int(adata.var["highly_variable"].sum())
    if n_hvg < args.min_hvg_for_clustering:
        if args.allow_all_genes_if_few_hvg:
            adata.var["highly_variable"] = True
            n_hvg = int(adata.n_vars)
            hvg_flavor += "_fallback_all_genes"
        else:
            raise ValueError(f"Only {n_hvg} HVGs were found.")

    if not normalized:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    adata.raw = adata

    n_components = int(min(args.n_pcs, adata.n_obs - 1, n_hvg - 1))
    if n_components < 2:
        raise ValueError(f"Too few cells/HVGs for PCA: {n_components}")
    sc.tl.pca(
        adata,
        n_comps=n_components,
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
            print(f"  Harmony failed; using X_pca: {error}")

    n_neighbors = min(args.n_neighbors, max(2, adata.n_obs - 1))
    n_pcs_neighbors = min(args.n_pcs_neighbors, n_components)
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
            resolution=args.leiden_resolution,
            key_added="leiden",
            flavor="igraph",
            n_iterations=2,
            directed=False,
            random_state=args.random_seed,
        )
    except Exception:
        sc.tl.leiden(
            adata,
            resolution=args.leiden_resolution,
            key_added="leiden",
            random_state=args.random_seed,
        )

    params = {
        "hvg_flavor_used": hvg_flavor,
        "n_hvg": n_hvg,
        "n_pcs": n_components,
        "n_pcs_neighbors": n_pcs_neighbors,
        "n_neighbors": n_neighbors,
        "neighbor_metric": args.neighbor_metric,
        "representation": representation,
        "harmony_requested": args.use_harmony,
        "harmony_used": harmony_used,
        "harmony_key": batch_key or "",
        "harmony_error": harmony_error,
        "umap_min_dist": args.umap_min_dist,
        "leiden_resolution": args.leiden_resolution,
        "random_seed": args.random_seed,
        "all_genes_retained_in_X": True,
        "figures_created_by_step03": False,
    }
    adata.uns["03_disease_cluster_params"] = params
    return adata, params


def write_condition_outputs(
    counts: ad.AnnData,
    spec: DatasetSpec,
    condition: str,
    output_dir: Path,
    load_report: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{spec.dataset_folder}_{clean_label(condition)}"
    primary_h5ad = output_dir / f"{prefix}_clustered.h5ad"
    run_summary_path = output_dir / f"{prefix}_run_summary.csv"
    if primary_h5ad.exists() and primary_h5ad.stat().st_size > 1000 and not args.force:
        if run_summary_path.exists():
            row = pd.read_csv(run_summary_path).iloc[0].to_dict()
        else:
            row = {
                "dataset_folder": spec.dataset_folder,
                "condition": condition,
                "output_h5ad": str(primary_h5ad),
            }
        row["status"] = "reused_existing_output"
        print(f"  Reusing existing output: {primary_h5ad}")
        return row

    qc_source_report = compute_qc_metrics(counts, args)
    qc_columns = [
        "qc_n_genes",
        "qc_total_counts",
        "qc_pct_mt",
        "qc_doublet_score",
        "qc_exon_fraction",
        "qc_entropy",
    ]
    counts.obs[qc_columns].describe().to_csv(
        output_dir / f"{prefix}_QC_metric_summary_before_filter.csv"
    )
    hard_keep, hard_flags = hard_qc_flags(counts, args)
    hard = counts[hard_keep.to_numpy()].copy()
    if hard.n_obs < args.min_cells_per_condition:
        raise ValueError(f"Only {hard.n_obs:,} cells remain after hard QC.")

    if args.use_mean_iqr_qc:
        iqr_keep, iqr_flags, thresholds = mean_iqr_qc(hard, args)
    else:
        iqr_keep = pd.Series(True, index=hard.obs_names)
        iqr_flags = pd.DataFrame(
            {"keep_after_meanIQR_QC": True}, index=hard.obs_names
        )
        thresholds = pd.DataFrame(
            [{"status": "meanIQR_QC_disabled_to_match_actual_aging_run"}]
        )
    thresholds.to_csv(
        output_dir / f"{prefix}_meanIQR_QC_thresholds.csv", index=False
    )
    filtered = hard[iqr_keep.to_numpy()].copy()
    if filtered.n_obs < args.min_cells_per_condition:
        raise ValueError(f"Only {filtered.n_obs:,} cells remain after all QC.")

    removed_hard = counts.obs.loc[~hard_keep].copy()
    if len(removed_hard):
        removed_hard = removed_hard[
            ["source_cell_id", "source_row_index", "donor_key", "condition_merged"]
            + qc_columns
        ].join(hard_flags.loc[~hard_keep])
        removed_hard["remove_stage"] = "hard_QC"
    removed_iqr = hard.obs.loc[~iqr_keep].copy()
    if len(removed_iqr):
        removed_iqr = removed_iqr[
            ["source_cell_id", "source_row_index", "donor_key", "condition_merged"]
            + qc_columns
        ].join(iqr_flags.loc[~iqr_keep])
        removed_iqr["remove_stage"] = "meanIQR_QC"
    removed = pd.concat([removed_hard, removed_iqr], axis=0, sort=False)
    removed.to_csv(output_dir / f"{prefix}_cells_removed_by_QC.csv", index=False)

    stage_counts = pd.DataFrame(
        [
            {
                "stage": "after_condition_region_policy_before_QC",
                "n_cells": int(counts.n_obs),
                "n_donors": int(counts.obs["donor_key"].nunique()),
            },
            {
                "stage": "after_hard_QC",
                "n_cells": int(hard.n_obs),
                "n_donors": int(hard.obs["donor_key"].nunique()),
            },
            {
                "stage": "after_all_QC",
                "n_cells": int(filtered.n_obs),
                "n_donors": int(filtered.obs["donor_key"].nunique()),
            },
        ]
    )
    stage_counts.to_csv(output_dir / f"{prefix}_QC_stage_counts.csv", index=False)

    del counts, hard
    gc.collect()
    clustered, cluster_params = preprocess_and_cluster(filtered, args)
    clustered.obs["leiden"] = clustered.obs["leiden"].astype(str)
    clustered.obs["step03_final_QC_pass"] = True
    clustered.uns["03_script_version"] = SCRIPT_VERSION
    clustered.uns["03_dataset_policy_json"] = json.dumps(
        asdict(spec), sort_keys=True
    )

    cluster_counts = (
        clustered.obs["leiden"]
        .value_counts()
        .sort_index(key=lambda index: index.map(natural_key))
        .rename_axis("leiden")
        .reset_index(name="n_cells")
    )
    cluster_counts.to_csv(output_dir / f"{prefix}_cluster_counts.csv", index=False)
    pd.crosstab(clustered.obs["leiden"], clustered.obs["donor_key"]).to_csv(
        output_dir / f"{prefix}_clusters_vs_donor.csv"
    )
    pd.crosstab(
        clustered.obs["leiden"], clustered.obs["region_standardized_analysis"]
    ).to_csv(output_dir / f"{prefix}_clusters_vs_region.csv")
    clustered.var.reset_index(names="var_name").to_csv(
        output_dir / f"{prefix}_gene_and_HVG_metadata.csv", index=False
    )

    if args.keep_counts_layer:
        raise RuntimeError(
            "Internal ordering error: counts layer must be created before normalization."
        )
    make_obs_h5ad_safe(clustered)
    temporary_h5ad = output_dir / f".{prefix}_clustered.partial.{os.getpid()}.h5ad"
    clustered.write_h5ad(temporary_h5ad, compression=args.h5ad_compression)
    os.replace(temporary_h5ad, primary_h5ad)

    summary = {
        "script_version": SCRIPT_VERSION,
        "dataset_folder": spec.dataset_folder,
        "dataset_id": spec.dataset_id,
        "condition": condition,
        "status": "completed",
        "region_policy": spec.region_policy,
        "region_filter_basis": spec.region_filter_basis,
        "regions_retained": ";".join(
            sorted(clustered.obs["region_standardized_analysis"].unique(), key=natural_key)
        ),
        "has_within_dataset_NF": spec.has_within_dataset_nf,
        "comparison_limitation": spec.comparison_limitation,
        "n_cells_before_QC": load_report["n_cells_after_condition_region_policy_before_QC"],
        "n_cells_after_hard_QC": int(stage_counts.iloc[1]["n_cells"]),
        "n_cells_after_all_QC": int(clustered.n_obs),
        "n_removed_total": int(load_report["n_cells_after_condition_region_policy_before_QC"] - clustered.n_obs),
        "n_donors_before_QC": load_report["n_donors_before_QC"],
        "n_donors_after_QC": int(clustered.obs["donor_key"].nunique()),
        "n_genes": int(clustered.n_vars),
        "n_clusters": int(clustered.obs["leiden"].nunique()),
        "output_h5ad": str(primary_h5ad),
        "count_h5_path": spec.count_h5_path,
        "hvg_flavor_used": cluster_params["hvg_flavor_used"],
        "n_hvg": cluster_params["n_hvg"],
        "harmony_used": cluster_params["harmony_used"],
        **qc_source_report,
    }
    pd.DataFrame([summary]).to_csv(run_summary_path, index=False)
    (output_dir / f"{prefix}_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    del clustered, filtered
    gc.collect()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter, QC, and independently cluster the approved disease "
            "dataset-condition cohorts."
        )
    )
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dataset_ids",
        default="all",
        help="Comma-separated dataset folders (e.g. dataset01,dataset11) or all.",
    )
    parser.add_argument(
        "--condition_labels",
        default="",
        help="Optional merged-condition subset such as DoxCM or DCM,NF.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate inputs and print cohort counts without creating outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and replace an existing condition output.",
    )

    parser.add_argument("--min_genes", type=float, default=150)
    parser.add_argument("--min_counts", type=float, default=150)
    parser.add_argument("--max_mito_percent", type=float, default=5.0)
    parser.add_argument("--min_cells_per_condition", type=int, default=200)
    parser.add_argument("--iqr_multiplier", type=float, default=1.5)
    parser.add_argument("--use_mean_iqr_qc", action="store_true", default=False)
    parser.add_argument("--two_sided_all_qc_outliers", action="store_true")
    parser.add_argument("--compute_entropy", action="store_true")

    parser.add_argument("--n_top_hvg", type=int, default=2000)
    parser.add_argument("--min_hvg_for_clustering", type=int, default=20)
    parser.add_argument(
        "--allow_all_genes_if_few_hvg",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--n_pcs", type=int, default=100)
    parser.add_argument("--n_pcs_neighbors", type=int, default=100)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--neighbor_metric", default="cosine")
    parser.add_argument("--umap_min_dist", type=float, default=0.2)
    parser.add_argument("--leiden_resolution", type=float, default=0.6)
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--no_harmony", action="store_false", dest="use_harmony")
    parser.set_defaults(use_harmony=True)

    parser.add_argument(
        "--inverse_count_integer_tolerance",
        type=float,
        default=DEFAULT_INVERSE_COUNT_INTEGER_TOLERANCE,
        help=(
            "Maximum distance from the nearest integer allowed when recovering "
            "counts from a log1p-normalized matrix. The exact per-cell total-count "
            "validation remains mandatory."
        ),
    )
    parser.add_argument("--h5_max_gap_rows", type=int, default=32)
    parser.add_argument("--h5_max_window_rows", type=int, default=20000)
    parser.add_argument(
        "--h5ad_compression",
        choices=["gzip", "lzf"],
        default="lzf",
        help="LZF is faster and reduces temporary disk pressure for large outputs.",
    )
    parser.add_argument(
        "--keep_counts_layer",
        action="store_true",
        help=(
            "Reserved for a future high-disk mode. Raw counts are recoverable "
            "from source_cell_id/source_row_index and are not duplicated by default."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.min_genes < 0 or args.min_counts < 0:
        raise ValueError("Minimum gene/count thresholds cannot be negative.")
    if not 0 <= args.max_mito_percent <= 100:
        raise ValueError("max_mito_percent must be between 0 and 100.")
    if args.min_cells_per_condition < 20:
        raise ValueError("min_cells_per_condition must be at least 20.")
    if args.iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be positive.")
    if args.n_top_hvg < 20 or args.n_pcs < 2 or args.n_neighbors < 2:
        raise ValueError("Clustering parameters are too small.")
    if not 0 < args.inverse_count_integer_tolerance < 0.5:
        raise ValueError(
            "inverse_count_integer_tolerance must be greater than 0 and less "
            "than 0.5 so nearest-integer rounding remains unambiguous."
        )
    if args.keep_counts_layer:
        raise ValueError(
            "--keep_counts_layer is intentionally disabled for this memory/disk-"
            "constrained workflow. Counts remain recoverable through provenance."
        )


def main() -> int:
    args = parse_args()
    validate_args(args)
    project_dir = find_project_dir()
    selected_datasets = parse_csv_set(args.dataset_ids)
    selected_conditions = parse_csv_set(args.condition_labels)

    specs = [
        spec
        for spec in DATASET_SPECS
        if selected_datasets is None or spec.dataset_folder in selected_datasets
    ]
    if selected_datasets is not None:
        missing = selected_datasets.difference(spec.dataset_folder for spec in specs)
        if missing:
            raise ValueError(f"Unknown/unapproved dataset folders: {sorted(missing)}")
    if not specs:
        raise ValueError("No approved datasets were selected.")

    print("=" * 88)
    print("Selected disease Step 03: independent dataset-condition QC/clustering")
    print("Project:", project_dir)
    print(
        "Exclusions: dataset02/dataset06 after LEC-usability screening; "
        "dataset10 RV-only; dataset12 COVID no longer needed"
    )
    print("Mean-IQR QC enabled:", args.use_mean_iqr_qc)
    print("=" * 88)

    audit_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    for spec in specs:
        rows, mappings = inspect_dataset(spec, project_dir)
        if selected_conditions is not None:
            rows = [row for row in rows if row["condition"] in selected_conditions]
        audit_rows.extend(rows)
        mapping_rows.extend(mappings)

    audit = pd.DataFrame(audit_rows)
    if audit.empty:
        raise ValueError("No dataset-condition combinations matched the requested filters.")
    print("\nRead-only cohort validation (before QC):")
    print(
        audit[
            [
                "dataset_folder",
                "condition",
                "n_cells_after_condition_region_policy_before_QC",
                "n_donors_before_QC",
                "regions_retained",
                "region_policy",
                "has_within_dataset_NF",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nTotal selected cells before QC: "
        f"{audit['n_cells_after_condition_region_policy_before_QC'].sum():,}"
    )
    print(
        "Sum of dataset-namespaced donor counts by condition: "
        f"{audit['n_donors_before_QC'].sum():,}"
    )
    if args.validate_only:
        print("\nValidation complete; no result directory or analysis output was created.")
        return 0

    output_root = resolve_project_path(project_dir, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_root / "03_dataset_condition_cohort_before_QC.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(
        output_root / "03_condition_and_region_mapping_audit.csv", index=False
    )
    policy_rows = []
    for spec in DATASET_SPECS:
        policy_rows.append(
            {
                "dataset_folder": spec.dataset_folder,
                "status": "included",
                "conditions": ";".join(spec.condition_order),
                "region_policy": spec.region_policy,
                "region_filter_basis": spec.region_filter_basis,
                "has_within_dataset_NF": spec.has_within_dataset_nf,
                "reason_or_limitation": spec.comparison_limitation,
            }
        )
    for excluded in EXCLUDED_DATASET_POLICY:
        policy_rows.append(
            {
                "dataset_folder": excluded["dataset_folder"],
                "status": excluded["status"],
                "conditions": "",
                "region_policy": "",
                "region_filter_basis": "",
                "has_within_dataset_NF": "",
                "reason_or_limitation": excluded.get("reason", ""),
            }
        )
    pd.DataFrame(policy_rows).to_csv(
        output_root / "03_dataset_inclusion_exclusion_policy.csv", index=False
    )

    config = {
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "project_dir": str(project_dir),
        "output_root": str(output_root),
        "implementation": "standalone_memory_conscious_CSR_row_reader",
        "covid_policy": "Exclude dataset10 RV-only and dataset12 COVID",
        "all_genes_retained": True,
        "figures_created": False,
        "arguments": vars(args),
        "dataset_specs": [asdict(spec) for spec in specs],
    }
    (output_root / "03_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    summaries: list[dict[str, object]] = []
    for spec in specs:
        requested_conditions = [
            condition
            for condition in spec.condition_order
            if selected_conditions is None or condition in selected_conditions
        ]
        for condition in requested_conditions:
            print("\n" + "=" * 88)
            print(f"Running {spec.dataset_folder}/{condition}")
            print("=" * 88)
            output_dir = output_root / spec.dataset_folder / clean_label(condition)
            error_path = (
                output_dir
                / f"{spec.dataset_folder}_{clean_label(condition)}_ERROR.txt"
            )
            try:
                primary = output_dir / f"{spec.dataset_folder}_{clean_label(condition)}_clustered.h5ad"
                if primary.exists() and primary.stat().st_size > 1000 and not args.force:
                    summary = write_condition_outputs(
                        ad.AnnData(), spec, condition, output_dir, {}, args
                    )
                else:
                    counts, load_report = load_condition_counts(
                        spec, condition, project_dir, args
                    )
                    summary = write_condition_outputs(
                        counts, spec, condition, output_dir, load_report, args
                    )
                if error_path.exists():
                    error_path.unlink()
                summaries.append(summary)
            except Exception as error:
                traceback.print_exc()
                output_dir.mkdir(parents=True, exist_ok=True)
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                summaries.append(
                    {
                        "dataset_folder": spec.dataset_folder,
                        "dataset_id": spec.dataset_id,
                        "condition": condition,
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            finally:
                gc.collect()
            pd.DataFrame(summaries).to_csv(
                output_root / "03_master_run_summary.csv", index=False
            )

    master = pd.DataFrame(summaries)
    n_errors = int(master["status"].eq("error").sum()) if "status" in master else 0
    print("\nStep 03 finished.")
    print("Master summary:", output_root / "03_master_run_summary.csv")
    if n_errors:
        print(f"Conditions with errors: {n_errors}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
