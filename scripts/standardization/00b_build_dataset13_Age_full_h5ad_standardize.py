"""
00b_build_dataset13_Age_full_h5ad_standardize.py

Build standardized h5ad from the full dataset13 RDS export made by:
  00a_export_dataset13_Age_full_RDS_to_mtx.R

Output:
  Data/converted/dataset_13_Age_full_30plus_age10yr_standardized.h5ad
  metadata/datasets_age10_30plus_full_temp.csv
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
import anndata as ad

DATASET_ID = "dataset_13_Age"
EXPORT_REL = Path("Data/converted/dataset_13_Age_5000feature_rds_export")
OUTPUT_H5AD_REL = Path("Data/converted/dataset_13_Age_full_30plus_age10yr_standardized.h5ad")
TEMP_METADATA_REL = Path("metadata/datasets_age10_30plus_full_temp.csv")


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[2]


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    exact = {str(c): c for c in df.columns}
    for c in candidates:
        if c in exact:
            return exact[c]
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def clean_str(s: pd.Series, default: str = "unknown") -> pd.Series:
    return s.astype(str).replace({"nan": default, "NaN": default, "<NA>": default, "None": default, "": default})


def extract_age_start(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def make_age_bin_10yr(age_start: pd.Series) -> pd.Series:
    bin_start = (age_start // 10 * 10).astype("Int64")
    out = bin_start.astype(str) + "-" + (bin_start + 9).astype(str)
    out = out.astype("string")
    out[(age_start < 30) | age_start.isna()] = pd.NA
    return out


def standardize_disease(x: str) -> str:
    raw = str(x).strip()
    compact = re.sub(r"[^a-z0-9]+", "", raw.lower().replace("_", " ").replace("-", " "))
    if compact in {"", "nan", "na", "none", "unknown"}:
        return "unknown"
    if compact in {"nf", "normal", "control", "healthy", "donor", "nonfailing"}:
        return "NF"
    if compact in {"heartfailurehf", "heartfailure", "hf"}:
        return "HF"
    return raw


def standardize_obs(metadata: pd.DataFrame, age_col_for_bins: str | None) -> pd.DataFrame:
    if "cell" not in metadata.columns:
        raise ValueError("metadata.csv must contain a cell column.")
    metadata = metadata.copy()
    metadata["cell"] = metadata["cell"].astype(str)
    metadata = metadata.drop_duplicates("cell", keep="first").set_index("cell")
    metadata["dataset_id"] = DATASET_ID
    metadata["source_id"] = metadata.get("source_id", "heart_sn_rna_list_rds")
    metadata["modality"] = metadata.get("modality", "single-nucleus")
    metadata["data_type"] = metadata.get("data_type", "snRNA-seq")

    disease_source = find_col(metadata, ["disease_original", "disease", "heart_failure_status", "Primary_diagnosis", "diagnosis", "Diagnosis", "group", "Group"])
    metadata["disease_original"] = metadata[disease_source].astype(str) if disease_source else "unknown"
    metadata["disease"] = metadata["disease_original"].map(standardize_disease)

    donor_col = find_col(metadata, ["donor_id", "donor", "Donor", "patient", "patient_id", "subject", "sample", "Sample", "biosample_id"])
    sample_col = find_col(metadata, ["biosample_id", "sample_id", "sample", "Sample", "orig.ident", "orig_ident", "library_id"])
    metadata["donor_id"] = metadata[donor_col].astype(str) if donor_col else "unknown_donor"
    metadata["biosample_id"] = metadata[sample_col].astype(str) if sample_col else metadata["donor_id"].astype(str)

    sex_col = find_col(metadata, ["sex", "Sex", "gender", "Gender"])
    age_col = find_col(metadata, ["age", "Age", "Age_range", "Age_range_normalized", "age_group"])
    metadata["sex"] = metadata[sex_col].astype(str) if sex_col else "unknown"
    metadata["age"] = metadata[age_col].astype(str) if age_col else "unknown"
    if "lvef" not in metadata.columns:
        metadata["lvef"] = "unknown"

    celltype_col = find_col(metadata, ["cell_type_leiden0.6", "cell_type", "CellType", "celltype", "broad_celltype_ref", "MainCellType", "annotation", "Annotation"])
    cluster_col = find_col(metadata, ["SubCluster", "subcluster", "Cluster", "cluster", "seurat_clusters"])
    metadata["cell_type_leiden0.6"] = metadata[celltype_col].astype(str) if celltype_col else "unknown"
    metadata["SubCluster"] = metadata[cluster_col].astype(str) if cluster_col else metadata["cell_type_leiden0.6"].astype(str)

    count_col = find_col(metadata, ["n_counts", "nCount_RNA", "total_counts", "nCount"])
    gene_col = find_col(metadata, ["n_genes", "nFeature_RNA", "n_genes_by_counts", "nFeature"])
    mito_col = find_col(metadata, ["percent_mito", "percent_mt", "percent.mito", "pct_counts_mt"])
    if count_col is not None:
        metadata["n_counts"] = pd.to_numeric(metadata[count_col], errors="coerce")
    if gene_col is not None:
        metadata["n_genes"] = pd.to_numeric(metadata[gene_col], errors="coerce")
    if mito_col is not None:
        metadata["percent_mito"] = pd.to_numeric(metadata[mito_col], errors="coerce")

    if age_col_for_bins is None:
        age_col_for_bins = find_col(metadata, ["Age_range_normalized", "Age_range", "age_group", "age", "Age", "age_years", "Age_years"])
    if age_col_for_bins is None:
        raise ValueError("No age column found. Available columns:\n" + "\n".join(metadata.columns.astype(str)))
    print("Using age column for 10-year bins:", age_col_for_bins)
    age_start = extract_age_start(metadata[age_col_for_bins])
    metadata["age_source_for_10yr_bin"] = age_col_for_bins
    metadata["age_start_numeric"] = age_start
    metadata["age_10yr_30plus"] = make_age_bin_10yr(age_start)

    for col in metadata.columns:
        if pd.api.types.is_object_dtype(metadata[col]) or pd.api.types.is_categorical_dtype(metadata[col]):
            metadata[col] = clean_str(metadata[col])
    return metadata


def compute_qc_if_missing(adata: ad.AnnData) -> ad.AnnData:
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    if "n_counts" not in adata.obs.columns or pd.to_numeric(adata.obs["n_counts"], errors="coerce").isna().all():
        adata.obs["n_counts"] = np.asarray(X.sum(axis=1)).ravel()
    else:
        adata.obs["n_counts"] = pd.to_numeric(adata.obs["n_counts"], errors="coerce").values
    if "n_genes" not in adata.obs.columns or pd.to_numeric(adata.obs["n_genes"], errors="coerce").isna().all():
        adata.obs["n_genes"] = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        adata.obs["n_genes"] = pd.to_numeric(adata.obs["n_genes"], errors="coerce").values
    if "percent_mito" not in adata.obs.columns or pd.to_numeric(adata.obs["percent_mito"], errors="coerce").isna().all():
        mt_mask = np.asarray(pd.Index(adata.var_names.astype(str)).str.upper().str.startswith("MT-"), dtype=bool)
        if mt_mask.any():
            total_counts = pd.to_numeric(adata.obs["n_counts"], errors="coerce").to_numpy()
            mt_counts = np.asarray(X[:, mt_mask].sum(axis=1)).ravel()
            adata.obs["percent_mito"] = np.where(total_counts > 0, mt_counts / total_counts * 100.0, 0.0)
        else:
            adata.obs["percent_mito"] = 0.0
    else:
        adata.obs["percent_mito"] = pd.to_numeric(adata.obs["percent_mito"], errors="coerce").fillna(0).values
    return adata


def clean_for_h5ad(adata: ad.AnnData) -> ad.AnnData:
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)
    adata.obs.index.name = None
    adata.var.index.name = None
    for col in adata.obs.columns:
        if pd.api.types.is_object_dtype(adata.obs[col]) or pd.api.types.is_categorical_dtype(adata.obs[col]):
            adata.obs[col] = adata.obs[col].astype(str).replace({"nan": "unknown", "NaN": "unknown", "<NA>": "unknown", "None": "unknown"})
    for col in adata.var.columns:
        if pd.api.types.is_object_dtype(adata.var[col]) or pd.api.types.is_categorical_dtype(adata.var[col]):
            adata.var[col] = adata.var[col].astype(str).replace({"nan": "unknown", "NaN": "unknown", "<NA>": "unknown", "None": "unknown"})
    adata.uns = {}
    return adata


def update_metadata_csv(project_dir: Path, adata: ad.AnnData, output_h5ad_rel: Path, temp_metadata_rel: Path):
    metadata_path = project_dir / "metadata" / "datasets.csv"
    meta = pd.read_csv(metadata_path)
    backup = metadata_path.parent / f"datasets_backup_before_dataset13_full_agebin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    meta.to_csv(backup, index=False)
    print("Backed up metadata:", backup)
    if "dataset_id" not in meta.columns:
        meta["dataset_id"] = ""
    idx = meta["dataset_id"].astype(str).eq(DATASET_ID)
    if not idx.any():
        new = {col: "" for col in meta.columns}
        new["dataset_id"] = DATASET_ID
        meta = pd.concat([meta, pd.DataFrame([new])], ignore_index=True)
        idx = meta["dataset_id"].astype(str).eq(DATASET_ID)
    values = {
        "dataset_short_name": "Age_full_30plus_10yr",
        "species": "Human",
        "tissue": "Heart",
        "data_type": "snRNA-seq",
        "modality": "single-nucleus",
        "source_id": "heart_sn_rna_list_rds",
        "raw_path": "Data/raw/dataset_13_Age/heart_sn_rna_list.rds",
        "h5ad_path": str(output_h5ad_rel).replace("\\", "/"),
        "converted_h5ad_path": str(output_h5ad_rel).replace("\\", "/"),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "disease_states": "; ".join(sorted(adata.obs["age_10yr_30plus"].astype(str).unique())),
        "disease_col": "age_10yr_30plus",
        "donor_col": "donor_id",
        "sample_col": "biosample_id",
        "sex_col": "sex",
        "age_col": "age",
        "lvef_col": "lvef",
        "celltype_col": "cell_type_leiden0.6",
        "subcluster_col": "SubCluster",
        "count_col": "n_counts",
        "gene_count_col": "n_genes",
        "mito_col": "percent_mito",
        "doublet_score_col": "",
        "exon_prop_col": "",
        "raw_count_layer": "",
        "analysis_matrix_note": "Full RDS export converted to h5ad. X stores raw counts. Cells below age 30 removed. Split column for 03 is age_10yr_30plus.",
        "include_in_cross_dataset_analysis": "No_temp_age_bin_run",
        "notes": "Temporary age-bin clustering metadata using full RDS-derived expression matrix.",
    }
    for col in values:
        if col not in meta.columns:
            meta[col] = ""
    for col, val in values.items():
        meta.loc[idx, col] = val
    out = project_dir / temp_metadata_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False)
    print("Saved temporary metadata:", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dir", default=str(EXPORT_REL))
    parser.add_argument("--output_h5ad", default=str(OUTPUT_H5AD_REL))
    parser.add_argument("--temp_metadata", default=str(TEMP_METADATA_REL))
    parser.add_argument("--age_col", default="")
    parser.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    args = parser.parse_args()

    project_dir = find_project_dir()
    export_dir = project_dir / args.export_dir
    mtx_path = export_dir / "counts_genes_by_cells.mtx"
    genes_path = export_dir / "genes.csv"
    cells_path = export_dir / "cells.csv"
    metadata_path = export_dir / "metadata.csv"
    for p in [mtx_path, genes_path, cells_path, metadata_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required export file:\n{p}")
    output_h5ad = project_dir / args.output_h5ad
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)

    print("Reading genes/cells/metadata...")
    genes = pd.read_csv(genes_path)
    cells = pd.read_csv(cells_path)
    metadata_raw = pd.read_csv(metadata_path)
    gene_col = "gene_symbol" if "gene_symbol" in genes.columns else genes.columns[0]
    cell_col = "cell" if "cell" in cells.columns else cells.columns[0]
    gene_names = genes[gene_col].astype(str).tolist()
    cell_names = cells[cell_col].astype(str).tolist()
    metadata = standardize_obs(metadata_raw, age_col_for_bins=args.age_col.strip() or None)
    metadata = metadata.reindex(cell_names)
    if metadata.isna().all(axis=1).any():
        raise ValueError(f"{int(metadata.isna().all(axis=1).sum())} cells from cells.csv are missing in metadata.csv.")
    print("Age-bin counts before filtering:")
    print(metadata["age_10yr_30plus"].value_counts(dropna=False).sort_index())
    keep_mask = metadata["age_10yr_30plus"].notna()
    if not keep_mask.any():
        raise ValueError("No cells remain after filtering age <30 / unknown age.")

    print("Reading MTX...")
    X = mmread(str(mtx_path))
    if not sp.issparse(X):
        X = sp.coo_matrix(X)
    X = X.tocsr()
    print("Raw MTX shape:", X.shape)
    if X.shape == (len(gene_names), len(cell_names)):
        print("Matrix is genes x cells; transposing to cells x genes.")
        X = X.T.tocsr()
    elif X.shape != (len(cell_names), len(gene_names)):
        raise ValueError(f"MTX shape {X.shape} does not match genes={len(gene_names)}, cells={len(cell_names)}.")

    keep_pos = np.where(keep_mask.to_numpy())[0]
    X = X[keep_pos, :].tocsr()
    metadata = metadata.iloc[keep_pos].copy()
    cell_names = metadata.index.astype(str).tolist()
    var = pd.DataFrame(index=pd.Index(gene_names).astype(str))
    var["gene_symbol"] = gene_names
    if "gene_original" in genes.columns:
        var["gene_original"] = genes["gene_original"].astype(str).values

    print("Creating AnnData...")
    adata = ad.AnnData(X=X, obs=metadata, var=var)
    adata.obs_names = pd.Index(cell_names).astype(str)
    adata.var_names = pd.Index(gene_names).astype(str)
    adata.var_names_make_unique()
    adata = compute_qc_if_missing(adata)
    adata = clean_for_h5ad(adata)
    print("Final AnnData:")
    print(adata)
    print("Final age-bin counts:")
    print(adata.obs["age_10yr_30plus"].value_counts().sort_index())
    marker_check = pd.DataFrame({
        "marker": ["PECAM1", "VWF", "CDH5", "PROX1", "LYVE1", "PDPN", "FLT4", "CCL21", "MMRN1", "TBX1"],
        "present": [g in adata.var_names for g in ["PECAM1", "VWF", "CDH5", "PROX1", "LYVE1", "PDPN", "FLT4", "CCL21", "MMRN1", "TBX1"]],
    })
    print("Marker check:")
    print(marker_check.to_string(index=False))
    pd.DataFrame([
        {"metric": "n_cells", "value": adata.n_obs},
        {"metric": "n_genes", "value": adata.n_vars},
        {"metric": "output_h5ad", "value": str(output_h5ad)},
        {"metric": "age_bins", "value": "; ".join(sorted(adata.obs["age_10yr_30plus"].astype(str).unique()))},
    ]).to_csv(output_h5ad.parent / "dataset_13_Age_full_30plus_age10yr_build_summary.csv", index=False)
    marker_check.to_csv(output_h5ad.parent / "dataset_13_Age_full_30plus_age10yr_marker_check.csv", index=False)
    print("Writing h5ad:", output_h5ad)
    compression = None if args.compression == "none" else args.compression
    adata.write_h5ad(output_h5ad, compression=compression)
    update_metadata_csv(project_dir, adata, Path(args.output_h5ad), Path(args.temp_metadata))
    print("Done.")


if __name__ == "__main__":
    main()
