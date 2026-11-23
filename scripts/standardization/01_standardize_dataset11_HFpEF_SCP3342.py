from pathlib import Path
import gc

import pandas as pd
import scanpy as sc


DATASET_ID = "dataset_11_HFpEF_SCP3342"


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[2]


def find_raw_h5ad(raw_dir: Path) -> Path:
    h5ads = [
        p for p in raw_dir.rglob("*.h5ad")
        if not p.name.startswith("._")
    ]

    if len(h5ads) == 0:
        raise FileNotFoundError(f"Cannot find raw h5ad under:\n{raw_dir}")

    if len(h5ads) > 1:
        print("\nMultiple h5ad files found. Using first:")
        for p in h5ads:
            print(p)

    return h5ads[0]


def clean_h5ad(adata):
    adata.obs.index.name = None
    adata.var.index.name = None

    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

    numeric_obs_cols = [
    "ncount",
    "count",
    "ngenes",
    "genes",
    "percent_mito",
    "entropy",
    "doublet_scores",
    "n_counts",
    "n_genes",
    "cellranger_percent_mito",
    "cellranger_doublet_scores",
]

    for col in numeric_obs_cols:
        if col in adata.obs.columns:
            adata.obs[col] = pd.to_numeric(
                adata.obs[col].astype(str),
                errors="coerce"
            )

    for col in adata.obs.columns:
        if col in numeric_obs_cols:
            continue

        if str(adata.obs[col].dtype) in ["object", "category"]:
            adata.obs[col] = adata.obs[col].astype(str)

    for col in adata.var.columns:
        if str(adata.var[col].dtype) in ["object", "category"]:
            adata.var[col] = adata.var[col].astype(str)

    return adata


def main():
    project_dir = find_project_dir()

    raw_dir = project_dir / "Data" / "raw" / DATASET_ID
    converted_dir = project_dir / "Data" / "converted"
    results_dir = project_dir / "results" / DATASET_ID

    converted_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_h5ad = find_raw_h5ad(raw_dir)
    output_h5ad = converted_dir / f"{DATASET_ID}_standardized.h5ad"

    print("Raw h5ad:")
    print(raw_h5ad)

    print("\nOutput standardized h5ad:")
    print(output_h5ad)

    adata = sc.read_h5ad(raw_h5ad)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # Clean possible hidden spaces in column names.
    adata.obs.columns = adata.obs.columns.astype(str).str.strip()
    adata.var.columns = adata.var.columns.astype(str).str.strip()

    print("\nLoaded raw dataset11:")
    print(adata)

    print("\nRaw obs columns:")
    print(adata.obs.columns.tolist())

    print("\nRaw layers:")
    print(list(adata.layers.keys()))

    # --------------------------------------------------
    # Memory-light count handling
    # --------------------------------------------------
    # Use one raw count matrix as X.
    # Do NOT keep duplicate count layers.
    # This prevents memory explosion in the largeDataset script.
    if "raw_count_cellbender" in adata.layers:
        print("\nUsing raw_count_cellbender as X raw counts.")
        adata.X = adata.layers["raw_count_cellbender"]
    elif "raw_count_cellranger" in adata.layers:
        print("\nUsing raw_count_cellranger as X raw counts.")
        adata.X = adata.layers["raw_count_cellranger"]
    else:
        print("\nNo raw count layer found. Keeping current X as raw counts.")

    print("\nDeleting all large layers to save memory.")
    for layer in list(adata.layers.keys()):
        del adata.layers[layer]

    gc.collect()

    print("\nLayers after deletion:")
    print(list(adata.layers.keys()))

    # Keep author UMAP as reference if present.
    if "X_umap" in adata.obsm:
        adata.obsm["X_umap_author_ref"] = adata.obsm["X_umap"].copy()

    # --------------------------------------------------
    # Standard dataset columns
    # --------------------------------------------------
    adata.obs["dataset_id"] = DATASET_ID
    adata.obs["source_id"] = "SCP3342"
    adata.obs["modality"] = "single-nucleus"

    # Use original_disease, not disease ontology IDs.
    if "original_disease" not in adata.obs.columns:
        raise ValueError(
            "Missing original_disease column.\n\nAvailable obs columns are:\n"
            + str(adata.obs.columns.tolist())
        )

    adata.obs["disease_original"] = adata.obs["original_disease"].astype(str)

    adata.obs["disease"] = (
        adata.obs["disease_original"]
        .replace({
            "control": "NF",
            "Control": "NF",
            "hfpef": "HFpEF",
            "HFpEF": "HFpEF",
        })
        .astype(str)
    )

    # Donor/sample
    if "donor_id" in adata.obs.columns:
        adata.obs["donor_id"] = adata.obs["donor_id"].astype(str)
    else:
        adata.obs["donor_id"] = "unknown_donor"

    if "biosample_id" in adata.obs.columns:
        adata.obs["biosample_id"] = adata.obs["biosample_id"].astype(str)
    else:
        adata.obs["biosample_id"] = adata.obs["donor_id"].astype(str)

    # Clinical fields
    if "sex" in adata.obs.columns:
        adata.obs["sex"] = adata.obs["sex"].astype(str)
    else:
        adata.obs["sex"] = "unknown"

    adata.obs["age"] = "unknown"
    adata.obs["lvef"] = "unknown"

    # Author cell type labels
    if "original_cell_type" in adata.obs.columns:
        adata.obs["cell_type_leiden0.6"] = adata.obs["original_cell_type"].astype(str)
    elif "cell_type_ontology_label" in adata.obs.columns:
        adata.obs["cell_type_leiden0.6"] = adata.obs["cell_type_ontology_label"].astype(str)
    else:
        adata.obs["cell_type_leiden0.6"] = "unknown"

    if "original_cell_type_label" in adata.obs.columns:
        adata.obs["SubCluster"] = adata.obs["original_cell_type_label"].astype(str)
    else:
        adata.obs["SubCluster"] = adata.obs["cell_type_leiden0.6"].astype(str)

    # --------------------------------------------------
    # QC aliases
    # Hard-coded for dataset11 based on actual obs columns.
    # This prevents 04_largeDataset from computing these from X.
    # --------------------------------------------------
    required_qc_source_cols = [
        "ncount",
        "ngenes",
        "percent_mito",
        "doublet_scores",
    ]

    missing_qc_source_cols = [
        col for col in required_qc_source_cols
        if col not in adata.obs.columns
    ]

    if len(missing_qc_source_cols) > 0:
        raise ValueError(
            "Dataset11 is missing expected QC source columns:\n"
            + str(missing_qc_source_cols)
            + "\n\nAvailable obs columns are:\n"
            + str(adata.obs.columns.tolist())
        )

    adata.obs["n_counts"] = pd.to_numeric(
        adata.obs["ncount"].astype(str),
        errors="coerce"
    )

    adata.obs["n_genes"] = pd.to_numeric(
        adata.obs["ngenes"].astype(str),
        errors="coerce"
    )

    adata.obs["percent_mito"] = pd.to_numeric(
        adata.obs["percent_mito"].astype(str),
        errors="coerce"
    )

    adata.obs["cellranger_percent_mito"] = adata.obs["percent_mito"]

    adata.obs["cellranger_doublet_scores"] = pd.to_numeric(
        adata.obs["doublet_scores"].astype(str),
        errors="coerce"
    )

    print("\nQC alias preview:")
    print(
        adata.obs[
            [
                "ncount",
                "ngenes",
                "percent_mito",
                "doublet_scores",
                "n_counts",
                "n_genes",
                "cellranger_percent_mito",
                "cellranger_doublet_scores",
            ]
        ].head()
    )

    # --------------------------------------------------
    # Marker check
    # --------------------------------------------------
    markers = [
        "CDH5",
        "VWF",
        "PECAM1",
        "PROX1",
        "LYVE1",
        "PDPN",
        "CCL21",
        "FLT4",
        "MMRN1",
        "TBX1",
    ]

    gene_set = {g.upper() for g in adata.var_names.astype(str)}

    marker_check = pd.DataFrame({
        "marker": markers,
        "present": [m.upper() in gene_set for m in markers],
    })

    print("\nMarker check:")
    print(marker_check)

    marker_check.to_csv(
        results_dir / "dataset11_marker_check_after_standardization.csv",
        index=False
    )

    print("\nDisease counts:")
    print(adata.obs["disease"].value_counts(dropna=False))

    print("\nAuthor cell type counts:")
    print(adata.obs["cell_type_leiden0.6"].value_counts(dropna=False).head(50))

    print("\nSubCluster counts:")
    print(adata.obs["SubCluster"].value_counts(dropna=False).head(50))

    # --------------------------------------------------
    # Clean and save
    # --------------------------------------------------
    adata = clean_h5ad(adata)

    if output_h5ad.exists():
        output_h5ad.unlink()

    adata.write_h5ad(output_h5ad, compression="gzip")

    summary = pd.DataFrame([{
        "dataset_id": DATASET_ID,
        "raw_h5ad_path": str(raw_h5ad.relative_to(project_dir)),
        "standardized_h5ad_path": str(output_h5ad.relative_to(project_dir)),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "disease_col": "disease",
        "donor_col": "donor_id",
        "sample_col": "biosample_id",
        "celltype_col": "cell_type_leiden0.6",
        "subcluster_col": "SubCluster",
        "raw_count_matrix": "X",
        "layers_kept": "none",
        "qc_count_col": "n_counts",
        "qc_gene_col": "n_genes",
        "qc_mito_col": "percent_mito",
        "qc_doublet_col": "cellranger_doublet_scores",
        "disease_mapping": "original_disease: control -> NF, hfpef -> HFpEF",
        "script03_recommendation": "Do not run Script03 for dataset11 on local 16GB memory. Use 04_largeDataset_QC_cellLevel_find_LEC.py.",
    }])

    summary.to_csv(
        converted_dir / f"{DATASET_ID}_standardization_summary.csv",
        index=False
    )

    print("\nSaved memory-light standardized h5ad:")
    print(output_h5ad)

    print("\nFinal standardized AnnData:")
    print(adata)

    print("\nDone.")


if __name__ == "__main__":
    main()