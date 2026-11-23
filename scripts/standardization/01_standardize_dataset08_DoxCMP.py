from pathlib import Path

import pandas as pd
import scanpy as sc


DATASET_ID = "dataset_08_DoxCMP"


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[2]


def clean_string_columns(adata):
    adata.obs.index.name = None
    adata.var.index.name = None

    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

    numeric_cols = [
        "nCount_RNA",
        "nFeature_RNA",
        "percent.mt",
        "nCount_SCT",
        "nFeature_SCT",
        "log10GenesPerUMI",
    ]

    for col in numeric_cols:
        if col in adata.obs.columns:
            adata.obs[col] = pd.to_numeric(
                adata.obs[col].astype(str),
                errors="coerce"
            )

    for col in adata.obs.columns:
        if col in numeric_cols:
            continue

        if str(adata.obs[col].dtype) in ["object", "category"]:
            adata.obs[col] = adata.obs[col].astype(str)

    for col in adata.var.columns:
        if str(adata.var[col].dtype) in ["object", "category"]:
            adata.var[col] = adata.var[col].astype(str)

    return adata


def main():
    project_dir = find_project_dir()

    raw_h5ad = (
        project_dir
        / "Data"
        / "raw"
        / DATASET_ID
        / f"{DATASET_ID}.h5ad"
    )

    converted_dir = project_dir / "Data" / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)

    output_h5ad = converted_dir / f"{DATASET_ID}_standardized.h5ad"

    print("Loading:")
    print(raw_h5ad)

    if not raw_h5ad.exists():
        raise FileNotFoundError(
            f"Cannot find raw h5ad:\n{raw_h5ad}\n\n"
            "Run 00_build_dataset08_DoxCMP_h5ad.py first."
        )

    adata = sc.read_h5ad(raw_h5ad)
    adata.var_names_make_unique()

    print("\nLoaded:")
    print(adata)

    if "counts" in adata.layers:
        print("\nLayer counts exists.")
        print("X is already raw RNA counts from the build step, so NOT copying counts into X.")
    else:
        print("\nNo counts layer found. Using current X.")

    # -----------------------------
    # Disease/group
    # -----------------------------
    if "group" not in adata.obs.columns:
        raise ValueError("Expected metadata column 'group' was not found.")

    adata.obs["disease_original"] = adata.obs["group"].astype(str)

    disease_map = {
        "Donor": "NF",
        "donor": "NF",
        "DoxCM": "DoxCM",
        "DOXCM": "DoxCM",
        "NICM": "NICM",
        "DCM": "DCM",
    }

    adata.obs["disease"] = adata.obs["disease_original"].replace(disease_map)

    # -----------------------------
    # Core standardized columns
    # -----------------------------
    adata.obs["dataset_id"] = DATASET_ID
    adata.obs["source_id"] = "GSE292067"
    adata.obs["modality"] = "single-nucleus"

    if "orig.ident" in adata.obs.columns:
        adata.obs["donor_id"] = adata.obs["orig.ident"].astype(str)
        adata.obs["biosample_id"] = adata.obs["orig.ident"].astype(str)
    else:
        adata.obs["donor_id"] = "unknown_donor"
        adata.obs["biosample_id"] = "unknown_sample"

    adata.obs["sex"] = pd.NA
    adata.obs["age"] = pd.NA
    adata.obs["lvef"] = pd.NA

    # -----------------------------
    # Author annotation
    # -----------------------------
    if "celltype" in adata.obs.columns:
        adata.obs["cell_type_leiden0.6"] = adata.obs["celltype"].astype(str)
    else:
        adata.obs["cell_type_leiden0.6"] = "unknown"

    if "seurat_clusters" in adata.obs.columns:
        adata.obs["SubCluster"] = adata.obs["seurat_clusters"].astype(str)
    else:
        adata.obs["SubCluster"] = adata.obs["cell_type_leiden0.6"].astype(str)

    for col in ["cell_type_leiden0.6", "SubCluster"]:
        adata.obs[col] = (
            adata.obs[col]
            .astype(str)
            .replace({"nan": "unknown", "NaN": "unknown", "<NA>": "unknown"})
        )

    # QC alias columns
    if "nCount_RNA" in adata.obs.columns:
        adata.obs["n_counts"] = pd.to_numeric(
            adata.obs["nCount_RNA"].astype(str),
            errors="coerce"
        )

    if "nFeature_RNA" in adata.obs.columns:
        adata.obs["n_genes"] = pd.to_numeric(
            adata.obs["nFeature_RNA"].astype(str),
            errors="coerce"
        )

    if "percent.mt" in adata.obs.columns:
        adata.obs["percent_mito"] = pd.to_numeric(
            adata.obs["percent.mt"].astype(str),
            errors="coerce"
        )

    markers = [
        "CDH5", "VWF", "PECAM1",
        "PROX1", "LYVE1", "PDPN",
        "CCL21", "FLT4", "MMRN1", "TBX1"
    ]

    marker_check = pd.DataFrame({
        "marker": markers,
        "present": [m.upper() in {g.upper() for g in adata.var_names} for m in markers]
    })

    print("\nMarker check:")
    print(marker_check)

    print("\nDisease counts:")
    print(adata.obs["disease"].value_counts(dropna=False))

    print("\nDonor counts:")
    print(adata.obs["donor_id"].value_counts(dropna=False).head(30))

    print("\nCell type counts:")
    print(adata.obs["cell_type_leiden0.6"].value_counts(dropna=False))

    # To save memory/disk, remove duplicate counts layer.
    # X already contains the raw count matrix.
    if "counts" in adata.layers:
        del adata.layers["counts"]

    adata.uns = {}
    adata = clean_string_columns(adata)

    print("\nSaving standardized h5ad:")
    print(output_h5ad)

    if output_h5ad.exists():
        output_h5ad.unlink()

    adata.write_h5ad(output_h5ad, compression="gzip")

    print("\nDone.")
    print(adata)


if __name__ == "__main__":
    main()