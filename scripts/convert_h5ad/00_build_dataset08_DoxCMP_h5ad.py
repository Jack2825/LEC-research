from pathlib import Path

import anndata as ad
import pandas as pd
from scipy.io import mmread


DATASET_ID = "dataset_08_DoxCMP"


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[2]


def clean_string_columns(adata):
    adata.obs.index.name = None
    adata.var.index.name = None

    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

    for col in adata.obs.columns:
        if str(adata.obs[col].dtype) in ["object", "category"]:
            adata.obs[col] = adata.obs[col].astype(str)

    for col in adata.var.columns:
        if str(adata.var[col].dtype) in ["object", "category"]:
            adata.var[col] = adata.var[col].astype(str)

    return adata


def main():
    project_dir = find_project_dir()

    raw_dir = project_dir / "Data" / "raw" / DATASET_ID
    export_dir = raw_dir / f"{DATASET_ID}_export"

    output_h5ad = raw_dir / f"{DATASET_ID}.h5ad"

    print("Export folder:")
    print(export_dir)

    matrix_file = export_dir / "counts_genes_by_cells.mtx"
    genes_file = export_dir / "genes.csv"
    cells_file = export_dir / "cells.csv"
    metadata_file = export_dir / "metadata.csv"

    for path in [matrix_file, genes_file, cells_file, metadata_file]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file:\n{path}")

    print("\nReading matrix...")
    # Matrix is genes x cells. AnnData needs cells x genes.
    X = mmread(matrix_file).T.tocsr()

    print("Reading genes/cells/metadata...")
    genes = pd.read_csv(genes_file)["gene"].astype(str).tolist()
    cells = pd.read_csv(cells_file)["cell"].astype(str).tolist()

    metadata = pd.read_csv(metadata_file)
    metadata["cell"] = metadata["cell"].astype(str)

    if metadata["cell"].duplicated().any():
        print("Warning: duplicated cells in metadata. Keeping first.")
        metadata = metadata.drop_duplicates("cell", keep="first")

    metadata = metadata.set_index("cell")
    metadata = metadata.loc[cells]

    var = pd.DataFrame(index=genes)
    obs = metadata

    print("\nCreating AnnData...")
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()

    adata.layers["counts"] = adata.X.copy()

    # Add UMAP if present
    umap_file = export_dir / "umap.csv"
    if umap_file.exists():
        print("Adding UMAP...")
        umap = pd.read_csv(umap_file)
        umap["cell"] = umap["cell"].astype(str)
        umap = umap.set_index("cell").loc[cells]
        adata.obsm["X_umap"] = umap.iloc[:, :2].to_numpy()

    # Add PCA if present
    pca_file = export_dir / "pca.csv"
    if pca_file.exists():
        print("Adding PCA...")
        pca = pd.read_csv(pca_file)
        pca["cell"] = pca["cell"].astype(str)
        pca = pca.set_index("cell").loc[cells]
        adata.obsm["X_pca"] = pca.drop(columns=[], errors="ignore").to_numpy()

    # Add harmony if present
    harmony_file = export_dir / "harmony.csv"
    if harmony_file.exists():
        print("Adding harmony...")
        harmony = pd.read_csv(harmony_file)
        harmony["cell"] = harmony["cell"].astype(str)
        harmony = harmony.set_index("cell").loc[cells]
        adata.obsm["X_harmony"] = harmony.to_numpy()

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

    adata.uns = {}
    adata = clean_string_columns(adata)

    print("\nWriting h5ad:")
    print(output_h5ad)

    if output_h5ad.exists():
        output_h5ad.unlink()

    adata.write_h5ad(output_h5ad, compression="gzip")

    print("\nDone.")
    print(adata)


if __name__ == "__main__":
    main()