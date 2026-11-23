#!/usr/bin/env python3
r"""
Step 08: combine final NF LECs from Q1-Q4 and recluster them together.

The final LEC identities come from Step 06 ``selected_as_LEC``. Expression is
recovered from the matching Step 03 ``adata.raw`` objects, rather than from the
Step 06 HVG matrices. This retains every gene available in the Step 03 inputs.
The four quartiles are concatenated with an OUTER gene join; a gene-presence
table records whether every gene was available in every input.

The combined expression matrix remains log-normalized and unscaled. Scaling
and Harmony are performed only in a temporary object used for PCA, neighbors,
UMAP, and Leiden clustering. Harmony uses ``Project`` by default and never uses
age quartile, exact age, sex, or donor as a batch variable.
The canonical resolution for this run is 0.3. State names are regenerated from
``state0`` upward and are therefore specific to this resolution and cell set.

Default input
-------------
new_result2_aging/03/dataset13/Q*/dataset13_Q*_clustered.h5ad
new_result2_aging/06_EC_recluster/dataset13/Q*/dataset13_Q*_EC_reclustered.h5ad

Default output
--------------
new_result2_aging/08_combined_LEC_recluster/dataset13/

Main outputs
------------
08_combined_final_LEC_reclustered.h5ad
08_combined_LEC_cell_metadata.csv.gz
08_state_counts.csv
08_state_by_age_quartile.csv
08_state_donor_project_QC.csv
08_state_marker_ranking_all_genes.csv.gz
08_state_top25_markers_long.csv
08_gene_presence_by_quartile.csv
08_method_and_interpretation_notes.txt
figures/*.png

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_aging\08_combine_recluster_final_LEC_age_quartiles.py
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
from typing import Iterable, Optional

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp


# The project Python installation does not include a functional Tk/Tcl GUI.
# Force a non-interactive backend so figure creation is terminal-safe.
matplotlib.use("Agg", force=True)


SCRIPT_VERSION = "08_dataset13_NF_combined_LEC_states_res0p3_2026_08_07"
DATASET = "dataset13"
REQUIRED_DISEASE = "NF"
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
DEFAULT_RESOLUTION = 0.3
DEFAULT_OUTPUT_PREFIX = "08"

DEFAULT_ALLCELL_ROOT = "new_result2_aging/03/dataset13"
DEFAULT_EC_ROOT = "new_result2_aging/06_EC_recluster/dataset13"
DEFAULT_OUTPUT_ROOT = "new_result2_aging/08_combined_LEC_recluster/dataset13"
DEFAULT_OUT_H5AD = "08_combined_final_LEC_reclustered.h5ad"


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine final Step 06 NF LECs across Q1-Q4, retain the outer "
            "union of Step 03 raw genes, and define combined LEC states."
        )
    )
    parser.add_argument("--allcell_root", default=DEFAULT_ALLCELL_ROOT)
    parser.add_argument("--ec_root", default=DEFAULT_EC_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--out_h5ad", default=DEFAULT_OUT_H5AD)
    parser.add_argument(
        "--quartiles", nargs="+", choices=QUARTILES, default=QUARTILES
    )
    parser.add_argument("--n_top_hvg", type=int, default=2000)
    parser.add_argument("--n_pcs", type=int, default=30)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument(
        "--harmony_batch_key",
        default="Project",
        help=(
            "Technical batch column used for Harmony on PCA. Default: Project. "
            "Age quartile is never used automatically."
        ),
    )
    parser.add_argument(
        "--no_harmony",
        action="store_true",
        help="Use ordinary PCA for neighbors/UMAP instead of Harmony-corrected PCA.",
    )
    parser.add_argument("--top_n_markers", type=int, default=25)
    parser.add_argument(
        "--exclude_donors",
        nargs="*",
        default=[],
        help=(
            "Exact donor_key values to remove before HVG selection, Harmony, "
            "neighbors, Leiden clustering, marker ranking, and all summaries."
        ),
    )
    parser.add_argument(
        "--output_prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=(
            "Prefix used for every output filename. The primary Step 08 run "
            "uses '08'; donor-exclusion sensitivity runs can use '08_2'."
        ),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--skip_figures", action="store_true", help="Skip all brief figures."
    )
    parser.add_argument(
        "--skip_markers", action="store_true", help="Skip state marker ranking."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing primary output h5ad.",
    )
    return parser.parse_args()


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def true_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return normalized_text(values).str.lower().isin(
        {"true", "1", "yes", "y", "lec"}
    )


def natural_key(value: object) -> tuple:
    text = str(value)
    pieces = re.split(r"(\d+(?:\.\d+)?)", text)
    out: list[tuple[int, object]] = []
    for piece in pieces:
        if not piece:
            continue
        try:
            out.append((0, float(piece)))
        except ValueError:
            out.append((1, piece.lower()))
    return tuple(out)


def first_existing(columns: Iterable[str], frame: pd.DataFrame) -> Optional[str]:
    return next((column for column in columns if column in frame.columns), None)


def donor_key_from_obs(obs: pd.DataFrame) -> pd.Series:
    if "donor_key" in obs.columns:
        key = normalized_text(obs["donor_key"])
        if key.ne("").all():
            return key
    if "donor_id_quartile_unique" in obs.columns:
        key = normalized_text(obs["donor_id_quartile_unique"])
        if key.ne("").all():
            return key
    project = first_existing(["Project", "project", "source_id"], obs)
    donor = first_existing(
        ["donor_id", "Donor", "donor", "sample_id", "biosample_id"], obs
    )
    if project and donor:
        return normalized_text(obs[project]) + "::" + normalized_text(obs[donor])
    if donor:
        return normalized_text(obs[donor])
    raise KeyError(
        "Cannot construct donor_key. Expected donor_id_quartile_unique, donor_id, "
        "Donor, sample_id, or biosample_id."
    )


def close_backed(adata: ad.AnnData) -> None:
    try:
        adata.file.close()
    except Exception:
        pass


def validate_nf_quartile(obs: pd.DataFrame, quartile: str, source: Path) -> None:
    if "age_quartile" in obs.columns:
        found = set(normalized_text(obs["age_quartile"]).unique())
        if found != {quartile}:
            raise ValueError(
                f"Age-quartile mismatch in {source}: expected {quartile}, found {found}."
            )
    disease_col = first_existing(["disease", "Disease", "condition"], obs)
    if disease_col and disease_col.lower() != "condition":
        found = set(normalized_text(obs[disease_col]).str.upper().unique())
        if found != {REQUIRED_DISEASE}:
            raise ValueError(
                f"Expected NF only in {source}; {disease_col} contains {found}."
            )


def load_quartile_lecs(
    quartile: str,
    allcell_root: Path,
    ec_root: Path,
) -> tuple[ad.AnnData, dict[str, object], set[str]]:
    """Use Step 06 for LEC identities and Step 03 raw for full expression."""
    prefix = f"dataset13_{quartile}"
    all_path = allcell_root / quartile / f"{prefix}_clustered.h5ad"
    ec_path = ec_root / quartile / f"{prefix}_EC_reclustered.h5ad"
    if not all_path.exists():
        raise FileNotFoundError(f"Step 03 input not found: {all_path}")
    if not ec_path.exists():
        raise FileNotFoundError(f"Step 06 input not found: {ec_path}")

    print(f"[{quartile}] reading Step 06 LEC identities: {ec_path}")
    ec_backed = ad.read_h5ad(ec_path, backed="r")
    ec_obs = ec_backed.obs.copy()
    validate_nf_quartile(ec_obs, quartile, ec_path)
    if "selected_as_LEC" not in ec_obs.columns:
        close_backed(ec_backed)
        raise KeyError(f"selected_as_LEC is absent from {ec_path}")
    lec_flag = true_mask(ec_obs["selected_as_LEC"])
    lec_meta = ec_obs.loc[lec_flag].copy()
    n_ec = int(ec_backed.n_obs)
    close_backed(ec_backed)
    if lec_meta.empty:
        raise ValueError(f"No final Step 06 LEC cells were selected for {quartile}.")

    if "celltypist_majority_vote" in lec_meta.columns:
        labels = set(normalized_text(lec_meta["celltypist_majority_vote"]).unique())
        if labels != {"EC8_ln"}:
            raise ValueError(
                f"Final LECs in {quartile} are not exclusively EC8_ln majority vote: "
                f"{sorted(labels)}"
            )

    print(f"[{quartile}] recovering full Step 03 raw expression: {all_path}")
    all_adata = ad.read_h5ad(all_path)
    validate_nf_quartile(all_adata.obs, quartile, all_path)
    selected_ids = pd.Index(lec_meta.index.astype(str))
    all_ids = pd.Index(all_adata.obs_names.astype(str))
    missing = selected_ids.difference(all_ids)
    if len(missing):
        raise ValueError(
            f"{len(missing):,} Step 06 LEC cells are absent from Step 03 for "
            f"{quartile}; examples: {missing[:5].tolist()}"
        )

    selected = all_adata[selected_ids].copy()
    if selected.raw is not None:
        expression = selected.raw.to_adata()
        expression.obs = selected.obs.copy()
        expression_source = "step03_raw"
    else:
        warnings.warn(
            f"Step 03 raw is absent for {quartile}; using Step 03 X instead."
        )
        expression = selected
        expression_source = "step03_X"
    del selected, all_adata
    gc.collect()

    # Carry Step 06 evidence without replacing Step 03 metadata.
    carry_columns = [
        "allcell_leiden",
        "step05_celltypist_raw_label",
        "step05_celltypist_majority_vote",
        "celltypist_raw_label",
        "celltypist_majority_vote",
        "celltypist_max_probability",
        "celltypist_probability_best_label",
        "selected_as_LEC",
        "final_LEC_by_majority_vote",
        "ec_recluster_id",
    ]
    aligned_meta = lec_meta.reindex(expression.obs_names.astype(str))
    for column in carry_columns:
        if column in aligned_meta.columns:
            expression.obs[column] = aligned_meta[column].to_numpy()
    if "leiden" in aligned_meta.columns:
        expression.obs["step06_ec_leiden"] = normalized_text(
            aligned_meta["leiden"]
        ).to_numpy()

    expression.var_names = expression.var_names.astype(str)
    expression.var_names_make_unique()
    original_ids = expression.obs_names.astype(str)
    expression.obs["source_cell_id"] = original_ids
    expression.obs["source_step03_h5ad"] = str(all_path)
    expression.obs["source_step06_h5ad"] = str(ec_path)
    expression.obs["dataset_folder"] = DATASET
    expression.obs["condition"] = quartile
    expression.obs["disease"] = REQUIRED_DISEASE
    expression.obs["age_quartile"] = quartile
    expression.obs["donor_key"] = donor_key_from_obs(expression.obs).to_numpy()
    expression.obs_names = pd.Index(
        [f"{DATASET}|{quartile}|{cell_id}" for cell_id in original_ids],
        dtype="object",
    )

    genes = set(expression.var_names.astype(str))
    summary = {
        "dataset": DATASET,
        "disease": REQUIRED_DISEASE,
        "age_quartile": quartile,
        "n_step06_EC_cells": n_ec,
        "n_final_LEC_cells": int(expression.n_obs),
        "n_final_LEC_donors": int(expression.obs["donor_key"].nunique()),
        "n_genes_recovered": int(expression.n_vars),
        "expression_source": expression_source,
        "step03_h5ad": str(all_path),
        "step06_h5ad": str(ec_path),
    }
    print(
        f"[{quartile}] retained {expression.n_obs:,} final LECs, "
        f"{summary['n_final_LEC_donors']} donors, and {expression.n_vars:,} genes."
    )
    return expression, summary, genes


def make_gene_presence_table(
    union_genes: Iterable[str], genes_by_quartile: dict[str, set[str]]
) -> pd.DataFrame:
    rows = []
    for gene in sorted(set(union_genes), key=natural_key):
        row: dict[str, object] = {"gene": gene}
        present = 0
        for quartile in QUARTILES:
            value = gene in genes_by_quartile.get(quartile, set())
            row[f"present_{quartile}"] = value
            present += int(value)
        row["n_quartiles_present"] = present
        row["present_all_requested_quartiles"] = present == len(genes_by_quartile)
        rows.append(row)
    return pd.DataFrame(rows)


def transfer_embedding(combined: ad.AnnData, work: ad.AnnData) -> None:
    for key in ["X_pca", "X_pca_harmony", "X_umap"]:
        if key in work.obsm:
            combined.obsm[key] = work.obsm[key].copy()
    for key in ["connectivities", "distances"]:
        if key in work.obsp:
            combined.obsp[key] = work.obsp[key].copy()
    for key in ["pca", "neighbors", "umap", "combined_leiden"]:
        if key in work.uns:
            combined.uns[key] = work.uns[key]


def compute_embedding_and_states(
    combined: ad.AnnData,
    args: argparse.Namespace,
) -> tuple[ad.AnnData, dict[str, object], pd.DataFrame]:
    import scanpy as sc

    if combined.n_obs < 3 or combined.n_vars < 3:
        raise ValueError(
            f"Too few observations/features: {combined.n_obs} cells, {combined.n_vars} genes."
        )
    work = combined.copy()
    n_top = min(max(50, int(args.n_top_hvg)), work.n_vars)
    try:
        sc.pp.highly_variable_genes(
            work, n_top_genes=n_top, flavor="seurat", subset=False
        )
        hvg_mask = work.var["highly_variable"].fillna(False).to_numpy(dtype=bool)
    except Exception as error:
        warnings.warn(f"HVG selection failed; using every retained gene: {error}")
        hvg_mask = np.ones(work.n_vars, dtype=bool)
    if int(hvg_mask.sum()) < 50:
        hvg_mask = np.ones(work.n_vars, dtype=bool)

    combined.var["highly_variable_for_combined_embedding"] = hvg_mask
    work = work[:, hvg_mask].copy()
    sc.pp.scale(work, max_value=10)
    n_pcs = min(int(args.n_pcs), work.n_obs - 1, work.n_vars - 1)
    if n_pcs < 2:
        raise ValueError(f"Too few dimensions for PCA: n_pcs={n_pcs}")
    sc.tl.pca(
        work,
        n_comps=n_pcs,
        svd_solver="arpack",
        random_state=args.random_state,
    )

    # Keep an uncorrected UMAP for a compact batch-correction sensitivity figure.
    uncorrected = work.copy()
    n_neighbors = max(2, min(int(args.n_neighbors), work.n_obs - 1))
    sc.pp.neighbors(
        uncorrected,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep="X_pca",
        random_state=args.random_state,
    )
    sc.tl.umap(uncorrected, random_state=args.random_state)
    combined.obsm["X_umap_uncorrected"] = uncorrected.obsm["X_umap"].copy()
    del uncorrected

    use_rep = "X_pca"
    harmony_status = "disabled"
    batch_key = None if args.no_harmony else str(args.harmony_batch_key).strip()
    if batch_key:
        if batch_key not in work.obs.columns:
            harmony_status = f"skipped_missing_batch_key_{batch_key}"
            warnings.warn(
                f"Harmony batch key {batch_key!r} is absent; using ordinary PCA."
            )
        else:
            # harmonypy 0.0.10 receives an object matrix from pandas StringDtype
            # dummy variables under recent pandas/numpy versions. A categorical
            # column keeps the design matrix numeric and avoids that failure.
            batch_labels = (
                normalized_text(work.obs[batch_key])
                .replace("", "unknown")
                .astype(str)
                .to_numpy()
            )
            work.obs[batch_key] = pd.Categorical(batch_labels)
            if work.obs[batch_key].nunique() < 2:
                harmony_status = "skipped_only_one_batch"
            else:
                try:
                    import scanpy.external as sce

                    sce.pp.harmony_integrate(
                        work,
                        key=batch_key,
                        basis="X_pca",
                        adjusted_basis="X_pca_harmony",
                    )
                    use_rep = "X_pca_harmony"
                    harmony_status = "ok"
                except Exception as error:
                    harmony_status = (
                        f"failed_{type(error).__name__}: {str(error)[:300]}"
                    )
                    warnings.warn(
                        f"Harmony failed; using ordinary PCA instead. Details: {error}"
                    )

    sc.pp.neighbors(
        work,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        random_state=args.random_state,
    )
    sc.tl.umap(work, random_state=args.random_state)
    sc.tl.leiden(
        work,
        resolution=float(args.resolution),
        key_added="combined_leiden",
        random_state=args.random_state,
    )
    transfer_embedding(combined, work)
    combined.obs["combined_leiden"] = normalized_text(
        work.obs["combined_leiden"]
    ).to_numpy()

    clusters = sorted(
        combined.obs["combined_leiden"].astype(str).unique(), key=natural_key
    )
    combined.obs["combined_leiden"] = pd.Categorical(
        combined.obs["combined_leiden"].astype(str),
        categories=clusters,
        ordered=True,
    )
    base_colors = list(matplotlib.colormaps["tab20"].colors)
    state_colors = [
        matplotlib.colors.to_hex(base_colors[index % len(base_colors)])
        for index in range(len(clusters))
    ]
    mapping = pd.DataFrame(
        {
            "combined_leiden": clusters,
            "LEC_state": [f"state{index}" for index in range(len(clusters))],
            "color_hex": state_colors,
        }
    )
    cluster_to_state = dict(
        zip(mapping["combined_leiden"], mapping["LEC_state"])
    )
    state_order = mapping["LEC_state"].tolist()
    combined.obs["LEC_state"] = pd.Categorical(
        combined.obs["combined_leiden"].map(cluster_to_state),
        categories=state_order,
        ordered=True,
    )
    # Persist one canonical state palette for every current and downstream plot.
    # Scanpy reads these keys automatically for categorical UMAP colors.
    combined.uns["LEC_state_colors"] = state_colors
    combined.uns["combined_leiden_colors"] = state_colors
    combined.uns["LEC_state_color_map"] = dict(zip(state_order, state_colors))

    params = {
        "method": "Harmony_on_PCA" if use_rep == "X_pca_harmony" else "PCA",
        "harmony_batch_key": batch_key or "",
        "harmony_status": harmony_status,
        "neighbors_representation": use_rep,
        "n_top_hvg_requested": int(args.n_top_hvg),
        "n_hvg_used": int(hvg_mask.sum()),
        "n_pcs_used": int(n_pcs),
        "n_neighbors_used": int(n_neighbors),
        "leiden_resolution": float(args.resolution),
        "random_state": int(args.random_state),
        "state_name_start": 0,
        "note": (
            "Harmony/scaling affect the embedding only. X and raw retain "
            "unscaled log-normalized expression and the outer gene union."
        ),
    }
    combined.uns["combined_LEC_embedding"] = params
    return combined, params, mapping


def state_summary_tables(
    combined: ad.AnnData,
) -> dict[str, pd.DataFrame]:
    obs = combined.obs.copy()
    state_order = list(obs["LEC_state"].cat.categories)
    obs["LEC_state"] = obs["LEC_state"].astype(str)
    obs["age_quartile"] = pd.Categorical(
        normalized_text(obs["age_quartile"]), categories=QUARTILES, ordered=True
    )

    counts = (
        obs.groupby("LEC_state", observed=True)
        .size()
        .reindex(state_order, fill_value=0)
        .rename("n_cells")
        .reset_index()
    )
    counts["fraction_of_all_LECs"] = counts["n_cells"] / len(obs)
    counts["percent_of_all_LECs"] = 100.0 * counts["fraction_of_all_LECs"]

    by_quartile = (
        obs.groupby(["age_quartile", "LEC_state"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = by_quartile.groupby("age_quartile", observed=True)["n_cells"].transform(
        "sum"
    )
    by_quartile["fraction_within_quartile"] = np.where(
        totals.gt(0), by_quartile["n_cells"] / totals, np.nan
    )
    by_quartile["percent_within_quartile"] = (
        100.0 * by_quartile["fraction_within_quartile"]
    )

    donor_counts = (
        obs.groupby(["donor_key", "age_quartile", "LEC_state"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    donor_total = (
        obs.groupby(["donor_key", "age_quartile"], observed=True)
        .size()
        .rename("n_total_LEC_cells")
        .reset_index()
    )
    donor_counts = donor_counts.merge(
        donor_total, on=["donor_key", "age_quartile"], how="left"
    )
    donor_counts["fraction_of_donor_LECs"] = (
        donor_counts["n_cells"] / donor_counts["n_total_LEC_cells"]
    )

    project_col = first_existing(["Project", "project", "source_id"], obs)
    qc_rows = []
    for state in state_order:
        group = obs.loc[obs["LEC_state"].eq(state)].copy()
        donor_vc = group["donor_key"].astype(str).value_counts()
        quartile_vc = group["age_quartile"].astype(str).value_counts()
        project_vc = (
            group[project_col].astype(str).value_counts()
            if project_col
            else pd.Series(dtype=int)
        )
        age_col = first_existing(
            ["age_numeric_resolved", "age_numeric", "age"], group
        )
        age = (
            pd.to_numeric(group[age_col], errors="coerce")
            if age_col
            else pd.Series(dtype=float)
        )
        n_cells = int(len(group))
        qc_rows.append(
            {
                "LEC_state": state,
                "combined_leiden": str(group["combined_leiden"].iloc[0]),
                "n_cells": n_cells,
                "n_donors": int(group["donor_key"].nunique()),
                "n_projects": int(group[project_col].nunique()) if project_col else np.nan,
                "n_age_quartiles": int(group["age_quartile"].nunique()),
                "age_min": float(age.min()) if age.notna().any() else np.nan,
                "age_max": float(age.max()) if age.notna().any() else np.nan,
                "top_donor": donor_vc.index[0] if len(donor_vc) else "",
                "top_donor_n_cells": int(donor_vc.iloc[0]) if len(donor_vc) else 0,
                "top_donor_fraction": (
                    float(donor_vc.iloc[0] / n_cells) if n_cells and len(donor_vc) else np.nan
                ),
                "donor_dominance_flag_over_50pct": (
                    bool(donor_vc.iloc[0] / n_cells > 0.5)
                    if n_cells and len(donor_vc)
                    else False
                ),
                "top_project": project_vc.index[0] if len(project_vc) else "",
                "top_project_fraction": (
                    float(project_vc.iloc[0] / n_cells)
                    if n_cells and len(project_vc)
                    else np.nan
                ),
                "top_age_quartile": quartile_vc.index[0] if len(quartile_vc) else "",
                "top_age_quartile_fraction": (
                    float(quartile_vc.iloc[0] / n_cells)
                    if n_cells and len(quartile_vc)
                    else np.nan
                ),
            }
        )
    state_qc = pd.DataFrame(qc_rows)

    donor_totals = (
        obs.groupby(["donor_key", "age_quartile"], observed=True)
        .agg(
            n_LEC_cells=("LEC_state", "size"),
            n_LEC_states=("LEC_state", "nunique"),
        )
        .reset_index()
    )
    for column in ["Project", "Sex", "sex", "age_numeric_resolved"]:
        if column in obs.columns:
            meta = (
                obs.groupby("donor_key", observed=True)[column]
                .first()
                .rename(column)
                .reset_index()
            )
            donor_totals = donor_totals.merge(meta, on="donor_key", how="left")

    return {
        "state_counts": counts,
        "state_by_quartile": by_quartile,
        "donor_state_counts": donor_counts,
        "state_qc": state_qc,
        "donor_totals": donor_totals,
    }


def rank_state_markers(
    combined: ad.AnnData,
    top_n: int,
    gene_presence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import scanpy as sc

    state_order = list(combined.obs["LEC_state"].cat.categories)
    sc.tl.rank_genes_groups(
        combined,
        groupby="LEC_state",
        method="wilcoxon",
        use_raw=True,
        pts=True,
        key_added="rank_genes_groups_LEC_state",
    )
    all_frames = []
    for state in state_order:
        frame = sc.get.rank_genes_groups_df(
            combined, group=state, key="rank_genes_groups_LEC_state"
        )
        frame.insert(0, "rank", np.arange(1, len(frame) + 1))
        frame.insert(0, "LEC_state", state)
        all_frames.append(frame)
    full = pd.concat(all_frames, ignore_index=True)
    coverage = gene_presence.rename(columns={"gene": "names"})
    full = full.merge(coverage, on="names", how="left")
    top = full.groupby("LEC_state", sort=False, observed=True).head(top_n).copy()
    wide = (
        top.assign(marker_label=lambda x: x["names"].astype(str))
        .pivot(index="LEC_state", columns="rank", values="marker_label")
        .reindex(state_order)
        .reset_index()
    )
    wide.columns = [
        "LEC_state" if column == "LEC_state" else f"marker_{int(column)}"
        for column in wide.columns
    ]
    return full, top, wide


def save_umap_summary(
    combined: ad.AnnData, path: Path, dpi: int, resolution: float
) -> None:
    import matplotlib.pyplot as plt
    import scanpy as sc

    age_col = first_existing(
        ["age_numeric_resolved", "age_numeric", "age"], combined.obs
    )
    project_col = first_existing(["Project", "project", "source_id"], combined.obs)
    sex_col = first_existing(["Sex", "sex"], combined.obs)
    specs = [
        ("LEC_state", "Combined LEC state"),
        ("age_quartile", "NF donor-age quartile"),
        (age_col, "Exact/resolved donor age"),
        (project_col, "Project"),
        (sex_col, "Sex"),
        ("combined_leiden", f"Combined Leiden (resolution {resolution:g})"),
    ]
    specs = [(key, title) for key, title in specs if key and key in combined.obs]
    ncols = 3
    nrows = math.ceil(len(specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 5.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, title) in zip(axes, specs):
        categorical = not pd.api.types.is_numeric_dtype(combined.obs[key])
        legend = "on data" if key in {"LEC_state", "combined_leiden"} else "right margin"
        sc.pl.umap(
            combined,
            color=key,
            ax=ax,
            show=False,
            frameon=False,
            size=16,
            alpha=0.8,
            title=title,
            legend_loc=legend if categorical else "right margin",
            color_map="viridis",
        )
    for ax in axes[len(specs) :]:
        ax.axis("off")
    fig.suptitle(
        "Dataset13 NF: final LECs combined across age quartiles",
        fontsize=16,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_batch_comparison(
    combined: ad.AnnData, batch_key: str, path: Path, dpi: int
) -> None:
    import matplotlib.pyplot as plt
    import scanpy as sc

    if batch_key not in combined.obs or "X_umap_uncorrected" not in combined.obsm:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sc.pl.embedding(
        combined,
        basis="umap_uncorrected",
        color=batch_key,
        ax=axes[0],
        show=False,
        frameon=False,
        size=16,
        title=f"Uncorrected PCA UMAP by {batch_key}",
    )
    sc.pl.umap(
        combined,
        color=batch_key,
        ax=axes[1],
        show=False,
        frameon=False,
        size=16,
        title=f"Primary UMAP by {batch_key}",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_state_composition(
    by_quartile: pd.DataFrame,
    state_color_map: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    pivot = by_quartile.pivot(
        index="age_quartile", columns="LEC_state", values="percent_within_quartile"
    ).fillna(0)
    pivot = pivot.reindex(QUARTILES)
    pivot = pivot.reindex(sorted(pivot.columns, key=natural_key), axis=1)
    colors = [state_color_map.get(str(state), "#808080") for state in pivot.columns]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    pivot.plot(kind="bar", stacked=True, color=colors, width=0.78, ax=ax)
    ax.set_xlabel("NF donor-age quartile")
    ax.set_ylabel("Cells within quartile (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Combined LEC-state composition across age quartiles")
    ax.legend(title="LEC state", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_state_support(
    state_qc: pd.DataFrame,
    state_color_map: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    states = state_qc["LEC_state"].astype(str).tolist()
    colors = [state_color_map.get(state, "#808080") for state in states]
    x = np.arange(len(states))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(x, state_qc["n_cells"], color=colors)
    axes[0].set_ylabel("Number of cells")
    axes[0].set_title("State size")
    axes[1].bar(x, state_qc["n_donors"], color=colors)
    axes[1].set_ylabel("Number of donors")
    axes[1].set_title("Donor support")
    axes[2].bar(x, 100 * state_qc["top_donor_fraction"], color=colors)
    axes[2].axhline(50, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Largest donor contribution (%)")
    axes[2].set_title("Donor dominance QC")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(states, rotation=45, ha="right")
        ax.set_xlabel("LEC state")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Combined aging LEC-state support", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_marker_dotplot(
    combined: ad.AnnData,
    top_markers: pd.DataFrame,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import scanpy as sc

    marker_dict: dict[str, list[str]] = {}
    for state, group in top_markers.groupby("LEC_state", sort=False, observed=True):
        marker_dict[str(state)] = group["names"].astype(str).head(3).tolist()
    if not marker_dict:
        return
    plot = sc.pl.dotplot(
        combined,
        var_names=marker_dict,
        groupby="LEC_state",
        use_raw=True,
        standard_scale="var",
        show=False,
        return_fig=True,
        title="Top combined-state markers",
    )
    plot.savefig(path)
    plt.close("all")


def make_obs_h5ad_safe(adata: ad.AnnData) -> None:
    for column in adata.obs.columns:
        values = adata.obs[column]
        if pd.api.types.is_string_dtype(values.dtype) and not isinstance(
            values.dtype, pd.CategoricalDtype
        ):
            adata.obs[column] = values.astype(object)
        if adata.obs[column].dtype != object:
            continue
        non_null = adata.obs[column].dropna()
        types = {type(value) for value in non_null.head(1000)}
        complex_value = any(
            isinstance(value, (list, tuple, dict, set, np.ndarray))
            for value in non_null.head(1000)
        )
        if len(types) > 1 or complex_value:
            adata.obs[column] = normalized_text(adata.obs[column]).astype(object)


def write_method_notes(
    path: Path,
    args: argparse.Namespace,
    run_summary: dict[str, object],
    state_qc: pd.DataFrame,
) -> None:
    excluded = [str(value) for value in args.exclude_donors if str(value).strip()]
    exclusion_text = "; ".join(excluded) if excluded else "none"
    dominant = state_qc.loc[
        state_qc["donor_dominance_flag_over_50pct"],
        ["LEC_state", "top_donor", "top_donor_fraction"],
    ]
    dominance_text = (
        dominant.to_csv(index=False).strip()
        if not dominant.empty
        else "No state contains more than 50% of its cells from one donor."
    )
    text = f"""Step 08 combined LEC-state method and interpretation notes
=========================================================

Run identity
------------
Script version: {SCRIPT_VERSION}
Leiden resolution: {args.resolution:g}
State labels: {run_summary['state_labels']}
Excluded donors before clustering: {exclusion_text}
LEC cells analyzed: {run_summary['n_final_LEC_cells']}

State-name rule
---------------
LEC state names are run-specific labels assigned from state0 upward after
Leiden clustering. They must not be carried across a different resolution or
donor-exclusion run. Downstream Steps 09-11 must read the state labels and
palette from this run rather than reuse names from an older Step 08 result.

Expression and embedding
------------------------
Final LEC identities come from Step 06 selected_as_LEC. Expression is recovered
from Step 03 raw, joined across quartiles with an outer gene union, and retained
unscaled for marker testing. Scaling and Harmony affect only PCA/neighbors/UMAP.
Harmony batch key: {run_summary['harmony_batch_key']}
Neighbors representation: {run_summary['neighbors_representation']}

Donor-support QC
----------------
{dominance_text}

Interpretation
--------------
Cell-level states and marker rankings are descriptive. Aging inference requires
donors as biological replicates. Donor-dominated or poorly supported states
must not be presented as general aging states without sensitivity analysis.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive.")
    if args.n_top_hvg <= 0 or args.n_pcs <= 1 or args.n_neighbors <= 1:
        raise ValueError("HVG/PCA/neighbors parameters must be positive and valid.")
    output_prefix = str(args.output_prefix).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_prefix):
        raise ValueError(
            "--output_prefix may contain only letters, numbers, underscores, and hyphens."
        )
    excluded_donors = {
        str(value).strip() for value in args.exclude_donors if str(value).strip()
    }

    project_dir = find_project_dir()
    allcell_root = resolve_path(args.allcell_root, project_dir)
    ec_root = resolve_path(args.ec_root, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    figure_dir = output_root / "figures"
    if not args.skip_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)
    output_h5ad = output_root / args.out_h5ad
    if output_h5ad.exists() and not args.overwrite:
        raise FileExistsError(
            f"Primary output already exists: {output_h5ad}. Use --overwrite to replace it."
        )

    print("Dataset13 NF aging Step 08: combine and recluster final LECs")
    print(f"Project directory: {project_dir}")
    print(f"Step 03 root: {allcell_root}")
    print(f"Step 06 root: {ec_root}")
    print(f"Output root: {output_root}")
    print(f"Leiden resolution: {args.resolution}")
    print(
        "Excluded donors before clustering: "
        + ("; ".join(sorted(excluded_donors)) if excluded_donors else "none")
    )
    print("State labels start from state0.")
    print("Gene join: OUTER union, using Step 03 raw expression.")

    subsets: list[ad.AnnData] = []
    input_summaries: list[dict[str, object]] = []
    genes_by_quartile: dict[str, set[str]] = {}
    found_excluded_donors: set[str] = set()
    n_excluded_cells = 0
    for quartile in args.quartiles:
        subset, summary, genes = load_quartile_lecs(
            quartile, allcell_root, ec_root
        )
        donor_values = normalized_text(subset.obs["donor_key"])
        found_here = set(donor_values.unique()).intersection(excluded_donors)
        found_excluded_donors.update(found_here)
        keep = ~donor_values.isin(excluded_donors)
        n_before_exclusion = int(subset.n_obs)
        n_removed = int((~keep).sum())
        if n_removed:
            subset = subset[keep.to_numpy()].copy()
            gc.collect()
        if subset.n_obs == 0:
            raise ValueError(
                f"Donor exclusion removed every LEC from {quartile}; cannot continue."
            )
        n_excluded_cells += n_removed
        summary["n_final_LEC_cells_before_donor_exclusion"] = n_before_exclusion
        summary["n_excluded_LEC_cells"] = n_removed
        summary["n_final_LEC_cells"] = int(subset.n_obs)
        summary["n_final_LEC_donors"] = int(subset.obs["donor_key"].nunique())
        subsets.append(subset)
        input_summaries.append(summary)
        genes_by_quartile[quartile] = genes

    missing_exclusions = excluded_donors.difference(found_excluded_donors)
    if missing_exclusions:
        raise ValueError(
            "Requested excluded donor(s) were not found in the Step 06 LEC inputs: "
            + "; ".join(sorted(missing_exclusions))
        )

    print(f"Combining {len(subsets)} quartile subsets with outer gene join...")
    combined = ad.concat(
        subsets,
        axis=0,
        join="outer",
        merge="first",
        uns_merge=None,
        label=None,
        index_unique=None,
        fill_value=0,
    )
    del subsets
    gc.collect()
    if not combined.obs_names.is_unique:
        raise ValueError("Combined cell IDs are not unique.")

    gene_presence = make_gene_presence_table(
        combined.var_names.astype(str), genes_by_quartile
    )
    coverage = gene_presence.set_index("gene").reindex(combined.var_names.astype(str))
    for column in coverage.columns:
        combined.var[column] = coverage[column].to_numpy()
    combined.var["gene_symbol"] = combined.var_names.astype(str)
    combined.var["outer_union_retained"] = True
    combined.raw = combined.copy()

    combined, embedding_params, state_mapping = compute_embedding_and_states(
        combined, args
    )
    state_color_map = dict(
        zip(state_mapping["LEC_state"], state_mapping["color_hex"])
    )
    tables = state_summary_tables(combined)

    marker_status = "skipped"
    marker_message = ""
    full_markers = pd.DataFrame()
    top_markers = pd.DataFrame()
    wide_markers = pd.DataFrame()
    if not args.skip_markers:
        try:
            print("Ranking state markers with retained unscaled expression...")
            full_markers, top_markers, wide_markers = rank_state_markers(
                combined, args.top_n_markers, gene_presence
            )
            marker_status = "ok"
        except Exception as error:
            marker_status = "failed"
            marker_message = f"{type(error).__name__}: {error}"
            warnings.warn(f"State marker ranking failed: {error}")

    # Tables needed for both interpretation and reviewer-facing QC.
    pd.DataFrame(input_summaries).to_csv(
        output_root / f"{output_prefix}_quartile_input_summary.csv", index=False
    )
    gene_presence.to_csv(
        output_root / f"{output_prefix}_gene_presence_by_quartile.csv", index=False
    )
    state_mapping.to_csv(
        output_root / f"{output_prefix}_state_name_mapping.csv", index=False
    )
    state_mapping[["LEC_state", "color_hex"]].to_csv(
        output_root / f"{output_prefix}_state_color_mapping.csv", index=False
    )
    tables["state_counts"].to_csv(
        output_root / f"{output_prefix}_state_counts.csv", index=False
    )
    tables["state_by_quartile"].to_csv(
        output_root / f"{output_prefix}_state_by_age_quartile.csv", index=False
    )
    tables["donor_state_counts"].to_csv(
        output_root / f"{output_prefix}_state_counts_by_donor.csv", index=False
    )
    tables["state_qc"].to_csv(
        output_root / f"{output_prefix}_state_donor_project_QC.csv", index=False
    )
    tables["donor_totals"].to_csv(
        output_root / f"{output_prefix}_donor_LEC_counts.csv", index=False
    )
    if "Project" in combined.obs.columns:
        pd.crosstab(combined.obs["age_quartile"], combined.obs["Project"]).to_csv(
            output_root / f"{output_prefix}_LEC_counts_age_quartile_by_project.csv"
        )
    if marker_status == "ok":
        full_markers.to_csv(
            output_root / f"{output_prefix}_state_marker_ranking_all_genes.csv.gz",
            index=False,
            compression="gzip",
        )
        top_markers.to_csv(
            output_root / f"{output_prefix}_state_top{args.top_n_markers}_markers_long.csv",
            index=False,
        )
        wide_markers.to_csv(
            output_root / f"{output_prefix}_state_top{args.top_n_markers}_markers_wide.csv",
            index=False,
        )

    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        figure_jobs = [
            (
                f"{output_prefix}_combined_LEC_umap_summary.png",
                "UMAP by state, quartile, exact age, project, sex, and Leiden",
                lambda path: save_umap_summary(
                    combined, path, args.dpi, args.resolution
                ),
            ),
            (
                f"{output_prefix}_combined_LEC_state_composition_by_quartile.png",
                "Cell-level state composition within each age quartile",
                lambda path: save_state_composition(
                    tables["state_by_quartile"], state_color_map, path, args.dpi
                ),
            ),
            (
                f"{output_prefix}_combined_LEC_state_support_QC.png",
                "State cell counts, donor support, and largest-donor contribution",
                lambda path: save_state_support(
                    tables["state_qc"], state_color_map, path, args.dpi
                ),
            ),
        ]
        if args.harmony_batch_key in combined.obs.columns:
            figure_jobs.append(
                (
                    f"{output_prefix}_project_batch_correction_comparison.png",
                    "Uncorrected and primary UMAPs colored by technical project",
                    lambda path: save_batch_comparison(
                        combined, args.harmony_batch_key, path, args.dpi
                    ),
                )
            )
        if marker_status == "ok" and not top_markers.empty:
            figure_jobs.append(
                (
                    f"{output_prefix}_combined_LEC_top_state_marker_dotplot.png",
                    "Top three marker genes per combined LEC state",
                    lambda path: save_marker_dotplot(combined, top_markers, path),
                )
            )
        for filename, description, function in figure_jobs:
            path = figure_dir / filename
            try:
                function(path)
                figure_manifest.append(
                    {"figure": str(path), "description": description, "status": "ok"}
                )
            except Exception as error:
                warnings.warn(f"Figure failed ({filename}): {error}")
                figure_manifest.append(
                    {
                        "figure": str(path),
                        "description": description,
                        "status": f"failed: {type(error).__name__}: {error}",
                    }
                )
        pd.DataFrame(figure_manifest).to_csv(
            output_root / f"{output_prefix}_figure_manifest.csv", index=False
        )

    combined.uns["step08_combine_info"] = {
        "script_version": SCRIPT_VERSION,
        "dataset": DATASET,
        "disease": REQUIRED_DISEASE,
        "quartiles": list(args.quartiles),
        "LEC_definition": "Step06 fresh EC-only CellTypist majority vote EC8_ln",
        "expression_source": "Step03 raw",
        "gene_join": "outer",
        "n_cells": int(combined.n_obs),
        "n_genes": int(combined.n_vars),
        "n_states": int(combined.obs["LEC_state"].nunique()),
        "state_name_start": 0,
        "leiden_resolution": float(args.resolution),
        "excluded_donors": sorted(excluded_donors),
        "n_excluded_LEC_cells": int(n_excluded_cells),
        "output_prefix": output_prefix,
    }
    make_obs_h5ad_safe(combined)
    combined.obs.reset_index(names="combined_cell_id").to_csv(
        output_root / f"{output_prefix}_combined_LEC_cell_metadata.csv.gz",
        index=False,
        compression="gzip",
    )
    print(f"Writing combined h5ad: {output_h5ad}")
    combined.write_h5ad(output_h5ad, compression="gzip")

    state_qc = tables["state_qc"]
    run_summary = {
        "script_version": SCRIPT_VERSION,
        "dataset": DATASET,
        "disease": REQUIRED_DISEASE,
        "status": "completed",
        "n_quartiles": len(args.quartiles),
        "n_final_LEC_cells": int(combined.n_obs),
        "n_final_LEC_donors": int(combined.obs["donor_key"].nunique()),
        "excluded_donors": ";".join(sorted(excluded_donors)),
        "n_excluded_LEC_cells": int(n_excluded_cells),
        "n_genes_outer_union": int(combined.n_vars),
        "n_genes_present_all_requested_quartiles": int(
            gene_presence["present_all_requested_quartiles"].sum()
        ),
        "n_states": int(combined.obs["LEC_state"].nunique()),
        "state_labels": ";".join(
            map(str, combined.obs["LEC_state"].cat.categories)
        ),
        "leiden_resolution": float(args.resolution),
        "harmony_batch_key": embedding_params["harmony_batch_key"],
        "harmony_status": embedding_params["harmony_status"],
        "neighbors_representation": embedding_params["neighbors_representation"],
        "n_hvg_used_for_embedding": embedding_params["n_hvg_used"],
        "n_pcs_used": embedding_params["n_pcs_used"],
        "n_neighbors_used": embedding_params["n_neighbors_used"],
        "n_states_donor_dominated_over_50pct": int(
            state_qc["donor_dominance_flag_over_50pct"].sum()
        ),
        "marker_status": marker_status,
        "marker_message": marker_message,
        "output_h5ad": str(output_h5ad),
    }
    pd.DataFrame([run_summary]).to_csv(
        output_root / f"{output_prefix}_run_summary.csv", index=False
    )
    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "allcell_root": str(allcell_root),
        "ec_root": str(ec_root),
        "output_root": str(output_root),
        "quartiles": args.quartiles,
        "gene_join": "outer",
        "expression_source": "Step03 raw",
        "n_top_hvg": args.n_top_hvg,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
        "resolution": args.resolution,
        "output_prefix": output_prefix,
        "excluded_donors": sorted(excluded_donors),
        "n_excluded_LEC_cells": int(n_excluded_cells),
        "random_state": args.random_state,
        "harmony_batch_key": args.harmony_batch_key,
        "harmony_enabled": not args.no_harmony,
        "top_n_markers": args.top_n_markers,
        "figures_enabled": not args.skip_figures,
    }
    with (output_root / f"{output_prefix}_run_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2)

    write_method_notes(
        output_root / f"{output_prefix}_method_and_interpretation_notes.txt",
        args,
        run_summary,
        state_qc,
    )

    print("\nDone.")
    print(f"Combined shape: {combined.n_obs:,} LECs x {combined.n_vars:,} genes")
    print(f"States: {run_summary['state_labels']}")
    print(f"Harmony: {run_summary['harmony_status']}")
    print(
        "Donor-dominated states (>50% from one donor): "
        f"{run_summary['n_states_donor_dominated_over_50pct']}"
    )
    print(f"Output: {output_h5ad}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
