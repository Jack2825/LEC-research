#!/usr/bin/env python3
r"""
Step 13.1: audit and QC the donor-replicated GSE271676 GeoMx disease data.

This replaces the former GSE135805 one-section-per-condition feasibility
script. GSE271676 is NanoString GeoMx WTA data: observations are segmented
areas of illumination (AOIs), not Visium spots or single cells. The primary
external-validation cohort is the CD31-enriched LV vessel compartment for
Control versus dilated cardiomyopathy (DCMP).

The step deliberately stops before biological hypothesis testing. It verifies
the DCC archive and three analysis tables, calculates negative-probe/LOQ QC,
reproduces upper-quartile (Q3) CPM scaling, audits the study design and mentor
targets, and writes a reusable H5AD. Later selection requires at least two of
CCL21, LYVE1, PDPN, FLT4, and PROX1 above the GeoMx LOQ. This step does not
perform a DCM-versus-Control test.

Default input
-------------
resources/spatial_transcriptomic/GSE271676_RAW/

Default output
--------------
new_result2_spatial_disease/13_1_spatial_disease_QC/
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import tarfile
from collections import OrderedDict
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_VERSION = "13_1_GSE271676_two_of_five_LEC_rule_2026_08_28"
DEFAULT_INPUT_ROOT = "resources/spatial_transcriptomic/GSE271676_RAW"
DEFAULT_OUTPUT = "new_result2_spatial_disease/13_1_spatial_disease_QC"
GEO_SERIES_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE271676"
MENDELEY_URL = "https://data.mendeley.com/datasets/gmgcdhmmdx/1"

PROBE_ANNOTATION_COLUMNS = [
    "ProbeName", "ProbeDisplayName", "TargetName", "HUGOSymbol", "Accessions",
    "GenomeBuild", "GenomicPosition", "AnalyteType", "CodeClass", "ProbePool",
    "TargetGroup", "GeneID",
]
REQUIRED_METADATA = [
    "Scan.name", "ROI..label.", "Segment..Name..Label.", "TMA.block",
    "Ventricle", "Segment.type", "PID", "Operation", "Origin_Structure",
    "Clinical_phenotype_path", "After_LVAD", "Clinical_phenotype_LV",
]
TARGET_GROUPS = OrderedDict(
    [
        ("LEC_selection_marker", ["CCL21", "LYVE1", "PDPN", "FLT4", "PROX1"]),
        ("homeostatic_LEC", ["MMRN1", "CCL21"]),
        ("disease_State0", ["ITGA9", "NRP2", "NPNT", "SEMA3D"]),
        ("disease_State3", ["PLSCR4", "CRIM1", "COLEC12", "PDGFC"]),
        ("activated_vascular_State5", ["FLT1", "ENG", "EPAS1"]),
        ("niche_ligand", ["BMP4"]),
        ("niche_receptor", ["BMPR1A", "BMPR2", "NRP1", "NRP2"]),
    ]
)


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and QC GSE271676 GeoMx data for DCM spatial validation."
    )
    parser.add_argument("--input_root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--robust_z_cutoff", type=float, default=3.5)
    parser.add_argument("--combined_caution_flags", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_dcc_validation", action="store_true")
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--skip_h5ad", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def find_initial_dataset(input_root: Path) -> Path:
    exact = input_root / "InitialDataset.csv"
    if exact.is_file():
        return exact
    candidates = sorted(input_root.glob("InitialDataset*.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Expected InitialDataset.csv or exactly one InitialDataset*.csv; "
            f"found {candidates}"
        )
    return candidates[0]


def aoi_keys(metadata: pd.DataFrame) -> pd.Index:
    roi = pd.to_numeric(metadata["ROI..label."], errors="raise").astype(int)
    return pd.Index(
        safe_text(metadata["Scan.name"])
        + " | " + roi.astype(str).str.zfill(3)
        + " | " + safe_text(metadata["Segment..Name..Label."]),
        name="aoi_id",
    )


def verify_dcc_tar(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty DCC archive: {path}")
    rows: list[dict[str, object]] = []
    with tarfile.open(path, mode="r") as archive:
        members = [x for x in archive.getmembers() if x.isfile()]
        for member in members:
            match = re.search(r"(GSM\d+)", Path(member.name).name)
            valid, uncompressed_bytes, version, error = False, 0, "", ""
            try:
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("tar member could not be opened")
                with gzip.GzipFile(fileobj=stream) as handle:
                    payload = handle.read()
                uncompressed_bytes = len(payload)
                header = payload[:2000].decode("utf-8", errors="replace")
                found = re.search(r'SoftwareVersion,"?([^"\r\n]+)', header)
                version = found.group(1) if found else ""
                valid = payload.startswith(b"<Header>") and uncompressed_bytes > 0
                if not valid:
                    error = "decompressed DCC lacks the expected header"
            except Exception as exc:
                error = str(exc)
            rows.append(
                {
                    "dcc_member": member.name,
                    "GSM": match.group(1) if match else "",
                    "compressed_bytes": int(member.size),
                    "uncompressed_bytes": int(uncompressed_bytes),
                    "gzip_and_DCC_header_valid": bool(valid),
                    "GeoMx_NGS_pipeline_version": version,
                    "error": error,
                }
            )
    inventory = pd.DataFrame(rows)
    if len(inventory) != 178:
        raise ValueError(f"Expected 178 DCC members; found {len(inventory)}")
    if not inventory["gzip_and_DCC_header_valid"].all():
        raise ValueError("One or more compressed DCC files failed integrity testing")
    if inventory["GSM"].eq("").any() or inventory["GSM"].duplicated().any():
        raise ValueError("DCC GSM accessions are missing or duplicated")
    return inventory


def robust_z(values: pd.Series) -> pd.Series:
    numeric = pd.Series(pd.to_numeric(values, errors="coerce"), index=values.index, dtype=float)
    median = float(numeric.median())
    mad = float((numeric - median).abs().median())
    if not np.isfinite(mad) or mad <= 0:
        sd = float(numeric.std(ddof=1))
        return (numeric - median) / sd if sd > 0 else pd.Series(0.0, index=values.index)
    return 0.6744897501960817 * (numeric - median) / mad


def q3_cpm(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce edgeR-style upper-quartile library scaling."""
    matrix = counts.to_numpy(dtype=float)
    library = matrix.sum(axis=0)
    if np.any(library <= 0):
        raise ValueError("A post-probe-QC AOI has a nonpositive library size")
    proportions = matrix / library[None, :]
    upper_quartile = np.quantile(proportions, 0.75, axis=0)
    if np.any(upper_quartile <= 0):
        raise ValueError("Upper-quartile normalization factor is nonpositive")
    geometric_mean = float(np.exp(np.mean(np.log(upper_quartile))))
    norm_factor = upper_quartile / geometric_mean
    effective_library = library * norm_factor
    normalized = matrix / effective_library[None, :] * 1_000_000.0
    factors = pd.DataFrame(
        {
            "aoi_id": counts.columns,
            "post_probe_QC_library_size": library,
            "upper_quartile_proportion": upper_quartile,
            "upper_quartile_norm_factor": norm_factor,
            "effective_library_size": effective_library,
        }
    ).set_index("aoi_id")
    return pd.DataFrame(normalized, index=counts.index, columns=counts.columns), factors


def target_registry(initial: pd.DataFrame, qc_genes: pd.Index) -> pd.DataFrame:
    endogenous = initial.loc[initial["CodeClass"].eq("Endogenous")]
    probe_counts = endogenous.groupby("TargetName", observed=True).size()
    roles: dict[str, list[str]] = {}
    for group, genes in TARGET_GROUPS.items():
        for gene in genes:
            roles.setdefault(gene, []).append(group)
    rows = []
    for gene, groups in roles.items():
        rows.append(
            {
                "gene": gene,
                "marker_groups": ";".join(groups),
                "n_endogenous_probes_initial": int(probe_counts.get(gene, 0)),
                "present_in_InitialDataset": gene in probe_counts.index,
                "passed_authors_1pct_probe_QC": gene in qc_genes,
                "primary_analysis_status": (
                    "available_post_probe_QC" if gene in qc_genes
                    else "raw_present_but_failed_authors_probe_QC"
                    if gene in probe_counts.index else "not_assayed"
                ),
                "use_in_two_of_five_LEC_selection": gene in TARGET_GROUPS["LEC_selection_marker"],
                "note": "CCL21 is also a homeostatic endpoint; selection-outcome overlap must be reported" if gene == "CCL21" else "",
            }
        )
    return pd.DataFrame(rows)


def design_membership(metadata: pd.DataFrame) -> pd.DataFrame:
    result = metadata.copy()
    lv_vessel = result["Ventricle"].eq("LV") & result["Segment.type"].eq("vessel")
    disease = result["Clinical_phenotype_LV"].isin(["Control", "DCMP"])
    result["analysis_primary_paper_LV_vessel"] = lv_vessel & disease
    result["analysis_strict_origin_LV_vessel"] = (
        result["analysis_primary_paper_LV_vessel"] & result["Origin_Structure"].eq("LV")
    )
    result["analysis_strict_noncardiac_control"] = (
        result["analysis_primary_paper_LV_vessel"]
        & result["Clinical_phenotype_path"].eq("Control")
    )
    result["analysis_pre_LVAD_sensitivity"] = (
        result["analysis_primary_paper_LV_vessel"] & ~safe_text(result["After_LVAD"]).eq("Y")
    )
    result["control_definition"] = np.select(
        [result["Clinical_phenotype_path"].eq("Control"), result["Clinical_phenotype_LV"].eq("Control")],
        ["strict_noncardiac_control", "histologically_normal_LV_extended_control"],
        default="not_control",
    )
    return result


def design_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    definitions = OrderedDict(
        [
            ("all_AOIs", pd.Series(True, index=metadata.index)),
            ("primary_paper_LV_vessel", metadata["analysis_primary_paper_LV_vessel"]),
            ("strict_origin_LV_vessel", metadata["analysis_strict_origin_LV_vessel"]),
            ("strict_noncardiac_control_only", metadata["analysis_strict_noncardiac_control"]),
            ("pre_LVAD_primary_sensitivity", metadata["analysis_pre_LVAD_sensitivity"]),
        ]
    )
    rows = []
    for analysis_set, mask in definitions.items():
        frame = metadata.loc[mask].copy()
        groups = ["Control"] if analysis_set == "strict_noncardiac_control_only" else list(
            dict.fromkeys(safe_text(frame["Clinical_phenotype_LV"]))
        )
        for group in groups:
            subset = frame.loc[safe_text(frame["Clinical_phenotype_LV"]).eq(group)]
            if subset.empty:
                continue
            rows.append(
                {
                    "analysis_set": analysis_set,
                    "Clinical_phenotype_LV": group,
                    "n_AOIs": int(len(subset)),
                    "n_donors": int(subset["PID"].nunique()),
                    "donors": ";".join(sorted(set(safe_text(subset["PID"])))),
                    "TMA_blocks": ";".join(sorted(set(safe_text(subset["TMA.block"])))),
                    "origin_structures": ";".join(sorted(set(safe_text(subset["Origin_Structure"])))),
                    "n_after_LVAD_AOIs": int(safe_text(subset["After_LVAD"]).eq("Y").sum()),
                }
            )
    return pd.DataFrame(rows)


def add_aoi_qc(
    metadata: pd.DataFrame,
    initial: pd.DataFrame,
    qc_counts: pd.DataFrame,
    q3_factors: pd.DataFrame,
    robust_z_cutoff: float,
    combined_caution_flags: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    negative = initial.loc[initial["CodeClass"].eq("Negative"), qc_counts.columns]
    negative_values = np.maximum(negative.to_numpy(dtype=float), 1.0)
    log_negative = np.log(negative_values)
    negative_geomean = np.exp(log_negative.mean(axis=0))
    negative_geosd = np.exp(log_negative.std(axis=0, ddof=1))
    # GeoMx NGS convention: two geometric SDs above negative-probe background.
    loq_per_probe = negative_geomean * np.square(negative_geosd)
    probe_number = (
        initial.loc[initial["CodeClass"].eq("Endogenous")]
        .groupby("TargetName", observed=True).size().reindex(qc_counts.index).astype(float)
    )
    target_loq = pd.DataFrame(
        probe_number.to_numpy()[:, None] * loq_per_probe[None, :],
        index=qc_counts.index,
        columns=qc_counts.columns,
    )
    above_loq = qc_counts.to_numpy(dtype=float) >= target_loq.to_numpy(dtype=float)
    endogenous = initial.loc[initial["CodeClass"].eq("Endogenous"), qc_counts.columns]

    result = metadata.copy().set_index("aoi_id", drop=False)
    result["raw_endogenous_probe_library_size"] = endogenous.sum(axis=0).reindex(result.index)
    result["negative_probe_geometric_mean"] = pd.Series(negative_geomean, index=qc_counts.columns).reindex(result.index)
    result["negative_probe_geometric_SD"] = pd.Series(negative_geosd, index=qc_counts.columns).reindex(result.index)
    result["LOQ_per_probe_geomean_times_geoSD2"] = pd.Series(loq_per_probe, index=qc_counts.columns).reindex(result.index)
    result = result.join(q3_factors, how="left")
    result["n_post_probe_QC_genes"] = int(qc_counts.shape[0])
    result["n_post_probe_QC_genes_above_LOQ"] = pd.Series(above_loq.sum(axis=0), index=qc_counts.columns).reindex(result.index)
    result["fraction_post_probe_QC_genes_above_LOQ"] = result["n_post_probe_QC_genes_above_LOQ"] / float(qc_counts.shape[0])
    result["qc_hard_fail"] = (
        ~np.isfinite(result["post_probe_QC_library_size"])
        | result["post_probe_QC_library_size"].le(0)
        | ~np.isfinite(result["LOQ_per_probe_geomean_times_geoSD2"])
        | result["LOQ_per_probe_geomean_times_geoSD2"].le(0)
    )
    pieces = []
    for _, frame in result.groupby("Segment.type", observed=True, sort=False):
        frame = frame.copy()
        frame["qc_robust_z_log_library_within_segment"] = robust_z(np.log1p(frame["post_probe_QC_library_size"]))
        frame["qc_robust_z_genes_above_LOQ_within_segment"] = robust_z(frame["n_post_probe_QC_genes_above_LOQ"])
        frame["qc_robust_z_log_negative_background_within_segment"] = robust_z(np.log1p(frame["negative_probe_geometric_mean"]))
        frame["qc_flag_low_library"] = frame["qc_robust_z_log_library_within_segment"].lt(-robust_z_cutoff)
        frame["qc_flag_low_genes_above_LOQ"] = frame["qc_robust_z_genes_above_LOQ_within_segment"].lt(-robust_z_cutoff)
        frame["qc_flag_high_negative_background"] = frame["qc_robust_z_log_negative_background_within_segment"].gt(robust_z_cutoff)
        pieces.append(frame)
    result = pd.concat(pieces).reindex(metadata["aoi_id"])
    flags = ["qc_flag_low_library", "qc_flag_low_genes_above_LOQ", "qc_flag_high_negative_background"]
    result["qc_caution_flag_count"] = result[flags].sum(axis=1).astype(int)
    result["qc_primary_keep"] = ~result["qc_hard_fail"]
    result["qc_sensitivity_keep"] = result["qc_primary_keep"] & result["qc_caution_flag_count"].lt(combined_caution_flags)
    result["qc_status"] = np.select(
        [result["qc_hard_fail"], ~result["qc_sensitivity_keep"]],
        ["hard_fail", "multiple_relative_cautions"], default="keep",
    )
    return result.reset_index(drop=True), target_loq


def build_target_detection(
    metadata: pd.DataFrame,
    registry: pd.DataFrame,
    raw_gene_counts: pd.DataFrame,
    qc_counts: pd.DataFrame,
    q3: pd.DataFrame,
    target_loq: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    indexed = metadata.set_index("aoi_id")
    for record in registry.itertuples(index=False):
        gene = record.gene
        raw_values = raw_gene_counts.loc[gene] if gene in raw_gene_counts.index else pd.Series(np.nan, index=qc_counts.columns)
        available = gene in qc_counts.index
        for aoi in qc_counts.columns:
            rows.append(
                {
                    "aoi_id": aoi,
                    "PID": indexed.at[aoi, "PID"],
                    "Clinical_phenotype_LV": indexed.at[aoi, "Clinical_phenotype_LV"],
                    "Ventricle": indexed.at[aoi, "Ventricle"],
                    "Origin_Structure": indexed.at[aoi, "Origin_Structure"],
                    "Segment.type": indexed.at[aoi, "Segment.type"],
                    "analysis_primary_paper_LV_vessel": indexed.at[aoi, "analysis_primary_paper_LV_vessel"],
                    "gene": gene,
                    "marker_groups": record.marker_groups,
                    "primary_QC_available": available,
                    "raw_initial_target_count": float(raw_values.get(aoi, np.nan)),
                    "post_probe_QC_count": float(qc_counts.at[gene, aoi]) if available else np.nan,
                    "target_LOQ": float(target_loq.at[gene, aoi]) if available else np.nan,
                    "above_LOQ": bool(qc_counts.at[gene, aoi] >= target_loq.at[gene, aoi]) if available else pd.NA,
                    "upper_quartile_CPM": float(q3.at[gene, aoi]) if available else np.nan,
                    "log2_upper_quartile_CPM_plus1": float(np.log2(q3.at[gene, aoi] + 1.0)) if available else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_target_detection(detection: pd.DataFrame) -> pd.DataFrame:
    frame = detection.loc[detection["analysis_primary_paper_LV_vessel"]].copy()
    rows = []
    for (group, gene, marker_groups), part in frame.groupby(
        ["Clinical_phenotype_LV", "gene", "marker_groups"], observed=True, sort=False
    ):
        available = bool(part["primary_QC_available"].all())
        valid = part.loc[part["above_LOQ"].notna()]
        n_above = int(valid["above_LOQ"].astype(bool).sum()) if available else 0
        rows.append(
            {
                "Clinical_phenotype_LV": group,
                "gene": gene,
                "marker_groups": marker_groups,
                "primary_QC_available": available,
                "n_AOIs": int(len(part)),
                "n_donors": int(part["PID"].nunique()),
                "n_AOIs_above_LOQ": n_above if available else pd.NA,
                "percent_AOIs_above_LOQ": 100.0 * n_above / len(valid) if available and len(valid) else np.nan,
                "median_log2_upper_quartile_CPM_plus1": float(part["log2_upper_quartile_CPM_plus1"].median()) if available else np.nan,
                "interpretation": "descriptive_AOI_summary_not_donor_inference",
            }
        )
    return pd.DataFrame(rows)


def pca_coordinates(log_q3: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    matrix = log_q3.to_numpy(dtype=float).T
    keep = np.isfinite(matrix.var(axis=0)) & (matrix.var(axis=0) > 0)
    centered = matrix[:, keep] - matrix[:, keep].mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[:2].T
    explained = singular**2 / np.sum(singular**2)
    result = metadata.copy()
    result["PC1"], result["PC2"] = scores[:, 0], scores[:, 1]
    result["PC1_variance_percent"] = float(explained[0] * 100)
    result["PC2_variance_percent"] = float(explained[1] * 100)
    return result


def plot_pca(pca: pd.DataFrame, path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for axis, (column, title) in zip(
        axes, [("Segment.type", "GeoMx segment"), ("Clinical_phenotype_LV", "LV clinical phenotype")]
    ):
        values = safe_text(pca[column])
        categories = list(dict.fromkeys(values))
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(categories), 1)))
        for category, color in zip(categories, colors):
            mask = values.eq(category)
            axis.scatter(pca.loc[mask, "PC1"], pca.loc[mask, "PC2"], s=34, alpha=0.8, label=category, color=color)
        axis.set_title(title)
        axis.set_xlabel(f"PC1 ({pca['PC1_variance_percent'].iloc[0]:.1f}%)")
        axis.set_ylabel(f"PC2 ({pca['PC2_variance_percent'].iloc[0]:.1f}%)")
        axis.axhline(0, color="0.85", linewidth=0.7)
        axis.axvline(0, color="0.85", linewidth=0.7)
        axis.legend(fontsize=7, frameon=False)
    figure.suptitle("GSE271676 log2 upper-quartile CPM: all 178 AOIs")
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_primary_qc(metadata: pd.DataFrame, path: Path, dpi: int) -> None:
    frame = metadata.loc[metadata["analysis_primary_paper_LV_vessel"]].copy()
    metrics = [
        ("post_probe_QC_library_size", "Post-probe-QC library size", True),
        ("negative_probe_geometric_mean", "Negative-probe geometric mean", True),
        ("LOQ_per_probe_geomean_times_geoSD2", "Per-probe LOQ", True),
        ("n_post_probe_QC_genes_above_LOQ", "Genes above LOQ", False),
    ]
    groups = ["Control", "DCMP"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    rng = np.random.default_rng(1301)
    for axis, (column, title, log_scale) in zip(axes.flat, metrics):
        arrays = [frame.loc[frame["Clinical_phenotype_LV"].eq(group), column].to_numpy(float) for group in groups]
        axis.boxplot(arrays, tick_labels=groups, showfliers=False)
        for position, values in enumerate(arrays, start=1):
            axis.scatter(position + rng.normal(0, 0.035, size=len(values)), values, color="#2f4b7c", s=28, alpha=0.8)
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Primary LV vessel AOI QC (6 Control, 10 DCMP)")
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_target_detection(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    gene_order = list(dict.fromkeys(gene for genes in TARGET_GROUPS.values() for gene in genes))
    matrix = summary.pivot(index="gene", columns="Clinical_phenotype_LV", values="percent_AOIs_above_LOQ").reindex(index=gene_order, columns=["Control", "DCMP"])
    figure, axis = plt.subplots(figsize=(6.5, max(8, 0.35 * len(gene_order))))
    masked = np.ma.masked_invalid(matrix.to_numpy(float))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    artist = axis.imshow(masked, aspect="auto", vmin=0, vmax=100, cmap=cmap)
    axis.set_xticks([0, 1], ["Control", "DCMP"])
    axis.set_yticks(range(len(gene_order)), gene_order)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            label = "QC fail" if pd.isna(value) else f"{value:.0f}%"
            axis.text(column, row, label, ha="center", va="center", fontsize=7, color="black" if pd.isna(value) or value < 55 else "white")
    figure.colorbar(artist, ax=axis).set_label("Primary LV vessel AOIs above LOQ (%)")
    axis.set_title("Mentor-target technical availability\n(descriptive, not a disease test)")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def build_h5ad(metadata: pd.DataFrame, qc_counts: pd.DataFrame, q3: pd.DataFrame, registry: pd.DataFrame) -> ad.AnnData:
    obs = metadata.copy().set_index("aoi_id", drop=False)
    # AnnData/HDF5 does not permit an index and retained column to share a name.
    obs.index = pd.Index(obs.index.astype(str).to_numpy(dtype=object), name="obs_id")
    for column in obs.columns:
        if obs[column].dtype == object or isinstance(obs[column].dtype, pd.StringDtype):
            obs[column] = safe_text(obs[column]).astype("object")
    var = pd.DataFrame(
        index=pd.Index(qc_counts.index.astype(str).to_numpy(dtype=object), name="gene")
    )
    var["TargetName"] = pd.Series(var.index.astype(str), index=var.index, dtype="object")
    indexed = registry.set_index("gene")
    var["mentor_marker_groups"] = indexed["marker_groups"].reindex(var.index).fillna("").astype("object")
    var["mentor_target"] = var["mentor_marker_groups"].ne("")
    raw_layer = sp.csr_matrix(qc_counts.T.to_numpy(dtype=np.float32))
    q3_layer = sp.csr_matrix(q3.T.to_numpy(dtype=np.float32))
    log_layer = sp.csr_matrix(np.log2(q3.T.to_numpy(dtype=np.float32) + 1.0))
    result = ad.AnnData(X=log_layer.copy(), obs=obs, var=var)
    result.layers["post_probe_QC_counts"] = raw_layer
    result.layers["upper_quartile_CPM"] = q3_layer
    result.layers["log2_upper_quartile_CPM_plus1"] = log_layer.copy()
    result.uns.update(
        {
            "dataset_id": "GSE271676",
            "platform": "NanoString_GeoMx_WTA",
            "observation_unit": "segmented_AOI_not_Visium_spot_or_single_cell",
            "script_version": SCRIPT_VERSION,
            "X_scale": "log2_upper_quartile_CPM_plus1",
            "primary_analysis": "LV_vessel_Control_vs_DCMP",
            "LEC_specificity_warning": "CD31-enriched vessel AOIs are not purified LECs",
            "binary_expression_gate_allowed": False,
            "formal_inference_completed": False,
            "source_GEO": GEO_SERIES_URL,
            "source_Mendeley": MENDELEY_URL,
        }
    )
    return result


def write_notes(path: Path, metadata: pd.DataFrame, registry: pd.DataFrame) -> None:
    primary = metadata.loc[metadata["analysis_primary_paper_LV_vessel"]]
    strict = metadata.loc[metadata["analysis_strict_origin_LV_vessel"]]
    failed = registry.loc[~registry["passed_authors_1pct_probe_QC"], "gene"].tolist()
    text = f"""Step 13.1 GSE271676 GeoMx disease audit and QC
==================================================

Replacement
-----------
This completely replaces the former GSE135805 Step 13.1. That dataset had one
normalized section per condition and supported only feasibility reporting.
GSE271676 has 178 GeoMx AOIs, repeated AOIs nested within donors, raw DCCs,
negative probes, post-probe-QC counts, and clinical/segment metadata.

Observation unit and scope
--------------------------
An observation is a segmented GeoMx AOI, not a Visium spot or a cell. Vessel
AOIs are CD31-enriched endothelial regions, not purified LECs. No continuous
tissue coordinates or local histology images are provided in these inputs, so
this branch cannot produce a Visium-like spatial map.

Primary design
--------------
Paper-matched LV vessel Control/DCMP AOIs: {len(primary)}.
Control: {int(primary['Clinical_phenotype_LV'].eq('Control').sum())} AOIs from {primary.loc[primary['Clinical_phenotype_LV'].eq('Control'), 'PID'].nunique()} donors.
DCMP: {int(primary['Clinical_phenotype_LV'].eq('DCMP').sum())} AOIs from {primary.loc[primary['Clinical_phenotype_LV'].eq('DCMP'), 'PID'].nunique()} donors.
Strict Origin_Structure=LV sensitivity: {len(strict)} AOIs. This removes the
AMC_020 IVS AOI. Strict noncardiac controls are three autopsy-control AOIs;
AMC_033 is used only in the paper-matched primary control definition.

QC and normalization
--------------------
InitialDataset negative probes define AOI-specific background. Per-probe LOQ
is negative-probe geometric mean multiplied by geometric SD squared; target
LOQ is scaled by target probe number. The authors' 1%-probe-QC matrix is the
primary expression input. Upper-quartile CPM is reproduced for descriptive
downstream scoring. No ComBat correction or biological test occurs here.

Mentor targets and LEC rule
---------------------------
All mentor targets are present in InitialDataset. Targets unavailable after
the authors' filter: {', '.join(failed) if failed else 'none'}. They cannot be
silently reintroduced into the primary analysis. Later steps select a vessel
AOI only when at least two of CCL21, LYVE1, PDPN, FLT4, and PROX1 are above the
AOI-specific GeoMx LOQ. No separate endothelial-expression panel is required.
CCL21 is also a homeostatic endpoint, so this selection-outcome overlap must be
reported in every biological interpretation.

Inference warning
-----------------
AOIs are nested within PID and are not donor replicates. DCM-versus-Control
testing must model PID/repeated AOIs and include strict-LV, strict-control, and
pre-LVAD sensitivities. This audit makes no final validation claim.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.robust_z_cutoff <= 0:
        raise ValueError("--robust_z_cutoff must be positive")
    if args.combined_caution_flags < 1 or args.combined_caution_flags > 3:
        raise ValueError("--combined_caution_flags must be between 1 and 3")

    project_dir = find_project_dir()
    input_root = resolve_path(args.input_root, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory is absent: {input_root}")
    sentinel = output_root / "13_1_run_summary.csv"
    if sentinel.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {sentinel}. Use --overwrite to replace it.")
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root = output_root / "figures"
    if not args.skip_figures:
        figure_root.mkdir(parents=True, exist_ok=True)

    initial_path = find_initial_dataset(input_root)
    qc_path = input_root / "Probe_QC_1percfilter.csv"
    metadata_path = input_root / "Spatial_Annotation.csv"
    dcc_path = input_root / "GSE271676_RAW.tar"
    for path in [initial_path, qc_path, metadata_path, dcc_path]:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty input: {path}")

    print("Step 13.1: GSE271676 GeoMx disease audit and QC")
    print("Reading:", initial_path)
    initial = pd.read_csv(initial_path, low_memory=False)
    print("Reading:", qc_path)
    qc_table = pd.read_csv(qc_path, low_memory=False)
    print("Reading:", metadata_path)
    metadata = pd.read_csv(metadata_path, low_memory=False)

    missing_probe = [x for x in PROBE_ANNOTATION_COLUMNS if x not in initial.columns]
    missing_metadata = [x for x in REQUIRED_METADATA if x not in metadata.columns]
    if missing_probe:
        raise KeyError(f"InitialDataset is missing columns: {missing_probe}")
    if missing_metadata:
        raise KeyError(f"Spatial_Annotation is missing columns: {missing_metadata}")
    if qc_table.columns[0] != "TargetName":
        raise KeyError("Probe_QC_1percfilter first column must be TargetName")
    expected_shapes = {(178, 52): metadata.shape, (18815, 190): initial.shape, (12801, 179): qc_table.shape}
    if metadata.shape != (178, 52) or initial.shape != (18815, 190) or qc_table.shape != (12801, 179):
        raise ValueError(f"Unexpected table shapes: annotation={metadata.shape}, initial={initial.shape}, probe_QC={qc_table.shape}")

    metadata["aoi_id"] = aoi_keys(metadata)
    if metadata["aoi_id"].duplicated().any():
        raise ValueError("Spatial annotation contains duplicated AOI keys")
    raw_aoi_columns = list(initial.columns[len(PROBE_ANNOTATION_COLUMNS):])
    qc_aoi_columns = list(qc_table.columns[1:])
    if set(metadata["aoi_id"]) != set(raw_aoi_columns) or set(metadata["aoi_id"]) != set(qc_aoi_columns):
        raise ValueError("AOI keys disagree across expression and annotation files")
    raw_aoi_columns = list(metadata["aoi_id"])
    initial.loc[:, raw_aoi_columns] = initial.loc[:, raw_aoi_columns].apply(pd.to_numeric, errors="raise")
    qc_table.loc[:, raw_aoi_columns] = qc_table.loc[:, raw_aoi_columns].apply(pd.to_numeric, errors="raise")
    qc_negative_summary = qc_table.loc[qc_table["TargetName"].eq("NegProbe-WTX")].copy()
    qc_gene_table = qc_table.loc[~qc_table["TargetName"].eq("NegProbe-WTX")].copy()
    if len(qc_negative_summary) != 1 or len(qc_gene_table) != 12800:
        raise ValueError(
            "Expected one NegProbe-WTX summary row and 12,800 biological genes "
            f"after probe QC; found {len(qc_negative_summary)} and {len(qc_gene_table)}"
        )
    raw_values = initial.loc[:, raw_aoi_columns].to_numpy(dtype=float)
    qc_values = qc_gene_table.loc[:, raw_aoi_columns].to_numpy(dtype=float)
    if not np.isfinite(raw_values).all() or not np.isfinite(qc_values).all():
        raise ValueError("Expression matrices contain nonfinite values")
    if (raw_values < 0).any() or (qc_values < 0).any():
        raise ValueError("Expression matrices contain negative values")
    if not np.allclose(raw_values, np.rint(raw_values), rtol=0, atol=1e-8) or not np.allclose(qc_values, np.rint(qc_values), rtol=0, atol=1e-8):
        raise ValueError("Expression matrices are not integer-like counts")
    if qc_table["TargetName"].duplicated().any():
        raise ValueError("Post-probe-QC TargetName values are not unique")

    endogenous = initial.loc[initial["CodeClass"].eq("Endogenous")].copy()
    negative = initial.loc[initial["CodeClass"].eq("Negative")].copy()
    if len(endogenous) != 18676 or len(negative) != 139:
        raise ValueError(f"Unexpected code classes: Endogenous={len(endogenous)}, Negative={len(negative)}")
    raw_gene_counts = endogenous.groupby("TargetName", observed=True)[raw_aoi_columns].sum()
    qc_counts = qc_gene_table.set_index("TargetName").loc[:, raw_aoi_columns]
    absent = qc_counts.index.difference(raw_gene_counts.index)
    if len(absent):
        raise ValueError(f"Post-QC genes absent from InitialDataset: {absent[:10].tolist()}")
    maximum_difference = float(np.max(np.abs(qc_counts.to_numpy(float) - raw_gene_counts.loc[qc_counts.index, raw_aoi_columns].to_numpy(float))))
    if maximum_difference != 0:
        raise ValueError(f"Post-QC counts do not match InitialDataset; maximum difference={maximum_difference}")

    print("Validating 178 compressed DCC files...")
    if args.skip_dcc_validation:
        dcc_inventory = pd.DataFrame(columns=[
            "dcc_member", "GSM", "compressed_bytes", "uncompressed_bytes",
            "gzip_and_DCC_header_valid", "GeoMx_NGS_pipeline_version", "error",
        ])
    else:
        dcc_inventory = verify_dcc_tar(dcc_path)
    dcc_inventory.to_csv(output_root / "13_1_DCC_inventory.csv", index=False)

    q3, q3_factors = q3_cpm(qc_counts)
    registry = target_registry(initial, qc_counts.index)
    if not registry["present_in_InitialDataset"].all():
        raise ValueError("One or more mentor targets are absent from InitialDataset")
    metadata = design_membership(metadata)
    metadata, target_loq = add_aoi_qc(
        metadata, initial, qc_counts, q3_factors,
        args.robust_z_cutoff, args.combined_caution_flags,
    )
    detection = build_target_detection(metadata, registry, raw_gene_counts, qc_counts, q3, target_loq)
    detection_summary = summarize_target_detection(detection)
    designs = design_summary(metadata)
    log_q3 = np.log2(q3 + 1.0)
    pca = pca_coordinates(log_q3, metadata)

    validation = pd.DataFrame([{
        "dataset": "GSE271676", "platform": "NanoString GeoMx WTA",
        "initial_file": str(initial_path), "probe_QC_file": str(qc_path),
        "annotation_file": str(metadata_path), "DCC_tar": str(dcc_path),
        "n_DCC_files": int(len(dcc_inventory)) if not args.skip_dcc_validation else 178,
        "all_DCC_gzip_and_headers_valid": bool(dcc_inventory["gzip_and_DCC_header_valid"].all()) if len(dcc_inventory) else pd.NA,
        "n_AOIs": int(len(metadata)), "n_annotation_columns": 52,
        "n_initial_rows": int(len(initial)), "n_endogenous_probes": int(len(endogenous)),
        "n_negative_probes": int(len(negative)),
        "NegProbe_WTX_summary_row_present": True,
        "n_post_probe_QC_table_rows_including_NegProbe_WTX": int(len(qc_table)),
        "n_post_probe_QC_genes": int(len(qc_counts)),
        "AOI_keys_exactly_aligned": True, "post_QC_counts_exactly_match_initial": maximum_difference == 0,
        "matrices_finite_nonnegative_integer_like": True, "resource_inputs_modified": False,
    }])
    validation.to_csv(output_root / "13_1_input_validation.csv", index=False)
    registry.to_csv(output_root / "13_1_DCM_target_registry.csv", index=False)
    metadata.to_csv(output_root / "13_1_AOI_metadata_and_QC.csv", index=False)
    metadata.loc[metadata["analysis_primary_paper_LV_vessel"]].to_csv(
        output_root / "13_1_primary_LV_vessel_Control_DCMP_design.csv", index=False
    )
    designs.to_csv(output_root / "13_1_design_summary.csv", index=False)
    q3_factors.reset_index().to_csv(output_root / "13_1_upper_quartile_normalization_factors.csv", index=False)
    detection.to_csv(output_root / "13_1_mentor_target_detection_by_AOI.csv.gz", index=False, compression="gzip")
    detection_summary.to_csv(output_root / "13_1_mentor_target_detection_summary.csv", index=False)
    pca.to_csv(output_root / "13_1_AOI_PCA_coordinates.csv", index=False)

    h5ad_path = output_root / "13_1_GSE271676_GeoMx_Q3_expression.h5ad"
    if not args.skip_h5ad:
        build_h5ad(metadata, qc_counts, q3, registry).write_h5ad(h5ad_path, compression="gzip")
        print("Wrote:", h5ad_path)

    figure_rows = []
    if not args.skip_figures:
        pca_path = figure_root / "13_1_all_AOI_upper_quartile_PCA.png"
        plot_pca(pca, pca_path, args.dpi)
        figure_rows.append({"figure": str(pca_path), "description": "All-AOI PCA by segment and phenotype"})
        qc_figure = figure_root / "13_1_primary_LV_vessel_QC.png"
        plot_primary_qc(metadata, qc_figure, args.dpi)
        figure_rows.append({"figure": str(qc_figure), "description": "Control/DCMP LV vessel AOI QC"})
        target_figure = figure_root / "13_1_primary_LV_vessel_target_detection.png"
        plot_target_detection(detection_summary, target_figure, args.dpi)
        figure_rows.append({"figure": str(target_figure), "description": "Mentor targets above LOQ"})
    pd.DataFrame(figure_rows, columns=["figure", "description"]).to_csv(output_root / "13_1_figure_manifest.csv", index=False)

    write_notes(output_root / "13_1_method_and_limitations.txt", metadata, registry)
    run_config = {
        "script_version": SCRIPT_VERSION, "project_dir": str(project_dir),
        "input_root": str(input_root),
        "input_files": {"InitialDataset": str(initial_path), "Probe_QC_1percfilter": str(qc_path), "Spatial_Annotation": str(metadata_path), "DCC_tar": str(dcc_path)},
        "output_root": str(output_root),
        "primary_filter": {"Ventricle": "LV", "Segment.type": "vessel", "Clinical_phenotype_LV": ["Control", "DCMP"]},
        "strict_origin_sensitivity": "Origin_Structure == LV",
        "strict_control_sensitivity": "Clinical_phenotype_path == Control",
        "pre_LVAD_sensitivity": "After_LVAD != Y",
        "normalization": "edgeR_style_upper_quartile_CPM",
        "LOQ": "negative_probe_geomean_times_geometric_SD_squared_scaled_by_target_probe_count",
        "LEC_selection_markers": TARGET_GROUPS["LEC_selection_marker"],
        "LEC_selection_rule": "at_least_2_of_5_markers_above_AOI_specific_LOQ",
        "CCL21_included_in_selection": True,
        "selection_outcome_overlap_warning": "CCL21_is_also_a_homeostatic_endpoint",
        "formal_inference_completed": False,
        "resource_inputs_modified": False, "DCC_validation_skipped": args.skip_dcc_validation,
        "figures_enabled": not args.skip_figures, "h5ad_written": not args.skip_h5ad,
        "h5ad": str(h5ad_path) if not args.skip_h5ad else "",
        "source_GEO": GEO_SERIES_URL, "source_Mendeley": MENDELEY_URL,
    }
    (output_root / "13_1_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    primary = metadata.loc[metadata["analysis_primary_paper_LV_vessel"]]
    strict = metadata.loc[metadata["analysis_strict_origin_LV_vessel"]]
    run_summary = pd.DataFrame([{
        "script_version": SCRIPT_VERSION, "status": "completed", "dataset": "GSE271676",
        "platform": "NanoString_GeoMx_WTA",
        "n_DCC_files_valid": int(dcc_inventory["gzip_and_DCC_header_valid"].sum()) if len(dcc_inventory) else pd.NA,
        "n_all_AOIs": int(len(metadata)), "n_all_donors": int(metadata["PID"].nunique()),
        "n_post_probe_QC_genes": int(len(qc_counts)), "n_primary_LV_vessel_AOIs": int(len(primary)),
        "n_primary_Control_AOIs": int(primary["Clinical_phenotype_LV"].eq("Control").sum()),
        "n_primary_Control_donors": int(primary.loc[primary["Clinical_phenotype_LV"].eq("Control"), "PID"].nunique()),
        "n_primary_DCMP_AOIs": int(primary["Clinical_phenotype_LV"].eq("DCMP").sum()),
        "n_primary_DCMP_donors": int(primary.loc[primary["Clinical_phenotype_LV"].eq("DCMP"), "PID"].nunique()),
        "n_strict_origin_LV_AOIs": int(len(strict)),
        "n_strict_noncardiac_Control_AOIs": int(metadata["analysis_strict_noncardiac_control"].sum()),
        "n_targets_raw_present": int(registry["present_in_InitialDataset"].sum()),
        "n_targets_post_probe_QC_available": int(registry["passed_authors_1pct_probe_QC"].sum()),
        "targets_failed_post_probe_QC": ";".join(registry.loc[~registry["passed_authors_1pct_probe_QC"], "gene"]),
        "n_qc_hard_fail_AOIs": int(metadata["qc_hard_fail"].sum()),
        "n_qc_sensitivity_keep_AOIs": int(metadata["qc_sensitivity_keep"].sum()),
        "formal_DCM_vs_Control_inference_completed": False,
        "current_safe_use": "GeoMx_AOI_QC_design_and_target_availability",
        "resource_inputs_modified": False, "output_root": str(output_root),
    }])
    run_summary.to_csv(sentinel, index=False)
    print("\nStep 13.1 completed.")
    print(run_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
