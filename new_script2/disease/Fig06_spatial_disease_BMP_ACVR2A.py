#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Figure 6: independent spatial assessment of the BMP4-ACVR2A axis.

The upstream disease MultiNicheNet analysis (Figure 5) prioritized an
endocardial-EC -> BMP4 -> ACVR2A -> LEC interaction in DCM.  This script uses
the independently processed GSE271676 GeoMx WTA resource to ask what part of
that hypothesis can be assessed spatially.

Two analysis levels are kept separate:

1. A donor-level focused TGF-beta/BMP/activin-family screen in all 16
   Control/DCMP LV vessel AOIs (4 Control and 7 DCMP donors).
2. A descriptive overlay in the seven AOIs satisfying the prespecified
   two-of-five LEC-marker rule (2 Control and 3 DCMP donors).

ACVR2A and ACVR2B were assayed in the raw WTA panel but were detected above
LOQ in only 1/178 AOIs and were excluded by the authors' 1% probe-QC filter.
Their raw counts are used only to explain assay availability; they are never
rescued for expression testing.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402


SCRIPT_VERSION = "Fig06_spatial_disease_BMP_ACVR2A_v2_2026_09_01"
DEFAULT_ALL_H5AD = (
    "new_result2_spatial_disease/13_1_spatial_disease_QC/"
    "13_1_GSE271676_GeoMx_Q3_expression.h5ad"
)
DEFAULT_SELECTED_H5AD = (
    "new_result2_spatial_disease/13_2_disease_LEC_reference/"
    "13_2_GSE271676_two_of_five_LEC_selected.h5ad"
)
DEFAULT_INITIAL = "resources/spatial_transcriptomic/GSE271676_RAW/InitialDataset.csv"
DEFAULT_REACTOME = "resources/gene_sets/c2.cp.reactome.v2026.1.Hs.symbols.gmt"
DEFAULT_UPSTREAM = (
    "new_result2/disease/Fig05D_disease_communication_modules/"
    "Fig05D_source_data.csv"
)
DEFAULT_OUTPUT = "new_result2/disease/Fig06_spatial_disease_BMP_ACVR2A"

EXPRESSION_LAYER = "log2_upper_quartile_CPM_plus1"
COUNT_LAYER = "post_probe_QC_counts"
LOQ_COLUMN = "LOQ_per_probe_geomean_times_geoSD2"
GROUP_ORDER = ["Control", "DCMP"]
LEC_MARKERS = ["CCL21", "LYVE1", "PDPN", "FLT4", "PROX1"]
LEC_RULE = "at least 2 of CCL21, LYVE1, PDPN, FLT4, PROX1 above AOI-specific LOQ"

NAVY = "#243B6B"
BLUE = "#4C78A8"
TEAL = "#178F89"
CORAL = "#E76F51"
GOLD = "#D4A72C"
SLATE = "#667085"
MID_GREY = "#98A2B3"
LIGHT_GREY = "#E6EAF0"
VERY_LIGHT = "#F6F8FA"
TEXT = "#202428"
GROUP_COLORS = {"Control": BLUE, "DCMP": CORAL}


# The family panel was fixed from direct ligand/receptor/SMAD components of
# Reactome Signaling by BMP, Signaling by Activin, and Signaling by TGF-beta
# receptor complex, supplemented by the mentor/upstream genes (BMP4, ACVR2A,
# ENG, NRP1/2) and canonical ID readouts.  It is intentionally narrower than
# the 161-gene umbrella Reactome family pathway.
FAMILY_GROUPS = OrderedDict(
    [
        (
            "Ligands",
            [
                "BMP2", "BMP4", "BMP5", "BMP6", "BMP7", "BMP10", "GDF2",
                "GDF11", "INHA", "INHBA", "INHBB", "TGFB1", "TGFB2", "TGFB3",
            ],
        ),
        (
            "Receptors / co-receptors",
            [
                "ACVR1", "ACVR1B", "ACVR1C", "ACVR2A", "ACVR2B", "ACVRL1",
                "BMPR1A", "BMPR1B", "BMPR2", "TGFBR1", "TGFBR2", "TGFBR3",
                "ENG", "NRP1", "NRP2",
            ],
        ),
        (
            "SMAD / readout / feedback",
            [
                "SMAD1", "SMAD2", "SMAD3", "SMAD4", "SMAD5", "SMAD6",
                "SMAD7", "SMAD9", "ID1", "ID2", "ID3", "BAMBI", "PMEPA1",
                "SERPINE1", "NOG", "FST", "FSTL1", "FSTL3",
            ],
        ),
    ]
)
FAMILY_GENES = [gene for genes in FAMILY_GROUPS.values() for gene in genes]
GENE_ROLE = {
    gene: group for group, genes in FAMILY_GROUPS.items() for gene in genes
}

# Compact display subset; the complete 47-gene audit remains in source data.
DISPLAY_GROUPS = OrderedDict(
    [
        (
            "Ligands",
            [
                "BMP2", "BMP4", "BMP5", "BMP6", "GDF2", "INHBA", "INHBB",
                "TGFB1", "TGFB2", "TGFB3",
            ],
        ),
        (
            "Receptors",
            [
                "ACVR2A", "ACVR2B", "ACVR1", "ACVR1B", "ACVRL1", "BMPR1A",
                "BMPR2", "TGFBR1", "TGFBR2", "TGFBR3",
            ],
        ),
        ("Co-receptors", ["ENG", "NRP1", "NRP2"]),
        (
            "SMAD / readouts",
            [
                "SMAD1", "SMAD5", "SMAD9", "SMAD2", "SMAD3", "SMAD4",
                "SMAD6", "SMAD7", "ID1", "ID2", "ID3",
            ],
        ),
    ]
)
DISPLAY_GENES = [gene for genes in DISPLAY_GROUPS.values() for gene in genes]
HEATMAP_GENES = ["BMP4", "BMPR2", "TGFBR2", "ENG", "SMAD5", "SMAD7", "ID1", "ID2", "ID3"]
PREFERRED_EFFECT_ORDER = [
    "BMP4", "TGFB1", "TGFB3", "ACVRL1", "BMPR2", "TGFBR2", "ENG", "NRP1",
    "SMAD2", "SMAD4", "SMAD5", "SMAD7", "ID1", "ID2", "ID3", "FSTL1", "FSTL3",
]

REACTOME_SETS = [
    "REACTOME_SIGNALING_BY_BMP",
    "REACTOME_SIGNALING_BY_ACTIVIN",
    "REACTOME_SIGNALING_BY_TGF_BETA_RECEPTOR_COMPLEX",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Figure 6 spatial disease BMP/ACVR2A validation panels."
    )
    parser.add_argument("--all_h5ad", default=DEFAULT_ALL_H5AD)
    parser.add_argument("--selected_h5ad", default=DEFAULT_SELECTED_H5AD)
    parser.add_argument("--initial_dataset", default=DEFAULT_INITIAL)
    parser.add_argument("--reactome_gmt", default=DEFAULT_REACTOME)
    parser.add_argument("--upstream_figure5_source", default=DEFAULT_UPSTREAM)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--screen_detection_fraction", type=float, default=0.50)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--render-only", action="store_true",
        help="Redraw the combined figure from its existing source CSVs; skip analysis.",
    )
    parser.add_argument(
        "--export_individual_panels", action="store_true",
        help="Opt in to legacy one-file-per-panel exports; off by default.",
    )
    return parser.parse_args()


def find_project_root() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype("string").fillna("").str.lower().isin({"true", "1", "yes", "y"})


def dense_matrix(matrix: object) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def dense_layer(adata: ad.AnnData, layer: str) -> pd.DataFrame:
    if layer not in adata.layers:
        raise KeyError(f"Required layer is absent: {layer}")
    values = dense_matrix(adata.layers[layer]).astype(float)
    if values.shape != adata.shape or not np.isfinite(values).all():
        raise ValueError(f"Invalid layer {layer}: {values.shape} versus {adata.shape}")
    return pd.DataFrame(
        values,
        index=pd.Index(adata.obs_names.astype(str), name="aoi_id"),
        columns=pd.Index(adata.var_names.astype(str), name="gene"),
    )


def read_reactome_membership(path: Path) -> dict[str, set[str]]:
    requested = set(REACTOME_SETS)
    result: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields and fields[0] in requested:
                result[fields[0]] = set(fields[2:])
    missing = requested.difference(result)
    if missing:
        raise ValueError(f"Missing requested Reactome sets: {sorted(missing)}")
    return result


def read_upstream_hypothesis(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    required = {"disease", "sender", "sender_label", "ligand", "receptor", "receiver", "prioritization_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Figure 5 source lacks columns: {sorted(missing)}")
    hit = frame.loc[
        frame["disease"].astype(str).eq("DCM")
        & frame["ligand"].astype(str).eq("BMP4")
        & frame["receptor"].astype(str).eq("ACVR2A")
        & frame["receiver"].astype(str).eq("LEC_EC8_ln")
    ]
    if len(hit) != 1:
        raise ValueError(f"Expected one DCM BMP4-ACVR2A Figure 5 axis, found {len(hit)}")
    row = hit.iloc[0]
    return {
        "sender": str(row["sender"]),
        "sender_label": str(row["sender_label"]),
        "ligand": "BMP4",
        "receptor": "ACVR2A",
        "receiver": "LEC_EC8_ln",
        "receiver_label": "LECs",
        "prioritization_score": float(row["prioritization_score"]),
        "source": str(path),
        "evidence_type": "MultiNicheNet prioritized interaction; not a spatial validation result",
    }


def donor_tables(
    expression: pd.DataFrame,
    detection: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annotation = pd.DataFrame(
        {
            "PID": metadata["PID"].astype(str).to_numpy(),
            "group": metadata["Clinical_phenotype_LV"].astype(str).to_numpy(),
        },
        index=expression.index,
    )
    donor_expression = expression.join(annotation).groupby(["PID", "group"], observed=True).mean()
    donor_detection = detection.join(annotation).groupby(["PID", "group"], observed=True).any()
    donor_n_aoi = annotation.groupby(["PID", "group"], observed=True).size().rename("n_AOIs")
    return donor_expression, donor_detection, donor_n_aoi.reset_index()


def make_cohort(
    adata: ad.AnnData,
    mask: pd.Series | np.ndarray,
    probe_counts: pd.Series,
) -> dict[str, object]:
    bool_mask = np.asarray(mask, dtype=bool)
    subset = adata[bool_mask].copy()
    groups = set(subset.obs["Clinical_phenotype_LV"].astype(str))
    if not groups.issubset(set(GROUP_ORDER)):
        raise ValueError(f"Unexpected phenotype in analysis cohort: {sorted(groups)}")
    expression = dense_layer(subset, EXPRESSION_LAYER)
    counts = dense_layer(subset, COUNT_LAYER)
    available = [gene for gene in FAMILY_GENES if gene in expression.columns]
    loq = subset.obs[LOQ_COLUMN].astype(float).reindex(expression.index)
    thresholds = np.outer(
        loq.to_numpy(), probe_counts.reindex(available).astype(float).to_numpy()
    )
    detection = pd.DataFrame(
        counts[available].to_numpy() >= thresholds,
        index=expression.index,
        columns=available,
    )
    donor_expression, donor_detection, donor_n_aoi = donor_tables(
        expression[available], detection, subset.obs
    )
    return {
        "adata": subset,
        "obs": subset.obs.copy(),
        "expression": expression,
        "counts": counts,
        "detection": detection,
        "donor_expression": donor_expression,
        "donor_detection": donor_detection,
        "donor_n_aoi": donor_n_aoi,
    }


def exact_label_permutation(values: np.ndarray, labels: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=str)
    control = labels == "Control"
    n_control = int(control.sum())
    n_dcmp = int((labels == "DCMP").sum())
    if n_control < 1 or n_dcmp < 1:
        return float("nan"), float("nan"), 0
    observed = float(values[~control].mean() - values[control].mean())
    null = []
    for indices in itertools.combinations(range(len(values)), n_control):
        candidate_control = np.zeros(len(values), dtype=bool)
        candidate_control[list(indices)] = True
        null.append(float(values[~candidate_control].mean() - values[candidate_control].mean()))
    null_array = np.asarray(null, dtype=float)
    p_value = float(np.mean(np.abs(null_array) >= abs(observed) - 1e-12))
    return observed, p_value, int(len(null_array))


def bh_fdr(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    ordered_index = valid.sort_values().index
    ordered = valid.loc[ordered_index].to_numpy()
    adjusted = ordered * len(ordered) / np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered_index] = np.minimum(adjusted, 1.0)
    return result


def assay_audit(
    initial: pd.DataFrame,
    all_adata: ad.AnnData,
    reactome: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.Series]:
    endogenous = initial.loc[initial["CodeClass"].astype(str).eq("Endogenous")].copy()
    probe_counts = endogenous.groupby("TargetName", observed=True).size().astype(int)
    post_qc = set(all_adata.var_names.astype(str))
    aoi_columns = list(all_adata.obs_names.astype(str))
    missing_columns = set(aoi_columns).difference(initial.columns)
    if missing_columns:
        raise KeyError(f"InitialDataset lacks {len(missing_columns)} AOI columns")
    loq = all_adata.obs[LOQ_COLUMN].astype(float).reindex(aoi_columns)
    primary_mask = as_bool(all_adata.obs["analysis_primary_paper_LV_vessel"])
    primary_ids = list(all_adata.obs_names[primary_mask].astype(str))

    rows = []
    for gene in FAMILY_GENES:
        raw_rows = endogenous.loc[endogenous["TargetName"].astype(str).eq(gene)]
        raw_present = not raw_rows.empty
        raw_detection = pd.Series(False, index=aoi_columns, dtype=bool)
        target_group = ""
        if raw_present:
            raw_counts = raw_rows[aoi_columns].apply(pd.to_numeric, errors="raise").sum(axis=0)
            raw_detection = raw_counts.ge(loq * len(raw_rows))
            target_group = ";".join(sorted(set(raw_rows["TargetGroup"].astype(str))))
        memberships = [name for name, genes in reactome.items() if gene in genes]
        rows.append(
            {
                "gene": gene,
                "pathway_group": GENE_ROLE[gene],
                "n_initial_endogenous_probes": int(len(raw_rows)),
                "present_in_InitialDataset": raw_present,
                "raw_TargetGroup": target_group,
                "n_raw_AOIs_above_LOQ_all_178": int(raw_detection.sum()) if raw_present else 0,
                "percent_raw_AOIs_above_LOQ_all_178": float(100 * raw_detection.mean()) if raw_present else 0.0,
                "n_raw_primary_AOIs_above_LOQ": int(raw_detection.reindex(primary_ids).sum()) if raw_present else 0,
                "present_after_authors_1pct_probe_QC": gene in post_qc,
                "allowed_for_expression_inference": gene in post_qc,
                "reactome_direct_membership": ";".join(memberships),
                "mentor_or_upstream_priority": gene in {"BMP4", "ACVR2A", "ENG", "NRP1", "NRP2"},
                "reason_if_unavailable": (
                    "raw probe present but excluded by authors' 1% probe-QC filter; raw counts used only for assay audit"
                    if raw_present and gene not in post_qc
                    else "not assayed"
                    if not raw_present
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows), probe_counts


def detection_summary(
    cohort_name: str,
    cohort: dict[str, object],
    audit: pd.DataFrame,
) -> pd.DataFrame:
    metadata = cohort["obs"]
    detection = cohort["detection"]
    donor_detection = cohort["donor_detection"]
    donor_expression = cohort["donor_expression"]
    rows = []
    audit_lookup = audit.set_index("gene")
    for gene in FAMILY_GENES:
        audit_row = audit_lookup.loc[gene]
        post_qc = bool(audit_row["present_after_authors_1pct_probe_QC"])
        record: dict[str, object] = {
            "cohort": cohort_name,
            "gene": gene,
            "pathway_group": GENE_ROLE[gene],
            "n_raw_AOIs_above_LOQ_all_178": int(audit_row["n_raw_AOIs_above_LOQ_all_178"]),
            "n_raw_primary_AOIs_above_LOQ": int(audit_row["n_raw_primary_AOIs_above_LOQ"]),
            "assay_status": (
                "available_post_probe_QC"
                if post_qc
                else "raw_present_but_failed_authors_probe_QC"
                if bool(audit_row["present_in_InitialDataset"])
                else "not_assayed"
            ),
            "n_AOIs": int(len(metadata)),
            "n_donors": int(metadata["PID"].astype(str).nunique()),
        }
        if post_qc:
            gene_detection = detection[gene]
            gene_donor_detection = donor_detection[gene]
            record.update(
                {
                    "n_AOIs_above_LOQ": int(gene_detection.sum()),
                    "percent_AOIs_above_LOQ": float(100.0 * gene_detection.mean()),
                    "n_donors_with_any_AOI_above_LOQ": int(gene_donor_detection.sum()),
                    "percent_donors_with_any_AOI_above_LOQ": float(100.0 * gene_donor_detection.mean()),
                }
            )
            for group in GROUP_ORDER:
                aoi_mask = metadata["Clinical_phenotype_LV"].astype(str).eq(group).to_numpy()
                donor_mask = donor_expression.index.get_level_values("group") == group
                record[f"{group}_n_AOIs"] = int(aoi_mask.sum())
                record[f"{group}_n_AOIs_above_LOQ"] = int(gene_detection.iloc[aoi_mask].sum())
                record[f"{group}_n_donors"] = int(donor_mask.sum())
                record[f"{group}_n_donors_above_LOQ"] = int(gene_donor_detection.iloc[donor_mask].sum())
                record[f"{group}_donor_mean_expression"] = float(donor_expression.iloc[donor_mask][gene].mean())
            record["DCMP_minus_Control_donor_mean_log2_Q3_CPM"] = (
                record["DCMP_donor_mean_expression"] - record["Control_donor_mean_expression"]
            )
        rows.append(record)
    return pd.DataFrame(rows)


def add_primary_inference(
    summary: pd.DataFrame,
    cohort: dict[str, object],
    detection_fraction: float,
) -> pd.DataFrame:
    result = summary.copy()
    donor_expression = cohort["donor_expression"]
    labels = donor_expression.index.get_level_values("group").to_numpy(dtype=str)
    n_control = int((labels == "Control").sum())
    n_dcmp = int((labels == "DCMP").sum())
    min_control = math.ceil(n_control * detection_fraction)
    min_dcmp = math.ceil(n_dcmp * detection_fraction)
    result["screen_detection_fraction_per_group"] = detection_fraction
    result["minimum_Control_donors_above_LOQ"] = min_control
    result["minimum_DCMP_donors_above_LOQ"] = min_dcmp
    result["screen_eligible"] = (
        result["assay_status"].eq("available_post_probe_QC")
        & result["Control_n_donors_above_LOQ"].fillna(0).ge(min_control)
        & result["DCMP_n_donors_above_LOQ"].fillna(0).ge(min_dcmp)
        & (n_control >= 3)
        & (n_dcmp >= 3)
    )
    result["exact_two_sided_permutation_p"] = np.nan
    result["n_exact_label_allocations"] = pd.NA
    for index, row in result.loc[result["screen_eligible"]].iterrows():
        effect, p_value, n_allocations = exact_label_permutation(
            donor_expression[row["gene"]].to_numpy(), labels
        )
        result.at[index, "DCMP_minus_Control_donor_mean_log2_Q3_CPM"] = effect
        result.at[index, "exact_two_sided_permutation_p"] = p_value
        result.at[index, "n_exact_label_allocations"] = n_allocations
    result["BH_FDR_q"] = bh_fdr(result["exact_two_sided_permutation_p"])
    result["FDR_lt_0.05"] = result["BH_FDR_q"].lt(0.05).fillna(False)
    result["inference_scope"] = np.where(
        result["screen_eligible"],
        "donor-level exact test; BH across detection-eligible fixed family panel",
        "descriptive only or unavailable",
    )
    return result


def sensitivity_inference(
    cohort_name: str,
    cohort: dict[str, object],
    eligible_genes: list[str],
) -> pd.DataFrame:
    donor_expression = cohort["donor_expression"]
    labels = donor_expression.index.get_level_values("group").to_numpy(dtype=str)
    rows = []
    for gene in eligible_genes:
        effect, p_value, n_allocations = exact_label_permutation(
            donor_expression[gene].to_numpy(), labels
        )
        rows.append(
            {
                "sensitivity_set": cohort_name,
                "gene": gene,
                "n_Control_donors": int((labels == "Control").sum()),
                "n_DCMP_donors": int((labels == "DCMP").sum()),
                "DCMP_minus_Control_donor_mean_log2_Q3_CPM": effect,
                "exact_two_sided_permutation_p": p_value,
                "n_exact_label_allocations": n_allocations,
            }
        )
    result = pd.DataFrame(rows)
    result["BH_FDR_q"] = bh_fdr(result["exact_two_sided_permutation_p"])
    result["FDR_lt_0.05"] = result["BH_FDR_q"].lt(0.05).fillna(False)
    return result


def selected_donor_heatmap_table(cohort: dict[str, object]) -> pd.DataFrame:
    donor_expression = cohort["donor_expression"]
    donor_detection = cohort["donor_detection"]
    donor_n_aoi = cohort["donor_n_aoi"].set_index(["PID", "group"])["n_AOIs"]
    available = [gene for gene in HEATMAP_GENES if gene in donor_expression.columns]
    matrix = donor_expression[available]
    center = matrix.median(axis=0)
    mad_scale = matrix.sub(center, axis=1).abs().median(axis=0) / 0.6744897501960817
    fallback = matrix.std(axis=0, ddof=1)
    scale = mad_scale.where(mad_scale > 0, fallback).where(lambda x: x > 0, 1.0).fillna(1.0)
    z = matrix.sub(center, axis=1).div(scale, axis=1).clip(-2.5, 2.5)
    rows = []
    for donor_key in matrix.index:
        pid, group = donor_key
        for gene in available:
            rows.append(
                {
                    "PID": pid,
                    "Clinical_phenotype_LV": group,
                    "display_condition": "DCM" if group == "DCMP" else group,
                    "n_selected_AOIs": int(donor_n_aoi.loc[donor_key]),
                    "gene": gene,
                    "donor_mean_log2_Q3_CPM_plus1": float(matrix.at[donor_key, gene]),
                    "gene_robust_z_across_selected_donors": float(z.at[donor_key, gene]),
                    "any_selected_AOI_above_LOQ": bool(donor_detection.at[donor_key, gene]),
                }
            )
    return pd.DataFrame(rows)


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "axes.edgecolor": "#C5CBD3",
            "axes.linewidth": 0.7,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_heading(axis: plt.Axes, label: str, title: str) -> None:
    """Draw a panel letter and title on one shared baseline."""
    axis.text(
        -0.055, 1.045, label, transform=axis.transAxes, ha="right", va="bottom",
        fontsize=12, fontweight="bold", color=TEXT,
    )
    axis.text(
        0.00, 1.045, title, transform=axis.transAxes, ha="left", va="bottom",
        fontsize=10, fontweight="bold", color=TEXT,
    )


def draw_resource(axis: plt.Axes, metrics: dict[str, int], hypothesis: dict[str, object], label: str) -> None:
    axis.set_axis_off()
    panel_heading(axis, label, "Data resource and analysis funnel")

    hypothesis_text = (
        f"Figure 5 hypothesis: {hypothesis['sender_label']}  →  BMP4  →  ACVR2A  →  LECs\n"
        f"MultiNicheNet priority {hypothesis['prioritization_score']:.2f}"
    )
    box = FancyBboxPatch(
        (0.02, 0.78), 0.96, 0.16, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0, edgecolor=GOLD, facecolor="#FFF8E6", transform=axis.transAxes,
    )
    axis.add_patch(box)
    axis.text(0.50, 0.86, hypothesis_text, transform=axis.transAxes, ha="center", va="center", fontsize=7.4, fontweight="bold", linespacing=1.3)

    steps = [
        (0.02, 0.47, 0.25, 0.16, f"{metrics['n_all_AOIs']} AOIs\n{metrics['n_all_donors']} donors / {metrics['n_cores']} cores", "#EDF1F6"),
        (0.37, 0.46, 0.25, 0.18, f"{metrics['n_post_QC_genes']:,} genes\nafter authors'\nprobe QC", "#E4F0F8"),
        (0.72, 0.47, 0.25, 0.16, f"{metrics['n_primary_AOIs']} LV vessel AOIs\n{metrics['n_primary_donors']} donors", "#DFF3EF"),
    ]
    for x, y, width, height, text_value, face in steps:
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0, facecolor=face, transform=axis.transAxes,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text_value, transform=axis.transAxes, ha="center", va="center", fontsize=8)
    axis.annotate("", xy=(0.37, 0.55), xytext=(0.27, 0.55), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=MID_GREY, lw=1.1))
    axis.annotate("", xy=(0.72, 0.55), xytext=(0.62, 0.55), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=MID_GREY, lw=1.1))

    selected = FancyBboxPatch(
        (0.10, 0.115), 0.80, 0.225, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0, edgecolor=TEAL, facecolor="#EBF7F5", transform=axis.transAxes,
    )
    axis.add_patch(selected)
    axis.text(0.50, 0.285, f"{metrics['n_selected_AOIs']} LEC-like AOIs | {metrics['n_selected_donors']} donors", transform=axis.transAxes, ha="center", va="center", fontsize=8.5, fontweight="bold", color=NAVY)
    axis.text(
        0.50, 0.175,
        "≥2 of 5 LEC markers above AOI-specific LOQ\nCCL21 · LYVE1 · PDPN · FLT4 · PROX1",
        transform=axis.transAxes, ha="center", va="center", fontsize=6.3,
        color=SLATE, linespacing=1.25,
    )
    axis.annotate("", xy=(0.50, 0.34), xytext=(0.845, 0.47), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.1))
    axis.text(0.26, 0.05, f"Control\n{metrics['n_selected_control_AOIs']} AOIs / {metrics['n_selected_control_donors']} donors", transform=axis.transAxes, ha="center", va="center", fontsize=7.5, color=BLUE, fontweight="bold")
    axis.text(0.74, 0.05, f"DCM\n{metrics['n_selected_dcmp_AOIs']} AOIs / {metrics['n_selected_dcmp_donors']} donors", transform=axis.transAxes, ha="center", va="center", fontsize=7.5, color=CORAL, fontweight="bold")


def draw_qc(axis: plt.Axes, qc: pd.DataFrame, label: str) -> None:
    panel_heading(axis, label, "Quality of Control/DCM LV vessel AOIs")
    for group in GROUP_ORDER:
        part = qc.loc[qc["Clinical_phenotype_LV"].astype(str).eq(group)]
        axis.scatter(
            part["log10_post_probe_QC_library_size"],
            part["percent_post_probe_QC_genes_above_LOQ"],
            s=30, color=GROUP_COLORS[group], edgecolor="white", linewidth=0.6,
            alpha=0.80, label="DCM" if group == "DCMP" else group, zorder=2,
        )
    selected = qc.loc[qc["selected_two_of_five"]]
    axis.scatter(
        selected["log10_post_probe_QC_library_size"],
        selected["percent_post_probe_QC_genes_above_LOQ"],
        s=70, facecolor="none", edgecolor="#3F4B52", linewidth=1.0,
        label="≥2/5 LEC-like", zorder=3,
    )
    axis.set_xlabel("log10 post-probe-QC library size")
    axis.set_ylabel("Genes above AOI-specific LOQ (%)")
    axis.grid(color=LIGHT_GREY, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=7, loc="lower right")
    medians = selected.groupby("Clinical_phenotype_LV", observed=True)["n_post_probe_QC_genes_above_LOQ"].median()
    text_value = "Selected AOIs, median genes above LOQ — " + " · ".join(
        f"{'DCM' if group == 'DCMP' else group}: {medians.get(group, float('nan')):,.0f}"
        for group in GROUP_ORDER
    )
    axis.text(
        0.00, 1.006, text_value, transform=axis.transAxes,
        ha="left", va="bottom", fontsize=6.1, color=SLATE, clip_on=False,
    )


def draw_detection(axis: plt.Axes, table: pd.DataFrame, label: str) -> None:
    panel_heading(axis, label, "TGF-β/BMP/activin assay coverage and direction")
    cohorts = ["primary_LV_vessel", "two_of_five_LEC_like"]
    cohort_labels = ["All LV vessel\n16 AOIs / 11 donors", "≥2/5 LEC-like\n7 AOIs / 5 donors"]
    plotted = table.loc[table["gene"].isin(DISPLAY_GENES)].copy()
    effect_abs = pd.to_numeric(plotted["DCMP_minus_Control_donor_mean_log2_Q3_CPM"], errors="coerce").abs().max()
    vmax = max(0.5, float(effect_abs) if np.isfinite(effect_abs) else 0.5)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    y_lookup = {gene: len(DISPLAY_GENES) - 1 - i for i, gene in enumerate(DISPLAY_GENES)}

    start = 0
    for group_name, genes in DISPLAY_GROUPS.items():
        end = start + len(genes)
        y_top = len(DISPLAY_GENES) - start - 0.5
        y_bottom = len(DISPLAY_GENES) - end - 0.5
        face = "#F7F9FB" if list(DISPLAY_GROUPS).index(group_name) % 2 == 0 else "#EEF4F6"
        axis.add_patch(Rectangle((-0.48, y_bottom), 1.96, y_top - y_bottom, facecolor=face, edgecolor="none", zorder=0))
        axis.text(1.46, (y_top + y_bottom) / 2, group_name, rotation=90, ha="center", va="center", fontsize=5.8, color=SLATE)
        start = end

    for x, cohort in enumerate(cohorts):
        frame = plotted.loc[plotted["cohort"].eq(cohort)].set_index("gene")
        for gene in DISPLAY_GENES:
            row = frame.loc[gene]
            y = y_lookup[gene]
            if row["assay_status"] != "available_post_probe_QC":
                axis.scatter(x, y, marker="x", s=35, linewidth=1.4, color=GOLD, zorder=3)
                continue
            detection_fraction = float(row["percent_AOIs_above_LOQ"]) / 100.0
            effect = float(row["DCMP_minus_Control_donor_mean_log2_Q3_CPM"])
            axis.scatter(x, y, s=18 + 150 * detection_fraction, color=cmap(norm(effect)), edgecolor="white", linewidth=0.5, zorder=3)
    axis.set_xlim(-0.5, 1.64)
    axis.set_ylim(-0.6, len(DISPLAY_GENES) - 0.4)
    axis.set_xticks([0, 1], cohort_labels)
    axis.set_yticks([y_lookup[gene] for gene in DISPLAY_GENES], [f"{gene}†" if gene in {"BMP4", "ACVR2A"} else gene for gene in DISPLAY_GENES], fontstyle="italic", fontsize=6.2)
    axis.tick_params(axis="x", labelsize=6.8, length=0)
    axis.tick_params(axis="y", length=0)
    axis.spines[:].set_visible(False)
    size_handles = [axis.scatter([], [], s=18 + 150 * fraction, color="#A9BBCB", edgecolor="white") for fraction in [0.25, 0.50, 1.0]]
    legend1 = axis.legend(size_handles, ["25%", "50%", "100%"], title="AOIs above LOQ", frameon=False, bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=6.2, title_fontsize=6.4)
    axis.add_artist(legend1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = axis.figure.colorbar(sm, ax=axis, fraction=0.035, pad=0.16)
    cbar.set_label("DCM − Control donor-mean\nlog2(Q3 CPM + 1)", fontsize=6.4)
    cbar.ax.tick_params(labelsize=6)
    acvr2a = plotted.loc[
        plotted["cohort"].eq("primary_LV_vessel") & plotted["gene"].eq("ACVR2A")
    ].iloc[0]
    axis.text(
        0.00, -0.085,
        f"† mentor/upstream-prioritized. ACVR2A raw probe: {int(acvr2a['n_raw_AOIs_above_LOQ_all_178'])}/178 above LOQ, "
        f"{int(acvr2a['n_raw_primary_AOIs_above_LOQ'])}/16 focused; gold × = failed author 1% probe QC.",
        transform=axis.transAxes, ha="left", va="top", fontsize=5.9, color=SLATE,
    )


def draw_heatmap(axis: plt.Axes, table: pd.DataFrame, label: str) -> None:
    panel_heading(axis, label, "Donor-level pathway context in LEC-like AOIs")
    donor_info = table[["PID", "Clinical_phenotype_LV"]].drop_duplicates().copy()
    donor_info["group_order"] = donor_info["Clinical_phenotype_LV"].map({"Control": 0, "DCMP": 1})
    donor_info = donor_info.sort_values(["group_order", "PID"])
    donors = donor_info["PID"].tolist()
    genes = [gene for gene in HEATMAP_GENES if gene in set(table["gene"])]
    matrix = table.pivot(index="gene", columns="PID", values="gene_robust_z_across_selected_donors").reindex(index=genes, columns=donors)
    detection = table.pivot(index="gene", columns="PID", values="any_selected_AOI_above_LOQ").reindex(index=genes, columns=donors)
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    for row in range(len(genes)):
        for column in range(len(donors)):
            if bool(detection.iloc[row, column]):
                axis.scatter(column, row, s=8, color="#263238", edgecolor="white", linewidth=0.25)
    axis.set_xticks(range(len(donors)), donors, rotation=45, ha="right", fontsize=6.5)
    axis.set_yticks(range(len(genes)), genes, fontstyle="italic", fontsize=6.8)
    axis.tick_params(length=0)
    axis.spines[:].set_visible(False)
    for tick, (_, row) in zip(axis.get_xticklabels(), donor_info.iterrows()):
        tick.set_color(GROUP_COLORS[str(row["Clinical_phenotype_LV"])])
        tick.set_fontweight("bold")
    axis.legend(
        handles=[Patch(facecolor=BLUE, label="Control"), Patch(facecolor=CORAL, label="DCM")],
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.82,
        fontsize=5.8, ncol=2, loc="upper right", bbox_to_anchor=(0.995, 0.995),
        borderpad=0.25, handlelength=1.0, columnspacing=0.8,
    )
    cbar = axis.figure.colorbar(image, ax=axis, fraction=0.04, pad=0.04)
    cbar.set_label("relative expression z", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)


def draw_effects(axis: plt.Axes, effects: pd.DataFrame, label: str) -> None:
    panel_heading(axis, label, "Focused pathway effects at the donor level")
    frame = effects.set_index("gene").reindex(PREFERRED_EFFECT_ORDER).dropna(subset=["primary_effect"], how="all")
    genes = frame.index.tolist()
    y = np.arange(len(genes))[::-1]
    axis.axvline(0, color=MID_GREY, linewidth=0.8)
    for ypos, (gene, row) in zip(y, frame.iterrows()):
        primary_effect = float(row["primary_effect"])
        selected_effect = float(row["selected_effect"]) if pd.notna(row["selected_effect"]) else np.nan
        if np.isfinite(selected_effect):
            axis.plot([primary_effect, selected_effect], [ypos, ypos], color="#D7DEE6", lw=0.8, zorder=1)
            axis.scatter(selected_effect, ypos, marker="D", s=27, color=CORAL, edgecolor="white", linewidth=0.5, zorder=3)
        axis.scatter(primary_effect, ypos, marker="o", s=28, color=NAVY, edgecolor="white", linewidth=0.5, zorder=4)
        if bool(row["primary_FDR_lt_0.05"]):
            axis.text(primary_effect, ypos + 0.30, "★", ha="center", va="center", fontsize=7.5, color=GOLD)
    labels = [f"{gene}†" if gene == "BMP4" else gene for gene in genes]
    axis.set_yticks(y, labels, fontstyle="italic", fontsize=6.7)
    for tick, gene in zip(axis.get_yticklabels(), genes):
        if bool(frame.loc[gene, "primary_FDR_lt_0.05"]):
            tick.set_fontweight("bold")
    axis.set_xlabel("DCM − Control donor-mean log2(Q3 CPM + 1)")
    axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(
        handles=[
            plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor=NAVY, markeredgecolor="white", label="All LV vessel (4 Control / 7 DCM donors)"),
            plt.Line2D([], [], marker="D", linestyle="none", markerfacecolor=CORAL, markeredgecolor="white", label="≥2/5 LEC-like (2 Control / 3 DCM donors; descriptive)"),
        ],
        frameon=False, fontsize=5.8, loc="upper right", bbox_to_anchor=(1.0, 1.075),
    )
    axis.text(
        0.00, -0.19,
        "★ BH FDR q<0.05 across the fixed detection-eligible family panel.\n"
        "† BMP4 retained by prior hypothesis but below the test gate; ACVR2A unavailable after probe QC.",
        transform=axis.transAxes, ha="left", va="top", fontsize=5.7, color=SLATE,
    )


def draw_sensitivity(axis: plt.Axes, sensitivity: pd.DataFrame, significant_genes: list[str], label: str) -> None:
    panel_heading(axis, label, "Robustness of the primary pathway hits")
    set_order = ["primary", "strict_anatomical_LV", "pre_LVAD", "strict_noncardiac_controls", "strict_controls_block1"]
    set_labels = ["Primary\n4C / 7D", "Strict LV\n4C / 6D", "Pre-LVAD\n4C / 6D", "Strict controls\n3C / 7D", "Block 1 + strict C\n3C / 4D"]
    frame = sensitivity.loc[sensitivity["gene"].isin(significant_genes)].copy()
    effect = frame.pivot(index="gene", columns="sensitivity_set", values="DCMP_minus_Control_donor_mean_log2_Q3_CPM").reindex(index=significant_genes, columns=set_order)
    qvalue = frame.pivot(index="gene", columns="sensitivity_set", values="BH_FDR_q").reindex(index=significant_genes, columns=set_order)
    vmax = max(0.5, float(np.nanmax(np.abs(effect.to_numpy()))))
    image = axis.imshow(effect.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    for row in range(effect.shape[0]):
        for column in range(effect.shape[1]):
            q = qvalue.iloc[row, column]
            value = effect.iloc[row, column]
            if pd.isna(value):
                continue
            text_value = "★" if pd.notna(q) and q < 0.05 else "·"
            color = "white" if abs(float(value)) > 0.62 * vmax else TEXT
            axis.text(column, row, text_value, ha="center", va="center", fontsize=8, color=color, fontweight="bold")
    axis.set_xticks(range(len(set_order)), set_labels, fontsize=6.2)
    axis.set_yticks(range(len(significant_genes)), significant_genes, fontstyle="italic", fontsize=7)
    axis.tick_params(length=0)
    axis.spines[:].set_visible(False)
    cbar = axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    cbar.set_label("DCM − Control effect", fontsize=6.4)
    cbar.ax.tick_params(labelsize=6)
    axis.text(
        0.0, -0.20,
        "★ within-sensitivity BH FDR q<0.05; · same test, q≥0.05.\n"
        "Low-n exact tests have coarse p-value resolution.",
        transform=axis.transAxes, ha="left", va="top", fontsize=5.7, color=SLATE,
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str, dpi: int) -> list[dict[str, object]]:
    rows = []
    for extension in ["png", "pdf", "jpg"]:
        path = directory / f"{stem}.{extension}"
        kwargs = {"dpi": dpi} if extension in {"png", "jpg"} else {}
        if extension == "jpg":
            kwargs["pil_kwargs"] = {"quality": 95, "subsampling": 0}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        rows.append({"figure": stem, "format": extension, "path": str(path), "dpi": dpi if extension in {"png", "jpg"} else "vector"})
    return rows


def write_readme(
    path: Path,
    metrics: dict[str, int],
    eligible_genes: list[str],
    significant_genes: list[str],
) -> None:
    text = f"""# Figure 6: spatial disease BMP-ACVR2A analysis

This figure tests the spatial context of the Figure-5 MultiNicheNet-prioritized
endocardial-EC -> BMP4 -> ACVR2A -> LEC axis using GSE271676 GeoMx WTA data.
The default export contains the assembled analysis summary and source tables
only; individual panel files require `--export_individual_panels`.

## Statistical unit and test

- Primary cohort: {metrics['n_primary_AOIs']} Control/DCM LV vessel AOIs from {metrics['n_primary_donors']} donors.
- Biological replicate: donor/PID. Repeated AOIs are averaged within donor.
- Test: two-sided exact donor-label permutation.
- Multiple testing: Benjamini-Hochberg across {len(eligible_genes)} genes from the fixed family panel that were above LOQ in at least 50% of donors in each group.
- Significant primary genes: {', '.join(significant_genes)}.

## LEC-like overlay

Seven AOIs from five donors meet the prespecified two-of-five LEC-marker rule.
Only two Control donors remain, so this overlay is descriptive and has no
p-values or FDR. GeoMx vessel AOIs are CD31-enriched mixed regions, not purified
LECs.

## ACVR2A boundary

ACVR2A was present as one raw WTA probe but was above the reconstructed LOQ in
only one of 178 AOIs (<1%). It was excluded by the authors' 1% probe-QC filter.
Raw ACVR2A counts are shown only in the assay audit and are not rescued for
expression testing.

## Interpretation

Higher SMAD7, ID1, ID2, ENG, and ID3 in DCM LV vessel AOIs supports an altered
BMP/TGF-beta-compatible response/feedback context. It does not directly validate
ACVR2A, ligand-receptor directionality, receptor occupancy, phospho-SMAD activity,
purified LEC signaling, or causality.
"""
    path.write_text(text, encoding="utf-8")


def make_combined_figure(metrics, hypothesis, qc, detection_table, heatmap, effects, sensitivity, sensitivity_display_genes):
    """Draw the existing six panels with no figure-wide heading."""
    set_publication_style()
    fig = plt.figure(figsize=(11.0, 14.2))
    grid = fig.add_gridspec(
        3, 2, height_ratios=[0.75, 1.62, 1.18], hspace=0.47, wspace=0.34,
        left=0.08, right=0.94, top=0.925, bottom=0.13,
    )
    draw_resource(fig.add_subplot(grid[0, 0]), metrics, hypothesis, "A")
    draw_qc(fig.add_subplot(grid[0, 1]), qc, "B")
    draw_detection(fig.add_subplot(grid[1, 0]), detection_table, "C")
    draw_heatmap(fig.add_subplot(grid[1, 1]), heatmap, "D")
    draw_effects(fig.add_subplot(grid[2, 0]), effects, "E")
    draw_sensitivity(fig.add_subplot(grid[2, 1]), sensitivity, sensitivity_display_genes, "F")
    fig.text(
        0.94, 0.022,
        "Expression/co-detection supports pathway context—not ACVR2A expression, ligand–receptor directionality, or causal signaling.",
        ha="right", va="bottom", fontsize=6.2, color=SLATE,
    )
    return fig


def render_saved_figure(args, root: Path) -> None:
    """Apply layout edits without recalculating any disease statistics."""
    output = resolve_path(root, args.output_root)
    sources = output / "source_data"
    resource = pd.read_csv(sources / "Fig06A_resource_and_design.csv")
    metrics = {
        row.metric: int(row.value)
        for row in resource.itertuples()
        if row.record_type == "resource_metric" and row.metric.startswith("n_")
    }
    hypothesis = read_upstream_hypothesis(resolve_path(root, args.upstream_figure5_source))
    qc = pd.read_csv(sources / "Fig06B_AOI_QC.csv")
    detection = pd.read_csv(sources / "Fig06C_pathway_detection.csv")
    heatmap = pd.read_csv(sources / "Fig06D_selected_donor_expression.csv")
    effects = pd.read_csv(sources / "Fig06E_pathway_effects.csv")
    sensitivity = pd.read_csv(sources / "Fig06F_pathway_sensitivity.csv")
    hits = set(effects.loc[as_bool(effects["primary_FDR_lt_0.05"]), "gene"])
    display_genes = [gene for gene in ["SMAD7", "ID1", "ID2", "ENG", "ID3"] if gene in hits]
    fig = make_combined_figure(metrics, hypothesis, qc, detection, heatmap, effects, sensitivity, display_genes)
    records = save_figure(fig, output / "figures", "Fig06_spatial_disease_BMP_ACVR2A", args.dpi)
    plt.close(fig)
    manifest_path = output / "Fig06_figure_manifest.csv"
    updated = pd.DataFrame(records)
    updated["script"] = str(Path(__file__).resolve())
    updated["generated_utc"] = datetime.now(timezone.utc).isoformat()
    if manifest_path.is_file():
        previous = pd.read_csv(manifest_path)
        updated = pd.concat([previous.loc[~previous["figure"].eq("Fig06_spatial_disease_BMP_ACVR2A")], updated], ignore_index=True)
    updated.to_csv(manifest_path, index=False)
    print(f"Redrew disease Figure 6 from unchanged source tables: {output}")


def main() -> None:
    args = parse_args()
    if not (0 < args.screen_detection_fraction <= 1):
        raise ValueError("--screen_detection_fraction must be in (0, 1]")
    root = find_project_root()
    if args.render_only:
        render_saved_figure(args, root)
        return
    all_path = resolve_path(root, args.all_h5ad)
    selected_path = resolve_path(root, args.selected_h5ad)
    initial_path = resolve_path(root, args.initial_dataset)
    reactome_path = resolve_path(root, args.reactome_gmt)
    upstream_path = resolve_path(root, args.upstream_figure5_source)
    output_root = resolve_path(root, args.output_root)
    for path, label in [
        (all_path, "Step 13.1 H5AD"),
        (selected_path, "Step 13.2 selected H5AD"),
        (initial_path, "InitialDataset.csv"),
        (reactome_path, "Reactome GMT"),
        (upstream_path, "Figure 5 source data"),
    ]:
        validate_file(path, label)
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; rerun with --overwrite: {output_root}")
    figure_dir = output_root / "figures"
    source_dir = output_root / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    all_adata = ad.read_h5ad(all_path)
    selected_reference = ad.read_h5ad(selected_path, backed="r")
    if all_adata.n_obs != 178 or all_adata.n_vars != 12800:
        raise ValueError(f"Unexpected Step-13.1 shape: {all_adata.shape}")
    for layer in [EXPRESSION_LAYER, COUNT_LAYER]:
        if layer not in all_adata.layers:
            raise KeyError(f"Missing required all-AOI layer: {layer}")
    initial = pd.read_csv(initial_path, low_memory=False)
    reactome = read_reactome_membership(reactome_path)
    hypothesis = read_upstream_hypothesis(upstream_path)
    audit, probe_counts = assay_audit(initial, all_adata, reactome)

    obs = all_adata.obs
    primary_mask = as_bool(obs["analysis_primary_paper_LV_vessel"])
    strict_lv_mask = as_bool(obs["analysis_strict_origin_LV_vessel"])
    pre_lvad_mask = as_bool(obs["analysis_pre_LVAD_sensitivity"])
    strict_control_flag = as_bool(obs["analysis_strict_noncardiac_control"])
    phenotype = obs["Clinical_phenotype_LV"].astype(str)
    strict_controls_mask = primary_mask & (phenotype.eq("DCMP") | strict_control_flag)
    block_one = pd.to_numeric(obs["TMA.block"], errors="raise").eq(1)
    strict_controls_block_one_mask = strict_controls_mask & block_one

    selected_ids = set(selected_reference.obs_names.astype(str))
    selected_mask = pd.Series(all_adata.obs_names.astype(str).isin(selected_ids), index=all_adata.obs_names)
    primary_ids = set(all_adata.obs_names[primary_mask].astype(str))
    if not selected_ids.issubset(primary_ids):
        raise ValueError("Step-13.2 selected AOIs are not a subset of the primary LV vessel cohort")

    cohorts = {
        "primary": make_cohort(all_adata, primary_mask, probe_counts),
        "strict_anatomical_LV": make_cohort(all_adata, strict_lv_mask, probe_counts),
        "pre_LVAD": make_cohort(all_adata, pre_lvad_mask, probe_counts),
        "strict_noncardiac_controls": make_cohort(all_adata, strict_controls_mask, probe_counts),
        "strict_controls_block1": make_cohort(all_adata, strict_controls_block_one_mask, probe_counts),
        "selected": make_cohort(all_adata, selected_mask, probe_counts),
    }

    selected_donors = cohorts["selected"]["donor_expression"].index.get_level_values("group")
    n_selected_control = int((selected_donors == "Control").sum())
    n_selected_dcmp = int((selected_donors == "DCMP").sum())
    if n_selected_control != 2 or n_selected_dcmp != 3:
        raise ValueError("Selected-cohort donor counts changed; review the no-inference rule")

    primary_detection = add_primary_inference(
        detection_summary("primary_LV_vessel", cohorts["primary"], audit),
        cohorts["primary"],
        args.screen_detection_fraction,
    )
    selected_detection = detection_summary(
        "two_of_five_LEC_like", cohorts["selected"], audit
    )
    selected_detection["screen_eligible"] = False
    selected_detection["exact_two_sided_permutation_p"] = np.nan
    selected_detection["BH_FDR_q"] = np.nan
    selected_detection["FDR_lt_0.05"] = False
    selected_detection["inference_scope"] = "descriptive only: two Control donors"
    detection_table = pd.concat([primary_detection, selected_detection], ignore_index=True, sort=False)

    eligible_genes = primary_detection.loc[primary_detection["screen_eligible"], "gene"].astype(str).tolist()
    if len(eligible_genes) < 1:
        raise ValueError("No pathway genes passed the fixed detection gate")
    sensitivity_tables = []
    for name in [
        "primary", "strict_anatomical_LV", "pre_LVAD",
        "strict_noncardiac_controls", "strict_controls_block1",
    ]:
        sensitivity_tables.append(sensitivity_inference(name, cohorts[name], eligible_genes))
    sensitivity = pd.concat(sensitivity_tables, ignore_index=True)

    primary_effects = primary_detection[
        [
            "gene", "pathway_group", "assay_status", "screen_eligible",
            "DCMP_minus_Control_donor_mean_log2_Q3_CPM", "exact_two_sided_permutation_p",
            "BH_FDR_q", "FDR_lt_0.05", "percent_AOIs_above_LOQ",
            "percent_donors_with_any_AOI_above_LOQ",
        ]
    ].rename(
        columns={
            "DCMP_minus_Control_donor_mean_log2_Q3_CPM": "primary_effect",
            "exact_two_sided_permutation_p": "primary_exact_p",
            "BH_FDR_q": "primary_BH_FDR_q",
            "FDR_lt_0.05": "primary_FDR_lt_0.05",
            "percent_AOIs_above_LOQ": "primary_percent_AOIs_above_LOQ",
            "percent_donors_with_any_AOI_above_LOQ": "primary_percent_donors_above_LOQ",
        }
    )
    selected_effects = selected_detection[
        [
            "gene", "DCMP_minus_Control_donor_mean_log2_Q3_CPM",
            "percent_AOIs_above_LOQ", "percent_donors_with_any_AOI_above_LOQ",
        ]
    ].rename(
        columns={
            "DCMP_minus_Control_donor_mean_log2_Q3_CPM": "selected_effect",
            "percent_AOIs_above_LOQ": "selected_percent_AOIs_above_LOQ",
            "percent_donors_with_any_AOI_above_LOQ": "selected_percent_donors_above_LOQ",
        }
    )
    effects = primary_effects.merge(selected_effects, on="gene", how="left")
    heatmap = selected_donor_heatmap_table(cohorts["selected"])

    primary_obs = cohorts["primary"]["obs"].copy()
    primary_obs["selected_two_of_five"] = primary_obs.index.astype(str).isin(selected_ids)
    primary_obs["log10_post_probe_QC_library_size"] = np.log10(primary_obs["post_probe_QC_library_size"].astype(float))
    primary_obs["percent_post_probe_QC_genes_above_LOQ"] = 100.0 * primary_obs["fraction_post_probe_QC_genes_above_LOQ"].astype(float)
    qc_columns = [
        "aoi_id", "PID", "Clinical_phenotype_LV", "TMA.block", "Scan.name",
        "Origin_Structure", "After_LVAD", "control_definition", "qc_status",
        "post_probe_QC_library_size", "n_post_probe_QC_genes_above_LOQ",
        "fraction_post_probe_QC_genes_above_LOQ", "log10_post_probe_QC_library_size",
        "percent_post_probe_QC_genes_above_LOQ", "selected_two_of_five",
    ]
    qc = primary_obs[qc_columns].copy()

    selected_obs = cohorts["selected"]["obs"]
    metrics = {
        "n_all_AOIs": int(all_adata.n_obs),
        "n_all_donors": int(obs["PID"].astype(str).nunique()),
        "n_cores": int(obs["ID.block.core"].astype(str).nunique()),
        "n_assayed_genes": int(initial.loc[initial["CodeClass"].astype(str).eq("Endogenous"), "TargetName"].astype(str).nunique()),
        "n_post_QC_genes": int(all_adata.n_vars),
        "n_all_AOIs_hard_QC_pass": int((~as_bool(obs["qc_hard_fail"])).sum()),
        "n_primary_AOIs": int(cohorts["primary"]["adata"].n_obs),
        "n_primary_donors": int(cohorts["primary"]["obs"]["PID"].astype(str).nunique()),
        "n_selected_AOIs": int(cohorts["selected"]["adata"].n_obs),
        "n_selected_donors": int(selected_obs["PID"].astype(str).nunique()),
        "n_selected_control_AOIs": int(selected_obs["Clinical_phenotype_LV"].astype(str).eq("Control").sum()),
        "n_selected_control_donors": n_selected_control,
        "n_selected_dcmp_AOIs": int(selected_obs["Clinical_phenotype_LV"].astype(str).eq("DCMP").sum()),
        "n_selected_dcmp_donors": n_selected_dcmp,
    }
    resource = pd.DataFrame(
        [{"record_type": "resource_metric", "metric": key, "value": value} for key, value in metrics.items()]
        + [
            {"record_type": "resource_metric", "metric": "data_resource", "value": "GSE271676"},
            {"record_type": "resource_metric", "metric": "platform", "value": "NanoString GeoMx Human Whole Transcriptome Atlas"},
            {"record_type": "resource_metric", "metric": "observation_unit", "value": "AOI (area of illumination)"},
            {"record_type": "resource_metric", "metric": "vessel_segment_definition", "value": "Vimentin+/CD31+/Troponin-I- enriched segment"},
            {"record_type": "resource_metric", "metric": "LEC_like_selection_rule", "value": LEC_RULE},
            {"record_type": "upstream_hypothesis", "metric": "Figure5_DCM_axis", "value": f"{hypothesis['sender_label']} -> BMP4 -> ACVR2A -> LECs; priority={hypothesis['prioritization_score']:.6f}"},
        ]
    )

    significant = primary_detection.loc[primary_detection["FDR_lt_0.05"]].sort_values(["BH_FDR_q", "gene"])
    significant_genes = significant["gene"].astype(str).tolist()
    expected_hits = {"SMAD7", "ID1", "ID2", "ENG", "ID3"}
    if set(significant_genes) != expected_hits:
        raise ValueError(f"Primary significant-gene set changed: {significant_genes}")
    sensitivity_display_genes = [gene for gene in ["SMAD7", "ID1", "ID2", "ENG", "ID3"] if gene in significant_genes]

    resource.to_csv(source_dir / "Fig06A_resource_and_design.csv", index=False)
    qc.to_csv(source_dir / "Fig06B_AOI_QC.csv", index=False)
    detection_table.to_csv(source_dir / "Fig06C_pathway_detection.csv", index=False)
    heatmap.to_csv(source_dir / "Fig06D_selected_donor_expression.csv", index=False)
    effects.to_csv(source_dir / "Fig06E_pathway_effects.csv", index=False)
    sensitivity.to_csv(source_dir / "Fig06F_pathway_sensitivity.csv", index=False)
    audit.to_csv(source_dir / "Fig06_assay_audit.csv", index=False)

    set_publication_style()
    manifest: list[dict[str, object]] = []
    fig = make_combined_figure(metrics, hypothesis, qc, detection_table, heatmap, effects, sensitivity, sensitivity_display_genes)
    manifest.extend(save_figure(fig, figure_dir, "Fig06_spatial_disease_BMP_ACVR2A", args.dpi))
    plt.close(fig)

    panel_specs = [
        ("Fig06A_resource_and_analysis_funnel", (7.3, 4.6), draw_resource, (metrics, hypothesis, "A")),
        ("Fig06B_LV_vessel_AOI_QC", (6.7, 5.2), draw_qc, (qc, "B")),
        ("Fig06C_pathway_coverage_direction", (7.8, 10.8), draw_detection, (detection_table, "C")),
        ("Fig06D_selected_donor_pathway_heatmap", (7.2, 6.0), draw_heatmap, (heatmap, "D")),
        ("Fig06E_donor_level_pathway_effects", (8.8, 7.4), draw_effects, (effects, "E")),
        ("Fig06F_pathway_sensitivity", (8.0, 5.2), draw_sensitivity, (sensitivity, sensitivity_display_genes, "F")),
    ]
    if args.export_individual_panels:
        for stem, size, drawer, drawer_args in panel_specs:
            panel_fig, panel_axis = plt.subplots(figsize=size)
            drawer(panel_axis, *drawer_args)
            panel_fig.tight_layout()
            manifest.extend(save_figure(panel_fig, figure_dir, stem, args.dpi))
            plt.close(panel_fig)
    else:
        for stem, *_ in panel_specs:
            for extension in ["png", "pdf"]:
                legacy_path = figure_dir / f"{stem}.{extension}"
                if legacy_path.is_file():
                    legacy_path.unlink()

    manifest_table = pd.DataFrame(manifest)
    manifest_table["script"] = str(Path(__file__).resolve())
    manifest_table["generated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_table.to_csv(output_root / "Fig06_figure_manifest.csv", index=False)

    input_manifest = pd.DataFrame(
        [
            {"input_role": "all_AOI_expression_and_QC", "path": str(all_path)},
            {"input_role": "two_of_five_selected_AOI_IDs", "path": str(selected_path)},
            {"input_role": "raw_probe_assay_audit_only", "path": str(initial_path)},
            {"input_role": "pathway_definition", "path": str(reactome_path)},
            {"input_role": "upstream_Figure5_hypothesis", "path": str(upstream_path)},
        ]
    )
    input_manifest.to_csv(output_root / "Fig06_input_manifest.csv", index=False)

    acvr2a = audit.loc[audit["gene"].eq("ACVR2A")].iloc[0]
    bmp4_primary = primary_detection.loc[primary_detection["gene"].eq("BMP4")].iloc[0]
    bmp4_selected = selected_detection.loc[selected_detection["gene"].eq("BMP4")].iloc[0]
    bmpr2_primary = primary_detection.loc[primary_detection["gene"].eq("BMPR2")].iloc[0]
    bmpr2_selected = selected_detection.loc[selected_detection["gene"].eq("BMPR2")].iloc[0]
    sensitivity_hits = (
        sensitivity.loc[sensitivity["FDR_lt_0.05"]]
        .groupby("sensitivity_set")["gene"]
        .apply(lambda values: ";".join(values.astype(str)))
        .to_dict()
    )

    conclusion = f"""FIGURE 6 SPATIAL DISEASE CONCLUSION

Upstream hypothesis
- Figure 5 prioritized {hypothesis['sender_label']} -> BMP4 -> ACVR2A -> LECs in DCM (MultiNicheNet priority {hypothesis['prioritization_score']:.3f}).

Resource and QC
- GSE271676 contains {metrics['n_all_AOIs']} GeoMx WTA AOIs from {metrics['n_all_donors']} donors and {metrics['n_cores']} tissue cores.
- {metrics['n_post_QC_genes']:,} of {metrics['n_assayed_genes']:,} endogenous targets remain after the authors' 1% probe-QC filter.
- No AOI triggered the reconstructed hard-outlier rules. The focused Control/DCMP LV vessel cohort contains {metrics['n_primary_AOIs']} AOIs from {metrics['n_primary_donors']} donors.
- {metrics['n_selected_AOIs']} AOIs from {metrics['n_selected_donors']} donors meet the prespecified two-of-five LEC-marker rule. Only {metrics['n_selected_control_donors']} Control donors remain, so this subset is descriptive.

BMP4 and ACVR2A feasibility
- BMP4 is above LOQ in {int(bmp4_primary['n_AOIs_above_LOQ'])}/{int(bmp4_primary['n_AOIs'])} LV vessel AOIs and {int(bmp4_selected['n_AOIs_above_LOQ'])}/{int(bmp4_selected['n_AOIs'])} LEC-like AOIs. It fails the 50%-per-group donor detection gate and is not a significant disease-group result.
- ACVR2A is above raw-probe LOQ in only {int(acvr2a['n_raw_AOIs_above_LOQ_all_178'])}/178 AOIs and 0/16 focused LV vessel AOIs. It failed the authors' 1% probe-QC filter, so direct spatial ACVR2A expression validation is unavailable; raw counts were not rescued.
- BMPR2 is above LOQ in {int(bmpr2_primary['n_AOIs_above_LOQ'])}/{int(bmpr2_primary['n_AOIs'])} LV vessel AOIs and {int(bmpr2_selected['n_AOIs_above_LOQ'])}/{int(bmpr2_selected['n_AOIs'])} LEC-like AOIs, but has no significant donor-level DCM shift.

Other TGF-beta/BMP/activin-family findings
- Across {len(eligible_genes)} detection-eligible genes in the fixed family panel, DCM LV vessel AOIs have higher {', '.join(significant_genes)} at BH FDR q<0.05.
- SMAD7, ID1, and ID2 remain FDR-supported in strict anatomical-LV, pre-LVAD, and strict-noncardiac-control sensitivities. ENG and ID3 are directionally stable but do not remain below FDR 0.05 in every sensitivity.
- The same five genes shift upward in the small LEC-like donor overlay, but that overlay has no formal test.

Interpretation boundary
The independent spatial dataset supports an altered BMP/TGF-beta-compatible transcriptional response/feedback context in DCM CD31-enriched vessel AOIs. It does not directly validate ACVR2A, BMP4 upregulation, purified LEC signaling, ligand-receptor directionality, receptor occupancy, phospho-SMAD activity, or causality.
"""
    (output_root / "Fig06_FINAL_CONCLUSION.txt").write_text(conclusion, encoding="utf-8")

    methods = f"""FIGURE 6 METHODS AND LIMITATIONS

Inputs
- {all_path}
- {selected_path}
- {initial_path}
- {reactome_path}
- {upstream_path}

QC and normalization
- The figure reuses Step 13.1 post-probe-QC counts, AOI metadata, reconstructed LOQ, and log2(Q3 CPM + 1). It does not renormalize the selected subset.
- AOI/target detection requires the summed post-QC count to be at or above the AOI per-probe LOQ multiplied by target probe number.
- Raw InitialDataset counts are used only to audit why ACVR2A/ACVR2B failed the authors' 1% probe-QC filter.

Analysis sets
- Primary focused screen: 16 Control/DCMP LV vessel AOIs, averaged within PID (4 Control and 7 DCM donors).
- Descriptive LEC-like overlay: seven AOIs requiring at least two of {', '.join(LEC_MARKERS)} above LOQ (2 Control and 3 DCM donors); no p-values or FDR.
- Sensitivities: strict anatomical LV, pre-LVAD, strict noncardiac controls, and a TMA-block-1 strict-control subset.

Pathway screen
- Fixed 47-gene family panel: direct ligand/receptor/SMAD members of {', '.join(REACTOME_SETS)}, plus mentor/upstream genes and canonical ID readouts.
- Test eligibility: above LOQ in at least {args.screen_detection_fraction:.0%} of donors in each group; {len(eligible_genes)} genes pass.
- Effect: DCM minus Control difference in equally weighted donor-mean log2(Q3 CPM + 1).
- Test: two-sided exact donor-label permutation; all label allocations enumerated.
- Multiplicity: Benjamini-Hochberg across the {len(eligible_genes)} eligible genes. Each sensitivity applies BH to the same eligible family panel.

Interpretation limits
- The authors used DREAM mixed models in their paper; this focused validation instead uses exact donor-level permutation because the target panel and donor counts are small.
- GeoMx observations are AOIs, not Visium spots or single cells. Vessel AOIs are Vimentin+/CD31+/Troponin-I- enriched mixed regions.
- Transcript abundance/co-detection cannot establish cell-level co-expression, signaling direction, receptor activation, phospho-SMAD activity, regulon activity, or causality.
"""
    (output_root / "Fig06_METHODS_AND_LIMITATIONS.txt").write_text(methods, encoding="utf-8")

    run_config = {
        "script_version": SCRIPT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "all_h5ad": str(all_path),
            "selected_h5ad": str(selected_path),
            "initial_dataset": str(initial_path),
            "reactome_gmt": str(reactome_path),
            "upstream_figure5_source": str(upstream_path),
        },
        "output_root": str(output_root),
        "expression_layer": EXPRESSION_LAYER,
        "LEC_rule": LEC_RULE,
        "screen_detection_fraction_per_group": args.screen_detection_fraction,
        "fixed_family_panel": FAMILY_GENES,
        "reactome_sets": REACTOME_SETS,
        "screen_eligible_genes": eligible_genes,
        "significant_primary_genes_BH_FDR_lt_0.05": significant_genes,
        "sensitivity_significant_genes": sensitivity_hits,
        "ACVR2A_result": "direct_validation_unavailable_failed_authors_probe_QC",
        "formal_inference_in_selected_LEC_like_subset": False,
        "individual_panel_files_exported": args.export_individual_panels,
        "dpi": args.dpi,
        "resource_inputs_modified": False,
    }
    (output_root / "Fig06_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    run_summary = pd.DataFrame(
        [
            {"metric": "script_version", "value": SCRIPT_VERSION},
            {"metric": "combined_figure", "value": str(figure_dir / "Fig06_spatial_disease_BMP_ACVR2A.png")},
            {"metric": "n_figure_stems", "value": int(manifest_table["figure"].nunique())},
            {"metric": "n_fixed_family_genes", "value": len(FAMILY_GENES)},
            {"metric": "n_detection_eligible_family_genes", "value": len(eligible_genes)},
            {"metric": "significant_primary_genes_BH_FDR_lt_0.05", "value": ";".join(significant_genes)},
            {"metric": "ACVR2A_result", "value": "direct_validation_unavailable_failed_authors_probe_QC"},
            {"metric": "selected_subset_inference", "value": "descriptive_only_two_Control_donors"},
            {"metric": "status", "value": "completed"},
        ]
    )
    run_summary.to_csv(output_root / "Fig06_run_summary.csv", index=False)
    write_readme(output_root / "README.md", metrics, eligible_genes, significant_genes)
    print(conclusion)
    print(f"Wrote Figure 6 outputs to: {output_root}")


if __name__ == "__main__":
    main()
