#!/usr/bin/env python3
r"""
Step 13.2: select GSE271676 DCM LEC-like vessel AOIs by a two-of-five rule.

A primary-QC LV vessel AOI is selected when at least two of CCL21, LYVE1,
PDPN, FLT4, and PROX1 are above its AOI-specific GeoMx LOQ. The rule is shared
with aging Step 12.3; no additional endothelial-expression requirement is
used. Aging uses nonzero Visium counts, whereas GeoMx uses above-LOQ evidence.

CCL21 is both a selection marker and a homeostatic endpoint. That overlap is
kept because it is part of the requested rule, but it is explicitly recorded
as a limitation. This step selects observations and writes the analysis H5AD;
it does not test Control versus DCMP biology.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_VERSION = "13_2_GSE271676_two_of_five_LEC_selection_2026_08_28"
DEFAULT_STEP13_1_H5AD = (
    "new_result2_spatial_disease/13_1_spatial_disease_QC/"
    "13_1_GSE271676_GeoMx_Q3_expression.h5ad"
)
DEFAULT_DETECTION = (
    "new_result2_spatial_disease/13_1_spatial_disease_QC/"
    "13_1_mentor_target_detection_by_AOI.csv.gz"
)
DEFAULT_REGISTRY = (
    "new_result2_spatial_disease/13_1_spatial_disease_QC/"
    "13_1_DCM_target_registry.csv"
)
DEFAULT_OUTPUT = "new_result2_spatial_disease/13_2_disease_LEC_reference"
LEC_MARKERS = ["CCL21", "LYVE1", "PDPN", "FLT4", "PROX1"]
AUTHOR_CODE_CANDIDATE_EC_OUTLIERS = {
    "PR79_1_WTA | 008 | vessel",
    "PR79_1_WTA | 038 | vessel",
}


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select GSE271676 LV vessel AOIs by at least two of five LEC markers."
    )
    parser.add_argument("--step13_1_h5ad", default=DEFAULT_STEP13_1_H5AD)
    parser.add_argument("--target_detection", default=DEFAULT_DETECTION)
    parser.add_argument("--target_registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum_LEC_markers_above_LOQ", type=int, default=2)
    parser.add_argument("--stringent_LEC_markers_above_LOQ", type=int, default=3)
    parser.add_argument("--minimum_donors_per_group", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--skip_h5ad", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype("string").fillna("").str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def layer_as_dataframe(adata: ad.AnnData, layer: str) -> pd.DataFrame:
    if layer not in adata.layers:
        raise KeyError(f"Required layer is absent: {layer}")
    matrix = adata.layers[layer]
    values = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    if values.shape != adata.shape:
        raise ValueError(f"Layer {layer} has shape {values.shape}, expected {adata.shape}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Layer {layer} contains invalid values")
    return pd.DataFrame(
        values.T.astype(float),
        index=pd.Index(adata.var_names.astype(str), name="gene"),
        columns=pd.Index(adata.obs_names.astype(str), name="aoi_id"),
    )


def q3_cpm(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = counts.to_numpy(float)
    library = matrix.sum(axis=0)
    if np.any(library <= 0):
        raise ValueError("Q3 input contains a nonpositive library size")
    proportions = matrix / library[None, :]
    upper_quartile = np.quantile(proportions, 0.75, axis=0)
    if np.any(upper_quartile <= 0):
        raise ValueError("Q3 input contains a nonpositive upper quartile")
    geometric_mean = float(np.exp(np.mean(np.log(upper_quartile))))
    factor = upper_quartile / geometric_mean
    effective_library = library * factor
    normalized = matrix / effective_library[None, :] * 1_000_000.0
    factors = pd.DataFrame(
        {
            "aoi_id": counts.columns,
            "library_size": library,
            "upper_quartile_proportion": upper_quartile,
            "upper_quartile_norm_factor": factor,
            "effective_library_size": effective_library,
        }
    ).set_index("aoi_id")
    return pd.DataFrame(normalized, index=counts.index, columns=counts.columns), factors


def robust_gene_z(log_expression: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    values = log_expression.to_numpy(float)
    median = np.median(values, axis=1)
    mad = np.median(np.abs(values - median[:, None]), axis=1)
    sd = np.std(values, axis=1, ddof=1)
    scale = mad / 0.6744897501960817
    scale = np.where(scale > 0, scale, np.where(sd > 0, sd, 1.0))
    z = np.clip((values - median[:, None]) / scale[:, None], -clip, clip)
    return pd.DataFrame(z, index=log_expression.index, columns=log_expression.columns)


def marker_detection_table(
    detection: pd.DataFrame, primary_ids: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = detection.loc[
        detection["aoi_id"].isin(primary_ids) & detection["gene"].isin(LEC_MARKERS)
    ].copy()
    frame["above_LOQ_bool"] = as_bool(frame["above_LOQ"])
    pivot = frame.pivot(index="aoi_id", columns="gene", values="above_LOQ_bool")
    pivot = pivot.reindex(index=primary_ids, columns=LEC_MARKERS, fill_value=False).fillna(False).astype(bool)
    summary = []
    for gene in LEC_MARKERS:
        gene_frame = frame.loc[frame["gene"].eq(gene)]
        detected = gene_frame.loc[gene_frame["above_LOQ_bool"]]
        by_group = detected.groupby("Clinical_phenotype_LV", observed=True)["PID"].nunique()
        summary.append(
            {
                "gene": gene,
                "n_primary_AOIs": len(primary_ids),
                "n_AOIs_above_LOQ": int(pivot[gene].sum()),
                "fraction_AOIs_above_LOQ": float(pivot[gene].mean()),
                "n_donors_above_LOQ": int(detected["PID"].nunique()),
                "n_Control_donors_above_LOQ": int(by_group.get("Control", 0)),
                "n_DCMP_donors_above_LOQ": int(by_group.get("DCMP", 0)),
                "used_in_two_of_five_selection": True,
            }
        )
    return pivot, pd.DataFrame(summary)


def add_selection_fields(
    metadata: pd.DataFrame,
    detection_pivot: pd.DataFrame,
    marker_score: pd.Series,
    minimum: int,
    stringent: int,
) -> pd.DataFrame:
    result = metadata.copy()
    result.index = result.index.astype(str)
    detection_pivot = detection_pivot.reindex(result.index, fill_value=False)
    result["n_LEC_markers_above_LOQ"] = detection_pivot.sum(axis=1).astype(int)
    result["LEC_markers_above_LOQ"] = detection_pivot.apply(
        lambda row: ";".join(row.index[row.to_numpy(bool)]), axis=1
    )
    result["LEC_marker_score_five"] = marker_score.reindex(result.index)
    result["LEC_selected_two_of_five"] = result["n_LEC_markers_above_LOQ"].ge(minimum)
    result["LEC_stringent_three_of_five"] = result["n_LEC_markers_above_LOQ"].ge(stringent)
    result["LEC_selection_tier"] = np.select(
        [result["LEC_stringent_three_of_five"], result["LEC_selected_two_of_five"]],
        ["selected_three_or_more_of_five", "selected_exactly_two_of_five"],
        default="excluded_fewer_than_two_of_five",
    )
    result["author_code_candidate_EC_outlier"] = result.index.isin(AUTHOR_CODE_CANDIDATE_EC_OUTLIERS)
    result["CCL21_selection_endpoint_overlap"] = True
    return result


def readiness_table(scores: pd.DataFrame, minimum_donors: int) -> pd.DataFrame:
    selected = as_bool(scores["LEC_selected_two_of_five"])
    strict_control = scores["Clinical_phenotype_LV"].eq("DCMP") | as_bool(scores["analysis_strict_noncardiac_control"])
    definitions = OrderedDict(
        [
            ("primary_two_of_five", selected),
            ("strict_anatomical_origin_LV", selected & as_bool(scores["analysis_strict_origin_LV_vessel"])),
            ("strict_noncardiac_controls", selected & strict_control),
            ("pre_LVAD_only", selected & as_bool(scores["analysis_pre_LVAD_sensitivity"])),
            ("exclude_author_candidate_EC_outliers", selected & ~as_bool(scores["author_code_candidate_EC_outlier"])),
            ("stringent_three_of_five", selected & as_bool(scores["LEC_stringent_three_of_five"])),
        ]
    )
    rows = []
    for name, mask in definitions.items():
        frame = scores.loc[mask]
        counts = frame.groupby("Clinical_phenotype_LV", observed=True).agg(
            n_AOIs=("aoi_id", "size"), n_donors=("PID", "nunique")
        )
        ready = all(
            int(counts.loc[group, "n_donors"]) >= minimum_donors if group in counts.index else False
            for group in ["Control", "DCMP"]
        )
        for group in ["Control", "DCMP"]:
            rows.append(
                {
                    "analysis_set": name,
                    "Clinical_phenotype_LV": group,
                    "n_AOIs": int(counts.loc[group, "n_AOIs"]) if group in counts.index else 0,
                    "n_donors": int(counts.loc[group, "n_donors"]) if group in counts.index else 0,
                    "minimum_donors_required": minimum_donors,
                    "both_groups_donor_threshold_met": ready,
                    "use_recommendation": "eligible_for_PID_level_inference" if ready else "descriptive_only_insufficient_donors",
                }
            )
    return pd.DataFrame(rows)


def selected_h5ad(
    source: ad.AnnData,
    selected_scores: pd.DataFrame,
    primary_counts: pd.DataFrame,
    primary_q3: pd.DataFrame,
) -> ad.AnnData:
    ids = selected_scores.index.astype(str).tolist()
    counts = primary_counts.loc[:, ids].T.to_numpy(float)
    q3 = primary_q3.loc[:, ids].T.to_numpy(float)
    log_q3 = np.log2(q3 + 1.0)
    var = source.var.copy()
    var.index = var.index.astype(str)
    var["LEC_selection_marker"] = var.index.isin(LEC_MARKERS)
    output = ad.AnnData(X=sp.csr_matrix(log_q3), obs=selected_scores.copy(), var=var)
    output.obs_names = pd.Index(ids, name="obs_id")
    output.var_names = source.var_names.astype(str)
    output.layers["post_probe_QC_counts"] = sp.csr_matrix(counts)
    output.layers["primary_vessel_upper_quartile_CPM"] = sp.csr_matrix(q3)
    output.layers["log2_primary_vessel_upper_quartile_CPM_plus1"] = sp.csr_matrix(log_q3)
    output.uns["step13_2_config_json"] = json.dumps(
        {
            "script_version": SCRIPT_VERSION,
            "LEC_selection_markers": LEC_MARKERS,
            "selection_rule": "at_least_2_of_5_above_AOI_specific_LOQ",
            "CCL21_selection_endpoint_overlap": True,
        }
    )
    return output


def plot_marker_matrix(scores: pd.DataFrame, pivot: pd.DataFrame, path: Path, dpi: int) -> None:
    order = scores.sort_values(
        ["Clinical_phenotype_LV", "LEC_selected_two_of_five", "n_LEC_markers_above_LOQ"],
        ascending=[True, False, False],
    ).index
    matrix = pivot.reindex(order)[LEC_MARKERS].T.to_numpy(int)
    fig, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    axis.imshow(matrix, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#E5E5E5", "#0072B2"]), vmin=0, vmax=1)
    axis.set_yticks(np.arange(len(LEC_MARKERS)), LEC_MARKERS)
    labels = [f"{scores.loc[aoi, 'PID']}\n{scores.loc[aoi, 'Clinical_phenotype_LV']}" for aoi in order]
    axis.set_xticks(np.arange(len(order)), labels, rotation=55, ha="right", fontsize=8)
    axis.set_title("Two-of-five LEC selection: marker detection above AOI-specific LOQ")
    for column, aoi in enumerate(order):
        if scores.loc[aoi, "LEC_selected_two_of_five"]:
            axis.plot([column - 0.48, column + 0.48], [-0.55, -0.55], color="#D55E00", linewidth=3)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_selection_counts(scores: pd.DataFrame, path: Path, dpi: int) -> None:
    rows = []
    for group in ["Control", "DCMP"]:
        frame = scores.loc[scores["Clinical_phenotype_LV"].eq(group)]
        selected = frame.loc[as_bool(frame["LEC_selected_two_of_five"])]
        rows.append((group, len(selected), selected["PID"].nunique()))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    colors = ["#4C78A8", "#E45756"]
    for axis, index, title in [(axes[0], 1, "Selected AOIs"), (axes[1], 2, "Selected donors")]:
        axis.bar([0, 1], [row[index] for row in rows], color=colors)
        axis.set_xticks([0, 1], [row[0] for row in rows])
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
        for position, row in enumerate(rows):
            axis.text(position, row[index] + 0.08, str(row[index]), ha="center")
    fig.suptitle("Primary two-of-five LEC-selected DCM cohort")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not (1 <= args.minimum_LEC_markers_above_LOQ < args.stringent_LEC_markers_above_LOQ <= 5):
        raise ValueError("Marker thresholds must satisfy 1 <= primary < stringent <= 5")
    project_dir = find_project_dir()
    input_h5ad = resolve_path(args.step13_1_h5ad, project_dir)
    detection_path = resolve_path(args.target_detection, project_dir)
    registry_path = resolve_path(args.target_registry, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    for path, label in [(input_h5ad, "Step 13.1 H5AD"), (detection_path, "target detection"), (registry_path, "target registry")]:
        validate_file(path, label)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is nonempty; rerun with --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    figures_dir.mkdir(exist_ok=True)

    adata = ad.read_h5ad(input_h5ad)
    counts = layer_as_dataframe(adata, "post_probe_QC_counts")
    metadata = adata.obs.copy()
    metadata.index = metadata.index.astype(str)
    metadata["aoi_id"] = metadata.index
    all_lv_mask = metadata["Ventricle"].astype(str).eq("LV") & metadata["Segment.type"].astype(str).eq("vessel")
    primary_mask = as_bool(metadata["analysis_primary_paper_LV_vessel"])
    all_lv_ids = metadata.index[all_lv_mask].tolist()
    primary_ids = metadata.index[primary_mask].tolist()
    if len(primary_ids) != 16:
        raise ValueError(f"Expected 16 primary LV vessel AOIs; found {len(primary_ids)}")

    all_lv_q3, all_lv_factors = q3_cpm(counts.loc[:, all_lv_ids])
    primary_q3, primary_factors = q3_cpm(counts.loc[:, primary_ids])
    all_lv_z = robust_gene_z(np.log2(all_lv_q3 + 1.0))
    missing_markers = [gene for gene in LEC_MARKERS if gene not in all_lv_z.index]
    if missing_markers:
        raise KeyError(f"Selection markers missing after probe QC: {missing_markers}")
    marker_score = all_lv_z.loc[LEC_MARKERS].mean(axis=0)

    detection = pd.read_csv(detection_path, low_memory=False)
    pivot, marker_summary = marker_detection_table(detection, primary_ids)
    primary_scores = add_selection_fields(
        metadata.loc[primary_ids], pivot, marker_score,
        args.minimum_LEC_markers_above_LOQ,
        args.stringent_LEC_markers_above_LOQ,
    )
    selected_scores = primary_scores.loc[as_bool(primary_scores["LEC_selected_two_of_five"])].copy()
    if selected_scores.empty:
        raise ValueError("The two-of-five rule selected zero AOIs")
    readiness = readiness_table(primary_scores, args.minimum_donors_per_group)
    donor_scores = selected_scores.groupby(["PID", "Clinical_phenotype_LV"], observed=True).agg(
        n_AOIs=("aoi_id", "size"),
        mean_n_LEC_markers_above_LOQ=("n_LEC_markers_above_LOQ", "mean"),
        mean_LEC_marker_score_five=("LEC_marker_score_five", "mean"),
    ).reset_index()

    primary_scores.to_csv(output_root / "13_2_all_primary_AOI_two_of_five_LEC_evidence.csv", index=False)
    selected_scores.to_csv(output_root / "13_2_selected_AOI_LEC_evidence.csv", index=False)
    donor_scores.to_csv(output_root / "13_2_selected_donor_LEC_evidence.csv", index=False)
    marker_summary.to_csv(output_root / "13_2_LEC_marker_detection_summary.csv", index=False)
    readiness.to_csv(output_root / "13_2_LEC_selection_readiness.csv", index=False)
    all_lv_factors.assign(normalization_reference="all_44_LV_vessel_AOIs").to_csv(output_root / "13_2_all_LV_vessel_Q3_factors.csv")
    primary_factors.assign(normalization_reference="all_16_primary_Control_DCMP_vessel_AOIs").to_csv(output_root / "13_2_primary_Q3_factors.csv")

    h5ad_path = output_root / "13_2_GSE271676_two_of_five_LEC_selected.h5ad"
    if not args.skip_h5ad:
        selected_h5ad(adata, selected_scores, counts.loc[:, primary_ids], primary_q3).write_h5ad(h5ad_path, compression="gzip")

    figure_rows = []
    if not args.skip_figures:
        marker_path = figures_dir / "13_2_two_of_five_marker_detection.png"
        plot_marker_matrix(primary_scores, pivot, marker_path, args.dpi)
        figure_rows.append({"figure": str(marker_path.relative_to(output_root)), "role": "Marker-by-AOI selection audit"})
        count_path = figures_dir / "13_2_two_of_five_selected_counts.png"
        plot_selection_counts(primary_scores, count_path, args.dpi)
        figure_rows.append({"figure": str(count_path.relative_to(output_root)), "role": "Selected AOI and donor counts"})
    pd.DataFrame(figure_rows).to_csv(output_root / "13_2_figure_manifest.csv", index=False)

    group_counts = selected_scores.groupby("Clinical_phenotype_LV", observed=True).agg(n_AOIs=("aoi_id", "size"), n_donors=("PID", "nunique"))
    control_donors = int(group_counts.loc["Control", "n_donors"])
    dcmp_donors = int(group_counts.loc["DCMP", "n_donors"])
    formal_ready = control_donors >= args.minimum_donors_per_group and dcmp_donors >= args.minimum_donors_per_group
    summary = pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "status": "completed",
                "n_primary_LV_vessel_AOIs_before_LEC_selection": len(primary_scores),
                "n_selected_AOIs": len(selected_scores),
                "n_selected_Control_AOIs": int(group_counts.loc["Control", "n_AOIs"]),
                "n_selected_Control_donors": control_donors,
                "n_selected_DCMP_AOIs": int(group_counts.loc["DCMP", "n_AOIs"]),
                "n_selected_DCMP_donors": dcmp_donors,
                "LEC_selection_markers": ";".join(LEC_MARKERS),
                "minimum_markers_above_LOQ": args.minimum_LEC_markers_above_LOQ,
                "primary_PID_inference_ready": formal_ready,
                "minimum_donors_required_each_group": args.minimum_donors_per_group,
                "formal_disease_test_completed": False,
                "selection_outcome_overlap_warning": "CCL21_is_also_a_homeostatic_endpoint",
                "resource_inputs_modified": False,
                "output_h5ad": str(h5ad_path) if not args.skip_h5ad else "",
            }
        ]
    )
    summary.to_csv(output_root / "13_2_run_summary.csv", index=False)
    config = {
        "script_version": SCRIPT_VERSION,
        "input_h5ad": str(input_h5ad),
        "target_detection": str(detection_path),
        "target_registry": str(registry_path),
        "output_root": str(output_root),
        "LEC_selection_markers": LEC_MARKERS,
        "selection_rule": "at_least_2_of_5_above_AOI_specific_LOQ",
        "minimum_LEC_markers_above_LOQ": args.minimum_LEC_markers_above_LOQ,
        "stringent_LEC_markers_above_LOQ": args.stringent_LEC_markers_above_LOQ,
        "CCL21_included_in_selection": True,
        "minimum_donors_per_group": args.minimum_donors_per_group,
        "normalization": "Q3_CPM_within_all_primary_16_before_selection",
    }
    (output_root / "13_2_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    notes = f"""Step 13.2 two-of-five LEC selection
=====================================

An LV vessel AOI is selected when at least {args.minimum_LEC_markers_above_LOQ}
of {', '.join(LEC_MARKERS)} are above its AOI-specific GeoMx LOQ. No other
expression panel or abundance requirement is used.

The rule selected {len(selected_scores)} of {len(primary_scores)} primary AOIs:
{int(group_counts.loc['Control', 'n_AOIs'])} Control AOIs from {control_donors}
donors and {int(group_counts.loc['DCMP', 'n_AOIs'])} DCMP AOIs from
{dcmp_donors} donors. The prespecified minimum is
{args.minimum_donors_per_group} donors per group, so formal donor-level disease
inference is {'available' if formal_ready else 'not available'}.

CCL21 is both an LEC-selection marker and a homeostatic endpoint. Selection can
therefore enrich the endpoint being tested. AOIs remain mixed vessel regions;
two detected markers do not prove co-expression in one cell.
"""
    (output_root / "13_2_method_and_interpretation_notes.txt").write_text(notes, encoding="utf-8")

    print(f"Step 13.2 complete: {output_root}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
