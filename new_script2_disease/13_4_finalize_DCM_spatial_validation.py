#!/usr/bin/env python3
r"""
Step 13.4: final evidence grading for GSE271676 DCM spatial validation.

This is a locked reporting layer for Steps 13.1-13.3. It does not select a
new cohort or rerun expression tests. Step 13.2 selected LV vessel AOIs when at
least two of CCL21, LYVE1, PDPN, FLT4, and PROX1 were above the AOI-specific
LOQ. Step 13.3 then averaged repeated AOIs within PID.

The selected cohort has two Control donors and three DCMP donors. Because the
Control group is below the prespecified minimum of three donors, Step 13.4
reports direction and coherence descriptively and forbids a formal validation
claim. CCL21 is both a selection marker and a homeostatic endpoint, which is
reported as an explicit selection/outcome overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402


SCRIPT_VERSION = "13_4_GSE271676_two_of_five_descriptive_DCM_grade_2026_08_28"
DEFAULT_STEP13_2 = "new_result2_spatial_disease/13_2_disease_LEC_reference"
DEFAULT_STEP13_3 = "new_result2_spatial_disease/13_3_prespecified_DCM_program_validation"
DEFAULT_OUTPUT = "new_result2_spatial_disease/13_4_final_DCM_spatial_validation"
LEC_MARKERS = ["CCL21", "LYVE1", "PDPN", "FLT4", "PROX1"]
MINIMUM_MARKERS = 2
MINIMUM_DONORS_PER_GROUP = 3

PROGRAM_ORDER = [
    "homeostatic_LEC",
    "disease_expanded_State0_like",
    "disease_expanded_State3_like",
    "activated_vascular_State5_like",
]
MENTOR_CLAIMS = {
    "homeostatic_LEC": "loss of the MMRN1/CCL21 homeostatic program in DCMP",
    "disease_expanded_State0_like": "expansion of the ITGA9/NRP2/NPNT/SEMA3D State0-like program in DCMP",
    "disease_expanded_State3_like": "expansion of the PLSCR4/CRIM1/COLEC12/PDGFC State3-like program in DCMP",
    "activated_vascular_State5_like": "activation of the FLT1/ENG/EPAS1 vascular State5-like program in DCMP",
}
DISPLAY_NAMES = {
    "homeostatic_LEC": "Homeostatic LEC",
    "disease_expanded_State0_like": "Disease-expanded State0-like",
    "disease_expanded_State3_like": "Disease-expanded State3-like",
    "activated_vascular_State5_like": "Activated vascular State5-like",
}
SENSITIVITY_SETS = [
    "strict_anatomical_origin_LV",
    "strict_noncardiac_controls",
    "pre_LVAD_only",
    "exclude_author_candidate_EC_outliers",
    "stringent_three_of_five",
]
GRADE_COLORS = {
    "DESCRIPTIVE_EXPECTED_DIRECTION_INSUFFICIENT_DONORS": "#E69F00",
    "DESCRIPTIVE_OPPOSITE_DIRECTION_INSUFFICIENT_DONORS": "#D55E00",
    "DESCRIPTIVE_NOT_ESTIMABLE": "#7F7F7F",
}


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the final descriptive DCM spatial evidence grade."
    )
    parser.add_argument("--step13_2_root", default=DEFAULT_STEP13_2)
    parser.add_argument("--step13_3_root", default=DEFAULT_STEP13_3)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype("string").fillna("").str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def semicolon_list(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [item for item in str(value).split(";") if item]


def grade_programs(
    registry: pd.DataFrame,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    genes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for program in PROGRAM_ORDER:
        reg = registry.loc[registry["program"].eq(program)].iloc[0]
        test = primary.loc[primary["program"].eq(program)].iloc[0]
        gene_frame = genes.loc[genes["program"].eq(program)].copy()
        sensitivity_frame = sensitivity.loc[
            sensitivity["program"].eq(program)
            & sensitivity["analysis_set"].isin(SENSITIVITY_SETS)
        ].copy()

        planned = semicolon_list(reg["planned_genes"])
        used = semicolon_list(reg["genes_used"])
        excluded = semicolon_list(reg["excluded_genes"])
        direction_supported = bool(test["direction_concordant"])
        inference_eligible = bool(test["minimum_donor_threshold_met"])
        if inference_eligible:
            raise ValueError(
                "This Step 13.4 version is locked to descriptive grading, but Step 13.3 "
                "reports an inference-eligible cohort. Re-review the grading design."
            )

        gene_flags = as_bool(gene_frame["direction_concordant"])
        gene_direction_count = int(gene_flags.sum())
        gene_direction_fraction = (
            gene_direction_count / len(gene_frame) if len(gene_frame) else np.nan
        )
        sensitivity_flags = as_bool(sensitivity_frame["direction_concordant"])
        sensitivity_count = int(sensitivity_flags.sum())
        sensitivity_fraction = (
            sensitivity_count / len(sensitivity_frame)
            if len(sensitivity_frame) else np.nan
        )

        if direction_supported:
            grade = "DESCRIPTIVE_EXPECTED_DIRECTION_INSUFFICIENT_DONORS"
            wording = (
                f"The {DISPLAY_NAMES[program]} score changed in the prespecified "
                f"direction (DCMP-Control={test['difference_DCMP_minus_Control']:.3f}), "
                f"with {gene_direction_count}/{len(gene_frame)} retained genes and "
                f"{sensitivity_count}/{len(sensitivity_frame)} descriptive sensitivity "
                "sets in that direction. This is a descriptive pattern, not formal "
                "validation, because only two Control donors passed LEC selection."
            )
        elif np.isfinite(float(test["difference_DCMP_minus_Control"])):
            grade = "DESCRIPTIVE_OPPOSITE_DIRECTION_INSUFFICIENT_DONORS"
            wording = (
                f"The {DISPLAY_NAMES[program]} score changed opposite to the "
                f"prespecified direction (DCMP-Control={test['difference_DCMP_minus_Control']:.3f}). "
                "The selected cohort is too small for an inferential contradiction claim."
            )
        else:
            grade = "DESCRIPTIVE_NOT_ESTIMABLE"
            wording = f"The {DISPLAY_NAMES[program]} effect was not estimable."

        overlap = program == "homeostatic_LEC"
        if overlap:
            wording += (
                " CCL21 is both a selection marker and an endpoint, so this homeostatic "
                "result is not selection-independent."
            )

        rows.append(
            {
                "program": program,
                "mentor_claim": MENTOR_CLAIMS[program],
                "expected_direction": test["expected_direction"],
                "n_Control_donors": int(test["n_Control_donors"]),
                "n_DCMP_donors": int(test["n_DCMP_donors"]),
                "primary_difference_DCMP_minus_Control": float(test["difference_DCMP_minus_Control"]),
                "primary_hedges_g_descriptive": float(test["hedges_g"]),
                "primary_direction_supported": direction_supported,
                "formal_inference_eligible": False,
                "primary_P_value": np.nan,
                "primary_FDR": np.nan,
                "planned_genes": ";".join(planned),
                "retained_genes": ";".join(used),
                "excluded_by_author_probe_QC": ";".join(excluded),
                "n_retained_genes_expected_direction": gene_direction_count,
                "n_retained_genes": len(gene_frame),
                "gene_expected_direction_fraction": gene_direction_fraction,
                "n_sensitivity_sets_expected_direction": sensitivity_count,
                "n_sensitivity_sets": len(sensitivity_frame),
                "sensitivity_expected_direction_fraction": sensitivity_fraction,
                "CCL21_selection_endpoint_overlap": overlap,
                "final_evidence_grade": grade,
                "supports_discovery_hypothesis": "descriptive_only" if direction_supported else "no",
                "complete_positive_validation_claim_allowed": False,
                "exact_recommended_wording": wording,
            }
        )

        checks = [
            ("two_of_five_LEC_selection_applied", True, "At least two of the five markers were above AOI-specific LOQ"),
            ("minimum_three_donors_each_group", False, "Only two Control donors passed selection"),
            ("primary_expected_direction", direction_supported, "Descriptive donor effect has the prespecified sign"),
            ("retained_gene_majority_expected_direction", bool(np.isfinite(gene_direction_fraction) and gene_direction_fraction >= 0.5), "At least half of retained genes have expected sign"),
            ("sensitivity_majority_expected_direction", bool(np.isfinite(sensitivity_fraction) and sensitivity_fraction >= 0.5), "At least half of estimable sensitivity sets have expected sign"),
            ("CCL21_endpoint_independent_of_selection", not overlap, "Homeostatic endpoint is not independent because CCL21 was used for selection"),
        ]
        for check, passed, rule in checks:
            audit_rows.append(
                {
                    "program": program,
                    "check": check,
                    "passed": passed,
                    "role": "descriptive_audit_only",
                    "rule": rule,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def niche_decision(niche_tests: pd.DataFrame, associations: pd.DataFrame) -> pd.DataFrame:
    valid = associations.dropna(subset=["Spearman_rho"]).copy()
    if len(valid):
        best = valid.loc[valid["Spearman_rho"].abs().idxmax()]
        best_pair = f"{best['source_score']} -> {best['target_score']}"
        best_rho = float(best["Spearman_rho"])
    else:
        best_pair, best_rho = "not_estimable", np.nan
    return pd.DataFrame(
        [
            {
                "hypothesis": "BMP4 with BMPR1A/BMPR2 and NRP1/NRP2 spatial context",
                "n_donors": int(associations["n_donors"].max()),
                "best_descriptive_association": best_pair,
                "best_descriptive_Spearman_rho": best_rho,
                "formal_P_or_FDR_available": False,
                "final_evidence_grade": "DESCRIPTIVE_ONLY_INSUFFICIENT_DONORS",
                "complete_positive_validation_claim_allowed": False,
                "exact_recommended_wording": (
                    "BMP4/receptor scores were examined descriptively in five selected "
                    "donors. The cohort is insufficient for inferential signaling "
                    "corroboration, and co-expression cannot establish direction or causality."
                ),
            }
        ]
    )


def context_decision(selected_aoi: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        selected_aoi,
        [
            "Clinical_phenotype_LV", "PID", "n_LEC_markers_above_LOQ",
            "LEC_selected_two_of_five", "LEC_markers_above_LOQ",
        ],
        "selected AOI evidence",
    )
    if not as_bool(selected_aoi["LEC_selected_two_of_five"]).all():
        raise ValueError("A Step-13.2 selected AOI does not satisfy the two-of-five rule")
    counts = selected_aoi.groupby("Clinical_phenotype_LV", observed=True).agg(
        n_AOIs=("PID", "size"), n_donors=("PID", "nunique")
    )
    return pd.DataFrame(
        [
            {
                "selection_markers": ";".join(LEC_MARKERS),
                "minimum_markers_above_AOI_LOQ": MINIMUM_MARKERS,
                "n_selected_AOIs": len(selected_aoi),
                "n_Control_AOIs": int(counts.loc["Control", "n_AOIs"]),
                "n_Control_donors": int(counts.loc["Control", "n_donors"]),
                "n_DCMP_AOIs": int(counts.loc["DCMP", "n_AOIs"]),
                "n_DCMP_donors": int(counts.loc["DCMP", "n_donors"]),
                "minimum_observed_markers_above_LOQ": int(selected_aoi["n_LEC_markers_above_LOQ"].min()),
                "CCL21_selection_endpoint_overlap": True,
                "final_evidence_grade": "TWO_OF_FIVE_MARKER_SELECTED_MIXED_AOI_CONTEXT",
                "exact_recommended_wording": (
                    "Seven LV vessel AOIs met the prespecified two-of-five LEC-marker "
                    "rule. GeoMx AOIs are mixtures and are not purified individual LECs."
                ),
            }
        ]
    )


def final_decisions(
    programs: pd.DataFrame, niche: pd.DataFrame, context: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for row in programs.itertuples(index=False):
        rows.append(
            {
                "decision_type": "prespecified_DCM_program",
                "hypothesis": row.mentor_claim,
                "final_evidence_grade": row.final_evidence_grade,
                "formal_validation_allowed": False,
                "recommended_wording": row.exact_recommended_wording,
                "canonical_source": "13_4_program_evidence_grades.csv",
            }
        )
    rows.extend(
        [
            {
                "decision_type": "niche_signaling_context",
                "hypothesis": niche.iloc[0]["hypothesis"],
                "final_evidence_grade": niche.iloc[0]["final_evidence_grade"],
                "formal_validation_allowed": False,
                "recommended_wording": niche.iloc[0]["exact_recommended_wording"],
                "canonical_source": "13_4_niche_evidence_grade.csv",
            },
            {
                "decision_type": "LEC_spatial_context",
                "hypothesis": "AOIs satisfy the requested LEC marker-selection rule",
                "final_evidence_grade": context.iloc[0]["final_evidence_grade"],
                "formal_validation_allowed": False,
                "recommended_wording": context.iloc[0]["exact_recommended_wording"],
                "canonical_source": "13_4_LEC_context_evidence_grade.csv",
            },
            {
                "decision_type": "SCENIC_TF_context",
                "hypothesis": "TF transcript expression validates SCENIC regulon activity",
                "final_evidence_grade": "EXPRESSION_ONLY_NOT_REGULON_VALIDATION",
                "formal_validation_allowed": False,
                "recommended_wording": (
                    "EPAS1 transcript expression is secondary expression context only; "
                    "GeoMx expression does not directly validate SCENIC regulon activity."
                ),
                "canonical_source": "13_3_gene_level_donor_tests.csv",
            },
        ]
    )
    return pd.DataFrame(rows)


def plot_final_summary(programs: pd.DataFrame, path: Path, dpi: int) -> None:
    frame = programs.set_index("program").loc[PROGRAM_ORDER].reset_index()
    y = np.arange(len(frame))[::-1]
    fig, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    for position, row in zip(y, frame.itertuples(index=False)):
        color = GRADE_COLORS[row.final_evidence_grade]
        axis.scatter(
            row.primary_difference_DCMP_minus_Control,
            position,
            s=85,
            color=color,
            edgecolor="white",
            zorder=3,
        )
        axis.text(
            frame["primary_difference_DCMP_minus_Control"].max() + 0.07,
            position,
            "expected direction; descriptive only"
            if row.primary_direction_supported
            else "opposite direction; descriptive only",
            va="center",
            fontsize=8,
        )
    axis.axvline(0, color="black", linewidth=0.9)
    axis.set_yticks(y, [DISPLAY_NAMES[p] for p in frame["program"]])
    axis.set_xlabel("Mean donor score difference: DCMP - Control")
    axis.set_title("DCM program directions after two-of-five LEC-marker selection")
    axis.text(
        0.01,
        -0.16,
        "No confidence intervals or P values: 2 Control donors < required minimum of 3.",
        transform=axis.transAxes,
        fontsize=9,
    )
    axis.spines[["top", "right"]].set_visible(False)
    left = float(frame["primary_difference_DCMP_minus_Control"].min())
    right = float(frame["primary_difference_DCMP_minus_Control"].max())
    axis.set_xlim(left - 0.12, right + max(0.7, (right - left) * 0.8))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_evidence_matrix(audit: pd.DataFrame, path: Path, dpi: int) -> None:
    checks = audit["check"].drop_duplicates().tolist()
    pivot = audit.pivot(index="program", columns="check", values="passed").reindex(
        index=PROGRAM_ORDER, columns=checks
    )
    values = pivot.astype(float).to_numpy()
    fig, axis = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    axis.imshow(
        values,
        aspect="auto",
        cmap=ListedColormap(["#D9D9D9", "#009E73"]),
        vmin=0,
        vmax=1,
    )
    axis.set_xticks(
        np.arange(len(checks)),
        [item.replace("_", " ") for item in checks],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(np.arange(len(PROGRAM_ORDER)), [DISPLAY_NAMES[p] for p in PROGRAM_ORDER])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                "PASS" if values[row, column] == 1 else "NO",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if values[row, column] == 1 else "#555555",
            )
    axis.set_title("Descriptive evidence audit; no check overrides donor insufficiency")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_gene_components(genes: pd.DataFrame, path: Path, dpi: int) -> None:
    frame = genes.copy()
    frame["program_order"] = frame["program"].map({p: i for i, p in enumerate(PROGRAM_ORDER)})
    frame = frame.sort_values(["program_order", "gene"]).reset_index(drop=True)
    y = np.arange(len(frame))[::-1]
    fig, axis = plt.subplots(figsize=(9.5, 7), constrained_layout=True)
    for position, row in zip(y, frame.itertuples(index=False)):
        color = "#009E73" if bool(row.direction_concordant) else "#D55E00"
        axis.scatter(
            row.difference_DCMP_minus_Control,
            position,
            s=48,
            color=color,
            edgecolor="white",
            zorder=3,
        )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, frame["gene"])
    axis.set_xlabel("DCMP - Control donor gene-z difference\nGreen = expected sign; orange = opposite sign")
    axis.set_title("Retained mentor targets: descriptive effects only")
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_final_statement(
    programs: pd.DataFrame, niche: pd.DataFrame, context: pd.DataFrame
) -> str:
    del niche
    effect_lines = []
    for row in programs.itertuples(index=False):
        direction = "expected" if row.primary_direction_supported else "opposite"
        effect_lines.append(
            f"- {DISPLAY_NAMES[row.program]}: difference={row.primary_difference_DCMP_minus_Control:.3f}; "
            f"{direction} direction; {row.final_evidence_grade}."
        )
    return f"""FINAL DCM SPATIAL VALIDATION STATEMENT
======================================

Canonical conclusion
--------------------
GSE271676 provides descriptive spatial consistency with the mentor-specified
DCM program shifts after the requested LEC selection, but it does not provide
formal validation. Step 13.2 selected {int(context.iloc[0]['n_selected_AOIs'])}
LV vessel AOIs requiring at least two of CCL21, LYVE1, PDPN, FLT4, and PROX1
above the AOI-specific LOQ. These comprise {int(context.iloc[0]['n_Control_AOIs'])}
Control AOIs from only {int(context.iloc[0]['n_Control_donors'])} donors and
{int(context.iloc[0]['n_DCMP_AOIs'])} DCMP AOIs from
{int(context.iloc[0]['n_DCMP_donors'])} donors. The Control group therefore
fails the prespecified minimum of {MINIMUM_DONORS_PER_GROUP} donors.

Program results
---------------
{chr(10).join(effect_lines)}

All four scores point in their prespecified directions in this selected
cohort. No P values, FDRs, bootstrap confidence intervals, or positive formal
validation grades are reported because donor replication is insufficient.

Required caveats
----------------
- CCL21 is used for LEC selection and in the MMRN1/CCL21 homeostatic endpoint;
  that result is not independent of selection.
- SEMA3D and PDGFC failed the authors' 1% probe-QC filter, leaving reduced
  State0-like and State3-like panels.
- GeoMx AOIs are mixed tissue regions, not individual or purified LECs.
- BMP4/BMPR/NRP results are descriptive co-expression, not signaling direction
  or causality.
- EPAS1 transcript expression is not SCENIC regulon validation.

Required reporting language
---------------------------
- Complete DCM spatial validation: NO.
- All four DCM program shifts: DESCRIPTIVE EXPECTED-DIRECTION SIGNALS ONLY.
- BMP4/receptor signaling: NOT FORMALLY TESTABLE IN THIS SELECTED COHORT.
- SCENIC TF layer: EXPRESSION CONTEXT ONLY, NOT REGULON VALIDATION.

Canonical decision table: 13_4_FINAL_VALIDATION_DECISIONS.csv
"""


def main() -> int:
    args = parse_args()
    project_dir = find_project_dir()
    step13_2 = resolve_path(args.step13_2_root, project_dir)
    step13_3 = resolve_path(args.step13_3_root, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    files = OrderedDict(
        [
            ("registry", step13_3 / "13_3_program_registry.csv"),
            ("primary", step13_3 / "13_3_primary_program_tests.csv"),
            ("sensitivity", step13_3 / "13_3_sensitivity_program_tests.csv"),
            ("genes", step13_3 / "13_3_gene_level_donor_tests.csv"),
            ("niche_tests", step13_3 / "13_3_secondary_niche_group_tests.csv"),
            ("associations", step13_3 / "13_3_niche_program_associations.csv"),
            ("selected_aoi", step13_2 / "13_2_selected_AOI_LEC_evidence.csv"),
        ]
    )
    for label, path in files.items():
        validate_file(path, label)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is nonempty; rerun with --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    tables = {label: pd.read_csv(path) for label, path in files.items()}

    require_columns(
        tables["primary"],
        [
            "program", "difference_DCMP_minus_Control", "direction_concordant",
            "minimum_donor_threshold_met", "n_Control_donors", "n_DCMP_donors",
        ],
        "primary tests",
    )
    if set(tables["primary"]["program"]) != set(PROGRAM_ORDER):
        raise ValueError("Primary-program identities differ from the locked registry")
    if as_bool(tables["primary"]["minimum_donor_threshold_met"]).any():
        raise ValueError("Current report expects a donor-insufficient descriptive cohort")
    if set(SENSITIVITY_SETS) - set(tables["sensitivity"]["analysis_set"]):
        raise ValueError("A prespecified Step-13.3 sensitivity set is absent")

    programs, audit = grade_programs(
        tables["registry"], tables["primary"], tables["sensitivity"], tables["genes"]
    )
    niche = niche_decision(tables["niche_tests"], tables["associations"])
    context = context_decision(tables["selected_aoi"])
    decisions = final_decisions(programs, niche, context)

    programs.to_csv(output_root / "13_4_program_evidence_grades.csv", index=False)
    audit.to_csv(output_root / "13_4_evidence_rule_audit.csv", index=False)
    niche.to_csv(output_root / "13_4_niche_evidence_grade.csv", index=False)
    context.to_csv(output_root / "13_4_LEC_context_evidence_grade.csv", index=False)
    decisions.to_csv(output_root / "13_4_FINAL_VALIDATION_DECISIONS.csv", index=False)

    limitations = pd.DataFrame(
        [
            {"limitation": "insufficient_Control_donors", "impact": "Only two Control donors pass the two-of-five selection rule.", "required_action": "Report descriptive directions only; no P value, FDR, or formal validation claim."},
            {"limitation": "mixed_AOI_observation", "impact": "GeoMx AOIs are tissue mixtures, not individual or purified LECs.", "required_action": "Use LEC-marker-selected AOI wording."},
            {"limitation": "CCL21_selection_outcome_overlap", "impact": "CCL21 contributes to both selection and the homeostatic endpoint.", "required_action": "Do not present the homeostatic result as selection-independent confirmation."},
            {"limitation": "incomplete_target_panels", "impact": "SEMA3D and PDGFC failed the authors' 1% probe-QC filter.", "required_action": "Document reduced State0/State3 panels; do not impute genes."},
            {"limitation": "no_continuous_coordinates", "impact": "Downloaded tables do not enable neighborhood-distance analysis.", "required_action": "Describe niche findings as within-AOI co-expression only."},
            {"limitation": "TF_expression_not_regulon", "impact": "GeoMx transcript abundance does not measure SCENIC regulon activity.", "required_action": "Treat EPAS1 expression as secondary context."},
        ]
    )
    limitations.to_csv(output_root / "13_4_design_limitations.csv", index=False)

    final_statement = build_final_statement(programs, niche, context)
    (output_root / "13_4_FINAL_VALIDATION_STATEMENT.txt").write_text(
        final_statement, encoding="utf-8"
    )
    locator = pd.DataFrame(
        [
            {"priority": 1, "file": "13_4_FINAL_VALIDATION_DECISIONS.csv", "use": "Canonical final decision for every mentor hypothesis"},
            {"priority": 2, "file": "13_4_FINAL_VALIDATION_STATEMENT.txt", "use": "Ready-to-use final narrative conclusion"},
            {"priority": 3, "file": "figures/13_4_final_program_evidence.png", "use": "Main descriptive program figure"},
            {"priority": 4, "file": "13_4_program_evidence_grades.csv", "use": "Numerical program directions and descriptive grades"},
            {"priority": 5, "file": "13_4_design_limitations.csv", "use": "Required interpretation boundaries"},
            {"priority": 6, "file": "../13_3_prespecified_DCM_program_validation/13_3_primary_program_tests.csv", "use": "Underlying donor-level descriptive effect table"},
        ]
    )
    locator.to_csv(output_root / "13_4_WHICH_RESULTS_TO_USE.csv", index=False)

    readme = f"""READ THIS FIRST: FINAL DCM SPATIAL VALIDATION
================================================

Use 13_4_FINAL_VALIDATION_DECISIONS.csv as the canonical decision table and
13_4_FINAL_VALIDATION_STATEMENT.txt as the final narrative.

Step 13.4 does not perform a new biological test. It reports the Step-13.3
donor-level directions after Step 13.2 selected AOIs using at least two of
CCL21, LYVE1, PDPN, FLT4, and PROX1 above AOI-specific LOQ.

The current selected cohort has {int(context.iloc[0]['n_Control_donors'])}
Control and {int(context.iloc[0]['n_DCMP_donors'])} DCMP donors. Because the
Control group is below the required minimum of {MINIMUM_DONORS_PER_GROUP}, no
formal hypothesis test or positive validation claim is permitted. The four
programs point in the expected direction and are descriptive signals only.

CCL21 is also part of the homeostatic outcome, so that endpoint is not
selection-independent. TF transcript expression remains secondary context and
does not validate SCENIC regulon activity.
"""
    (output_root / "13_4_READ_ME_FINAL_VALIDATION.txt").write_text(
        readme, encoding="utf-8"
    )

    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        jobs = [
            ("13_4_final_program_evidence.png", lambda path: plot_final_summary(programs, path, args.dpi), "Canonical descriptive program-effect figure"),
            ("13_4_evidence_check_matrix.png", lambda path: plot_evidence_matrix(audit, path, args.dpi), "Audit matrix including donor insufficiency and selection overlap"),
            ("13_4_gene_component_evidence.png", lambda path: plot_gene_components(tables["genes"], path, args.dpi), "Retained mentor-gene descriptive effects"),
        ]
        for filename, function, role in jobs:
            path = figures_dir / filename
            function(path)
            figure_manifest.append({"figure": str(path.relative_to(output_root)), "role": role})
    pd.DataFrame(figure_manifest).to_csv(output_root / "13_4_figure_manifest.csv", index=False)

    config = {
        "script_version": SCRIPT_VERSION,
        "step13_2_root": str(step13_2),
        "step13_3_root": str(step13_3),
        "output_root": str(output_root),
        "LEC_selection_markers": LEC_MARKERS,
        "minimum_markers_above_AOI_LOQ": MINIMUM_MARKERS,
        "minimum_donors_per_group": MINIMUM_DONORS_PER_GROUP,
        "formal_inference_performed": False,
        "CCL21_selection_endpoint_overlap": True,
        "new_biological_test_performed": False,
    }
    (output_root / "13_4_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    run_summary = pd.DataFrame(
        [
            {"metric": "script_version", "value": SCRIPT_VERSION},
            {"metric": "status", "value": "completed"},
            {"metric": "new_biological_test_performed", "value": False},
            {"metric": "formal_inference_performed", "value": False},
            {"metric": "n_selected_AOIs", "value": int(context.iloc[0]["n_selected_AOIs"])},
            {"metric": "Control_donors", "value": int(context.iloc[0]["n_Control_donors"])},
            {"metric": "DCMP_donors", "value": int(context.iloc[0]["n_DCMP_donors"])},
            {"metric": "n_programs_expected_direction", "value": int(programs["primary_direction_supported"].sum())},
            {"metric": "n_formally_validated_programs", "value": 0},
            {"metric": "overall_complete_DCM_spatial_validation", "value": "NO"},
            {"metric": "canonical_decision_file", "value": "13_4_FINAL_VALIDATION_DECISIONS.csv"},
        ]
    )
    run_summary.to_csv(output_root / "13_4_run_summary.csv", index=False)

    print(f"Step 13.4 complete: {output_root}")
    print(
        programs[
            [
                "program", "primary_difference_DCMP_minus_Control",
                "primary_direction_supported", "final_evidence_grade",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
