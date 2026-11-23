#!/usr/bin/env python3
"""Create Figure 1A: study design and analytical workflow.

The schematic summarizes the completed aging and cardiovascular-disease
analysis branches without implying that their LEC state spaces or downstream
inference were combined. It uses the current run summaries and final spatial
evidence tables only as workflow provenance; no biological statistic is
recomputed here.

Outputs
-------
new_result2/Fig01A_research_workflow/
    Fig01A_research_workflow.png
    Fig01A_research_workflow.pdf
    Fig01A_research_workflow.jpg
    Fig01A_source_data.csv
    Fig01A_input_manifest.csv
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import pandas as pd  # noqa: E402


SCRIPT_VERSION = "Fig01A_parallel_aging_disease_workflow_v2_2026_08_31"
OUTPUT_RELATIVE = Path("new_result2/Fig01A_research_workflow")

INPUTS = {
    "aging_cohort_qc": (
        Path("new_result2_aging/03/03_master_run_summary_age_quartile_QC.csv"),
        "aging cohort definition, cell-level QC, Harmony, Leiden and UMAP",
    ),
    "disease_cohort_qc": (
        Path("new_result2_disease/03/03_master_run_summary.csv"),
        "active disease cohorts and cell-level QC workflow",
    ),
    "aging_EC_LEC_refinement": (
        Path(
            "new_result2_aging/06_EC_recluster/dataset13/"
            "06_EC_recluster_run_summary.csv"
        ),
        "aging EC-only reclustering and LEC identification",
    ),
    "disease_EC_LEC_refinement": (
        Path(
            "new_result2_disease/06_EC_recluster/"
            "06_EC_recluster_run_summary.csv"
        ),
        "disease EC-only reclustering and LEC identification",
    ),
    "aging_LEC_state_discovery": (
        Path(
            "new_result2_aging/08_combined_LEC_recluster/dataset13/"
            "08_run_summary.csv"
        ),
        "aging LEC-only integration and state discovery",
    ),
    "disease_LEC_state_discovery": (
        Path(
            "new_result2_disease/08_combined_LEC_recluster/"
            "08_run_summary.csv"
        ),
        "disease LEC-only integration and state discovery",
    ),
    "aging_donor_filter": (
        Path(
            "new_result2_aging/08_7_min25_LEC_filtered_age_analysis/dataset13/"
            "08_7_run_summary.csv"
        ),
        "aging donor-level minimum-25 LEC analysis tier",
    ),
    "disease_donor_filter": (
        Path(
            "new_result2_disease/08_3_min25_LEC_filtered_analysis/"
            "08_3_run_summary.csv"
        ),
        "primary disease donor-level minimum-25 LEC tier",
    ),
    "HFpEF_min10": (
        Path(
            "new_result2_disease/12_NicheNet_donor_pseudobulk/"
            "dataset11_HFpEF_vs_NF_min10/12_run_summary.csv"
        ),
        "dataset11 HFpEF-versus-NF exploratory minimum-10 tier",
    ),
    "HFpEF_min15": (
        Path(
            "new_result2_disease/12_NicheNet_donor_pseudobulk/"
            "dataset11_HFpEF_vs_NF_min15/12_run_summary.csv"
        ),
        "dataset11 HFpEF-versus-NF minimum-15 sensitivity tier",
    ),
    "aging_GSEA": (
        Path(
            "new_result2_aging/09_LEC_state_GSEA/dataset13/"
            "09_figure_manifest.csv"
        ),
        "aging LEC-state marker and preranked GSEA characterization",
    ),
    "disease_GSEA": (
        Path("new_result2_disease/09_LEC_state_GSEA/09_figure_manifest.csv"),
        "disease LEC-state marker and preranked GSEA characterization",
    ),
    "aging_SCENIC": (
        Path(
            "new_result2_aging/11_SCENIC_analysis/dataset13/"
            "11_figure_manifest.csv"
        ),
        "aging pySCENIC regulon inference and robustness",
    ),
    "disease_SCENIC": (
        Path("new_result2_disease/11_SCENIC_analysis/11_figure_manifest.csv"),
        "disease pySCENIC regulon inference and robustness",
    ),
    "aging_communication": (
        Path(
            "new_result2_aging/13_NicheNet_donor_pseudobulk/dataset13/"
            "13_run_summary.csv"
        ),
        "aging donor-pseudobulk edgeR and MultiNicheNet",
    ),
    "disease_communication": (
        Path(
            "new_result2_disease/12_NicheNet_donor_pseudobulk/"
            "12_run_summary.csv"
        ),
        "disease donor-pseudobulk edgeR and MultiNicheNet",
    ),
    "aging_spatial_validation": (
        Path(
            "new_result2_spatial_aging/"
            "12_4_spatial_aging_LEC_program_robustness_and_evidence_grade/"
            "12_4_FINAL_VALIDATION_DECISIONS.csv"
        ),
        "independent aging spatial program validation and evidence grading",
    ),
    "disease_spatial_validation": (
        Path(
            "new_result2_spatial_disease/13_4_final_DCM_spatial_validation/"
            "13_4_FINAL_VALIDATION_DECISIONS.csv"
        ),
        "independent DCM GeoMx program validation and evidence grading",
    ),
}

TEXT = "#20252A"
SUBTEXT = "#59636C"
NAVY = "#22364D"
SHARED_FILL = "#F3F6F7"
SHARED_EDGE = "#71808B"
AGING = "#3973A9"
AGING_FILL = "#E8F1F8"
DISEASE = "#C26443"
DISEASE_FILL = "#FAECE7"
VALIDATION = "#287C78"
VALIDATION_FILL = "#E7F4F2"


def project_root() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_provenance(root: Path) -> dict[str, Path]:
    paths = {name: root / relative for name, (relative, _) in INPUTS.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing workflow provenance files: {missing}")

    aging_filter = pd.read_csv(paths["aging_donor_filter"])
    disease_filter = pd.read_csv(paths["disease_donor_filter"])
    min10 = pd.read_csv(paths["HFpEF_min10"])
    min15 = pd.read_csv(paths["HFpEF_min15"])
    if len(aging_filter) != 1 or int(aging_filter.iloc[0]["minimum_LECs"]) != 25:
        raise ValueError("The current aging donor filter is not minimum 25.")
    if len(disease_filter) != 1 or int(disease_filter.iloc[0]["minimum_LECs"]) != 25:
        raise ValueError("The current primary disease donor filter is not minimum 25.")
    expected = {
        10: (min10, 6, 7, 349),
        15: (min15, 4, 5, 301),
    }
    for threshold, (summary, n_nf, n_hfpef, n_cells) in expected.items():
        if len(summary) != 1:
            raise ValueError(f"Expected one dataset11 summary row at {threshold}.")
        row = summary.iloc[0]
        checks = [
            int(row["receiver_min_cells"]) == threshold,
            int(row["n_reference_donors"]) == n_nf,
            int(row["n_disease_donors"]) == n_hfpef,
            int(row["n_receiver_cells"]) == n_cells,
            str(row["validation_status"]).lower() == "passed",
            str(row["status"]).lower() == "completed",
        ]
        if not all(checks):
            raise ValueError(f"Dataset11 threshold validation failed at {threshold}.")
    return paths


def rounded_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    *,
    facecolor: str = SHARED_FILL,
    edgecolor: str = SHARED_EDGE,
    title_color: str = TEXT,
    title_size: float = 11.5,
    subtitle_size: float = 8.5,
    linewidth: float = 1.25,
    rounding: float = 0.018,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.008,rounding_size={rounding}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=2,
        )
    )
    title_y = y + height * (0.63 if subtitle else 0.50)
    axis.text(
        x + width / 2,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        zorder=3,
    )
    if subtitle:
        axis.text(
            x + width / 2,
            y + height * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=SUBTEXT,
            linespacing=1.25,
            zorder=3,
        )


def connector(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY,
    arrow: bool = True,
    linewidth: float = 1.8,
) -> None:
    style = "-|>" if arrow else "-"
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            color=color,
            linewidth=linewidth,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0",
            zorder=1,
        )
    )


def draw_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    figure, axis = plt.subplots(figsize=(14.0, 8.4), facecolor="white")
    figure.subplots_adjust(left=0.025, right=0.985, top=0.985, bottom=0.035)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(0.012, 0.972, "A", fontsize=20, fontweight="bold", color=TEXT, va="top")
    axis.text(
        0.055,
        0.972,
        "Study design and analytical workflow",
        fontsize=19,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    axis.text(
        0.055,
        0.935,
        "Parallel analysis of aging and cardiovascular-disease LEC states",
        fontsize=9.8,
        color=SUBTEXT,
        va="top",
    )

    # Cohort container and cohort lanes.
    rounded_box(
        axis,
        0.10,
        0.795,
        0.80,
        0.115,
        "",
        "",
        facecolor="white",
        edgecolor=SHARED_EDGE,
        title_size=12.5,
    )
    axis.text(
        0.50,
        0.885,
        "Human cardiac sc/snRNA-seq cohorts",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=TEXT,
        zorder=3,
    )
    rounded_box(
        axis,
        0.135,
        0.805,
        0.33,
        0.050,
        "Physiological aging",
        "NF • LV/AP • age ≥30 • Q1–Q4",
        facecolor=AGING_FILL,
        edgecolor=AGING,
        title_color=AGING,
        title_size=9.5,
        subtitle_size=7.2,
        linewidth=1.1,
        rounding=0.012,
    )
    rounded_box(
        axis,
        0.535,
        0.805,
        0.33,
        0.050,
        "Cardiovascular disease",
        "DCM • HCM • DoxCM • HFpEF + NF",
        facecolor=DISEASE_FILL,
        edgecolor=DISEASE,
        title_color=DISEASE,
        title_size=9.5,
        subtitle_size=7.2,
        linewidth=1.1,
        rounding=0.012,
    )

    connector(axis, (0.50, 0.795), (0.50, 0.755))
    rounded_box(
        axis,
        0.14,
        0.665,
        0.72,
        0.090,
        "Cell-level preprocessing and annotation",
        "Quality control • normalization • Harmony • clustering • annotation",
        title_size=11.7,
        subtitle_size=8.0,
    )

    connector(axis, (0.50, 0.665), (0.50, 0.625))
    rounded_box(
        axis,
        0.14,
        0.535,
        0.72,
        0.090,
        "EC refinement and LEC identification",
        "EC reclustering • lymphatic EC selection • marker and doublet review",
        title_size=11.7,
        subtitle_size=8.0,
    )

    # Split into independent aging and disease state discovery.
    connector(axis, (0.50, 0.535), (0.50, 0.505), arrow=False)
    connector(axis, (0.30, 0.505), (0.70, 0.505), arrow=False)
    connector(axis, (0.30, 0.505), (0.30, 0.475), color=AGING)
    connector(axis, (0.70, 0.505), (0.70, 0.475), color=DISEASE)
    rounded_box(
        axis,
        0.08,
        0.375,
        0.39,
        0.100,
        "Aging LEC states",
        "LEC-only integration • donor-normalized age analysis",
        facecolor=AGING_FILL,
        edgecolor=AGING,
        title_color=AGING,
        title_size=11.2,
        subtitle_size=8.2,
        linewidth=1.4,
    )
    rounded_box(
        axis,
        0.53,
        0.375,
        0.39,
        0.100,
        "Disease LEC states",
        "LEC-only integration • within-dataset donor analysis",
        facecolor=DISEASE_FILL,
        edgecolor=DISEASE,
        title_color=DISEASE,
        title_size=11.2,
        subtitle_size=8.2,
        linewidth=1.4,
    )

    connector(axis, (0.275, 0.375), (0.275, 0.335), color=AGING)
    connector(axis, (0.725, 0.375), (0.725, 0.335), color=DISEASE)
    rounded_box(
        axis,
        0.075,
        0.190,
        0.85,
        0.145,
        "",
        "",
        facecolor="white",
        edgecolor=SHARED_EDGE,
        title_size=11.8,
        subtitle_size=8.1,
    )
    axis.text(
        0.50,
        0.305,
        "Parallel downstream analyses",
        ha="center",
        va="center",
        fontsize=11.8,
        fontweight="bold",
        color=TEXT,
        zorder=3,
    )
    axis.text(
        0.50,
        0.279,
        "Donor-aware and performed separately in aging and disease",
        ha="center",
        va="center",
        fontsize=8.1,
        color=SUBTEXT,
        zorder=3,
    )
    module_specs = [
        (0.105, "State\nabundance"),
        (0.315, "Marker genes\n+ GSEA"),
        (0.525, "pySCENIC\nregulons"),
        (0.735, "edgeR +\nMultiNicheNet"),
    ]
    for x, label in module_specs:
        rounded_box(
            axis,
            x,
            0.205,
            0.16,
            0.058,
            label,
            "",
            facecolor=SHARED_FILL,
            edgecolor=SHARED_EDGE,
            title_size=8.7,
            linewidth=0.9,
            rounding=0.012,
        )

    connector(axis, (0.50, 0.190), (0.50, 0.160), color=VALIDATION)
    rounded_box(
        axis,
        0.18,
        0.045,
        0.64,
        0.115,
        "",
        "",
        facecolor=VALIDATION_FILL,
        edgecolor=VALIDATION,
        title_color=VALIDATION,
        title_size=11.4,
        subtitle_size=8.0,
        linewidth=1.4,
    )
    axis.text(
        0.50,
        0.139,
        "Independent spatial validation",
        ha="center",
        va="center",
        fontsize=11.4,
        fontweight="bold",
        color=VALIDATION,
        zorder=3,
    )
    axis.text(
        0.50,
        0.114,
        "Prespecified LEC programs • robustness and evidence grading",
        ha="center",
        va="center",
        fontsize=7.2,
        color=SUBTEXT,
        zorder=3,
    )
    rounded_box(
        axis,
        0.215,
        0.057,
        0.25,
        0.038,
        "Aging spatial transcriptomics",
        "",
        facecolor="white",
        edgecolor=AGING,
        title_color=AGING,
        title_size=8.2,
        linewidth=0.9,
        rounding=0.010,
    )
    rounded_box(
        axis,
        0.535,
        0.057,
        0.25,
        0.038,
        "DCM GeoMx",
        "",
        facecolor="white",
        edgecolor=DISEASE,
        title_color=DISEASE,
        title_size=8.2,
        linewidth=0.9,
        rounding=0.010,
    )

    axis.text(
        0.50,
        0.012,
        "",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=SUBTEXT,
    )
    return figure


def source_records() -> pd.DataFrame:
    nodes = [
        ("cohort", "shared", 1, "Human cardiac sc/snRNA-seq cohorts", "Aging and disease cohorts"),
        ("aging_cohort", "aging", 2, "Physiological aging", "NF; LV/AP; age >=30; donor quartiles Q1-Q4"),
        ("disease_cohort", "disease", 2, "Cardiovascular disease", "DCM; HCM; DoxCM; HFpEF; within-dataset NF"),
        ("preprocessing", "shared", 3, "Cell-level preprocessing and annotation", "QC; normalization; HVGs/PCA; Harmony; Leiden/UMAP; marker genes; CellTypist"),
        ("LEC_identification", "shared", 4, "EC refinement and LEC identification", "EC reclustering; EC8 lymphatic majority label; LEC markers; identity/doublet review"),
        ("aging_states", "aging", 5, "Aging LEC states", "Independent LEC-only integration and donor-normalized age comparisons"),
        ("disease_states", "disease", 5, "Disease LEC states", "Independent LEC-only integration and within-dataset donor comparisons"),
        ("functional_characterization", "parallel", 6, "Parallel downstream analyses", "Donor-aware and performed separately in aging and disease"),
        ("state_abundance", "parallel", 7, "State abundance", "Performed independently in aging and disease"),
        ("GSEA", "parallel", 7, "Marker genes and GSEA", "State characterization; not a disease-effect test"),
        ("SCENIC", "parallel", 7, "pySCENIC regulons", "Regulon activity and robustness"),
        ("communication", "parallel", 7, "edgeR and MultiNicheNet", "Raw-count donor pseudobulk and communication prioritization"),
        ("spatial", "validation", 8, "Independent spatial validation", "Aging spatial transcriptomics and DCM GeoMx; robustness/evidence grading"),
    ]
    records: list[dict[str, object]] = []
    for element_id, lane, order, title, detail in nodes:
        records.append(
            {
                "script_version": SCRIPT_VERSION,
                "record_type": "workflow_node",
                "element_id": element_id,
                "lane": lane,
                "display_order": order,
                "title": title,
                "detail": detail,
                "source_element": "",
                "target_element": "",
            }
        )
    edges = [
        ("cohort", "preprocessing"),
        ("preprocessing", "LEC_identification"),
        ("LEC_identification", "aging_states"),
        ("LEC_identification", "disease_states"),
        ("aging_states", "functional_characterization"),
        ("disease_states", "functional_characterization"),
        ("functional_characterization", "state_abundance"),
        ("functional_characterization", "GSEA"),
        ("functional_characterization", "SCENIC"),
        ("functional_characterization", "communication"),
        ("state_abundance", "spatial"),
        ("GSEA", "spatial"),
        ("SCENIC", "spatial"),
        ("communication", "spatial"),
    ]
    for index, (source, target) in enumerate(edges, start=1):
        records.append(
            {
                "script_version": SCRIPT_VERSION,
                "record_type": "workflow_edge",
                "element_id": f"edge_{index:02d}",
                "lane": "workflow",
                "display_order": index,
                "title": "",
                "detail": "",
                "source_element": source,
                "target_element": target,
            }
        )
    records.append(
        {
            "script_version": SCRIPT_VERSION,
            "record_type": "analysis_tier_note",
            "element_id": "donor_cutoffs",
            "lane": "donor_inference",
            "display_order": 11,
            "title": "Donor analysis thresholds",
            "detail": "Primary >=25 final LECs/donor; dataset11 HFpEF-NF exploratory >=10 and sensitivity >=15; symmetric within comparison",
            "source_element": "",
            "target_element": "",
        }
    )
    return pd.DataFrame.from_records(records)


def input_manifest(root: Path, paths: dict[str, Path]) -> pd.DataFrame:
    records = []
    for name, path in paths.items():
        relative, purpose = INPUTS[name]
        stat = path.stat()
        records.append(
            {
                "script_version": SCRIPT_VERSION,
                "input_id": name,
                "input_file": relative.as_posix(),
                "input_type": path.suffix.lower().lstrip("."),
                "used_for": purpose,
                "use_mode": "workflow provenance and current-result validation",
                "size_bytes": stat.st_size,
                "last_modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> int:
    root = project_root()
    paths = validate_provenance(root)
    output = root / OUTPUT_RELATIVE
    output.mkdir(parents=True, exist_ok=True)

    source_records().to_csv(output / "Fig01A_source_data.csv", index=False)
    input_manifest(root, paths).to_csv(
        output / "Fig01A_input_manifest.csv", index=False
    )

    figure = draw_figure()
    figure.savefig(
        output / "Fig01A_research_workflow.pdf",
        format="pdf",
        metadata={
            "Title": "Figure 1A: study design and analytical workflow",
            "Author": "Current-result figure pipeline",
            "Subject": SCRIPT_VERSION,
        },
    )
    figure.savefig(output / "Fig01A_research_workflow.png", dpi=450)
    figure.savefig(
        output / "Fig01A_research_workflow.jpg",
        dpi=450,
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)

    print(f"Completed {SCRIPT_VERSION}")
    print(f"Output: {output}")
    print(f"Workflow provenance files validated: {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
