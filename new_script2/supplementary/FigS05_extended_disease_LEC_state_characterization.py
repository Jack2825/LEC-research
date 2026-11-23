#!/usr/bin/env python3
"""Build Supplementary Figure S5: extended disease LEC-state analyses.

The six panels extend main Figure 4 with donor-level, less-abundant-state,
marker/pathway, and focused State 4/7 identity and technical QC results.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = (
    ROOT
    / "new_result2"
    / "supplementary"
    / "FigS05_extended_disease_LEC_state_characterization"
)

DISEASE_ROOT = ROOT / "new_result2_disease"
STEP08 = DISEASE_ROOT / "08_combined_LEC_recluster"
IDENTITY_QC = DISEASE_ROOT / "08_1_LEC_identity_doublet_QC" / "tables"
FILTERED = DISEASE_ROOT / "08_3_min25_LEC_filtered_analysis" / "tables"
GSEA_DIR = DISEASE_ROOT / "09_LEC_state_GSEA"

H5AD = STEP08 / "08_combined_final_LEC_reclustered.h5ad"
CSV_INPUTS = {
    "all_state_effects": ROOT
    / "new_result2"
    / "disease"
    / "FigS04_disease_all_state_effects"
    / "FigS04_source_data.csv",
    "main_effect_source": ROOT
    / "new_result2"
    / "disease"
    / "Fig04D_disease_representative_state_effects"
    / "Fig04D_source_data.csv",
    "retained_donor_states": FILTERED / "08_3_retained_donor_state_counts_percentages.csv.gz",
    "main_marker_pathway_source": ROOT
    / "new_result2"
    / "disease"
    / "Fig04C_disease_marker_GSEA_map"
    / "Fig04C_source_data.csv",
    "state_top_markers": STEP08 / "08_state_top25_markers_long.csv",
    "gsea_top_pathways": GSEA_DIR / "09_GSEA_top_pathways_per_state.csv",
    "marker_expression_qc": IDENTITY_QC / "08_1_marker_expression_by_state.csv",
    "state_identity_qc": IDENTITY_QC / "08_1_state_identity_summary.csv",
    "state_dataset_donor_qc": STEP08 / "08_state_dataset_donor_QC.csv",
    "cell_identity_doublet_qc": IDENTITY_QC / "08_1_LEC_cell_identity_doublet_QC.csv.gz",
    "state_colors": STEP08 / "08_state_color_mapping.csv",
}

ALL_STATES = [f"state{index}" for index in range(9)]
LESS_ABUNDANT_STATES = ["state5", "state6", "state7", "state8"]
FOCUS_STATES = ["state4", "state7"]

COHORT_ORDER = [
    "dataset01 | NF",
    "dataset01 | DCM",
    "dataset01 | HCM",
    "dataset08 | NF",
    "dataset08 | DoxCM",
    "dataset11 | NF",
    "dataset11 | HFpEF",
]
COHORT_LABELS = {
    "dataset01 | NF": "D01 NF",
    "dataset01 | DCM": "D01 DCM",
    "dataset01 | HCM": "D01 HCM",
    "dataset08 | NF": "D08 NF",
    "dataset08 | DoxCM": "D08 DoxCM",
    "dataset11 | NF": "D11 NF",
    "dataset11 | HFpEF": "D11 HFpEF",
}
CONDITION_COLORS = {
    "NF": "#7D8992",
    "DCM": "#C65F47",
    "HCM": "#7B65A8",
    "DoxCM": "#C9912C",
    "HFpEF": "#2B8C87",
}
COMPARISON_ORDER = [
    "DCM vs NF (≥25)",
    "HCM vs NF (≥25)",
    "DoxCM vs NF (≥25)",
    "HFpEF vs NF (≥15)",
]
COMPARISON_COLORS = {
    "DCM vs NF (≥25)": CONDITION_COLORS["DCM"],
    "HCM vs NF (≥25)": CONDITION_COLORS["HCM"],
    "DoxCM vs NF (≥25)": CONDITION_COLORS["DoxCM"],
    "HFpEF vs NF (≥15)": CONDITION_COLORS["HFpEF"],
}

IDENTITY_MARKERS = ["CCL21", "FLT4", "NRP2", "PROX1", "ERG", "PECAM1"]
CONTRACTILE_MARKERS = ["TTN", "RYR2", "MYH7", "MYL2", "TNNT2", "ACTC1"]

DARK = "#202529"
SLATE = "#687786"
GRID = "#DDE2E6"
LIGHT_GREY = "#D8DEE2"
NAVY = "#2C5F8A"
TEAL = "#2B8C87"
RED = "#C65F47"
BLUE = "#3E78A5"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.0,
            "axes.edgecolor": "#59636C",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def state_number(state: str) -> int:
    return int(str(state).replace("state", ""))


def state_label(state: str) -> str:
    return f"State {state_number(state)}"


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.tick_params(length=3, color="#65717B")


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.14, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color=DARK,
        clip_on=False,
    )


def require_inputs() -> None:
    paths = [H5AD, *CSV_INPUTS.values()]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n" + "\n".join(missing))


def read_inputs() -> dict[str, pd.DataFrame]:
    require_inputs()
    frames: dict[str, pd.DataFrame] = {}
    for name, path in CSV_INPUTS.items():
        if name == "cell_identity_doublet_qc":
            frames[name] = pd.read_csv(
                path,
                usecols=[
                    "LEC_state",
                    "dataset_folder",
                    "cardiomyocyte_zscore_within_dataset_LECs",
                    "doublet_score_percentile_within_final_LECs",
                    "mixed_cardiac_endothelial_review_flag",
                    "doublet_percentile_ge_99",
                ],
                low_memory=False,
            )
        else:
            frames[name] = pd.read_csv(path, low_memory=False)
    return frames


def build_panel_a(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    main = data["main_effect_source"]
    main_states = set(
        main.loc[
            main["record_type"].isin(
                ["primary_min25_disease_vs_NF_effect", "HFpEF_vs_NF_threshold_effect"]
            ),
            "LEC_state",
        ].dropna()
    )
    additional_states = [state for state in ALL_STATES if state not in main_states]
    if additional_states != ["state2", "state5", "state6", "state7", "state8"]:
        raise ValueError(f"Unexpected main/additional state split: {additional_states}")

    source = data["all_state_effects"].copy()
    primary = source.loc[
        source["record_type"].eq("primary_min25_disease_vs_NF_effect")
        & source["LEC_state"].isin(additional_states)
    ].copy()
    hfpef = source.loc[
        source["record_type"].eq("HFpEF_vs_NF_threshold_effect")
        & source["minimum_LECs_per_donor"].eq(15)
        & source["LEC_state"].isin(additional_states)
    ].copy()
    frame = pd.concat([primary, hfpef], ignore_index=True)
    comparison_map = {
        "dataset01: DCM vs NF": "DCM vs NF (≥25)",
        "dataset01: HCM vs NF": "HCM vs NF (≥25)",
        "dataset08: DoxCM vs NF": "DoxCM vs NF (≥25)",
        "dataset11: HFpEF vs NF": "HFpEF vs NF (≥15)",
    }
    frame["comparison_display"] = frame["comparison"].map(comparison_map)
    frame["state_number"] = frame["LEC_state"].map(state_number)
    frame["state_display"] = frame["LEC_state"].map(state_label)
    frame["global_BH_q"] = np.where(
        frame["record_type"].eq("primary_min25_disease_vs_NF_effect"),
        frame["global_BH_with_HFpEF_min15"],
        frame["global_BH_four_comparison_family"],
    )
    frame["global_BH_q_lt_0_05"] = pd.to_numeric(frame["global_BH_q"], errors="coerce").lt(0.05)
    frame["selection_rule"] = (
        "All disease effects for States 2 and 5–8, which are not emphasized in main Figure 4D; "
        "Dataset01/08 use ≥25 and Dataset11 uses the ≥15 sensitivity cohort"
    )
    return frame.sort_values(["state_number", "comparison_display"]).reset_index(drop=True)


def build_analysis_donor_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    primary = data["retained_donor_states"].copy()
    primary = primary.loc[~primary["dataset_folder"].eq("dataset11")].copy()
    primary = primary.rename(
        columns={
            "n_final_LECs": "donor_n_final_LECs",
            "n_state_LECs": "n_state_LECs",
            "state_percentage_within_donor": "state_percentage_of_donor_LECs",
        }
    )
    primary["analysis_threshold"] = 25

    legacy = data["all_state_effects"]
    dataset11 = legacy.loc[
        legacy["record_type"].eq("HFpEF_NF_donor_state_abundance")
        & legacy["minimum_LECs_per_donor"].eq(15)
    ].copy()
    dataset11["dataset_folder"] = "dataset11"
    dataset11["cohort"] = "dataset11 | " + dataset11["condition"].astype(str)
    dataset11["analysis_threshold"] = 15

    columns = [
        "donor_key",
        "LEC_state",
        "dataset_folder",
        "condition",
        "cohort",
        "donor_n_final_LECs",
        "n_state_LECs",
        "state_percentage_of_donor_LECs",
        "analysis_threshold",
    ]
    frame = pd.concat([primary[columns], dataset11[columns]], ignore_index=True)
    frame["state_number"] = frame["LEC_state"].map(state_number)
    frame["state_display"] = frame["LEC_state"].map(state_label)
    frame["cohort_display"] = frame["cohort"].map(COHORT_LABELS)
    if frame["cohort_display"].isna().any():
        raise ValueError("Unexpected cohort in the combined donor-level disease table.")
    if frame.duplicated(["donor_key", "LEC_state"]).any():
        raise ValueError("Duplicate donor-state rows in the analysis donor table.")
    if not frame.groupby("donor_key")["LEC_state"].nunique().eq(9).all():
        raise ValueError("Every retained donor must have a complete nine-state grid.")
    return frame.sort_values(["cohort", "donor_key", "state_number"]).reset_index(drop=True)


def build_panel_b(donors: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (cohort, state), subset in donors.groupby(["cohort", "LEC_state"], sort=False):
        values = subset.set_index("donor_key")["state_percentage_of_donor_LECs"].astype(float)
        original = float(values.mean())
        leaveout = pd.Series(
            {donor: float(values.drop(index=donor).mean()) for donor in values.index},
            name="leave_one_donor_out_mean",
        )
        signed = leaveout - original
        influential = signed.abs().idxmax()
        records.append(
            {
                "cohort": cohort,
                "cohort_display": COHORT_LABELS[cohort],
                "LEC_state": state,
                "state_number": state_number(state),
                "state_display": state_label(state),
                "analysis_threshold": int(subset["analysis_threshold"].iloc[0]),
                "n_donors": int(values.size),
                "original_equal_donor_mean_percentage": original,
                "most_influential_donor": influential,
                "leave_one_donor_out_mean_percentage": float(leaveout.loc[influential]),
                "signed_change_percentage_points": float(signed.loc[influential]),
                "maximum_absolute_change_percentage_points": float(signed.loc[influential].__abs__()),
                "selection_rule": (
                    "Maximum absolute change in the equal-donor cohort mean after omitting one retained donor; "
                    "this is deterministic leave-one-donor-out sensitivity, not resampling"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def build_panel_c(donors: pd.DataFrame) -> pd.DataFrame:
    frame = donors.loc[donors["LEC_state"].isin(LESS_ABUNDANT_STATES)].copy()
    frame["selection_rule"] = (
        "All retained donor-level abundances for States 5–8; Dataset01/08 use ≥25 and Dataset11 uses ≥15"
    )
    return frame


def select_additional_markers(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    main = data["main_marker_pathway_source"]
    used_main = set(
        main.loc[main["record_type"].eq("marker_expression"), "gene"].dropna().astype(str)
    )
    ranking = data["state_top_markers"].sort_values(["LEC_state", "rank"]).copy()
    selected_frames: list[pd.DataFrame] = []
    selected_genes: set[str] = set()
    for state in ALL_STATES:
        candidates = ranking.loc[
            ranking["LEC_state"].eq(state)
            & ~ranking["names"].astype(str).isin(used_main | selected_genes)
        ].head(2)
        if len(candidates) != 2:
            raise ValueError(f"Could not select two unique additional markers for {state}.")
        selected_frames.append(candidates)
        selected_genes.update(candidates["names"].astype(str))
    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.rename(columns={"names": "gene", "LEC_state": "defining_state"})
    selected["defining_state_number"] = selected["defining_state"].map(state_number)
    selected["defining_state_display"] = selected["defining_state"].map(state_label)
    selected["selection_rule"] = (
        "First two ranked, globally unique state markers not displayed in main Figure 4C"
    )
    selected["row_order"] = np.arange(len(selected))
    return selected


def calculate_selected_marker_expression(selected: pd.DataFrame) -> pd.DataFrame:
    genes = selected.sort_values("row_order")["gene"].tolist()
    adata = ad.read_h5ad(H5AD)
    try:
        if adata.n_obs != 6291:
            raise ValueError(f"Expected 6,291 final disease LECs; found {adata.n_obs}.")
        if "LEC_state" not in adata.obs:
            raise KeyError("LEC_state is absent from the Step08 h5ad.")
        expression_source = adata.raw if adata.raw is not None else adata
        missing = [gene for gene in genes if gene not in expression_source.var_names]
        if missing:
            raise ValueError(f"Selected markers absent from Step08 raw: {missing}")
        matrix = expression_source[:, genes].X
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=float)
        obs_states = adata.obs["LEC_state"].astype(str).to_numpy()
    finally:
        del adata

    records: list[dict[str, object]] = []
    for state in ALL_STATES:
        mask = obs_states == state
        subset = matrix[mask]
        means = subset.mean(axis=0)
        percentages = 100 * (subset > 0).mean(axis=0)
        for index, gene in enumerate(genes):
            records.append(
                {
                    "comparison_state": state,
                    "comparison_state_number": state_number(state),
                    "comparison_state_display": state_label(state),
                    "gene": gene,
                    "n_cells_in_state": int(mask.sum()),
                    "n_cells_expressing": int((subset[:, index] > 0).sum()),
                    "percent_expressing": float(percentages[index]),
                    "mean_log1p_cp10k_all_cells": float(means[index]),
                }
            )
    expression = pd.DataFrame.from_records(records)
    minimum = expression.groupby("gene")["mean_log1p_cp10k_all_cells"].transform("min")
    maximum = expression.groupby("gene")["mean_log1p_cp10k_all_cells"].transform("max")
    span = maximum - minimum
    expression["mean_expression_scaled_0_1_within_gene"] = np.where(
        span > 0,
        (expression["mean_log1p_cp10k_all_cells"] - minimum) / span,
        0.0,
    )
    metadata = selected[
        [
            "gene",
            "defining_state",
            "defining_state_number",
            "defining_state_display",
            "rank",
            "scores",
            "logfoldchanges",
            "pvals_adj",
            "pct_nz_group",
            "pct_nz_reference",
            "n_cohorts_present",
            "present_all_cohorts",
            "selection_rule",
            "row_order",
        ]
    ]
    return expression.merge(metadata, on="gene", how="left", validate="many_to_one")


def select_additional_pathways(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    main = data["main_marker_pathway_source"]
    used = set(
        main.loc[main["record_type"].eq("GSEA_pathway"), "pathway_name"].dropna().astype(str)
    )
    pathways = data["gsea_top_pathways"].copy()
    pathways["absolute_NES"] = pathways["NES"].abs()
    pathways = pathways.sort_values(
        ["LEC_state", "FDR_q_value", "absolute_NES"], ascending=[True, True, False]
    )
    selected: list[pd.DataFrame] = []
    newly_used: set[str] = set()
    focus_pattern = r"CARDIAC|MUSCLE|CONTRACT|MYOSIN|VGCC|RYR"
    for state in ALL_STATES:
        candidates = pathways.loc[
            pathways["LEC_state"].eq(state)
            & ~pathways["pathway_name"].astype(str).isin(used | newly_used)
        ].copy()
        if state in FOCUS_STATES:
            focused = candidates.loc[
                candidates["pathway_display_name"].str.contains(
                    focus_pattern, case=False, regex=True, na=False
                )
                & candidates["NES"].gt(0)
            ]
            if not focused.empty:
                candidates = focused
                rule = (
                    "Lowest-FDR additional positive contractile/cardiac pathway for focused State 4/7"
                )
            else:
                rule = "Lowest-FDR additional pathway not displayed in main Figure 4C"
        else:
            rule = "Lowest-FDR additional pathway not displayed in main Figure 4C"
        choice = candidates.head(1).copy()
        if choice.empty:
            raise ValueError(f"No additional GSEA pathway available for {state}.")
        choice["selection_rule"] = rule
        selected.append(choice)
        newly_used.update(choice["pathway_name"].astype(str))
    frame = pd.concat(selected, ignore_index=True)
    frame["state_number"] = frame["LEC_state"].map(state_number)
    frame["state_display"] = frame["LEC_state"].map(state_label)
    frame["pathway_short"] = frame["pathway_display_name"].map(
        lambda value: "\n".join(textwrap.wrap(str(value).replace("_", " "), width=28))
    )
    frame["significant_FDR_0_25"] = frame["FDR_q_value"].le(0.25)
    return frame


def build_panel_d(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_markers = select_additional_markers(data)
    marker_expression = calculate_selected_marker_expression(selected_markers)
    pathways = select_additional_pathways(data)
    return marker_expression, pathways


def build_panel_e(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = data["marker_expression_qc"].copy()
    genes = IDENTITY_MARKERS + CONTRACTILE_MARKERS
    frame = frame.loc[
        frame["LEC_state"].isin(FOCUS_STATES) & frame["gene"].isin(genes)
    ].copy()
    frame["state_display"] = frame["LEC_state"].map(state_label)
    frame["gene_group"] = np.where(
        frame["gene"].isin(IDENTITY_MARKERS),
        "LEC / endothelial identity",
        "Cardiac / contractile",
    )
    order = {gene: index for index, gene in enumerate(genes)}
    frame["gene_order"] = frame["gene"].map(order)
    frame["selection_rule"] = (
        "Six prespecified LEC/endothelial and six contractile markers in focused States 4 and 7"
    )
    return frame.sort_values(["gene_order", "LEC_state"]).reset_index(drop=True)


def build_panel_f(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = data["state_identity_qc"].copy().merge(
        data["state_dataset_donor_qc"][
            ["LEC_state", "top_dataset", "top_dataset_fraction", "top_condition", "top_condition_fraction"]
        ],
        on="LEC_state",
        how="left",
        validate="one_to_one",
    )
    summary["state_number"] = summary["LEC_state"].map(state_number)
    summary["state_display"] = summary["LEC_state"].map(state_label)
    summary["mixed_review_percent"] = 100 * summary["fraction_mixed_cardiac_endothelial_review_flags"]
    summary["doublet_99_percent"] = 100 * summary["fraction_doublet_percentile_ge_99_among_available"]
    summary["same_state_neighbor_percent"] = 100 * summary["mean_neighbor_same_LEC_state_fraction"]
    summary["top_donor_percent"] = 100 * summary["top_donor_fraction"]
    summary["top_dataset_percent"] = 100 * summary["top_dataset_fraction"]
    summary["focused_contractile_state"] = summary["LEC_state"].isin(FOCUS_STATES)
    summary["selection_rule"] = (
        "All-state final-LEC identity, doublet-percentile, graph-neighborhood, donor, and dataset QC"
    )

    cells = data["cell_identity_doublet_qc"].copy()
    cells = cells.loc[cells["LEC_state"].isin(FOCUS_STATES)].copy()
    cells["state_display"] = cells["LEC_state"].map(state_label)
    cells["mixed_cardiac_endothelial_review_flag"] = (
        cells["mixed_cardiac_endothelial_review_flag"].astype(str).str.lower().eq("true")
    )
    cells["doublet_percentile_ge_99"] = (
        cells["doublet_percentile_ge_99"].astype(str).str.lower().eq("true")
    )
    cells["doublet_score_percent"] = 100 * pd.to_numeric(
        cells["doublet_score_percentile_within_final_LECs"], errors="coerce"
    )
    cells["selection_rule"] = (
        "All State 4/7 final LECs with an available upstream numeric doublet score; "
        "Dataset08 has no numeric score"
    )
    return summary.sort_values("state_number"), cells


def validate_tables(
    panel_a: pd.DataFrame,
    donors: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    marker_expression: pd.DataFrame,
    pathways: pd.DataFrame,
    panel_e: pd.DataFrame,
    panel_f_summary: pd.DataFrame,
    panel_f_cells: pd.DataFrame,
) -> None:
    if len(panel_a) != 20 or panel_a["comparison_display"].nunique() != 4:
        raise ValueError("Panel A requires five additional states across four disease comparisons.")
    if set(donors["cohort"]) != set(COHORT_ORDER):
        raise ValueError("The donor table must cover all seven studied disease/control cohorts.")
    if len(panel_b) != 63:
        raise ValueError("Panel B requires seven cohorts by nine states.")
    if set(panel_c["LEC_state"]) != set(LESS_ABUNDANT_STATES):
        raise ValueError("Panel C must contain States 5–8.")
    if marker_expression["gene"].nunique() != 18 or len(marker_expression) != 162:
        raise ValueError("Panel D marker map requires 18 unique markers across nine states.")
    if pathways["LEC_state"].nunique() != 9 or len(pathways) != 9:
        raise ValueError("Panel D requires one additional pathway per state.")
    if len(panel_e) != 24:
        raise ValueError("Panel E requires 12 markers in each of States 4 and 7.")
    if len(panel_f_summary) != 9 or set(panel_f_summary["LEC_state"]) != set(ALL_STATES):
        raise ValueError("Panel F summary must cover all nine states.")
    if panel_f_cells.dropna(subset=["doublet_score_percent"])["LEC_state"].nunique() != 2:
        raise ValueError("Panel F doublet-score comparison requires both State 4 and State 7.")


def draw_panel_a(fig: plt.Figure, spec, frame: pd.DataFrame) -> list[plt.Axes]:
    ax = fig.add_subplot(spec)
    states = ["state2", "state5", "state6", "state7", "state8"]
    positions = {state: value for state, value in zip(states, np.arange(len(states))[::-1])}
    offsets = dict(zip(COMPARISON_ORDER, [0.24, 0.08, -0.08, -0.24]))
    for comparison in COMPARISON_ORDER:
        subset = frame.loc[frame["comparison_display"].eq(comparison)]
        for _, row in subset.iterrows():
            y = positions[row["LEC_state"]] + offsets[comparison]
            color = COMPARISON_COLORS[comparison]
            filled = bool(row["global_BH_q_lt_0_05"])
            ax.errorbar(
                row["mean_difference_percentage_points"],
                y,
                xerr=[
                    [row["mean_difference_percentage_points"] - row["mean_difference_ci_low"]],
                    [row["mean_difference_ci_high"] - row["mean_difference_percentage_points"]],
                ],
                fmt="o",
                markersize=5.8,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.2,
                ecolor=color,
                elinewidth=1.05,
                capsize=2,
                zorder=3,
            )
    labels = []
    definition_map = frame.drop_duplicates("LEC_state").set_index("LEC_state")["state_definition"].to_dict()
    for state in states:
        definition = str(definition_map.get(state, "")).replace("*", "")
        labels.append(f"{state_label(state)}\n{definition}")
    ax.set_yticks([positions[state] for state in states], labels)
    ax.axvline(0, color="#8B969E", linewidth=0.9, linestyle=(0, (3, 2)))
    ax.set_xlabel("Disease − matched NF abundance difference (percentage points)")
    ax.set_title(
        "Additional donor-level disease effects",
        loc="left",
        pad=9,
    )
    clean_axis(ax, "x")
    comparison_handles = [
        Line2D([0], [0], color=COMPARISON_COLORS[name], marker="o", linewidth=1.2, label=name)
        for name in COMPARISON_ORDER
    ]
    fill_handles = [
        Line2D([0], [0], color=DARK, marker="o", markerfacecolor=DARK, linewidth=0, label="Global BH q < 0.05"),
        Line2D([0], [0], color=DARK, marker="o", markerfacecolor="white", linewidth=0, label="Global BH q ≥ 0.05"),
    ]
    legend1 = ax.legend(handles=comparison_handles, frameon=False, loc="lower left", fontsize=7.5)
    ax.add_artist(legend1)
    ax.legend(handles=fill_handles, frameon=False, loc="lower right", fontsize=7.5)
    ax.text(
        0,
        -0.16,
        "States shown here were not selected for the main Figure 4 effect panel. Bars are 95% donor-bootstrap intervals.",
        transform=ax.transAxes,
        fontsize=7.7,
        color=SLATE,
    )
    panel_letter(ax, "A", x=-0.18, y=1.03)
    return [ax]


def draw_panel_b(fig: plt.Figure, spec, frame: pd.DataFrame) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 2, width_ratios=[1, 0.035], wspace=0.045)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    matrix = (
        frame.pivot(index="cohort", columns="state_number", values="maximum_absolute_change_percentage_points")
        .reindex(index=COHORT_ORDER, columns=range(9))
    )
    vmax = max(1.0, float(matrix.to_numpy().max()) * 1.02)
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(9), [f"State {i}" for i in range(9)], rotation=45, ha="right")
    row_labels = [
        f"{COHORT_LABELS[cohort]}  ({'≥15' if cohort.startswith('dataset11') else '≥25'})"
        for cohort in COHORT_ORDER
    ]
    ax.set_yticks(range(len(COHORT_ORDER)), row_labels)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix.iloc[y, x]
            ax.text(
                x,
                y,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.1,
                color="white" if value > vmax * 0.58 else DARK,
            )
    for x in [4, 7]:
        ax.add_patch(plt.Rectangle((x - 0.48, -0.48), 0.96, 6.96, fill=False, edgecolor=DARK, linewidth=1.0))
    ax.tick_params(length=0)
    ax.set_title(
        "Leave-one-donor-out influence across every studied cohort",
        loc="left",
        pad=9,
    )
    ax.set_xlabel("LEC state")
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Maximum change in equal-donor mean (pp)", fontsize=8.0)
    colorbar.ax.tick_params(labelsize=7.3)
    ax.text(
        0,
        -0.23,
        "Each value is the largest absolute change after omitting one donor. This is deterministic influence analysis, not resampling.",
        transform=ax.transAxes,
        fontsize=7.7,
        color=SLATE,
    )
    panel_letter(ax, "B", x=-0.22, y=1.03)
    return [ax, cax]


def draw_panel_c(fig: plt.Figure, spec, frame: pd.DataFrame) -> list[plt.Axes]:
    grid = spec.subgridspec(2, 2, hspace=0.32, wspace=0.18)
    rng = np.random.default_rng(2701)
    axes: list[plt.Axes] = []
    for index, state in enumerate(LESS_ABUNDANT_STATES):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        subset = frame.loc[frame["LEC_state"].eq(state)]
        for x, cohort in enumerate(COHORT_ORDER):
            values = subset.loc[
                subset["cohort"].eq(cohort), "state_percentage_of_donor_LECs"
            ].to_numpy(float)
            condition = cohort.split(" | ")[1]
            jitter = rng.uniform(-0.13, 0.13, size=len(values))
            ax.scatter(
                x + jitter,
                values,
                s=22,
                color=CONDITION_COLORS[condition],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.35,
                zorder=2,
            )
            ax.scatter(
                x,
                values.mean(),
                s=55,
                marker="D",
                color=CONDITION_COLORS[condition],
                edgecolor=DARK,
                linewidth=0.55,
                zorder=3,
            )
        ax.set_xticks(range(len(COHORT_ORDER)), [COHORT_LABELS[value] for value in COHORT_ORDER], rotation=35, ha="right")
        ax.set_ylabel("Donor LECs in state (%)")
        ax.set_title(f"{state_label(state)}", loc="left", pad=7)
        ax.set_ylim(bottom=-0.6)
        clean_axis(ax, "y")
        if index == 0:
            panel_letter(ax, "C", x=-0.18, y=1.14)
            ax.text(
                0,
                1.12,
                "Less-abundant state distributions",
                transform=ax.transAxes,
                fontsize=11.5,
                fontweight="bold",
                color=DARK,
            )
        axes.append(ax)
    axes[-1].text(
        1.0,
        -0.48,
        "Small points: donors; diamonds: equal-donor means. D01/D08 ≥25; D11 ≥15.",
        transform=axes[-1].transAxes,
        ha="right",
        fontsize=7.5,
        color=SLATE,
    )
    return axes


def draw_panel_d(
    fig: plt.Figure,
    spec,
    markers: pd.DataFrame,
    pathways: pd.DataFrame,
) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.42)
    ax_marker = fig.add_subplot(grid[0, 0])
    ax_pathway = fig.add_subplot(grid[0, 1])

    marker_rows = (
        markers[["row_order", "gene", "defining_state"]]
        .drop_duplicates()
        .sort_values("row_order")
    )
    row_lookup = marker_rows.set_index("gene")["row_order"].to_dict()
    x = markers["comparison_state_number"].to_numpy()
    y = markers["gene"].map(row_lookup).to_numpy()
    sizes = 10 + 0.78 * markers["percent_expressing"].to_numpy()
    scatter = ax_marker.scatter(
        x,
        y,
        s=sizes,
        c=markers["mean_expression_scaled_0_1_within_gene"],
        cmap="Reds",
        norm=Normalize(0, 1),
        edgecolor="#D4D9DD",
        linewidth=0.35,
    )
    for _, row in marker_rows.iterrows():
        ax_marker.add_patch(
            plt.Rectangle(
                (state_number(row["defining_state"]) - 0.45, row["row_order"] - 0.45),
                0.9,
                0.9,
                fill=False,
                edgecolor=DARK,
                linewidth=0.9,
            )
        )
    defining_numbers = marker_rows["defining_state"].map(state_number).to_numpy()
    for yline in np.where(np.diff(defining_numbers) != 0)[0] + 0.5:
        ax_marker.axhline(yline, color="#BCC5CA", linewidth=0.8)
    labels = [
        f"{state_label(state)} | {gene}"
        for state, gene in zip(marker_rows["defining_state"], marker_rows["gene"])
    ]
    ax_marker.set_yticks(marker_rows["row_order"], labels)
    ax_marker.set_xticks(range(9), [f"State {index}" for index in range(9)], rotation=45, ha="right")
    ax_marker.invert_yaxis()
    ax_marker.set_xlabel("LEC state")
    ax_marker.set_title("Additional state-marker expression", loc="left", pad=8)
    ax_marker.tick_params(length=0)
    colorbar = fig.colorbar(scatter, ax=ax_marker, fraction=0.035, pad=0.025)
    colorbar.set_label("Per-gene scaled mean log1p expression", fontsize=7.7)
    colorbar.ax.tick_params(labelsize=7.0)
    for value in [25, 50, 75, 100]:
        ax_marker.scatter([], [], s=10 + 0.78 * value, facecolor="#B44C48", edgecolor="#D4D9DD", label=f"{value}%")
    ax_marker.legend(
        title="Cells expressing",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0, -0.27),
        ncol=4,
        fontsize=7.0,
        title_fontsize=7.2,
    )

    ordered = pathways.sort_values("state_number", ascending=False).reset_index(drop=True)
    y_path = np.arange(len(ordered))
    colors = np.where(ordered["NES"].ge(0), RED, BLUE)
    for yi, value, color in zip(y_path, ordered["NES"], colors):
        ax_pathway.hlines(yi, 0, value, color=color, linewidth=1.8, alpha=0.78)
    for yi, (_, row), color in zip(y_path, ordered.iterrows(), colors):
        ax_pathway.scatter(
            row["NES"],
            yi,
            s=52,
            facecolor=color if row["significant_FDR_0_25"] else "white",
            edgecolor=color,
            linewidth=1.0,
            zorder=3,
        )
    pathway_labels = [
        f"{row['state_display']} | {row['pathway_short']}"
        for _, row in ordered.iterrows()
    ]
    ax_pathway.set_yticks(y_path, pathway_labels)
    ax_pathway.axvline(0, color="#9CA6AD", linewidth=0.8)
    limit = max(2.55, float(ordered["NES"].abs().max()) * 1.12)
    ax_pathway.set_xlim(-limit, limit)
    ax_pathway.set_xlabel("Normalized enrichment score (NES)")
    ax_pathway.set_title("Additional state GSEA evidence", loc="left", pad=8)
    clean_axis(ax_pathway, "x")
    ax_pathway.scatter([], [], s=48, color=RED, label="NES > 0")
    ax_pathway.scatter([], [], s=48, color=BLUE, label="NES < 0")
    ax_pathway.scatter([], [], s=48, facecolor="white", edgecolor=DARK, label="FDR > 0.25")
    ax_pathway.legend(frameon=False, loc="lower right", fontsize=7.2)

    panel_letter(ax_marker, "D", x=-0.26, y=1.15)
    ax_marker.text(
        0,
        1.12,
        "Additional marker and pathway support",
        transform=ax_marker.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=DARK,
    )
    return [ax_marker, ax_pathway]


def draw_panel_e(fig: plt.Figure, spec, frame: pd.DataFrame) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 2, width_ratios=[1, 0.035], wspace=0.05)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    genes = IDENTITY_MARKERS + CONTRACTILE_MARKERS
    gene_order = {gene: index for index, gene in enumerate(genes)}
    state_order = {"state4": 0, "state7": 1}
    x = frame["LEC_state"].map(state_order).to_numpy()
    y = frame["gene"].map(gene_order).to_numpy()
    sizes = 18 + 1.0 * frame["pct_detected"].to_numpy()
    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        c=frame["mean_logexpr"],
        cmap="viridis",
        edgecolor="white",
        linewidth=0.6,
    )
    state_counts = frame.groupby("LEC_state")["n_LEC_cells"].first().to_dict()
    ax.set_xticks(
        [0, 1],
        [f"State 4\n{int(state_counts['state4'])} cells", f"State 7\n{int(state_counts['state7'])} cells"],
    )
    ax.set_yticks(range(len(genes)), genes)
    ax.invert_yaxis()
    ax.set_xlim(-0.55, 1.55)
    ax.axhline(len(IDENTITY_MARKERS) - 0.5, color="#9CA6AD", linewidth=1.1)
    ax.text(
        1.03,
        0.77,
        "LEC / endothelial\nidentity",
        transform=ax.transAxes,
        va="center",
        fontsize=8.2,
        color=NAVY,
        fontweight="bold",
    )
    ax.text(
        1.03,
        0.25,
        "Cardiac / contractile",
        transform=ax.transAxes,
        va="center",
        fontsize=8.2,
        color=RED,
        fontweight="bold",
    )
    ax.set_title("State 4/7 identity and contractile-marker detection", loc="left", pad=9)
    ax.tick_params(length=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    colorbar = fig.colorbar(scatter, cax=cax)
    colorbar.set_label("Mean log1p expression", fontsize=8.0)
    colorbar.ax.tick_params(labelsize=7.2)
    for value in [25, 50, 75, 100]:
        ax.scatter([], [], s=18 + value, color="#4D968F", edgecolor="white", label=f"{value}%")
    ax.legend(
        title="Cells detected",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=4,
        fontsize=7.2,
        title_fontsize=7.4,
    )
    ax.text(
        0,
        -0.16,
        "These are state-level detection summaries and do not establish same-cell co-expression.",
        transform=ax.transAxes,
        fontsize=7.7,
        color=SLATE,
    )
    panel_letter(ax, "E", x=-0.20, y=1.03)
    return [ax, cax]


def draw_panel_f(
    fig: plt.Figure,
    spec,
    summary: pd.DataFrame,
    cells: pd.DataFrame,
    colors: dict[str, str],
) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 3, width_ratios=[0.78, 0.92, 1.58], wspace=0.31)
    ax_identity = fig.add_subplot(grid[0, 0])
    ax_doublet = fig.add_subplot(grid[0, 1])
    ax_qc = fig.add_subplot(grid[0, 2])

    for _, row in summary.iterrows():
        focus = row["LEC_state"] in FOCUS_STATES
        ax_identity.scatter(
            row["mean_LEC_endothelial_identity_zscore"],
            row["mean_cardiomyocyte_zscore"],
            s=80 if focus else 48,
            color=colors[row["LEC_state"]],
            edgecolor=DARK if focus else "white",
            linewidth=1.25 if focus else 0.55,
            zorder=3,
        )
        ax_identity.annotate(
            row["state_display"],
            (row["mean_LEC_endothelial_identity_zscore"], row["mean_cardiomyocyte_zscore"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.3,
            fontweight="bold" if focus else "normal",
        )
    ax_identity.axhline(0, color=GRID, linewidth=0.8)
    ax_identity.axvline(0, color=GRID, linewidth=0.8)
    ax_identity.set_xlabel("Mean LEC/endothelial\nidentity z score")
    ax_identity.set_ylabel("Mean cardiomyocyte z score")
    ax_identity.set_title("State-level identity balance", loc="left", pad=8)
    clean_axis(ax_identity)
    panel_letter(ax_identity, "F", x=-0.30, y=1.18)
    ax_identity.text(
        0,
        1.15,
        "Focused contractile-state QC",
        transform=ax_identity.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=DARK,
    )

    available = cells.dropna(
        subset=["cardiomyocyte_zscore_within_dataset_LECs", "doublet_score_percent"]
    ).copy()
    for state in FOCUS_STATES:
        subset = available.loc[available["LEC_state"].eq(state)]
        ax_doublet.scatter(
            subset["cardiomyocyte_zscore_within_dataset_LECs"],
            subset["doublet_score_percent"],
            s=15,
            color=colors[state],
            alpha=0.38,
            edgecolor="none",
            rasterized=True,
            label=state_label(state),
        )
        rho = subset["cardiomyocyte_zscore_within_dataset_LECs"].corr(
            subset["doublet_score_percent"], method="spearman"
        )
        ax_doublet.text(
            0.04,
            0.95 - 0.07 * FOCUS_STATES.index(state),
            f"{state_label(state)}: ρ = {rho:.2f} (n={len(subset)})",
            transform=ax_doublet.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            color=colors[state],
        )
    flagged = available.loc[available["doublet_percentile_ge_99"]]
    ax_doublet.scatter(
        flagged["cardiomyocyte_zscore_within_dataset_LECs"],
        flagged["doublet_score_percent"],
        s=26,
        facecolor="none",
        edgecolor=DARK,
        linewidth=0.6,
        zorder=3,
    )
    ax_doublet.axhline(99, color="#8C979E", linewidth=0.9, linestyle=(0, (3, 2)))
    ax_doublet.set_ylim(-2, 102)
    ax_doublet.set_xlabel("Cardiomyocyte signature z score")
    ax_doublet.set_ylabel("Within-sample doublet-score percentile")
    ax_doublet.set_title("Cardiac signal vs doublet score", loc="left", pad=8)
    clean_axis(ax_doublet, "y")
    ax_doublet.legend(frameon=False, loc="lower right", fontsize=7.2)

    metric_columns = [
        "mixed_review_percent",
        "doublet_99_percent",
        "same_state_neighbor_percent",
        "top_donor_percent",
        "top_dataset_percent",
    ]
    metric_labels = [
        "Mixed cardiac +\nendothelial flag",
        "≥99th-percentile\ndoublet score",
        "Same-state\nneighbors",
        "Top-donor\nshare",
        "Top-dataset\nshare",
    ]
    matrix = summary.set_index("LEC_state").reindex(ALL_STATES)[metric_columns].to_numpy(float)
    image = ax_qc.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ylabels = [
        f"State {index}  ({int(row.n_LEC_cells):,} cells; {int(row.n_LEC_positive_donors)} donors)"
        for index, row in enumerate(summary.set_index("LEC_state").reindex(ALL_STATES).itertuples())
    ]
    ax_qc.set_yticks(range(9), ylabels)
    ax_qc.set_xticks(range(len(metric_columns)), metric_labels)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            ax_qc.text(
                x,
                y,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if value >= 58 else DARK,
            )
    for state in FOCUS_STATES:
        y = state_number(state)
        ax_qc.add_patch(
            plt.Rectangle(
                (-0.48, y - 0.48),
                len(metric_columns) - 0.04,
                0.96,
                fill=False,
                edgecolor=DARK,
                linewidth=1.2,
            )
        )
    ax_qc.tick_params(length=0)
    ax_qc.set_title("All-state technical context (%)", loc="left", pad=8)
    colorbar = fig.colorbar(image, ax=ax_qc, fraction=0.036, pad=0.025)
    colorbar.set_label("Percent", fontsize=7.8)
    colorbar.ax.tick_params(labelsize=7.0)
    ax_qc.text(
        1.0,
        -0.20,
        "Descriptive final-LEC QC only: relative doublet percentiles are not probabilities; Dataset08 lacks a numeric score.\n"
        "These analyses evaluate technical alternatives but do not rule out ambient RNA.",
        transform=ax_qc.transAxes,
        ha="right",
        fontsize=7.4,
        color=SLATE,
    )
    return [ax_identity, ax_doublet, ax_qc]


def save_figure(fig: plt.Figure, basename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "jpg"):
        kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.18,
            "facecolor": "white",
        }
        if extension in {"png", "jpg"}:
            kwargs["dpi"] = 300
        if extension == "jpg":
            kwargs["pil_kwargs"] = {"quality": 95, "subsampling": 0}
        fig.savefig(OUT_DIR / f"{basename}.{extension}", **kwargs)
    plt.close(fig)


def make_individual_panels(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    marker_expression: pd.DataFrame,
    pathways: pd.DataFrame,
    panel_e: pd.DataFrame,
    panel_f_summary: pd.DataFrame,
    panel_f_cells: pd.DataFrame,
    colors: dict[str, str],
) -> None:
    fig = plt.figure(figsize=(12.5, 6.3), constrained_layout=True)
    draw_panel_a(fig, GridSpec(1, 1, figure=fig)[0], panel_a)
    save_figure(fig, "FigS05A_additional_donor_level_disease_effects")

    fig = plt.figure(figsize=(12.8, 5.7), constrained_layout=True)
    draw_panel_b(fig, GridSpec(1, 1, figure=fig)[0], panel_b)
    save_figure(fig, "FigS05B_leave_one_donor_out_influence")

    fig = plt.figure(figsize=(14.2, 8.2), constrained_layout=True)
    draw_panel_c(fig, GridSpec(1, 1, figure=fig)[0], panel_c)
    save_figure(fig, "FigS05C_less_abundant_state_distributions")

    fig = plt.figure(figsize=(16.5, 8.4), constrained_layout=True)
    draw_panel_d(fig, GridSpec(1, 1, figure=fig)[0], marker_expression, pathways)
    save_figure(fig, "FigS05D_additional_marker_pathway_results")

    fig = plt.figure(figsize=(7.7, 6.8), constrained_layout=True)
    draw_panel_e(fig, GridSpec(1, 1, figure=fig)[0], panel_e)
    save_figure(fig, "FigS05E_state4_state7_marker_QC")

    fig = plt.figure(figsize=(17.0, 6.5), constrained_layout=True)
    draw_panel_f(
        fig,
        GridSpec(1, 1, figure=fig)[0],
        panel_f_summary,
        panel_f_cells,
        colors,
    )
    save_figure(fig, "FigS05F_state4_state7_technical_QC")


def make_combined_figure(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    marker_expression: pd.DataFrame,
    pathways: pd.DataFrame,
    panel_e: pd.DataFrame,
    panel_f_summary: pd.DataFrame,
    panel_f_cells: pd.DataFrame,
    colors: dict[str, str],
) -> None:
    fig = plt.figure(figsize=(18.0, 42.0), constrained_layout=False)
    grid = fig.add_gridspec(
        6,
        1,
        height_ratios=[6.2, 5.6, 8.0, 8.2, 6.5, 6.3],
        left=0.12,
        right=0.965,
        top=0.972,
        bottom=0.025,
        hspace=0.48,
    )
    draw_panel_a(fig, grid[0, 0], panel_a)
    draw_panel_b(fig, grid[1, 0], panel_b)
    draw_panel_c(fig, grid[2, 0], panel_c)
    draw_panel_d(fig, grid[3, 0], marker_expression, pathways)
    draw_panel_e(fig, grid[4, 0], panel_e)
    draw_panel_f(fig, grid[5, 0], panel_f_summary, panel_f_cells, colors)
    fig.suptitle(
        "Figure S5 | Extended disease LEC-state characterization and robustness",
        fontsize=18,
        fontweight="bold",
        x=0.02,
        y=0.993,
        ha="left",
    )
    save_figure(fig, "FigS05_extended_disease_LEC_state_characterization")


def write_traceability(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    marker_expression: pd.DataFrame,
    pathways: pd.DataFrame,
    panel_e: pd.DataFrame,
    panel_f_summary: pd.DataFrame,
    panel_f_cells: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for panel, record_type, frame in [
        ("A", "additional_donor_level_disease_effect", panel_a),
        ("B", "leave_one_donor_out_influence", panel_b),
        ("C", "less_abundant_state_donor_abundance", panel_c),
        ("D", "additional_marker_expression", marker_expression),
        ("D", "additional_GSEA_pathway", pathways),
        ("E", "state4_state7_marker_identity_QC", panel_e),
        ("F", "all_state_identity_technical_QC", panel_f_summary),
        ("F", "state4_state7_cell_doublet_QC", panel_f_cells),
    ]:
        output = frame.copy()
        if "panel" in output.columns:
            output = output.rename(columns={"panel": "source_panel"})
        output.insert(0, "panel", panel)
        output.insert(1, "record_type_S5", record_type)
        frames.append(output)
    pd.concat(frames, ignore_index=True, sort=False).to_csv(
        OUT_DIR / "FigS05_source_data.csv", index=False
    )

    panel_map = {
        "all_state_effects": "A; B; C",
        "main_effect_source": "A exclusion check",
        "retained_donor_states": "B; C",
        "main_marker_pathway_source": "D exclusion check and state definitions",
        "state_top_markers": "D",
        "gsea_top_pathways": "D",
        "marker_expression_qc": "E",
        "state_identity_qc": "F",
        "state_dataset_donor_qc": "F",
        "cell_identity_doublet_qc": "F",
        "state_colors": "F",
    }
    purpose_map = {
        "all_state_effects": "Existing all-state primary and Dataset11 threshold-effect results plus Dataset11 donor-state abundances",
        "main_effect_source": "Identify states already emphasized in main Figure 4D",
        "retained_donor_states": "Post-≥25 donor-state abundances for Dataset01 and Dataset08",
        "main_marker_pathway_source": "Exclude marker genes and pathways already displayed in main Figure 4C",
        "state_top_markers": "Objective selection of two additional ranked markers per state",
        "gsea_top_pathways": "Objective selection of one additional pathway per state with focused contractile logic for States 4/7",
        "marker_expression_qc": "Existing final-LEC marker-detection summaries for focused States 4/7",
        "state_identity_qc": "Existing final-LEC identity, relative doublet, neighborhood, and donor QC",
        "state_dataset_donor_qc": "Dataset and condition dominance context for every state",
        "cell_identity_doublet_qc": "Cell-level State 4/7 cardiac-signature and relative doublet-score comparison",
        "state_colors": "Established disease-state color mapping",
    }
    records = []
    for input_id, path in CSV_INPUTS.items():
        records.append(
            {
                "input_id": input_id,
                "panels": panel_map[input_id],
                "input_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "input_type": "CSV.GZ" if path.suffix == ".gz" else "CSV",
                "purpose": purpose_map[input_id],
            }
        )
    records.append(
        {
            "input_id": "state_expression_h5ad",
            "panels": "D",
            "input_file": str(H5AD.relative_to(ROOT)).replace("\\", "/"),
            "input_type": "H5AD",
            "purpose": "Mean log1p expression and detection percentage for additional marker genes across all disease LEC states",
        }
    )
    pd.DataFrame(records).to_csv(OUT_DIR / "FigS05_input_manifest.csv", index=False)


def main() -> None:
    configure_style()
    data = read_inputs()
    colors = data["state_colors"].set_index("LEC_state")["color_hex"].to_dict()
    panel_a = build_panel_a(data)
    donors = build_analysis_donor_table(data)
    panel_b = build_panel_b(donors)
    panel_c = build_panel_c(donors)
    marker_expression, pathways = build_panel_d(data)
    panel_e = build_panel_e(data)
    panel_f_summary, panel_f_cells = build_panel_f(data)
    validate_tables(
        panel_a,
        donors,
        panel_b,
        panel_c,
        marker_expression,
        pathways,
        panel_e,
        panel_f_summary,
        panel_f_cells,
    )
    write_traceability(
        panel_a,
        panel_b,
        panel_c,
        marker_expression,
        pathways,
        panel_e,
        panel_f_summary,
        panel_f_cells,
    )
    make_individual_panels(
        panel_a,
        panel_b,
        panel_c,
        marker_expression,
        pathways,
        panel_e,
        panel_f_summary,
        panel_f_cells,
        colors,
    )
    make_combined_figure(
        panel_a,
        panel_b,
        panel_c,
        marker_expression,
        pathways,
        panel_e,
        panel_f_summary,
        panel_f_cells,
        colors,
    )
    print(f"Supplementary Figure S5 written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
