"""Create disease Figure 4D with primary and exploratory donor-level effects.

Dataset01 DCM/HCM and dataset08 DoxCM remain primary comparisons requiring
>=25 final LECs per donor. Dataset11 HFpEF versus matched NF is an explicitly
exploratory comparison at >=10 LECs per donor, with >=15 shown as a threshold
sensitivity analysis. The relaxed dataset11 threshold is applied to both
HFpEF and NF donors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_CSV = (
    ROOT
    / "new_result2_disease/08_4_donor_level_LEC_state_disease_analysis/tables/"
    "08_4_primary_within_dataset_state_comparisons.csv"
)
CELL_CSV = (
    ROOT
    / "new_result2_disease/08_3_min25_LEC_filtered_analysis/tables/"
    "08_3_final_LEC_cell_filter_mapping.csv.gz"
)
MIN10_SUMMARY = (
    ROOT
    / "new_result2_disease/12_NicheNet_donor_pseudobulk/"
    "dataset11_HFpEF_vs_NF_min10/12_run_summary.csv"
)
MIN15_SUMMARY = (
    ROOT
    / "new_result2_disease/12_NicheNet_donor_pseudobulk/"
    "dataset11_HFpEF_vs_NF_min15/12_run_summary.csv"
)

OUTPUT_DIR = ROOT / "new_result2/disease/Fig04D_disease_representative_state_effects"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_PNG = OUTPUT_DIR / "Fig04D_disease_representative_state_effects.png"
FIGURE_PDF = OUTPUT_DIR / "Fig04D_disease_representative_state_effects.pdf"
FIGURE_JPG = OUTPUT_DIR / "Fig04D_disease_representative_state_effects.jpg"
SOURCE_CSV = OUTPUT_DIR / "Fig04D_source_data.csv"
MANIFEST_CSV = OUTPUT_DIR / "Fig04D_input_manifest.csv"

SCRIPT_VERSION = "Fig04D_HFpEF_min10_min15_sensitivity_v3_2026_08_31"
BOOTSTRAP_ITERATIONS = 10_000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 20260826

SELECTED_STATES = ["state0", "state1", "state3", "state4"]
ALL_STATES = [f"state{value}" for value in range(9)]
STATE_DEFINITIONS = {
    "state0": "Disease-associated remodeling LEC",
    "state1": "Homeostatic maintenance LEC",
    "state3": "Disease-associated endothelial regulatory LEC",
    "state4": "Cardiomyopathy-associated transitional state",
}
PRIMARY_COMPARISONS = [
    "dataset01: DCM vs NF",
    "dataset01: HCM vs NF",
    "dataset08: DoxCM vs NF",
]
COMPARISON_LABELS = {
    "dataset01: DCM vs NF": "DCM - NF",
    "dataset01: HCM vs NF": "HCM - NF",
    "dataset08: DoxCM vs NF": "DoxCM - NF",
}
COMPARISON_COLORS = {
    "dataset01: DCM vs NF": "#C75B43",
    "dataset01: HCM vs NF": "#7A64A8",
    "dataset08: DoxCM vs NF": "#C7902E",
}
HFPEF_STYLES = {
    10: {"color": "#287C78", "label": "≥10 LECs/donor · exploratory"},
    15: {"color": "#687279", "label": "≥15 LECs/donor · sensitivity"},
}
EXPECTED_DONORS = {10: {"NF": 6, "HFpEF": 7}, 15: {"NF": 4, "HFpEF": 5}}
EXPECTED_RECEIVER_CELLS = {10: 349, 15: 301}

TEXT = "#202428"
SUBTEXT = "#5C646A"
GRID = "#DDE2E5"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def bh_adjust(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    adjusted = np.full(len(numeric), np.nan, dtype=float)
    finite = np.isfinite(numeric.to_numpy(float))
    if finite.any():
        adjusted[finite] = multipletests(numeric.to_numpy(float)[finite], method="fdr_bh")[1]
    return pd.Series(adjusted, index=values.index)


def bootstrap_mean_difference(
    disease: np.ndarray, reference: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    disease_indices = rng.integers(0, len(disease), size=(BOOTSTRAP_ITERATIONS, len(disease)))
    reference_indices = rng.integers(
        0, len(reference), size=(BOOTSTRAP_ITERATIONS, len(reference))
    )
    differences = disease[disease_indices].mean(axis=1) - reference[reference_indices].mean(axis=1)
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    low, high = np.quantile(differences, [alpha, 1 - alpha])
    return float(low), float(high)


def validate_fig07_threshold_summary(path: Path, threshold: int) -> None:
    summary = pd.read_csv(path)
    require_columns(
        summary,
        {
            "receiver_min_cells",
            "n_reference_donors",
            "n_disease_donors",
            "n_receiver_cells",
            "validation_status",
            "status",
        },
        path,
    )
    if len(summary) != 1:
        raise ValueError(f"Expected one completed threshold-summary row in {path}.")
    row = summary.iloc[0]
    expected = EXPECTED_DONORS[threshold]
    checks = {
        "receiver_min_cells": int(row["receiver_min_cells"]) == threshold,
        "n_reference_donors": int(row["n_reference_donors"]) == expected["NF"],
        "n_disease_donors": int(row["n_disease_donors"]) == expected["HFpEF"],
        "n_receiver_cells": int(row["n_receiver_cells"]) == EXPECTED_RECEIVER_CELLS[threshold],
        "validation_status": str(row["validation_status"]).lower() == "passed",
        "status": str(row["status"]).lower() == "completed",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Dataset11 threshold provenance failed for ≥{threshold}: {failed}")


def donor_state_table(cells: pd.DataFrame, threshold: int) -> pd.DataFrame:
    subset = cells.loc[
        cells["dataset_folder"].eq("dataset11")
        & cells["condition"].isin(["NF", "HFpEF"])
        & (cells["donor_n_final_LECs"] >= threshold)
    ].copy()
    donor_meta = subset[
        ["donor_key", "condition", "donor_n_final_LECs"]
    ].drop_duplicates("donor_key")
    observed = donor_meta.groupby("condition")["donor_key"].nunique().to_dict()
    if observed != EXPECTED_DONORS[threshold]:
        raise ValueError(f"Unexpected dataset11 donor counts at ≥{threshold}: {observed}")
    if len(subset) != EXPECTED_RECEIVER_CELLS[threshold]:
        raise ValueError(
            f"Unexpected dataset11 LEC count at ≥{threshold}: "
            f"{len(subset)} != {EXPECTED_RECEIVER_CELLS[threshold]}"
        )

    counts = (
        subset.groupby(["donor_key", "condition", "LEC_state"], observed=True)
        .size()
        .rename("n_state_LECs")
        .reset_index()
    )
    complete = pd.MultiIndex.from_product(
        [donor_meta["donor_key"], ALL_STATES], names=["donor_key", "LEC_state"]
    ).to_frame(index=False)
    complete = complete.merge(donor_meta, on="donor_key", how="left", validate="many_to_one")
    complete = complete.merge(
        counts,
        on=["donor_key", "condition", "LEC_state"],
        how="left",
        validate="one_to_one",
    )
    complete["n_state_LECs"] = complete["n_state_LECs"].fillna(0).astype(int)
    complete["state_percentage_of_donor_LECs"] = (
        100 * complete["n_state_LECs"] / complete["donor_n_final_LECs"]
    )
    complete["minimum_LECs_per_donor"] = threshold
    complete["analysis_tier"] = "exploratory_min10" if threshold == 10 else "sensitivity_min15"
    return complete


def hfpef_effects(donors: pd.DataFrame, threshold: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + threshold)
    records: list[dict[str, object]] = []
    for state in ALL_STATES:
        state_rows = donors.loc[donors["LEC_state"].eq(state)]
        disease = state_rows.loc[
            state_rows["condition"].eq("HFpEF"), "state_percentage_of_donor_LECs"
        ].to_numpy(float)
        reference = state_rows.loc[
            state_rows["condition"].eq("NF"), "state_percentage_of_donor_LECs"
        ].to_numpy(float)
        if not len(disease) or not len(reference):
            raise ValueError(f"Missing HFpEF or NF donors for {state} at ≥{threshold}.")
        mean_difference = float(disease.mean() - reference.mean())
        ci_low, ci_high = bootstrap_mean_difference(disease, reference, rng)
        identical_constant = (
            np.all(disease == disease[0])
            and np.all(reference == reference[0])
            and disease[0] == reference[0]
        )
        p_value = (
            np.nan
            if identical_constant
            else float(
                stats.mannwhitneyu(
                    disease, reference, alternative="two-sided", method="asymptotic"
                ).pvalue
            )
        )
        records.append(
            {
                "comparison": "dataset11: HFpEF vs NF",
                "dataset_folder": "dataset11",
                "reference_condition": "NF",
                "disease_condition": "HFpEF",
                "LEC_state": state,
                "n_reference_donors": len(reference),
                "n_disease_donors": len(disease),
                "reference_mean_percentage": float(reference.mean()),
                "disease_mean_percentage": float(disease.mean()),
                "mean_difference_percentage_points": mean_difference,
                "mean_difference_ci_low": ci_low,
                "mean_difference_ci_high": ci_high,
                "mann_whitney_p": p_value,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "confidence_level": CONFIDENCE_LEVEL,
                "minimum_LECs_per_donor": threshold,
                "analysis_tier": (
                    "exploratory_min10" if threshold == 10 else "sensitivity_min15"
                ),
            }
        )
    output = pd.DataFrame.from_records(records)
    output["mann_whitney_FDR_within_dataset11_threshold"] = bh_adjust(
        output["mann_whitney_p"]
    )
    return output


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [PRIMARY_CSV, CELL_CSV, MIN10_SUMMARY, MIN15_SUMMARY]:
        if not path.exists():
            raise FileNotFoundError(path)
    validate_fig07_threshold_summary(MIN10_SUMMARY, 10)
    validate_fig07_threshold_summary(MIN15_SUMMARY, 15)

    primary_all = pd.read_csv(PRIMARY_CSV, low_memory=False)
    cells = pd.read_csv(CELL_CSV, low_memory=False)
    require_columns(
        primary_all,
        {
            "comparison",
            "LEC_state",
            "mean_difference_percentage_points",
            "mean_difference_ci_low",
            "mean_difference_ci_high",
            "mann_whitney_p",
        },
        PRIMARY_CSV,
    )
    require_columns(
        cells,
        {
            "dataset_folder",
            "condition",
            "donor_key",
            "LEC_state",
            "donor_n_final_LECs",
        },
        CELL_CSV,
    )
    primary_all = primary_all.loc[primary_all["comparison"].isin(PRIMARY_COMPARISONS)].copy()
    if len(primary_all) != len(PRIMARY_COMPARISONS) * len(ALL_STATES):
        raise ValueError("The primary three-comparison by nine-state matrix is incomplete.")

    donors10 = donor_state_table(cells, 10)
    donors15 = donor_state_table(cells, 15)
    effects10 = hfpef_effects(donors10, 10)
    effects15 = hfpef_effects(donors15, 15)

    for threshold_effects, name in [(effects10, "min10"), (effects15, "min15")]:
        combined_p = pd.concat(
            [primary_all["mann_whitney_p"], threshold_effects["mann_whitney_p"]],
            ignore_index=True,
        )
        combined_q = bh_adjust(combined_p)
        primary_all[f"global_BH_with_HFpEF_{name}"] = combined_q.iloc[: len(primary_all)].to_numpy()
        threshold_effects["global_BH_four_comparison_family"] = combined_q.iloc[
            len(primary_all) :
        ].to_numpy()

    primary = primary_all.loc[primary_all["LEC_state"].isin(SELECTED_STATES)].copy()
    if len(primary) != len(PRIMARY_COMPARISONS) * len(SELECTED_STATES):
        raise ValueError("The selected primary effect matrix is incomplete.")
    return primary, effects10, effects15, pd.concat([donors10, donors15], ignore_index=True)


def draw_primary_effects(axis: plt.Axes, effects: pd.DataFrame) -> None:
    state_positions = dict(zip(SELECTED_STATES, [3, 2, 1, 0], strict=True))
    offsets = {
        "dataset01: DCM vs NF": 0.22,
        "dataset01: HCM vs NF": 0.0,
        "dataset08: DoxCM vs NF": -0.22,
    }
    for comparison in PRIMARY_COMPARISONS:
        part = effects.loc[effects["comparison"].eq(comparison)].set_index("LEC_state")
        for state in SELECTED_STATES:
            row = part.loc[state]
            y = state_positions[state] + offsets[comparison]
            mean = float(row["mean_difference_percentage_points"])
            low = float(row["mean_difference_ci_low"])
            high = float(row["mean_difference_ci_high"])
            significant = float(row["global_BH_with_HFpEF_min10"]) < 0.05
            color = COMPARISON_COLORS[comparison]
            axis.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                fmt="none",
                ecolor=color,
                elinewidth=1.5,
                capsize=2.8,
                zorder=2,
            )
            axis.scatter(
                mean,
                y,
                s=58,
                facecolor=color if significant else "white",
                edgecolor=color,
                linewidth=1.4,
                zorder=3,
            )
    axis.axvline(0, color="#5E676D", linewidth=1.0, linestyle="--", zorder=0)
    axis.set_yticks(
        [state_positions[state] for state in SELECTED_STATES],
        [f"{state.replace('state', 'State ')}\n{STATE_DEFINITIONS[state]}" for state in SELECTED_STATES],
    )
    axis.set_ylim(-0.55, 3.55)
    axis.set_xlim(-96, 63)
    axis.set_xlabel("Disease - matched NF abundance difference (percentage points)", fontweight="bold")
    axis.set_title("Primary within-dataset effects · ≥25 LECs/donor", loc="left", fontweight="bold", pad=10)
    axis.set_axisbelow(True)
    axis.xaxis.grid(True, color=GRID, linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#8C959B")
    axis.tick_params(axis="y", length=0, pad=7)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=COMPARISON_COLORS[comparison],
            markerfacecolor=COMPARISON_COLORS[comparison],
            linewidth=1.4,
            label=COMPARISON_LABELS[comparison],
        )
        for comparison in PRIMARY_COMPARISONS
    ]
    axis.legend(handles=handles, frameon=False, loc="lower right", fontsize=8.7)


def draw_hfpef_effects(axis: plt.Axes, effects10: pd.DataFrame, effects15: pd.DataFrame) -> None:
    state_positions = dict(zip(SELECTED_STATES, [3, 2, 1, 0], strict=True))
    for threshold, effects, offset in [(10, effects10, 0.12), (15, effects15, -0.12)]:
        part = effects.loc[effects["LEC_state"].isin(SELECTED_STATES)].set_index("LEC_state")
        style = HFPEF_STYLES[threshold]
        for state in SELECTED_STATES:
            row = part.loc[state]
            y = state_positions[state] + offset
            mean = float(row["mean_difference_percentage_points"])
            low = float(row["mean_difference_ci_low"])
            high = float(row["mean_difference_ci_high"])
            q_value = float(row["global_BH_four_comparison_family"])
            significant = np.isfinite(q_value) and q_value < 0.05
            linestyle = "-" if threshold == 10 else "--"
            axis.plot([low, high], [y, y], color=style["color"], linewidth=1.45, linestyle=linestyle, zorder=2)
            axis.plot([low, low], [y - 0.055, y + 0.055], color=style["color"], linewidth=0.9)
            axis.plot([high, high], [y - 0.055, y + 0.055], color=style["color"], linewidth=0.9)
            axis.scatter(
                mean,
                y,
                s=55 if threshold == 10 else 45,
                marker="D",
                facecolor=style["color"] if significant else "white",
                edgecolor=style["color"],
                linewidth=1.3,
                zorder=3,
            )
    axis.axvline(0, color="#5E676D", linewidth=1.0, linestyle="--", zorder=0)
    axis.set_yticks(
        [state_positions[state] for state in SELECTED_STATES],
        [state.replace("state", "State ") for state in SELECTED_STATES],
    )
    axis.set_ylim(-0.55, 3.55)
    axis.set_xlim(-58, 62)
    axis.set_xlabel("HFpEF - matched NF difference (percentage points)", fontweight="bold")
    axis.set_title(
        "D11 HFpEF vs NF · threshold sensitivity\nRelaxed cutoff applies to both groups",
        loc="left",
        fontweight="bold",
        pad=10,
    )
    axis.set_axisbelow(True)
    axis.xaxis.grid(True, color=GRID, linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#8C959B")
    axis.tick_params(axis="y", length=0)
    handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color=HFPEF_STYLES[threshold]["color"],
            markerfacecolor="white",
            linestyle="-" if threshold == 10 else "--",
            linewidth=1.4,
            label=HFPEF_STYLES[threshold]["label"],
        )
        for threshold in [10, 15]
    ]
    axis.legend(handles=handles, frameon=False, loc="upper left", fontsize=8.0)


def write_outputs(
    primary: pd.DataFrame,
    effects10: pd.DataFrame,
    effects15: pd.DataFrame,
    donors: pd.DataFrame,
) -> None:
    primary_source = primary.copy()
    primary_source.insert(0, "figure_panel", "4D")
    primary_source.insert(1, "record_type", "primary_min25_disease_vs_NF_effect")
    primary_source["minimum_LECs_per_donor"] = 25
    primary_source["analysis_tier"] = "primary_min25"
    primary_source["state_definition"] = primary_source["LEC_state"].map(STATE_DEFINITIONS)
    primary_source["source_file"] = PRIMARY_CSV.relative_to(ROOT).as_posix()

    hfpef_sources = []
    for table in [effects10, effects15]:
        selected = table.loc[table["LEC_state"].isin(SELECTED_STATES)].copy()
        selected.insert(0, "figure_panel", "4D")
        selected.insert(1, "record_type", "HFpEF_vs_NF_threshold_effect")
        selected["state_definition"] = selected["LEC_state"].map(STATE_DEFINITIONS)
        selected["source_file"] = CELL_CSV.relative_to(ROOT).as_posix()
        hfpef_sources.append(selected)

    donor_source = donors.loc[donors["LEC_state"].isin(SELECTED_STATES)].copy()
    donor_source.insert(0, "figure_panel", "4D")
    donor_source.insert(1, "record_type", "HFpEF_NF_donor_state_abundance")
    donor_source["state_definition"] = donor_source["LEC_state"].map(STATE_DEFINITIONS)
    donor_source["source_file"] = CELL_CSV.relative_to(ROOT).as_posix()
    pd.concat(
        [primary_source, *hfpef_sources, donor_source], ignore_index=True, sort=False
    ).to_csv(SOURCE_CSV, index=False)

    pd.DataFrame(
        [
            {
                "figure_panel": "4D primary effects",
                "input_file": PRIMARY_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "comparison;LEC_state;mann_whitney_p;mean_difference_percentage_points;mean_difference_ci_low;mean_difference_ci_high",
                "purpose": "primary within-dataset effects for DCM HCM and DoxCM at >=25 LECs per donor",
            },
            {
                "figure_panel": "4D HFpEF threshold effects",
                "input_file": CELL_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "dataset_folder;condition;donor_key;LEC_state;donor_n_final_LECs",
                "purpose": "donor-state percentages and HFpEF-versus-NF effects at dataset11 thresholds >=10 and >=15",
            },
            {
                "figure_panel": "4D threshold provenance",
                "input_file": MIN10_SUMMARY.relative_to(ROOT).as_posix(),
                "used_fields": "receiver_min_cells;n_reference_donors;n_disease_donors;n_receiver_cells;validation_status;status",
                "purpose": "validate the completed exploratory >=10 dataset11 tier",
            },
            {
                "figure_panel": "4D threshold provenance",
                "input_file": MIN15_SUMMARY.relative_to(ROOT).as_posix(),
                "used_fields": "receiver_min_cells;n_reference_donors;n_disease_donors;n_receiver_cells;validation_status;status",
                "purpose": "validate the completed >=15 dataset11 sensitivity tier",
            },
        ]
    ).to_csv(MANIFEST_CSV, index=False)


def main() -> None:
    primary, effects10, effects15, donors = load_inputs()
    write_outputs(primary, effects10, effects15, donors)

    figure = plt.figure(figsize=(14.8, 6.6), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[2.0, 1.08],
        left=0.225,
        right=0.975,
        top=0.755,
        bottom=0.19,
        wspace=0.22,
    )
    ax_primary = figure.add_subplot(grid[0, 0])
    ax_hfpef = figure.add_subplot(grid[0, 1])
    draw_primary_effects(ax_primary, primary)
    draw_hfpef_effects(ax_hfpef, effects10, effects15)

    figure.text(0.018, 0.955, "D", fontsize=18, fontweight="bold", color=TEXT, va="top")
    figure.text(
        0.075,
        0.955,
        "Shared and disease-specific changes in representative LEC states",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    figure.text(
        0.075,
        0.895,
        "Points are donor-level effects; horizontal bars are 95% donor-bootstrap confidence intervals.",
        fontsize=10.2,
        color=SUBTEXT,
        va="top",
    )
    figure.text(
        0.075,
        0.842,
        "D01/D08: primary ≥25 LECs/donor. D11 HFpEF vs NF: exploratory ≥10; ≥15 sensitivity. Comparisons remain within dataset.",
        fontsize=9.7,
        color=SUBTEXT,
        va="top",
    )
    figure.text(
        0.075,
        0.045,
        "Filled markers indicate global BH q<0.05: primary and ≥10 markers use the four-comparison ≥10 family; ≥15 is corrected in its corresponding sensitivity family. HFpEF direction is consistent, but significance is threshold-sensitive.",
        fontsize=8.8,
        color=SUBTEXT,
        va="bottom",
    )
    figure.savefig(FIGURE_PNG, dpi=450, facecolor="white")
    figure.savefig(FIGURE_PDF, facecolor="white")
    figure.savefig(
        FIGURE_JPG,
        dpi=450,
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)

    print(f"Figure 4D written to: {OUTPUT_DIR}")
    print("Primary: DCM/HCM/DoxCM at >=25 LECs per donor")
    print("HFpEF vs NF: >=10 exploratory (7 vs 6 donors); >=15 sensitivity (5 vs 4 donors)")


if __name__ == "__main__":
    main()
