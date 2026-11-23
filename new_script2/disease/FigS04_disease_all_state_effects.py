"""Create the supplementary all-state donor-level disease comparison panel.

Dataset01 DCM/HCM and dataset08 DoxCM remain primary comparisons at >=25
final LECs per donor. Dataset11 HFpEF versus matched NF is included at >=10 as
an exploratory analysis, with >=15 shown as a nested threshold sensitivity.
The dataset11 cutoff is applied to both HFpEF and NF donors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
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

OUTPUT_DIR = ROOT / "new_result2/disease/FigS04_disease_all_state_effects"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_PNG = OUTPUT_DIR / "FigS04_disease_all_state_effects.png"
FIGURE_PDF = OUTPUT_DIR / "FigS04_disease_all_state_effects.pdf"
FIGURE_JPG = OUTPUT_DIR / "FigS04_disease_all_state_effects.jpg"
SOURCE_CSV = OUTPUT_DIR / "FigS04_source_data.csv"
MANIFEST_CSV = OUTPUT_DIR / "FigS04_input_manifest.csv"

SCRIPT_VERSION = "FigS04_all_state_HFpEF_min10_min15_v2_2026_08_26"
BOOTSTRAP_ITERATIONS = 10_000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 20260826

STATES = [f"state{value}" for value in range(9)]
COMPARISONS = [
    "dataset01: DCM vs NF",
    "dataset01: HCM vs NF",
    "dataset08: DoxCM vs NF",
]
LABELS = {
    "dataset01: DCM vs NF": "DCM - NF\n(dataset01)",
    "dataset01: HCM vs NF": "HCM - NF\n(dataset01)",
    "dataset08: DoxCM vs NF": "DoxCM - NF\n(dataset08)",
}
DEFINITIONS = {
    "state0": "Disease-associated remodeling LEC",
    "state1": "Homeostatic maintenance LEC",
    "state2": "Structural remodeling LEC",
    "state3": "Disease-associated endothelial regulatory LEC",
    "state4": "Cardiomyopathy-associated transitional state",
    "state5": "Activated vascular endothelial LEC",
    "state6": "Matrix-associated LEC",
    "state7": "Rare contractile-associated state",
    "state8": "Rare regulatory LEC",
}
EXPECTED_DONORS = {10: {"NF": 6, "HFpEF": 7}, 15: {"NF": 4, "HFpEF": 5}}
EXPECTED_RECEIVER_CELLS = {10: 349, 15: 301}

TEXT = "#202428"
SUBTEXT = "#5C646A"
GRID = "#D9E0E4"
EXPLORATORY = "#287C78"
SENSITIVITY = "#687279"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
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
        adjusted[finite] = multipletests(
            numeric.to_numpy(float)[finite], method="fdr_bh"
        )[1]
    return pd.Series(adjusted, index=values.index)


def bootstrap_mean_difference(
    disease: np.ndarray, reference: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    disease_indices = rng.integers(
        0, len(disease), size=(BOOTSTRAP_ITERATIONS, len(disease))
    )
    reference_indices = rng.integers(
        0, len(reference), size=(BOOTSTRAP_ITERATIONS, len(reference))
    )
    differences = (
        disease[disease_indices].mean(axis=1)
        - reference[reference_indices].mean(axis=1)
    )
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    low, high = np.quantile(differences, [alpha, 1 - alpha])
    return float(low), float(high)


def validate_threshold_summary(path: Path, threshold: int) -> None:
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
        raise ValueError(f"Expected one threshold-summary row in {path}.")
    row = summary.iloc[0]
    expected = EXPECTED_DONORS[threshold]
    checks = {
        "receiver_min_cells": int(row["receiver_min_cells"]) == threshold,
        "n_reference_donors": int(row["n_reference_donors"]) == expected["NF"],
        "n_disease_donors": int(row["n_disease_donors"]) == expected["HFpEF"],
        "n_receiver_cells": (
            int(row["n_receiver_cells"]) == EXPECTED_RECEIVER_CELLS[threshold]
        ),
        "validation_status": str(row["validation_status"]).lower() == "passed",
        "status": str(row["status"]).lower() == "completed",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"Dataset11 threshold provenance failed for >={threshold}: {failed}"
        )


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
        raise ValueError(
            f"Unexpected dataset11 donor counts at >={threshold}: {observed}"
        )
    if len(subset) != EXPECTED_RECEIVER_CELLS[threshold]:
        raise ValueError(
            f"Unexpected dataset11 LEC count at >={threshold}: {len(subset)}"
        )

    counts = (
        subset.groupby(["donor_key", "condition", "LEC_state"], observed=True)
        .size()
        .rename("n_state_LECs")
        .reset_index()
    )
    complete = pd.MultiIndex.from_product(
        [donor_meta["donor_key"], STATES], names=["donor_key", "LEC_state"]
    ).to_frame(index=False)
    complete = complete.merge(
        donor_meta, on="donor_key", how="left", validate="many_to_one"
    )
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
    complete["analysis_tier"] = (
        "exploratory_min10" if threshold == 10 else "sensitivity_min15"
    )
    return complete


def hfpef_effects(donors: pd.DataFrame, threshold: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + threshold)
    records: list[dict[str, object]] = []
    for state in STATES:
        state_rows = donors.loc[donors["LEC_state"].eq(state)]
        disease = state_rows.loc[
            state_rows["condition"].eq("HFpEF"),
            "state_percentage_of_donor_LECs",
        ].to_numpy(float)
        reference = state_rows.loc[
            state_rows["condition"].eq("NF"),
            "state_percentage_of_donor_LECs",
        ].to_numpy(float)
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
                    disease,
                    reference,
                    alternative="two-sided",
                    method="asymptotic",
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
                "mean_difference_percentage_points": float(
                    disease.mean() - reference.mean()
                ),
                "mean_difference_ci_low": ci_low,
                "mean_difference_ci_high": ci_high,
                "mann_whitney_p": p_value,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "confidence_level": CONFIDENCE_LEVEL,
                "minimum_LECs_per_donor": threshold,
                "analysis_tier": (
                    "exploratory_min10"
                    if threshold == 10
                    else "sensitivity_min15"
                ),
            }
        )
    output = pd.DataFrame.from_records(records)
    output["mann_whitney_FDR_within_dataset11_threshold"] = bh_adjust(
        output["mann_whitney_p"]
    )
    return output


def load_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    for path in [PRIMARY_CSV, CELL_CSV, MIN10_SUMMARY, MIN15_SUMMARY]:
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_threshold_summary(MIN10_SUMMARY, 10)
    validate_threshold_summary(MIN15_SUMMARY, 15)

    primary = pd.read_csv(PRIMARY_CSV, low_memory=False)
    cells = pd.read_csv(CELL_CSV, low_memory=False)
    require_columns(
        primary,
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
    primary = primary.loc[primary["comparison"].isin(COMPARISONS)].copy()
    if len(primary) != len(COMPARISONS) * len(STATES):
        raise ValueError("The primary three-comparison by nine-state matrix is incomplete.")

    donors10 = donor_state_table(cells, 10)
    donors15 = donor_state_table(cells, 15)
    effects10 = hfpef_effects(donors10, 10)
    effects15 = hfpef_effects(donors15, 15)
    for effects, name in [(effects10, "min10"), (effects15, "min15")]:
        combined_p = pd.concat(
            [primary["mann_whitney_p"], effects["mann_whitney_p"]],
            ignore_index=True,
        )
        combined_q = bh_adjust(combined_p)
        primary[f"global_BH_with_HFpEF_{name}"] = combined_q.iloc[
            : len(primary)
        ].to_numpy()
        effects["global_BH_four_comparison_family"] = combined_q.iloc[
            len(primary) :
        ].to_numpy()
    donors = pd.concat([donors10, donors15], ignore_index=True)
    return primary, effects10, effects15, donors


def draw_effect_matrix(
    axis: plt.Axes, effects: pd.DataFrame, figure: plt.Figure
) -> None:
    matrix = (
        effects.pivot(
            index="LEC_state",
            columns="comparison",
            values="mean_difference_percentage_points",
        )
        .reindex(index=STATES, columns=COMPARISONS)
    )
    significance = (
        effects.pivot(
            index="LEC_state",
            columns="comparison",
            values="global_BH_with_HFpEF_min10",
        )
        .reindex(index=STATES, columns=COMPARISONS)
        .lt(0.05)
    )
    if matrix.isna().any().any():
        raise ValueError("The all-state primary disease-effect matrix is incomplete.")
    limit = float(np.ceil(np.abs(matrix.to_numpy(float)).max() / 10) * 10)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = mpl.colormaps["RdBu_r"]
    axis.set_xlim(-0.5, len(COMPARISONS) - 0.5)
    axis.set_ylim(len(STATES) - 0.5, -0.5)
    axis.set_facecolor("white")
    for row, state in enumerate(STATES):
        for column, comparison in enumerate(COMPARISONS):
            value = float(matrix.loc[state, comparison])
            strong = bool(significance.loc[state, comparison])
            axis.add_patch(
                FancyBboxPatch(
                    (column - 0.37, row - 0.31),
                    0.74,
                    0.62,
                    boxstyle="round,pad=0.01,rounding_size=0.08",
                    facecolor=cmap(norm(value)),
                    edgecolor=TEXT if strong else "#BCC6CC",
                    linewidth=1.7 if strong else 0.65,
                )
            )
            axis.text(
                column,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="bold" if strong else "normal",
                color="white" if abs(value) > 0.55 * limit else TEXT,
            )
    axis.set_xticks(
        np.arange(len(COMPARISONS)), [LABELS[value] for value in COMPARISONS]
    )
    axis.set_yticks(
        np.arange(len(STATES)),
        [
            f"{state.replace('state', 'State ')}  |  {DEFINITIONS[state]}"
            for state in STATES
        ],
    )
    axis.tick_params(length=0, pad=7)
    axis.set_xticks(np.arange(-0.5, len(COMPARISONS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(STATES), 1), minor=True)
    axis.grid(which="minor", color=GRID, linewidth=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        "Primary within-dataset effects | \u226525 LECs/donor",
        loc="left",
        fontweight="bold",
        pad=12,
    )
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array([])
    cbar = figure.colorbar(
        mapper, ax=axis, orientation="vertical", fraction=0.045, pad=0.035
    )
    cbar.set_label(
        "Difference in donor LEC abundance (percentage points)",
        rotation=90,
        labelpad=10,
    )
    cbar.outline.set_visible(False)
    axis.text(
        0.0,
        -0.095,
        "Bold outline and number: global BH q<0.05 in the four-comparison \u226510 family",
        transform=axis.transAxes,
        fontsize=8.5,
        color=SUBTEXT,
        ha="left",
        va="top",
    )


def draw_hfpef_effects(
    axis: plt.Axes, effects10: pd.DataFrame, effects15: pd.DataFrame
) -> None:
    positions = {state: index for index, state in enumerate(STATES)}
    styles = {
        10: {
            "effects": effects10,
            "offset": -0.12,
            "color": EXPLORATORY,
            "linestyle": "-",
            "label": "\u226510 exploratory (7 HFpEF, 6 NF)",
        },
        15: {
            "effects": effects15,
            "offset": 0.12,
            "color": SENSITIVITY,
            "linestyle": "--",
            "label": "\u226515 sensitivity (5 HFpEF, 4 NF)",
        },
    }
    for style in styles.values():
        for row in style["effects"].itertuples(index=False):
            y = positions[row.LEC_state] + style["offset"]
            effect = float(row.mean_difference_percentage_points)
            low = float(row.mean_difference_ci_low)
            high = float(row.mean_difference_ci_high)
            q_value = row.global_BH_four_comparison_family
            significant = pd.notna(q_value) and float(q_value) < 0.05
            axis.plot(
                [low, high],
                [y, y],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.6,
                zorder=2,
            )
            axis.plot(
                effect,
                y,
                marker="D",
                markersize=6.8,
                markerfacecolor=style["color"] if significant else "white",
                markeredgecolor=style["color"],
                markeredgewidth=1.4,
                linestyle="none",
                zorder=3,
            )
    axis.axvline(0, color="#B8C0C5", linestyle="--", linewidth=1.0)
    axis.set_ylim(len(STATES) - 0.5, -0.5)
    axis.set_yticks(
        np.arange(len(STATES)), [state.replace("state", "State ") for state in STATES]
    )
    axis.set_xlabel("HFpEF - matched NF difference (percentage points)")
    axis.set_title(
        "Dataset11 HFpEF threshold sensitivity\nRelaxed cutoff applies to both groups",
        loc="left",
        fontweight="bold",
        pad=12,
    )
    axis.set_axisbelow(True)
    axis.xaxis.grid(True, color=GRID, linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#8C959B")
    axis.tick_params(axis="y", length=0)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                color=EXPLORATORY,
                markerfacecolor="white",
                markeredgecolor=EXPLORATORY,
                linewidth=1.5,
                label=styles[10]["label"],
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color=SENSITIVITY,
                markerfacecolor="white",
                markeredgecolor=SENSITIVITY,
                linestyle="--",
                linewidth=1.5,
                label=styles[15]["label"],
            ),
        ],
        frameon=False,
        fontsize=8.4,
        loc="upper left",
    )
    axis.text(
        0.0,
        -0.095,
        "Bars are 95% donor-bootstrap CIs; filled diamonds indicate global BH q<0.05.",
        transform=axis.transAxes,
        fontsize=8.5,
        color=SUBTEXT,
        ha="left",
        va="top",
    )


def write_source_and_manifest(
    primary: pd.DataFrame,
    effects10: pd.DataFrame,
    effects15: pd.DataFrame,
    donors: pd.DataFrame,
) -> None:
    primary_source = primary.copy()
    primary_source.insert(0, "figure_panel", "FigS04")
    primary_source.insert(1, "record_type", "primary_min25_disease_vs_NF_effect")
    primary_source["minimum_LECs_per_donor"] = 25
    primary_source["analysis_tier"] = "primary_min25"
    primary_source["state_definition"] = primary_source["LEC_state"].map(DEFINITIONS)
    primary_source["source_file"] = PRIMARY_CSV.relative_to(ROOT).as_posix()

    threshold_sources = []
    for effects in [effects10, effects15]:
        selected = effects.copy()
        selected.insert(0, "figure_panel", "FigS04")
        selected.insert(1, "record_type", "HFpEF_vs_NF_threshold_effect")
        selected["state_definition"] = selected["LEC_state"].map(DEFINITIONS)
        selected["source_file"] = CELL_CSV.relative_to(ROOT).as_posix()
        threshold_sources.append(selected)

    donor_source = donors.copy()
    donor_source.insert(0, "figure_panel", "FigS04")
    donor_source.insert(1, "record_type", "HFpEF_NF_donor_state_abundance")
    donor_source["state_definition"] = donor_source["LEC_state"].map(DEFINITIONS)
    donor_source["source_file"] = CELL_CSV.relative_to(ROOT).as_posix()

    source = pd.concat(
        [primary_source, *threshold_sources, donor_source],
        ignore_index=True,
        sort=False,
    )
    source.insert(0, "script_version", SCRIPT_VERSION)
    source.to_csv(SOURCE_CSV, index=False)

    pd.DataFrame(
        [
            {
                "figure_panel": "FigS04 primary effects",
                "input_file": PRIMARY_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "comparison;LEC_state;mean_difference_percentage_points;mean_difference_ci_low;mean_difference_ci_high;mann_whitney_p",
                "purpose": "all-state DCM HCM and DoxCM effects at >=25 LECs per donor",
            },
            {
                "figure_panel": "FigS04 HFpEF effects",
                "input_file": CELL_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "dataset_folder;condition;donor_key;LEC_state;donor_n_final_LECs",
                "purpose": "dataset11 donor-state percentages and HFpEF-versus-NF effects at >=10 and >=15",
            },
            {
                "figure_panel": "FigS04 threshold validation",
                "input_file": MIN10_SUMMARY.relative_to(ROOT).as_posix(),
                "used_fields": "receiver_min_cells;n_reference_donors;n_disease_donors;n_receiver_cells;validation_status;status",
                "purpose": "validate the completed dataset11 >=10 exploratory cohort",
            },
            {
                "figure_panel": "FigS04 threshold validation",
                "input_file": MIN15_SUMMARY.relative_to(ROOT).as_posix(),
                "used_fields": "receiver_min_cells;n_reference_donors;n_disease_donors;n_receiver_cells;validation_status;status",
                "purpose": "validate the completed dataset11 >=15 sensitivity cohort",
            },
        ]
    ).to_csv(MANIFEST_CSV, index=False)


def main() -> None:
    primary, effects10, effects15, donors = load_inputs()
    write_source_and_manifest(primary, effects10, effects15, donors)

    figure = plt.figure(figsize=(16.8, 9.4), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.45, 1.0],
        left=0.285,
        right=0.975,
        top=0.80,
        bottom=0.16,
        wspace=0.28,
    )
    ax_primary = figure.add_subplot(grid[0, 0])
    ax_hfpef = figure.add_subplot(grid[0, 1])
    draw_effect_matrix(ax_primary, primary, figure)
    draw_hfpef_effects(ax_hfpef, effects10, effects15)

    figure.text(
        0.07,
        0.955,
        "Supplementary Figure 4 | Donor-level disease effects across all LEC states",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    figure.text(
        0.07,
        0.895,
        "Primary comparisons use \u226525 LECs/donor. Dataset11 HFpEF versus matched NF is exploratory at \u226510, with \u226515 sensitivity.",
        fontsize=10.2,
        color=SUBTEXT,
        va="top",
    )
    figure.text(
        0.07,
        0.045,
        "Effects are differences in donor-normalized state abundance and include zero-count donor-state combinations. The relaxed dataset11 cutoff is applied equally to HFpEF and NF.",
        fontsize=9.2,
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
    print(f"Supplementary all-state disease panel written to: {OUTPUT_DIR}")
    print("HFpEF >=10: 7 vs 6 donors; >=15: 5 vs 4 donors")


if __name__ == "__main__":
    main()
