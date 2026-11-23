"""Create aging Figure 2D as a donor-level effect-size forest plot.

This revision mirrors the presentation logic of disease Figure 5E. Q1 is the
reference group; Q2+Q3 and Q4 effects are differences in mean donor-level LEC
state abundance. Horizontal bars are percentile 95% donor-bootstrap confidence
intervals. Pairwise Mann-Whitney P values are BH-adjusted across the complete
11-state by two-contrast family. The existing state-level Kruskal-Wallis BH FDR
is retained as complementary omnibus evidence.

Outputs are written as PNG, PDF, JPG, figure source data, and an input manifest.
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
INPUT_DIR = (
    ROOT
    / "new_result2_aging/08_7_min25_LEC_filtered_age_analysis/dataset13/"
    "three_groups_Q2_Q3_merged/donor_level"
)
DONOR_CSV = INPUT_DIR / "08_7_Q2_Q3_merged_donor_state_counts_percentages.csv.gz"
SUMMARY_CSV = INPUT_DIR / "08_7_Q2_Q3_merged_state_group_summary_mean_sem.csv"
TEST_CSV = INPUT_DIR / "08_7_Q2_Q3_merged_state_omnibus_tests.csv"

OUTPUT_DIR = ROOT / "new_result2/aging/Fig02D_aging_representative_state_abundance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_STEM = OUTPUT_DIR / "Fig02D_aging_representative_state_abundance"
SOURCE_CSV = OUTPUT_DIR / "Fig02D_source_data.csv"
MANIFEST_CSV = OUTPUT_DIR / "Fig02D_input_manifest.csv"

SCRIPT_VERSION = "Fig02D_Q1_referenced_effect_forest_v2_2026_08_31"
BOOTSTRAP_ITERATIONS = 10_000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 20260831

AGE_GROUPS = ["Q1", "Q2+Q3", "Q4"]
EXPECTED_DONORS = {"Q1": 4, "Q2+Q3": 4, "Q4": 5}
ALL_STATES = [f"state{value}" for value in range(11)]
SELECTED_STATES = ["state1", "state4", "state6"]
STATE_DEFINITIONS = {
    "state1": "Matrix-remodeling and signaling LEC",
    "state4": "Cytokine-responsive immune LEC",
    "state6": "Activated endothelial LEC",
}
STATE_SELECTION_RATIONALE = {
    "state1": "high-support representative state",
    "state4": "strongest decreasing representative pattern",
    "state6": "lower-abundance activated representative state",
}
CONTRASTS = [
    {
        "comparison_group": "Q2+Q3",
        "reference_group": "Q1",
        "label": "Q2+Q3 - Q1",
        "color": "#758794",
        "offset": 0.12,
    },
    {
        "comparison_group": "Q4",
        "reference_group": "Q1",
        "label": "Q4 - Q1",
        "color": "#2F6F9F",
        "offset": -0.12,
    },
]

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
        adjusted[finite] = multipletests(
            numeric.to_numpy(float)[finite], method="fdr_bh"
        )[1]
    return pd.Series(adjusted, index=values.index)


def bootstrap_mean_difference(
    comparison: np.ndarray,
    reference: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    comparison_indices = rng.integers(
        0, len(comparison), size=(BOOTSTRAP_ITERATIONS, len(comparison))
    )
    reference_indices = rng.integers(
        0, len(reference), size=(BOOTSTRAP_ITERATIONS, len(reference))
    )
    differences = (
        comparison[comparison_indices].mean(axis=1)
        - reference[reference_indices].mean(axis=1)
    )
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    low, high = np.quantile(differences, [alpha, 1 - alpha])
    return float(low), float(high)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [DONOR_CSV, SUMMARY_CSV, TEST_CSV]:
        if not path.is_file():
            raise FileNotFoundError(path)

    donor = pd.read_csv(DONOR_CSV, low_memory=False)
    summary = pd.read_csv(SUMMARY_CSV)
    tests = pd.read_csv(TEST_CSV)
    require_columns(
        donor,
        {
            "donor_key",
            "LEC_state",
            "age_group",
            "n_total_LEC_cells",
            "cell_state_percentage",
        },
        DONOR_CSV,
    )
    require_columns(
        summary,
        {"LEC_state", "age_group", "n_donors", "mean_percentage"},
        SUMMARY_CSV,
    )
    require_columns(
        tests,
        {"LEC_state", "kruskal_p", "kruskal_fdr_bh", "epsilon_squared"},
        TEST_CSV,
    )

    donor["cell_state_percentage"] = pd.to_numeric(
        donor["cell_state_percentage"], errors="raise"
    )
    donor["n_total_LEC_cells"] = pd.to_numeric(
        donor["n_total_LEC_cells"], errors="raise"
    )
    if (donor["n_total_LEC_cells"] < 25).any():
        raise ValueError("A retained donor has fewer than 25 total LECs.")
    if donor.duplicated(["donor_key", "LEC_state"]).any():
        raise ValueError("Donor/state rows are not unique.")
    if set(donor["LEC_state"]) != set(ALL_STATES):
        raise ValueError("The expected 11-state donor table is incomplete.")
    rows_per_donor = donor.groupby("donor_key").size()
    if not rows_per_donor.eq(len(ALL_STATES)).all():
        raise ValueError("Each donor must have one row for every LEC state.")

    group_counts = (
        donor[["donor_key", "age_group"]]
        .drop_duplicates()["age_group"]
        .value_counts()
        .to_dict()
    )
    if group_counts != EXPECTED_DONORS:
        raise ValueError(f"Unexpected retained donor support: {group_counts}")

    calculated = (
        donor.groupby(["LEC_state", "age_group"], observed=True)
        ["cell_state_percentage"]
        .agg(n_donors="size", mean_percentage="mean")
        .reset_index()
    )
    check = calculated.merge(
        summary[["LEC_state", "age_group", "n_donors", "mean_percentage"]],
        on=["LEC_state", "age_group"],
        suffixes=("_calculated", "_recorded"),
        validate="one_to_one",
    )
    if len(check) != len(ALL_STATES) * len(AGE_GROUPS):
        raise ValueError("The recorded state-by-age summary is incomplete.")
    if not (
        check["n_donors_calculated"].eq(check["n_donors_recorded"]).all()
        and np.allclose(
            check["mean_percentage_calculated"],
            check["mean_percentage_recorded"],
            rtol=0,
            atol=1e-10,
        )
    ):
        raise ValueError("Donor data do not reproduce the recorded group summary.")
    if set(tests["LEC_state"]) != set(ALL_STATES) or len(tests) != len(ALL_STATES):
        raise ValueError("The state-level omnibus-test table is incomplete.")
    return donor, summary, tests


def calculate_effects(donor: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        comparison_group = str(contrast["comparison_group"])
        reference_group = str(contrast["reference_group"])
        for state_index, state in enumerate(ALL_STATES):
            state_rows = donor.loc[donor["LEC_state"].eq(state)]
            comparison = state_rows.loc[
                state_rows["age_group"].eq(comparison_group),
                "cell_state_percentage",
            ].to_numpy(float)
            reference = state_rows.loc[
                state_rows["age_group"].eq(reference_group),
                "cell_state_percentage",
            ].to_numpy(float)
            if len(comparison) != EXPECTED_DONORS[comparison_group]:
                raise ValueError(f"Unexpected {comparison_group} support for {state}.")
            if len(reference) != EXPECTED_DONORS[reference_group]:
                raise ValueError(f"Unexpected {reference_group} support for {state}.")

            rng = np.random.default_rng(
                RANDOM_SEED + 100 * contrast_index + state_index
            )
            low, high = bootstrap_mean_difference(comparison, reference, rng)
            identical_constant = (
                np.all(comparison == comparison[0])
                and np.all(reference == reference[0])
                and comparison[0] == reference[0]
            )
            p_value = (
                np.nan
                if identical_constant
                else float(
                    stats.mannwhitneyu(
                        comparison,
                        reference,
                        alternative="two-sided",
                        method="asymptotic",
                    ).pvalue
                )
            )
            records.append(
                {
                    "contrast": str(contrast["label"]),
                    "comparison_group": comparison_group,
                    "reference_group": reference_group,
                    "LEC_state": state,
                    "n_comparison_donors": len(comparison),
                    "n_reference_donors": len(reference),
                    "comparison_mean_percentage": float(comparison.mean()),
                    "reference_mean_percentage": float(reference.mean()),
                    "mean_difference_percentage_points": float(
                        comparison.mean() - reference.mean()
                    ),
                    "mean_difference_ci_low": low,
                    "mean_difference_ci_high": high,
                    "mann_whitney_p": p_value,
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "confidence_level": CONFIDENCE_LEVEL,
                }
            )

    effects = pd.DataFrame.from_records(records)
    effects["global_BH_q_22_tests"] = bh_adjust(effects["mann_whitney_p"])
    effects["global_BH_significant"] = effects["global_BH_q_22_tests"] < 0.05
    effects = effects.merge(
        tests[["LEC_state", "kruskal_p", "kruskal_fdr_bh", "epsilon_squared"]],
        on="LEC_state",
        how="left",
        validate="many_to_one",
    )
    effects["display_in_panel"] = effects["LEC_state"].isin(SELECTED_STATES)
    effects["state_definition"] = effects["LEC_state"].map(STATE_DEFINITIONS)
    effects["state_selection_rationale"] = effects["LEC_state"].map(
        STATE_SELECTION_RATIONALE
    )
    return effects


def draw_effects(axis: plt.Axes, effects: pd.DataFrame) -> None:
    state_positions = dict(zip(SELECTED_STATES, [2, 1, 0], strict=True))
    for contrast in CONTRASTS:
        part = effects.loc[
            effects["contrast"].eq(contrast["label"])
            & effects["LEC_state"].isin(SELECTED_STATES)
        ].set_index("LEC_state")
        for state in SELECTED_STATES:
            row = part.loc[state]
            y = state_positions[state] + float(contrast["offset"])
            mean = float(row["mean_difference_percentage_points"])
            low = float(row["mean_difference_ci_low"])
            high = float(row["mean_difference_ci_high"])
            color = str(contrast["color"])
            axis.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                fmt="none",
                ecolor=color,
                elinewidth=1.6,
                capsize=3.0,
                zorder=2,
            )
            axis.scatter(
                mean,
                y,
                s=62,
                facecolor=color,
                edgecolor=color,
                linewidth=1.5,
                zorder=3,
            )

    displayed = effects.loc[effects["display_in_panel"]]
    limit = float(
        np.nanmax(
            np.abs(
                displayed[
                    ["mean_difference_ci_low", "mean_difference_ci_high"]
                ].to_numpy(float)
            )
        )
    )
    limit = max(10.0, np.ceil((limit + 5.0) / 10.0) * 10.0)
    axis.axvline(0, color="#687279", linewidth=1.0, linestyle="--", zorder=0)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-0.50, 2.50)
    axis.set_yticks(
        [state_positions[state] for state in SELECTED_STATES],
        [
            f"{state.replace('state', 'State ')}\n{STATE_DEFINITIONS[state]}"
            for state in SELECTED_STATES
        ],
    )
    axis.set_xlabel(
        "Age-group abundance difference from Q1 (percentage points)",
        fontweight="bold",
    )
    axis.set_title(
        "Donor-level aging effects · Q1 reference",
        loc="left",
        fontweight="bold",
        pad=11,
    )
    axis.set_axisbelow(True)
    axis.xaxis.grid(True, color=GRID, linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#8C959B")
    axis.tick_params(axis="y", length=0, pad=8)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=str(contrast["color"]),
            markerfacecolor=str(contrast["color"]),
            markeredgewidth=1.4,
            linewidth=1.5,
            label=str(contrast["label"]),
        )
        for contrast in CONTRASTS
    ]
    axis.legend(handles=handles, frameon=False, loc="lower right", fontsize=9.2)


def write_outputs(
    donor: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    effects: pd.DataFrame,
) -> None:
    effect_source = effects.copy()
    effect_source.insert(0, "script_version", SCRIPT_VERSION)
    effect_source.insert(1, "figure_panel", "2D")
    effect_source.insert(2, "record_type", "age_group_contrast_effect")
    effect_source["minimum_LECs_per_donor"] = 25
    effect_source["source_file"] = DONOR_CSV.relative_to(ROOT).as_posix()

    donor_source = donor.loc[donor["LEC_state"].isin(SELECTED_STATES)].copy()
    donor_source = donor_source.merge(
        summary[
            [
                "LEC_state",
                "age_group",
                "n_donors",
                "mean_percentage",
                "sem_percentage",
                "median_percentage",
            ]
        ],
        on=["LEC_state", "age_group"],
        how="left",
        validate="many_to_one",
    ).merge(
        tests[["LEC_state", "kruskal_p", "kruskal_fdr_bh", "epsilon_squared"]],
        on="LEC_state",
        how="left",
        validate="many_to_one",
    )
    donor_source.insert(0, "script_version", SCRIPT_VERSION)
    donor_source.insert(1, "figure_panel", "2D")
    donor_source.insert(2, "record_type", "displayed_donor_state_abundance")
    donor_source["display_in_panel"] = True
    donor_source["state_definition"] = donor_source["LEC_state"].map(
        STATE_DEFINITIONS
    )
    donor_source["state_selection_rationale"] = donor_source["LEC_state"].map(
        STATE_SELECTION_RATIONALE
    )
    donor_source["minimum_LECs_per_donor"] = 25
    donor_source["source_file"] = DONOR_CSV.relative_to(ROOT).as_posix()

    pd.concat([effect_source, donor_source], ignore_index=True, sort=False).to_csv(
        SOURCE_CSV, index=False
    )

    pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "figure_panel": "2D effects and donor values",
                "input_file": DONOR_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "donor_key;LEC_state;age_group;n_total_LEC_cells;cell_state_percentage",
                "purpose": "Q1-referenced mean differences, donor-bootstrap confidence intervals, pairwise tests, and displayed donor values after the >=25-LEC filter",
            },
            {
                "script_version": SCRIPT_VERSION,
                "figure_panel": "2D validation",
                "input_file": SUMMARY_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "LEC_state;age_group;n_donors;mean_percentage;sem_percentage;median_percentage",
                "purpose": "validate donor-level group support and descriptive statistics",
            },
            {
                "script_version": SCRIPT_VERSION,
                "figure_panel": "2D omnibus evidence",
                "input_file": TEST_CSV.relative_to(ROOT).as_posix(),
                "used_fields": "LEC_state;kruskal_p;kruskal_fdr_bh;epsilon_squared",
                "purpose": "retain the existing 11-state Kruskal-Wallis BH-FDR results as complementary omnibus evidence",
            },
        ]
    ).to_csv(MANIFEST_CSV, index=False)


def main() -> None:
    donor, summary, tests = load_inputs()
    effects = calculate_effects(donor, tests)
    write_outputs(donor, summary, tests, effects)

    figure, axis = plt.subplots(figsize=(13.6, 5.7), facecolor="white")
    figure.subplots_adjust(left=0.33, right=0.965, top=0.72, bottom=0.16)
    draw_effects(axis, effects)

    figure.text(0.018, 0.955, "D", fontsize=18, fontweight="bold", color=TEXT, va="top")
    figure.text(
        0.075,
        0.955,
        "Age-associated changes in representative LEC states",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    figure.text(
        0.075,
        0.885,
        "Points are differences in donor-level mean abundance; horizontal bars are 95% donor-bootstrap confidence intervals.",
        fontsize=10.2,
        color=SUBTEXT,
        va="top",
    )
    figure.text(
        0.075,
        0.825,
        "Primary analysis: ≥25 total LECs/donor; Q1 n=4, Q2+Q3 n=4, Q4 n=5. Q1 is the reference group.",
        fontsize=9.7,
        color=SUBTEXT,
        va="top",
    )

    n_pairwise_significant = int(effects["global_BH_significant"].sum())

    figure.savefig(FIGURE_STEM.with_suffix(".png"), dpi=450, facecolor="white")
    figure.savefig(FIGURE_STEM.with_suffix(".pdf"), facecolor="white")
    figure.savefig(
        FIGURE_STEM.with_suffix(".jpg"),
        dpi=450,
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)

    print(f"Figure 2D written to: {OUTPUT_DIR}")
    print("Displayed states: state1, state4, state6")
    print("Contrasts: Q2+Q3 - Q1; Q4 - Q1")
    print(f"Global BH-significant pairwise tests: {n_pairwise_significant}/22")


if __name__ == "__main__":
    main()
