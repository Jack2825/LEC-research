#!/usr/bin/env python3
"""Build Supplementary Figure S2: LEC-number threshold rationale and sensitivity.

All panels use existing aging and disease analysis outputs. The script writes
each panel separately and a combined figure in PNG, PDF, and JPG formats, plus
one consolidated source-data CSV and one input manifest.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "new_result2" / "supplementary" / "FigS02_LEC_number_threshold_sensitivity"

INPUTS = {
    "aging_retention": ROOT
    / "new_result2_aging"
    / "08_6_donor_LEC_minimum_reliability"
    / "dataset13"
    / "08_6_candidate_filter_retention_impact.csv",
    "aging_filter_mapping": ROOT
    / "new_result2_aging"
    / "08_7_min25_LEC_filtered_age_analysis"
    / "dataset13"
    / "08_7_all_donor_filter_mapping.csv",
    "aging_state_counts": ROOT
    / "new_result2_aging"
    / "08_combined_LEC_recluster"
    / "dataset13"
    / "08_state_counts_by_donor.csv",
    "aging_main_figure_source": ROOT
    / "new_result2"
    / "aging"
    / "Fig02D_aging_representative_state_abundance"
    / "Fig02D_source_data.csv",
    "disease_retention": ROOT
    / "new_result2_disease"
    / "08_2_donor_state_reliability"
    / "tables"
    / "08_2_candidate_threshold_retention.csv",
    "disease_donor_state_composition": ROOT
    / "new_result2_disease"
    / "08_2_donor_state_reliability"
    / "tables"
    / "08_2_donor_state_composition_long.csv",
    "disease_main_figure_source": ROOT
    / "new_result2"
    / "disease"
    / "Fig05C_disease_state_abundance"
    / "Fig05C_source_data.csv",
}

NAVY = "#2C5F8A"
TEAL = "#3A8D8C"
GOLD = "#C6902F"
PLUM = "#80658A"
SLATE = "#687786"
LIGHT = "#D9DEE3"
PALE_TEAL = "#E8F1F0"
DARK = "#202529"
GRID = "#DDE2E6"

AGING_STATES = ["state1", "state4", "state6"]
AGING_STATE_LABELS = {
    "state1": "State 1",
    "state4": "State 4",
    "state6": "State 6",
}
AGING_STATE_COLORS = {"state1": NAVY, "state4": GOLD, "state6": TEAL}
AGING_STATE_MARKERS = {"state1": "o", "state4": "s", "state6": "^"}

DISEASE_SERIES = [
    {
        "series": "D01 DCM–NF · State 0",
        "dataset_folder": "dataset01",
        "disease_condition": "DCM",
        "reference_condition": "NF",
        "LEC_state": "state0",
        "color": NAVY,
        "marker": "o",
        "thresholds": [5, 10, 15, 20, 25, 30],
    },
    {
        "series": "D01 HCM–NF · State 0",
        "dataset_folder": "dataset01",
        "disease_condition": "HCM",
        "reference_condition": "NF",
        "LEC_state": "state0",
        "color": TEAL,
        "marker": "s",
        "thresholds": [5, 10, 15, 20, 25, 30],
    },
    {
        "series": "D08 DoxCM–NF · State 3",
        "dataset_folder": "dataset08",
        "disease_condition": "DoxCM",
        "reference_condition": "NF",
        "LEC_state": "state3",
        "color": GOLD,
        "marker": "^",
        "thresholds": [5, 10, 15, 20, 25, 30],
    },
    {
        "series": "D11 HFpEF–NF · State 0",
        "dataset_folder": "dataset11",
        "disease_condition": "HFpEF",
        "reference_condition": "NF",
        "LEC_state": "state0",
        "color": PLUM,
        "marker": "D",
        "thresholds": [10, 15],
    },
]

DISEASE_COHORT_LABELS = {
    "dataset01 | DCM": "D01 DCM",
    "dataset01 | HCM": "D01 HCM",
    "dataset01 | NF": "D01 NF",
    "dataset08 | DoxCM": "D08 DoxCM",
    "dataset08 | NF": "D08 NF",
    "dataset11 | HFpEF": "D11 HFpEF",
    "dataset11 | NF": "D11 NF",
}
DISEASE_COHORT_ORDER = list(DISEASE_COHORT_LABELS.values())
DISEASE_COHORT_COLORS = {
    "D01 DCM": NAVY,
    "D01 HCM": TEAL,
    "D01 NF": "#8C969E",
    "D08 DoxCM": GOLD,
    "D08 NF": "#A4ADB4",
    "D11 HFpEF": PLUM,
    "D11 NF": "#BEC4C9",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#59636C",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def require_inputs() -> None:
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input CSV(s):\n" + "\n".join(missing))


def read_inputs() -> dict[str, pd.DataFrame]:
    require_inputs()
    return {name: pd.read_csv(path) for name, path in INPUTS.items()}


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.tick_params(length=3, color="#65717B")


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.18, y: float = 1.10) -> None:
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


def build_retention_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    aging = data["aging_retention"].copy()
    aging_records = pd.DataFrame(
        {
            "scope": "Aging",
            "minimum_LECs": aging["minimum_LECs"].astype(int),
            "n_donors_total": 48,
            "n_donors_retained": aging["n_donors_retained"].astype(int),
            "donor_retention_percent": aging["percent_donors_retained"].astype(float),
            "n_LEC_cells_total": 1119,
            "n_LEC_cells_retained": aging["n_LEC_cells_retained"].astype(int),
            "LEC_retention_percent": aging["percent_LEC_cells_retained"].astype(float),
        }
    )

    donor_long = data["disease_donor_state_composition"]
    disease_donors = donor_long[
        ["donor_key", "dataset_folder", "condition", "cohort", "n_final_LECs"]
    ].drop_duplicates()
    if disease_donors["donor_key"].duplicated().any():
        raise ValueError("Disease donor metadata are not unique by donor_key.")

    disease_records: list[dict[str, object]] = []
    for scope, mask in [
        ("D01/D08 primary", disease_donors["dataset_folder"].isin(["dataset01", "dataset08"])),
        ("D11 HFpEF/NF", disease_donors["dataset_folder"].eq("dataset11")),
    ]:
        subset = disease_donors.loc[mask]
        for threshold in [5, 10, 15, 20, 25, 30]:
            retained = subset["n_final_LECs"].ge(threshold)
            disease_records.append(
                {
                    "scope": scope,
                    "minimum_LECs": threshold,
                    "n_donors_total": len(subset),
                    "n_donors_retained": int(retained.sum()),
                    "donor_retention_percent": 100.0 * retained.mean(),
                    "n_LEC_cells_total": int(subset["n_final_LECs"].sum()),
                    "n_LEC_cells_retained": int(subset.loc[retained, "n_final_LECs"].sum()),
                    "LEC_retention_percent": 100.0
                    * subset.loc[retained, "n_final_LECs"].sum()
                    / subset["n_final_LECs"].sum(),
                }
            )
    disease_retention_source = data["disease_retention"].copy()
    disease_source = disease_retention_source.loc[
        disease_retention_source["scope_type"].eq("cohort")
        & disease_retention_source["minimum_LECs"].isin([5, 10, 15, 20, 25, 30])
    ].copy()
    disease_source["scope"] = disease_source["scope"].map(DISEASE_COHORT_LABELS)
    if disease_source["scope"].isna().any():
        raise ValueError("Unexpected cohort in the disease threshold-retention table.")
    disease_source_records = disease_source.assign(
        donor_retention_percent=100.0 * disease_source["donor_fraction_retained"],
        LEC_retention_percent=100.0 * disease_source["cell_fraction_retained"],
    )[
        [
            "scope",
            "minimum_LECs",
            "n_donors_total",
            "n_donors_retained",
            "donor_retention_percent",
            "n_LEC_cells_total",
            "n_LEC_cells_retained",
            "LEC_retention_percent",
        ]
    ]
    disease_overall = disease_retention_source.loc[
        disease_retention_source["scope_type"].eq("overall")
        & disease_retention_source["scope"].eq("all")
        & disease_retention_source["minimum_LECs"].isin([5, 10, 15, 20, 25, 30])
    ].copy()
    disease_overall_records = disease_overall.assign(
        scope="Disease overall",
        donor_retention_percent=100.0 * disease_overall["donor_fraction_retained"],
        LEC_retention_percent=100.0 * disease_overall["cell_fraction_retained"],
    )[
        [
            "scope",
            "minimum_LECs",
            "n_donors_total",
            "n_donors_retained",
            "donor_retention_percent",
            "n_LEC_cells_total",
            "n_LEC_cells_retained",
            "LEC_retention_percent",
        ]
    ]
    result = pd.concat(
        [
            aging_records,
            pd.DataFrame(disease_records),
            disease_source_records,
            disease_overall_records,
        ],
        ignore_index=True,
    )
    return result


def build_aging_group_retention(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mapping = data["aging_filter_mapping"].copy()
    thresholds = [3, 5, 8, 10, 12, 15, 20, 25]
    records: list[dict[str, object]] = []
    for group in ["Q1", "Q2", "Q3", "Q4"]:
        subset = mapping.loc[mapping["age_group"].eq(group)].copy()
        total_donors = int(subset["donor_key"].nunique())
        total_cells = int(subset["n_total_LEC_cells"].sum())
        for threshold in thresholds:
            retained = subset.loc[subset["n_total_LEC_cells"].ge(threshold)]
            retained_donors = int(retained["donor_key"].nunique())
            retained_cells = int(retained["n_total_LEC_cells"].sum())
            records.append(
                {
                    "scope": group,
                    "minimum_LECs": threshold,
                    "n_donors_total": total_donors,
                    "n_donors_retained": retained_donors,
                    "donor_retention_percent": 100.0 * retained_donors / total_donors,
                    "n_LEC_cells_total": total_cells,
                    "n_LEC_cells_retained": retained_cells,
                    "LEC_retention_percent": 100.0 * retained_cells / total_cells,
                }
            )
    return pd.DataFrame.from_records(records)


def build_resolution_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    aging = data["aging_filter_mapping"][
        ["donor_key", "n_total_LEC_cells"]
    ].drop_duplicates()
    aging = aging.rename(columns={"n_total_LEC_cells": "n_LECs"})
    aging.insert(0, "scope", "Aging")

    disease = data["disease_donor_state_composition"][
        ["donor_key", "cohort", "n_final_LECs"]
    ].drop_duplicates()
    if disease["donor_key"].duplicated().any():
        raise ValueError("Disease donor depth is not unique by donor_key.")
    disease["scope"] = disease["cohort"].map(DISEASE_COHORT_LABELS)
    if disease["scope"].isna().any():
        raise ValueError("Unexpected cohort while building donor-level resolution data.")
    disease = disease.rename(columns={"n_final_LECs": "n_LECs"})[
        ["scope", "donor_key", "n_LECs"]
    ]

    result = pd.concat([aging, disease], ignore_index=True)
    result["n_LECs"] = result["n_LECs"].astype(int)
    result["one_cell_abundance_step_pp"] = 100.0 / result["n_LECs"]
    return result


def build_aging_effect_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mapping = data["aging_filter_mapping"][
        ["donor_key", "age_group", "n_total_LEC_cells"]
    ].copy()
    counts = data["aging_state_counts"][["donor_key", "LEC_state", "n_cells"]].copy()

    all_states = sorted(
        counts["LEC_state"].astype(str).unique(), key=lambda value: int(value.replace("state", ""))
    )
    grid = pd.MultiIndex.from_product(
        [mapping["donor_key"], all_states], names=["donor_key", "LEC_state"]
    ).to_frame(index=False)
    grid = grid.merge(mapping, on="donor_key", validate="many_to_one")
    grid = grid.merge(counts, on=["donor_key", "LEC_state"], how="left", validate="one_to_one")
    grid["n_cells"] = grid["n_cells"].fillna(0).astype(int)
    grid["donor_state_percentage"] = (
        100.0 * grid["n_cells"] / grid["n_total_LEC_cells"]
    )
    grid["merged_age_group"] = grid["age_group"].replace({"Q2": "Q2+Q3", "Q3": "Q2+Q3"})

    records: list[dict[str, object]] = []
    for threshold in [5, 10, 15, 20, 25]:
        retained = grid.loc[grid["n_total_LEC_cells"].ge(threshold)]
        donor_support = retained[["donor_key", "merged_age_group"]].drop_duplicates()
        support = donor_support["merged_age_group"].value_counts().to_dict()
        for comparison_group, contrast in [("Q2+Q3", "Q2+Q3 − Q1"), ("Q4", "Q4 − Q1")]:
            for state in AGING_STATES:
                comparison_values = retained.loc[
                    retained["merged_age_group"].eq(comparison_group)
                    & retained["LEC_state"].eq(state),
                    "donor_state_percentage",
                ]
                reference_values = retained.loc[
                    retained["merged_age_group"].eq("Q1")
                    & retained["LEC_state"].eq(state),
                    "donor_state_percentage",
                ]
                records.append(
                    {
                        "analysis": "Aging",
                        "minimum_LECs": threshold,
                        "contrast": contrast,
                        "comparison_group": comparison_group,
                        "reference_group": "Q1",
                        "LEC_state": state,
                        "n_comparison_donors": int(support.get(comparison_group, 0)),
                        "n_reference_donors": int(support.get("Q1", 0)),
                        "comparison_mean_percentage": float(comparison_values.mean()),
                        "reference_mean_percentage": float(reference_values.mean()),
                        "mean_difference_percentage_points": float(
                            comparison_values.mean() - reference_values.mean()
                        ),
                    }
                )
    return pd.DataFrame(records)


def build_disease_effect_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    donor_state = data["disease_donor_state_composition"].copy()
    records: list[dict[str, object]] = []
    for specification in DISEASE_SERIES:
        for threshold in specification["thresholds"]:
            subset = donor_state.loc[
                donor_state["dataset_folder"].eq(specification["dataset_folder"])
                & donor_state["LEC_state"].eq(specification["LEC_state"])
                & donor_state["n_final_LECs"].ge(threshold)
            ]
            disease_values = 100.0 * subset.loc[
                subset["condition"].eq(specification["disease_condition"]),
                "state_fraction_within_donor",
            ]
            reference_values = 100.0 * subset.loc[
                subset["condition"].eq(specification["reference_condition"]),
                "state_fraction_within_donor",
            ]
            if disease_values.empty or reference_values.empty:
                continue
            records.append(
                {
                    "analysis": "Disease",
                    "series": specification["series"],
                    "dataset_folder": specification["dataset_folder"],
                    "disease_condition": specification["disease_condition"],
                    "reference_condition": specification["reference_condition"],
                    "LEC_state": specification["LEC_state"],
                    "minimum_LECs": threshold,
                    "n_disease_donors": len(disease_values),
                    "n_reference_donors": len(reference_values),
                    "disease_mean_percentage": float(disease_values.mean()),
                    "reference_mean_percentage": float(reference_values.mean()),
                    "mean_difference_percentage_points": float(
                        disease_values.mean() - reference_values.mean()
                    ),
                }
            )
    return pd.DataFrame(records)


def validate_tables(
    data: dict[str, pd.DataFrame],
    retention: pd.DataFrame,
    aging_group_retention: pd.DataFrame,
    resolution: pd.DataFrame,
    aging_effects: pd.DataFrame,
    disease_effects: pd.DataFrame,
) -> None:
    def retention_tuple(scope: str, threshold: int) -> tuple[int, int, int, int]:
        row = retention.loc[
            retention["scope"].eq(scope) & retention["minimum_LECs"].eq(threshold)
        ].iloc[0]
        return (
            int(row["n_donors_retained"]),
            int(row["n_donors_total"]),
            int(row["n_LEC_cells_retained"]),
            int(row["n_LEC_cells_total"]),
        )

    expected_retention = {
        ("Aging", 25): (13, 48, 818, 1119),
        ("D01/D08 primary", 25): (50, 61, 5708, 5868),
        ("D11 HFpEF/NF", 10): (13, 34, 349, 423),
        ("D11 HFpEF/NF", 15): (9, 34, 301, 423),
        ("Disease overall", 25): (55, 95, 5940, 6291),
    }
    for key, expected in expected_retention.items():
        observed = retention_tuple(*key)
        if observed != expected:
            raise ValueError(f"Unexpected retention for {key}: {observed}; expected {expected}.")

    aging_group_n25 = aging_group_retention.loc[
        aging_group_retention["minimum_LECs"].eq(25)
    ]
    if int(aging_group_n25["n_donors_retained"].sum()) != 13:
        raise ValueError("Aging group-specific retention does not reproduce 13 donors at N=25.")

    disease_retention_source = data["disease_retention"]
    for scope, datasets in [
        ("D01/D08 primary", ["dataset01", "dataset08"]),
        ("D11 HFpEF/NF", ["dataset11"]),
    ]:
        source_rows = disease_retention_source.loc[
            disease_retention_source["scope_type"].eq("dataset")
            & disease_retention_source["scope"].isin(datasets)
        ]
        for threshold in retention.loc[retention["scope"].eq(scope), "minimum_LECs"]:
            source_at_threshold = source_rows.loc[source_rows["minimum_LECs"].eq(threshold)]
            expected = (
                int(source_at_threshold["n_donors_retained"].sum()),
                int(source_at_threshold["n_donors_total"].sum()),
                int(source_at_threshold["n_LEC_cells_retained"].sum()),
                int(source_at_threshold["n_LEC_cells_total"].sum()),
            )
            observed = retention_tuple(scope, int(threshold))
            if observed != expected:
                raise ValueError(
                    f"Derived retention does not match the existing disease threshold table "
                    f"for {scope}, N={threshold}: {observed} versus {expected}."
                )

    expected_resolution_scopes = {"Aging", *DISEASE_COHORT_ORDER}
    if set(resolution["scope"]) != expected_resolution_scopes:
        raise ValueError("Panel B donor-resolution data do not include every required cohort.")
    if not np.allclose(
        resolution["one_cell_abundance_step_pp"], 100.0 / resolution["n_LECs"]
    ):
        raise ValueError("Panel B donor-level abundance resolution was calculated incorrectly.")

    aging_main = data["aging_main_figure_source"]
    aging_main = aging_main.loc[
        aging_main["record_type"].eq("age_group_contrast_effect")
        & aging_main["display_in_panel"].astype(str).str.lower().eq("true")
    ]
    for _, row in aging_main.iterrows():
        contrast = str(row["contrast"]).replace(" - ", " − ")
        observed = aging_effects.loc[
            aging_effects["minimum_LECs"].eq(25)
            & aging_effects["contrast"].eq(contrast)
            & aging_effects["LEC_state"].eq(row["LEC_state"]),
            "mean_difference_percentage_points",
        ]
        if len(observed) != 1 or not np.isclose(
            float(observed.iloc[0]), float(row["mean_difference_percentage_points"]), atol=1e-6
        ):
            raise ValueError(
                f"Aging sensitivity does not reproduce main Figure 2 for {contrast}, {row['LEC_state']}."
            )

    disease_main = data["disease_main_figure_source"]
    disease_checks = [
        ("D01 DCM–NF · State 0", 25, "within_dataset_disease_minus_NF_effect", "dataset01: DCM vs NF", "state0"),
        ("D01 HCM–NF · State 0", 25, "within_dataset_disease_minus_NF_effect", "dataset01: HCM vs NF", "state0"),
        ("D08 DoxCM–NF · State 3", 25, "within_dataset_disease_minus_NF_effect", "dataset08: DoxCM vs NF", "state3"),
        ("D11 HFpEF–NF · State 0", 10, "HFpEF_vs_NF_threshold_effect", "dataset11: HFpEF vs NF", "state0"),
        ("D11 HFpEF–NF · State 0", 15, "HFpEF_vs_NF_threshold_effect", "dataset11: HFpEF vs NF", "state0"),
    ]
    for series, threshold, record_type, comparison, state in disease_checks:
        expected = disease_main.loc[
            disease_main["record_type"].eq(record_type)
            & disease_main["comparison"].eq(comparison)
            & disease_main["LEC_state"].eq(state)
        ]
        if record_type == "HFpEF_vs_NF_threshold_effect":
            expected = expected.loc[expected["minimum_LECs_per_donor"].eq(threshold)]
        observed = disease_effects.loc[
            disease_effects["series"].eq(series)
            & disease_effects["minimum_LECs"].eq(threshold),
            "mean_difference_percentage_points",
        ]
        if len(expected) != 1 or len(observed) != 1 or not np.isclose(
            float(observed.iloc[0]),
            float(expected.iloc[0]["mean_difference_percentage_points"]),
            atol=1e-6,
        ):
            raise ValueError(f"Disease sensitivity validation failed for {series}, N={threshold}.")


def draw_panel_a(fig: plt.Figure, spec, retention: pd.DataFrame) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 2, wspace=0.13)
    ax_aging = fig.add_subplot(grid[0, 0])
    aging = retention.loc[retention["scope"].eq("Aging")].sort_values("minimum_LECs")
    ax_aging.plot(
        aging["minimum_LECs"], aging["donor_retention_percent"],
        color=NAVY, marker="o", linewidth=2, markersize=4.5, label="Donors",
    )
    ax_aging.plot(
        aging["minimum_LECs"], aging["LEC_retention_percent"],
        color=TEAL, marker="s", linestyle="--", linewidth=1.8,
        markersize=4.2, label="LEC cells",
    )
    ax_aging.axvline(25, color=DARK, linewidth=1.1, linestyle=(0, (3, 2)))
    ax_aging.text(
        0.03, 0.06, "N=25: 13/48 donors; 73.1% of LECs",
        transform=ax_aging.transAxes, ha="left", va="bottom", fontsize=8, color=SLATE,
    )
    ax_aging.set_title("Aging", loc="left", pad=8)
    ax_aging.set_xlabel("Minimum LECs per donor")
    ax_aging.set_ylabel("Retention (%)")
    ax_aging.set_ylim(0, 104)
    ax_aging.set_xticks(aging["minimum_LECs"].astype(int))
    ax_aging.legend(frameon=False, loc="upper right")
    clean_axis(ax_aging, "y")
    panel_letter(ax_aging, "A", x=-0.25, y=1.13)

    ax_disease = fig.add_subplot(grid[0, 1])
    for scope in DISEASE_COHORT_ORDER:
        subset = retention.loc[retention["scope"].eq(scope)].sort_values("minimum_LECs")
        ax_disease.plot(
            subset["minimum_LECs"],
            subset["donor_retention_percent"],
            color=DISEASE_COHORT_COLORS[scope],
            marker="o",
            linestyle="--" if scope.endswith("NF") else "-",
            linewidth=1.8,
            markersize=3.8,
            label=scope,
        )
    ax_disease.axvspan(10, 15, color=PALE_TEAL, zorder=0)
    ax_disease.axvline(25, color=DARK, linewidth=1.1, linestyle=(0, (3, 2)))
    ax_disease.set_title("Disease cohorts: donor retention", loc="left", pad=8)
    ax_disease.set_xlabel("Minimum LECs per donor")
    ax_disease.set_ylim(0, 104)
    ax_disease.set_xticks([5, 10, 15, 20, 25, 30])
    ax_disease.set_yticklabels([])
    clean_axis(ax_disease, "y")
    ax_disease.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        fontsize=7.2,
        columnspacing=0.9,
        handlelength=2.0,
    )
    return [ax_aging, ax_disease]


def draw_panel_b(fig: plt.Figure, spec, resolution: pd.DataFrame) -> plt.Axes:
    ax = fig.add_subplot(spec)
    scopes = ["Aging", *DISEASE_COHORT_ORDER]
    labels = ["Aging", "D01\nDCM", "D01\nHCM", "D01\nNF", "D08\nDoxCM", "D08\nNF", "D11\nHFpEF", "D11\nNF"]
    colors = [NAVY, *[DISEASE_COHORT_COLORS[scope] for scope in DISEASE_COHORT_ORDER]]
    values = [
        resolution.loc[resolution["scope"].eq(scope), "one_cell_abundance_step_pp"].to_numpy(float)
        for scope in scopes
    ]
    boxplot = ax.boxplot(
        values,
        positions=np.arange(len(scopes)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": DARK, "linewidth": 1.4},
        whiskerprops={"color": SLATE, "linewidth": 1.0},
        capprops={"color": SLATE, "linewidth": 1.0},
    )
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)
    rng = np.random.default_rng(20260903)
    for position, (group_values, color) in enumerate(zip(values, colors)):
        jitter = rng.uniform(-0.17, 0.17, size=len(group_values))
        ax.scatter(
            position + jitter,
            group_values,
            s=18,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    for n_cells, color in [(10, TEAL), (15, TEAL), (25, DARK)]:
        value = 100.0 / n_cells
        ax.axhline(value, color=color, linewidth=0.9, linestyle=(0, (3, 2)), zorder=0)
        ax.text(
            len(scopes) - 0.42,
            value * 1.04,
            f"N={n_cells}: {value:.1f} pp",
            ha="right",
            va="bottom",
            fontsize=7.2,
            color=color,
        )
    ax.set_yscale("log")
    ax.set_ylim(0.2, 125)
    ax.set_yticks([0.25, 0.5, 1, 2, 4, 10, 20, 50, 100])
    ax.set_yticklabels(["0.25", "0.5", "1", "2", "4", "10", "20", "50", "100"])
    ax.set_xticks(np.arange(len(scopes)), labels)
    ax.set_ylabel("One-cell abundance step, 100/N (pp; log scale)")
    ax.set_xlabel("Observed donor group")
    ax.set_title(
        "Donor-level abundance resolution from observed LEC counts\n"
        "Lower values indicate finer percentage resolution",
        loc="left",
        pad=8,
    )
    clean_axis(ax, "y")
    panel_letter(ax, "B", x=-0.10, y=1.14)
    return ax


def draw_retention_matrix(
    fig: plt.Figure,
    ax: plt.Axes,
    table: pd.DataFrame,
    row_order: list[str],
    thresholds: list[int],
    title: str,
    outline_specs: list[tuple[float, float, float, float, str]],
) -> None:
    cell_matrix = (
        table.pivot(index="scope", columns="minimum_LECs", values="LEC_retention_percent")
        .reindex(index=row_order, columns=thresholds)
        .to_numpy(dtype=float)
    )
    donor_retained = (
        table.pivot(index="scope", columns="minimum_LECs", values="n_donors_retained")
        .reindex(index=row_order, columns=thresholds)
        .to_numpy(dtype=int)
    )
    donor_total = (
        table.pivot(index="scope", columns="minimum_LECs", values="n_donors_total")
        .reindex(index=row_order, columns=thresholds)
        .to_numpy(dtype=int)
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "lec_retention", ["#F3F6F7", "#BFD7D8", TEAL]
    )
    matrix_image = ax.imshow(cell_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    for row_idx in range(len(row_order)):
        for col_idx in range(len(thresholds)):
            value = cell_matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{donor_retained[row_idx, col_idx]}/{donor_total[row_idx, col_idx]}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value >= 68 else DARK,
            )
    ax.set_xticks(np.arange(len(thresholds)), thresholds)
    ax.set_yticks(np.arange(len(row_order)), row_order)
    ax.set_xlabel("Minimum LECs per donor")
    ax.set_title(f"{title}\nCell text: retained / total donors", loc="left", pad=8)
    for x_position in np.arange(-0.5, len(thresholds), 1):
        ax.axvline(x_position, color="white", linewidth=1)
    for y_position in np.arange(-0.5, len(row_order), 1):
        ax.axhline(y_position, color="white", linewidth=1)
    for x, y, width, height, color in outline_specs:
        ax.add_patch(
            Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=1.6)
        )
    colorbar = fig.colorbar(matrix_image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("LEC cells retained (%)", fontsize=8.3)
    colorbar.ax.tick_params(labelsize=7.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_panel_c(
    fig: plt.Figure, spec, aging_group_retention: pd.DataFrame
) -> plt.Axes:
    ax = fig.add_subplot(spec)
    draw_retention_matrix(
        fig,
        ax,
        aging_group_retention,
        ["Q1", "Q2", "Q3", "Q4"],
        [3, 5, 8, 10, 12, 15, 20, 25],
        "Aging threshold analysis by age group",
        [(6.5, -0.45, 1.0, 3.9, DARK)],
    )
    ax.set_xlabel("Minimum LECs per donor\nBlack outline: primary N=25")
    panel_letter(ax, "C", x=-0.19, y=1.13)
    return ax


def draw_panel_d(fig: plt.Figure, spec, retention: pd.DataFrame) -> plt.Axes:
    ax = fig.add_subplot(spec)
    cohort_rows = retention.loc[retention["scope"].isin(DISEASE_COHORT_ORDER)].copy()
    draw_retention_matrix(
        fig,
        ax,
        cohort_rows,
        DISEASE_COHORT_ORDER,
        [5, 10, 15, 20, 25, 30],
        "Disease threshold analysis: every studied cohort",
        [(3.5, -0.45, 1.0, 5.0, DARK), (0.5, 4.55, 2.0, 1.9, TEAL)],
    )
    ax.set_xlabel(
        "Minimum LECs per donor\n"
        "Black: D01/D08 primary N=25; teal: D11 N=10–15 sensitivity tiers"
    )
    panel_letter(ax, "D", x=-0.21, y=1.13)
    return ax


def draw_panel_e(
    fig: plt.Figure,
    spec,
    aging_effects: pd.DataFrame,
    disease_effects: pd.DataFrame,
) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 3, wspace=0.16)
    axes: list[plt.Axes] = []
    for idx, contrast in enumerate(["Q2+Q3 − Q1", "Q4 − Q1"]):
        ax = fig.add_subplot(grid[0, idx])
        subset = aging_effects.loc[aging_effects["contrast"].eq(contrast)]
        for state in AGING_STATES:
            state_rows = subset.loc[subset["LEC_state"].eq(state)].sort_values("minimum_LECs")
            ax.plot(
                state_rows["minimum_LECs"],
                state_rows["mean_difference_percentage_points"],
                color=AGING_STATE_COLORS[state],
                marker=AGING_STATE_MARKERS[state],
                linewidth=2,
                markersize=4.5,
                label=AGING_STATE_LABELS[state],
            )
        ax.axhline(0, color="#AEB6BD", linewidth=0.9)
        ax.axvline(25, color=DARK, linewidth=1.1, linestyle=(0, (3, 2)))
        ax.set_xlim(4, 27)
        ax.set_xticks([5, 10, 15, 20, 25])
        ax.set_ylim(-30, 20)
        ax.set_xlabel("Minimum LECs per donor")
        ax.set_title(f"Aging: {contrast}", loc="left", pad=8)
        clean_axis(ax, "both")
        if idx == 0:
            ax.set_ylabel("Mean abundance difference (pp)")
            ax.legend(frameon=False, loc="upper right", fontsize=7.4)
            panel_letter(ax, "E", x=-0.27, y=1.13)
        else:
            ax.set_yticklabels([])
        axes.append(ax)

    ax = fig.add_subplot(grid[0, 2])
    for specification in DISEASE_SERIES:
        subset = disease_effects.loc[
            disease_effects["series"].eq(specification["series"])
        ].sort_values("minimum_LECs")
        linestyle = "--" if specification["dataset_folder"] == "dataset11" else "-"
        ax.plot(
            subset["minimum_LECs"],
            subset["mean_difference_percentage_points"],
            color=specification["color"],
            marker=specification["marker"],
            linestyle=linestyle,
            linewidth=2,
            markersize=4.6,
            label=specification["series"],
        )
    ax.axhline(0, color="#AEB6BD", linewidth=0.9)
    ax.axvspan(10, 15, color=PALE_TEAL, zorder=0)
    ax.axvline(25, color=DARK, linewidth=1.1, linestyle=(0, (3, 2)))
    ax.text(
        12.5,
        44,
        "D11 sensitivity",
        ha="center",
        va="top",
        fontsize=7.5,
        color=TEAL,
    )
    ax.set_xlim(4, 31)
    ax.set_ylim(0, 45)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Minimum LECs per donor")
    ax.set_ylabel("Disease − matched NF difference (pp)")
    ax.set_title("Disease: representative state effects", loc="left", pad=8)
    clean_axis(ax, "both")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        fontsize=7.1,
        handlelength=2.0,
    )
    axes.append(ax)
    return axes


def save_figure(fig: plt.Figure, basename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "jpg"):
        kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.16,
            "facecolor": "white",
        }
        if extension in {"png", "jpg"}:
            kwargs["dpi"] = 300
        if extension == "jpg":
            kwargs["pil_kwargs"] = {"quality": 95, "subsampling": 0}
        fig.savefig(OUT_DIR / f"{basename}.{extension}", **kwargs)
    plt.close(fig)


def make_individual_panels(
    retention: pd.DataFrame,
    aging_group_retention: pd.DataFrame,
    resolution: pd.DataFrame,
    aging_effects: pd.DataFrame,
    disease_effects: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(11.8, 4.9), constrained_layout=True)
    draw_panel_a(fig, GridSpec(1, 1, figure=fig)[0], retention)
    save_figure(fig, "FigS02A_donor_retention_across_thresholds")

    fig = plt.figure(figsize=(10.6, 4.8), constrained_layout=True)
    draw_panel_b(fig, GridSpec(1, 1, figure=fig)[0], resolution)
    save_figure(fig, "FigS02B_donor_abundance_resolution")

    fig = plt.figure(figsize=(8.4, 4.5), constrained_layout=True)
    draw_panel_c(fig, GridSpec(1, 1, figure=fig)[0], aging_group_retention)
    save_figure(fig, "FigS02C_aging_threshold_analysis")

    fig = plt.figure(figsize=(9.0, 5.1), constrained_layout=True)
    draw_panel_d(fig, GridSpec(1, 1, figure=fig)[0], retention)
    save_figure(fig, "FigS02D_disease_threshold_analysis")

    fig = plt.figure(figsize=(15.4, 5.2), constrained_layout=True)
    draw_panel_e(
        fig,
        GridSpec(1, 1, figure=fig)[0],
        aging_effects,
        disease_effects,
    )
    save_figure(fig, "FigS02E_state_abundance_sensitivity")


def make_combined_figure(
    retention: pd.DataFrame,
    aging_group_retention: pd.DataFrame,
    resolution: pd.DataFrame,
    aging_effects: pd.DataFrame,
    disease_effects: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(16.5, 18.0), constrained_layout=True)
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.78, 0.82, 0.94, 1.0],
        width_ratios=[0.88, 1.12],
        hspace=0.18,
        wspace=0.12,
    )
    draw_panel_a(fig, grid[0, :], retention)
    draw_panel_b(fig, grid[1, :], resolution)
    draw_panel_c(fig, grid[2, 0], aging_group_retention)
    draw_panel_d(fig, grid[2, 1], retention)
    draw_panel_e(fig, grid[3, :], aging_effects, disease_effects)
    fig.suptitle(
        "Figure S2 | LEC-number threshold rationale and sensitivity",
        fontsize=18,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    save_figure(fig, "FigS02_LEC_number_threshold_sensitivity")


def write_traceability(
    retention: pd.DataFrame,
    aging_group_retention: pd.DataFrame,
    resolution: pd.DataFrame,
    aging_effects: pd.DataFrame,
    disease_effects: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    panel_a_source = retention.loc[
        retention["scope"].isin(["Aging", *DISEASE_COHORT_ORDER])
    ]
    panel_d_source = retention.loc[retention["scope"].isin(DISEASE_COHORT_ORDER)]
    for panel, record_type, frame in [
        ("A", "donor_retention_across_thresholds", panel_a_source),
        ("B", "observed_donor_abundance_resolution", resolution),
        ("C", "aging_group_threshold_analysis", aging_group_retention),
        ("D", "disease_cohort_threshold_analysis", panel_d_source),
        ("E", "aging_state_effect_sensitivity", aging_effects),
        ("E", "disease_state_effect_sensitivity", disease_effects),
    ]:
        output = frame.copy()
        output.insert(0, "panel", panel)
        output.insert(1, "record_type", record_type)
        frames.append(output)
    pd.concat(frames, ignore_index=True, sort=False).to_csv(
        OUT_DIR / "FigS02_source_data.csv", index=False
    )

    panel_map = {
        "aging_retention": "A",
        "aging_filter_mapping": "B; C; E",
        "aging_state_counts": "E",
        "aging_main_figure_source": "E validation",
        "disease_retention": "A; D",
        "disease_donor_state_composition": "B; E",
        "disease_main_figure_source": "E validation",
    }
    purpose_map = {
        "aging_retention": "Aging donor and LEC-cell retention across candidate thresholds",
        "aging_filter_mapping": "Aging donor depth for observed abundance resolution and age-group analyses",
        "aging_state_counts": "Aging donor-state counts used to recalculate representative effects",
        "aging_main_figure_source": "Validate that N=25 sensitivity values reproduce main Figure 2",
        "disease_retention": "Overall and every-cohort disease retention across candidate thresholds",
        "disease_donor_state_composition": "Every disease cohort's observed donor depth and representative state-abundance effects",
        "disease_main_figure_source": "Validate N=25 and D11 N=10/N=15 effects against main Figure 5",
    }
    manifest = pd.DataFrame(
        [
            {
                "input_id": name,
                "panels": panel_map[name],
                "input_csv": str(path.relative_to(ROOT)),
                "purpose": purpose_map[name],
            }
            for name, path in INPUTS.items()
        ]
    )
    manifest.to_csv(OUT_DIR / "FigS02_input_manifest.csv", index=False)


def main() -> None:
    configure_style()
    data = read_inputs()
    retention = build_retention_table(data)
    aging_group_retention = build_aging_group_retention(data)
    resolution = build_resolution_table(data)
    aging_effects = build_aging_effect_table(data)
    disease_effects = build_disease_effect_table(data)
    validate_tables(
        data,
        retention,
        aging_group_retention,
        resolution,
        aging_effects,
        disease_effects,
    )
    write_traceability(
        retention,
        aging_group_retention,
        resolution,
        aging_effects,
        disease_effects,
    )
    make_individual_panels(
        retention,
        aging_group_retention,
        resolution,
        aging_effects,
        disease_effects,
    )
    make_combined_figure(
        retention,
        aging_group_retention,
        resolution,
        aging_effects,
        disease_effects,
    )
    print(f"Supplementary Figure S2 written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
