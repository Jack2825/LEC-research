#!/usr/bin/env python3
"""Create disease Figure 5A: state-specific regulon activity."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


SCRIPT_VERSION = "Fig05A_disease_state_specific_regulons_v1_2026_08_31"

SCENIC_FULL_CSV = Path(
    "new_result2_disease/11_SCENIC_analysis/tables/"
    "11_SCENIC_state_regulon_full_summary.csv"
)
SCENIC_ZSCORE_CSV = Path(
    "new_result2_disease/11_SCENIC_analysis/tables/"
    "11_dataset_donor_balanced_mean_AUCell_zscore.csv"
)
CELL_FILTER_CSV = Path(
    "new_result2_disease/08_3_min25_LEC_filtered_analysis/tables/"
    "08_3_final_LEC_cell_filter_mapping.csv.gz"
)
HFPEF_MIN10_SUMMARY_CSV = Path(
    "new_result2_disease/12_NicheNet_donor_pseudobulk/"
    "dataset11_HFpEF_vs_NF_min10/12_run_summary.csv"
)
HFPEF_MIN15_SUMMARY_CSV = Path(
    "new_result2_disease/12_NicheNet_donor_pseudobulk/"
    "dataset11_HFpEF_vs_NF_min15/12_run_summary.csv"
)
COMPARISON_CSV = Path(
    "new_result2_disease/08_4_donor_level_LEC_state_disease_analysis/tables/"
    "08_4_primary_within_dataset_state_comparisons.csv"
)
OUTPUT_DIR = Path("new_result2/disease/Fig05A_disease_state_specific_regulons")


# Exact examples from the disease SCENIC specification. "SMAD1/5" is represented by
# both named regulons, SMAD1 and SMAD5.
SELECTED_REGULONS: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("state0", ["ETS1", "ERG", "PROX1", "SMAD1", "SMAD5"]),
        ("state1", ["JUN", "JUNB", "JUND", "FOS", "KLF2"]),
        ("state3", ["PPARG", "FOXO1", "NFATC1", "ERG", "PROX1"]),
        ("state4", ["STAT1", "STAT2", "STAT6", "CUX1"]),
        ("state5", ["EPAS1", "TEAD4", "IRF3"]),
    ]
)
STATE_ORDER = list(SELECTED_REGULONS)
ALL_STATES = [f"state{value}" for value in range(9)]
BOOTSTRAP_ITERATIONS = 10_000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 20260826
HFPEF_EXPECTED_DONORS = {
    10: {"NF": 6, "HFpEF": 7},
    15: {"NF": 4, "HFpEF": 5},
}
HFPEF_EXPECTED_CELLS = {10: 349, 15: 301}
HFPEF_STYLES = {
    10: {"color": "#287C78", "label": "≥10 · exploratory"},
    15: {"color": "#687279", "label": "≥15 · sensitivity"},
}

# These are the exact template-listed regulons that pass robust_candidate_flag
# in the current Step 11 disease results. Only these are promoted into panel B.
ROBUSTNESS_FEATURES = [
    ("state1", "JUN"),
    ("state1", "JUNB"),
    ("state1", "JUND"),
    ("state1", "KLF2"),
    ("state3", "PPARG"),
    ("state5", "EPAS1"),
]

COMPARISON_ORDER = [
    "dataset01: DCM vs NF",
    "dataset01: HCM vs NF",
    "dataset08: DoxCM vs NF",
]
COMPARISON_TITLES = {
    "dataset01: DCM vs NF": "D01 · DCM vs NF",
    "dataset01: HCM vs NF": "D01 · HCM vs NF",
    "dataset08: DoxCM vs NF": "D08 · DoxCM vs NF",
}
C_STATE_LABELS = {
    "state0": "State 0\nDisease-expanded",
    "state1": "State 1\nNF-dominant",
    "state3": "State 3\nDisease-expanded pattern",
    "state4": "State 4\nCardiomyopathy-linked",
    "state5": "State 5\nSelected-disease enrichment",
}

TEXT = "#22272B"
SUBTEXT = "#5B6369"
NEUTRAL = "#4B555D"
LIGHT = "#B8BEC2"
GRID = "#DDE1E3"


def project_root() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[2]


def display_state(state: str) -> str:
    return f"State {int(state.removeprefix('state'))}"


def as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
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


def validate_hfpef_threshold_summary(
    summary: pd.DataFrame, source: Path, threshold: int
) -> None:
    require_columns(
        summary,
        [
            "receiver_min_cells",
            "n_reference_donors",
            "n_disease_donors",
            "n_receiver_cells",
            "validation_status",
            "status",
        ],
        source,
    )
    if len(summary) != 1:
        raise ValueError(f"Expected one threshold-summary row in {source}.")
    row = summary.iloc[0]
    expected = HFPEF_EXPECTED_DONORS[threshold]
    checks = {
        "receiver_min_cells": int(row["receiver_min_cells"]) == threshold,
        "n_reference_donors": int(row["n_reference_donors"]) == expected["NF"],
        "n_disease_donors": int(row["n_disease_donors"]) == expected["HFpEF"],
        "n_receiver_cells": int(row["n_receiver_cells"]) == HFPEF_EXPECTED_CELLS[threshold],
        "validation_status": str(row["validation_status"]).lower() == "passed",
        "status": str(row["status"]).lower() == "completed",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"HFpEF threshold provenance failed at ≥{threshold}: {failed}")


def hfpef_donor_state_table(cells: pd.DataFrame, threshold: int) -> pd.DataFrame:
    subset = cells.loc[
        cells["dataset_folder"].eq("dataset11")
        & cells["condition"].isin(["NF", "HFpEF"])
        & (cells["donor_n_final_LECs"] >= threshold)
    ].copy()
    donor_meta = subset[
        ["donor_key", "condition", "donor_n_final_LECs"]
    ].drop_duplicates("donor_key")
    observed = donor_meta.groupby("condition")["donor_key"].nunique().to_dict()
    if observed != HFPEF_EXPECTED_DONORS[threshold]:
        raise ValueError(f"Unexpected dataset11 donor counts at ≥{threshold}: {observed}")
    if len(subset) != HFPEF_EXPECTED_CELLS[threshold]:
        raise ValueError(
            f"Unexpected dataset11 LEC count at ≥{threshold}: "
            f"{len(subset)} != {HFPEF_EXPECTED_CELLS[threshold]}"
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


def hfpef_effect_table_from_donors(
    donors: pd.DataFrame, threshold: int
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + threshold)
    records: list[dict[str, object]] = []
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    for state in ALL_STATES:
        state_rows = donors.loc[donors["LEC_state"].eq(state)]
        disease = state_rows.loc[
            state_rows["condition"].eq("HFpEF"), "state_percentage_of_donor_LECs"
        ].to_numpy(float)
        reference = state_rows.loc[
            state_rows["condition"].eq("NF"), "state_percentage_of_donor_LECs"
        ].to_numpy(float)
        disease_index = rng.integers(
            0, len(disease), size=(BOOTSTRAP_ITERATIONS, len(disease))
        )
        reference_index = rng.integers(
            0, len(reference), size=(BOOTSTRAP_ITERATIONS, len(reference))
        )
        boot = disease[disease_index].mean(axis=1) - reference[reference_index].mean(axis=1)
        ci_low, ci_high = np.quantile(boot, [alpha, 1 - alpha])
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
                "mean_difference_percentage_points": float(
                    disease.mean() - reference.mean()
                ),
                "mean_difference_ci_low": float(ci_low),
                "mean_difference_ci_high": float(ci_high),
                "mann_whitney_p": p_value,
                "minimum_LECs_per_donor": threshold,
                "analysis_tier": (
                    "exploratory_min10" if threshold == 10 else "sensitivity_min15"
                ),
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "confidence_level": CONFIDENCE_LEVEL,
            }
        )
    output = pd.DataFrame.from_records(records)
    output["mann_whitney_FDR_within_dataset11_threshold"] = bh_adjust(
        output["mann_whitney_p"]
    )
    return output


def load_inputs(root: Path) -> dict[str, object]:
    paths = {
        "scenic_full": root / SCENIC_FULL_CSV,
        "scenic_zscore": root / SCENIC_ZSCORE_CSV,
        "cell_filter": root / CELL_FILTER_CSV,
        "hfpef_min10_summary": root / HFPEF_MIN10_SUMMARY_CSV,
        "hfpef_min15_summary": root / HFPEF_MIN15_SUMMARY_CSV,
        "comparison": root / COMPARISON_CSV,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    full = pd.read_csv(paths["scenic_full"], low_memory=False)
    zscore = pd.read_csv(paths["scenic_zscore"], index_col=0)
    cells = pd.read_csv(paths["cell_filter"], low_memory=False)
    min10_summary = pd.read_csv(paths["hfpef_min10_summary"], low_memory=False)
    min15_summary = pd.read_csv(paths["hfpef_min15_summary"], low_memory=False)
    comparison = pd.read_csv(paths["comparison"], low_memory=False)

    require_columns(
        full,
        [
            "LEC_state",
            "regulon",
            "TF",
            "dataset_donor_balanced_mean_AUCell_zscore",
            "dataset_donor_bootstrap_RSS_median",
            "dataset_donor_bootstrap_RSS_q025",
            "dataset_donor_bootstrap_RSS_q975",
            "dataset_donor_bootstrap_RSS_top_frequency",
            "consensus_rank_within_state",
            "robust_candidate_flag",
        ],
        paths["scenic_full"],
    )
    require_columns(
        cells,
        [
            "donor_key",
            "dataset_folder",
            "condition",
            "donor_n_final_LECs",
            "LEC_state",
        ],
        paths["cell_filter"],
    )
    require_columns(
        comparison,
        [
            "comparison",
            "LEC_state",
            "mean_difference_percentage_points",
            "mean_difference_ci_low",
            "mean_difference_ci_high",
            "mann_whitney_p",
            "mann_whitney_FDR_within_comparison",
            "n_reference_donors",
            "n_disease_donors",
        ],
        paths["comparison"],
    )
    validate_hfpef_threshold_summary(
        min10_summary, paths["hfpef_min10_summary"], 10
    )
    validate_hfpef_threshold_summary(
        min15_summary, paths["hfpef_min15_summary"], 15
    )

    selected_rows: list[pd.Series] = []
    for state, genes in SELECTED_REGULONS.items():
        for gene in genes:
            matched = full.loc[full["LEC_state"].eq(state) & full["TF"].eq(gene)]
            if len(matched) != 1:
                raise ValueError(
                    f"Expected one current SCENIC row for {state} {gene}; found {len(matched)}"
                )
            selected_rows.append(matched.iloc[0])
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["robust_candidate_flag"] = as_bool(selected["robust_candidate_flag"])

    expected_robust = set(ROBUSTNESS_FEATURES)
    observed_robust = set(
        selected.loc[selected["robust_candidate_flag"], ["LEC_state", "TF"]]
        .itertuples(index=False, name=None)
    )
    if observed_robust != expected_robust:
        raise ValueError(
            "The current robust template-regulon set changed: "
            f"expected={sorted(expected_robust)}, observed={sorted(observed_robust)}"
        )

    expected_regulons = {
        f"{gene}(+)" for genes in SELECTED_REGULONS.values() for gene in genes
    }
    missing_zscore = sorted(expected_regulons - set(zscore.columns.astype(str)))
    missing_states = sorted(set(STATE_ORDER) - set(zscore.index.astype(str)))
    if missing_zscore or missing_states:
        raise ValueError(
            f"Dataset/donor z-score matrix is missing regulons={missing_zscore}, "
            f"states={missing_states}"
        )

    cells["donor_n_final_LECs"] = pd.to_numeric(
        cells["donor_n_final_LECs"], errors="raise"
    )
    primary_all = comparison.loc[
        comparison["comparison"].isin(COMPARISON_ORDER)
        & comparison["LEC_state"].isin(ALL_STATES)
    ].copy()
    if len(primary_all) != len(COMPARISON_ORDER) * len(ALL_STATES):
        raise ValueError("The primary three-comparison by nine-state matrix is incomplete.")

    hfpef_donors10 = hfpef_donor_state_table(cells, 10)
    hfpef_donors15 = hfpef_donor_state_table(cells, 15)
    hfpef_effects10 = hfpef_effect_table_from_donors(hfpef_donors10, 10)
    hfpef_effects15 = hfpef_effect_table_from_donors(hfpef_donors15, 15)
    for threshold_effects, name in [
        (hfpef_effects10, "min10"),
        (hfpef_effects15, "min15"),
    ]:
        combined_p = pd.concat(
            [primary_all["mann_whitney_p"], threshold_effects["mann_whitney_p"]],
            ignore_index=True,
        )
        combined_q = bh_adjust(combined_p)
        primary_all[f"global_BH_with_HFpEF_{name}"] = combined_q.iloc[
            : len(primary_all)
        ].to_numpy()
        threshold_effects["global_BH_four_comparison_family"] = combined_q.iloc[
            len(primary_all) :
        ].to_numpy()

    effects = primary_all.loc[primary_all["LEC_state"].isin(STATE_ORDER)].copy()
    if len(effects) != len(COMPARISON_ORDER) * len(STATE_ORDER):
        raise ValueError("Disease-comparison table is missing a requested state/comparison row.")
    if effects.duplicated(["comparison", "LEC_state"]).any():
        raise ValueError("Disease-comparison state rows are not unique.")

    # Validate the explicit biological pattern in the disease SCENIC specification.
    effect_pivot = effects.pivot(
        index="LEC_state", columns="comparison", values="mean_difference_percentage_points"
    )
    if not (effect_pivot.loc["state1"] < 0).all():
        raise ValueError("Current effects no longer show State 1 as NF-dominant.")
    if not (effect_pivot.loc["state0"] > 0).all():
        raise ValueError("Current effects no longer show disease expansion of State 0.")
    if not (effect_pivot.loc["state3"] > 0).all():
        raise ValueError("Current effects no longer show the State 3 disease-expansion pattern.")
    for threshold, donor_table in [(10, hfpef_donors10), (15, hfpef_donors15)]:
        state4_total = int(
            donor_table.loc[donor_table["LEC_state"].eq("state4"), "n_state_LECs"].sum()
        )
        if state4_total != 0:
            raise ValueError(f"Dataset11 State 4 is no longer absent at ≥{threshold}.")

    return {
        "paths": paths,
        "full": full,
        "zscore": zscore,
        "selected": selected,
        "effects": effects,
        "hfpef_donors10": hfpef_donors10,
        "hfpef_donors15": hfpef_donors15,
        "hfpef_effects10": hfpef_effects10,
        "hfpef_effects15": hfpef_effects15,
    }


def heatmap_table(inputs: dict[str, object]) -> pd.DataFrame:
    zscore: pd.DataFrame = inputs["zscore"]  # type: ignore[assignment]
    selected: pd.DataFrame = inputs["selected"]  # type: ignore[assignment]
    metrics = selected.set_index(["LEC_state", "TF"])
    records: list[dict[str, object]] = []
    row_order = 0
    for target_state, genes in SELECTED_REGULONS.items():
        for gene in genes:
            metric = metrics.loc[(target_state, gene)]
            for heatmap_state in STATE_ORDER:
                records.append(
                    {
                        "panel": "A",
                        "record_type": "template_regulon_activity",
                        "target_state": target_state,
                        "target_state_display": display_state(target_state),
                        "TF": gene,
                        "regulon": f"{gene}(+)",
                        "row_order": row_order,
                        "heatmap_state": heatmap_state,
                        "heatmap_state_display": display_state(heatmap_state),
                        "dataset_donor_balanced_mean_AUCell_zscore": float(
                            zscore.loc[heatmap_state, f"{gene}(+)"]
                        ),
                        "target_dataset_donor_bootstrap_RSS_median": float(
                            metric["dataset_donor_bootstrap_RSS_median"]
                        ),
                        "target_dataset_donor_bootstrap_RSS_top_frequency": float(
                            metric["dataset_donor_bootstrap_RSS_top_frequency"]
                        ),
                        "target_consensus_rank": int(metric["consensus_rank_within_state"]),
                        "robust_candidate_flag": bool(metric["robust_candidate_flag"]),
                        "source_file": SCENIC_ZSCORE_CSV.as_posix()
                        + ";"
                        + SCENIC_FULL_CSV.as_posix(),
                    }
                )
            row_order += 1
    return pd.DataFrame.from_records(records)


def robustness_table(inputs: dict[str, object]) -> pd.DataFrame:
    full: pd.DataFrame = inputs["full"]  # type: ignore[assignment]
    rows: list[pd.Series] = []
    for state, gene in ROBUSTNESS_FEATURES:
        rows.append(full.loc[full["LEC_state"].eq(state) & full["TF"].eq(gene)].iloc[0])
    output = pd.DataFrame(rows).reset_index(drop=True)
    output.insert(0, "panel", "B")
    output.insert(1, "record_type", "dataset_donor_bootstrap_regulon_robustness")
    output["state_display"] = output["LEC_state"].map(display_state)
    output["source_file"] = SCENIC_FULL_CSV.as_posix()
    return output


def effect_table(inputs: dict[str, object]) -> pd.DataFrame:
    effects: pd.DataFrame = inputs["effects"]  # type: ignore[assignment]
    output = effects.copy()
    output.insert(0, "panel", "C")
    output.insert(1, "record_type", "within_dataset_disease_minus_NF_effect")
    output["state_display"] = output["LEC_state"].map(display_state)
    output["source_file"] = COMPARISON_CSV.as_posix()
    return output


def hfpef_table(inputs: dict[str, object]) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for threshold in [10, 15]:
        effects: pd.DataFrame = inputs[f"hfpef_effects{threshold}"]  # type: ignore[assignment]
        selected_effects = effects.loc[effects["LEC_state"].isin(STATE_ORDER)].copy()
        selected_effects.insert(0, "panel", "C")
        selected_effects.insert(1, "record_type", "HFpEF_vs_NF_threshold_effect")
        selected_effects["state_display"] = selected_effects["LEC_state"].map(display_state)
        selected_effects["source_file"] = CELL_FILTER_CSV.as_posix()
        tables.append(selected_effects)

        donors: pd.DataFrame = inputs[f"hfpef_donors{threshold}"]  # type: ignore[assignment]
        selected_donors = donors.loc[donors["LEC_state"].isin(STATE_ORDER)].copy()
        selected_donors.insert(0, "panel", "C")
        selected_donors.insert(1, "record_type", "HFpEF_NF_donor_state_abundance")
        selected_donors["state_display"] = selected_donors["LEC_state"].map(display_state)
        selected_donors["source_file"] = CELL_FILTER_CSV.as_posix()
        tables.append(selected_donors)
    return pd.concat(tables, ignore_index=True, sort=False)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.4,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def panel_label(axis: plt.Axes, label: str, x: float = -0.08, y: float = 1.035) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )


def draw_panel_a(axis: plt.Axes, figure: plt.Figure, heat: pd.DataFrame) -> None:
    rows = (
        heat[["row_order", "target_state", "TF", "robust_candidate_flag"]]
        .drop_duplicates()
        .sort_values("row_order")
    )
    matrix = (
        heat.pivot(
            index="row_order",
            columns="heatmap_state",
            values="dataset_donor_balanced_mean_AUCell_zscore",
        )
        .reindex(index=rows["row_order"], columns=STATE_ORDER)
        .to_numpy(float)
    )
    limit = max(2.5, float(np.ceil(np.nanmax(np.abs(matrix)) * 2) / 2))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(STATE_ORDER)), [display_state(state) for state in STATE_ORDER])
    axis.set_yticks(range(len(rows)), rows["TF"].tolist(), fontstyle="italic")
    for label, robust in zip(axis.get_yticklabels(), rows["robust_candidate_flag"], strict=True):
        label.set_fontweight("bold" if bool(robust) else "normal")
    axis.tick_params(length=0)
    axis.set_xlabel("LEC state")
    axis.set_title("State-specific regulon activity", loc="left", fontweight="bold", pad=10)

    start = 0
    for state, genes in SELECTED_REGULONS.items():
        n_genes = len(genes)
        midpoint = start + (n_genes - 1) / 2
        axis.text(
            -0.31,
            midpoint,
            display_state(state),
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color="#30363B",
            clip_on=False,
        )
        column = STATE_ORDER.index(state)
        for row_index in range(start, start + n_genes):
            axis.add_patch(
                Rectangle(
                    (column - 0.49, row_index - 0.49),
                    0.98,
                    0.98,
                    fill=False,
                    edgecolor="#30363B",
                    linewidth=1.05,
                )
            )
        start += n_genes
        if start < len(rows):
            axis.axhline(start - 0.5, color="white", linewidth=2.2)

    colorbar = figure.colorbar(image, ax=axis, orientation="horizontal", fraction=0.042, pad=0.075)
    colorbar.set_label(
        "Dataset/donor-balanced mean AUCell (state-wise z score)", fontsize=7.5
    )
    colorbar.ax.tick_params(labelsize=6.9, length=2)


def draw_panel_b(axis: plt.Axes, robust: pd.DataFrame) -> None:
    robust = robust.reset_index(drop=True)
    y_positions = np.arange(len(robust))[::-1]
    high_max = float(pd.to_numeric(robust["dataset_donor_bootstrap_RSS_q975"]).max())
    x_limit = high_max + 0.080
    for y, row in zip(y_positions, robust.itertuples(index=False), strict=True):
        median = float(row.dataset_donor_bootstrap_RSS_median)
        low = float(row.dataset_donor_bootstrap_RSS_q025)
        high = float(row.dataset_donor_bootstrap_RSS_q975)
        frequency = float(row.dataset_donor_bootstrap_RSS_top_frequency)
        axis.plot([low, high], [y, y], color=NEUTRAL, linewidth=1.5, zorder=2)
        axis.plot([low, low], [y - 0.08, y + 0.08], color=NEUTRAL, linewidth=1)
        axis.plot([high, high], [y - 0.08, y + 0.08], color=NEUTRAL, linewidth=1)
        axis.scatter(
            median,
            y,
            s=38 + 105 * frequency,
            color=NEUTRAL,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.text(
            high + 0.008,
            y,
            f"{100 * frequency:.0f}%",
            ha="left",
            va="center",
            fontsize=7.2,
            color="#333333",
        )

    labels = [
        f"{display_state(row.LEC_state)}  ·  {row.TF}"
        for row in robust.itertuples(index=False)
    ]
    axis.set_yticks(y_positions, labels)
    for label in axis.get_yticklabels():
        label.set_fontweight("bold")
        label.set_color("#30363B")
    axis.set_xlim(0, x_limit)
    axis.set_ylim(-0.7, len(robust) - 0.3)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel("Dataset/donor-bootstrap RSS median (95% interval)")
    axis.set_title(
        "Regulon specificity and cross-dataset robustness",
        loc="left",
        fontweight="bold",
        pad=10,
    )
    axis.text(
        1,
        1.005,
        "Within-state top-8 frequency (%)",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=SUBTEXT,
    )
    axis.text(
        0,
        -0.14,
        "Dot size and percentages: how often a regulon ranks among the top eight\n"
        "within its state across dataset/donor-bootstrap repetitions.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.9,
        color=SUBTEXT,
    )


def effect_xlim(effects: pd.DataFrame) -> tuple[float, float]:
    low = float(effects["mean_difference_ci_low"].min())
    high = float(effects["mean_difference_ci_high"].max())
    return min(-90.0, np.floor(low / 10) * 10), max(60.0, np.ceil(high / 10) * 10)


def draw_effect_axis(
    axis: plt.Axes,
    comparison_name: str,
    effects: pd.DataFrame,
    x_limits: tuple[float, float],
    show_labels: bool,
) -> None:
    subset = effects.loc[effects["comparison"].eq(comparison_name)].set_index("LEC_state").reindex(STATE_ORDER)
    y_positions = np.arange(len(STATE_ORDER))[::-1]
    axis.axvline(0, color="#60686E", linestyle="--", linewidth=0.9, zorder=1)
    for y, state in zip(y_positions, STATE_ORDER, strict=True):
        row = subset.loc[state]
        effect = float(row["mean_difference_percentage_points"])
        low = float(row["mean_difference_ci_low"])
        high = float(row["mean_difference_ci_high"])
        fdr = float(row["global_BH_with_HFpEF_min10"])
        significant = fdr <= 0.05
        axis.plot([low, high], [y, y], color=NEUTRAL, linewidth=1.4, zorder=2)
        axis.plot([low, low], [y - 0.09, y + 0.09], color=NEUTRAL, linewidth=0.9)
        axis.plot([high, high], [y - 0.09, y + 0.09], color=NEUTRAL, linewidth=0.9)
        axis.scatter(
            effect,
            y,
            s=48,
            facecolor=NEUTRAL if significant else "white",
            edgecolor=NEUTRAL,
            linewidth=1.1,
            zorder=3,
        )
        axis.text(
            effect,
            y + 0.24,
            f"{effect:+.1f}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color=TEXT,
        )

    axis.set_xlim(*x_limits)
    axis.set_ylim(-0.65, len(STATE_ORDER) - 0.35)
    axis.set_yticks(
        y_positions,
        [C_STATE_LABELS[state] for state in STATE_ORDER] if show_labels else [""] * len(STATE_ORDER),
    )
    axis.tick_params(axis="y", length=0, pad=7)
    if show_labels:
        for label in axis.get_yticklabels():
            label.set_fontweight("semibold")
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.set_xlabel("Disease − NF (pp)")
    axis.set_title(COMPARISON_TITLES[comparison_name], fontweight="bold", pad=8)


def draw_hfpef_axis(
    axis: plt.Axes,
    donors: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(20260824)
    summary = summary.set_index("LEC_state").reindex(STATE_ORDER)
    y_positions = np.arange(len(STATE_ORDER))[::-1]
    for y, state in zip(y_positions, STATE_ORDER, strict=True):
        values = donors.loc[
            donors["LEC_state"].eq(state), "state_percentage_of_donor_LECs"
        ].to_numpy(float)
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        axis.scatter(values, np.full(len(values), y) + jitter, s=20, color=LIGHT, alpha=0.72, edgecolor="none", zorder=2)
        mean = float(summary.loc[state, "mean_donor_percentage"])
        sem = float(summary.loc[state, "sem_donor_percentage"])
        axis.errorbar(
            mean,
            y,
            xerr=sem,
            fmt="D",
            markersize=5.5,
            color=NEUTRAL,
            markeredgecolor="#252A2E",
            markeredgewidth=0.7,
            ecolor=NEUTRAL,
            elinewidth=1.2,
            capsize=2.3,
            zorder=4,
        )
        axis.text(
            min(mean + sem + 2.0, 78.0),
            y,
            f"{mean:.1f}%",
            ha="left",
            va="center",
            fontsize=6.9,
            color=TEXT,
        )
    axis.set_xlim(-2, 82)
    axis.set_ylim(-0.65, len(STATE_ORDER) - 0.35)
    axis.tick_params(axis="y", left=False, labelleft=False)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.set_xlabel("% of donor LECs")
    axis.set_title("D11 · HFpEF (descriptive)", fontweight="bold", pad=8)
    axis.text(
        0.98,
        0.985,
        "n=5; no retained D11 NF",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color=SUBTEXT,
    )


def draw_hfpef_effect_axis(
    axis: plt.Axes,
    effects10: pd.DataFrame,
    effects15: pd.DataFrame,
) -> None:
    y_positions = np.arange(len(STATE_ORDER))[::-1]
    for threshold, effects, offset in [(10, effects10, 0.12), (15, effects15, -0.12)]:
        subset = effects.loc[effects["LEC_state"].isin(STATE_ORDER)].set_index(
            "LEC_state"
        )
        style = HFPEF_STYLES[threshold]
        for y, state in zip(y_positions, STATE_ORDER, strict=True):
            row = subset.loc[state]
            effect = float(row["mean_difference_percentage_points"])
            low = float(row["mean_difference_ci_low"])
            high = float(row["mean_difference_ci_high"])
            q_value = float(row["global_BH_four_comparison_family"])
            significant = np.isfinite(q_value) and q_value < 0.05
            line_style = "-" if threshold == 10 else "--"
            axis.plot(
                [low, high],
                [y + offset, y + offset],
                color=style["color"],
                linewidth=1.3,
                linestyle=line_style,
                zorder=2,
            )
            axis.plot(
                [low, low],
                [y + offset - 0.055, y + offset + 0.055],
                color=style["color"],
                linewidth=0.8,
            )
            axis.plot(
                [high, high],
                [y + offset - 0.055, y + offset + 0.055],
                color=style["color"],
                linewidth=0.8,
            )
            axis.scatter(
                effect,
                y + offset,
                marker="D",
                s=42 if threshold == 10 else 34,
                facecolor=style["color"] if significant else "white",
                edgecolor=style["color"],
                linewidth=1.1,
                zorder=3,
            )
    axis.axvline(0, color="#60686E", linestyle="--", linewidth=0.9, zorder=1)
    axis.set_xlim(-60, 65)
    axis.set_ylim(-0.65, len(STATE_ORDER) - 0.35)
    axis.tick_params(axis="y", left=False, labelleft=False)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.set_xlabel("HFpEF - NF (pp)")
    axis.set_title("D11 · HFpEF vs NF (exploratory)", fontweight="bold", pad=8)
    handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color=HFPEF_STYLES[threshold]["color"],
            markerfacecolor="white",
            linestyle="-" if threshold == 10 else "--",
            linewidth=1.2,
            label=HFPEF_STYLES[threshold]["label"],
        )
        for threshold in [10, 15]
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=6.6,
        handletextpad=0.4,
    )


def source_data_table(
    heat: pd.DataFrame,
    robust: pd.DataFrame,
    effects: pd.DataFrame,
    hfpef: pd.DataFrame,
) -> pd.DataFrame:
    return pd.concat([heat, robust, effects, hfpef], ignore_index=True, sort=False)


def write_outputs(root: Path, figure: plt.Figure, heat: pd.DataFrame) -> None:
    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    heat.to_csv(output / "Fig05A_source_data.csv", index=False)
    manifest = pd.DataFrame(
        [
            {
                "input_file": SCENIC_FULL_CSV.as_posix(),
                "used_fields": (
                    "LEC_state;TF;regulon;dataset_donor_bootstrap_RSS_*;"
                    "consensus_rank_within_state;robust_candidate_flag"
                ),
                "figure_use": "A exact-regulon validation and robust-label status",
            },
            {
                "input_file": SCENIC_ZSCORE_CSV.as_posix(),
                "used_fields": "state rows; exact template regulon columns",
                "figure_use": "A dataset/donor-balanced regulon activity heatmap",
            },
            {
                "input_file": "Figure 5A regulon specification",
                "used_fields": "key states; example regulons",
                "figure_use": "A exact regulator selection",
            },
        ]
    )
    manifest.insert(0, "script_version", SCRIPT_VERSION)
    manifest.to_csv(output / "Fig05A_input_manifest.csv", index=False)
    figure.savefig(output / "Fig05A_disease_state_specific_regulons.pdf", bbox_inches="tight")
    figure.savefig(
        output / "Fig05A_disease_state_specific_regulons.png",
        dpi=600,
        bbox_inches="tight",
    )
    figure.savefig(
        output / "Fig05A_disease_state_specific_regulons.jpg",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )


def main() -> int:
    root = project_root()
    inputs = load_inputs(root)
    heat = heatmap_table(inputs)
    configure_style()
    figure, axis = plt.subplots(figsize=(8.5, 7.2), facecolor="white")
    figure.subplots_adjust(left=0.25, right=0.965, bottom=0.15, top=0.82)
    draw_panel_a(axis, figure, heat)
    figure.text(0.025, 0.965, "A", ha="left", va="top", fontsize=18, fontweight="bold", color=TEXT)
    figure.text(
        0.085,
        0.965,
        "Regulons defining disease-associated LEC states",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.965,
        0.024,
        "Dataset/donor-balanced Step 11 SCENIC analysis of 6,291 disease-branch LECs.",
        ha="right",
        va="bottom",
        fontsize=7,
        color=SUBTEXT,
    )
    write_outputs(root, figure, heat)
    plt.close(figure)
    output = root / OUTPUT_DIR
    print(f"Figure 5A written to: {output}")
    print(f"Panel A template regulons: {sum(len(genes) for genes in SELECTED_REGULONS.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
