#!/usr/bin/env python3
"""Create aging Figure 3A: state-specific regulon activity."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


SCRIPT_VERSION = "Fig03A_aging_state_specific_regulons_v1_2026_08_31"

SCENIC_FULL_CSV = Path(
    "new_result2_aging/11_SCENIC_analysis/dataset13/tables/"
    "11_SCENIC_state_regulon_full_summary.csv"
)
SCENIC_ZSCORE_CSV = Path(
    "new_result2_aging/11_SCENIC_analysis/dataset13/tables/"
    "11_donor_balanced_mean_AUCell_zscore.csv"
)
DONOR_STATE_CSV = Path(
    "new_result2_aging/08_7_min25_LEC_filtered_age_analysis/dataset13/"
    "three_groups_Q2_Q3_merged/donor_level/"
    "08_7_Q2_Q3_merged_donor_state_counts_percentages.csv.gz"
)
AGE_SUMMARY_CSV = Path(
    "new_result2_aging/08_7_min25_LEC_filtered_age_analysis/dataset13/"
    "three_groups_Q2_Q3_merged/donor_level/"
    "08_7_Q2_Q3_merged_state_group_summary_mean_sem.csv"
)
AGE_TEST_CSV = Path(
    "new_result2_aging/08_7_min25_LEC_filtered_age_analysis/dataset13/"
    "three_groups_Q2_Q3_merged/donor_level/"
    "08_7_Q2_Q3_merged_state_omnibus_tests.csv"
)
OUTPUT_DIR = Path("new_result2/aging/Fig03A_aging_state_specific_regulons")


# This ordered selection reproduces the examples in the supplied Figure 3 text.
# "NR2F2/DDIT3" is represented by both named regulons.
SELECTED_REGULONS: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("state0", ["NFE2L3", "WT1", "MAFB", "RUNX1"]),
        ("state1", ["EGR3", "SOX18", "HEYL", "NFATC2"]),
        ("state2", ["TBX15", "HMGA2", "FOXO1", "SMAD9"]),
        ("state4", ["CEBPB", "IRF4", "PRDM6", "MYC"]),
        ("state5", ["FOXO1", "EPAS1", "SMAD9", "NR2F2", "DDIT3"]),
    ]
)
STATE_ORDER = list(SELECTED_REGULONS)
ROBUSTNESS_FEATURES = [
    ("state0", "NFE2L3"),
    ("state0", "WT1"),
    ("state0", "MAFB"),
    ("state0", "RUNX1"),
    ("state4", "CEBPB"),
    ("state5", "FOXO1"),
    ("state5", "EPAS1"),
]
AGE_STATES = ["state0", "state4", "state5"]
AGE_GROUPS = ["Q1", "Q2+Q3", "Q4"]
EXPECTED_GROUP_DONORS = {"Q1": 4, "Q2+Q3": 4, "Q4": 5}


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


def load_inputs(root: Path) -> dict[str, object]:
    paths = {
        "scenic_full": root / SCENIC_FULL_CSV,
        "scenic_zscore": root / SCENIC_ZSCORE_CSV,
        "donor_state": root / DONOR_STATE_CSV,
        "age_summary": root / AGE_SUMMARY_CSV,
        "age_test": root / AGE_TEST_CSV,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    full = pd.read_csv(paths["scenic_full"], low_memory=False)
    zscore = pd.read_csv(paths["scenic_zscore"], index_col=0)
    donor = pd.read_csv(paths["donor_state"], low_memory=False)
    age_summary = pd.read_csv(paths["age_summary"], low_memory=False)
    age_test = pd.read_csv(paths["age_test"], low_memory=False)

    require_columns(
        full,
        [
            "LEC_state",
            "regulon",
            "TF",
            "donor_balanced_mean_AUCell_zscore",
            "donor_bootstrap_RSS_median",
            "donor_bootstrap_RSS_q025",
            "donor_bootstrap_RSS_q975",
            "donor_bootstrap_RSS_top_frequency",
            "consensus_rank_within_state",
            "robust_candidate_flag",
        ],
        paths["scenic_full"],
    )
    require_columns(
        donor,
        [
            "donor_key",
            "LEC_state",
            "age_group",
            "age_numeric",
            "n_total_LEC_cells",
            "cell_state_percentage",
        ],
        paths["donor_state"],
    )
    require_columns(
        age_summary,
        ["LEC_state", "age_group", "n_donors", "mean_percentage", "sem_percentage"],
        paths["age_summary"],
    )
    require_columns(
        age_test,
        ["LEC_state", "kruskal_p", "kruskal_fdr_bh", "epsilon_squared"],
        paths["age_test"],
    )
    # Validate the exact template regulons and executed robustness flags.
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
    if not as_bool(selected["robust_candidate_flag"]).all():
        failed = selected.loc[~as_bool(selected["robust_candidate_flag"]), ["LEC_state", "TF"]]
        raise ValueError(f"Requested examples are not all robust candidates:\n{failed}")

    expected_regulons = {f"{gene}(+)" for genes in SELECTED_REGULONS.values() for gene in genes}
    missing_zscore = sorted(expected_regulons - set(zscore.columns.astype(str)))
    missing_states = sorted(set(STATE_ORDER) - set(zscore.index.astype(str)))
    if missing_zscore or missing_states:
        raise ValueError(
            f"Donor-balanced z-score matrix is missing regulons={missing_zscore}, "
            f"states={missing_states}"
        )

    donor = donor.loc[donor["LEC_state"].isin(AGE_STATES)].copy()
    donor["cell_state_percentage"] = pd.to_numeric(
        donor["cell_state_percentage"], errors="raise"
    )
    if donor.duplicated(["donor_key", "LEC_state"]).any():
        raise ValueError("Donor/state rows are not unique in the age-composition table.")
    if donor["donor_key"].nunique() != 13:
        raise ValueError(
            f"Expected 13 donors after the >=25-LEC filter; found "
            f"{donor['donor_key'].nunique()}"
        )
    per_group = donor[["donor_key", "age_group"]].drop_duplicates()["age_group"].value_counts()
    observed_group_donors = {
        group: int(per_group.get(group, 0)) for group in AGE_GROUPS
    }
    if observed_group_donors != EXPECTED_GROUP_DONORS:
        raise ValueError(
            f"Unexpected post-filter merged group support: {observed_group_donors}"
        )
    if (pd.to_numeric(donor["n_total_LEC_cells"], errors="raise") < 25).any():
        raise ValueError("Panel C contains a donor with fewer than 25 total LECs.")

    age_summary = age_summary.loc[
        age_summary["LEC_state"].isin(AGE_STATES)
        & age_summary["age_group"].isin(AGE_GROUPS)
    ].copy()
    if len(age_summary) != len(AGE_STATES) * len(AGE_GROUPS):
        raise ValueError("Age-summary table does not contain every requested state/quartile.")
    age_test = age_test.loc[age_test["LEC_state"].isin(AGE_STATES)].copy()
    if len(age_test) != len(AGE_STATES):
        raise ValueError("Age-test table does not contain every requested state.")

    return {
        "paths": paths,
        "full": full,
        "zscore": zscore,
        "selected": selected,
        "donor": donor,
        "age_summary": age_summary,
        "age_test": age_test,
    }


def heatmap_table(inputs: dict[str, object]) -> pd.DataFrame:
    zscore: pd.DataFrame = inputs["zscore"]  # type: ignore[assignment]
    selected: pd.DataFrame = inputs["selected"]  # type: ignore[assignment]
    metrics = selected.set_index(["LEC_state", "TF"])
    records: list[dict[str, object]] = []
    row_order = 0
    for target_state, genes in SELECTED_REGULONS.items():
        for gene in genes:
            selected_metric = metrics.loc[(target_state, gene)]
            for heatmap_state in STATE_ORDER:
                records.append(
                    {
                        "panel": "A",
                        "record_type": "state_specific_regulon_heatmap",
                        "target_state": target_state,
                        "target_state_display": display_state(target_state),
                        "TF": gene,
                        "regulon": f"{gene}(+)",
                        "row_order": row_order,
                        "heatmap_state": heatmap_state,
                        "heatmap_state_display": display_state(heatmap_state),
                        "donor_balanced_mean_AUCell_zscore": float(
                            zscore.loc[heatmap_state, f"{gene}(+)"]
                        ),
                        "target_donor_bootstrap_RSS_median": float(
                            selected_metric["donor_bootstrap_RSS_median"]
                        ),
                        "target_donor_bootstrap_RSS_top_frequency": float(
                            selected_metric["donor_bootstrap_RSS_top_frequency"]
                        ),
                        "target_consensus_rank": int(
                            selected_metric["consensus_rank_within_state"]
                        ),
                        "robust_candidate_flag": bool(
                            as_bool(pd.Series([selected_metric["robust_candidate_flag"]])).iloc[0]
                        ),
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
    output.insert(1, "record_type", "donor_bootstrap_regulon_robustness")
    output["state_display"] = output["LEC_state"].map(display_state)
    output["source_file"] = SCENIC_FULL_CSV.as_posix()
    return output


def age_table(inputs: dict[str, object]) -> pd.DataFrame:
    donor: pd.DataFrame = inputs["donor"]  # type: ignore[assignment]
    summary: pd.DataFrame = inputs["age_summary"]  # type: ignore[assignment]
    tests: pd.DataFrame = inputs["age_test"]  # type: ignore[assignment]
    output = donor.merge(
        summary[
            ["LEC_state", "age_group", "n_donors", "mean_percentage", "sem_percentage"]
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
    output.insert(0, "panel", "C")
    output.insert(1, "record_type", "donor_normalized_state_abundance")
    output["state_display"] = output["LEC_state"].map(display_state)
    output["source_file"] = (
        DONOR_STATE_CSV.as_posix()
        + ";"
        + AGE_SUMMARY_CSV.as_posix()
        + ";"
        + AGE_TEST_CSV.as_posix()
    )
    return output


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
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
    )


def draw_panel_a(
    axis: plt.Axes,
    figure: plt.Figure,
    heat: pd.DataFrame,
) -> None:
    rows = heat[["row_order", "target_state", "TF"]].drop_duplicates().sort_values("row_order")
    matrix = (
        heat.pivot(index="row_order", columns="heatmap_state", values="donor_balanced_mean_AUCell_zscore")
        .reindex(index=rows["row_order"], columns=STATE_ORDER)
        .to_numpy(float)
    )
    limit = max(2.5, float(np.ceil(np.nanmax(np.abs(matrix)) * 2) / 2))
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(range(len(STATE_ORDER)), [display_state(state) for state in STATE_ORDER])
    axis.set_yticks(range(len(rows)), rows["TF"].tolist(), fontstyle="italic")
    axis.tick_params(length=0)
    axis.set_xlabel("LEC state")
    axis.set_title(
        "State-specific regulon activity",
        loc="left",
        fontweight="bold",
        pad=10,
    )

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
            fontsize=7.5,
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
    colorbar.set_label("Donor-balanced mean AUCell (state-wise z score)", fontsize=7.3)
    colorbar.ax.tick_params(labelsize=6.8, length=2)


def draw_panel_b(axis: plt.Axes, robust: pd.DataFrame) -> None:
    robust = robust.reset_index(drop=True)
    y_positions = np.arange(len(robust))[::-1]
    high_max = float(pd.to_numeric(robust["donor_bootstrap_RSS_q975"]).max())
    x_limit = high_max + 0.075
    for y, row in zip(y_positions, robust.itertuples(index=False), strict=True):
        median = float(row.donor_bootstrap_RSS_median)
        low = float(row.donor_bootstrap_RSS_q025)
        high = float(row.donor_bootstrap_RSS_q975)
        frequency = float(row.donor_bootstrap_RSS_top_frequency)
        axis.plot([low, high], [y, y], color="#4B555D", linewidth=1.5, zorder=2)
        axis.plot([low, low], [y - 0.08, y + 0.08], color="#4B555D", linewidth=1)
        axis.plot([high, high], [y - 0.08, y + 0.08], color="#4B555D", linewidth=1)
        axis.scatter(
            median,
            y,
            s=38 + 105 * frequency,
            color="#4B555D",
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
            fontsize=7,
            color="#333333",
        )

    labels = [
        f"{display_state(row.LEC_state)}  ·  {row.TF}"
        for row in robust.itertuples(index=False)
    ]
    axis.set_yticks(y_positions, labels)
    for label, row in zip(axis.get_yticklabels(), robust.itertuples(index=False), strict=True):
        label.set_color("#30363B")
        label.set_fontweight("bold" if row.TF in {"CEBPB", "NFE2L3", "FOXO1", "EPAS1"} else "normal")
    axis.set_xlim(0, x_limit)
    axis.set_ylim(-0.7, len(robust) - 0.3)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel("Donor-bootstrap RSS median (95% interval)")
    axis.set_title(
        "Regulon specificity and donor robustness",
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
        fontsize=6.8,
        color="#555555",
    )
    axis.text(
        0,
        -0.14,
        "Dot size and percentages: how often a regulon ranks among the top eight\n"
        "within its state across donor-bootstrap repetitions.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color="#555555",
    )


def draw_age_axis(
    axis: plt.Axes,
    state: str,
    age: pd.DataFrame,
    rng: np.random.Generator,
    show_panel_label: bool,
) -> None:
    subset = age.loc[age["LEC_state"].eq(state)].copy()
    summaries = (
        subset[
            ["age_group", "mean_percentage", "sem_percentage", "kruskal_p", "kruskal_fdr_bh"]
        ]
        .drop_duplicates("age_group")
        .set_index("age_group")
        .reindex(AGE_GROUPS)
    )
    means = summaries["mean_percentage"].to_numpy(float)
    sem = summaries["sem_percentage"].fillna(0).to_numpy(float)
    positions = np.arange(len(AGE_GROUPS), dtype=float)

    for position, group in zip(positions, AGE_GROUPS, strict=True):
        values = subset.loc[subset["age_group"].eq(group), "cell_state_percentage"].to_numpy(float)
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        axis.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=18,
            color="#AAB1B6",
            alpha=0.58,
            edgecolor="none",
            zorder=2,
        )
    axis.plot(positions, means, color="#3F484F", linewidth=1.8, zorder=3)
    axis.errorbar(
        positions,
        means,
        yerr=sem,
        fmt="o",
        markersize=5.7,
        color="#3F484F",
        markeredgecolor="#222222",
        markeredgewidth=0.65,
        ecolor="#333333",
        elinewidth=0.9,
        capsize=2.3,
        zorder=4,
    )

    delta = means[-1] - means[0]
    max_value = float(subset["cell_state_percentage"].max())
    upper = min(105.0, max(20.0, np.ceil(max(max_value, np.max(means + sem)) * 1.12 / 10) * 10))
    axis.set_ylim(0, upper)
    axis.set_xlim(-0.42, 3.42)
    axis.set_xticks(positions, ["Q1\nyounger", "Q2+Q3", "Q4\nolder"])
    axis.grid(axis="y", color="#E1E1E1", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylabel("% of donor LECs")
    axis.set_title(
        f"{display_state(state)}   Q1→Q4: {delta:+.1f} pp",
        color="#252A2E",
        fontweight="bold",
        pad=6,
    )
    p_value = float(summaries["kruskal_p"].dropna().iloc[0])
    fdr = float(summaries["kruskal_fdr_bh"].dropna().iloc[0])
    axis.text(
        0.98,
        0.96,
        f"Kruskal–Wallis P={p_value:.3g}; FDR={fdr:.3g}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )
    if show_panel_label:
        panel_label(axis, "C", x=-0.16, y=1.12)
        axis.text(
            0,
            1.15,
            "Age-dependent abundance of regulon-defined states",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.3,
            fontweight="bold",
        )
        axis.text(
            0,
            1.09,
            "≥25 LECs per donor; Q2 and Q3 merged (n=4, 4, and 5 donors)",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            color="#555555",
        )


def write_outputs(root: Path, heat: pd.DataFrame) -> Path:
    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    source = heat.copy()
    source.insert(0, "script_version", SCRIPT_VERSION)
    source.to_csv(output / "Fig03A_source_data.csv", index=False)
    manifest = pd.DataFrame(
        [
            {
                "input_file": SCENIC_FULL_CSV.as_posix(),
                "used_fields": "LEC_state;TF;regulon;donor_bootstrap_RSS_*;consensus_rank_within_state;robust_candidate_flag",
                "figure_use": "A exact-regulon validation",
            },
            {
                "input_file": SCENIC_ZSCORE_CSV.as_posix(),
                "used_fields": "state rows; exact regulon columns",
                "figure_use": "A donor-balanced regulon activity heatmap",
            },
        ]
    )
    manifest.insert(0, "script_version", SCRIPT_VERSION)
    manifest.to_csv(output / "Fig03A_input_manifest.csv", index=False)

    configure_style()
    figure, axis = plt.subplots(figsize=(8.3, 7.1), facecolor="white")
    figure.subplots_adjust(left=0.25, right=0.965, bottom=0.15, top=0.82)
    draw_panel_a(axis, figure, heat)
    figure.text(0.025, 0.965, "A", ha="left", va="top", fontsize=18, fontweight="bold")
    figure.text(
        0.085,
        0.965,
        "Regulons defining aging-relevant LEC states",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.965,
        0.025,
        "Donor-balanced Step 11 SCENIC analysis of 1,119 LECs.",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#666666",
    )
    figure.savefig(output / "Fig03A_aging_state_specific_regulons.pdf", bbox_inches="tight")
    figure.savefig(output / "Fig03A_aging_state_specific_regulons.png", dpi=600, bbox_inches="tight")
    figure.savefig(
        output / "Fig03A_aging_state_specific_regulons.jpg",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)
    return output


def main() -> int:
    root = project_root()
    inputs = load_inputs(root)
    heat = heatmap_table(inputs)
    output = write_outputs(root, heat)
    print(f"Figure 3A written to: {output}")
    print(f"Exact template regulon selections: {sum(map(len, SELECTED_REGULONS.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
