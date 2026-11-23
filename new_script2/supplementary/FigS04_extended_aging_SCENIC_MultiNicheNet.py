#!/usr/bin/env python3
"""Build Supplementary Figure S4: extended aging SCENIC and MultiNicheNet.

Seven panels provide the extended and sensitivity evidence underlying main
Figure 3 without reproducing its selected presentation panels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = (
    ROOT
    / "new_result2"
    / "supplementary"
    / "FigS04_extended_aging_SCENIC_MultiNicheNet"
)

SCENIC_DIR = ROOT / "new_result2_aging" / "11_SCENIC_analysis" / "dataset13" / "tables"
NICHE_DIR = (
    ROOT
    / "new_result2_aging"
    / "13_NicheNet_donor_pseudobulk"
    / "dataset13"
    / "tables"
)

INPUTS = {
    "scenic_full_summary": SCENIC_DIR / "11_SCENIC_state_regulon_full_summary.csv",
    "main_regulon_source": ROOT
    / "new_result2"
    / "aging"
    / "Fig03A_aging_state_specific_regulons"
    / "Fig03A_source_data.csv",
    "ligand_activities": NICHE_DIR / "13_Sydney_equivalent_ligand_activities.csv",
    "main_ligand_source": ROOT
    / "new_result2"
    / "aging"
    / "Fig03D_aging_ligand_activity"
    / "Fig03D_source_data.csv",
    "top50_sender_edges": NICHE_DIR / "13_MultiNicheNet_top50_sender_to_LEC_per_group.csv",
    "main_sender_source": ROOT
    / "new_result2"
    / "aging"
    / "Fig03E_aging_sender_ligand_network"
    / "Fig03E_source_data.csv",
    "ligand_receptor_links": NICHE_DIR / "13_Sydney_equivalent_ligand_receptor_links.csv",
    "ligand_target_links": NICHE_DIR / "13_Sydney_equivalent_ligand_target_links.csv",
    "main_lr_source": ROOT
    / "new_result2"
    / "aging"
    / "Fig03F_aging_ligand_receptor_interactions"
    / "Fig03F_source_data.csv",
    "de_primary": NICHE_DIR / "13_edgeR_primary_LEC_Q4_vs_Q1_all_genes.csv",
    "de_gse183852": NICHE_DIR / "13_edgeR_sensitivity_GSE183852_only_all_genes.csv",
    "de_project_sex": NICHE_DIR
    / "13_edgeR_sensitivity_Project_Sex_adjusted_all_genes.csv",
}

ALL_STATES = [f"state{index}" for index in range(11)]
DARK = "#202529"
SLATE = "#65727C"
GRID = "#DDE2E6"
LIGHT_GREY = "#D8DEE2"
NAVY = "#2C5F8A"
TEAL = "#2B8C87"
ORANGE = "#D07A36"
DOWN = "#3E78A5"
UP = "#C65F47"

SENDER_LABELS = {
    "EC5_art": "Arterial ECs",
    "EC7_endocardial": "Endocardial ECs",
    "LYVE1.IGF1.MP": "LYVE1+IGF1 macrophages",
    "SMC1_basic": "SMCs",
    "Mast": "Mast cells",
    "NC1_glial": "Glial cells",
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


def require_inputs() -> None:
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input CSV(s):\n" + "\n".join(missing))


def read_inputs() -> dict[str, pd.DataFrame]:
    require_inputs()
    return {name: pd.read_csv(path, low_memory=False) for name, path in INPUTS.items()}


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


def direction_display(direction: str) -> str:
    return {
        "Q4_downregulated": "Q4-downregulated LEC gene program",
        "Q4_upregulated": "Q4-upregulated LEC gene program",
    }[direction]


def direction_color(direction: str) -> str:
    return DOWN if direction == "Q4_downregulated" else UP


def build_extended_regulons(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = data["scenic_full_summary"].copy()
    full["state_number"] = full["LEC_state"].map(state_number)
    full["robust_candidate_flag"] = (
        full["robust_candidate_flag"].astype(str).str.lower().eq("true")
    )

    main = data["main_regulon_source"].copy()
    main_pairs = set(zip(main["target_state"], main["regulon"]))
    full["shown_in_main_target_pair"] = [
        (state, regulon) in main_pairs
        for state, regulon in zip(full["LEC_state"], full["regulon"])
    ]

    candidates = full.loc[
        full["robust_candidate_flag"] & ~full["shown_in_main_target_pair"]
    ].copy()
    candidates = candidates.sort_values(
        ["state_number", "consensus_rank_within_state", "regulon"]
    )
    selected = candidates.groupby("LEC_state", sort=False, as_index=False).head(2).copy()
    selected["selection_rule"] = (
        "Top two remaining robust candidates per state by consensus rank after "
        "excluding target-state/regulon pairs displayed in main Figure 3A"
    )
    selected["target_state"] = selected["LEC_state"]
    selected["target_state_display"] = selected["LEC_state"].map(state_label)
    selected["row_label"] = (
        selected["target_state_display"] + "  |  " + selected["TF"].astype(str)
    )
    selected = selected.sort_values(
        ["state_number", "consensus_rank_within_state", "regulon"]
    ).reset_index(drop=True)
    selected["row_order"] = np.arange(len(selected))

    heatmap = selected[
        [
            "target_state",
            "target_state_display",
            "state_number",
            "TF",
            "regulon",
            "row_label",
            "row_order",
            "consensus_rank_within_state",
            "donor_bootstrap_RSS_median",
            "donor_bootstrap_RSS_q025",
            "donor_bootstrap_RSS_q975",
            "donor_bootstrap_RSS_top_frequency",
            "equal_cell_RSS_top_frequency",
            "selection_rule",
        ]
    ].merge(
        full[
            [
                "LEC_state",
                "regulon",
                "donor_balanced_mean_AUCell_zscore",
            ]
        ].rename(columns={"LEC_state": "heatmap_state"}),
        on="regulon",
        how="left",
        validate="one_to_many",
    )
    heatmap["heatmap_state_number"] = heatmap["heatmap_state"].map(state_number)
    heatmap["heatmap_state_display"] = heatmap["heatmap_state"].map(state_label)
    heatmap = heatmap.sort_values(["row_order", "heatmap_state_number"]).reset_index(drop=True)

    robustness = selected.copy()
    robustness["target_state_color"] = robustness["state_color"]

    all_pairs = full.copy()
    selected_pairs = set(zip(selected["target_state"], selected["regulon"]))
    all_pairs["selected_extended_regulon"] = [
        (state, regulon) in selected_pairs
        for state, regulon in zip(all_pairs["LEC_state"], all_pairs["regulon"])
    ]
    all_pairs["selection_rule"] = (
        "All state-regulon pairs provide context; outlined points are robust candidates; "
        "filled points are the extended regulons selected for panels A and B"
    )
    return heatmap, robustness, all_pairs


def build_extended_ligands(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = data["ligand_activities"].copy()
    frame = frame.dropna(subset=["test_ligand", "aupr_corrected", "direction"])
    frame = frame.loc[
        frame["direction"].isin(["Q4_downregulated", "Q4_upregulated"])
    ].copy()
    frame = frame.sort_values(
        ["direction", "aupr_corrected"], ascending=[True, False]
    )
    frame["rank_in_complete_table"] = (
        frame.groupby("direction")["aupr_corrected"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    main_ligands = set(data["main_ligand_source"]["ligand"].dropna().astype(str))
    frame["shown_in_main_figure_3"] = frame["test_ligand"].isin(main_ligands)
    frame = frame.loc[~frame["shown_in_main_figure_3"]].copy()
    frame = frame.groupby("direction", sort=False, as_index=False).head(8).copy()
    frame["extended_rank"] = frame.groupby("direction").cumcount() + 1
    frame["direction_display"] = frame["direction"].map(direction_display)
    frame["selection_rule"] = (
        "Top eight corrected-AUPR ligand activities per direction after excluding all "
        "ligands displayed in main Figure 3D"
    )
    return frame.reset_index(drop=True)


def build_sender_contributions(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    edges = data["top50_sender_edges"].copy()
    main_labels = (
        data["main_sender_source"]
        .dropna(subset=["sender", "sender_label"])
        .drop_duplicates("sender")
        .set_index("sender")["sender_label"]
        .to_dict()
    )
    labels = {**SENDER_LABELS, **main_labels}
    summary = (
        edges.groupby("sender", as_index=False)
        .agg(
            top50_edge_count=("id", "size"),
            unique_ligand_count=("ligand", "nunique"),
            unique_receptor_count=("receptor", "nunique"),
            prioritization_score_sum=("prioritization_score", "sum"),
            prioritization_score_mean=("prioritization_score", "mean"),
            prioritization_score_max=("prioritization_score", "max"),
        )
        .sort_values("prioritization_score_sum", ascending=False)
        .reset_index(drop=True)
    )
    summary["sender_label"] = summary["sender"].map(labels).fillna(summary["sender"])
    summary["contribution_percent"] = (
        100
        * summary["prioritization_score_sum"]
        / summary["prioritization_score_sum"].sum()
    )
    summary["selection_rule"] = (
        "All sender types represented among the Q1-prioritized top-50 sender-to-LEC edges"
    )
    return summary


def build_extended_interactions(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = data["main_lr_source"].dropna(subset=["ligand", "receptor"])
    main_pairs = set(zip(main["ligand"].astype(str), main["receptor"].astype(str)))

    lr = data["ligand_receptor_links"].copy().rename(
        columns={"from": "ligand", "to": "receptor"}
    )
    lr["shown_in_main_figure_3"] = [
        (ligand, receptor) in main_pairs
        for ligand, receptor in zip(lr["ligand"], lr["receptor"])
    ]
    lr = lr.loc[~lr["shown_in_main_figure_3"]].copy()
    lr = lr.sort_values(["direction", "weight"], ascending=[True, False])
    lr = lr.groupby("direction", sort=False, as_index=False).head(4).copy()
    lr["rank_within_direction"] = lr.groupby("direction").cumcount() + 1
    lr["interaction"] = lr["ligand"] + " → " + lr["receptor"]
    lr["direction_display"] = lr["direction"].map(direction_display)
    lr["selection_rule"] = (
        "Top four ligand-receptor links per direction after excluding interactions "
        "displayed in main Figure 3F"
    )

    targets = data["ligand_target_links"].copy()
    targets = targets.sort_values(["direction", "weight"], ascending=[True, False])
    targets = targets.groupby("direction", sort=False, as_index=False).head(4).copy()
    targets["rank_within_direction"] = targets.groupby("direction").cumcount() + 1
    targets["interaction"] = targets["ligand"] + " → " + targets["target"]
    targets["direction_display"] = targets["direction"].map(direction_display)
    targets["selection_rule"] = "Top four ligand-target links per direction by link weight"
    return lr.reset_index(drop=True), targets.reset_index(drop=True)


def build_de_sensitivity(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    primary = data["de_primary"][["gene", "logFC", "FDR"]].rename(
        columns={"logFC": "primary_logFC", "FDR": "primary_FDR"}
    )
    top_genes = set(primary.nsmallest(5, "primary_FDR")["gene"])
    records: list[pd.DataFrame] = []
    for input_id, label in [
        ("de_gse183852", "GSE183852-only model"),
        ("de_project_sex", "Project- and sex-adjusted model"),
    ]:
        sensitivity = data[input_id][["gene", "logFC", "FDR", "model"]].rename(
            columns={"logFC": "sensitivity_logFC", "FDR": "sensitivity_FDR"}
        )
        merged = primary.merge(sensitivity, on="gene", how="inner", validate="one_to_one")
        merged["sensitivity_analysis"] = label
        merged["highlight_primary_top5_FDR"] = merged["gene"].isin(top_genes)
        merged["pearson_r"] = merged["primary_logFC"].corr(
            merged["sensitivity_logFC"], method="pearson"
        )
        merged["spearman_rho"] = merged["primary_logFC"].corr(
            merged["sensitivity_logFC"], method="spearman"
        )
        merged["selection_rule"] = (
            "All genes shared with the primary analysis; labels mark the five genes "
            "with the lowest primary-model FDR"
        )
        records.append(merged)
    return pd.concat(records, ignore_index=True)


def validate_tables(
    data: dict[str, pd.DataFrame],
    heatmap: pd.DataFrame,
    robustness: pd.DataFrame,
    all_pairs: pd.DataFrame,
    ligands: pd.DataFrame,
    senders: pd.DataFrame,
    lr: pd.DataFrame,
    targets: pd.DataFrame,
    de: pd.DataFrame,
) -> None:
    if set(robustness["target_state"]) != set(ALL_STATES):
        raise ValueError("Extended regulon selection must cover every aging LEC state.")
    counts = robustness.groupby("target_state").size()
    if counts.min() < 1 or counts.max() > 2:
        raise ValueError("Expected one or two extended regulons per state.")
    if heatmap.groupby("row_order")["heatmap_state"].nunique().ne(11).any():
        raise ValueError("Each extended regulon must be evaluated across all 11 states.")
    if all_pairs.shape[0] != data["scenic_full_summary"].shape[0]:
        raise ValueError("Panel C must retain the complete state-regulon summary.")
    if not ligands.groupby("direction").size().eq(8).all():
        raise ValueError("Panel D requires eight additional ligands per direction.")
    if senders["sender"].nunique() != data["top50_sender_edges"]["sender"].nunique():
        raise ValueError("Panel E must retain every sender in the top-50 table.")
    if not lr.groupby("direction").size().eq(4).all():
        raise ValueError("Panel F requires four additional ligand-receptor links per direction.")
    if not targets.groupby("direction").size().eq(4).all():
        raise ValueError("Panel F requires four ligand-target links per direction.")
    if de["sensitivity_analysis"].nunique() != 2:
        raise ValueError("Panel G requires both prespecified sensitivity models.")


def draw_panel_a(
    fig: plt.Figure,
    spec,
    heatmap: pd.DataFrame,
) -> list[plt.Axes]:
    order = (
        heatmap[["row_order", "row_label", "target_state"]]
        .drop_duplicates()
        .sort_values("row_order")
    )
    matrix = (
        heatmap.pivot(
            index="row_order",
            columns="heatmap_state_number",
            values="donor_balanced_mean_AUCell_zscore",
        )
        .reindex(index=order["row_order"], columns=range(11))
    )
    grid = spec.subgridspec(1, 2, width_ratios=[1, 0.035], wspace=0.05)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
    )
    ax.set_xticks(range(11), [f"State {i}" for i in range(11)], rotation=45, ha="right")
    ax.set_yticks(range(len(order)), order["row_label"])
    ax.set_xlabel("LEC state")
    ax.set_title(
        "Extended state-specific regulon activity",
        loc="left",
        pad=9,
    )
    ax.tick_params(length=0)
    for y, state in enumerate(order["target_state"]):
        x = state_number(state)
        ax.add_patch(
            plt.Rectangle(
                (x - 0.48, y - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor=DARK,
                linewidth=1.15,
            )
        )
    state_numbers = order["target_state"].map(state_number).to_numpy()
    for y in np.where(np.diff(state_numbers) != 0)[0] + 0.5:
        ax.axhline(y, color="white", linewidth=1.8)
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Donor-balanced mean AUCell\n(state-wise z score)", fontsize=8.2)
    colorbar.ax.tick_params(labelsize=7.5)
    panel_letter(ax, "A", x=-0.21, y=1.03)
    return [ax, cax]


def draw_panel_b(fig: plt.Figure, spec, frame: pd.DataFrame) -> list[plt.Axes]:
    ax = fig.add_subplot(spec)
    ordered = frame.sort_values(
        ["state_number", "consensus_rank_within_state", "regulon"], ascending=[False, False, False]
    ).reset_index(drop=True)
    y = np.arange(len(ordered))
    x = ordered["donor_bootstrap_RSS_median"].to_numpy()
    xerr = np.vstack(
        [
            x - ordered["donor_bootstrap_RSS_q025"].to_numpy(),
            ordered["donor_bootstrap_RSS_q975"].to_numpy() - x,
        ]
    )
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#76828B",
        elinewidth=1.1,
        capsize=2,
        zorder=1,
    )
    size = 30 + 105 * ordered["donor_bootstrap_RSS_top_frequency"].to_numpy()
    scatter = ax.scatter(
        x,
        y,
        s=size,
        c=ordered["donor_bootstrap_RSS_top_frequency"],
        cmap="viridis",
        norm=Normalize(0, 1),
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
    )
    ax.set_yticks(y, ordered["row_label"])
    ax.set_xlabel("Donor-bootstrap RSS median (95% interval)")
    ax.set_title("Additional donor-bootstrap reproducibility", loc="left", pad=9)
    clean_axis(ax, "x")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("Within-state top-8 frequency", fontsize=8.0)
    colorbar.ax.tick_params(labelsize=7.3)
    panel_letter(ax, "B", x=-0.29, y=1.03)
    return [ax]


def draw_panel_c(fig: plt.Figure, spec, frame: pd.DataFrame) -> list[plt.Axes]:
    ax = fig.add_subplot(spec)
    ax.scatter(
        frame["equal_cell_RSS_top_frequency"],
        frame["donor_bootstrap_RSS_top_frequency"],
        s=12,
        color=LIGHT_GREY,
        alpha=0.55,
        edgecolor="none",
        rasterized=True,
        label="All state-regulon pairs",
    )
    robust = frame.loc[frame["robust_candidate_flag"]]
    ax.scatter(
        robust["equal_cell_RSS_top_frequency"],
        robust["donor_bootstrap_RSS_top_frequency"],
        s=32,
        facecolor="white",
        edgecolor=TEAL,
        linewidth=0.9,
        alpha=0.9,
        label="Robust candidates",
    )
    selected = frame.loc[frame["selected_extended_regulon"]]
    ax.scatter(
        selected["equal_cell_RSS_top_frequency"],
        selected["donor_bootstrap_RSS_top_frequency"],
        s=42,
        color=TEAL,
        edgecolor="white",
        linewidth=0.5,
        label="Panels A–B selection",
        zorder=4,
    )
    ax.plot([0, 1], [0, 1], color="#9AA4AC", linestyle=(0, (3, 2)), linewidth=0.9)
    pearson = frame["equal_cell_RSS_top_frequency"].corr(
        frame["donor_bootstrap_RSS_top_frequency"], method="pearson"
    )
    spearman = frame["equal_cell_RSS_top_frequency"].corr(
        frame["donor_bootstrap_RSS_top_frequency"], method="spearman"
    )
    ax.text(
        0.04,
        0.96,
        f"All pairs: Pearson r = {pearson:.2f}\nSpearman ρ = {spearman:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        color=DARK,
    )
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.025, 1.025)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Equal-cell resampling: within-state top-8 frequency")
    ax.set_ylabel("Donor bootstrap: within-state top-8 frequency")
    ax.set_title("Regulon robustness across analysis schemes", loc="left", pad=9)
    clean_axis(ax, "both")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    panel_letter(ax, "C", x=-0.22, y=1.03)
    return [ax]


def draw_panel_d(fig: plt.Figure, spec, frame: pd.DataFrame) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 2, wspace=0.28)
    axes: list[plt.Axes] = []
    for index, direction in enumerate(["Q4_downregulated", "Q4_upregulated"]):
        ax = fig.add_subplot(grid[0, index])
        subset = frame.loc[frame["direction"].eq(direction)].sort_values(
            "aupr_corrected", ascending=True
        )
        y = np.arange(len(subset))
        color = direction_color(direction)
        ax.barh(y, subset["aupr_corrected"], height=0.62, color=color, alpha=0.92)
        ax.set_yticks(y, subset["test_ligand"].map(lambda x: f"{x}  (rank {int(subset.loc[subset['test_ligand'].eq(x), 'rank_in_complete_table'].iloc[0])})"))
        xmax = subset["aupr_corrected"].max() * 1.22
        ax.set_xlim(0, xmax)
        for yi, value in zip(y, subset["aupr_corrected"]):
            ax.text(value + xmax * 0.018, yi, f"{value:.3f}", va="center", fontsize=7.3)
        ax.set_xlabel("Corrected AUPR")
        ax.set_title(direction_display(direction), loc="left", color=color, pad=8)
        clean_axis(ax, "x")
        if index == 0:
            panel_letter(ax, "D", x=-0.24, y=1.16)
        axes.append(ax)
    axes[0].text(
        0,
        1.12,
        "Extended MultiNicheNet ligand rankings",
        transform=axes[0].transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=DARK,
        ha="left",
    )
    return axes


def draw_panel_e(fig: plt.Figure, spec, frame: pd.DataFrame) -> list[plt.Axes]:
    ax = fig.add_subplot(spec)
    subset = frame.sort_values("contribution_percent", ascending=True)
    y = np.arange(len(subset))
    bars = ax.barh(y, subset["contribution_percent"], height=0.62, color=NAVY, alpha=0.9)
    ax.set_yticks(y, subset["sender_label"])
    ax.set_xlabel("Share of summed prioritization score (%)")
    ax.set_title("Sender-cell contributions", loc="left", pad=9)
    xmax = subset["contribution_percent"].max() * 1.32
    ax.set_xlim(0, xmax)
    for bar, (_, row) in zip(bars, subset.iterrows()):
        ax.text(
            bar.get_width() + xmax * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['top50_edge_count'])} edges",
            va="center",
            fontsize=7.8,
            color=DARK,
        )
    clean_axis(ax, "x")
    ax.text(
        0,
        -0.20,
        "All sender types in the Q1-prioritized top-50 sender-to-LEC table.",
        transform=ax.transAxes,
        fontsize=7.7,
        color=SLATE,
    )
    panel_letter(ax, "E", x=-0.32, y=1.03)
    return [ax]


def draw_lollipop_interactions(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    entity_column: str,
) -> None:
    ordered = frame.sort_values(["direction", "weight"], ascending=[False, True]).reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = [direction_color(value) for value in ordered["direction"]]
    for yi, value, color in zip(y, ordered["weight"], colors):
        ax.hlines(yi, 0, value, color=color, linewidth=2.0, alpha=0.75)
    ax.scatter(ordered["weight"], y, s=55, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(y, ordered[entity_column])
    ax.set_xlabel("Link weight")
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlim(0, ordered["weight"].max() * 1.13)
    clean_axis(ax, "x")


def draw_panel_f(
    fig: plt.Figure,
    spec,
    lr: pd.DataFrame,
    targets: pd.DataFrame,
) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 2, wspace=0.38)
    ax_lr = fig.add_subplot(grid[0, 0])
    ax_target = fig.add_subplot(grid[0, 1])
    draw_lollipop_interactions(ax_lr, lr, "Additional ligand–receptor links", "interaction")
    draw_lollipop_interactions(ax_target, targets, "Additional ligand–target links", "interaction")
    ax_lr.scatter([], [], s=50, color=DOWN, label="Q4-downregulated program")
    ax_lr.scatter([], [], s=50, color=UP, label="Q4-upregulated program")
    ax_lr.legend(frameon=False, loc="lower right", fontsize=7.2)
    panel_letter(ax_lr, "F", x=-0.30, y=1.15)
    ax_lr.text(
        0,
        1.12,
        "Extended signaling relationships",
        transform=ax_lr.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=DARK,
        ha="left",
    )
    return [ax_lr, ax_target]


def draw_panel_g(fig: plt.Figure, spec, frame: pd.DataFrame) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 2, wspace=0.22)
    axes: list[plt.Axes] = []
    analyses = ["GSE183852-only model", "Project- and sex-adjusted model"]
    for index, analysis in enumerate(analyses):
        ax = fig.add_subplot(grid[0, index])
        subset = frame.loc[frame["sensitivity_analysis"].eq(analysis)].copy()
        values = np.r_[subset["primary_logFC"], subset["sensitivity_logFC"]]
        bound = max(5.0, float(np.nanmax(np.abs(values))) * 1.06)
        ax.scatter(
            subset["primary_logFC"],
            subset["sensitivity_logFC"],
            s=13,
            color="#AFB8BE",
            alpha=0.48,
            edgecolor="none",
            rasterized=True,
        )
        highlighted = subset.loc[subset["highlight_primary_top5_FDR"]].sort_values("primary_FDR")
        ax.scatter(
            highlighted["primary_logFC"],
            highlighted["sensitivity_logFC"],
            s=42,
            color=ORANGE,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        offsets = [(5, 5), (5, -11), (-6, 6), (5, 5), (-6, -12)]
        for offset, (_, row) in zip(offsets, highlighted.iterrows()):
            ax.annotate(
                row["gene"],
                (row["primary_logFC"], row["sensitivity_logFC"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.2,
                ha="right" if offset[0] < 0 else "left",
                color=DARK,
            )
        ax.plot([-bound, bound], [-bound, bound], color="#87939B", linewidth=0.9, linestyle=(0, (3, 2)))
        ax.axhline(0, color=GRID, linewidth=0.7)
        ax.axvline(0, color=GRID, linewidth=0.7)
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Primary donor-pseudobulk logFC (Q4 − Q1)")
        if index == 0:
            ax.set_ylabel("Sensitivity-model logFC (Q4 − Q1)")
            panel_letter(ax, "G", x=-0.22, y=1.16)
        ax.set_title(analysis, loc="left", pad=8)
        ax.text(
            0.04,
            0.96,
            f"Pearson r = {subset['pearson_r'].iloc[0]:.2f}\nSpearman ρ = {subset['spearman_rho'].iloc[0]:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            color=DARK,
        )
        clean_axis(ax)
        axes.append(ax)
    axes[0].text(
        0,
        1.12,
        "Differential-expression sensitivity underlying MultiNicheNet",
        transform=axes[0].transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=DARK,
        ha="left",
    )
    return axes


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
    heatmap: pd.DataFrame,
    robustness: pd.DataFrame,
    all_pairs: pd.DataFrame,
    ligands: pd.DataFrame,
    senders: pd.DataFrame,
    lr: pd.DataFrame,
    targets: pd.DataFrame,
    de: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(13.5, 8.4), constrained_layout=True)
    draw_panel_a(fig, GridSpec(1, 1, figure=fig)[0], heatmap)
    save_figure(fig, "FigS04A_extended_SCENIC_regulons")

    fig = plt.figure(figsize=(8.7, 8.2), constrained_layout=True)
    draw_panel_b(fig, GridSpec(1, 1, figure=fig)[0], robustness)
    save_figure(fig, "FigS04B_additional_donor_bootstrap_reproducibility")

    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    draw_panel_c(fig, GridSpec(1, 1, figure=fig)[0], all_pairs)
    save_figure(fig, "FigS04C_regulon_robustness")

    fig = plt.figure(figsize=(12.8, 5.8), constrained_layout=True)
    draw_panel_d(fig, GridSpec(1, 1, figure=fig)[0], ligands)
    save_figure(fig, "FigS04D_extended_ligand_rankings")

    fig = plt.figure(figsize=(7.8, 5.2), constrained_layout=True)
    draw_panel_e(fig, GridSpec(1, 1, figure=fig)[0], senders)
    save_figure(fig, "FigS04E_sender_cell_contributions")

    fig = plt.figure(figsize=(13.0, 5.6), constrained_layout=True)
    draw_panel_f(fig, GridSpec(1, 1, figure=fig)[0], lr, targets)
    save_figure(fig, "FigS04F_extended_ligand_receptor_target_links")

    fig = plt.figure(figsize=(12.2, 5.6), constrained_layout=True)
    draw_panel_g(fig, GridSpec(1, 1, figure=fig)[0], de)
    save_figure(fig, "FigS04G_MultiNicheNet_DE_sensitivity")


def make_combined_figure(
    heatmap: pd.DataFrame,
    robustness: pd.DataFrame,
    all_pairs: pd.DataFrame,
    ligands: pd.DataFrame,
    senders: pd.DataFrame,
    lr: pd.DataFrame,
    targets: pd.DataFrame,
    de: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(18.0, 31.0), constrained_layout=True)
    grid = fig.add_gridspec(
        6,
        2,
        height_ratios=[1.30, 1.24, 0.88, 0.64, 0.88, 1.00],
        width_ratios=[1, 1],
        hspace=0.16,
        wspace=0.14,
    )
    draw_panel_a(fig, grid[0, :], heatmap)
    draw_panel_b(fig, grid[1, 0], robustness)
    draw_panel_c(fig, grid[1, 1], all_pairs)
    draw_panel_d(fig, grid[2, :], ligands)
    draw_panel_e(fig, grid[3, :], senders)
    draw_panel_f(fig, grid[4, :], lr, targets)
    draw_panel_g(fig, grid[5, :], de)
    fig.suptitle(
        "Figure S4 | Extended aging SCENIC and MultiNicheNet analyses",
        fontsize=18,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    save_figure(fig, "FigS04_extended_aging_SCENIC_MultiNicheNet")


def write_traceability(
    heatmap: pd.DataFrame,
    robustness: pd.DataFrame,
    all_pairs: pd.DataFrame,
    ligands: pd.DataFrame,
    senders: pd.DataFrame,
    lr: pd.DataFrame,
    targets: pd.DataFrame,
    de: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for panel, record_type, frame in [
        ("A", "extended_regulon_heatmap", heatmap),
        ("B", "extended_regulon_donor_bootstrap", robustness),
        ("C", "all_regulon_robustness_comparison", all_pairs),
        ("D", "extended_ligand_activity", ligands),
        ("E", "sender_cell_contribution", senders),
        ("F", "additional_ligand_receptor_link", lr),
        ("F", "additional_ligand_target_link", targets),
        ("G", "differential_expression_sensitivity", de),
    ]:
        output = frame.copy()
        output.insert(0, "panel", panel)
        output.insert(1, "record_type", record_type)
        frames.append(output)
    pd.concat(frames, ignore_index=True, sort=False).to_csv(
        OUT_DIR / "FigS04_source_data.csv", index=False
    )

    panel_map = {
        "scenic_full_summary": "A; B; C",
        "main_regulon_source": "A; B exclusion check",
        "ligand_activities": "D",
        "main_ligand_source": "D exclusion check",
        "top50_sender_edges": "E",
        "main_sender_source": "E sender display labels",
        "ligand_receptor_links": "F",
        "ligand_target_links": "F",
        "main_lr_source": "F exclusion check",
        "de_primary": "G",
        "de_gse183852": "G",
        "de_project_sex": "G",
    }
    purpose_map = {
        "scenic_full_summary": "Complete state-regulon metrics used for extended selection and robustness comparisons",
        "main_regulon_source": "Exclude target-state/regulon pairs already displayed in main Figure 3A",
        "ligand_activities": "Complete corrected-AUPR ligand ranking in both age directions",
        "main_ligand_source": "Exclude ligands already displayed in main Figure 3D",
        "top50_sender_edges": "Quantify contributions from every sender represented among the top 50 Q1-prioritized edges",
        "main_sender_source": "Reuse established sender-cell display labels",
        "ligand_receptor_links": "Additional directional ligand-receptor links",
        "ligand_target_links": "Additional directional ligand-target links",
        "main_lr_source": "Exclude ligand-receptor pairs already displayed in main Figure 3F",
        "de_primary": "Primary donor-pseudobulk age contrast underlying the receiver gene set",
        "de_gse183852": "Single-project differential-expression sensitivity analysis",
        "de_project_sex": "Project- and sex-adjusted differential-expression sensitivity analysis",
    }
    manifest = pd.DataFrame(
        [
            {
                "input_id": input_id,
                "panels": panel_map[input_id],
                "input_csv": str(path.relative_to(ROOT)).replace("\\", "/"),
                "purpose": purpose_map[input_id],
            }
            for input_id, path in INPUTS.items()
        ]
    )
    manifest.to_csv(OUT_DIR / "FigS04_input_manifest.csv", index=False)


def main() -> None:
    configure_style()
    data = read_inputs()
    heatmap, robustness, all_pairs = build_extended_regulons(data)
    ligands = build_extended_ligands(data)
    senders = build_sender_contributions(data)
    lr, targets = build_extended_interactions(data)
    de = build_de_sensitivity(data)
    validate_tables(
        data,
        heatmap,
        robustness,
        all_pairs,
        ligands,
        senders,
        lr,
        targets,
        de,
    )
    write_traceability(
        heatmap,
        robustness,
        all_pairs,
        ligands,
        senders,
        lr,
        targets,
        de,
    )
    make_individual_panels(
        heatmap,
        robustness,
        all_pairs,
        ligands,
        senders,
        lr,
        targets,
        de,
    )
    make_combined_figure(
        heatmap,
        robustness,
        all_pairs,
        ligands,
        senders,
        lr,
        targets,
        de,
    )
    print(f"Supplementary Figure S4 written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
