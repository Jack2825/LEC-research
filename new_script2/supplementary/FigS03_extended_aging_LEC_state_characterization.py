#!/usr/bin/env python3
"""Build Supplementary Figure S3: extended aging LEC-state analyses.

The five panels add donor-level, rare-state, molecular, and sensitivity evidence
without repeating the principal UMAP, composition, marker/pathway, or selected
effect panels already presented in main Figure 2.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = (
    ROOT
    / "new_result2"
    / "supplementary"
    / "FigS03_extended_aging_LEC_state_characterization"
)

DONOR_DIR = (
    ROOT
    / "new_result2_aging"
    / "08_7_min25_LEC_filtered_age_analysis"
    / "dataset13"
    / "three_groups_Q2_Q3_merged"
    / "donor_level"
)

INPUTS = {
    "continuous_age": DONOR_DIR / "08_7_Q2_Q3_merged_continuous_age_spearman.csv",
    "donor_dominance": DONOR_DIR
    / "08_7_Q2_Q3_merged_highest_donor_contribution_by_group_state.csv",
    "donor_state_abundance": DONOR_DIR
    / "08_7_Q2_Q3_merged_donor_state_counts_percentages.csv.gz",
    "leave_top_donor_out": DONOR_DIR
    / "08_7_Q2_Q3_merged_leave_top_donor_out_sensitivity.csv",
    "rare_state_markers": ROOT
    / "new_result2_aging"
    / "08_combined_LEC_recluster"
    / "dataset13"
    / "08_state_top25_markers_long.csv",
    "gsea_top_pathways": ROOT
    / "new_result2_aging"
    / "09_LEC_state_GSEA"
    / "dataset13"
    / "09_GSEA_top_pathways_per_state.csv",
    "main_fig02c_source": ROOT
    / "new_result2"
    / "aging"
    / "Fig02C_aging_marker_GSEA_map"
    / "Fig02C_source_data.csv",
    "state_colors": ROOT
    / "new_result2_aging"
    / "08_combined_LEC_recluster"
    / "dataset13"
    / "08_state_color_mapping.csv",
}

ALL_STATES = [f"state{index}" for index in range(11)]
RARE_STATES = ["state7", "state8", "state9", "state10"]
GROUPS = ["Q1", "Q2+Q3", "Q4"]
GROUP_LABELS = ["Q1\nyounger", "Q2+Q3\nmiddle", "Q4\nolder"]
RARE_DEFINITIONS = {
    "state7": "Rare stress-associated LEC",
    "state8": "Rare regulatory LEC",
    "state9": "Rare transitional LEC",
    "state10": "Rare mature LEC",
}

DARK = "#202529"
SLATE = "#687786"
GRID = "#DDE2E6"
NAVY = "#2C5F8A"
TEAL = "#3A8D8C"
PALE = "#F2F5F6"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
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


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.18, y: float = 1.11) -> None:
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


def build_panel_a(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = data["continuous_age"].copy()
    frame["state_number"] = frame["LEC_state"].map(state_number)
    frame["state_label"] = frame["LEC_state"].map(state_label)
    return frame.sort_values("state_number").reset_index(drop=True)


def build_panel_b(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = data["donor_dominance"].copy()
    frame["state_number"] = frame["LEC_state"].map(state_number)
    return frame.sort_values(["state_number", "age_group"]).reset_index(drop=True)


def build_panel_c(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = data["donor_state_abundance"].copy()
    frame["cell_state_percentage"] = pd.to_numeric(
        frame["cell_state_percentage"], errors="raise"
    )
    return frame.loc[frame["LEC_state"].isin(RARE_STATES)].copy()


def build_panel_d(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    markers = data["rare_state_markers"].copy()
    markers = markers.loc[
        markers["LEC_state"].isin(RARE_STATES) & markers["rank"].le(3)
    ].copy()
    markers["marker_fdr_05"] = markers["pvals_adj"].le(0.05)
    markers["selection_rule"] = "Top three ranked markers per rare state"
    markers = markers.sort_values(
        ["LEC_state", "rank"], key=lambda col: col.map(state_number) if col.name == "LEC_state" else col
    ).reset_index(drop=True)

    main = data["main_fig02c_source"].copy()
    main = main.loc[main["record_type"].eq("GSEA_pathway")].copy()
    main["LEC_state"] = "state" + main["defining_state"].astype(str)
    used_pairs = set(zip(main["LEC_state"], main["pathway_name"]))

    pathways = data["gsea_top_pathways"].copy()
    pathways = pathways.loc[pathways["LEC_state"].isin(RARE_STATES)].copy()
    pathways = pathways.loc[
        [
            (state, pathway) not in used_pairs
            for state, pathway in zip(pathways["LEC_state"], pathways["pathway_name"])
        ]
    ].copy()
    pathways["absolute_NES"] = pathways["NES"].abs()
    pathways = pathways.sort_values(
        ["LEC_state", "FDR_q_value", "absolute_NES"],
        ascending=[True, True, False],
        key=lambda col: col.map(state_number) if col.name == "LEC_state" else col,
    )
    pathways = pathways.groupby("LEC_state", sort=False, as_index=False).head(2).copy()
    pathways["pathway_fdr_025"] = pathways["FDR_q_value"].le(0.25)
    pathways["selection_rule"] = (
        "Two lowest-FDR pathways per rare state after excluding pathways shown in Figure 2"
    )
    pathways = pathways.reset_index(drop=True)
    return markers, pathways


def build_panel_e(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    donor = data["donor_state_abundance"].copy()
    donor["cell_state_percentage"] = pd.to_numeric(
        donor["cell_state_percentage"], errors="raise"
    )
    leaveout = data["leave_top_donor_out"].set_index("LEC_state")
    records: list[dict[str, object]] = []
    for state in ALL_STATES:
        state_rows = donor.loc[donor["LEC_state"].eq(state)].copy()
        excluded = str(leaveout.loc[state, "excluded_top_donor"])
        reduced = state_rows.loc[state_rows["donor_key"].ne(excluded)].copy()
        for group, contrast in [("Q2+Q3", "Q2+Q3 − Q1"), ("Q4", "Q4 − Q1")]:
            original_effect = float(
                state_rows.loc[state_rows["age_group"].eq(group), "cell_state_percentage"].mean()
                - state_rows.loc[
                    state_rows["age_group"].eq("Q1"), "cell_state_percentage"
                ].mean()
            )
            leaveout_effect = float(
                reduced.loc[reduced["age_group"].eq(group), "cell_state_percentage"].mean()
                - reduced.loc[
                    reduced["age_group"].eq("Q1"), "cell_state_percentage"
                ].mean()
            )
            records.append(
                {
                    "LEC_state": state,
                    "state_number": state_number(state),
                    "contrast": contrast,
                    "excluded_top_donor": excluded,
                    "original_effect_pp": original_effect,
                    "leaveout_effect_pp": leaveout_effect,
                    "leaveout_change_pp": leaveout_effect - original_effect,
                }
            )
    return pd.DataFrame.from_records(records)


def validate_tables(
    data: dict[str, pd.DataFrame],
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    marker_data: pd.DataFrame,
    pathway_data: pd.DataFrame,
    panel_e: pd.DataFrame,
) -> None:
    donor = data["donor_state_abundance"]
    if donor["donor_key"].nunique() != 13 or donor["n_total_LEC_cells"].min() < 25:
        raise ValueError("S3 donor-level inputs are not the expected 13 donors with at least 25 LECs.")
    if set(panel_a["LEC_state"]) != set(ALL_STATES):
        raise ValueError("Panel A does not contain all 11 aging LEC states.")
    if set(panel_b["LEC_state"]) != set(ALL_STATES) or set(panel_b["age_group"]) != set(GROUPS):
        raise ValueError("Panel B does not contain every state and age group.")
    if set(panel_c["LEC_state"]) != set(RARE_STATES):
        raise ValueError("Panel C does not contain all four less-abundant states.")
    if not (marker_data.groupby("LEC_state").size() == 3).all():
        raise ValueError("Panel D must contain exactly three ranked markers per rare state.")
    if not (pathway_data.groupby("LEC_state").size() == 2).all():
        raise ValueError("Panel D must contain exactly two additional pathways per rare state.")
    main = data["main_fig02c_source"]
    main = main.loc[main["record_type"].eq("GSEA_pathway")]
    main_pairs = set(
        zip("state" + main["defining_state"].astype(str), main["pathway_name"])
    )
    if any(
        (state, pathway) in main_pairs
        for state, pathway in zip(pathway_data["LEC_state"], pathway_data["pathway_name"])
    ):
        raise ValueError("Panel D repeats a pathway already shown in main Figure 2.")
    if len(panel_e) != 22 or set(panel_e["LEC_state"]) != set(ALL_STATES):
        raise ValueError("Panel E leave-top-donor-out sensitivity table is incomplete.")


def draw_panel_a(
    fig: plt.Figure,
    spec,
    frame: pd.DataFrame,
    colors: dict[str, str],
) -> plt.Axes:
    ax = fig.add_subplot(spec)
    y = np.arange(len(frame))
    rho = frame["spearman_rho_age_vs_state_percentage"].to_numpy(float)
    ax.axvline(0, color="#AEB6BD", linewidth=0.9)
    for position, (_, row) in enumerate(frame.iterrows()):
        state = row["LEC_state"]
        ax.hlines(position, 0, row["spearman_rho_age_vs_state_percentage"], color=colors[state], linewidth=1.5)
        ax.scatter(
            row["spearman_rho_age_vs_state_percentage"],
            position,
            s=52,
            color=colors[state],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.text(
            0.77,
            position,
            f"q={row['spearman_fdr_bh']:.2f}",
            ha="right",
            va="center",
            fontsize=7.5,
            color=SLATE,
        )
    ax.set_yticks(y, frame["state_label"])
    ax.invert_yaxis()
    ax.set_xlim(-0.80, 0.80)
    ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax.set_xlabel("Spearman correlation with age (ρ)")
    ax.set_title(
        "Continuous-age associations across all states\n13 donors with ≥25 LECs; q values are BH FDR",
        loc="left",
        pad=8,
    )
    clean_axis(ax, "x")
    panel_letter(ax, "A", x=-0.31, y=1.10)
    return ax


def draw_panel_b(fig: plt.Figure, spec, frame: pd.DataFrame) -> plt.Axes:
    ax = fig.add_subplot(spec)
    contribution = (
        frame.pivot(index="LEC_state", columns="age_group", values="top_donor_percent_of_group_state")
        .reindex(index=ALL_STATES, columns=GROUPS)
    )
    donor_support = (
        frame.pivot(index="LEC_state", columns="age_group", values="n_donors_with_state")
        .reindex(index=ALL_STATES, columns=GROUPS)
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "donor_dominance", ["#F5F7F8", "#B9D0D4", NAVY]
    )
    image = ax.imshow(contribution.to_numpy(float), aspect="auto", cmap=cmap, vmin=0, vmax=100)
    for row_index in range(len(ALL_STATES)):
        for column_index in range(len(GROUPS)):
            value = contribution.iloc[row_index, column_index]
            support = donor_support.iloc[row_index, column_index]
            if np.isnan(value):
                label = "not detected"
                color = SLATE
            else:
                label = f"{value:.0f}%\n(n={int(support)})"
                color = "white" if value >= 62 else DARK
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7.1,
                color=color,
            )
    ax.set_xticks(np.arange(len(GROUPS)), GROUP_LABELS)
    ax.set_yticks(np.arange(len(ALL_STATES)), [state_label(state) for state in ALL_STATES])
    ax.set_title(
        "Maximum single-donor contribution\nCell text: top-donor share; n=donors containing the state",
        loc="left",
        pad=8,
    )
    for position in np.arange(-0.5, len(GROUPS), 1):
        ax.axvline(position, color="white", linewidth=1)
    for position in np.arange(-0.5, len(ALL_STATES), 1):
        ax.axhline(position, color="white", linewidth=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Top donor's share of state cells (%)", fontsize=8.3)
    colorbar.ax.tick_params(labelsize=7.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_letter(ax, "B", x=-0.30, y=1.10)
    return ax


def donor_offsets(n: int) -> np.ndarray:
    return np.linspace(-0.13, 0.13, n) if n > 1 else np.array([0.0])


def draw_panel_c(
    fig: plt.Figure,
    spec,
    frame: pd.DataFrame,
    colors: dict[str, str],
) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 4, wspace=0.12)
    axes: list[plt.Axes] = []
    for index, state in enumerate(RARE_STATES):
        ax = fig.add_subplot(grid[0, index])
        state_rows = frame.loc[frame["LEC_state"].eq(state)].copy()
        grouped = [
            state_rows.loc[
                state_rows["age_group"].eq(group), "cell_state_percentage"
            ].to_numpy(float)
            for group in GROUPS
        ]
        boxes = ax.boxplot(
            grouped,
            positions=np.arange(3),
            widths=0.50,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            medianprops={"color": DARK, "linewidth": 1.3},
            boxprops={"edgecolor": colors[state], "linewidth": 1.0},
            whiskerprops={"color": colors[state], "linewidth": 0.9},
            capprops={"color": colors[state], "linewidth": 0.9},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(mpl.colors.to_rgba(colors[state], 0.16))
        for position, group in enumerate(GROUPS):
            values = state_rows.loc[state_rows["age_group"].eq(group)].sort_values("donor_key")
            ax.scatter(
                position + donor_offsets(len(values)),
                values["cell_state_percentage"],
                s=31,
                color=colors[state],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
        ax.set_xticks(np.arange(3), ["Q1", "Q2+Q3", "Q4"])
        ax.set_ylim(-0.5, 15)
        ax.set_title(
            f"{state_label(state)}\n{RARE_DEFINITIONS[state]}",
            loc="left",
            pad=8,
        )
        clean_axis(ax, "y")
        if index == 0:
            ax.set_ylabel("Abundance within donor (%)")
            panel_letter(ax, "C", x=-0.30, y=1.12)
        else:
            ax.set_yticklabels([])
        axes.append(ax)
    return axes


def marker_label(row: pd.Series) -> str:
    return f"S{state_number(row['LEC_state'])} · {row['names']}"


def pathway_label(row: pd.Series) -> str:
    collection = str(row["collection"]).replace("KEGG_Medicus", "KEGG")
    name = str(row["pathway_display_name"]).replace("_", " ").title()
    return f"S{state_number(row['LEC_state'])} · {collection}: {name}"


def draw_panel_d(
    fig: plt.Figure,
    spec,
    markers: pd.DataFrame,
    pathways: pd.DataFrame,
    colors: dict[str, str],
) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.46)
    ax_marker = fig.add_subplot(grid[0, 0])
    marker_plot = markers.sort_values(
        ["LEC_state", "rank"],
        ascending=[False, False],
        key=lambda col: col.map(state_number) if col.name == "LEC_state" else col,
    ).reset_index(drop=True)
    y = np.arange(len(marker_plot))
    for position, (_, row) in enumerate(marker_plot.iterrows()):
        color = colors[row["LEC_state"]]
        significant = bool(row["marker_fdr_05"])
        ax_marker.hlines(position, 0, row["logfoldchanges"], color=color, linewidth=1.2)
        ax_marker.scatter(
            row["logfoldchanges"],
            position,
            s=28 + 90 * float(row["pct_nz_group"]),
            facecolor=color if significant else "white",
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )
    ax_marker.axvline(0, color="#AEB6BD", linewidth=0.8)
    ax_marker.set_yticks(y, [marker_label(row) for _, row in marker_plot.iterrows()])
    ax_marker.set_xlabel("Marker log fold-change")
    ax_marker.set_title(
        "Additional marker evidence\nTop three ranked genes per rare state",
        loc="left",
        pad=8,
    )
    clean_axis(ax_marker, "x")
    ax_marker.scatter([], [], s=65, facecolor=DARK, edgecolor=DARK, label="BH FDR < 0.05")
    ax_marker.scatter([], [], s=65, facecolor="white", edgecolor=DARK, label="BH FDR ≥ 0.05")
    ax_marker.legend(frameon=False, loc="lower right", fontsize=7.4)
    panel_letter(ax_marker, "D", x=-0.42, y=1.10)

    ax_pathway = fig.add_subplot(grid[0, 1])
    pathway_plot = pathways.sort_values(
        ["LEC_state", "FDR_q_value"],
        ascending=[False, False],
        key=lambda col: col.map(state_number) if col.name == "LEC_state" else col,
    ).reset_index(drop=True)
    y_pathway = np.arange(len(pathway_plot))
    for position, (_, row) in enumerate(pathway_plot.iterrows()):
        color = "#B8574D" if float(row["NES"]) > 0 else NAVY
        significant = bool(row["pathway_fdr_025"])
        ax_pathway.hlines(position, 0, row["NES"], color=color, linewidth=1.3)
        ax_pathway.scatter(
            row["NES"],
            position,
            s=66,
            facecolor=color if significant else "white",
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )
        ax_pathway.text(
            2.38,
            position,
            f"q={row['FDR_q_value']:.3f}",
            ha="right",
            va="center",
            fontsize=7.2,
            color=SLATE,
        )
    labels = [
        textwrap.fill(pathway_label(row), width=44)
        for _, row in pathway_plot.iterrows()
    ]
    ax_pathway.axvline(0, color="#AEB6BD", linewidth=0.8)
    ax_pathway.set_yticks(y_pathway, labels)
    ax_pathway.set_xlim(-2.45, 2.45)
    ax_pathway.set_xlabel("Normalized enrichment score (NES)")
    ax_pathway.set_title(
        "Additional pathway evidence\nTwo lowest-FDR pathways not shown in Figure 2",
        loc="left",
        pad=8,
    )
    clean_axis(ax_pathway, "x")
    ax_pathway.scatter([], [], s=55, facecolor=DARK, edgecolor=DARK, label="FDR ≤ 0.25")
    ax_pathway.scatter([], [], s=55, facecolor="white", edgecolor=DARK, label="FDR > 0.25")
    ax_pathway.legend(frameon=False, loc="lower right", fontsize=7.4)
    return [ax_marker, ax_pathway]


def draw_panel_e(
    fig: plt.Figure,
    spec,
    frame: pd.DataFrame,
    colors: dict[str, str],
) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 2, wspace=0.13)
    axes: list[plt.Axes] = []
    limits = (-28, 53)
    for index, contrast in enumerate(["Q2+Q3 − Q1", "Q4 − Q1"]):
        ax = fig.add_subplot(grid[0, index])
        subset = frame.loc[frame["contrast"].eq(contrast)].sort_values("state_number")
        ax.plot(limits, limits, color="#AEB6BD", linewidth=1.0, linestyle=(0, (3, 2)))
        for _, row in subset.iterrows():
            state = row["LEC_state"]
            ax.scatter(
                row["original_effect_pp"],
                row["leaveout_effect_pp"],
                s=58,
                color=colors[state],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            ax.annotate(
                str(row["state_number"]),
                (row["original_effect_pp"], row["leaveout_effect_pp"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7.2,
                color=DARK,
            )
        ax.axhline(0, color="#D0D5D9", linewidth=0.7)
        ax.axvline(0, color="#D0D5D9", linewidth=0.7)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Original donor-level effect (pp)")
        ax.set_title(contrast, loc="left", pad=8)
        clean_axis(ax, "both")
        if index == 0:
            ax.set_ylabel("Effect after removing state-specific top donor (pp)")
            panel_letter(ax, "E", x=-0.28, y=1.12)
        else:
            ax.set_yticklabels([])
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
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    marker_data: pd.DataFrame,
    pathway_data: pd.DataFrame,
    panel_e: pd.DataFrame,
    colors: dict[str, str],
) -> None:
    fig = plt.figure(figsize=(7.5, 5.8), constrained_layout=True)
    draw_panel_a(fig, GridSpec(1, 1, figure=fig)[0], panel_a, colors)
    save_figure(fig, "FigS03A_continuous_age_associations")

    fig = plt.figure(figsize=(7.7, 5.8), constrained_layout=True)
    draw_panel_b(fig, GridSpec(1, 1, figure=fig)[0], panel_b)
    save_figure(fig, "FigS03B_single_donor_contribution")

    fig = plt.figure(figsize=(13.6, 4.7), constrained_layout=True)
    draw_panel_c(fig, GridSpec(1, 1, figure=fig)[0], panel_c, colors)
    save_figure(fig, "FigS03C_less_abundant_state_characterization")

    fig = plt.figure(figsize=(16.5, 6.4), constrained_layout=True)
    draw_panel_d(
        fig,
        GridSpec(1, 1, figure=fig)[0],
        marker_data,
        pathway_data,
        colors,
    )
    save_figure(fig, "FigS03D_additional_marker_pathway_evidence")

    fig = plt.figure(figsize=(11.5, 5.4), constrained_layout=True)
    draw_panel_e(fig, GridSpec(1, 1, figure=fig)[0], panel_e, colors)
    save_figure(fig, "FigS03E_leave_top_donor_out_sensitivity")


def make_combined_figure(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    marker_data: pd.DataFrame,
    pathway_data: pd.DataFrame,
    panel_e: pd.DataFrame,
    colors: dict[str, str],
) -> None:
    fig = plt.figure(figsize=(17.0, 22.0), constrained_layout=True)
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.15, 0.86, 1.18, 1.0],
        width_ratios=[0.82, 1.18],
        hspace=0.16,
        wspace=0.13,
    )
    draw_panel_a(fig, grid[0, 0], panel_a, colors)
    draw_panel_b(fig, grid[0, 1], panel_b)
    draw_panel_c(fig, grid[1, :], panel_c, colors)
    draw_panel_d(fig, grid[2, :], marker_data, pathway_data, colors)
    draw_panel_e(fig, grid[3, :], panel_e, colors)
    fig.suptitle(
        "Figure S3 | Extended aging LEC-state characterization and robustness",
        fontsize=18,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    save_figure(fig, "FigS03_extended_aging_LEC_state_characterization")


def write_traceability(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    marker_data: pd.DataFrame,
    pathway_data: pd.DataFrame,
    panel_e: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for panel, record_type, frame in [
        ("A", "continuous_age_association", panel_a),
        ("B", "single_donor_contribution", panel_b),
        ("C", "rare_state_donor_abundance", panel_c),
        ("D", "rare_state_ranked_marker", marker_data),
        ("D", "additional_rare_state_pathway", pathway_data),
        ("E", "leave_top_donor_out_effect", panel_e),
    ]:
        output = frame.copy()
        output.insert(0, "panel", panel)
        output.insert(1, "record_type", record_type)
        frames.append(output)
    pd.concat(frames, ignore_index=True, sort=False).to_csv(
        OUT_DIR / "FigS03_source_data.csv", index=False
    )

    panel_map = {
        "continuous_age": "A",
        "donor_dominance": "B",
        "donor_state_abundance": "C; E",
        "leave_top_donor_out": "E",
        "rare_state_markers": "D",
        "gsea_top_pathways": "D",
        "main_fig02c_source": "D exclusion check",
        "state_colors": "A; C; D; E",
    }
    purpose_map = {
        "continuous_age": "All-state donor-level continuous-age associations after the primary ≥25-LEC filter",
        "donor_dominance": "Maximum single-donor contribution within each age-group/state combination",
        "donor_state_abundance": "Donor-level rare-state distributions and leave-top-donor-out effect calculations",
        "leave_top_donor_out": "State-specific donor selected for the influence sensitivity analysis",
        "rare_state_markers": "Objective top-three ranked marker selection for States 7–10",
        "gsea_top_pathways": "Candidate additional GSEA pathways for States 7–10",
        "main_fig02c_source": "Exclude pathways already displayed in main Figure 2",
        "state_colors": "Maintain the established aging-state color mapping",
    }
    manifest = pd.DataFrame(
        [
            {
                "input_id": input_id,
                "panels": panel_map[input_id],
                "input_csv": str(path.relative_to(ROOT)),
                "purpose": purpose_map[input_id],
            }
            for input_id, path in INPUTS.items()
        ]
    )
    manifest.to_csv(OUT_DIR / "FigS03_input_manifest.csv", index=False)


def main() -> None:
    configure_style()
    data = read_inputs()
    colors = data["state_colors"].set_index("LEC_state")["color_hex"].to_dict()
    panel_a = build_panel_a(data)
    panel_b = build_panel_b(data)
    panel_c = build_panel_c(data)
    marker_data, pathway_data = build_panel_d(data)
    panel_e = build_panel_e(data)
    validate_tables(
        data,
        panel_a,
        panel_b,
        panel_c,
        marker_data,
        pathway_data,
        panel_e,
    )
    write_traceability(
        panel_a,
        panel_b,
        panel_c,
        marker_data,
        pathway_data,
        panel_e,
    )
    make_individual_panels(
        panel_a,
        panel_b,
        panel_c,
        marker_data,
        pathway_data,
        panel_e,
        colors,
    )
    make_combined_figure(
        panel_a,
        panel_b,
        panel_c,
        marker_data,
        pathway_data,
        panel_e,
        colors,
    )
    print(f"Supplementary Figure S3 written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
