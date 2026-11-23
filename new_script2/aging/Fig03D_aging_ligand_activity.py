"""Create aging Figure 3D: prioritized ligand activity."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = (
    ROOT
    / "new_result2_aging"
    / "13_NicheNet_donor_pseudobulk"
    / "dataset13"
)
TABLE_DIR = INPUT_DIR / "tables"
OUTPUT_DIR = ROOT / "new_result2" / "aging" / "Fig03D_aging_ligand_activity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITY_CSV = TABLE_DIR / "13_Sydney_equivalent_ligand_activities.csv"
INTERACTION_CSV = (
    TABLE_DIR / "13_MultiNicheNet_top50_sender_to_LEC_per_group.csv"
)
CELLTYPE_MAP_CSV = TABLE_DIR / "13_MultiNicheNet_celltype_name_map.csv"
DONOR_CSV = TABLE_DIR / "13_selected_Q1_Q4_donors.csv"

FIGURE_PNG = OUTPUT_DIR / "Fig03D_aging_ligand_activity.png"
FIGURE_PDF = OUTPUT_DIR / "Fig03D_aging_ligand_activity.pdf"
FIGURE_JPG = OUTPUT_DIR / "Fig03D_aging_ligand_activity.jpg"
SOURCE_CSV = OUTPUT_DIR / "Fig03D_source_data.csv"
MANIFEST_CSV = OUTPUT_DIR / "Fig03D_input_manifest.csv"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


DOWN_COLOR = "#3F78A8"
UP_COLOR = "#C65F43"
TEXT_COLOR = "#202428"
GRID_COLOR = "#D9DEE2"
EDGE_COLOR = "#76818A"

SENDER_ORDER = [
    "EC5_art",
    "EC7_endocardial",
    "LYVE1.IGF1.MP",
    "SMC1_basic",
    "Mast",
    "NC1_glial",
]
SENDER_LABELS = {
    "EC5_art": "Arterial\nECs",
    "EC7_endocardial": "Endocardial\nECs",
    "LYVE1.IGF1.MP": "LYVE1+IGF1\nmacrophages",
    "SMC1_basic": "SMCs",
    "Mast": "Mast\ncells",
    "NC1_glial": "Glial\ncells",
}
SENDER_COLORS = {
    "EC5_art": "#BFD7EA",
    "EC7_endocardial": "#A8D5BA",
    "LYVE1.IGF1.MP": "#F1CF9B",
    "SMC1_basic": "#D7C4E8",
    "Mast": "#EDB7B7",
    "NC1_glial": "#C8D1D6",
}


def require_columns(data: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(data.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    activity = pd.read_csv(ACTIVITY_CSV)
    interactions = pd.read_csv(INTERACTION_CSV)
    donors = pd.read_csv(DONOR_CSV)
    celltype_map = pd.read_csv(CELLTYPE_MAP_CSV)

    require_columns(
        activity,
        {"test_ligand", "aupr_corrected", "direction"},
        ACTIVITY_CSV.name,
    )
    require_columns(
        interactions,
        {
            "group",
            "sender",
            "receiver",
            "ligand",
            "receptor",
            "prioritization_score",
            "max_scaled_activity",
            "scaled_pb_ligand",
            "scaled_pb_receptor",
        },
        INTERACTION_CSV.name,
    )
    require_columns(
        donors,
        {"final_four_group", "n_final_LEC_cells", "retained_min25_LECs"},
        DONOR_CSV.name,
    )
    require_columns(
        celltype_map,
        {"analysis_celltype", "celltype_mnn"},
        CELLTYPE_MAP_CSV.name,
    )

    retained = donors["retained_min25_LECs"].astype(str).str.lower().eq("true")
    group_counts = donors["final_four_group"].value_counts().to_dict()
    if len(donors) != 9 or group_counts.get("Q1") != 4 or group_counts.get("Q4") != 5:
        raise ValueError("Expected the post-filter donor design Q1=4 and Q4=5.")
    if (pd.to_numeric(donors["n_final_LEC_cells"]) < 25).any() or not retained.all():
        raise ValueError("At least one Figure 3 donor fails the >=25 final-LEC rule.")

    mapped_celltypes = set(celltype_map["celltype_mnn"].astype(str))
    if not set(SENDER_ORDER).issubset(mapped_celltypes):
        raise ValueError("A requested sender is absent from the cell-type name map.")

    activity["aupr_corrected"] = pd.to_numeric(
        activity["aupr_corrected"], errors="coerce"
    )
    activity = activity.dropna(subset=["test_ligand", "direction", "aupr_corrected"])
    interactions["prioritization_score"] = pd.to_numeric(
        interactions["prioritization_score"], errors="coerce"
    )
    interactions["max_scaled_activity"] = pd.to_numeric(
        interactions["max_scaled_activity"], errors="coerce"
    )
    interactions = interactions.dropna(
        subset=[
            "sender",
            "receiver",
            "ligand",
            "receptor",
            "prioritization_score",
            "max_scaled_activity",
        ]
    )
    return activity, interactions, donors


def select_panel_a(activity: pd.DataFrame) -> pd.DataFrame:
    directions = ["Q4_downregulated", "Q4_upregulated"]
    ranked = activity[activity["direction"].isin(directions)].copy()
    ranked = ranked.sort_values(
        ["direction", "aupr_corrected"], ascending=[True, False]
    )
    ranked["rank_within_direction"] = (
        ranked.groupby("direction").cumcount() + 1
    )

    # Start with the top eight in each direction, then keep each ligand only in
    # the direction where it has the better rank. This gives 15 non-redundant
    # entries for the current table (VEGFC is shared and ranks better for Q4-up).
    candidates = ranked[ranked["rank_within_direction"] <= 8].copy()
    selected = (
        candidates.sort_values(
            ["test_ligand", "rank_within_direction", "aupr_corrected"],
            ascending=[True, True, False],
        )
        .drop_duplicates("test_ligand", keep="first")
        .copy()
    )

    if len(selected) < 15:
        additions = ranked[~ranked["test_ligand"].isin(selected["test_ligand"])]
        additions = additions.sort_values(
            ["rank_within_direction", "aupr_corrected"],
            ascending=[True, False],
        )
        selected = pd.concat(
            [selected, additions.head(15 - len(selected))], ignore_index=True
        )
    if len(selected) > 15:
        selected = selected.sort_values(
            ["rank_within_direction", "aupr_corrected"],
            ascending=[True, False],
        ).head(15)

    if len(selected) != 15 or selected["test_ligand"].nunique() != 15:
        raise ValueError(
            "Panel D selection must contain 15 ligand entries without duplicated names."
        )

    direction_labels = {
        "Q4_downregulated": "Q4-downregulated LEC genes",
        "Q4_upregulated": "Q4-upregulated LEC genes",
    }
    selected["direction_label"] = selected["direction"].map(direction_labels)
    selected["highlighted_example"] = selected["test_ligand"].eq("VEGFC")
    return selected.sort_values(
        ["direction", "rank_within_direction"], ascending=[True, True]
    ).reset_index(drop=True)


def select_panels_b_c(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    finite = interactions[
        interactions["sender"].isin(SENDER_ORDER)
        & interactions["receiver"].eq("LEC_EC8_ln")
    ].copy()
    if finite.empty:
        raise ValueError("No finite sender-to-LEC interactions are available.")

    ligand_ranking = (
        finite.groupby("ligand", as_index=False)["prioritization_score"]
        .max()
        .sort_values("prioritization_score", ascending=False)
        .head(8)
    )
    ligand_order = ligand_ranking["ligand"].tolist()
    if len(ligand_order) != 8:
        raise ValueError("Panel E requires exactly eight prioritized ligands.")

    selected = finite[finite["ligand"].isin(ligand_order)].copy()
    observed_senders = set(selected["sender"])
    if observed_senders != set(SENDER_ORDER):
        raise ValueError(
            "The eight selected ligands must connect exactly the six requested senders."
        )

    sender_ligand = (
        selected.sort_values("prioritization_score", ascending=False)
        .groupby(["sender", "ligand"], as_index=False)
        .agg(
            prioritization_score=("prioritization_score", "max"),
            max_scaled_activity=("max_scaled_activity", "max"),
            receptors=("receptor", lambda x: ";".join(sorted(set(map(str, x))))),
            group=("group", "first"),
            receiver=("receiver", "first"),
        )
    )
    sender_ligand["sender_label"] = sender_ligand["sender"].map(
        lambda x: SENDER_LABELS[x].replace("\n", " ")
    )

    receptor_edges = (
        selected.sort_values("prioritization_score", ascending=False)
        .drop_duplicates(["sender", "ligand", "receptor"], keep="first")
        .copy()
    )
    receptor_edges["sender_label"] = receptor_edges["sender"].map(
        lambda x: SENDER_LABELS[x].replace("\n", " ")
    )
    receptor_edges["ligand_receptor"] = (
        receptor_edges["ligand"].astype(str)
        + "\N{EN DASH}"
        + receptor_edges["receptor"].astype(str)
    )
    return sender_ligand, receptor_edges, ligand_order


def style_bar_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors="#525A60")
    ax.set_axisbelow(True)


def draw_panel_a(
    fig: plt.Figure,
    outer_slot,
    data: pd.DataFrame,
) -> list[plt.Axes]:
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_slot, wspace=0.56)
    axes: list[plt.Axes] = []
    configs = [
        ("Q4_downregulated", DOWN_COLOR, "Q4-downregulated\nLEC gene program"),
        ("Q4_upregulated", UP_COLOR, "Q4-upregulated\nLEC gene program"),
    ]
    for index, (direction, color, title) in enumerate(configs):
        ax = fig.add_subplot(inner[0, index])
        subset = data[data["direction"].eq(direction)].sort_values(
            "rank_within_direction", ascending=True
        )
        y = np.arange(len(subset))
        bars = ax.barh(
            y,
            subset["aupr_corrected"],
            color=color,
            edgecolor="none",
            height=0.68,
            zorder=2,
        )
        ax.set_yticks(y, subset["test_ligand"])
        ax.invert_yaxis()
        max_value = max(float(subset["aupr_corrected"].max()), 1e-6)
        ax.set_xlim(0, max_value * 1.24)
        ax.set_xlabel("Corrected AUPR")
        ax.set_title(title, loc="left", fontweight="bold", color=color, pad=8)
        style_bar_axis(ax)

        for row_number, (bar, (_, row)) in enumerate(zip(bars, subset.iterrows())):
            value = float(row["aupr_corrected"])
            ax.text(
                value + max_value * 0.025,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=7.5,
                color="#4E565C",
            )
            tick = ax.get_yticklabels()[row_number]
            tick.set_fontstyle("italic")
            if bool(row["highlighted_example"]):
                bar.set_edgecolor(TEXT_COLOR)
                bar.set_linewidth(1.4)
                tick.set_fontweight("bold")
                tick.set_color(TEXT_COLOR)
        axes.append(ax)
    return axes


def add_node(
    ax: plt.Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    label: str,
    facecolor: str,
    edgecolor: str = "#4D5961",
    fontsize: float = 8.2,
    italic: bool = False,
) -> None:
    x, y = center
    node = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
        zorder=4,
    )
    ax.add_patch(node)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="semibold",
        fontstyle="italic" if italic else "normal",
        color=TEXT_COLOR,
        zorder=5,
        linespacing=1.05,
    )


def add_network_edge(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    linewidth: float,
    color: str,
    alpha: float,
    rad: float,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=1,
    )
    ax.add_patch(arrow)


def draw_panel_b(
    ax: plt.Axes,
    edges: pd.DataFrame,
    ligand_order: list[str],
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    sender_y = dict(zip(SENDER_ORDER, np.linspace(0.84, 0.16, len(SENDER_ORDER))))
    ligand_y = dict(zip(ligand_order, np.linspace(0.88, 0.12, len(ligand_order))))
    sender_x, ligand_x, receiver_x = 0.14, 0.52, 0.89

    score_min = float(edges["prioritization_score"].min())
    score_max = float(edges["prioritization_score"].max())
    score_span = max(score_max - score_min, 1e-9)

    for _, row in edges.iterrows():
        score = float(row["prioritization_score"])
        width = 0.7 + 2.0 * (score - score_min) / score_span
        dy = ligand_y[row["ligand"]] - sender_y[row["sender"]]
        add_network_edge(
            ax,
            (sender_x + 0.115, sender_y[row["sender"]]),
            (ligand_x - 0.075, ligand_y[row["ligand"]]),
            width,
            EDGE_COLOR,
            0.42,
            float(np.clip(dy * 0.11, -0.08, 0.08)),
        )

    ligand_scores = edges.groupby("ligand")["prioritization_score"].max()
    for ligand in ligand_order:
        score = float(ligand_scores.loc[ligand])
        width = 0.8 + 2.1 * (score - score_min) / score_span
        dy = 0.5 - ligand_y[ligand]
        add_network_edge(
            ax,
            (ligand_x + 0.075, ligand_y[ligand]),
            (receiver_x - 0.065, 0.5),
            width,
            "#A27743",
            0.50,
            float(np.clip(dy * 0.12, -0.08, 0.08)),
        )

    for sender in SENDER_ORDER:
        add_node(
            ax,
            (sender_x, sender_y[sender]),
            0.23,
            0.078,
            SENDER_LABELS[sender],
            SENDER_COLORS[sender],
            fontsize=7.6,
        )
    for ligand in ligand_order:
        add_node(
            ax,
            (ligand_x, ligand_y[ligand]),
            0.145,
            0.058,
            ligand,
            "#F4DFC1",
            edgecolor="#8A6841",
            fontsize=8.0,
            italic=True,
        )
    add_node(
        ax,
        (receiver_x, 0.5),
        0.125,
        0.12,
        "LEC\nreceiver",
        "#DCEAF3",
        edgecolor="#3D6A82",
        fontsize=8.5,
    )

    ax.text(sender_x, 0.98, "Sender cells (6)", ha="center", va="top", fontweight="bold")
    ax.text(ligand_x, 0.98, "Ligands (8)", ha="center", va="top", fontweight="bold")
    ax.text(receiver_x, 0.98, "Receiver", ha="center", va="top", fontweight="bold")
    ax.text(
        0.5,
        0.015,
        "Finite top-ranked interactions are Q1-prioritized; line width reflects priority score.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#596168",
    )


def draw_panel_c(
    fig: plt.Figure,
    ax: plt.Axes,
    edges: pd.DataFrame,
    ligand_order: list[str],
) -> None:
    pair_order: list[str] = []
    for ligand in ligand_order:
        ligand_rows = edges[edges["ligand"].eq(ligand)]
        ordered_pairs = (
            ligand_rows.groupby("ligand_receptor")["prioritization_score"]
            .max()
            .sort_values(ascending=False)
            .index.tolist()
        )
        pair_order.extend(ordered_pairs)

    # A transposed layout (sender rows, interaction columns) produces compact,
    # readable cells instead of the very wide sparse cells created by six
    # sender columns and twelve interaction rows.
    matrix = (
        edges.pivot_table(
            index="sender",
            columns="ligand_receptor",
            values="prioritization_score",
            aggfunc="max",
        )
        .reindex(index=SENDER_ORDER, columns=pair_order)
    )
    values = matrix.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    vmin = min(0.78, float(finite_values.min()))
    vmax = max(0.88, float(finite_values.max()))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps["Blues"].copy()
    ax.set_facecolor("white")
    ax.set_xlim(-0.5, len(pair_order) - 0.5)
    ax.set_ylim(len(SENDER_ORDER) - 0.5, -0.5)

    ax.set_xticks(np.arange(len(pair_order)))
    ax.set_xticklabels(
        pair_order,
        rotation=42,
        ha="right",
        va="top",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks(np.arange(len(SENDER_ORDER)))
    ax.set_yticklabels([SENDER_LABELS[s] for s in SENDER_ORDER])
    ax.tick_params(axis="y", length=0, pad=12)
    for label, sender in zip(ax.get_yticklabels(), SENDER_ORDER):
        label.set_bbox(
            {
                "boxstyle": "round,pad=0.30",
                "facecolor": SENDER_COLORS[sender],
                "edgecolor": "#68737B",
                "linewidth": 0.7,
            }
        )

    emphasized_pairs = {
        "VEGFC\N{EN DASH}NRP2",
        "TGM2\N{EN DASH}ITGA9",
        "TGM2\N{EN DASH}SDC4",
        "HBEGF\N{EN DASH}ERBB4",
        "HBEGF\N{EN DASH}CD44",
        "HBEGF\N{EN DASH}CD9",
    }
    for label in ax.get_xticklabels():
        if label.get_text() in emphasized_pairs:
            label.set_fontweight("bold")

    ax.set_xticks(np.arange(-0.5, len(pair_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SENDER_ORDER), 1), minor=True)
    ax.grid(which="minor", color="#D9E0E4", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Slightly stronger separators identify ligand blocks while preserving a
    # single coherent heatmap.
    column_cursor = 0
    for ligand in ligand_order[:-1]:
        column_cursor += sum(
            pair.startswith(f"{ligand}\N{EN DASH}") for pair in pair_order
        )
        ax.axvline(column_cursor - 0.5, color="#AEB7BD", linewidth=1.25)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Encode every observed sender-by-interaction value as an inset tile.  The
    # surrounding white cell and visible grid make clear that the value belongs
    # only to this exact matrix entry and has no implication for adjacent cells.
    threshold = (vmin + vmax) / 2
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if np.isfinite(value):
                ax.add_patch(
                    FancyBboxPatch(
                        (column_index - 0.36, row_index - 0.31),
                        0.72,
                        0.62,
                        boxstyle="round,pad=0.01,rounding_size=0.08",
                        facecolor=cmap(norm(value)),
                        edgecolor="#C7D0D6",
                        linewidth=0.55,
                        zorder=2,
                    )
                )
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="semibold",
                    color="white" if value >= threshold else "#263238",
                    zorder=3,
                )

    score_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    score_map.set_array([])
    cbar = fig.colorbar(
        score_map,
        ax=ax,
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("MultiNicheNet prioritization score", rotation=90, labelpad=10)
    cbar.outline.set_visible(False)
    ax.set_xlabel("Ligand\N{EN DASH}receptor interaction", labelpad=16)
    ax.set_ylabel("Predicted sender cell type", labelpad=16)


def build_source_data(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
) -> pd.DataFrame:
    a = pd.DataFrame(
        {
            "panel": "D",
            "record_type": "ligand_activity",
            "direction": panel_a["direction"],
            "rank_within_direction": panel_a["rank_within_direction"],
            "sender": "",
            "sender_label": "",
            "ligand": panel_a["test_ligand"],
            "receptor": "",
            "receiver": "LEC_EC8_ln",
            "activity_metric": "aupr_corrected",
            "activity_value": panel_a["aupr_corrected"],
            "prioritization_score": np.nan,
            "ligand_activity_scaled": np.nan,
            "source_csv": ACTIVITY_CSV.relative_to(ROOT).as_posix(),
            "selection_note": np.where(
                panel_a["highlighted_example"],
                "template example that ranks within selected activity entries",
                "top-ranked ligand entry",
            ),
        }
    )
    b = pd.DataFrame(
        {
            "panel": "E",
            "record_type": "sender_ligand_network_edge",
            "direction": panel_b["group"].map(lambda x: f"{x}_prioritized"),
            "rank_within_direction": np.nan,
            "sender": panel_b["sender"],
            "sender_label": panel_b["sender_label"],
            "ligand": panel_b["ligand"],
            "receptor": panel_b["receptors"],
            "receiver": panel_b["receiver"],
            "activity_metric": "",
            "activity_value": np.nan,
            "prioritization_score": panel_b["prioritization_score"],
            "ligand_activity_scaled": panel_b["max_scaled_activity"],
            "source_csv": INTERACTION_CSV.relative_to(ROOT).as_posix(),
            "selection_note": "six-sender/eight-ligand simplified network",
        }
    )
    c = pd.DataFrame(
        {
            "panel": "F",
            "record_type": "ligand_receptor_interaction",
            "direction": panel_c["group"].map(lambda x: f"{x}_prioritized"),
            "rank_within_direction": np.nan,
            "sender": panel_c["sender"],
            "sender_label": panel_c["sender_label"],
            "ligand": panel_c["ligand"],
            "receptor": panel_c["receptor"],
            "receiver": panel_c["receiver"],
            "activity_metric": "",
            "activity_value": np.nan,
            "prioritization_score": panel_c["prioritization_score"],
            "ligand_activity_scaled": panel_c["max_scaled_activity"],
            "source_csv": INTERACTION_CSV.relative_to(ROOT).as_posix(),
            "selection_note": "receptor-resolved edge for one of the eight selected ligands",
        }
    )
    return pd.concat([a, b, c], ignore_index=True)


def write_manifest(donors: pd.DataFrame) -> None:
    group_counts = donors["final_four_group"].value_counts().to_dict()
    records = [
        {
            "figure_panels": "D",
            "input_csv": ACTIVITY_CSV.relative_to(ROOT).as_posix(),
            "use": "bidirectional ligand activity ranked by corrected AUPR",
            "filter_or_selection": (
                "exploratory donor-edgeR visualization layer; top eight per direction, "
                "duplicate ligand names retained only in their better-ranked direction; 15 entries total"
            ),
        },
        {
            "figure_panels": "D provenance",
            "input_csv": DONOR_CSV.relative_to(ROOT).as_posix(),
            "use": "verify the post-filter donor population",
            "filter_or_selection": (
                f">=25 final LECs per donor; Q1={group_counts.get('Q1', 0)}, "
                f"Q4={group_counts.get('Q4', 0)}; total={len(donors)}"
            ),
        },
    ]
    pd.DataFrame(records).to_csv(MANIFEST_CSV, index=False)


def main() -> None:
    activity, interactions, donors = load_and_validate()
    panel_a = select_panel_a(activity)
    panel_b, panel_c, ligand_order = select_panels_b_c(interactions)
    build_source_data(panel_a, panel_b, panel_c).query("panel == 'D'").to_csv(
        SOURCE_CSV, index=False
    )
    write_manifest(donors)
    fig = plt.figure(figsize=(9.6, 6.8), facecolor="white")
    grid = GridSpec(
        1,
        1,
        figure=fig,
        left=0.10,
        right=0.97,
        top=0.76,
        bottom=0.16,
    )
    draw_panel_a(fig, grid[0, 0], panel_a)
    fig.text(0.025, 0.96, "D", fontsize=18, fontweight="bold", color=TEXT_COLOR, va="top")
    fig.text(
        0.08,
        0.96,
        "Prioritized ligand activity with aging",
        fontsize=17,
        fontweight="bold",
        color=TEXT_COLOR,
        va="top",
    )
    fig.text(
        0.08,
        0.90,
        "Exploratory receiver-gene layer (FDR ≤ 0.25); bars show corrected AUPR.",
        fontsize=9,
        color="#5A6268",
        va="top",
    )
    min_cells = int(pd.to_numeric(donors["n_final_LEC_cells"]).min())
    fig.text(
        0.08,
        0.027,
        (
            f"Donor-aware Q1 vs Q4 analysis after the \N{GREATER-THAN OR EQUAL TO}{25}-LEC filter: "
            f"Q1 n=4, Q4 n=5 (minimum {min_cells} LECs/donor)."
        ),
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#555E64",
    )

    fig.savefig(FIGURE_PNG, dpi=450, facecolor="white")
    fig.savefig(FIGURE_PDF, facecolor="white")
    fig.savefig(
        FIGURE_JPG,
        dpi=450,
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(fig)

    print(f"Figure 3D written to: {OUTPUT_DIR}")
    print(f"Panel D ligand entries: {len(panel_a)}")
    print(
        "Post-filter donors: "
        + "; ".join(
            f"{group}={count}"
            for group, count in donors["final_four_group"].value_counts().sort_index().items()
        )
    )


if __name__ == "__main__":
    main()
