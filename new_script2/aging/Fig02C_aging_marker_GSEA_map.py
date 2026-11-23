"""Create the integrated aging Figure 2 panel C from Step08/GSEA results.

The state-marker dot plot, proposed definitions, and exact Aging-sheet
pathways are aligned to shared state boundaries as one compound panel.

The full 1,119-cell Step08 object is used so these state-definition panels
remain consistent with Figure 2A. This is descriptive state characterization,
not the later donor >=25-LEC aging-inference subset.
"""

from __future__ import annotations

from pathlib import Path
import re
import textwrap

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
STEP08_DIR = (
    ROOT
    / "new_result2_aging"
    / "08_combined_LEC_recluster"
    / "dataset13"
)
H5AD = STEP08_DIR / "08_combined_final_LEC_reclustered.h5ad"
STATE_COUNTS_CSV = STEP08_DIR / "08_state_counts.csv"
GSEA_XLSX = ROOT / "FigureGSEAPathways.xlsx"

OUTPUT_DIR = ROOT / "new_result2" / "aging" / "Fig02C_aging_marker_GSEA_map"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_PNG = OUTPUT_DIR / "Fig02C_aging_marker_GSEA_map.png"
FIGURE_PDF = OUTPUT_DIR / "Fig02C_aging_marker_GSEA_map.pdf"
FIGURE_JPG = OUTPUT_DIR / "Fig02C_aging_marker_GSEA_map.jpg"
SOURCE_CSV = OUTPUT_DIR / "Fig02C_source_data.csv"
MANIFEST_CSV = OUTPUT_DIR / "Fig02C_input_manifest.csv"

EXPECTED_N_CELLS = 1119
STATES = list(range(11))
STATE_KEYS = [f"state{state}" for state in STATES]


# Exact genes and state definitions transcribed from the user-supplied table.
# HLA-A/C/E is represented by the three genes HLA-A, HLA-C and HLA-E.
MARKERS: dict[int, list[str]] = {
    0: ["CCL21", "MMRN1", "TFPI", "TIMP3", "HLA-A", "HLA-C", "HLA-E", "B2M"],
    1: ["IGF1", "CALCRL", "NPNT", "HMCN1"],
    2: ["RELN", "CD36", "PIEZO2", "MYH10"],
    3: ["MYH7", "MYL2", "TPM1", "TNNT2"],
    4: ["ADAMTS9", "NAMPT", "AKAP12"],
    5: ["TLL1", "TFPI", "LRRC1", "PARD6G"],
    6: ["MECOM", "EMCN", "FLT1", "VWF"],
    7: [],
    8: [],
    9: [],
    10: [],
}

STATE_DEFINITIONS = {
    0: "Homeostatic immune-surveillance LEC",
    1: "Matrix-remodeling and signaling LEC",
    2: "Structural / remodeling / mechanosensory LEC",
    3: "Differentiated contractile-associated LEC",
    4: "Cytokine-responsive immune LEC",
    5: "Polarity-associated LEC",
    6: "Activated endothelial LEC",
    7: "Rare stress-associated LEC",
    8: "Rare regulatory LEC",
    9: "Rare transitional LEC",
    10: "Rare mature LEC",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

TEXT = "#202428"
SUBTEXT = "#5C646A"
BOUNDARY = "#99A2A8"
GRID = "#E2E6E9"
POSITIVE = "#B84B3D"
NEGATIVE = "#356F9F"
RARE_BAND = "#F0F2F3"


def state_number(value: object) -> int:
    match = re.fullmatch(r"state(\d+)", str(value).strip().lower())
    if match is None:
        raise ValueError(f"Unexpected state label: {value!r}")
    return int(match.group(1))


def clean_text(value: object) -> str:
    text = str(value).replace("\u200b", "").strip()
    replacements = {
        "TNF��/NF��B": "TNFα/NF-κB",
        "IFN��": "IFNγ",
        "IFN-��": "IFN-γ",
        "TGF��": "TGFβ",
        "WNT/��-catenin": "WNT/β-catenin",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def display_pathway_name(value: object) -> str:
    """Keep the pathway term exact while making the database prefix readable."""
    text = clean_text(value)
    prefixes = [
        ("KEGG_MEDICUS_REFERENCE_", "KEGG: "),
        ("REACTOME_", "Reactome: "),
        ("HALLMARK_", "Hallmark: "),
        ("GOBP_", "GO BP: "),
    ]
    for prefix, replacement in prefixes:
        if text.startswith(prefix):
            text = replacement + text[len(prefix) :]
            break
    return text.replace("_", " ")


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_gsea() -> tuple[pd.DataFrame, int]:
    gsea = pd.read_excel(GSEA_XLSX, sheet_name="Aging")
    require_columns(
        gsea,
        {"State", "In-Figure Annotation", "Pathway Name", "NES", "FDR"},
        f"{GSEA_XLSX.name}:Aging",
    )
    gsea = gsea.copy()
    gsea["State"] = pd.to_numeric(gsea["State"], errors="raise").astype(int)
    gsea["NES"] = pd.to_numeric(gsea["NES"], errors="raise")
    gsea["FDR"] = pd.to_numeric(gsea["FDR"], errors="raise")
    if sorted(gsea["State"].unique().tolist()) != STATES:
        raise ValueError("The Aging GSEA sheet does not contain exactly States 0-10.")
    if gsea[["Pathway Name", "NES", "FDR"]].isna().any().any():
        raise ValueError("The Aging GSEA sheet contains missing pathway evidence.")

    gsea["annotation_display"] = gsea["In-Figure Annotation"].map(clean_text)
    gsea["pathway_display"] = gsea["Pathway Name"].map(display_pathway_name)
    duplicate_mask = gsea.duplicated(
        subset=["State", "Pathway Name", "NES", "FDR"], keep="first"
    )
    n_duplicates = int(duplicate_mask.sum())
    gsea = gsea.loc[~duplicate_mask].reset_index(drop=True)
    gsea["direction"] = np.where(gsea["NES"] >= 0, "positive", "negative")
    gsea["fdr_for_size"] = gsea["FDR"].clip(lower=1e-4)
    gsea["minus_log10_fdr"] = -np.log10(gsea["fdr_for_size"])
    gsea["passes_exploratory_fdr_0.25"] = gsea["FDR"] <= 0.25
    return gsea, n_duplicates


def load_expression() -> tuple[pd.DataFrame, pd.DataFrame]:
    state_counts = pd.read_csv(STATE_COUNTS_CSV)
    require_columns(
        state_counts,
        {"LEC_state", "n_cells", "percent_of_all_LECs"},
        STATE_COUNTS_CSV.name,
    )
    state_counts["state_number"] = state_counts["LEC_state"].map(state_number)
    state_counts = state_counts.sort_values("state_number")
    if state_counts["state_number"].tolist() != STATES:
        raise ValueError("The Step08 state-count CSV does not contain States 0-10.")
    if int(pd.to_numeric(state_counts["n_cells"]).sum()) != EXPECTED_N_CELLS:
        raise ValueError("The Step08 state-count CSV does not sum to 1,119 LECs.")

    adata = ad.read_h5ad(H5AD)
    try:
        if adata.n_obs != EXPECTED_N_CELLS:
            raise ValueError(f"Expected {EXPECTED_N_CELLS} LECs; found {adata.n_obs}.")
        if "LEC_state" not in adata.obs:
            raise KeyError("LEC_state is absent from the Step08 h5ad.")
        observed_states = sorted(
            adata.obs["LEC_state"].astype(str).unique(), key=state_number
        )
        if observed_states != STATE_KEYS:
            raise ValueError(f"Expected {STATE_KEYS}; found {observed_states}.")

        unique_markers = list(
            dict.fromkeys(gene for state in STATES for gene in MARKERS[state])
        )
        expression_source = adata.raw if adata.raw is not None else adata
        missing = [gene for gene in unique_markers if gene not in expression_source.var_names]
        if missing:
            raise ValueError(f"Requested marker genes absent from Step08 raw: {missing}")

        matrix = expression_source[:, unique_markers].X
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=float)
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("Step08 marker expression contains invalid values.")
        obs_states = adata.obs["LEC_state"].astype(str).to_numpy()
    finally:
        # In-memory AnnData does not require explicit file closure.
        del adata

    records: list[dict[str, object]] = []
    for comparison_state in STATES:
        mask = obs_states == f"state{comparison_state}"
        subset = matrix[mask]
        means = subset.mean(axis=0)
        percentages = (subset > 0).mean(axis=0) * 100.0
        for gene_index, gene in enumerate(unique_markers):
            records.append(
                {
                    "comparison_state": comparison_state,
                    "comparison_state_label": f"State {comparison_state}",
                    "gene": gene,
                    "n_cells_in_state": int(mask.sum()),
                    "n_cells_expressing": int((subset[:, gene_index] > 0).sum()),
                    "percent_expressing": float(percentages[gene_index]),
                    "mean_log1p_cp10k_all_cells": float(means[gene_index]),
                }
            )
    stats = pd.DataFrame(records)
    gene_min = stats.groupby("gene")["mean_log1p_cp10k_all_cells"].transform("min")
    gene_max = stats.groupby("gene")["mean_log1p_cp10k_all_cells"].transform("max")
    span = gene_max - gene_min
    stats["mean_expression_scaled_0_1_within_gene"] = np.where(
        span > 0,
        (stats["mean_log1p_cp10k_all_cells"] - gene_min) / span,
        0.0,
    )

    observed = stats.groupby("comparison_state")["n_cells_in_state"].first().to_dict()
    reported = (
        state_counts.set_index("state_number")["n_cells"].astype(int).to_dict()
    )
    if observed != reported:
        raise ValueError(f"H5AD state counts differ from CSV counts: {observed} != {reported}")
    return stats, state_counts


def build_layout(gsea: pd.DataFrame) -> tuple[dict[int, dict[str, float]], float]:
    layout: dict[int, dict[str, float]] = {}
    cursor = 0.0
    for state in STATES:
        marker_rows = max(len(MARKERS[state]), 1)
        pathway_rows = int((gsea["State"] == state).sum())
        height = float(max(4, marker_rows, pathway_rows))
        layout[state] = {
            "start": cursor,
            "end": cursor + height,
            "height": height,
            "center": cursor + height / 2,
        }
        cursor += height
    return layout, cursor


def positions_in_band(start: float, height: float, n_items: int) -> list[float]:
    if n_items < 1:
        return []
    offset = (height - n_items) / 2.0
    return [start + offset + index + 0.5 for index in range(n_items)]


def add_state_bands(
    axis: plt.Axes,
    layout: dict[int, dict[str, float]],
    total_height: float,
) -> None:
    for state in STATES:
        band = layout[state]
        if state % 2 == 1:
            axis.axhspan(
                band["start"], band["end"], facecolor="#F7F8F9", zorder=-10
            )
        if state >= 7:
            axis.axhspan(
                band["start"], band["end"], facecolor=RARE_BAND, alpha=0.50, zorder=-9
            )
        axis.axhline(band["start"], color=BOUNDARY, linewidth=0.75, zorder=8)
    axis.axhline(total_height, color=BOUNDARY, linewidth=0.75, zorder=8)
    axis.set_ylim(total_height, 0)


def dot_size(percent: np.ndarray | float) -> np.ndarray | float:
    return 12.0 + 1.25 * np.asarray(percent)


def fdr_dot_size(fdr: np.ndarray | float) -> np.ndarray | float:
    fdr_array = np.asarray(fdr, dtype=float)
    strength = np.clip(-np.log10(np.maximum(fdr_array, 1e-4)), 0.0, 4.0)
    return 20.0 + 18.0 * strength


def draw_panel_d(
    ax: plt.Axes,
    stats: pd.DataFrame,
    layout: dict[int, dict[str, float]],
    total_height: float,
) -> dict[tuple[int, str], float]:
    add_state_bands(ax, layout, total_height)
    ax.set_xlim(-0.55, 10.55)
    for x in STATES:
        ax.axvline(x, color=GRID, linewidth=0.7, zorder=-5)

    row_positions: dict[tuple[int, str], float] = {}
    y_ticks: list[float] = []
    y_labels: list[str] = []
    plot_rows: list[dict[str, float | int | str]] = []

    for defining_state in STATES:
        band = layout[defining_state]
        genes = MARKERS[defining_state]
        if genes:
            positions = positions_in_band(band["start"], band["height"], len(genes))
            for gene, y in zip(genes, positions):
                row_positions[(defining_state, gene)] = y
                y_ticks.append(y)
                y_labels.append(gene)
                gene_stats = stats[stats["gene"].eq(gene)]
                for _, row in gene_stats.iterrows():
                    plot_rows.append(
                        {
                            "x": int(row["comparison_state"]),
                            "y": y,
                            "percent": float(row["percent_expressing"]),
                            "scaled_mean": float(
                                row["mean_expression_scaled_0_1_within_gene"]
                            ),
                        }
                    )
        else:
            y = band["center"]
            y_ticks.append(y)
            y_labels.append("")
            ax.text(
                5.0,
                y,
                "rare state — no robust major-marker set supplied",
                ha="center",
                va="center",
                fontsize=9.0,
                fontstyle="italic",
                color="#70787E",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "#C8CDD1",
                    "linewidth": 0.6,
                },
                zorder=4,
            )

    plot_frame = pd.DataFrame(plot_rows)
    cmap = mpl.colormaps["Reds"]
    ax.scatter(
        plot_frame["x"],
        plot_frame["y"],
        s=dot_size(plot_frame["percent"].to_numpy()),
        c=plot_frame["scaled_mean"],
        cmap=cmap,
        norm=Normalize(0, 1),
        edgecolors="#B7BDC1",
        linewidths=0.35,
        zorder=3,
    )

    ax.set_xticks(STATES, [str(state) for state in STATES])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("LEC state", labelpad=8, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks(y_ticks, y_labels)
    ax.tick_params(axis="y", length=0, pad=6)
    for label in ax.get_yticklabels():
        if label.get_text():
            label.set_fontstyle("italic")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#B7BDC1")
    return row_positions


def draw_text_columns(
    ax_state: plt.Axes,
    ax_definition: plt.Axes,
    ax_pathway: plt.Axes,
    layout: dict[int, dict[str, float]],
    total_height: float,
    gsea: pd.DataFrame,
) -> dict[int, float]:
    for axis in [ax_state, ax_definition, ax_pathway]:
        add_state_bands(axis, layout, total_height)
        axis.set_xlim(0, 1)
        axis.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        axis.spines[["top", "bottom", "left"]].set_visible(False)
        axis.spines["right"].set_visible(True)
        axis.spines["right"].set_color("#B7BDC1")
        axis.spines["right"].set_linewidth(0.8)

    pathway_positions: dict[int, float] = {}
    for state in STATES:
        band = layout[state]
        ax_state.text(
            0.5,
            band["center"],
            f"State\n{state}",
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=TEXT,
        )
        definition = textwrap.fill(STATE_DEFINITIONS[state], width=22)
        ax_definition.text(
            0.06,
            band["center"],
            definition,
            ha="left",
            va="center",
            fontsize=10.2,
            fontweight="semibold" if state <= 6 else "normal",
            color=TEXT,
            linespacing=1.2,
        )

        state_pathways = gsea[gsea["State"].eq(state)]
        positions = positions_in_band(
            band["start"], band["height"], len(state_pathways)
        )
        for (_, row), y in zip(state_pathways.iterrows(), positions):
            pathway_positions[int(row.name)] = y
            name = str(row["pathway_display"])
            ax_pathway.text(
                0.025,
                y,
                name,
                ha="left",
                va="center",
                fontsize=9.2,
                color=TEXT if bool(row["passes_exploratory_fdr_0.25"]) else "#747C82",
                linespacing=1.0,
            )
    return pathway_positions


def draw_gsea_axis(
    ax: plt.Axes,
    layout: dict[int, dict[str, float]],
    total_height: float,
    gsea: pd.DataFrame,
    pathway_positions: dict[int, float],
) -> None:
    add_state_bands(ax, layout, total_height)
    x_min = min(-2.8, float(gsea["NES"].min()) - 0.2)
    x_max = max(2.6, float(gsea["NES"].max()) + 0.2)
    ax.set_xlim(x_min, x_max)
    ax.axvline(0, color="#596168", linewidth=0.9, zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=-4)

    for index, row in gsea.iterrows():
        y = pathway_positions[int(index)]
        nes = float(row["NES"])
        color = POSITIVE if nes >= 0 else NEGATIVE
        passes = bool(row["passes_exploratory_fdr_0.25"])
        ax.hlines(
            y,
            0,
            nes,
            color=color,
            linewidth=1.25,
            alpha=0.72 if passes else 0.38,
            zorder=2,
        )
        ax.scatter(
            [nes],
            [y],
            s=float(fdr_dot_size(float(row["FDR"]))),
            facecolors=color if passes else "white",
            edgecolors=color,
            linewidths=1.0,
            zorder=3,
        )

    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#70787E")
    ax.set_xlabel("Normalized enrichment score (NES)", labelpad=8)


def build_source_data(
    stats: pd.DataFrame,
    gsea: pd.DataFrame,
) -> pd.DataFrame:
    expression_rows: list[pd.DataFrame] = []
    for defining_state in STATES:
        for marker_order, gene in enumerate(MARKERS[defining_state], start=1):
            subset = stats[stats["gene"].eq(gene)].copy()
            subset["panel"] = "C"
            subset["record_type"] = "marker_expression"
            subset["defining_state"] = defining_state
            subset["state_definition"] = STATE_DEFINITIONS[defining_state]
            subset["marker_order"] = marker_order
            subset["pathway_annotation"] = ""
            subset["pathway_name"] = ""
            subset["pathway_display"] = ""
            subset["NES"] = np.nan
            subset["FDR"] = np.nan
            subset["source_file"] = H5AD.relative_to(ROOT).as_posix()
            subset["notes"] = (
                "dot size = percent expressing (>0); color = per-gene min-max "
                "scaled mean log1p(CP10K), including zero-expression cells"
            )
            expression_rows.append(subset)

    expression = pd.concat(expression_rows, ignore_index=True)
    rare = pd.DataFrame(
        {
            "panel": "C",
            "record_type": "rare_state_marker_placeholder",
            "defining_state": [7, 8, 9, 10],
            "state_definition": [STATE_DEFINITIONS[state] for state in [7, 8, 9, 10]],
            "marker_order": np.nan,
            "comparison_state": np.nan,
            "comparison_state_label": "",
            "gene": "",
            "n_cells_in_state": np.nan,
            "n_cells_expressing": np.nan,
            "percent_expressing": np.nan,
            "mean_log1p_cp10k_all_cells": np.nan,
            "mean_expression_scaled_0_1_within_gene": np.nan,
            "pathway_annotation": "",
            "pathway_name": "",
            "pathway_display": "",
            "NES": np.nan,
            "FDR": np.nan,
            "source_file": "Figure 2 marker and state-definition specification",
            "notes": "No named major marker genes were supplied for this rare state.",
        }
    )
    definitions = pd.DataFrame(
        {
            "panel": "C",
            "record_type": "state_definition",
            "defining_state": STATES,
            "state_definition": [STATE_DEFINITIONS[state] for state in STATES],
            "marker_order": np.nan,
            "comparison_state": np.nan,
            "comparison_state_label": "",
            "gene": "",
            "n_cells_in_state": np.nan,
            "n_cells_expressing": np.nan,
            "percent_expressing": np.nan,
            "mean_log1p_cp10k_all_cells": np.nan,
            "mean_expression_scaled_0_1_within_gene": np.nan,
            "pathway_annotation": "",
            "pathway_name": "",
            "pathway_display": "",
            "NES": np.nan,
            "FDR": np.nan,
            "source_file": "Figure 2 marker and state-definition specification",
            "notes": "Exact proposed definition supplied for Figure 2.",
        }
    )
    pathway = pd.DataFrame(
        {
            "panel": "C",
            "record_type": "GSEA_pathway",
            "defining_state": gsea["State"],
            "state_definition": gsea["State"].map(STATE_DEFINITIONS),
            "marker_order": np.nan,
            "comparison_state": np.nan,
            "comparison_state_label": "",
            "gene": "",
            "n_cells_in_state": np.nan,
            "n_cells_expressing": np.nan,
            "percent_expressing": np.nan,
            "mean_log1p_cp10k_all_cells": np.nan,
            "mean_expression_scaled_0_1_within_gene": np.nan,
            "pathway_annotation": gsea["annotation_display"],
            "pathway_name": gsea["Pathway Name"],
            "pathway_display": gsea["pathway_display"],
            "NES": gsea["NES"],
            "FDR": gsea["FDR"],
            "source_file": (
                GSEA_XLSX.relative_to(ROOT).as_posix() + " [sheet=Aging]"
            ),
            "notes": np.where(
                gsea["passes_exploratory_fdr_0.25"],
                "filled point: FDR <= 0.25",
                "open point: FDR > 0.25",
            ),
        }
    )
    return pd.concat([expression, rare, definitions, pathway], ignore_index=True)


def write_manifest(n_duplicate_gsea_rows: int) -> None:
    records = [
        {
            "figure_panel": "C",
            "input_file": H5AD.relative_to(ROOT).as_posix(),
            "input_type": "H5AD",
            "used_fields": "raw/X; obs/LEC_state",
            "purpose": (
                "mean log1p(CP10K) and percent expressing for the exact marker set "
                "across all 1,119 Step08 LECs"
            ),
        },
        {
            "figure_panel": "C validation",
            "input_file": STATE_COUNTS_CSV.relative_to(ROOT).as_posix(),
            "input_type": "CSV",
            "used_fields": "LEC_state;n_cells;percent_of_all_LECs",
            "purpose": "validate State 0-10 identities and the 1,119-cell population",
        },
        {
            "figure_panel": "C",
            "input_file": GSEA_XLSX.relative_to(ROOT).as_posix(),
            "input_type": "XLSX sheet Aging",
            "used_fields": "State;In-Figure Annotation;Pathway Name;NES;FDR",
            "purpose": (
                "state-aligned GSEA pathways; "
                f"{n_duplicate_gsea_rows} exact duplicate row removed for display"
            ),
        },
        {
            "figure_panel": "C",
            "input_file": "Figure 2 marker and state-definition specification",
            "input_type": "figure specification",
            "used_fields": "State;Proposed definition;Major marker genes",
            "purpose": (
                "exact marker lists and proposed state definitions; States 7-10 "
                "retained without invented marker genes"
            ),
        },
    ]
    pd.DataFrame(records).to_csv(MANIFEST_CSV, index=False)


def add_legends(fig: plt.Figure, ax_d: plt.Axes, ax_nes: plt.Axes) -> None:
    d_position = ax_d.get_position()
    color_axis = fig.add_axes(
        [d_position.x0, d_position.y0 - 0.055, d_position.width * 0.43, 0.010]
    )
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=mpl.colormaps["Reds"]),
        cax=color_axis,
        orientation="horizontal",
    )
    colorbar.outline.set_visible(False)
    colorbar.set_ticks([0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.set_label(
        "Mean log1p(CP10K), per-gene scaled 0-1", fontsize=9.0, labelpad=2
    )

    pct_values = [0, 25, 50, 75, 100]
    size_handles = [
        ax_d.scatter(
            [],
            [],
            s=float(dot_size(value)),
            facecolor="#D65C4D",
            edgecolor="#AAB1B6",
            linewidth=0.35,
            label=str(value),
        )
        for value in pct_values
    ]
    legend_d = ax_d.legend(
        handles=size_handles,
        title="Cells expressing (%)",
        loc="upper left",
        bbox_to_anchor=(0.48, -0.055),
        ncol=5,
        frameon=False,
        fontsize=9,
        title_fontsize=9.2,
        handletextpad=0.2,
        columnspacing=0.7,
        borderaxespad=0,
    )
    ax_d.add_artist(legend_d)

    sign_handles = [
        Line2D([0], [0], color=POSITIVE, marker="o", linewidth=1.4, label="NES > 0"),
        Line2D([0], [0], color=NEGATIVE, marker="o", linewidth=1.4, label="NES < 0"),
        Line2D(
            [0],
            [0],
            color="#727B81",
            marker="o",
            markerfacecolor="white",
            linewidth=0,
            label="FDR > 0.25",
        ),
    ]
    ax_nes.legend(
        handles=sign_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=1.1,
        handletextpad=0.45,
        borderaxespad=0,
    )


def main() -> None:
    gsea, n_duplicate_gsea_rows = load_gsea()
    expression_stats, state_counts = load_expression()
    layout, total_height = build_layout(gsea)

    source_data = build_source_data(expression_stats, gsea)
    source_data.to_csv(SOURCE_CSV, index=False)
    write_manifest(n_duplicate_gsea_rows)

    figure = plt.figure(figsize=(25.5, 19.6), facecolor="white")
    outer = GridSpec(
        1,
        2,
        figure=figure,
        width_ratios=[1.00, 2.15],
        left=0.065,
        right=0.985,
        top=0.895,
        bottom=0.105,
        wspace=0.055,
    )
    ax_d = figure.add_subplot(outer[0, 0])
    inner_e = GridSpecFromSubplotSpec(
        1,
        4,
        subplot_spec=outer[0, 1],
        width_ratios=[0.52, 2.35, 6.70, 2.60],
        wspace=0.015,
    )
    ax_state = figure.add_subplot(inner_e[0, 0], sharey=ax_d)
    ax_definition = figure.add_subplot(inner_e[0, 1], sharey=ax_d)
    ax_pathway = figure.add_subplot(inner_e[0, 2], sharey=ax_d)
    ax_nes = figure.add_subplot(inner_e[0, 3], sharey=ax_d)

    draw_panel_d(ax_d, expression_stats, layout, total_height)
    pathway_positions = draw_text_columns(
        ax_state,
        ax_definition,
        ax_pathway,
        layout,
        total_height,
        gsea,
    )
    draw_gsea_axis(ax_nes, layout, total_height, gsea, pathway_positions)
    add_legends(figure, ax_d, ax_nes)

    figure.text(
        0.065,
        0.968,
        "Aging LEC state marker and pathway map",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.065,
        0.943,
        "State boundaries are shared across the marker dot plot, proposed definitions, and supporting GSEA evidence.",
        ha="left",
        va="top",
        fontsize=12,
        color=SUBTEXT,
    )

    d_position = ax_d.get_position()
    figure.text(
        d_position.x0 - 0.030,
        0.968,
        "C",
        va="top",
        fontsize=24,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        d_position.x0,
        0.915,
        "State-defining markers, definitions, and supporting GSEA pathways",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
    )

    header_y = 0.899
    for axis, label, align in [
        (ax_state, "State", "center"),
        (ax_definition, "Proposed definition", "left"),
        (ax_pathway, "Supporting GSEA pathway", "left"),
        (ax_nes, "GSEA effect", "center"),
    ]:
        position = axis.get_position()
        x = (position.x0 + position.x1) / 2 if align == "center" else position.x0 + 0.004
        figure.text(
            x,
            header_y,
            label,
            ha=align,
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=TEXT,
        )

    figure.text(
        0.065,
        0.020,
        (
            "Panel C uses all 1,119 LECs in the same full Step08 state-definition object as Figure 2A "
            "(before the later donor >=25-LEC inference filter). Dot size is the percentage of cells with "
            "expression >0; color is the state mean log1p(CP10K), scaled 0-1 separately for each gene. "
            "GSEA labels reproduce the Aging sheet pathway terms; point area reflects FDR and open points have FDR >0.25."
        ),
        ha="left",
        va="bottom",
        fontsize=9.3,
        color=SUBTEXT,
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

    print(f"Integrated Figure 2C written to: {OUTPUT_DIR}")
    print(f"Step08 LECs: {int(state_counts['n_cells'].sum())}")
    print(f"Named marker rows: {sum(len(MARKERS[state]) for state in STATES)}")
    print(f"Unique marker genes: {len(set(gene for genes in MARKERS.values() for gene in genes))}")
    print(f"Displayed Aging GSEA pathways: {len(gsea)}")
    print(f"Exact duplicate GSEA rows removed: {n_duplicate_gsea_rows}")


if __name__ == "__main__":
    main()
