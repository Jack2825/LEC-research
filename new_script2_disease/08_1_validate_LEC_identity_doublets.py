#!/usr/bin/env python3
r"""
Step 08.1: LEC-only identity, doublet, neighborhood, and donor-description QC.

Every calculation and figure in this script is restricted to the final LECs
present in the current Step 08 combined clustering object. It never adds ECs,
other cell types, or donors with zero final LECs. It is read-only: no cell is
removed, no state is renamed, and no h5ad is written.

Important interpretation
------------------------
The doublet scores were created upstream. This script ranks those existing
scores only among final LECs from the matching technical sample, with an
LEC-only cohort/dataset fallback when a sample contains too few LECs. A score
percentile is a relative rank, not a doublet probability or removal call.

Because non-LEC reference populations are intentionally excluded, cardiac and
other lineage signatures are relative differences among final LECs. This is a
state-review analysis, not an external cell-identity classifier.

Default input
-------------
new_result2_disease/08_combined_LEC_recluster/
    08_combined_final_LEC_reclustered.h5ad

Default output
--------------
new_result2_disease/08_1_LEC_identity_doublet_QC/

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\08_1_validate_LEC_identity_doublets.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCRIPT_VERSION = "08_1_disease_final_Step08_LECs_only_identity_doublet_QC_2026_08_11"
DEFAULT_STEP08 = (
    "new_result2_disease/08_combined_LEC_recluster/"
    "08_combined_final_LEC_reclustered.h5ad"
)
DEFAULT_OUTPUT = "new_result2_disease/08_1_LEC_identity_doublet_QC"
DEFAULT_FOCUS_STATES = ["state4", "state5", "state6", "state7", "state8"]
DEFAULT_FOCUS_DONORS = ["dataset01::P1430", "dataset01::P1508"]

EXPECTED_DATASETS = ["dataset01", "dataset08", "dataset11"]
EXPECTED_CONDITIONS = ["NF", "DCM", "HCM", "DoxCM", "HFpEF"]
EXPECTED_REGIONS = {"dataset01": "LV", "dataset08": "LV", "dataset11": "S"}
CONDITION_COLORS = {
    "NF": "#0072B2",
    "DCM": "#D55E00",
    "HCM": "#CC79A7",
    "DoxCM": "#56B4E9",
    "HFpEF": "#E69F00",
}

MARKER_PANELS: dict[str, list[str]] = {
    "LEC_specific": [
        "PROX1", "PDPN", "FLT4", "CCL21", "LYVE1", "NRP2", "RELN", "FOXC2",
    ],
    "pan_endothelial": [
        "PECAM1", "CDH5", "CLDN5", "VWF", "KDR", "ERG", "ESAM", "EMCN", "PLVAP",
    ],
    "cardiomyocyte": [
        "TTN", "RYR2", "SLC8A1", "DMD", "SORBS2", "MYH7", "MYH6", "MYL2",
        "MYL3", "TNNI3", "TNNT2", "ACTC1", "CKM", "PLN", "MYBPC3", "TCAP",
    ],
    "fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "COL6A1"],
    "pericyte_smc": [
        "RGS5", "PDGFRB", "CSPG4", "ACTA2", "MCAM", "MYH11", "TAGLN",
    ],
    "immune": ["PTPRC", "LST1", "TYROBP", "FCER1G", "CD3D", "CD79A", "MS4A1"],
}
MARKER_FIGURE_GENES = [
    "PROX1", "PDPN", "FLT4", "CCL21", "LYVE1", "NRP2",
    "PECAM1", "CDH5", "CLDN5", "VWF", "KDR", "ERG",
    "TTN", "RYR2", "SLC8A1", "MYH7", "MYL2", "TNNI3", "TNNT2", "ACTC1",
    "COL1A1", "DCN", "RGS5", "PDGFRB", "PTPRC", "LST1",
]

STATE_MIXED_MARKER_REVIEW_FRACTION = 0.05
STATE_DOUBLET_Q99_REVIEW_FRACTION = 0.08
STATE_DONOR_DOMINANCE_REVIEW_FRACTION = 0.50


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def natural_key(value: object) -> tuple[object, ...]:
    pieces = re.split(r"(\d+(?:\.\d+)?)", str(value))
    return tuple(
        (0, float(piece)) if piece and piece.replace(".", "", 1).isdigit()
        else (1, piece.casefold())
        for piece in pieces if piece != ""
    )


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def safe_numeric(values: pd.Series) -> pd.Series:
    cleaned = normalized_text(values).replace(
        {"": pd.NA, "<NA>": pd.NA, "NA": pd.NA, "unknown": pd.NA, "Unknown": pd.NA}
    )
    return pd.to_numeric(cleaned, errors="coerce")


def first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    reference = np.sort(reference[np.isfinite(reference)])
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if reference.size:
        result[valid] = np.searchsorted(reference, values[valid], side="right") / reference.size
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identity, doublet, neighborhood, and descriptive QC using final Step 08 LECs only."
    )
    parser.add_argument("--step08_h5ad", default=DEFAULT_STEP08)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--focus_states", nargs="+", default=DEFAULT_FOCUS_STATES)
    parser.add_argument("--focus_donors", nargs="+", default=DEFAULT_FOCUS_DONORS)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--minimum_sample_LECs_for_percentile", type=int, default=20)
    parser.add_argument("--cardiac_LEC_quantile", type=float, default=0.99)
    parser.add_argument("--minimum_cardiac_genes", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_neighborhood", action="store_true")
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def state_palette(adata: ad.AnnData, states: list[str]) -> dict[str, str]:
    colors = list(adata.uns.get("LEC_state_colors", []))
    values = adata.obs["LEC_state"]
    if isinstance(values.dtype, pd.CategoricalDtype):
        categories = [str(x) for x in values.cat.categories]
    else:
        categories = states
    if len(colors) == len(categories):
        mapped = dict(zip(categories, colors))
        if all(state in mapped for state in states):
            return {state: mapped[state] for state in states}
    fallback = list(plt.get_cmap("tab20").colors)
    return {state: fallback[i % len(fallback)] for i, state in enumerate(states)}


def panel_metrics(
    matrix: sp.csr_matrix,
    genes: pd.Index,
    panels: dict[str, list[str]],
) -> dict[str, dict[str, np.ndarray]]:
    lookup = {str(gene): index for index, gene in enumerate(genes)}
    result: dict[str, dict[str, np.ndarray]] = {}
    for panel, panel_genes in panels.items():
        indices = [lookup[gene] for gene in panel_genes]
        part = matrix[:, indices]
        detected = np.asarray((part > 0).sum(axis=1)).ravel().astype(int)
        result[panel] = {
            "mean_logexpr": np.asarray(part.mean(axis=1)).ravel(),
            "genes_detected": detected,
            "fraction_detected": detected / len(indices),
        }
    return result


def add_LEC_only_doublet_percentiles(
    cells: pd.DataFrame,
    minimum_sample_LECs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = cells.copy()
    result["doublet_score"] = pd.to_numeric(result["qc_doublet_score"], errors="coerce")
    result["doublet_score_percentile_within_final_LECs"] = np.nan
    result["doublet_percentile_reference"] = "numeric_score_unavailable"
    threshold_rows: list[dict[str, object]] = []

    for sample_key, sample in result.groupby("sample_key", observed=True, dropna=False):
        available = sample["doublet_score"].notna()
        if not available.any():
            continue
        sample_reference = sample.loc[available, "doublet_score"].to_numpy(float)
        reference = sample_reference
        reference_label = "matching_sample_final_LECs"
        if len(reference) < minimum_sample_LECs:
            dataset = str(sample["dataset_folder"].iloc[0])
            condition = str(sample["condition"].iloc[0])
            cohort_mask = (
                result["dataset_folder"].astype(str).eq(dataset)
                & result["condition"].astype(str).eq(condition)
                & result["doublet_score"].notna()
            )
            reference = result.loc[cohort_mask, "doublet_score"].to_numpy(float)
            reference_label = "matching_dataset_condition_final_LECs_fallback"
            if len(reference) < minimum_sample_LECs:
                dataset_mask = (
                    result["dataset_folder"].astype(str).eq(dataset)
                    & result["doublet_score"].notna()
                )
                reference = result.loc[dataset_mask, "doublet_score"].to_numpy(float)
                reference_label = "matching_dataset_final_LECs_fallback"
        index = sample.index[available]
        result.loc[index, "doublet_score_percentile_within_final_LECs"] = empirical_percentile(
            reference, result.loc[index, "doublet_score"].to_numpy(float)
        )
        result.loc[index, "doublet_percentile_reference"] = reference_label
        threshold_rows.append(
            {
                "sample_key": str(sample_key),
                "dataset_folder": str(sample["dataset_folder"].iloc[0]),
                "condition": str(sample["condition"].iloc[0]),
                "n_final_LECs_in_sample_with_numeric_score": int(len(sample_reference)),
                "reference_scope": reference_label,
                "n_final_LECs_in_reference": int(len(reference)),
                "q90": float(np.quantile(reference, 0.90)),
                "q95": float(np.quantile(reference, 0.95)),
                "q99": float(np.quantile(reference, 0.99)),
            }
        )
    result["doublet_percentile_ge_95"] = result[
        "doublet_score_percentile_within_final_LECs"
    ].ge(0.95)
    result["doublet_percentile_ge_99"] = result[
        "doublet_score_percentile_within_final_LECs"
    ].ge(0.99)
    scrublet_column = first_existing(result, ["scrublet", "predicted_doublet"])
    result["author_doublet_call"] = (
        normalized_text(result[scrublet_column]) if scrublet_column else ""
    )
    result["doublet_evidence_source"] = np.where(
        result["doublet_score"].notna(),
        "existing_numeric_score_ranked_within_final_LECs",
        np.where(
            result["author_doublet_call"].ne(""),
            "existing_categorical_author_call",
            "unavailable",
        ),
    )
    return result, pd.DataFrame(threshold_rows)


def add_LEC_neighborhood_metrics(
    cells: pd.DataFrame,
    graph: sp.csr_matrix,
    n_neighbors: int,
) -> pd.DataFrame:
    graph = graph.tocsr()
    state = cells["LEC_state"].astype(str).to_numpy()
    donor = cells["donor_key"].astype(str).to_numpy()
    dataset = cells["dataset_folder"].astype(str).to_numpy()
    condition = cells["condition"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for position in range(len(cells)):
        start, stop = graph.indptr[position], graph.indptr[position + 1]
        neighbors = graph.indices[start:stop]
        weights = graph.data[start:stop]
        keep = neighbors != position
        neighbors, weights = neighbors[keep], weights[keep]
        if len(neighbors) > n_neighbors:
            chosen = np.argpartition(weights, -n_neighbors)[-n_neighbors:]
            neighbors = neighbors[chosen]
        if len(neighbors):
            rows.append(
                {
                    "n_LEC_neighbors_evaluated": int(len(neighbors)),
                    "neighbor_same_LEC_state_fraction": float(np.mean(state[neighbors] == state[position])),
                    "neighbor_same_donor_fraction": float(np.mean(donor[neighbors] == donor[position])),
                    "neighbor_same_dataset_fraction": float(np.mean(dataset[neighbors] == dataset[position])),
                    "neighbor_same_condition_fraction": float(np.mean(condition[neighbors] == condition[position])),
                }
            )
        else:
            rows.append(
                {
                    "n_LEC_neighbors_evaluated": 0,
                    "neighbor_same_LEC_state_fraction": np.nan,
                    "neighbor_same_donor_fraction": np.nan,
                    "neighbor_same_dataset_fraction": np.nan,
                    "neighbor_same_condition_fraction": np.nan,
                }
            )
    return pd.concat([cells.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summarize_cells(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, part in frame.groupby(groups, observed=True, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, map(str, keys)))
        available = part["doublet_score_percentile_within_final_LECs"].notna()
        row.update(
            {
                "n_LEC_cells": int(len(part)),
                "n_LEC_positive_donors": int(part["donor_key"].nunique()),
                "n_datasets": int(part["dataset_folder"].nunique()),
                "conditions": ";".join(sorted(part["condition"].astype(str).unique())),
                "median_LEC_specific_genes_detected": float(part["lec_specific_genes_detected"].median()),
                "median_pan_endothelial_genes_detected": float(part["pan_endothelial_genes_detected"].median()),
                "median_cardiomyocyte_genes_detected": float(part["cardiomyocyte_genes_detected"].median()),
                "mean_LEC_endothelial_identity_zscore": float(part["LEC_endothelial_identity_zscore"].mean()),
                "mean_cardiomyocyte_zscore": float(part["cardiomyocyte_zscore_within_dataset_LECs"].mean()),
                "n_mixed_cardiac_endothelial_review_flags": int(part["mixed_cardiac_endothelial_review_flag"].sum()),
                "fraction_mixed_cardiac_endothelial_review_flags": float(part["mixed_cardiac_endothelial_review_flag"].mean()),
                "n_numeric_doublet_scores": int(available.sum()),
                "fraction_doublet_percentile_ge_95_among_available": (
                    float(part.loc[available, "doublet_percentile_ge_95"].mean()) if available.any() else np.nan
                ),
                "fraction_doublet_percentile_ge_99_among_available": (
                    float(part.loc[available, "doublet_percentile_ge_99"].mean()) if available.any() else np.nan
                ),
                "mean_neighbor_same_LEC_state_fraction": float(part["neighbor_same_LEC_state_fraction"].mean()),
                "mean_neighbor_same_donor_fraction": float(part["neighbor_same_donor_fraction"].mean()),
                "fraction_any_review_signal": float(part["any_review_signal"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def color_state_ticks(ax: plt.Axes, palette: dict[str, str], axis: str = "x") -> None:
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for label in labels:
        if label.get_text() in palette:
            label.set_color(palette[label.get_text()])
            label.set_fontweight("bold")


def save_demographics(
    donors: pd.DataFrame,
    cohort_order: list[str],
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)
    rng = np.random.default_rng(8012026)

    ax = axes[0, 0]
    for x, cohort in enumerate(cohort_order):
        part = donors.loc[donors["cohort"].eq(cohort)]
        ages = part["age_numeric"].dropna().to_numpy(float)
        color = CONDITION_COLORS.get(str(part["condition"].iloc[0]), "#777777")
        if len(ages):
            ax.scatter(x + rng.uniform(-0.14, 0.14, len(ages)), ages, s=34, color=color,
                       alpha=0.8, edgecolor="white", linewidth=0.4)
            ax.plot([x - 0.18, x + 0.18], [np.median(ages)] * 2, color="black", lw=2)
        ax.text(
            x,
            0.98,
            f"age available: {len(ages)}\nLEC+ donors: {len(part)}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
        )
    labels = [x.replace("dataset", "D") for x in cohort_order]
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("Age (years)")
    ax.set_title("Age metadata among final-LEC-positive donors")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    sex_colors = {"Female": "#CC79A7", "Male": "#0072B2", "Unknown": "#BDBDBD"}
    bottom = np.zeros(len(cohort_order))
    for sex in ["Female", "Male", "Unknown"]:
        values = [int(((donors["cohort"] == cohort) & (donors["sex_standardized_QC"] == sex)).sum()) for cohort in cohort_order]
        ax.bar(range(len(cohort_order)), values, bottom=bottom, color=sex_colors[sex],
               label=sex, edgecolor="white")
        bottom += np.asarray(values)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("Final-LEC-positive donors")
    ax.set_title("Sex information for donors represented in Step 08")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    donor_counts = [int((donors["cohort"] == cohort).sum()) for cohort in cohort_order]
    cell_counts = [int(donors.loc[donors["cohort"].eq(cohort), "n_LEC_cells"].sum()) for cohort in cohort_order]
    x = np.arange(len(cohort_order))
    bars = ax.bar(x, cell_counts, color=[CONDITION_COLORS.get(cohort.split(" | ")[1], "#777777") for cohort in cohort_order])
    for bar, n_donors in zip(bars, donor_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{n_donors} donors",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Final Step 08 LEC cells")
    ax.set_title("Final LEC cells and contributing donors")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    for x_value, cohort in enumerate(cohort_order):
        part = donors.loc[donors["cohort"].eq(cohort)]
        values = part["n_LEC_cells"].to_numpy(float)
        ax.scatter(x_value + rng.uniform(-0.15, 0.15, len(values)), values, s=30,
                   color=CONDITION_COLORS.get(str(part["condition"].iloc[0]), "#777777"),
                   alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axhline(5, color="#E69F00", ls="--", label="5 LECs")
    ax.axhline(10, color="#D55E00", ls=":", label="10 LECs")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("Final LEC cells per represented donor (log scale)")
    ax.set_title("Only donors with at least one final LEC")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Final Step 08 LEC donor demographics and support", fontsize=17)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_signature_heatmap(
    state_summary: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    panels = list(MARKER_PANELS)
    columns = [f"mean_{panel}_zscore" for panel in panels]
    matrix = state_summary.set_index("LEC_state").reindex(states)[columns].to_numpy(float)
    extent = max(float(np.nanmax(np.abs(matrix))), 0.1)
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-extent, vmax=extent, aspect="auto")
    ax.set_xticks(range(len(panels)), [x.replace("_", " ") for x in panels], rotation=30, ha="right")
    ax.set_yticks(range(len(states)), states)
    color_state_ticks(ax, palette, "y")
    for i in range(len(states)):
        for j in range(len(panels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Signatures standardized among final LECs within each dataset")
    fig.colorbar(image, ax=ax, label="Mean within-dataset final-LEC z-score")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_marker_dotplot(
    marker: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    genes = [gene for gene in MARKER_FIGURE_GENES if gene in set(marker["gene"])]
    work = marker.loc[marker["gene"].isin(genes)].copy()
    work["gene"] = pd.Categorical(work["gene"], categories=genes, ordered=True)
    work["LEC_state"] = pd.Categorical(work["LEC_state"], categories=states, ordered=True)
    work["x"] = work["LEC_state"].cat.codes
    work["y"] = work["gene"].cat.codes
    work["gene_z"] = work.groupby("gene", observed=True)["mean_logexpr"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0)
    )
    extent = max(float(work["gene_z"].abs().max()), 0.1)
    fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)
    scatter = ax.scatter(work["x"], work["y"], s=15 + 2.3 * work["pct_detected"],
                         c=work["gene_z"], cmap="RdBu_r", vmin=-extent, vmax=extent,
                         edgecolor="#444444", linewidth=0.25)
    ax.set_xticks(range(len(states)), states, rotation=35, ha="right")
    ax.set_yticks(range(len(genes)), genes)
    ax.invert_yaxis()
    color_state_ticks(ax, palette, "x")
    ax.grid(alpha=0.15)
    ax.set_title("Marker expression among final Step 08 LECs")
    fig.colorbar(scatter, ax=ax, label="Gene-wise z-score across LEC states")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_identity_scatter(
    cells: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    focus_donors: list[str],
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for ax, dataset in zip(axes, EXPECTED_DATASETS):
        part = cells.loc[cells["dataset_folder"].astype(str).eq(dataset)]
        for state in states:
            selected = part.loc[part["LEC_state"].astype(str).eq(state)]
            ax.scatter(selected["LEC_endothelial_identity_zscore"],
                       selected["cardiomyocyte_zscore_within_dataset_LECs"],
                       s=11, color=palette[state], alpha=0.5, linewidth=0)
        focus = part.loc[part["donor_key"].astype(str).isin(focus_donors)]
        ax.scatter(focus["LEC_endothelial_identity_zscore"],
                   focus["cardiomyocyte_zscore_within_dataset_LECs"],
                   s=34, facecolors="none", edgecolors="black", linewidth=0.8)
        flagged = part.loc[part["mixed_cardiac_endothelial_review_flag"]]
        ax.scatter(flagged["LEC_endothelial_identity_zscore"],
                   flagged["cardiomyocyte_zscore_within_dataset_LECs"],
                   s=28, marker="x", color="#D62728", linewidth=0.9)
        ax.axhline(0, color="#777777", ls="--", lw=0.7)
        ax.set_title(dataset)
        ax.set_xlabel("LEC/endothelial identity z-score among final LECs")
        ax.set_ylabel("Cardiomyocyte z-score among final LECs")
        ax.grid(alpha=0.15)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[s], label=s, markersize=7) for s in states]
    handles.extend([
        Line2D([0], [0], marker="o", color="black", markerfacecolor="none", label="focus donor", markersize=7, linestyle="none"),
        Line2D([0], [0], marker="x", color="#D62728", label="relative cardiac review", markersize=7, linestyle="none"),
    ])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=6, frameon=False)
    fig.suptitle("Relative identity and cardiac-expression evidence within final LECs", fontsize=16, y=1.03)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_doublet_qc(
    cells: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    available = cells.loc[cells["doublet_score_percentile_within_final_LECs"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    data = [available.loc[available["LEC_state"].astype(str).eq(state), "doublet_score_percentile_within_final_LECs"].to_numpy(float) for state in states]
    box = axes[0].boxplot(data, positions=np.arange(len(states)), widths=0.6,
                          patch_artist=True, showfliers=False)
    for patch, state in zip(box["boxes"], states):
        patch.set_facecolor(palette[state]); patch.set_alpha(0.7)
    axes[0].axhline(0.95, color="#E69F00", ls="--", label="95th percentile")
    axes[0].axhline(0.99, color="#D55E00", ls=":", label="99th percentile")
    axes[0].set_xticks(range(len(states)), states, rotation=35, ha="right")
    color_state_ticks(axes[0], palette, "x")
    axes[0].set_ylabel("Existing score percentile among final LECs")
    axes[0].set_title("LEC-only relative score distributions")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    f95, f99 = [], []
    for state in states:
        part = available.loc[available["LEC_state"].astype(str).eq(state)]
        f95.append(float(part["doublet_percentile_ge_95"].mean()) if len(part) else np.nan)
        f99.append(float(part["doublet_percentile_ge_99"].mean()) if len(part) else np.nan)
    x = np.arange(len(states))
    axes[1].bar(x - 0.2, f95, width=0.4, color="#E69F00", label=">=95th")
    axes[1].bar(x + 0.2, f99, width=0.4, color="#D55E00", label=">=99th")
    axes[1].set_xticks(x, states, rotation=35, ha="right")
    color_state_ticks(axes[1], palette, "x")
    axes[1].set_ylabel("Fraction among final LECs with numeric scores")
    axes[1].set_title("High-tail relative scores\n(dataset08 has categorical Singlet calls only)")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Doublet percentiles are LEC-only ranks, not probabilities", fontsize=16)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_neighbor_qc(
    state_summary: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    work = state_summary.set_index("LEC_state").reindex(states)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    x = np.arange(len(states))
    axes[0].bar(x, work["mean_neighbor_same_LEC_state_fraction"],
                color=[palette[state] for state in states], edgecolor="#444444", linewidth=0.5)
    axes[0].set_xticks(x, states, rotation=35, ha="right")
    color_state_ticks(axes[0], palette, "x")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Mean same-state fraction")
    axes[0].set_title("Final-LEC graph state purity")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, work["mean_neighbor_same_donor_fraction"],
                color=[palette[state] for state in states], edgecolor="#444444", linewidth=0.5)
    axes[1].set_xticks(x, states, rotation=35, ha="right")
    color_state_ticks(axes[1], palette, "x")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean same-donor fraction")
    axes[1].set_title("Final-LEC graph donor locality")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Step 08 graph neighborhoods contain final LECs only", fontsize=16)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_focus_umap(
    adata: ad.AnnData,
    cells: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    focus_states: list[str],
    focus_donors: list[str],
    path: Path,
    dpi: int,
) -> None:
    umap = np.asarray(adata.obsm["X_umap"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for state in states:
        mask = cells["LEC_state"].astype(str).eq(state).to_numpy()
        axes[0, 0].scatter(umap[mask, 0], umap[mask, 1], s=4, color=palette[state],
                           alpha=0.7, linewidth=0, label=state)
    axes[0, 0].set_title("All final Step 08 LEC states")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0, 1].scatter(umap[:, 0], umap[:, 1], s=3, color="#D9D9D9", linewidth=0)
    for state in focus_states:
        mask = cells["LEC_state"].astype(str).eq(state).to_numpy()
        axes[0, 1].scatter(umap[mask, 0], umap[mask, 1], s=7, color=palette[state],
                           alpha=0.85, linewidth=0, label=state)
    axes[0, 1].set_title("Prespecified focus states")
    axes[0, 1].legend(frameon=False, ncol=2)
    for ax, donor in zip(axes[1], focus_donors[:2]):
        mask = cells["donor_key"].astype(str).eq(donor).to_numpy()
        ax.scatter(umap[:, 0], umap[:, 1], s=3, color="#D9D9D9", linewidth=0)
        ax.scatter(umap[mask, 0], umap[mask, 1], s=10,
                   c=[palette[str(state)] for state in cells.loc[mask, "LEC_state"]],
                   edgecolor="black", linewidth=0.2)
        ax.set_title(f"{donor}\n{int(mask.sum())} final LECs")
    for ax in axes.ravel():
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Final Step 08 LECs only", fontsize=17)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.neighbors < 1:
        raise ValueError("--neighbors must be positive.")
    if args.minimum_sample_LECs_for_percentile < 2:
        raise ValueError("--minimum_sample_LECs_for_percentile must be at least 2.")
    if not 0 < args.cardiac_LEC_quantile < 1:
        raise ValueError("--cardiac_LEC_quantile must be between 0 and 1.")

    project_dir = find_project_dir()
    step08_path = resolve_path(args.step08_h5ad, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    if not step08_path.exists():
        raise FileNotFoundError(step08_path)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {output_root}. Pass --overwrite.")
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading final Step 08 LEC object: {step08_path}")
    adata = ad.read_h5ad(step08_path)
    required = {
        "dataset_folder", "condition", "donor_key", "sample_key", "LEC_state",
        "region_standardized_analysis", "step03_final_QC_pass", "qc_doublet_score",
    }
    missing = sorted(required - set(adata.obs.columns))
    if missing:
        raise KeyError(f"Step 08 is missing required columns: {missing}")
    if not adata.obs_names.is_unique:
        raise ValueError("Step 08 cell identifiers are not unique.")
    if not adata.obs["step03_final_QC_pass"].astype(bool).all():
        raise ValueError("Step 08 contains cells that did not pass Step 03 QC.")

    cells = adata.obs.copy()
    cells.index = cells.index.astype(str)
    cells.index.name = "step08_cell_id"
    cells = cells.reset_index()
    cells["dataset_folder"] = normalized_text(cells["dataset_folder"])
    cells["condition"] = normalized_text(cells["condition"])
    cells["donor_key"] = normalized_text(cells["donor_key"])
    cells["sample_key"] = normalized_text(cells["sample_key"])
    cells["LEC_state"] = normalized_text(cells["LEC_state"])
    if len(cells) != 6291:
        warnings.warn(f"Current Step 08 has {len(cells)} final LECs rather than the previous 6,291.")
    if cells["donor_key"].isna().any() or cells["donor_key"].eq("").any():
        raise ValueError("Every final LEC must have a donor_key.")

    states = sorted(cells["LEC_state"].unique(), key=natural_key)
    datasets = sorted(cells["dataset_folder"].unique())
    conditions = sorted(cells["condition"].unique())
    if datasets != EXPECTED_DATASETS:
        raise ValueError(f"Unexpected datasets: {datasets}")
    if sorted(conditions) != sorted(EXPECTED_CONDITIONS):
        raise ValueError(f"Unexpected conditions: {conditions}")
    region_check = cells.groupby("dataset_folder", observed=True)["region_standardized_analysis"].agg(
        lambda x: ";".join(sorted(normalized_text(x).unique()))
    ).to_dict()
    if region_check != EXPECTED_REGIONS:
        raise ValueError(f"Unexpected regions: {region_check}")
    focus_states = [str(x) for x in args.focus_states]
    focus_donors = [str(x) for x in args.focus_donors]
    if set(focus_states) - set(states):
        raise ValueError(f"Missing focus states: {sorted(set(focus_states) - set(states))}")
    if set(focus_donors) - set(cells["donor_key"]):
        raise ValueError(f"Missing focus donors: {sorted(set(focus_donors) - set(cells['donor_key']))}")
    palette = state_palette(adata, states)

    available = set(map(str, adata.var_names))
    panels = {panel: [gene for gene in genes if gene in available] for panel, genes in MARKER_PANELS.items()}
    if any(not genes for genes in panels.values()):
        raise ValueError(f"Empty marker panel after gene matching: {[k for k, v in panels.items() if not v]}")
    marker_genes = sorted({gene for genes in panels.values() for gene in genes}, key=adata.var_names.get_loc)
    marker_indices = [int(adata.var_names.get_loc(gene)) for gene in marker_genes]
    marker_matrix = sp.csr_matrix(adata[:, marker_indices].X, dtype=np.float64)
    marker_index = pd.Index(marker_genes)
    metrics = panel_metrics(marker_matrix, marker_index, panels)

    for panel, values in metrics.items():
        prefix = panel.lower()
        cells[f"{prefix}_mean_logexpr"] = values["mean_logexpr"]
        cells[f"{prefix}_genes_detected"] = values["genes_detected"]
        cells[f"{prefix}_fraction_detected"] = values["fraction_detected"]
        cells[f"{prefix}_zscore_within_dataset_LECs"] = cells.groupby(
            "dataset_folder", observed=True
        )[f"{prefix}_mean_logexpr"].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0)
        )
        cells[f"{prefix}_percentile_within_dataset_LECs"] = np.nan
        for dataset, index in cells.groupby("dataset_folder", observed=True).groups.items():
            scores = cells.loc[index, f"{prefix}_mean_logexpr"].to_numpy(float)
            cells.loc[index, f"{prefix}_percentile_within_dataset_LECs"] = empirical_percentile(scores, scores)

    cells["LEC_endothelial_identity_zscore"] = 0.5 * (
        cells["lec_specific_zscore_within_dataset_LECs"]
        + cells["pan_endothelial_zscore_within_dataset_LECs"]
    )
    cells["cardiac_final_LEC_review_threshold"] = np.nan
    for dataset, index in cells.groupby("dataset_folder", observed=True).groups.items():
        scores = cells.loc[index, "cardiomyocyte_mean_logexpr"].to_numpy(float)
        cells.loc[index, "cardiac_final_LEC_review_threshold"] = float(
            np.quantile(scores, args.cardiac_LEC_quantile)
        )
    cells["mixed_cardiac_endothelial_review_flag"] = (
        cells["cardiomyocyte_mean_logexpr"].ge(cells["cardiac_final_LEC_review_threshold"])
        & cells["cardiomyocyte_genes_detected"].ge(args.minimum_cardiac_genes)
        & cells["pan_endothelial_genes_detected"].ge(2)
    )

    cells, doublet_thresholds = add_LEC_only_doublet_percentiles(
        cells, args.minimum_sample_LECs_for_percentile
    )
    if not args.skip_neighborhood:
        if "connectivities" not in adata.obsp:
            raise KeyError("Step 08 lacks connectivities.")
        cells = add_LEC_neighborhood_metrics(cells, adata.obsp["connectivities"], args.neighbors)
    else:
        for column in [
            "neighbor_same_LEC_state_fraction", "neighbor_same_donor_fraction",
            "neighbor_same_dataset_fraction", "neighbor_same_condition_fraction",
        ]:
            cells[column] = np.nan
        cells["n_LEC_neighbors_evaluated"] = 0
    cells["focus_state"] = cells["LEC_state"].isin(focus_states)
    cells["focus_donor"] = cells["donor_key"].isin(focus_donors)
    cells["any_review_signal"] = (
        cells["mixed_cardiac_endothelial_review_flag"]
        | cells["doublet_percentile_ge_99"]
    )

    marker_lookup = {gene: i for i, gene in enumerate(marker_genes)}
    gene_to_panel = {gene: panel for panel, genes in panels.items() for gene in genes}
    marker_rows: list[dict[str, object]] = []
    for state in states:
        state_mask = cells["LEC_state"].eq(state).to_numpy()
        for gene in marker_genes:
            values = np.asarray(marker_matrix[state_mask, marker_lookup[gene]].toarray()).ravel()
            marker_rows.append(
                {
                    "LEC_state": state,
                    "gene": gene,
                    "panel": gene_to_panel[gene],
                    "n_LEC_cells": int(len(values)),
                    "mean_logexpr": float(np.mean(values)),
                    "median_logexpr": float(np.median(values)),
                    "pct_detected": float(np.mean(values > 0) * 100),
                }
            )
    marker_summary = pd.DataFrame(marker_rows)

    state_summary = summarize_cells(cells, ["LEC_state"])
    for panel in MARKER_PANELS:
        means = cells.groupby("LEC_state", observed=True)[f"{panel.lower()}_zscore_within_dataset_LECs"].mean()
        state_summary[f"mean_{panel}_zscore"] = state_summary["LEC_state"].map(means)
    donor_summary = summarize_cells(cells, ["dataset_folder", "condition", "donor_key"])
    cohort_summary = summarize_cells(cells, ["dataset_folder", "condition"])

    donor_state = cells.groupby(["LEC_state", "donor_key"], observed=True).size().rename("n").reset_index()
    totals = donor_state.groupby("LEC_state", observed=True)["n"].sum().to_dict()
    top = donor_state.sort_values(["LEC_state", "n"], ascending=[True, False]).drop_duplicates("LEC_state")
    top["top_donor_fraction"] = [float(n) / float(totals[state]) for state, n in zip(top["LEC_state"], top["n"])]
    top = top.rename(columns={"donor_key": "top_donor", "n": "top_donor_n_LECs"})
    state_summary = state_summary.merge(
        top[["LEC_state", "top_donor", "top_donor_n_LECs", "top_donor_fraction"]],
        on="LEC_state", how="left",
    )
    state_summary["prespecified_focus_state"] = state_summary["LEC_state"].isin(focus_states)

    def review_reasons(row: pd.Series) -> str:
        reasons: list[str] = []
        if bool(row["prespecified_focus_state"]):
            reasons.append("prespecified_focus")
        if float(row["fraction_mixed_cardiac_endothelial_review_flags"]) >= STATE_MIXED_MARKER_REVIEW_FRACTION:
            reasons.append("relative_cardiac_mixed_fraction_ge_5pct")
        value = row["fraction_doublet_percentile_ge_99_among_available"]
        if pd.notna(value) and float(value) >= STATE_DOUBLET_Q99_REVIEW_FRACTION:
            reasons.append("relative_doublet_q99_fraction_ge_8pct")
        if float(row["top_donor_fraction"]) > STATE_DONOR_DOMINANCE_REVIEW_FRACTION:
            reasons.append("single_donor_fraction_over_50pct")
        return "REVIEW_" + ";".join(reasons) if reasons else "DESCRIPTIVE_no_automatic_exclusion"

    state_summary["interpretation_flag"] = state_summary.apply(review_reasons, axis=1)

    donor_metadata_columns = [
        column for column in [
            "dataset_folder", "condition", "donor_key", "sex", "age", "lvef",
            "region_standardized_analysis",
        ] if column in cells.columns
    ]
    donor_meta = cells[donor_metadata_columns].drop_duplicates("donor_key").copy()
    donor_counts = cells.groupby("donor_key", observed=True).size().rename("n_LEC_cells")
    donor_states = cells.groupby("donor_key", observed=True)["LEC_state"].nunique().rename("n_LEC_states")
    donor_meta = donor_meta.merge(donor_counts, on="donor_key").merge(donor_states, on="donor_key")
    donor_meta["cohort"] = donor_meta["dataset_folder"].astype(str) + " | " + donor_meta["condition"].astype(str)
    donor_meta["age_numeric"] = safe_numeric(donor_meta["age"]) if "age" in donor_meta else np.nan
    sex = normalized_text(donor_meta["sex"]).str.casefold() if "sex" in donor_meta else pd.Series("", index=donor_meta.index)
    donor_meta["sex_standardized_QC"] = np.select(
        [sex.eq("female"), sex.eq("male")], ["Female", "Male"], default="Unknown"
    )
    donor_meta["has_at_least_one_final_LEC"] = True
    if len(donor_meta) != cells["donor_key"].nunique() or (donor_meta["n_LEC_cells"] <= 0).any():
        raise ValueError("LEC-only donor table contains an invalid or zero-LEC donor.")

    cohort_order: list[str] = []
    for dataset, order in [
        ("dataset01", ["NF", "DCM", "HCM"]),
        ("dataset08", ["NF", "DoxCM"]),
        ("dataset11", ["NF", "HFpEF"]),
    ]:
        for condition in order:
            label = f"{dataset} | {condition}"
            if label in set(donor_meta["cohort"]):
                cohort_order.append(label)

    availability = pd.DataFrame(
        [
            {"panel": panel, "gene": gene, "available_in_final_Step08_LEC_object": gene in available}
            for panel, genes in MARKER_PANELS.items() for gene in genes
        ]
    )
    input_validation = pd.DataFrame(
        [
            {
                "step08_h5ad": str(step08_path),
                "n_final_LEC_cells": int(len(cells)),
                "n_final_LEC_positive_donors": int(cells["donor_key"].nunique()),
                "n_states": int(len(states)),
                "datasets": ";".join(datasets),
                "conditions": ";".join(conditions),
                "regions_by_dataset": ";".join(f"{k}:{v}" for k, v in region_check.items()),
                "non_LEC_cells_used": 0,
                "zero_LEC_donors_used": 0,
                "status": "validated_final_Step08_LECs_only",
            }
        ]
    )

    cells.to_csv(tables_dir / "08_1_LEC_cell_identity_doublet_QC.csv.gz", index=False, compression="gzip")
    state_summary.to_csv(tables_dir / "08_1_state_identity_summary.csv", index=False)
    donor_summary.to_csv(tables_dir / "08_1_donor_identity_summary.csv", index=False)
    cohort_summary.to_csv(tables_dir / "08_1_dataset_condition_identity_summary.csv", index=False)
    donor_meta.to_csv(tables_dir / "08_1_final_LEC_donor_demographics.csv", index=False)
    marker_summary.to_csv(tables_dir / "08_1_marker_expression_by_state.csv", index=False)
    availability.to_csv(tables_dir / "08_1_signature_gene_availability.csv", index=False)
    doublet_thresholds.to_csv(tables_dir / "08_1_doublet_LEC_only_reference_thresholds.csv", index=False)
    input_validation.to_csv(tables_dir / "08_1_input_validation.csv", index=False)
    pd.DataFrame([{"LEC_state": state, "state_color": palette[state]} for state in states]).to_csv(
        tables_dir / "08_1_state_color_mapping.csv", index=False
    )

    figure_manifest: list[dict[str, str]] = []
    if not args.skip_figures:
        jobs = [
            ("08_1_final_LEC_demographics_and_support.png", "Demographics and support of final-LEC-positive donors only",
             lambda p: save_demographics(donor_meta, cohort_order, p, args.dpi)),
            ("08_1_state_signature_heatmap.png", "Relative lineage signatures among final LECs",
             lambda p: save_signature_heatmap(state_summary, states, palette, p, args.dpi)),
            ("08_1_state_marker_dotplot.png", "Marker expression among final LECs",
             lambda p: save_marker_dotplot(marker_summary, states, palette, p, args.dpi)),
            ("08_1_identity_vs_cardiac_scatter.png", "Relative identity/cardiac expression among final LECs",
             lambda p: save_identity_scatter(cells, states, palette, focus_donors, p, args.dpi)),
            ("08_1_doublet_QC.png", "Existing doublet scores ranked among final LECs only",
             lambda p: save_doublet_qc(cells, states, palette, p, args.dpi)),
            ("08_1_focus_states_and_donors_UMAP.png", "Focus states and donors among final LECs",
             lambda p: save_focus_umap(adata, cells, states, palette, focus_states, focus_donors, p, args.dpi)),
        ]
        if not args.skip_neighborhood:
            jobs.append(
                ("08_1_final_LEC_graph_neighbor_QC.png", "State and donor locality in the final-LEC Step 08 graph",
                 lambda p: save_neighbor_qc(state_summary, states, palette, p, args.dpi))
            )
        for filename, description, function in jobs:
            try:
                function(figures_dir / filename)
                figure_manifest.append({"figure": filename, "description": description, "status": "ok"})
            except Exception as error:
                warnings.warn(f"Figure failed ({filename}): {type(error).__name__}: {error}")
                figure_manifest.append({"figure": filename, "description": description,
                                        "status": f"failed: {type(error).__name__}: {error}"})
    pd.DataFrame(figure_manifest).to_csv(output_root / "08_1_figure_manifest.csv", index=False)

    dataset08_calls = sorted(set(cells.loc[cells["dataset_folder"].eq("dataset08"), "author_doublet_call"].astype(str)))
    notes = f"""Step 08.1 final-LEC-only identity and doublet QC
=====================================================

Scope
-----
All calculations use exactly the {len(cells):,} final cells from the current
Step 08 LEC clustering object. These cells represent {cells['donor_key'].nunique()}
donors, and every represented donor has at least one final LEC. No EC-only,
non-LEC, author-reference, or zero-LEC donor observations are used.

Expression
----------
The existing Step 08 log-normalized values are used without another
normalization. Signature z-scores and percentiles are standardized only among
final LECs within the same dataset. Consequently, a cardiac review flag means
high cardiac expression relative to other final LECs; it does not show that a
cell matches an external cardiomyocyte reference.

Doublet percentile
------------------
Existing upstream numeric scores are ranked among final LECs in the matching
technical sample. Samples with fewer than {args.minimum_sample_LECs_for_percentile}
numeric-score LECs use a final-LEC-only dataset/condition or dataset fallback.
The 95th and 99th percentiles are relative ranks, not probabilities and not
automatic doublet calls. Dataset08 has no numeric score; its retained author
calls are: {'; '.join(dataset08_calls) if dataset08_calls else 'none'}.

Neighborhood
------------
Neighborhood summaries use the Step 08 combined-LEC graph. Every graph node is
a final LEC. The outputs describe state and donor locality only; they do not
contain EC or cardiomyocyte neighbors.

Mutation policy
---------------
This workflow is read-only. It does not delete cells, change state labels,
recluster, or write an h5ad.
"""
    (output_root / "08_1_method_and_interpretation_notes.txt").write_text(notes, encoding="utf-8")

    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "step08_h5ad": str(step08_path),
        "output_root": str(output_root),
        "cell_scope": "exact_current_Step08_final_LECs_only",
        "n_final_LEC_cells": int(len(cells)),
        "n_final_LEC_positive_donors": int(cells["donor_key"].nunique()),
        "non_LEC_cells_used": 0,
        "zero_LEC_donors_used": 0,
        "states": states,
        "focus_states": focus_states,
        "focus_donors": focus_donors,
        "datasets": datasets,
        "conditions": conditions,
        "regions_by_dataset": region_check,
        "doublet_percentile_scope": "final_LECs_only_matching_sample_with_LEC_only_fallback",
        "minimum_sample_LECs_for_percentile": int(args.minimum_sample_LECs_for_percentile),
        "cardiac_signature_scope": "relative_among_final_LECs_within_dataset",
        "cardiac_LEC_quantile": float(args.cardiac_LEC_quantile),
        "minimum_cardiac_genes": int(args.minimum_cardiac_genes),
        "neighbors": int(args.neighbors),
        "neighborhood_scope": "Step08_final_LEC_graph_only",
        "marker_panels": MARKER_PANELS,
        "mutation_policy": "read_only_no_cell_removal_no_h5ad_write",
    }
    with (output_root / "08_1_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    run_summary = {
        "script_version": SCRIPT_VERSION,
        "status": "completed",
        "cell_scope": "exact_current_Step08_final_LECs_only",
        "n_final_LEC_cells": int(len(cells)),
        "n_final_LEC_positive_donors": int(cells["donor_key"].nunique()),
        "n_zero_LEC_donors": 0,
        "n_non_LEC_cells": 0,
        "n_states": int(len(states)),
        "n_numeric_doublet_scores": int(cells["doublet_score"].notna().sum()),
        "n_relative_mixed_cardiac_endothelial_flags": int(cells["mixed_cardiac_endothelial_review_flag"].sum()),
        "n_relative_doublet_percentile_ge_99": int(cells["doublet_percentile_ge_99"].sum()),
        "output_root": str(output_root),
    }
    pd.DataFrame([run_summary]).to_csv(output_root / "08_1_run_summary.csv", index=False)

    print("\nCompleted final-Step08-LEC-only QC.")
    print(pd.DataFrame([run_summary]).to_string(index=False))
    print("\nState summary:")
    print(state_summary[[
        "LEC_state", "n_LEC_cells", "n_LEC_positive_donors", "top_donor",
        "top_donor_fraction", "fraction_mixed_cardiac_endothelial_review_flags",
        "fraction_doublet_percentile_ge_99_among_available",
        "mean_neighbor_same_LEC_state_fraction", "interpretation_flag",
    ]].to_string(index=False))
    print(f"\nOutput: {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
