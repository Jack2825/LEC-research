#!/usr/bin/env python
r"""
Step 04: brief visualization of the selected disease Step 03 results.

This script mirrors the role of aging Step 04. It is read-only with respect to
Step 03: it does not recalculate QC, normalization, HVGs, Harmony, UMAP, Leiden,
or marker genes. It validates the completed cohort list, opens each clustered
h5ad in backed mode, and creates concise cohort, donor, QC, region, and UMAP
summaries.

Dataset02, dataset06, ICM/dataset03, RV-only dataset10, and dataset12 COVID are
excluded. The expected active cohorts are dataset01 DCM/HCM/NF, dataset08
DoxCM/NF (NICM excluded), and dataset11 HFpEF/NF.

Default input
-------------
new_result2_disease/03/

Default output
--------------
new_result2_disease/04/

PowerShell
----------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\04_visualize_selected_disease_datasets.py
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Patch

try:
    import scanpy as sc
except ImportError as error:
    raise ImportError(
        "scanpy is required. Run Step 04 with the project .venv."
    ) from error


SCRIPT_VERSION = "04_selected_disease_exclude_dataset02_dataset06_dataset12_2026_08_11"
DEFAULT_STEP03_ROOT = "new_result2_disease/03"
DEFAULT_OUTPUT_ROOT = "new_result2_disease/04"

EXPECTED_COHORTS = (
    ("dataset01", "DCM"),
    ("dataset01", "HCM"),
    ("dataset01", "NF"),
    ("dataset08", "DoxCM"),
    ("dataset08", "NF"),
    ("dataset11", "HFpEF"),
    ("dataset11", "NF"),
)
# Keep this semantic condition palette identical to disease Step 07.
CONDITION_ORDER = ("NF", "DCM", "HCM", "DoxCM", "HFpEF")
CONDITION_COLORS = {
    "NF": "#0072B2",
    "DCM": "#D55E00",
    "HCM": "#CC79A7",
    "DoxCM": "#56B4E9",
    "HFpEF": "#E69F00",
}
SEX_COLORS = {
    "Female": "#E377C2",
    "Male": "#4C78A8",
    "Unknown/other": "#B7B7B7",
}
REGION_ORDER = ("LV", "AP", "RV", "S", "Unknown")
REGION_COLORS = {
    "LV": "#4C78A8",
    "AP": "#F28E2B",
    "RV": "#E15759",
    "S": "#59A14F",
    "Unknown": "#B7B7B7",
}


def natural_key(value: object) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    ]


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    source: Path,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns in {source}: {missing}")


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def first_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def cohort_label(dataset_folder: str, condition: str) -> str:
    return f"{dataset_folder} | {condition}"


def axis_cohort_label(dataset_folder: str, condition: str) -> str:
    return f"{dataset_folder}\n{condition}"


def canonical_sex(value: object) -> str:
    text = str(value).strip().casefold()
    if text in {"female", "f", "woman"}:
        return "Female"
    if text in {"male", "m", "man"}:
        return "Male"
    return "Unknown/other"


def canonical_region(value: object) -> str:
    text = str(value).strip().upper()
    return text if text in {"LV", "AP", "RV", "S"} else "Unknown"


def mode_or_conflict(values: pd.Series) -> tuple[str, bool]:
    cleaned = normalized_text(values)
    cleaned = cleaned.loc[~cleaned.str.casefold().isin({"", "nan", "none"})]
    if cleaned.empty:
        return "Unknown", False
    counts = cleaned.value_counts()
    mode = str(counts.index[0])
    return mode, len(counts) > 1


def save_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def condition_legend_handles(include_before_qc: bool = False) -> list[Patch]:
    handles: list[Patch] = []
    if include_before_qc:
        handles.append(Patch(facecolor="#C7C7C7", label="Before QC"))
    handles.extend(
        Patch(facecolor=CONDITION_COLORS[condition], label=condition)
        for condition in CONDITION_ORDER
    )
    return handles


def validate_master(master: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = [
        "dataset_folder",
        "dataset_id",
        "condition",
        "status",
        "n_cells_before_QC",
        "n_cells_after_all_QC",
        "n_removed_total",
        "n_donors_before_QC",
        "n_donors_after_QC",
        "n_genes",
        "n_clusters",
        "regions_retained",
        "output_h5ad",
    ]
    require_columns(master, required, source)
    master = master.copy()
    master["dataset_folder"] = normalized_text(master["dataset_folder"])
    master["condition"] = normalized_text(master["condition"])
    forbidden = master["dataset_folder"].isin(
        {"dataset02", "dataset03", "dataset06", "dataset10", "dataset12"}
    )
    if forbidden.any():
        raise ValueError(
            "An excluded dataset is present in the Step 03 master summary: "
            f"{sorted(master.loc[forbidden, 'dataset_folder'].unique())}. "
            "dataset02, dataset06, ICM/dataset03, RV-only dataset10, and "
            "dataset12 COVID are "
            "excluded. Rerun the "
            "revised Step 03 before Step 04."
        )
    error_rows = master[master["status"].astype(str).str.casefold().eq("error")]
    if not error_rows.empty:
        raise ValueError(
            "Step 03 contains failed cohorts:\n"
            + error_rows[["dataset_folder", "condition"]].to_string(index=False)
        )
    if master.duplicated(["dataset_folder", "condition"]).any():
        raise ValueError("Step 03 master summary has duplicate cohort rows.")

    observed = set(zip(master["dataset_folder"], master["condition"]))
    expected = set(EXPECTED_COHORTS)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "Step 03 cohort list is incomplete or unexpected. "
            f"Missing={missing}; unexpected={unexpected}. Run Step 03 without "
            "dataset/condition subsetting to regenerate the complete master summary."
        )
    order = {pair: index for index, pair in enumerate(EXPECTED_COHORTS)}
    master["cohort_order"] = [
        order[(dataset, condition)]
        for dataset, condition in zip(master["dataset_folder"], master["condition"])
    ]
    master = master.sort_values("cohort_order").reset_index(drop=True)
    for column in [
        "n_cells_before_QC",
        "n_cells_after_all_QC",
        "n_removed_total",
        "n_donors_before_QC",
        "n_donors_after_QC",
        "n_genes",
        "n_clusters",
    ]:
        master[column] = pd.to_numeric(master[column], errors="raise")
    master["percent_cells_retained"] = np.divide(
        100.0 * master["n_cells_after_all_QC"],
        master["n_cells_before_QC"],
    )
    master["cohort"] = [
        cohort_label(dataset, condition)
        for dataset, condition in zip(master["dataset_folder"], master["condition"])
    ]
    return master


def h5ad_path_from_row(project_dir: Path, row: pd.Series) -> Path:
    path = resolve_project_path(project_dir, str(row["output_h5ad"]))
    if not path.exists():
        raise FileNotFoundError(f"Missing clustered Step 03 h5ad: {path}")
    return path


def sample_indices(n_cells: int, maximum: int, seed: int) -> np.ndarray:
    if maximum < 1:
        raise ValueError("Visualization sample sizes must be at least 1.")
    if n_cells <= maximum:
        return np.arange(n_cells, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_cells, size=maximum, replace=False))


def load_visualization_inputs(
    master: pd.DataFrame,
    project_dir: Path,
    max_cells_per_cohort: int,
    max_umap_cells_per_cohort: int,
    random_seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    donor_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []
    qc_parts: list[pd.DataFrame] = []
    umap_parts: list[pd.DataFrame] = []
    cluster_rows: list[dict[str, object]] = []
    region_count_rows: list[dict[str, object]] = []
    h5ad_rows: list[dict[str, object]] = []

    for index, row in master.iterrows():
        dataset = str(row["dataset_folder"])
        condition = str(row["condition"])
        cohort = str(row["cohort"])
        path = h5ad_path_from_row(project_dir, row)
        print(f"Reading {cohort}: {path}")
        adata = sc.read_h5ad(path, backed="r")
        try:
            if int(adata.n_obs) != int(row["n_cells_after_all_QC"]):
                raise ValueError(
                    f"Cell-count mismatch for {cohort}: master has "
                    f"{int(row['n_cells_after_all_QC']):,}, h5ad has "
                    f"{adata.n_obs:,}."
                )
            condition_column = first_column(
                adata.obs, ["condition_merged", "disease", "condition"]
            )
            if condition_column is None:
                raise KeyError(f"No condition column in {path}")
            observed_conditions = set(
                normalized_text(adata.obs[condition_column]).replace("", np.nan).dropna()
            )
            if observed_conditions != {condition}:
                raise ValueError(
                    f"Condition mismatch in {path}: expected {condition}, "
                    f"observed {sorted(observed_conditions)}"
                )
            require_columns(
                adata.obs,
                [
                    "donor_key",
                    "leiden",
                    "qc_n_genes",
                    "qc_total_counts",
                    "qc_pct_mt",
                    "region_standardized_analysis",
                ],
                path,
            )
            if "X_umap" not in adata.obsm:
                raise KeyError(f"X_umap is missing from {path}")

            obs = adata.obs
            donor_count = int(normalized_text(obs["donor_key"]).nunique())
            if donor_count != int(row["n_donors_after_QC"]):
                raise ValueError(
                    f"Donor-count mismatch for {cohort}: master has "
                    f"{int(row['n_donors_after_QC'])}, h5ad has {donor_count}."
                )
            cell_regions = normalized_text(
                obs["region_standardized_analysis"]
            ).map(canonical_region)
            for region, n_cells in cell_regions.value_counts().items():
                region_count_rows.append(
                    {
                        "dataset_folder": dataset,
                        "condition": condition,
                        "cohort": cohort,
                        "region": str(region),
                        "n_cells": int(n_cells),
                    }
                )
            sex_column = first_column(obs, ["sex", "Sex", "gender"])
            donor_frame = pd.DataFrame(
                {
                    "donor_key": normalized_text(obs["donor_key"]),
                    "region": normalized_text(obs["region_standardized_analysis"]),
                },
                index=obs.index,
            )
            if sex_column is not None:
                donor_frame["sex_source"] = normalized_text(obs[sex_column])
            else:
                donor_frame["sex_source"] = "Unknown"
            for donor_key, group in donor_frame.groupby("donor_key", observed=True):
                sex_value, sex_conflict = mode_or_conflict(group["sex_source"])
                source_regions = sorted(
                    set(normalized_text(group["region"])) - {""},
                    key=natural_key,
                )
                canonical_regions = sorted(
                    {canonical_region(value) for value in source_regions},
                    key=lambda value: REGION_ORDER.index(value),
                )
                region_multiple = len(canonical_regions) > 1
                donor_rows.append(
                    {
                        "dataset_folder": dataset,
                        "condition": condition,
                        "cohort": cohort,
                        "donor_key": donor_key,
                        "n_cells_after_QC": int(len(group)),
                        "sex_source": sex_value,
                        "sex": canonical_sex(sex_value),
                        "region_source": ";".join(source_regions) or "Unknown",
                        "region": ";".join(canonical_regions) or "Unknown",
                        "sex_conflict_within_donor": sex_conflict,
                        "multiple_regions_within_donor": region_multiple,
                    }
                )
                if sex_conflict or region_multiple:
                    conflict_rows.append(
                        {
                            "dataset_folder": dataset,
                            "condition": condition,
                            "donor_key": donor_key,
                            "sex_conflict": sex_conflict,
                            "multiple_regions": region_multiple,
                        }
                    )

            qc_index = sample_indices(
                adata.n_obs,
                max_cells_per_cohort,
                random_seed + 1009 * index,
            )
            qc = obs.iloc[qc_index][
                ["qc_n_genes", "qc_total_counts", "qc_pct_mt"]
            ].copy()
            qc.insert(0, "cohort", cohort)
            qc.insert(0, "condition", condition)
            qc.insert(0, "dataset_folder", dataset)
            qc_parts.append(qc.reset_index(drop=True))

            umap_index = sample_indices(
                adata.n_obs,
                max_umap_cells_per_cohort,
                random_seed + 2027 * index,
            )
            coordinates = np.asarray(adata.obsm["X_umap"])[umap_index]
            umap = pd.DataFrame(
                {
                    "dataset_folder": dataset,
                    "condition": condition,
                    "cohort": cohort,
                    "leiden": normalized_text(obs.iloc[umap_index]["leiden"]).to_numpy(),
                    "UMAP1": coordinates[:, 0],
                    "UMAP2": coordinates[:, 1],
                }
            )
            umap_parts.append(umap)

            counts = normalized_text(obs["leiden"]).value_counts()
            for leiden, n_cells in counts.items():
                cluster_rows.append(
                    {
                        "dataset_folder": dataset,
                        "condition": condition,
                        "cohort": cohort,
                        "leiden": str(leiden),
                        "n_cells": int(n_cells),
                        "percent_cells": 100.0 * int(n_cells) / adata.n_obs,
                    }
                )
            h5ad_rows.append(
                {
                    "dataset_folder": dataset,
                    "condition": condition,
                    "cohort": cohort,
                    "input_h5ad": str(path),
                    "n_cells": int(adata.n_obs),
                    "n_genes": int(adata.n_vars),
                    "n_donors": donor_count,
                    "n_clusters": int(counts.size),
                    "n_cells_sampled_for_QC_figures": int(len(qc_index)),
                    "n_cells_sampled_for_UMAP": int(len(umap_index)),
                    "condition_validation_column": condition_column,
                    "sex_source_column": sex_column or "",
                }
            )
        finally:
            adata.file.close()

    donors = pd.DataFrame(donor_rows)
    conflicts = pd.DataFrame(conflict_rows)
    sampled_qc = pd.concat(qc_parts, ignore_index=True)
    sampled_umap = pd.concat(umap_parts, ignore_index=True)
    clusters = pd.DataFrame(cluster_rows)
    region_counts = pd.DataFrame(region_count_rows)
    h5ad_summary = pd.DataFrame(h5ad_rows)
    if donors.duplicated(["cohort", "donor_key"]).any():
        raise ValueError("Donor summary contains duplicate cohort-donor rows.")
    return (
        donors,
        conflicts,
        sampled_qc,
        sampled_umap,
        clusters,
        region_counts,
        h5ad_summary,
    )


def leiden_color_map(labels: list[str]) -> dict[str, str]:
    ordered = sorted(set(map(str, labels)), key=natural_key)
    if len(ordered) <= 20:
        cmap = plt.get_cmap("tab20")
        return {label: to_hex(cmap(index % 20)) for index, label in enumerate(ordered)}
    cmap = plt.get_cmap("gist_ncar")
    return {
        label: to_hex(cmap(index / max(len(ordered) - 1, 1)))
        for index, label in enumerate(ordered)
    }


def create_summary_figures(
    master: pd.DataFrame,
    donors: pd.DataFrame,
    region_counts: pd.DataFrame,
    sampled_qc: pd.DataFrame,
    sampled_umap: pd.DataFrame,
    cluster_colors: dict[str, str],
    output_dir: Path,
    dpi: int,
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    labels = [
        axis_cohort_label(dataset, condition)
        for dataset, condition in zip(master["dataset_folder"], master["condition"])
    ]
    conditions = master["condition"].astype(str).tolist()
    condition_colors = [CONDITION_COLORS[value] for value in conditions]
    x = np.arange(len(master))
    cohort_figure_width = max(13.5, 1.25 * len(master))

    fig, ax = plt.subplots(figsize=(cohort_figure_width, 6.2))
    width = 0.38
    ax.bar(
        x - width / 2,
        master["n_cells_before_QC"],
        width,
        color="#C7C7C7",
        label="Before QC",
    )
    ax.bar(
        x + width / 2,
        master["n_cells_after_all_QC"],
        width,
        color=condition_colors,
        label="After QC",
    )
    for position, (_, row) in enumerate(master.iterrows()):
        ax.text(
            position + width / 2,
            float(row["n_cells_after_all_QC"]),
            f"{float(row['percent_cells_retained']):.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Number of cells")
    ax.set_title("Selected disease cohorts before and after Step 03 QC")
    ax.ticklabel_format(axis="y", style="plain")
    ax.legend(
        handles=condition_legend_handles(include_before_qc=True),
        title="After-QC condition colors",
        frameon=False,
    )
    path = output_dir / "01_cells_before_after_QC_by_cohort.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "Cells before and after QC"})

    fig, ax = plt.subplots(figsize=(cohort_figure_width, 5.8))
    retained = master["percent_cells_retained"].to_numpy(float)
    ax.bar(x, retained, color=condition_colors)
    ax.axhline(100, color="#777777", linewidth=0.8, linestyle="--")
    for position, (_, row) in enumerate(master.iterrows()):
        ax.text(
            position,
            retained[position] + 0.8,
            f"kept {retained[position]:.1f}%\nremoved {int(row['n_removed_total']):,}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylim(0, 112)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cells retained (%)")
    ax.set_title("Step 03 QC retention by dataset-condition cohort")
    ax.legend(handles=condition_legend_handles(), title="Condition", frameon=False)
    path = output_dir / "02_QC_retention_by_cohort.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "QC retention"})

    fig, ax = plt.subplots(figsize=(cohort_figure_width, 5.8))
    ax.bar(
        x - width / 2,
        master["n_donors_before_QC"],
        width,
        color="#C7C7C7",
        label="Before QC",
    )
    ax.bar(
        x + width / 2,
        master["n_donors_after_QC"],
        width,
        color=condition_colors,
        label="After QC",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Number of donors")
    ax.set_title("Donors retained after Step 03 QC")
    ax.legend(
        handles=condition_legend_handles(include_before_qc=True),
        title="After-QC condition colors",
        frameon=False,
    )
    path = output_dir / "03_donors_before_after_QC_by_cohort.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "Donors before and after QC"})

    donor_values = [
        donors.loc[donors["cohort"].eq(cohort), "n_cells_after_QC"].to_numpy(float)
        for cohort in master["cohort"]
    ]
    fig, ax = plt.subplots(figsize=(cohort_figure_width, 6.2))
    boxplot = ax.boxplot(donor_values, tick_labels=labels, patch_artist=True, showfliers=False)
    rng = np.random.default_rng(0)
    for position, (patch, color, values) in enumerate(
        zip(boxplot["boxes"], condition_colors, donor_values), 1
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.50)
        ax.scatter(
            np.full(len(values), position) + rng.normal(0, 0.055, len(values)),
            values,
            s=18,
            alpha=0.68,
            color=color,
        )
    ax.set_ylabel("QC-passed cells per donor")
    ax.set_title("Donor contributions within each disease cohort")
    ax.legend(handles=condition_legend_handles(), title="Condition", frameon=False)
    path = output_dir / "04_cells_per_donor_by_cohort.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "Cells per donor"})

    sex_counts = pd.crosstab(donors["cohort"], donors["sex"]).reindex(master["cohort"])
    sex_counts = sex_counts.reindex(columns=list(SEX_COLORS), fill_value=0)
    fig, ax = plt.subplots(figsize=(cohort_figure_width, 6.2))
    bottom = np.zeros(len(sex_counts), dtype=float)
    for sex in SEX_COLORS:
        values = sex_counts[sex].to_numpy(float)
        ax.bar(x, values, bottom=bottom, color=SEX_COLORS[sex], label=sex)
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_ylabel("Number of donors")
    ax.set_title("Donor sex distribution by dataset-condition cohort")
    ax.legend(frameon=False)
    path = output_dir / "05_donor_sex_distribution_by_cohort.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "Donor sex distribution"})

    dataset_order = list(dict.fromkeys(master["dataset_folder"].astype(str)))
    condition_cells = pd.pivot_table(
        master,
        index="dataset_folder",
        columns="condition",
        values="n_cells_after_all_QC",
        aggfunc="sum",
        fill_value=0,
    ).reindex(dataset_order)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bottom = np.zeros(len(condition_cells), dtype=float)
    for condition in CONDITION_ORDER:
        if condition not in condition_cells.columns:
            continue
        values = condition_cells[condition].to_numpy(float)
        ax.bar(
            dataset_order,
            values,
            bottom=bottom,
            color=CONDITION_COLORS[condition],
            label=condition,
        )
        bottom += values
    ax.set_ylabel("QC-passed cells")
    ax.set_title("Condition composition within each retained dataset")
    ax.ticklabel_format(axis="y", style="plain")
    ax.legend(frameon=False)
    path = output_dir / "06_condition_cell_composition_by_dataset.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "Condition composition"})

    region_cells = (
        region_counts.pivot_table(
            index="cohort",
            columns="region",
            values="n_cells",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(master["cohort"])
        .reindex(columns=REGION_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(cohort_figure_width, 6.2))
    bottom = np.zeros(len(region_cells), dtype=float)
    for region in REGION_ORDER:
        values = region_cells[region].to_numpy(float)
        if values.sum() == 0:
            continue
        ax.bar(x, values, bottom=bottom, color=REGION_COLORS[region], label=region)
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_ylabel("QC-passed cells")
    ax.set_title("Heart-region composition after the approved cohort restrictions")
    ax.ticklabel_format(axis="y", style="plain")
    ax.legend(frameon=False)
    path = output_dir / "07_region_cell_composition_by_cohort.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "Heart-region composition"})

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.4))
    qc_specs = [
        ("qc_n_genes", "Detected genes per cell", "Detected genes"),
        ("qc_total_counts", "Total counts per cell", "Total counts"),
        ("qc_pct_mt", "Mitochondrial percentage", "Percent mitochondrial"),
    ]
    for ax, (column, title, ylabel) in zip(axes, qc_specs):
        values = [
            pd.to_numeric(
                sampled_qc.loc[sampled_qc["cohort"].eq(cohort), column],
                errors="coerce",
            ).dropna().to_numpy()
            for cohort in master["cohort"]
        ]
        boxplot = ax.boxplot(
            values,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            vert=False,
        )
        for patch, color in zip(boxplot["boxes"], condition_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.58)
        ax.set_title(title)
        ax.set_xlabel(ylabel)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Cell-level distributions after Step 03 QC (sampled for display)")
    fig.tight_layout()
    path = output_dir / "08_cell_gene_count_mito_distributions_by_cohort.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "Cell QC distributions"})

    n_panels = len(master)
    n_columns = 4
    n_rows = math.ceil(n_panels / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(4.1 * n_columns, 3.8 * n_rows))
    axes_flat = np.asarray(axes).ravel()
    for panel, (_, row) in enumerate(master.iterrows()):
        ax = axes_flat[panel]
        cohort = str(row["cohort"])
        subset = sampled_umap[sampled_umap["cohort"].eq(cohort)]
        for leiden in sorted(subset["leiden"].unique(), key=natural_key):
            points = subset[subset["leiden"].eq(leiden)]
            ax.scatter(
                points["UMAP1"],
                points["UMAP2"],
                s=1.1,
                alpha=0.60,
                linewidths=0,
                rasterized=True,
                color=cluster_colors[str(leiden)],
            )
            if len(points) >= 3:
                ax.text(
                    float(points["UMAP1"].median()),
                    float(points["UMAP2"].median()),
                    str(leiden),
                    ha="center",
                    va="center",
                    fontsize=6,
                    weight="bold",
                    color="black",
                )
        ax.set_title(f"{row['dataset_folder']} | {row['condition']}\n{int(row['n_clusters'])} clusters")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP1", fontsize=7)
        ax.set_ylabel("UMAP2", fontsize=7)
    for panel in range(n_panels, len(axes_flat)):
        axes_flat[panel].axis("off")
    fig.suptitle(
        "Independent Step 03 Leiden clustering by disease cohort\n"
        "Numeric Leiden IDs are condition-specific; matching colors do not imply matching biology",
        y=1.01,
    )
    fig.tight_layout()
    path = output_dir / "09_UMAP_Leiden_by_cohort.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "Sampled UMAPs by Leiden cluster"})
    return manifest


def qc_sample_summary(sampled_qc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, condition, cohort), group in sampled_qc.groupby(
        ["dataset_folder", "condition", "cohort"], observed=True
    ):
        for metric in ["qc_n_genes", "qc_total_counts", "qc_pct_mt"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "dataset_folder": dataset,
                    "condition": condition,
                    "cohort": cohort,
                    "metric": metric,
                    "n_sampled_nonmissing": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def color_key_table(
    cluster_colors: dict[str, str],
    observed_regions: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for condition in CONDITION_ORDER:
        rows.append(
            {
                "variable": "condition",
                "label": condition,
                "color": CONDITION_COLORS[condition],
                "note": "Same condition uses the same color in all Step 04 figures",
            }
        )
    for sex, color in SEX_COLORS.items():
        rows.append({"variable": "sex", "label": sex, "color": color, "note": ""})
    for region in REGION_ORDER:
        if region not in observed_regions:
            continue
        rows.append(
            {
                "variable": "region",
                "label": region,
                "color": REGION_COLORS[region],
                "note": "S denotes the approved HFpEF septum exception",
            }
        )
    for leiden in sorted(cluster_colors, key=natural_key):
        rows.append(
            {
                "variable": "leiden",
                "label": leiden,
                "color": cluster_colors[leiden],
                "note": (
                    "Color is consistent for the numeric label, but Leiden labels "
                    "from independently clustered cohorts are not biological matches"
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brief visualization of the selected disease Step 03 cohorts."
    )
    parser.add_argument("--step03_root", default=DEFAULT_STEP03_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--max_cells_per_cohort", type=int, default=20000)
    parser.add_argument("--max_umap_cells_per_cohort", type=int, default=30000)
    parser.add_argument("--random_seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")
    project_dir = find_project_dir()
    step03_root = resolve_project_path(project_dir, args.step03_root)
    output_dir = resolve_project_path(project_dir, args.output_root)
    master_path = step03_root / "03_master_run_summary.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing Step 03 master summary: {master_path}")

    master = validate_master(pd.read_csv(master_path), master_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("Selected disease Step 04: brief visualization")
    print("Step 03 input:", step03_root)
    print("Step 04 output:", output_dir)
    print("Excluded: dataset02, dataset06, ICM/dataset03, RV-only dataset10, dataset12 COVID, and dataset08 NICM")
    print("=" * 88)

    (
        donors,
        conflicts,
        sampled_qc,
        sampled_umap,
        clusters,
        region_counts,
        h5ad_summary,
    ) = load_visualization_inputs(
        master,
        project_dir,
        args.max_cells_per_cohort,
        args.max_umap_cells_per_cohort,
        args.random_seed,
    )
    cluster_colors = leiden_color_map(clusters["leiden"].astype(str).tolist())
    manifest = create_summary_figures(
        master,
        donors,
        region_counts,
        sampled_qc,
        sampled_umap,
        cluster_colors,
        output_dir,
        args.dpi,
    )

    master.to_csv(output_dir / "04_cohort_summary.csv", index=False)
    donors.to_csv(output_dir / "04_donor_summary.csv", index=False)
    conflicts.to_csv(output_dir / "04_donor_metadata_conflicts.csv", index=False)
    clusters.to_csv(output_dir / "04_cluster_counts.csv", index=False)
    region_counts.to_csv(output_dir / "04_region_cell_counts.csv", index=False)
    h5ad_summary.to_csv(
        output_dir / "04_h5ad_visualization_input_summary.csv", index=False
    )
    qc_sample_summary(sampled_qc).to_csv(
        output_dir / "04_sampled_cell_QC_summary.csv", index=False
    )
    color_key_table(
        cluster_colors,
        set(region_counts["region"].astype(str)),
    ).to_csv(
        output_dir / "04_color_key.csv", index=False
    )
    manifest_frame = pd.DataFrame(manifest)
    manifest_frame.insert(0, "script_version", SCRIPT_VERSION)
    manifest_frame.to_csv(output_dir / "04_visualization_manifest.csv", index=False)

    unique_donors = int(donors["donor_key"].nunique())
    with open(output_dir / "04_visualization_summary.txt", "w", encoding="utf-8") as handle:
        handle.write(f"script_version: {SCRIPT_VERSION}\n")
        handle.write(f"step03_input: {step03_root}\n")
        handle.write(f"output: {output_dir}\n")
        handle.write("ICM_dataset03_included: False\n")
        handle.write("RV_only_dataset10_included: False\n")
        handle.write("dataset02_included: False\n")
        handle.write("dataset06_included: False\n")
        handle.write("dataset12_COVID_included: False\n")
        handle.write("dataset08_NICM_included: False\n")
        handle.write("dataset08_DoxCM_and_NF_included: True\n")
        handle.write(f"n_cohorts: {len(master)}\n")
        handle.write(f"n_cells_after_QC: {int(master['n_cells_after_all_QC'].sum())}\n")
        handle.write(f"sum_cohort_donor_counts: {int(master['n_donors_after_QC'].sum())}\n")
        handle.write(f"n_unique_dataset_namespaced_donors: {unique_donors}\n")
        handle.write(f"n_donors_with_metadata_multiple_values: {len(conflicts)}\n")
        handle.write(f"n_figures: {len(manifest)}\n")
        handle.write("marker_genes_calculated: False\n")
        handle.write("cell_QC_figure_stage: after_Step03_QC\n")
        handle.write(
            "Leiden_note: independently clustered cohort labels are not directly comparable\n"
        )

    print(f"Created {len(manifest)} figures and summary tables.")
    print("Manifest:", output_dir / "04_visualization_manifest.csv")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
