#!/usr/bin/env python
r"""
Step 04: brief visualization of the dataset13 NF-only aging quartile analysis.

This script is independent of the Step 03 Python file. It reads only the
tables and completed clustered h5ad files produced by Step 03; it never
recalculates quartiles, QC, Harmony, UMAP, Leiden clusters, or marker genes.
The h5ad files are opened in backed/read-only mode.

Default input
-------------
new_result2_aging/03/dataset13/

Default output
--------------
new_result2_aging/04/dataset13/

Example
-------
PowerShell:

    .\.venv\Scripts\python.exe `
      .\new_script2_aging\04_visualize_dataset13_age_quartiles.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scanpy as sc
except ImportError as error:
    raise ImportError(
        "scanpy is required. Activate the project .venv before running Step 04."
    ) from error


SCRIPT_VERSION = "04_dataset13_NF_only_age_quartile_visualization_2026_08_06"
DEFAULT_STEP03_ROOT = "new_result2_aging/03/dataset13"
DEFAULT_OUTPUT_ROOT = "new_result2_aging/04/dataset13"
REQUIRED_DISEASE = "NF"
QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]
UNIQUE_DONOR_COL = "donor_id_quartile_unique"
COLORS = {
    "Q1": "#4C78A8",
    "Q2": "#59A14F",
    "Q3": "#F28E2B",
    "Q4": "#E15759",
}


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_dir / path


def format_age(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def save_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in {source}: {missing}")


def first_metadata_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def normalized_text(values: pd.Series) -> pd.Series:
    """Convert object or categorical metadata to stripped nullable text."""
    return values.astype("string").fillna("Missing").str.strip()


def validate_nf_only(
    frame: pd.DataFrame,
    required_disease: str,
    source: Path,
) -> str:
    """Reject mixed-disease Step 03 results instead of plotting them as NF."""
    disease_column = first_metadata_column(frame, ["disease", "Disease"])
    if disease_column is None:
        raise KeyError(
            f"Cannot verify the NF-only cohort because {source} has no disease column."
        )
    values = normalized_text(frame[disease_column])
    keep = values.str.casefold().eq(required_disease.casefold())
    if not keep.all():
        observed = values.value_counts(dropna=False).to_dict()
        raise ValueError(
            "Step 04 requires NF-only Step 03 results, but mixed diseases were "
            f"found in {source}: {observed}. Rerun the updated Step 03 first."
        )
    return disease_column


def plot_stacked_category(
    donors: pd.DataFrame,
    metadata_column: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    values = donors[metadata_column].fillna("Missing").astype(str)
    values = values.replace(
        {"nan": "Missing", "None": "Missing", "": "Missing"}
    )
    counts = pd.crosstab(
        donors["age_quartile"], values, dropna=False
    ).reindex(QUARTILE_LABELS, fill_value=0)
    ax = counts.plot(
        kind="bar",
        stacked=True,
        figsize=(8.5, 5.2),
        colormap="tab20",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_title(title)
    ax.set_xlabel("Age quartile")
    ax.set_ylabel("Number of donors")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(
        title=metadata_column,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    save_figure(output_path, dpi)


def create_donor_figures(
    donors: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[dict]:
    """Create age, quartile, donor, sex, NF-confirmation, and project figures."""
    manifest: list[dict] = []
    age_values = pd.to_numeric(donors["age_numeric_resolved"], errors="coerce")
    summary_plot = summary.set_index("age_quartile").reindex(QUARTILE_LABELS)
    age_axis_min = 5.0 * np.floor((float(age_values.min()) - 5.0) / 5.0)
    age_axis_max = 5.0 * np.ceil((float(age_values.max()) + 5.0) / 5.0)
    age_major_ticks = np.arange(age_axis_min, age_axis_max + 0.1, 5.0)
    age_minor_ticks = np.arange(age_axis_min, age_axis_max + 0.1, 1.0)
    age_boundaries: list[tuple[float, str, bool]] = []
    for index in range(len(QUARTILE_LABELS) - 1):
        this_label = QUARTILE_LABELS[index]
        next_label = QUARTILE_LABELS[index + 1]
        this_max = float(summary_plot.loc[this_label, "age_max"])
        next_min = float(summary_plot.loc[next_label, "age_min"])
        tied = bool(np.isclose(this_max, next_min))
        boundary = this_max if tied else (this_max + next_min) / 2.0
        age_boundaries.append((boundary, f"{this_label}/{next_label}", tied))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    age_counts = pd.crosstab(
        age_values,
        donors["age_quartile"],
        dropna=False,
    ).reindex(columns=QUARTILE_LABELS, fill_value=0).sort_index()
    stacked_bottom = np.zeros(age_counts.shape[0], dtype=float)
    for label in QUARTILE_LABELS:
        values = age_counts[label].to_numpy(dtype=float)
        ax.bar(
            age_counts.index.to_numpy(dtype=float),
            values,
            bottom=stacked_bottom,
            width=0.42,
            align="center",
            alpha=0.78,
            color=COLORS[label],
            label=f"{label} (n={int(values.sum())})",
            edgecolor="white",
            linewidth=0.5,
        )
        stacked_bottom += values
    for boundary, boundary_label, tied in age_boundaries:
        ax.axvline(
            boundary,
            color="#555555" if not tied else "#8B008B",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.text(
            boundary + 0.15,
            0.97,
            boundary_label + (" tie" if tied else ""),
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="#555555" if not tied else "#8B008B",
        )
    ax.set_title("NF donor-age distribution by equal-count quartile")
    ax.set_xlabel(
        "Resolved age (years; each bar is centered on the displayed age value)"
    )
    ax.set_ylabel("Number of donors")
    ax.set_xlim(age_axis_min, age_axis_max)
    ax.set_xticks(age_major_ticks)
    ax.set_xticks(age_minor_ticks, minor=True)
    ax.grid(axis="x", which="major", alpha=0.22)
    ax.grid(axis="x", which="minor", alpha=0.07)
    ax.legend(frameon=False)
    path = output_dir / "01_donor_age_distribution_by_quartile.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "Donor age distribution"})

    sort_columns = ["age_numeric_resolved"]
    if UNIQUE_DONOR_COL in donors.columns:
        sort_columns.append(UNIQUE_DONOR_COL)
    ordered = donors.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for label in QUARTILE_LABELS:
        for age_source, marker, source_label in [
            ("exact", "o", "exact age"),
            ("reported_range_midpoint", "^", "range midpoint"),
        ]:
            mask = ordered["age_quartile"].eq(label) & ordered[
                "age_source"
            ].eq(age_source)
            if mask.any():
                ax.scatter(
                    ordered.loc[mask, "age_numeric_resolved"],
                    ordered.index[mask] + 1,
                    color=COLORS[label],
                    marker=marker,
                    s=30,
                    alpha=0.82,
                    label=f"{label}: {source_label}",
                )
    rank_boundaries = np.cumsum(
        [
            int(ordered["age_quartile"].eq(label).sum())
            for label in QUARTILE_LABELS[:-1]
        ]
    )
    for rank_boundary in rank_boundaries:
        ax.axhline(
            rank_boundary + 0.5,
            color="#999999",
            linestyle=":",
            linewidth=0.8,
        )
    for boundary, _, tied in age_boundaries:
        ax.axvline(
            boundary,
            color="#555555" if not tied else "#8B008B",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )
    ax.set_title("Sorted NF donor ages and quartile boundaries")
    ax.set_xlabel("Resolved age (years)")
    ax.set_ylabel("Donor rank after sorting by resolved age")
    ax.set_xlim(age_axis_min, age_axis_max)
    ax.set_xticks(age_major_ticks)
    ax.set_xticks(age_minor_ticks, minor=True)
    ax.grid(axis="x", which="major", alpha=0.18)
    ax.grid(axis="x", which="minor", alpha=0.06)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    path = output_dir / "02_sorted_donor_ages_exact_vs_midpoint.png"
    save_figure(path, dpi)
    manifest.append(
        {"file": path.name, "description": "Exact ages versus range midpoints"}
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].bar(
        QUARTILE_LABELS,
        summary_plot["n_donors"],
        color=[COLORS[label] for label in QUARTILE_LABELS],
    )
    axes[0].set_title("NF donors per quartile")
    axes[0].set_ylabel("Number of donors")
    for index, value in enumerate(summary_plot["n_donors"]):
        axes[0].text(index, value, f"{int(value)}", ha="center", va="bottom")
    axes[1].bar(
        QUARTILE_LABELS,
        summary_plot["n_cells_before_QC"],
        color=[COLORS[label] for label in QUARTILE_LABELS],
    )
    axes[1].set_title("Cells per quartile before QC")
    axes[1].set_ylabel("Number of cells")
    axes[1].ticklabel_format(axis="y", style="plain")
    for index, value in enumerate(summary_plot["n_cells_before_QC"]):
        axes[1].text(
            index,
            value,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    path = output_dir / "03_quartile_donor_and_cell_counts.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "Quartile sizes"})

    donor_cells = [
        donors.loc[
            donors["age_quartile"].eq(label), "n_cells_before_QC"
        ].astype(float).to_numpy()
        for label in QUARTILE_LABELS
    ]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    boxplot = ax.boxplot(
        donor_cells,
        tick_labels=QUARTILE_LABELS,
        patch_artist=True,
    )
    for patch, label in zip(boxplot["boxes"], QUARTILE_LABELS):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.48)
    for index, (label, values) in enumerate(zip(QUARTILE_LABELS, donor_cells), 1):
        ax.scatter(
            np.full(len(values), index) + rng.normal(0, 0.055, size=len(values)),
            values,
            color=COLORS[label],
            s=20,
            alpha=0.72,
        )
    ax.set_title("Cell contribution of each donor")
    ax.set_xlabel("Age quartile")
    ax.set_ylabel("Cells per donor before QC")
    path = output_dir / "04_cells_per_donor_by_quartile.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "Cells per donor"})

    category_specs = [
        (["sex", "Sex"], "NF donor sex distribution by quartile", "05_sex_distribution_by_quartile.png"),
        (["disease", "Disease"], "NF-only cohort confirmation by quartile", "06_disease_distribution_by_quartile.png"),
        (["Project", "project"], "NF donor source-project distribution by quartile", "07_project_distribution_by_quartile.png"),
    ]
    for candidates, title, filename in category_specs:
        metadata_column = first_metadata_column(donors, candidates)
        if metadata_column is None:
            print(f"Skipping {filename}; none of {candidates} is present.")
            continue
        path = output_dir / filename
        plot_stacked_category(donors, metadata_column, title, path, dpi)
        manifest.append({"file": path.name, "description": title})

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for y_position, label in enumerate(QUARTILE_LABELS):
        row = summary_plot.loc[label]
        low = float(row["age_min"])
        high = float(row["age_max"])
        ax.barh(
            y_position,
            width=max(high - low, 0.15),
            left=low,
            height=0.28,
            color=COLORS[label],
            alpha=0.88,
        )
        ax.scatter([low, high], [y_position, y_position], color=COLORS[label], s=45)
        ax.text(
            high + 0.7,
            y_position,
            f"{format_age(low)}-{format_age(high)} years; n={int(row['n_donors'])}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(np.arange(4), QUARTILE_LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Resolved donor age (years)")
    ax.set_title("Resolved-age range in each NF donor quartile")
    ax.set_xlim(age_axis_min, age_axis_max)
    ax.set_xticks(age_major_ticks)
    ax.set_xticks(age_minor_ticks, minor=True)
    for boundary, _, tied in age_boundaries:
        ax.axvline(
            boundary,
            color="#555555" if not tied else "#8B008B",
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
            zorder=0,
        )
    ax.grid(axis="x", which="major", alpha=0.22)
    ax.grid(axis="x", which="minor", alpha=0.07)
    path = output_dir / "08_resolved_age_range_by_quartile.png"
    save_figure(path, dpi)
    manifest.append({"file": path.name, "description": "Resolved-age ranges"})
    return manifest


def read_sampled_cell_qc(
    step03_root: Path,
    max_cells_per_quartile: int,
    random_seed: int,
    required_disease: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a reproducible cell sample from each completed Step 03 h5ad."""
    if max_cells_per_quartile < 1:
        raise ValueError("--max_cells_per_quartile must be at least 1.")
    columns = ["age_quartile", "qc_n_genes", "qc_total_counts", "qc_pct_mt"]
    obs_parts: list[pd.DataFrame] = []
    h5ad_rows: list[dict] = []
    for index, label in enumerate(QUARTILE_LABELS):
        h5ad_path = step03_root / label / f"dataset13_{label}_clustered.h5ad"
        if not h5ad_path.exists():
            raise FileNotFoundError(f"Missing Step 03 clustered file: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path, backed="r")
        try:
            disease_column = first_metadata_column(
                adata.obs, ["disease", "Disease"]
            )
            if disease_column is None:
                raise KeyError(
                    f"Cannot verify NF-only cells in {h5ad_path}; disease is absent."
                )
            disease_values = normalized_text(adata.obs[disease_column])
            if not disease_values.str.casefold().eq(
                required_disease.casefold()
            ).all():
                observed = disease_values.value_counts(dropna=False).to_dict()
                raise ValueError(
                    f"Mixed diseases were found in {h5ad_path}: {observed}. "
                    "Rerun the updated NF-only Step 03."
                )
            missing = [column for column in columns if column not in adata.obs.columns]
            if missing:
                raise KeyError(f"Missing columns in {h5ad_path}: {missing}")
            obs = adata.obs[columns].copy()
            n_cells = int(adata.n_obs)
            n_genes = int(adata.n_vars)
            if obs.shape[0] > max_cells_per_quartile:
                obs = obs.sample(
                    n=max_cells_per_quartile,
                    random_state=random_seed + index,
                )
            obs_parts.append(obs)
            h5ad_rows.append(
                {
                    "age_quartile": label,
                    "input_h5ad": str(h5ad_path),
                    "n_cells_after_step03_QC": n_cells,
                    "n_genes": n_genes,
                    "n_cells_sampled_for_visualization": int(obs.shape[0]),
                }
            )
        finally:
            adata.file.close()
    return pd.concat(obs_parts, ignore_index=True), pd.DataFrame(h5ad_rows)


def create_cell_qc_figures(
    sampled: pd.DataFrame,
    output_dir: Path,
    max_cells_per_quartile: int,
    dpi: int,
) -> list[dict]:
    """Visualize gene/count/mitochondrial distributions after Step 03 QC."""
    manifest: list[dict] = []
    color_list = [COLORS[label] for label in QUARTILE_LABELS]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    specs = [
        ("qc_n_genes", "Genes detected per cell", "Detected genes"),
        ("qc_total_counts", "Total counts per cell", "Total counts"),
        ("qc_pct_mt", "Mitochondrial percentage", "Percent mitochondrial"),
    ]
    for ax, (column, title, y_label) in zip(axes, specs):
        values = [
            pd.to_numeric(
                sampled.loc[sampled["age_quartile"].astype(str).eq(label), column],
                errors="coerce",
            ).dropna().to_numpy()
            for label in QUARTILE_LABELS
        ]
        boxplot = ax.boxplot(
            values,
            tick_labels=QUARTILE_LABELS,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(boxplot["boxes"], color_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.58)
        ax.set_title(title)
        ax.set_xlabel("Age quartile")
        ax.set_ylabel(y_label)
    fig.suptitle(
        "NF cell-level distributions after Step 03 QC "
        f"(up to {max_cells_per_quartile:,} cells per quartile)"
    )
    fig.tight_layout()
    path = output_dir / "09_cell_gene_count_mito_distributions_by_quartile.png"
    save_figure(path, dpi)
    manifest.append(
        {"file": path.name, "description": "Cell QC distributions after Step 03 QC"}
    )

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label in QUARTILE_LABELS:
        values = pd.to_numeric(
            sampled.loc[
                sampled["age_quartile"].astype(str).eq(label), "qc_n_genes"
            ],
            errors="coerce",
        ).dropna()
        ax.hist(
            values,
            bins=50,
            density=True,
            histtype="step",
            linewidth=1.7,
            color=COLORS[label],
            label=f"{label} (sample n={len(values):,})",
        )
    ax.set_title("Genes detected per NF cell after Step 03 QC")
    ax.set_xlabel("Detected genes")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    path = output_dir / "10_genes_detected_per_cell_distribution.png"
    save_figure(path, dpi)
    manifest.append(
        {"file": path.name, "description": "Detected-gene distribution after QC"}
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Brief visualization of completed dataset13 NF-only age-quartile "
            "Step 03 results."
        )
    )
    parser.add_argument("--step03_root", default=DEFAULT_STEP03_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--max_cells_per_quartile", type=int, default=20000)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument(
        "--skip_cell_qc",
        action="store_true",
        help="Create donor-level figures without opening the large h5ad files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = find_project_dir()
    step03_root = resolve_project_path(project_dir, args.step03_root)
    output_dir = resolve_project_path(project_dir, args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    donor_csv = step03_root / "dataset13_age_quartile_donor_assignments.csv"
    summary_csv = step03_root / "dataset13_age_quartile_summary_before_QC.csv"
    if not donor_csv.exists():
        raise FileNotFoundError(f"Missing Step 03 donor table: {donor_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing Step 03 quartile summary: {summary_csv}")

    donors = pd.read_csv(donor_csv)
    summary = pd.read_csv(summary_csv)
    require_columns(
        donors,
        [
            "age_quartile",
            "age_numeric_resolved",
            "age_source",
            "n_cells_before_QC",
        ],
        donor_csv,
    )
    require_columns(
        summary,
        ["age_quartile", "n_donors", "n_cells_before_QC", "age_min", "age_max"],
        summary_csv,
    )

    print("=" * 88)
    required_disease = REQUIRED_DISEASE
    disease_column = validate_nf_only(donors, required_disease, donor_csv)

    print("Dataset13 NF-only age-quartile Step 04: brief visualization")
    print("Step 03 input:", step03_root)
    print("Step 04 output:", output_dir)
    print("Verified disease:", required_disease)
    print("=" * 88)

    manifest = create_donor_figures(donors, summary, output_dir, args.dpi)
    h5ad_summary = pd.DataFrame()
    if not args.skip_cell_qc:
        sampled, h5ad_summary = read_sampled_cell_qc(
            step03_root,
            args.max_cells_per_quartile,
            args.random_seed,
            required_disease,
        )
        manifest.extend(
            create_cell_qc_figures(
                sampled,
                output_dir,
                args.max_cells_per_quartile,
                args.dpi,
            )
        )
        h5ad_summary.to_csv(
            output_dir / "04_h5ad_visualization_input_summary.csv", index=False
        )

    manifest_frame = pd.DataFrame(manifest)
    manifest_frame.insert(0, "script_version", SCRIPT_VERSION)
    manifest_frame.to_csv(output_dir / "04_visualization_manifest.csv", index=False)

    exact_count = int(donors["age_source"].eq("exact").sum())
    midpoint_count = int(
        donors["age_source"].eq("reported_range_midpoint").sum()
    )
    with open(
        output_dir / "04_visualization_summary.txt", "w", encoding="utf-8"
    ) as handle:
        handle.write(f"script_version: {SCRIPT_VERSION}\n")
        handle.write(f"step03_input: {step03_root}\n")
        handle.write(f"output: {output_dir}\n")
        handle.write(f"required_disease: {required_disease}\n")
        handle.write(f"verified_disease_column: {disease_column}\n")
        handle.write(f"n_donors: {len(donors)}\n")
        handle.write(f"n_exact_age_donors: {exact_count}\n")
        handle.write(f"n_range_midpoint_age_donors: {midpoint_count}\n")
        handle.write(f"n_figures: {len(manifest)}\n")
        handle.write("cell_QC_figure_stage: after_Step03_QC\n")

    print(f"Created {len(manifest)} figures.")
    print("Manifest:", output_dir / "04_visualization_manifest.csv")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
