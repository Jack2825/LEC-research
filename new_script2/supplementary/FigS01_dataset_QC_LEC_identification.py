#!/usr/bin/env python3
"""Build Supplementary Figure S1 from the existing aging and disease results.

The script writes every panel separately and as one combined figure in PNG, PDF,
and JPG formats. It also writes one consolidated source-data table and one input
manifest so every plotted value can be traced to its analysis output.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "new_result2" / "supplementary" / "FigS01_dataset_QC_LEC_identification"

INPUTS = {
    "aging_step03_qc": ROOT
    / "new_result2_aging"
    / "03"
    / "03_master_run_summary_age_quartile_QC.csv",
    "aging_donor_assignments": ROOT
    / "new_result2_aging"
    / "03"
    / "dataset13"
    / "dataset13_age_quartile_donor_assignments.csv",
    "aging_ec_lec_summary": ROOT
    / "new_result2_aging"
    / "06_EC_recluster"
    / "dataset13"
    / "06_EC_recluster_run_summary.csv",
    "aging_ec_marker_summary": ROOT
    / "new_result2_aging"
    / "06_EC_recluster"
    / "dataset13"
    / "06_EC_recluster_marker_summary_all_quartiles.csv",
    "aging_donor_lec_counts": ROOT
    / "new_result2_aging"
    / "08_combined_LEC_recluster"
    / "dataset13"
    / "08_donor_LEC_counts.csv",
    "disease_step03_qc": ROOT
    / "new_result2_disease"
    / "03"
    / "03_master_run_summary.csv",
    "disease_ec_lec_summary": ROOT
    / "new_result2_disease"
    / "06_EC_recluster"
    / "06_EC_recluster_run_summary.csv",
    "disease_ec_marker_summary": ROOT
    / "new_result2_disease"
    / "06_EC_recluster"
    / "06_EC_recluster_marker_summary_all_cohorts.csv",
    "disease_donor_lec_counts": ROOT
    / "new_result2_disease"
    / "08_combined_LEC_recluster"
    / "08_donor_LEC_counts.csv",
}

NAVY = "#2C5F8A"
TEAL = "#3A8D8C"
GOLD = "#C6902F"
SLATE = "#687786"
LIGHT = "#D9DEE3"
PALE = "#EEF2F5"
DARK = "#202529"
EXCLUDED = "#C7CDD3"

COHORT_ORDER = [
    "Aging Q1",
    "Aging Q2",
    "Aging Q3",
    "Aging Q4",
    "D01 DCM",
    "D01 HCM",
    "D01 NF",
    "D08 DoxCM",
    "D08 NF",
    "D11 HFpEF",
    "D11 NF",
]

DISEASE_LABELS = {
    "dataset01 | DCM": "D01 DCM",
    "dataset01 | HCM": "D01 HCM",
    "dataset01 | NF": "D01 NF",
    "dataset08 | DoxCM": "D08 DoxCM",
    "dataset08 | NF": "D08 NF",
    "dataset11 | HFpEF": "D11 HFpEF",
    "dataset11 | NF": "D11 NF",
}

MARKER_GROUPS = {
    "Endothelial identity": ["PECAM1", "VWF", "EMCN"],
    "LEC identity": ["PROX1", "FLT4", "PDPN", "LYVE1", "CCL21", "MMRN1", "NRP2"],
    "Lineage exclusion": ["COL1A1", "PTPRC", "RGS5", "ACTA2", "TNNT2"],
}
MARKERS = [gene for genes in MARKER_GROUPS.values() for gene in genes]


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
            "legend.fontsize": 8.5,
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


def fmt_count(value: float | int) -> str:
    value = int(round(float(value)))
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 100_000:
        return f"{value / 1_000:.0f}k"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def cohort_separator(ax: plt.Axes, y: float = 3.5) -> None:
    ax.axhline(y, color="#AAB2BA", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#DDE2E6", linewidth=0.7, zorder=0)
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


def build_tables(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    age_qc = data["aging_step03_qc"].copy()
    age_assign = data["aging_donor_assignments"].copy()
    age_yield = data["aging_ec_lec_summary"].copy()
    age_donor = data["aging_donor_lec_counts"].copy()
    disease_qc = data["disease_step03_qc"].copy()
    disease_yield = data["disease_ec_lec_summary"].copy()
    disease_donor = data["disease_donor_lec_counts"].copy()

    age_all_donors = age_assign.groupby("age_quartile")["donor_id_quartile_unique"].nunique()
    age_positive_donors = age_donor.groupby("age_quartile")["donor_key"].nunique()
    disease_positive_donors = disease_donor.groupby(["dataset_folder", "condition"])[
        "donor_key"
    ].nunique()

    cohort_records: list[dict[str, object]] = []
    for _, row in age_qc.iterrows():
        quartile = str(row["condition"])
        yield_row = age_yield.loc[age_yield["age_quartile"].astype(str).eq(quartile)].iloc[0]
        cohort_records.append(
            {
                "analysis": "Aging",
                "cohort": f"Aging {quartile}",
                "dataset": "dataset13",
                "condition": quartile,
                "n_donors_all_cell": int(age_all_donors.loc[quartile]),
                "n_donors_LEC_positive": int(age_positive_donors.get(quartile, 0)),
                "n_cells_before_step03_QC": int(row["n_cells_start"]),
                "n_cells_after_step03_QC": int(row["n_cells_after_meanIQR_QC"]),
                "percent_cells_retained": float(row["percent_final_kept"]),
                "step03_status": str(row["status"]),
                "n_EC_candidates": int(yield_row["n_EC_cells_after_filter"]),
                "n_selected_LECs": int(yield_row["n_selected_LEC_cells"]),
                "selected_LEC_cluster": str(yield_row["selected_LEC_clusters"]),
            }
        )

    for _, row in disease_qc.iterrows():
        key = (str(row["dataset_folder"]), str(row["condition"]))
        cohort_key = f"{key[0]} | {key[1]}"
        yield_row = disease_yield.loc[disease_yield["cohort"].astype(str).eq(cohort_key)].iloc[0]
        cohort_records.append(
            {
                "analysis": "Disease",
                "cohort": DISEASE_LABELS[cohort_key],
                "dataset": key[0],
                "condition": key[1],
                "n_donors_all_cell": int(row["n_donors_after_QC"]),
                "n_donors_LEC_positive": int(disease_positive_donors.get(key, 0)),
                "n_cells_before_step03_QC": int(row["n_cells_before_QC"]),
                "n_cells_after_step03_QC": int(row["n_cells_after_all_QC"]),
                "percent_cells_retained": 100.0
                * float(row["n_cells_after_all_QC"])
                / float(row["n_cells_before_QC"]),
                "step03_status": str(row["status"]),
                "n_EC_candidates": int(yield_row["n_EC_cells"]),
                "n_selected_LECs": int(yield_row["n_LEC_cells"]),
                "selected_LEC_cluster": str(yield_row["selected_LEC_clusters"]),
            }
        )

    cohorts = pd.DataFrame(cohort_records)
    cohorts["cohort"] = pd.Categorical(cohorts["cohort"], COHORT_ORDER, ordered=True)
    cohorts = cohorts.sort_values("cohort").reset_index(drop=True)
    cohorts["cohort"] = cohorts["cohort"].astype(str)

    age_donor_records = age_donor.assign(
        analysis="Aging",
        dataset="dataset13",
        condition=age_donor["age_quartile"].astype(str),
        cohort="Aging " + age_donor["age_quartile"].astype(str),
    )[["analysis", "dataset", "condition", "cohort", "donor_key", "n_LEC_cells"]]
    disease_donor_records = disease_donor.assign(
        analysis="Disease",
        dataset=disease_donor["dataset_folder"].astype(str),
        cohort=disease_donor.apply(
            lambda row: DISEASE_LABELS[f"{row['dataset_folder']} | {row['condition']}"], axis=1
        ),
    )[["analysis", "dataset", "condition", "cohort", "donor_key", "n_LEC_cells"]]
    donors = pd.concat([age_donor_records, disease_donor_records], ignore_index=True)
    donors["n_LEC_cells"] = donors["n_LEC_cells"].astype(int)
    donors["passes_LEC25"] = donors["n_LEC_cells"].ge(25)

    markers = build_marker_table(data, cohorts)
    validate_tables(cohorts, donors, markers)
    return cohorts, donors, markers


def build_marker_table(data: dict[str, pd.DataFrame], cohorts: pd.DataFrame) -> pd.DataFrame:
    age = data["aging_ec_marker_summary"].copy()
    disease = data["disease_ec_marker_summary"].copy()
    selected = cohorts.set_index("cohort")["selected_LEC_cluster"].to_dict()

    marker_records: list[dict[str, object]] = []
    for _, row in age.iterrows():
        cohort = f"Aging {row['age_quartile']}"
        if float(row["ec_leiden"]) != float(selected[cohort]):
            continue
        for group, genes in MARKER_GROUPS.items():
            for gene in genes:
                marker_records.append(
                    {
                        "analysis": "Aging",
                        "cohort": cohort,
                        "selected_LEC_cluster": str(int(float(row["ec_leiden"]))),
                        "marker_group": group,
                        "gene": gene,
                        "mean_log1p_expression": float(row[f"mean_expr_{gene}"]),
                        "fraction_expressing": float(row[f"pct_expr_{gene}"]),
                    }
                )

    for _, row in disease.iterrows():
        cohort_key = str(row["cohort"])
        cohort = DISEASE_LABELS.get(cohort_key)
        if cohort is None or float(row["ec_leiden"]) != float(selected[cohort]):
            continue
        for group, genes in MARKER_GROUPS.items():
            for gene in genes:
                marker_records.append(
                    {
                        "analysis": "Disease",
                        "cohort": cohort,
                        "selected_LEC_cluster": str(int(float(row["ec_leiden"]))),
                        "marker_group": group,
                        "gene": gene,
                        "mean_log1p_expression": float(row[f"mean_expr_{gene}"]),
                        "fraction_expressing": float(row[f"pct_expr_{gene}"]),
                    }
                )
    return pd.DataFrame(marker_records)


def validate_tables(cohorts: pd.DataFrame, donors: pd.DataFrame, markers: pd.DataFrame) -> None:
    if cohorts["cohort"].tolist() != COHORT_ORDER:
        raise ValueError("Cohort ordering or membership does not match the active atlases.")
    if len(donors.query("analysis == 'Aging'")) != 48:
        raise ValueError("Expected 48 aging donors with at least one LEC.")
    if len(donors.query("analysis == 'Disease'")) != 95:
        raise ValueError("Expected 95 disease donors with at least one LEC.")
    if int(cohorts.query("analysis == 'Aging'")["n_selected_LECs"].sum()) != 1119:
        raise ValueError("Aging selected-LEC total does not equal the combined atlas total (1,119).")
    if int(cohorts.query("analysis == 'Disease'")["n_selected_LECs"].sum()) != 6291:
        raise ValueError("Disease selected-LEC total does not equal the combined atlas total (6,291).")
    expected_marker_rows = len(COHORT_ORDER) * len(MARKERS)
    if len(markers) != expected_marker_rows:
        raise ValueError(
            f"Expected {expected_marker_rows} selected-cluster marker rows; found {len(markers)}."
        )
    if not markers["fraction_expressing"].between(0, 1).all():
        raise ValueError("Marker fractions must lie between 0 and 1.")


def draw_panel_a(fig: plt.Figure, spec, cohorts: pd.DataFrame) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 2, width_ratios=[0.85, 1.45], wspace=0.12)
    ax_donor = fig.add_subplot(grid[0, 0])
    ax_cells = fig.add_subplot(grid[0, 1])
    y = np.arange(len(cohorts))

    ax_donor.barh(
        y,
        cohorts["n_donors_all_cell"],
        height=0.64,
        color=LIGHT,
        edgecolor="none",
        label="All-cell QC object",
        zorder=1,
    )
    ax_donor.barh(
        y,
        cohorts["n_donors_LEC_positive"],
        height=0.34,
        color=NAVY,
        edgecolor="none",
        label="At least 1 final LEC",
        zorder=2,
    )
    ax_donor.set_yticks(y, cohorts["cohort"])
    ax_donor.set_ylim(len(cohorts) - 0.5, -1.0)
    ax_donor.set_xlabel("Donors")
    ax_donor.set_title("Donor representation", loc="left", pad=10)
    ax_donor.set_xlim(0, max(26, int(cohorts["n_donors_all_cell"].max()) + 3))
    ax_donor.legend(frameon=False, loc="upper right", handlelength=1.4)
    clean_axis(ax_donor, "x")
    cohort_separator(ax_donor)
    panel_letter(ax_donor, "A", x=-0.33)

    for idx, row in cohorts.iterrows():
        start = float(row["n_cells_before_step03_QC"])
        final = float(row["n_cells_after_step03_QC"])
        ax_cells.plot([final, start], [idx, idx], color="#AEB7BF", linewidth=1.6, zorder=1)
        ax_cells.scatter(
            start,
            idx,
            s=42,
            facecolor="white",
            edgecolor=SLATE,
            linewidth=1.2,
            zorder=3,
        )
        ax_cells.scatter(final, idx, s=42, color=TEAL, edgecolor="white", linewidth=0.5, zorder=4)
        ax_cells.annotate(
            f"{row['percent_cells_retained']:.1f}%",
            (max(start, final), idx),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            color=SLATE,
        )
    ax_cells.set_xscale("log")
    ax_cells.set_xlim(1.5e4, 3.8e5)
    ax_cells.set_yticks(y)
    ax_cells.set_yticklabels([])
    ax_cells.set_ylim(len(cohorts) - 0.5, -1.0)
    ax_cells.set_xlabel("Cells / nuclei")
    ax_cells.set_title("Step 03 QC retention", loc="left", pad=10)
    ax_cells.text(
        0.99,
        0.99,
        "open: entering Step 03   filled: retained\n"
        "disease rows: imported/reused Step 03 QC objects",
        transform=ax_cells.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=SLATE,
    )
    clean_axis(ax_cells, "x")
    cohort_separator(ax_cells)
    return [ax_donor, ax_cells]


def draw_panel_b(fig: plt.Figure, spec, cohorts: pd.DataFrame) -> plt.Axes:
    ax = fig.add_subplot(spec)
    y = np.arange(len(cohorts))
    for idx, row in cohorts.iterrows():
        values = [
            float(row["n_cells_after_step03_QC"]),
            float(row["n_EC_candidates"]),
            float(row["n_selected_LECs"]),
        ]
        ax.plot(values, [idx] * 3, color="#AFB8C0", linewidth=1.5, zorder=1)
        ax.scatter(values[0], idx, s=46, color=SLATE, edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(values[1], idx, s=46, color=TEAL, edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(values[2], idx, s=46, color=GOLD, edgecolor="white", linewidth=0.5, zorder=3)
        ax.annotate(
            fmt_count(values[2]),
            (values[2], idx),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7.3,
            color="#6E531B",
        )
    ax.set_xscale("log")
    ax.set_xlim(75, 4.2e5)
    ax.set_yticks(y, cohorts["cohort"])
    ax.invert_yaxis()
    ax.set_xlabel("Cells / nuclei (log scale)")
    ax.set_title("EC-to-LEC selection yield", loc="left", pad=10)
    legend = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=SLATE, markeredgecolor="white", label="Post-QC"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=TEAL, markeredgecolor="white", label="EC candidates"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=GOLD, markeredgecolor="white", label="Selected LECs"),
    ]
    ax.legend(
        handles=legend,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        handletextpad=0.3,
        columnspacing=0.8,
    )
    clean_axis(ax, "x")
    cohort_separator(ax)
    panel_letter(ax, "B", x=-0.25)
    return ax


def draw_panel_c(fig: plt.Figure, spec, donors: pd.DataFrame) -> list[plt.Axes]:
    grid = spec.subgridspec(1, 2, wspace=0.12)
    axes: list[plt.Axes] = []
    for idx, analysis in enumerate(["Aging", "Disease"]):
        ax = fig.add_subplot(grid[0, idx])
        subset = donors.query("analysis == @analysis").sort_values("n_LEC_cells").reset_index(drop=True)
        x = np.arange(1, len(subset) + 1)
        colors = np.where(subset["passes_LEC25"], NAVY if analysis == "Aging" else TEAL, EXCLUDED)
        ax.scatter(x, subset["n_LEC_cells"], c=colors, s=27, edgecolor="white", linewidth=0.35, zorder=3)
        ax.axhline(25, color="#A26B19", linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 600)
        ax.set_yticks([1, 5, 10, 25, 100, 500])
        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel("Donors ranked by LEC count")
        ax.set_title(f"{analysis} (LEC-positive donors)", loc="left", pad=8)
        retained = int(subset["passes_LEC25"].sum())
        ax.text(
            0.98,
            0.08,
            f"LEC25 retained: {retained}/{len(subset)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.2,
            color=DARK,
        )
        clean_axis(ax, "y")
        if idx == 0:
            ax.set_ylabel("Final LECs per donor (log scale)")
            panel_letter(ax, "C", x=-0.30, y=1.18)
        else:
            ax.set_yticklabels([])
    axes.extend(fig.axes[-2:])
    return axes


def cumulative_donor_table(donors: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for analysis, subset in donors.groupby("analysis", sort=False):
        ranked = subset.sort_values("n_LEC_cells", ascending=False).reset_index(drop=True)
        total = ranked["n_LEC_cells"].sum()
        for rank, row in ranked.iterrows():
            records.append(
                {
                    "analysis": analysis,
                    "donor_rank_descending": rank + 1,
                    "cumulative_donor_percent": 100.0 * (rank + 1) / len(ranked),
                    "cumulative_LEC_percent": 100.0
                    * ranked.loc[:rank, "n_LEC_cells"].sum()
                    / total,
                    "donor_key": row["donor_key"],
                    "n_LEC_cells": int(row["n_LEC_cells"]),
                }
            )
    return pd.DataFrame(records)


def draw_panel_d(fig: plt.Figure, spec, donors: pd.DataFrame) -> plt.Axes:
    ax = fig.add_subplot(spec)
    cumulative = cumulative_donor_table(donors)
    for analysis, color in [("Aging", NAVY), ("Disease", TEAL)]:
        subset = cumulative.query("analysis == @analysis")
        ax.plot(
            np.r_[0, subset["cumulative_donor_percent"]],
            np.r_[0, subset["cumulative_LEC_percent"]],
            color=color,
            linewidth=2.1,
            label=f"{analysis} (n={len(subset)})",
        )
        nearest = subset.iloc[(subset["cumulative_donor_percent"] - 20).abs().argmin()]
        ax.scatter(nearest["cumulative_donor_percent"], nearest["cumulative_LEC_percent"], color=color, s=25, zorder=4)
        ax.annotate(
            f"{nearest['cumulative_LEC_percent']:.0f}%",
            (nearest["cumulative_donor_percent"], nearest["cumulative_LEC_percent"]),
            xytext=(5, -3 if analysis == "Disease" else 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
            fontweight="bold",
        )
    ax.plot([0, 100], [0, 100], color="#BFC5CA", linewidth=1, linestyle=(0, (3, 3)), label="Equal contribution")
    ax.axvline(20, color="#D7DCE0", linewidth=0.8, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    ax.set_xlabel("Cumulative donors, ranked by LEC count (%)")
    ax.set_ylabel("Cumulative final LECs (%)")
    ax.set_title("Donor contribution to the final LEC atlases", loc="left", pad=10)
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax, "both")
    panel_letter(ax, "D", x=-0.23)
    return ax


def draw_panel_e(fig: plt.Figure, spec, markers: pd.DataFrame, cohorts: pd.DataFrame) -> plt.Axes:
    ax = fig.add_subplot(spec)
    x_map = {gene: idx for idx, gene in enumerate(MARKERS)}
    y_map = {cohort: idx for idx, cohort in enumerate(COHORT_ORDER)}
    selected = cohorts.set_index("cohort")["selected_LEC_cluster"].to_dict()

    x = markers["gene"].map(x_map).to_numpy()
    y = markers["cohort"].map(y_map).to_numpy()
    fractions = markers["fraction_expressing"].to_numpy(dtype=float)
    means = markers["mean_log1p_expression"].to_numpy(dtype=float)
    sizes = 12 + 170 * fractions
    norm = mpl.colors.Normalize(vmin=0, vmax=max(3.5, float(np.nanmax(means))))
    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        c=means,
        cmap="Blues",
        norm=norm,
        edgecolor="#6F7D88",
        linewidth=0.35,
        zorder=3,
    )

    row_labels = [f"{cohort}  ·  EC{selected[cohort]}" for cohort in COHORT_ORDER]
    ax.set_xticks(np.arange(len(MARKERS)), MARKERS, rotation=45, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=3)
    ax.set_yticks(np.arange(len(COHORT_ORDER)), row_labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.6, len(MARKERS) - 0.4)
    ax.set_ylim(len(COHORT_ORDER) - 0.35, -0.65)
    ax.set_title(
        "Selected-cluster endothelial/LEC identity and lineage exclusion",
        loc="left",
        y=1.28,
        pad=0,
    )
    ax.set_facecolor("white")
    ax.grid(color="#E4E8EB", linewidth=0.65, zorder=0)
    cohort_separator(ax)

    starts = [0, len(MARKER_GROUPS["Endothelial identity"]), 10]
    ends = [2, 9, 14]
    group_colors = [NAVY, TEAL, SLATE]
    for (group, _), start, end, color in zip(MARKER_GROUPS.items(), starts, ends, group_colors):
        center = (start + end) / 2
        ax.text(
            center,
            1.14,
            group,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            color=color,
            clip_on=False,
        )
        ax.plot(
            [start - 0.38, end + 0.38],
            [1.10, 1.10],
            transform=ax.get_xaxis_transform(),
            color=color,
            linewidth=2.3,
            solid_capstyle="butt",
            clip_on=False,
        )
    ax.axvline(2.5, color="#AAB2BA", linewidth=0.9)
    ax.axvline(9.5, color="#AAB2BA", linewidth=0.9)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean log1p expression", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    size_handles = [
        ax.scatter([], [], s=12 + 170 * value, facecolor="#91B8D6", edgecolor="#6F7D88", linewidth=0.35)
        for value in [0.25, 0.50, 0.75, 1.00]
    ]
    ax.legend(
        size_handles,
        ["25%", "50%", "75%", "100%"],
        title="Cells expressing",
        frameon=False,
        bbox_to_anchor=(1.08, 0.02),
        loc="lower left",
        labelspacing=1.15,
        borderaxespad=0,
        fontsize=7.8,
        title_fontsize=8.2,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_letter(ax, "E", x=-0.14, y=1.25)
    return ax


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


def make_individual_panels(cohorts: pd.DataFrame, donors: pd.DataFrame, markers: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11.2, 6.4), constrained_layout=True)
    draw_panel_a(fig, GridSpec(1, 1, figure=fig)[0], cohorts)
    save_figure(fig, "FigS01A_dataset_QC_and_donor_representation")

    fig = plt.figure(figsize=(7.7, 6.4), constrained_layout=True)
    draw_panel_b(fig, GridSpec(1, 1, figure=fig)[0], cohorts)
    save_figure(fig, "FigS01B_EC_to_LEC_selection_yield")

    fig = plt.figure(figsize=(10.8, 5.0), constrained_layout=True)
    draw_panel_c(fig, GridSpec(1, 1, figure=fig)[0], donors)
    save_figure(fig, "FigS01C_donor_LEC_counts_and_LEC25_cutoff")

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    draw_panel_d(fig, GridSpec(1, 1, figure=fig)[0], donors)
    save_figure(fig, "FigS01D_donor_contribution_to_LEC_atlases")

    fig = plt.figure(figsize=(13.8, 7.3), constrained_layout=True)
    draw_panel_e(fig, GridSpec(1, 1, figure=fig)[0], markers, cohorts)
    save_figure(fig, "FigS01E_selected_LEC_cluster_identity_QC")


def make_combined_figure(cohorts: pd.DataFrame, donors: pd.DataFrame, markers: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(17.0, 15.0), constrained_layout=True)
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 0.78, 1.20],
        width_ratios=[1.25, 0.92],
        hspace=0.18,
        wspace=0.12,
    )
    draw_panel_a(fig, grid[0, 0], cohorts)
    draw_panel_b(fig, grid[0, 1], cohorts)
    draw_panel_c(fig, grid[1, 0], donors)
    draw_panel_d(fig, grid[1, 1], donors)
    draw_panel_e(fig, grid[2, :], markers, cohorts)
    fig.suptitle(
        "Figure S1 | Dataset-level QC and LEC identification",
        fontsize=18,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    save_figure(fig, "FigS01_dataset_QC_and_LEC_identification")


def source_data_table(
    cohorts: pd.DataFrame, donors: pd.DataFrame, markers: pd.DataFrame
) -> pd.DataFrame:
    panel_a = cohorts[
        [
            "analysis",
            "cohort",
            "dataset",
            "condition",
            "n_donors_all_cell",
            "n_donors_LEC_positive",
            "n_cells_before_step03_QC",
            "n_cells_after_step03_QC",
            "percent_cells_retained",
            "step03_status",
        ]
    ].copy()
    panel_a.insert(0, "panel", "A")
    panel_a.insert(1, "record_type", "cohort_QC_summary")

    panel_b = cohorts[
        [
            "analysis",
            "cohort",
            "dataset",
            "condition",
            "n_cells_after_step03_QC",
            "n_EC_candidates",
            "n_selected_LECs",
            "selected_LEC_cluster",
        ]
    ].copy()
    panel_b.insert(0, "panel", "B")
    panel_b.insert(1, "record_type", "EC_to_LEC_yield")

    panel_c = donors.copy()
    panel_c.insert(0, "panel", "C")
    panel_c.insert(1, "record_type", "donor_LEC_count")

    panel_d = cumulative_donor_table(donors)
    panel_d.insert(0, "panel", "D")
    panel_d.insert(1, "record_type", "cumulative_donor_contribution")

    panel_e = markers.copy()
    panel_e.insert(0, "panel", "E")
    panel_e.insert(1, "record_type", "selected_cluster_marker_QC")

    return pd.concat([panel_a, panel_b, panel_c, panel_d, panel_e], ignore_index=True, sort=False)


def write_traceability(cohorts: pd.DataFrame, donors: pd.DataFrame, markers: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_data_table(cohorts, donors, markers).to_csv(
        OUT_DIR / "FigS01_source_data.csv", index=False
    )

    panel_map = {
        "aging_step03_qc": "A; B",
        "aging_donor_assignments": "A",
        "aging_ec_lec_summary": "A; B; E",
        "aging_ec_marker_summary": "E",
        "aging_donor_lec_counts": "A; C; D",
        "disease_step03_qc": "A; B",
        "disease_ec_lec_summary": "A; B; E",
        "disease_ec_marker_summary": "E",
        "disease_donor_lec_counts": "A; C; D",
    }
    purpose_map = {
        "aging_step03_qc": "Aging input and post-QC cell/nucleus counts",
        "aging_donor_assignments": "Aging donor counts per age quartile before LEC selection",
        "aging_ec_lec_summary": "Aging EC candidate and selected LEC counts and selected clusters",
        "aging_ec_marker_summary": "Aging selected-cluster expression and detection fractions",
        "aging_donor_lec_counts": "Aging donor-level final LEC counts",
        "disease_step03_qc": "Disease input and post-QC cell/nucleus and donor counts",
        "disease_ec_lec_summary": "Disease EC candidate and selected LEC counts and selected clusters",
        "disease_ec_marker_summary": "Disease selected-cluster expression and detection fractions",
        "disease_donor_lec_counts": "Disease donor-level final LEC counts",
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
    manifest.to_csv(OUT_DIR / "FigS01_input_manifest.csv", index=False)


def main() -> None:
    configure_style()
    data = read_inputs()
    cohorts, donors, markers = build_tables(data)
    write_traceability(cohorts, donors, markers)
    make_individual_panels(cohorts, donors, markers)
    make_combined_figure(cohorts, donors, markers)
    print(f"Supplementary Figure S1 written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
