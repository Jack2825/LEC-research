#!/usr/bin/env python3
"""Create Figure 1B from the completed aging and disease result CSV files.

The figure follows the supplied two-column template:

    Aging                         Diseases
    donor age distribution       donor age distribution
    final LEC counts              final LEC counts
    LEC marker dot plot across all cardiac EC candidates

Aging donor ages and LEC bars use only donors retained after the >=25-total-LEC
filter and show Q1-Q4 plus their combined NF total. Disease donor ages and LEC
bars use a tiered analysis cohort: dataset01/08 DCM, HCM, DoxCM, and matched NF
remain primary at >=25 LECs/donor; dataset11 HFpEF and its matched NF use >=10
as an exploratory cohort, with the nested >=15 cohort shown as sensitivity.
Donor ages are never inferred. Groups lacking numeric ages in the current
result table are marked NR (not reported). The marker dot plot remains a
selection-QC view across all Step 06 cardiac EC candidates and is not
donor-filtered.

Outputs
-------
new_result2/Fig01B_donor_age_and_LEC_counts/
    Fig01B_donor_age_and_LEC_counts.pdf
    Fig01B_donor_age_and_LEC_counts.png
    Fig01B_donor_age_and_LEC_counts.jpg
    Fig01B_source_data.csv
    Fig01B_input_csv_manifest.csv
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


SCRIPT_VERSION = "Fig01B_tiered_disease_cohort_v4_2026_08_31"
RANDOM_SEED = 23

AGING_FILTER_CSV = Path(
    "new_result2_aging/08_7_min25_LEC_filtered_age_analysis/dataset13/"
    "08_7_all_donor_filter_mapping.csv"
)
DISEASE_FILTER_CSV = Path(
    "new_result2_disease/08_3_min25_LEC_filtered_analysis/tables/"
    "08_3_all_donor_filter_mapping.csv"
)
DISEASE_COLOR_CSV = Path(
    "new_result2_disease/07_LEC_description/07_color_key.csv"
)
AGING_EC_MARKER_CSV = Path(
    "new_result2_aging/06_EC_recluster/dataset13/"
    "06_EC_recluster_marker_summary_all_quartiles.csv"
)
AGING_LEC_CLUSTER_CSV = Path(
    "new_result2_aging/06_EC_recluster/dataset13/"
    "06_final_LEC_candidate_clusters_all_quartiles.csv"
)
DISEASE_EC_MARKER_CSV = Path(
    "new_result2_disease/06_EC_recluster/"
    "06_EC_recluster_marker_summary_all_cohorts.csv"
)
DISEASE_LEC_CLUSTER_CSV = Path(
    "new_result2_disease/06_EC_recluster/"
    "06_final_LEC_candidate_clusters_all_cohorts.csv"
)
OUTPUT_RELATIVE = Path("new_result2/Fig01B_donor_age_and_LEC_counts")

AGING_ORDER = ["Q1", "Q2", "Q3", "Q4"]
AGING_COLORS = {
    "Q1": "#4C78A8",
    "Q2": "#59A14F",
    "Q3": "#F28E2B",
    "Q4": "#E15759",
    "Combined NF": "#4D4D4D",
}
DISEASE_ORDER = ["DCM", "HCM", "DoxCM", "D01/D08 NF", "HFpEF", "D11 NF"]
PRIMARY_DISEASE_GROUPS = ["DCM", "HCM", "DoxCM", "D01/D08 NF"]
DATASET11_GROUPS = ["HFpEF", "D11 NF"]
PRIMARY_MINIMUM_LECS = 25
HFPEF_EXPLORATORY_MINIMUM_LECS = 10
HFPEF_SENSITIVITY_MINIMUM_LECS = 15
LEC_MARKERS = ["PROX1", "FLT4", "PDPN", "LYVE1", "CCL21", "MMRN1", "NRP2", "MRC1"]


def project_root() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_string(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def load_inputs(root: Path) -> dict[str, object]:
    paths = {
        "aging_filter": root / AGING_FILTER_CSV,
        "disease_filter": root / DISEASE_FILTER_CSV,
        "disease_color": root / DISEASE_COLOR_CSV,
        "aging_ec_marker": root / AGING_EC_MARKER_CSV,
        "aging_lec_cluster": root / AGING_LEC_CLUSTER_CSV,
        "disease_ec_marker": root / DISEASE_EC_MARKER_CSV,
        "disease_lec_cluster": root / DISEASE_LEC_CLUSTER_CSV,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} CSV: {path}")

    aging_filter = pd.read_csv(paths["aging_filter"], low_memory=False)
    disease_filter = pd.read_csv(paths["disease_filter"], low_memory=False)
    disease_color = pd.read_csv(paths["disease_color"], low_memory=False)
    aging_ec_marker = pd.read_csv(paths["aging_ec_marker"], low_memory=False)
    aging_lec_cluster = pd.read_csv(paths["aging_lec_cluster"], low_memory=False)
    disease_ec_marker = pd.read_csv(paths["disease_ec_marker"], low_memory=False)
    disease_lec_cluster = pd.read_csv(paths["disease_lec_cluster"], low_memory=False)

    require_columns(
        aging_filter,
        [
            "donor_key",
            "age_group",
            "age_numeric",
            "n_total_LEC_cells",
            "retained_min25_LECs",
        ],
        paths["aging_filter"],
    )
    require_columns(
        disease_filter,
        [
            "donor_key",
            "dataset_folder",
            "condition",
            "age",
            "n_final_LECs",
            "retained_minimum_LECs",
        ],
        paths["disease_filter"],
    )
    require_columns(
        disease_color,
        ["category_type", "name", "color_hex"],
        paths["disease_color"],
    )
    marker_columns = [
        column_type + gene
        for gene in LEC_MARKERS
        for column_type in ("mean_expr_", "pct_expr_")
    ]
    require_columns(
        aging_ec_marker,
        ["age_quartile", "ec_leiden", "n_cells", *marker_columns],
        paths["aging_ec_marker"],
    )
    require_columns(
        aging_lec_cluster,
        ["age_quartile", "ec_leiden", "selected_as_LEC"],
        paths["aging_lec_cluster"],
    )
    require_columns(
        disease_ec_marker,
        [
            "dataset_folder",
            "condition",
            "ec_leiden",
            "n_cells_marker_summary",
            *marker_columns,
        ],
        paths["disease_ec_marker"],
    )
    require_columns(
        disease_lec_cluster,
        ["dataset_folder", "condition", "ec_leiden", "selected_as_LEC"],
        paths["disease_lec_cluster"],
    )

    return {
        "paths": paths,
        "aging_filter": aging_filter,
        "disease_filter": disease_filter,
        "disease_color": disease_color,
        "aging_ec_marker": aging_ec_marker,
        "aging_lec_cluster": aging_lec_cluster,
        "disease_ec_marker": disease_ec_marker,
        "disease_lec_cluster": disease_lec_cluster,
    }


def prepare_aging_age(frame: pd.DataFrame) -> pd.DataFrame:
    retained = frame.loc[true_values(frame["retained_min25_LECs"])].copy()
    result = retained[["donor_key", "age_numeric", "age_group"]].rename(
        columns={
            "donor_key": "donor_id_quartile_unique",
            "age_numeric": "age_numeric_resolved",
            "age_group": "age_quartile",
        }
    )
    result["age_numeric_resolved"] = pd.to_numeric(
        result["age_numeric_resolved"], errors="coerce"
    )
    result = result.drop_duplicates("donor_id_quartile_unique")
    if result["age_numeric_resolved"].isna().any():
        raise ValueError("The aging donor assignment table contains missing ages.")
    unexpected = sorted(set(result["age_quartile"]) - set(AGING_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected aging quartiles: {unexpected}")
    observed_counts = result.groupby("age_quartile", observed=True).size()
    expected_counts = {"Q1": 4, "Q2": 1, "Q3": 3, "Q4": 5}
    if observed_counts.to_dict() != expected_counts:
        raise ValueError(
            "Retained aging donor counts do not match the validated Step 08_7 cohort: "
            f"{observed_counts.to_dict()}"
        )
    return result


def prepare_disease_donors(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["n_final_LECs"] = pd.to_numeric(
        working["n_final_LECs"], errors="raise"
    )
    primary = working.loc[
        working["dataset_folder"].isin(["dataset01", "dataset08"])
        & (working["n_final_LECs"] >= PRIMARY_MINIMUM_LECS)
    ].copy()
    primary["display_group"] = primary["condition"].where(
        primary["condition"] != "NF", "D01/D08 NF"
    )
    primary["minimum_LECs_per_donor"] = PRIMARY_MINIMUM_LECS
    primary["analysis_tier"] = "primary_min25"
    primary["included_in_min15_sensitivity"] = False

    dataset11 = working.loc[
        working["dataset_folder"].eq("dataset11")
        & working["condition"].isin(["HFpEF", "NF"])
        & (working["n_final_LECs"] >= HFPEF_EXPLORATORY_MINIMUM_LECS)
    ].copy()
    dataset11["display_group"] = dataset11["condition"].where(
        dataset11["condition"] != "NF", "D11 NF"
    )
    dataset11["minimum_LECs_per_donor"] = HFPEF_EXPLORATORY_MINIMUM_LECS
    dataset11["analysis_tier"] = "exploratory_min10"
    dataset11["included_in_min15_sensitivity"] = (
        dataset11["n_final_LECs"] >= HFPEF_SENSITIVITY_MINIMUM_LECS
    )

    retained = pd.concat([primary, dataset11], ignore_index=True)
    result = retained[
        [
            "dataset_folder",
            "condition",
            "donor_key",
            "age",
            "n_final_LECs",
            "display_group",
            "minimum_LECs_per_donor",
            "analysis_tier",
            "included_in_min15_sensitivity",
        ]
    ].copy()
    if result["donor_key"].duplicated().any():
        duplicated = result.loc[result["donor_key"].duplicated(), "donor_key"]
        raise ValueError(f"Disease donor keys are not unique: {duplicated.tolist()}")
    result["age_numeric"] = pd.to_numeric(result["age"], errors="coerce")
    unexpected = sorted(set(result["display_group"]) - set(DISEASE_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected disease display groups: {unexpected}")
    observed = result.groupby("display_group", observed=True).agg(
        donors=("donor_key", "nunique"), LECs=("n_final_LECs", "sum")
    )
    expected = {
        "DCM": (10, 1372),
        "HCM": (15, 2669),
        "DoxCM": (9, 465),
        "D01/D08 NF": (16, 1202),
        "HFpEF": (7, 259),
        "D11 NF": (6, 90),
    }
    observed_values = {
        group: (int(row.donors), int(row.LECs))
        for group, row in observed.iterrows()
    }
    if observed_values != expected:
        raise ValueError(
            "Tiered disease cohort does not match the validated current data: "
            f"{observed_values}"
        )
    sensitivity = (
        result.loc[result["included_in_min15_sensitivity"]]
        .groupby("display_group", observed=True)
        .agg(donors=("donor_key", "nunique"), LECs=("n_final_LECs", "sum"))
    )
    expected_sensitivity = {"HFpEF": (5, 232), "D11 NF": (4, 69)}
    sensitivity_values = {
        group: (int(row.donors), int(row.LECs))
        for group, row in sensitivity.iterrows()
    }
    if sensitivity_values != expected_sensitivity:
        raise ValueError(
            "Dataset11 >=15 sensitivity cohort does not match the validated "
            f"current data: {sensitivity_values}"
        )
    return result


def prepare_aging_lec(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[true_values(frame["retained_min25_LECs"])].copy()
    result["n_total_LEC_cells"] = pd.to_numeric(
        result["n_total_LEC_cells"], errors="raise"
    )
    if (result["n_total_LEC_cells"] < 25).any():
        raise ValueError("A retained aging donor has fewer than 25 final LECs.")
    result = result.groupby("age_group", as_index=False, observed=True).agg(
        n_donors_all=("donor_key", "nunique"),
        n_LEC_cells=("n_total_LEC_cells", "sum"),
    )
    result = result.rename(columns={"age_group": "age_quartile"})
    result = result.set_index("age_quartile").reindex(AGING_ORDER).reset_index()
    if result.isna().any().any():
        raise ValueError("Aging LEC summary does not contain all Q1-Q4 rows.")
    combined = pd.DataFrame(
        {
            "age_quartile": ["Combined NF"],
            "n_donors_all": [int(result["n_donors_all"].sum())],
            "n_LEC_cells": [int(result["n_LEC_cells"].sum())],
        }
    )
    return pd.concat([result, combined], ignore_index=True)


def prepare_disease_lec(frame: pd.DataFrame) -> pd.DataFrame:
    donors = prepare_disease_donors(frame)
    result = donors.groupby("display_group", as_index=False, observed=True).agg(
        n_donors_all=("donor_key", "nunique"),
        n_LEC_cells=("n_final_LECs", "sum"),
    )
    result = result.set_index("display_group").reindex(DISEASE_ORDER).reset_index()
    if result.isna().any().any():
        raise ValueError("Disease LEC summary does not contain every display group.")
    result["minimum_LECs_per_donor"] = result["display_group"].map(
        {**{group: 25 for group in PRIMARY_DISEASE_GROUPS}, **{group: 10 for group in DATASET11_GROUPS}}
    )
    result["analysis_tier"] = result["display_group"].map(
        {**{group: "primary_min25" for group in PRIMARY_DISEASE_GROUPS}, **{group: "exploratory_min10" for group in DATASET11_GROUPS}}
    )
    sensitivity = (
        donors.loc[donors["included_in_min15_sensitivity"]]
        .groupby("display_group", observed=True)
        .agg(
            n_donors_min15=("donor_key", "nunique"),
            n_LEC_cells_min15=("n_final_LECs", "sum"),
        )
    )
    result = result.merge(
        sensitivity,
        left_on="display_group",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    return result


def condition_colors(frame: pd.DataFrame) -> dict[str, str]:
    subset = frame.loc[frame["category_type"].eq("condition")]
    colors = dict(zip(subset["name"], subset["color_hex"], strict=False))
    required = ["DCM", "HCM", "DoxCM", "HFpEF", "NF"]
    missing = [group for group in required if group not in colors]
    if missing:
        raise ValueError(f"Missing disease colors for: {missing}")
    colors["D01/D08 NF"] = colors["NF"]
    colors["D11 NF"] = colors["NF"]
    return colors


def normalized_cluster(values: pd.Series) -> pd.Series:
    """Normalize integer-like Leiden labels without changing nonnumeric labels."""
    numeric = pd.to_numeric(values, errors="coerce")
    output = values.astype(str).str.strip()
    integer_mask = numeric.notna() & np.isclose(numeric, np.round(numeric))
    output.loc[integer_mask] = numeric.loc[integer_mask].astype(int).astype(str)
    return output


def true_values(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def prepare_marker_dotplot_data(
    aging_markers: pd.DataFrame,
    aging_lec_clusters: pd.DataFrame,
    disease_markers: pd.DataFrame,
    disease_lec_clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Pool executed cluster summaries using cell-count weights."""
    aging = aging_markers.copy()
    aging["ec_leiden"] = normalized_cluster(aging["ec_leiden"])
    aging_candidates = aging_lec_clusters.loc[
        true_values(aging_lec_clusters["selected_as_LEC"])
    ].copy()
    aging_candidates["ec_leiden"] = normalized_cluster(aging_candidates["ec_leiden"])
    aging_keys = set(
        zip(
            aging_candidates["age_quartile"].astype(str),
            aging_candidates["ec_leiden"],
            strict=False,
        )
    )
    if len(aging_keys) != 4:
        raise ValueError(f"Expected four aging LEC clusters; found {len(aging_keys)}")
    aging["is_LEC"] = [
        (str(group), str(cluster)) in aging_keys
        for group, cluster in zip(
            aging["age_quartile"], aging["ec_leiden"], strict=False
        )
    ]
    aging["n_cells_plot"] = pd.to_numeric(aging["n_cells"], errors="raise")

    disease = disease_markers.copy()
    disease["ec_leiden"] = normalized_cluster(disease["ec_leiden"])
    disease_candidates = disease_lec_clusters.loc[
        true_values(disease_lec_clusters["selected_as_LEC"])
    ].copy()
    disease_candidates["ec_leiden"] = normalized_cluster(
        disease_candidates["ec_leiden"]
    )
    disease_keys = set(
        zip(
            disease_candidates["dataset_folder"].astype(str),
            disease_candidates["condition"].astype(str),
            disease_candidates["ec_leiden"],
            strict=False,
        )
    )
    if len(disease_keys) != 7:
        raise ValueError(f"Expected seven disease LEC clusters; found {len(disease_keys)}")
    disease["is_LEC"] = [
        (str(dataset), str(condition), str(cluster)) in disease_keys
        for dataset, condition, cluster in zip(
            disease["dataset_folder"],
            disease["condition"],
            disease["ec_leiden"],
            strict=False,
        )
    ]
    disease["n_cells_plot"] = pd.to_numeric(
        disease["n_cells_marker_summary"], errors="raise"
    )

    records: list[dict[str, object]] = []
    for analysis, table in [("Aging", aging), ("Diseases", disease)]:
        for is_lec, cell_class in [(True, "LEC"), (False, "Other ECs")]:
            subset = table.loc[table["is_LEC"].eq(is_lec)].copy()
            weights = subset["n_cells_plot"].to_numpy(dtype=float)
            n_cells = int(weights.sum())
            if n_cells <= 0:
                raise ValueError(f"No cells available for {analysis} {cell_class}")
            for gene in LEC_MARKERS:
                mean_values = pd.to_numeric(
                    subset[f"mean_expr_{gene}"], errors="raise"
                ).to_numpy(dtype=float)
                detection_values = pd.to_numeric(
                    subset[f"pct_expr_{gene}"], errors="raise"
                ).to_numpy(dtype=float)
                if np.nanmax(detection_values) > 1.000001:
                    raise ValueError(
                        f"Expected fractional detection values for {analysis} {gene}"
                    )
                records.append(
                    {
                        "analysis": analysis,
                        "cell_class": cell_class,
                        "display_group": f"{analysis} — {cell_class}",
                        "marker_gene": gene,
                        "mean_log1p_expression": float(
                            np.average(mean_values, weights=weights)
                        ),
                        "percent_expressing": float(
                            100.0 * np.average(detection_values, weights=weights)
                        ),
                        "n_cells": n_cells,
                    }
                )
    return pd.DataFrame.from_records(records)


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(width=0.8, length=3, labelsize=7.5)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)


def draw_age_distribution(
    axis: plt.Axes,
    frame: pd.DataFrame,
    group_column: str,
    age_column: str,
    groups: list[str],
    colors: dict[str, str],
    total_counts: dict[str, int],
    rng: np.random.Generator,
) -> None:
    positions = np.arange(len(groups), dtype=float)
    labels: list[str] = []

    for position, group in zip(positions, groups, strict=True):
        values = (
            frame.loc[frame[group_column].eq(group), age_column]
            .dropna()
            .astype(float)
            .to_numpy()
        )
        total = int(total_counts[group])
        available = int(values.size)
        if available:
            plot = axis.boxplot(
                [values],
                positions=[position],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111111", "linewidth": 1.2},
                whiskerprops={"color": "#333333", "linewidth": 0.8},
                capprops={"color": "#333333", "linewidth": 0.8},
                boxprops={"edgecolor": "#333333", "linewidth": 0.8},
            )
            plot["boxes"][0].set_facecolor(colors[group])
            plot["boxes"][0].set_alpha(0.35)
            jitter = rng.uniform(-0.16, 0.16, size=available)
            axis.scatter(
                np.full(available, position) + jitter,
                values,
                s=19,
                c=colors[group],
                edgecolors="white",
                linewidths=0.45,
                alpha=0.9,
                zorder=3,
            )
        else:
            axis.text(
                position,
                49,
                "NR",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#6B6B6B",
            )

        if available == total:
            labels.append(f"{group}\nn={total}")
        elif available == 0:
            labels.append(f"{group}\n0/{total}")
        else:
            labels.append(f"{group}\n{available}/{total}")

    axis.set_xlim(-0.55, len(groups) - 0.45)
    axis.set_ylim(20, 80)
    axis.yaxis.set_major_locator(MultipleLocator(10))
    axis.set_xticks(positions, labels)
    style_axis(axis)


def draw_lec_bars(
    axis: plt.Axes,
    frame: pd.DataFrame,
    group_column: str,
    groups: list[str],
    colors: dict[str, str],
    shared_upper_limit: float,
) -> None:
    indexed = frame.set_index(group_column)
    values = [int(indexed.loc[group, "n_LEC_cells"]) for group in groups]
    donors = [int(indexed.loc[group, "n_donors_all"]) for group in groups]
    positions = np.arange(len(groups), dtype=float)
    bars = axis.bar(
        positions,
        values,
        width=0.7,
        color=[colors[group] for group in groups],
        edgecolor="#222222",
        linewidth=0.8,
        zorder=2,
    )
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + shared_upper_limit * 0.018,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
        )
    labels = [f"{group}\nn={donor}" for group, donor in zip(groups, donors, strict=True)]
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, shared_upper_limit)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    style_axis(axis)


def draw_disease_lec_bars(
    axis: plt.Axes,
    frame: pd.DataFrame,
    colors: dict[str, str],
    shared_upper_limit: float,
) -> None:
    indexed = frame.set_index("display_group")
    positions = np.arange(len(DISEASE_ORDER), dtype=float)
    values = [int(indexed.loc[group, "n_LEC_cells"]) for group in DISEASE_ORDER]
    donors = [int(indexed.loc[group, "n_donors_all"]) for group in DISEASE_ORDER]
    bars = axis.bar(
        positions,
        values,
        width=0.72,
        color=[colors[group] for group in DISEASE_ORDER],
        edgecolor="#222222",
        linewidth=0.8,
        zorder=2,
    )
    for index, group in enumerate(DISEASE_ORDER):
        if group in DATASET11_GROUPS:
            bars[index].set_alpha(0.55)
            bars[index].set_hatch("///")
            sensitivity_value = int(indexed.loc[group, "n_LEC_cells_min15"])
            sensitivity_donors = int(indexed.loc[group, "n_donors_min15"])
            axis.bar(
                positions[index],
                sensitivity_value,
                width=0.38,
                color=colors[group],
                edgecolor="#222222",
                linewidth=0.9,
                zorder=3,
            )
            annotation_x = positions[index]
            annotation_y = shared_upper_limit * (
                0.29 if group == "HFpEF" else 0.10
            )
            axis.text(
                annotation_x,
                annotation_y,
                (
                    f"\u226510: {values[index]} (n={donors[index]})\n"
                    f"\u226515: {sensitivity_value} (n={sensitivity_donors})"
                ),
                ha="center",
                va="bottom",
                fontsize=6.2,
                fontweight="bold",
            )
        else:
            axis.text(
                positions[index],
                values[index] + shared_upper_limit * 0.018,
                f"{values[index]:,}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
            )

    labels = []
    for group, donor in zip(DISEASE_ORDER, donors, strict=True):
        if group in DATASET11_GROUPS:
            labels.append(f"{group}\n\u226510: n={donor}")
        else:
            labels.append(f"{group}\nn={donor}")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, shared_upper_limit)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    style_axis(axis)
    axis.legend(
        handles=[
            Patch(facecolor="#6F8794", edgecolor="#222222", label="\u226525 primary"),
            Patch(
                facecolor="#B8C7CC",
                edgecolor="#222222",
                hatch="///",
                label="D11 \u226510 exploratory",
            ),
            Patch(facecolor="#6F8794", edgecolor="#222222", label="D11 \u226515 subset"),
        ],
        frameon=False,
        fontsize=6.4,
        loc="upper right",
        handlelength=1.5,
    )


def draw_marker_dotplot(axis: plt.Axes, marker_data: pd.DataFrame) -> None:
    row_order = [
        "Aging — LEC",
        "Aging — Other ECs",
        "Diseases — LEC",
        "Diseases — Other ECs",
    ]
    row_positions = {label: index for index, label in enumerate(row_order)}
    gene_positions = {gene: index for index, gene in enumerate(LEC_MARKERS)}

    for index, label in enumerate(row_order):
        if label.endswith("— LEC"):
            axis.axhspan(index - 0.48, index + 0.48, color="#FFF0F0", zorder=0)

    x = marker_data["marker_gene"].map(gene_positions).to_numpy(dtype=float)
    y = marker_data["display_group"].map(row_positions).to_numpy(dtype=float)
    dot_size_scale = 3.4
    size = marker_data["percent_expressing"].to_numpy(dtype=float) * dot_size_scale
    expression = marker_data["mean_log1p_expression"].to_numpy(dtype=float)
    color_upper = max(0.5, float(np.ceil(expression.max() * 10) / 10))
    dots = axis.scatter(
        x,
        y,
        s=size,
        c=expression,
        cmap="viridis",
        vmin=0,
        vmax=color_upper,
        edgecolors="#252525",
        linewidths=0.35,
        zorder=3,
    )

    cell_counts = (
        marker_data[["display_group", "n_cells"]]
        .drop_duplicates()
        .set_index("display_group")["n_cells"]
        .to_dict()
    )
    row_labels = [f"{label}\nn={int(cell_counts[label]):,} cells" for label in row_order]
    axis.set_xticks(range(len(LEC_MARKERS)), LEC_MARKERS, fontstyle="italic")
    axis.set_yticks(range(len(row_order)), row_labels)
    axis.set_xlim(-0.55, len(LEC_MARKERS) - 0.45)
    axis.set_ylim(len(row_order) - 0.5, -0.5)
    axis.tick_params(axis="x", labelsize=8.2, pad=4)
    axis.tick_params(axis="y", labelsize=7.5, length=0)
    axis.grid(color="#E1E1E1", linewidth=0.55, zorder=1)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)

    colorbar = axis.figure.colorbar(dots, ax=axis, fraction=0.024, pad=0.025)
    colorbar.set_label("Mean log1p expression", fontsize=7.5)
    colorbar.ax.tick_params(labelsize=6.8, width=0.6, length=2.5)
    colorbar.outline.set_linewidth(0.6)

    legend_handles = [
        axis.scatter(
            [],
            [],
            s=percent * dot_size_scale,
            facecolor="#8F8F8F",
            edgecolor="#252525",
            linewidth=0.35,
            label=f"{percent}%",
        )
        for percent in (25, 50, 75, 100)
    ]
    axis.legend(
        handles=legend_handles,
        title="Cells expressing (%)",
        frameon=False,
        fontsize=6.8,
        title_fontsize=7.2,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        handletextpad=0.5,
        columnspacing=1.1,
    )


def build_source_data(
    root: Path,
    aging_age: pd.DataFrame,
    disease_donor: pd.DataFrame,
    aging_lec: pd.DataFrame,
    disease_lec: pd.DataFrame,
    marker_data: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    aging_totals = aging_age.groupby("age_quartile", observed=True).size().to_dict()
    for row in aging_age.itertuples(index=False):
        records.append(
            {
                "panel": "aging_donor_age",
                "record_type": "donor_observation",
                "group": row.age_quartile,
                "donor_id": row.donor_id_quartile_unique,
                "value": row.age_numeric_resolved,
                "unit": "years",
                "n_donors_with_value": aging_totals[row.age_quartile],
                "n_donors_total": aging_totals[row.age_quartile],
                "source_csv": AGING_FILTER_CSV.as_posix(),
                "source_columns": (
                    "donor_key;age_numeric;age_group;retained_min25_LECs"
                ),
                "derivation": (
                    "one row per retained donor with >=25 total final LECs; "
                    "no age inference"
                ),
            }
        )

    disease_totals = disease_donor.groupby("display_group", observed=True).size()
    disease_available = disease_donor.groupby("display_group", observed=True)[
        "age_numeric"
    ].count()
    for group in DISEASE_ORDER:
        group_rows = disease_donor.loc[disease_donor["display_group"].eq(group)]
        threshold = int(group_rows["minimum_LECs_per_donor"].iloc[0])
        tier = str(group_rows["analysis_tier"].iloc[0])
        sensitivity_n = int(group_rows["included_in_min15_sensitivity"].sum())
        records.append(
            {
                "panel": "disease_donor_age",
                "record_type": "availability_summary",
                "group": group,
                "donor_id": "",
                "value": np.nan,
                "unit": "years",
                "n_donors_with_value": int(disease_available.get(group, 0)),
                "n_donors_total": int(disease_totals.get(group, 0)),
                "source_csv": DISEASE_FILTER_CSV.as_posix(),
                "source_columns": (
                    "dataset_folder;condition;donor_key;age;n_final_LECs"
                ),
                "derivation": (
                    "dataset01/08 groups use >=25; dataset11 HFpEF and NF use >=10; "
                    "nonnumeric age retained as missing and never inferred"
                ),
                "minimum_LECs_per_donor": threshold,
                "analysis_tier": tier,
                "n_donors_in_min15_sensitivity": (
                    sensitivity_n if group in DATASET11_GROUPS else np.nan
                ),
            }
        )
    for row in disease_donor.loc[disease_donor["age_numeric"].notna()].itertuples(
        index=False
    ):
        group = row.display_group
        records.append(
            {
                "panel": "disease_donor_age",
                "record_type": "donor_observation",
                "group": group,
                "donor_id": row.donor_key,
                "value": row.age_numeric,
                "unit": "years",
                "n_donors_with_value": int(disease_available[group]),
                "n_donors_total": int(disease_totals[group]),
                "source_csv": DISEASE_FILTER_CSV.as_posix(),
                "source_columns": (
                    "dataset_folder;condition;donor_key;age;n_final_LECs"
                ),
                "derivation": (
                    "tier-retained donor; numeric age plotted as reported"
                ),
                "minimum_LECs_per_donor": int(row.minimum_LECs_per_donor),
                "analysis_tier": row.analysis_tier,
                "included_in_min15_sensitivity": bool(
                    row.included_in_min15_sensitivity
                ),
            }
        )

    for row in aging_lec.itertuples(index=False):
        combined = row.age_quartile == "Combined NF"
        records.append(
            {
                "panel": "aging_LEC_count",
                "record_type": "bar_summary",
                "group": row.age_quartile,
                "donor_id": "",
                "value": int(row.n_LEC_cells),
                "unit": "final LEC cells",
                "n_donors_with_value": int(row.n_donors_all),
                "n_donors_total": int(row.n_donors_all),
                "source_csv": AGING_FILTER_CSV.as_posix(),
                "source_columns": (
                    "donor_key;age_group;n_total_LEC_cells;retained_min25_LECs"
                ),
                "derivation": (
                    "sum across retained Q1-Q4 donors"
                    if combined
                    else "sum of final LECs across retained donors in the quartile"
                ),
            }
        )

    for row in disease_lec.itertuples(index=False):
        records.append(
            {
                "panel": "disease_LEC_count",
                "record_type": "bar_summary",
                "group": row.display_group,
                "donor_id": "",
                "value": int(row.n_LEC_cells),
                "unit": "final LEC cells",
                "n_donors_with_value": int(row.n_donors_all),
                "n_donors_total": int(row.n_donors_all),
                "source_csv": DISEASE_FILTER_CSV.as_posix(),
                "source_columns": (
                    "donor_key;dataset_folder;condition;n_final_LECs"
                ),
                "derivation": (
                    "sum of final LECs across donors retained at the displayed "
                    "analysis-tier threshold"
                ),
                "minimum_LECs_per_donor": int(row.minimum_LECs_per_donor),
                "analysis_tier": row.analysis_tier,
            }
        )
        if row.display_group in DATASET11_GROUPS:
            records.append(
                {
                    "panel": "disease_LEC_count",
                    "record_type": "sensitivity_bar_summary",
                    "group": row.display_group,
                    "donor_id": "",
                    "value": int(row.n_LEC_cells_min15),
                    "unit": "final LEC cells",
                    "n_donors_with_value": int(row.n_donors_min15),
                    "n_donors_total": int(row.n_donors_min15),
                    "source_csv": DISEASE_FILTER_CSV.as_posix(),
                    "source_columns": (
                        "donor_key;dataset_folder;condition;n_final_LECs"
                    ),
                    "derivation": (
                        "nested dataset11 sensitivity subset with >=15 final "
                        "LECs per donor"
                    ),
                    "minimum_LECs_per_donor": 15,
                    "analysis_tier": "sensitivity_min15",
                }
            )

    for row in marker_data.itertuples(index=False):
        if row.analysis == "Aging":
            source_csvs = (
                f"{AGING_EC_MARKER_CSV.as_posix()};"
                f"{AGING_LEC_CLUSTER_CSV.as_posix()}"
            )
        else:
            source_csvs = (
                f"{DISEASE_EC_MARKER_CSV.as_posix()};"
                f"{DISEASE_LEC_CLUSTER_CSV.as_posix()}"
            )
        records.append(
            {
                "panel": "LEC_marker_dotplot_all_cardiac_ECs",
                "record_type": "marker_dot_summary",
                "group": row.display_group,
                "donor_id": "",
                "value": np.nan,
                "unit": "log1p expression and percent of cells",
                "n_donors_with_value": np.nan,
                "n_donors_total": np.nan,
                "source_csv": source_csvs,
                "source_columns": (
                    f"n_cells;mean_expr_{row.marker_gene};"
                    f"pct_expr_{row.marker_gene};selected_as_LEC"
                ),
                "derivation": (
                    "Step06 pct_expr is mean(X>0) within each EC cluster; displayed "
                    "percent=100*sum(n_cluster*pct_expr)/sum(n_cluster). Mean "
                    "expression is pooled with the same n_cluster weights; LEC "
                    "status comes from the executed Step06 candidate-cluster table"
                ),
                "marker_gene": row.marker_gene,
                "mean_log1p_expression": row.mean_log1p_expression,
                "percent_expressing": row.percent_expressing,
                "n_cells": row.n_cells,
            }
        )

    output = pd.DataFrame.from_records(records)
    output.insert(0, "script_version", SCRIPT_VERSION)
    return output


def build_input_manifest(root: Path, paths: dict[str, Path]) -> pd.DataFrame:
    uses = {
        "aging_filter": (
            "aging retained-donor ages and LEC bars",
            "donor_key;age_group;age_numeric;n_total_LEC_cells;retained_min25_LECs",
            "retain only donors with >=25 total final LECs; aggregate by Q1-Q4",
        ),
        "disease_filter": (
            "tiered disease donor ages and LEC bars",
            "donor_key;dataset_folder;condition;age;n_final_LECs",
            "dataset01/08 use >=25; dataset11 HFpEF and matched NF use >=10 with >=15 sensitivity",
        ),
        "disease_color": (
            "disease condition colors",
            "category_type;name;color_hex",
            "current Step07 condition color mapping",
        ),
        "aging_ec_marker": (
            "aging LEC marker dot plot",
            "age_quartile;ec_leiden;n_cells;mean_expr_*;pct_expr_*",
            "Step06 mean(X>0) detection fractions pooled with n_cells weights",
        ),
        "aging_lec_cluster": (
            "aging LEC marker dot plot row classification",
            "age_quartile;ec_leiden;selected_as_LEC",
            "executed Step06 LEC cluster calls; pre-minimum-25 filter",
        ),
        "disease_ec_marker": (
            "disease LEC marker dot plot",
            "dataset_folder;condition;ec_leiden;n_cells_marker_summary;mean_expr_*;pct_expr_*",
            "active Step06 mean(X>0) fractions pooled with cell-count weights",
        ),
        "disease_lec_cluster": (
            "disease LEC marker dot plot row classification",
            "dataset_folder;condition;ec_leiden;selected_as_LEC",
            "executed Step06 LEC cluster calls; pre-minimum-25 filter",
        ),
    }
    records = []
    for label, path in paths.items():
        stat = path.stat()
        purpose, columns, aggregation = uses[label]
        records.append(
            {
                "script_version": SCRIPT_VERSION,
                "input_id": label,
                "source_csv": relative_string(path, root),
                "used_for": purpose,
                "columns_used": columns,
                "aggregation_or_filter": aggregation,
                "size_bytes": stat.st_size,
                "last_modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame.from_records(records)


def create_figure(
    aging_age: pd.DataFrame,
    disease_donor: pd.DataFrame,
    aging_lec: pd.DataFrame,
    disease_lec: pd.DataFrame,
    disease_colors: dict[str, str],
    marker_data: pd.DataFrame,
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    figure = plt.figure(figsize=(9.8, 9.2), facecolor="white")
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 0.95, 0.78],
        left=0.125,
        right=0.965,
        bottom=0.20,
        top=0.90,
        wspace=0.17,
        hspace=0.62,
    )
    age_aging_axis = figure.add_subplot(grid[0, 0])
    age_disease_axis = figure.add_subplot(grid[0, 1], sharey=age_aging_axis)
    lec_aging_axis = figure.add_subplot(grid[1, 0])
    lec_disease_axis = figure.add_subplot(grid[1, 1], sharey=lec_aging_axis)
    marker_axis = figure.add_subplot(grid[2, :])

    rng = np.random.default_rng(RANDOM_SEED)
    aging_age_totals = aging_age.groupby("age_quartile", observed=True).size().to_dict()
    draw_age_distribution(
        age_aging_axis,
        aging_age,
        "age_quartile",
        "age_numeric_resolved",
        AGING_ORDER,
        AGING_COLORS,
        aging_age_totals,
        rng,
    )

    disease_age_totals = (
        disease_donor.groupby("display_group", observed=True).size().to_dict()
    )
    draw_age_distribution(
        age_disease_axis,
        disease_donor,
        "display_group",
        "age_numeric",
        DISEASE_ORDER,
        disease_colors,
        disease_age_totals,
        rng,
    )
    disease_age_available = disease_donor.groupby(
        "display_group", observed=True
    )["age_numeric"].count()
    disease_age_labels = []
    for group in DISEASE_ORDER:
        total = int(disease_age_totals[group])
        available = int(disease_age_available.get(group, 0))
        label = (
            f"{group}\n{available}/{total}"
            if available != total
            else f"{group}\nn={total}"
        )
        if group in DATASET11_GROUPS:
            sensitivity_n = int(
                disease_donor.loc[
                    disease_donor["display_group"].eq(group),
                    "included_in_min15_sensitivity",
                ].sum()
            )
            label += f"\n\u226515: n={sensitivity_n}"
        disease_age_labels.append(label)
    age_disease_axis.set_xticks(
        np.arange(len(DISEASE_ORDER), dtype=float), disease_age_labels
    )

    shared_upper_limit = max(
        float(aging_lec["n_LEC_cells"].max()),
        float(disease_lec["n_LEC_cells"].max()),
    ) * 1.18
    draw_lec_bars(
        lec_aging_axis,
        aging_lec,
        "age_quartile",
        AGING_ORDER + ["Combined NF"],
        AGING_COLORS,
        shared_upper_limit,
    )
    draw_disease_lec_bars(
        lec_disease_axis, disease_lec, disease_colors, shared_upper_limit
    )
    draw_marker_dotplot(marker_axis, marker_data)

    figure.text(0.292, 0.962, "Aging", ha="center", va="center", fontsize=11, fontweight="bold")
    figure.text(0.758, 0.962, "Diseases", ha="center", va="center", fontsize=11, fontweight="bold")
    figure.text(0.018, 0.975, "B", ha="left", va="top", fontsize=12, fontweight="bold")

    age_aging_axis.set_title("Retained-donor age distribution", pad=7, fontweight="bold")
    age_disease_axis.set_title("Analysis-cohort age distribution", pad=7, fontweight="bold")
    lec_aging_axis.set_title("LECs in retained donors", pad=7, fontweight="bold")
    lec_disease_axis.set_title("LECs in tier-retained donors", pad=7, fontweight="bold")
    marker_axis.set_title(
        "LEC marker expression across all cardiac EC candidates",
        pad=17,
        fontweight="bold",
    )

    age_aging_axis.set_ylabel("Donor age (years)")
    lec_aging_axis.set_ylabel("Final LECs (cells)")
    plt.setp(age_disease_axis.get_yticklabels(), visible=False)
    plt.setp(lec_disease_axis.get_yticklabels(), visible=False)
    age_disease_axis.tick_params(axis="y", length=0)
    lec_disease_axis.tick_params(axis="y", length=0)

    reported = disease_donor.loc[disease_donor["age_numeric"].notna()]
    reported_datasets = ", ".join(sorted(reported["dataset_folder"].unique()))
    age_disease_axis.text(
        0.99,
        0.98,
        f"Numeric age: {len(reported)}/{len(disease_donor)} donors\n({reported_datasets} only)",
        transform=age_disease_axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color="#555555",
    )

    figure.text(
        0.5,
        0.025,
        "Aging and D01/D08 disease cohorts use \u226525 final LECs/donor.\n"
        "D11 HFpEF and matched NF use \u226510 (exploratory), with the nested \u226515 subset overlaid; disease labels show numeric-age/retained donors; NR, not reported.\n"
        "The marker dot plot is selection QC across all Step 06 cardiac EC candidates and is intentionally not donor-filtered.",
        ha="center",
        va="center",
        fontsize=6.3,
        color="#3F3F3F",
    )

    return figure


def main() -> int:
    root = project_root()
    inputs = load_inputs(root)

    aging_age = prepare_aging_age(inputs["aging_filter"])
    disease_donor = prepare_disease_donors(inputs["disease_filter"])
    aging_lec = prepare_aging_lec(inputs["aging_filter"])
    disease_lec = prepare_disease_lec(inputs["disease_filter"])
    colors = condition_colors(inputs["disease_color"])
    marker_data = prepare_marker_dotplot_data(
        inputs["aging_ec_marker"],
        inputs["aging_lec_cluster"],
        inputs["disease_ec_marker"],
        inputs["disease_lec_cluster"],
    )

    output = root / OUTPUT_RELATIVE
    output.mkdir(parents=True, exist_ok=True)

    source_data = build_source_data(
        root, aging_age, disease_donor, aging_lec, disease_lec, marker_data
    )
    source_data.to_csv(output / "Fig01B_source_data.csv", index=False)
    build_input_manifest(root, inputs["paths"]).to_csv(
        output / "Fig01B_input_csv_manifest.csv", index=False
    )

    figure = create_figure(
        aging_age, disease_donor, aging_lec, disease_lec, colors, marker_data
    )
    figure.savefig(
        output / "Fig01B_donor_age_and_LEC_counts.pdf",
        format="pdf",
        metadata={
            "Title": "Figure 1B: donor age distributions and final LEC counts",
            "Author": "Current-result figure pipeline",
            "Subject": SCRIPT_VERSION,
        },
    )
    figure.savefig(
        output / "Fig01B_donor_age_and_LEC_counts.png",
        format="png",
        dpi=600,
    )
    figure.savefig(
        output / "Fig01B_donor_age_and_LEC_counts.jpg",
        format="jpg",
        dpi=600,
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)

    print(f"Completed {SCRIPT_VERSION}")
    print(f"Output: {output}")
    print(f"Aging donors with age: {len(aging_age)}/{len(aging_age)}")
    print(
        "Disease donors with age: "
        f"{disease_donor['age_numeric'].notna().sum()}/{len(disease_donor)}"
    )
    print(
        "Aging LEC bars: "
        + "; ".join(
            f"{row.age_quartile}={int(row.n_LEC_cells)}"
            for row in aging_lec.itertuples(index=False)
        )
    )
    print(
        "Disease LEC bars: "
        + "; ".join(
            f"{row.display_group}={int(row.n_LEC_cells)}"
            for row in disease_lec.itertuples(index=False)
        )
    )
    marker_counts = (
        marker_data[["display_group", "n_cells"]]
        .drop_duplicates()
        .set_index("display_group")["n_cells"]
    )
    print(
        "Marker dot-plot cells: "
        + "; ".join(
            f"{group}={int(count)}" for group, count in marker_counts.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
