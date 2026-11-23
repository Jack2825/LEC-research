#!/usr/bin/env python3
"""Create the publication Figure 2A UMAP from the current aging Step08 result.

The plot uses the executed Step08 UMAP coordinates and state assignments without
reclustering or changing any cell. State labels are moved away from the cell
clouds and connected to robust state centroids with leader lines.

Outputs
-------
new_result2/aging/Fig02A_aging_LEC_state_UMAP/
    Fig02A_aging_LEC_state_UMAP.png
    Fig02A_aging_LEC_state_UMAP.pdf
    Fig02A_aging_LEC_state_UMAP.jpg
    Fig02A_source_data.csv
    Fig02A_input_manifest.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_VERSION = "Fig02A_aging_LEC_state_UMAP_v1_2026_08_23"

INPUT_H5AD = Path(
    "new_result2_aging/08_combined_LEC_recluster/dataset13/"
    "08_combined_final_LEC_reclustered.h5ad"
)
STATE_MAPPING_CSV = Path(
    "new_result2_aging/08_combined_LEC_recluster/dataset13/"
    "08_state_name_mapping.csv"
)
STATE_COUNTS_CSV = Path(
    "new_result2_aging/08_combined_LEC_recluster/dataset13/08_state_counts.csv"
)
OUTPUT_DIR = Path("new_result2/aging/Fig02A_aging_LEC_state_UMAP")

EXPECTED_N_CELLS = 1119
EXPECTED_STATES = [f"state{number}" for number in range(11)]

# Offsets are in UMAP coordinate units and apply to the current, fixed Step08
# embedding. They keep text off the cells while preserving an unambiguous link
# to each state's median UMAP position.
LABEL_OFFSETS = {
    "state0": (3.8, 2.6),
    "state1": (3.2, 0.0),
    "state2": (-3.2, -0.5),
    "state3": (2.4, 0.2),
    "state4": (-1.5, -1.7),
    "state5": (2.0, 1.2),
    "state6": (1.5, -1.7),
    "state7": (-2.0, 1.1),
    "state8": (-2.1, 1.1),
    "state9": (0.1, 1.8),
    "state10": (-2.3, 0.7),
}


def project_root() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[2]


def state_number(value: object) -> int:
    match = re.fullmatch(r"state(\d+)", str(value).strip())
    if match is None:
        raise ValueError(f"Unexpected state label: {value!r}")
    return int(match.group(1))


def load_current_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    h5ad_path = root / INPUT_H5AD
    mapping_path = root / STATE_MAPPING_CSV
    counts_path = root / STATE_COUNTS_CSV
    for path in [h5ad_path, mapping_path, counts_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    mapping = pd.read_csv(mapping_path)
    counts = pd.read_csv(counts_path)
    required_mapping = {"combined_leiden", "LEC_state", "color_hex"}
    required_counts = {
        "LEC_state",
        "n_cells",
        "fraction_of_all_LECs",
        "percent_of_all_LECs",
    }
    if not required_mapping.issubset(mapping.columns):
        raise ValueError(f"State mapping is missing {required_mapping - set(mapping.columns)}")
    if not required_counts.issubset(counts.columns):
        raise ValueError(f"State counts are missing {required_counts - set(counts.columns)}")

    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if "X_umap" not in backed.obsm:
            raise KeyError("X_umap is absent from the Step08 h5ad.")
        for column in ["LEC_state", "combined_leiden"]:
            if column not in backed.obs:
                raise KeyError(f"{column} is absent from the Step08 h5ad.")
        coordinates = np.asarray(backed.obsm["X_umap"], dtype=float)
        frame = pd.DataFrame(
            {
                "cell_id": backed.obs_names.astype(str),
                "UMAP1": coordinates[:, 0],
                "UMAP2": coordinates[:, 1],
                "LEC_state": backed.obs["LEC_state"].astype(str).to_numpy(),
                "combined_leiden": backed.obs["combined_leiden"].astype(str).to_numpy(),
            }
        )
    finally:
        backed.file.close()

    if len(frame) != EXPECTED_N_CELLS:
        raise ValueError(f"Expected {EXPECTED_N_CELLS} LECs; found {len(frame)}")
    if not np.isfinite(frame[["UMAP1", "UMAP2"]].to_numpy()).all():
        raise ValueError("The Step08 UMAP contains non-finite coordinates.")

    observed_states = sorted(frame["LEC_state"].unique(), key=state_number)
    if observed_states != EXPECTED_STATES:
        raise ValueError(f"Expected {EXPECTED_STATES}; found {observed_states}")

    mapping = mapping.copy()
    mapping["combined_leiden"] = mapping["combined_leiden"].astype(str)
    mapping["LEC_state"] = mapping["LEC_state"].astype(str)
    counts = counts.copy()
    counts["LEC_state"] = counts["LEC_state"].astype(str)
    counts["n_cells"] = pd.to_numeric(counts["n_cells"], errors="raise").astype(int)
    counts["percent_of_all_LECs"] = pd.to_numeric(
        counts["percent_of_all_LECs"], errors="raise"
    )

    merged = frame.merge(
        mapping[["combined_leiden", "LEC_state", "color_hex"]],
        on=["combined_leiden", "LEC_state"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        counts[["LEC_state", "n_cells", "percent_of_all_LECs"]],
        on="LEC_state",
        how="left",
        validate="many_to_one",
    )
    if merged[["color_hex", "n_cells", "percent_of_all_LECs"]].isna().any().any():
        raise ValueError("State mapping/count metadata did not cover every plotted cell.")

    observed_counts = merged["LEC_state"].value_counts().to_dict()
    reported_counts = counts.set_index("LEC_state")["n_cells"].to_dict()
    if observed_counts != reported_counts:
        raise ValueError(
            f"Step08 h5ad state counts do not match the state-count CSV: "
            f"{observed_counts} != {reported_counts}"
        )

    merged["state_number"] = merged["LEC_state"].map(state_number)
    merged["display_label"] = merged["state_number"].map(lambda value: f"State {value}")
    merged = merged.rename(
        columns={
            "n_cells": "state_n_cells",
            "percent_of_all_LECs": "state_percent_of_all_LECs",
        }
    )
    return merged, counts.sort_values("LEC_state", key=lambda x: x.map(state_number))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def make_figure(frame: pd.DataFrame, counts: pd.DataFrame) -> plt.Figure:
    configure_style()
    figure = plt.figure(figsize=(8.35, 4.7), constrained_layout=False)
    axis = figure.add_axes([0.055, 0.09, 0.735, 0.79])

    palettes = (
        frame[["LEC_state", "color_hex"]]
        .drop_duplicates()
        .set_index("LEC_state")["color_hex"]
        .to_dict()
    )
    order = sorted(EXPECTED_STATES, key=lambda state: -int((frame["LEC_state"] == state).sum()))
    for state in order:
        subset = frame.loc[frame["LEC_state"].eq(state)]
        axis.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=10,
            c=palettes[state],
            alpha=0.86,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    centroids = frame.groupby("LEC_state", observed=True)[["UMAP1", "UMAP2"]].median()
    for state in EXPECTED_STATES:
        center_x = float(centroids.loc[state, "UMAP1"])
        center_y = float(centroids.loc[state, "UMAP2"])
        dx, dy = LABEL_OFFSETS[state]
        axis.annotate(
            f"State {state_number(state)}",
            xy=(center_x, center_y),
            xytext=(center_x + dx, center_y + dy),
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="#111111",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": palettes[state],
                "linewidth": 1.15,
                "alpha": 0.97,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": palettes[state],
                "linewidth": 0.9,
                "shrinkA": 2,
                "shrinkB": 2,
                "connectionstyle": "arc3,rad=0.05",
            },
            zorder=5,
        )

    x_values = frame["UMAP1"].to_numpy(float)
    y_values = frame["UMAP2"].to_numpy(float)
    x_pad = 0.045 * (x_values.max() - x_values.min())
    y_pad = 0.12 * (y_values.max() - y_values.min())
    axis.set_xlim(x_values.min() - x_pad, x_values.max() + x_pad)
    axis.set_ylim(y_values.min() - y_pad, y_values.max() + y_pad)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()

    figure.text(0.018, 0.965, "A", ha="left", va="top", fontsize=15, fontweight="bold")
    figure.text(0.055, 0.965, "Aging LEC states", ha="left", va="top", fontsize=12, fontweight="bold")
    figure.text(
        0.055,
        0.918,
        "UMAP of 1,119 final LECs  |  Leiden resolution 0.3",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4D4D4D",
    )

    figure.text(0.815, 0.865, "State composition", fontsize=9, fontweight="bold", ha="left")
    figure.text(0.815, 0.83, "Cells (% of LECs)", fontsize=7.4, color="#555555", ha="left")
    y_start = 0.785
    y_step = 0.061
    count_lookup = counts.set_index("LEC_state")
    for index, state in enumerate(EXPECTED_STATES):
        y = y_start - index * y_step
        n_cells = int(count_lookup.loc[state, "n_cells"])
        percent = float(count_lookup.loc[state, "percent_of_all_LECs"])
        figure.text(0.819, y, "●", color=palettes[state], fontsize=10.5, ha="left", va="center")
        figure.text(
            0.842,
            y,
            f"State {state_number(state)}",
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="center",
        )
        figure.text(
            0.965,
            y,
            f"{n_cells:,} ({percent:.1f}%)",
            fontsize=7.6,
            ha="right",
            va="center",
            color="#333333",
        )

    figure.text(
        0.055,
        0.025,
        "State labels identify the executed Step08 clustering; no cells or coordinates were recalculated.",
        fontsize=7,
        color="#555555",
        ha="left",
        va="bottom",
    )
    return figure


def write_outputs(root: Path, frame: pd.DataFrame, counts: pd.DataFrame) -> Path:
    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)

    source = frame[
        [
            "cell_id",
            "UMAP1",
            "UMAP2",
            "combined_leiden",
            "LEC_state",
            "state_number",
            "display_label",
            "color_hex",
            "state_n_cells",
            "state_percent_of_all_LECs",
        ]
    ].copy()
    source.insert(0, "script_version", SCRIPT_VERSION)
    source.insert(1, "source_h5ad", INPUT_H5AD.as_posix())
    source.to_csv(output / "Fig02A_source_data.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "input_file": INPUT_H5AD.as_posix(),
                "input_type": "H5AD",
                "used_fields": "obsm/X_umap; obs/LEC_state; obs/combined_leiden",
                "purpose": "current executed Step08 cell coordinates and state assignments",
            },
            {
                "input_file": STATE_MAPPING_CSV.as_posix(),
                "input_type": "CSV",
                "used_fields": "combined_leiden;LEC_state;color_hex",
                "purpose": "executed state-name and color mapping",
            },
            {
                "input_file": STATE_COUNTS_CSV.as_posix(),
                "input_type": "CSV",
                "used_fields": "LEC_state;n_cells;percent_of_all_LECs",
                "purpose": "state-count validation and composition key",
            },
        ]
    )
    manifest.insert(0, "script_version", SCRIPT_VERSION)
    manifest.to_csv(output / "Fig02A_input_manifest.csv", index=False)

    figure = make_figure(frame, counts)
    figure.savefig(output / "Fig02A_aging_LEC_state_UMAP.pdf", bbox_inches="tight")
    figure.savefig(
        output / "Fig02A_aging_LEC_state_UMAP.png",
        dpi=600,
        bbox_inches="tight",
    )
    figure.savefig(
        output / "Fig02A_aging_LEC_state_UMAP.jpg",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)
    return output


def main() -> int:
    root = project_root()
    frame, counts = load_current_data(root)
    output = write_outputs(root, frame, counts)
    print(f"Figure 2A written to: {output}")
    print(f"Cells: {len(frame):,}")
    print(f"States: {frame['LEC_state'].nunique()}")
    print("Inputs: Step08 h5ad + state mapping CSV + state counts CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
