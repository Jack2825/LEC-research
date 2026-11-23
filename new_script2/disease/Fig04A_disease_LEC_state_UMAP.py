"""Create Figure 4A: disease LEC-state UMAP from the current Step08 result."""

from __future__ import annotations

from pathlib import Path
import re

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "new_result2_disease" / "08_combined_LEC_recluster"
INPUT_H5AD = INPUT_DIR / "08_combined_final_LEC_reclustered.h5ad"
STATE_MAPPING_CSV = INPUT_DIR / "08_state_name_mapping.csv"
STATE_COUNTS_CSV = INPUT_DIR / "08_state_counts.csv"
OUTPUT_DIR = ROOT / "new_result2" / "disease" / "Fig04A_disease_LEC_state_UMAP"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG = OUTPUT_DIR / "Fig04A_disease_LEC_state_UMAP.png"
PDF = OUTPUT_DIR / "Fig04A_disease_LEC_state_UMAP.pdf"
JPG = OUTPUT_DIR / "Fig04A_disease_LEC_state_UMAP.jpg"
SOURCE_CSV = OUTPUT_DIR / "Fig04A_source_data.csv"
MANIFEST_CSV = OUTPUT_DIR / "Fig04A_input_manifest.csv"

EXPECTED_N_CELLS = 6291
EXPECTED_STATES = [f"state{state}" for state in range(9)]

# Fixed offsets for the executed Step08 embedding. Labels are outside dense
# state cores and connected back to state medians with short leader lines.
LABEL_OFFSETS = {
    "state0": (-1.55, -0.15),
    "state1": (0.05, 1.15),
    "state2": (1.55, 0.05),
    "state3": (-1.45, 0.70),
    "state4": (-0.05, -1.05),
    "state5": (1.35, 0.55),
    "state6": (1.30, 0.35),
    "state7": (0.95, -1.00),
    "state8": (-1.35, -0.65),
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def state_number(value: object) -> int:
    match = re.fullmatch(r"state(\d+)", str(value).strip())
    if match is None:
        raise ValueError(f"Unexpected state label: {value!r}")
    return int(match.group(1))


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(STATE_MAPPING_CSV)
    counts = pd.read_csv(STATE_COUNTS_CSV)
    require_columns(
        mapping,
        {"combined_leiden", "LEC_state", "color_hex"},
        STATE_MAPPING_CSV.name,
    )
    require_columns(
        counts,
        {"LEC_state", "n_cells", "percent_of_all_LECs"},
        STATE_COUNTS_CSV.name,
    )

    backed = ad.read_h5ad(INPUT_H5AD, backed="r")
    try:
        if backed.n_obs != EXPECTED_N_CELLS:
            raise ValueError(f"Expected {EXPECTED_N_CELLS} LECs; found {backed.n_obs}.")
        if "X_umap" not in backed.obsm:
            raise KeyError("X_umap is absent from the disease Step08 h5ad.")
        for column in ["LEC_state", "combined_leiden"]:
            if column not in backed.obs:
                raise KeyError(f"{column} is absent from the disease Step08 h5ad.")
        coordinates = np.asarray(backed.obsm["X_umap"], dtype=float)
        cells = pd.DataFrame(
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

    observed_states = sorted(cells["LEC_state"].unique(), key=state_number)
    if observed_states != EXPECTED_STATES:
        raise ValueError(f"Expected {EXPECTED_STATES}; found {observed_states}.")
    if not np.isfinite(cells[["UMAP1", "UMAP2"]].to_numpy()).all():
        raise ValueError("The disease UMAP contains non-finite coordinates.")

    mapping = mapping.copy()
    mapping["combined_leiden"] = mapping["combined_leiden"].astype(str)
    mapping["LEC_state"] = mapping["LEC_state"].astype(str)
    counts = counts.copy()
    counts["LEC_state"] = counts["LEC_state"].astype(str)
    counts["n_cells"] = pd.to_numeric(counts["n_cells"], errors="raise").astype(int)
    counts["percent_of_all_LECs"] = pd.to_numeric(
        counts["percent_of_all_LECs"], errors="raise"
    )

    cells = cells.merge(
        mapping[["combined_leiden", "LEC_state", "color_hex"]],
        on=["combined_leiden", "LEC_state"],
        how="left",
        validate="many_to_one",
    ).merge(
        counts[["LEC_state", "n_cells", "percent_of_all_LECs"]],
        on="LEC_state",
        how="left",
        validate="many_to_one",
    )
    if cells[["color_hex", "n_cells", "percent_of_all_LECs"]].isna().any().any():
        raise ValueError("Disease state mapping/counts did not cover every cell.")

    observed_counts = cells["LEC_state"].value_counts().to_dict()
    reported_counts = counts.set_index("LEC_state")["n_cells"].to_dict()
    if observed_counts != reported_counts:
        raise ValueError(f"H5AD and state-count CSV differ: {observed_counts} != {reported_counts}")

    cells["state_number"] = cells["LEC_state"].map(state_number)
    cells["display_label"] = cells["state_number"].map(lambda x: f"State {x}")
    cells = cells.rename(
        columns={
            "n_cells": "state_n_cells",
            "percent_of_all_LECs": "state_percent_of_all_LECs",
        }
    )
    counts["state_number"] = counts["LEC_state"].map(state_number)
    return cells, counts.sort_values("state_number")


def draw_figure(cells: pd.DataFrame, counts: pd.DataFrame) -> plt.Figure:
    figure = plt.figure(figsize=(9.2, 6.3), facecolor="white")
    axis = figure.add_axes([0.055, 0.095, 0.735, 0.78])

    palettes = (
        cells[["LEC_state", "color_hex"]]
        .drop_duplicates()
        .set_index("LEC_state")["color_hex"]
        .to_dict()
    )
    order = sorted(
        EXPECTED_STATES,
        key=lambda state: -int((cells["LEC_state"] == state).sum()),
    )
    for state in order:
        subset = cells[cells["LEC_state"].eq(state)]
        axis.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=5.5,
            c=palettes[state],
            alpha=0.78,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    medians = cells.groupby("LEC_state")[["UMAP1", "UMAP2"]].median()
    for state in EXPECTED_STATES:
        x = float(medians.loc[state, "UMAP1"])
        y = float(medians.loc[state, "UMAP2"])
        dx, dy = LABEL_OFFSETS[state]
        axis.annotate(
            f"State {state_number(state)}",
            xy=(x, y),
            xytext=(x + dx, y + dy),
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#16191B",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": palettes[state],
                "linewidth": 1.2,
                "alpha": 0.98,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": palettes[state],
                "linewidth": 1.0,
                "shrinkA": 2,
                "shrinkB": 2,
                "connectionstyle": "arc3,rad=0.05",
            },
            zorder=5,
        )

    x_values = cells["UMAP1"].to_numpy(float)
    y_values = cells["UMAP2"].to_numpy(float)
    x_range = x_values.max() - x_values.min()
    y_range = y_values.max() - y_values.min()
    axis.set_xlim(x_values.min() - 0.13 * x_range, x_values.max() + 0.15 * x_range)
    axis.set_ylim(y_values.min() - 0.15 * y_range, y_values.max() + 0.15 * y_range)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()

    figure.text(0.018, 0.965, "A", fontsize=16, fontweight="bold", va="top")
    figure.text(0.055, 0.965, "Disease LEC states", fontsize=13, fontweight="bold", va="top")
    figure.text(
        0.055,
        0.922,
        "UMAP of 6,291 final LECs  |  Leiden resolution 0.6",
        fontsize=9,
        color="#4D555A",
        va="top",
    )

    figure.text(0.815, 0.865, "State composition", fontsize=10, fontweight="bold")
    figure.text(0.815, 0.832, "Cells (% of disease LECs)", fontsize=8, color="#555D62")
    count_lookup = counts.set_index("LEC_state")
    y_start = 0.785
    y_step = 0.073
    for index, state in enumerate(EXPECTED_STATES):
        y = y_start - index * y_step
        n_cells = int(count_lookup.loc[state, "n_cells"])
        percent = float(count_lookup.loc[state, "percent_of_all_LECs"])
        figure.text(0.818, y, "●", color=palettes[state], fontsize=12, va="center")
        figure.text(
            0.842,
            y,
            f"State {state_number(state)}",
            fontsize=8.8,
            fontweight="bold",
            va="center",
        )
        figure.text(
            0.975,
            y,
            f"{n_cells:,} ({percent:.1f}%)",
            fontsize=8.2,
            ha="right",
            va="center",
            color="#33383B",
        )

    figure.text(
        0.055,
        0.026,
        "State labels and UMAP coordinates are taken directly from the executed disease Step08 object; no reclustering was performed.",
        fontsize=7.6,
        color="#555D62",
        va="bottom",
    )
    return figure


def write_outputs(cells: pd.DataFrame, counts: pd.DataFrame) -> None:
    source = cells[
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
    source.insert(0, "source_h5ad", INPUT_H5AD.relative_to(ROOT).as_posix())
    source.to_csv(SOURCE_CSV, index=False)

    manifest = pd.DataFrame(
        [
            {
                "input_file": INPUT_H5AD.relative_to(ROOT).as_posix(),
                "input_type": "H5AD",
                "used_fields": "obsm/X_umap;obs/LEC_state;obs/combined_leiden",
                "purpose": "executed disease Step08 coordinates and state assignments",
            },
            {
                "input_file": STATE_MAPPING_CSV.relative_to(ROOT).as_posix(),
                "input_type": "CSV",
                "used_fields": "combined_leiden;LEC_state;color_hex",
                "purpose": "state-name and color mapping",
            },
            {
                "input_file": STATE_COUNTS_CSV.relative_to(ROOT).as_posix(),
                "input_type": "CSV",
                "used_fields": "LEC_state;n_cells;percent_of_all_LECs",
                "purpose": "state-count validation and composition key",
            },
        ]
    )
    manifest.to_csv(MANIFEST_CSV, index=False)

    figure = draw_figure(cells, counts)
    figure.savefig(PNG, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(PDF, bbox_inches="tight", facecolor="white")
    figure.savefig(
        JPG,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(figure)


def main() -> None:
    cells, counts = load_data()
    write_outputs(cells, counts)
    print(f"Figure 4A written to: {OUTPUT_DIR}")
    print(f"Cells: {len(cells):,}")
    print(f"States: {cells['LEC_state'].nunique()}")


if __name__ == "__main__":
    main()
