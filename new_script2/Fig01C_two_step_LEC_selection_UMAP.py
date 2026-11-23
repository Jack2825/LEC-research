"""Create Figure 1C: representative two-step cardiac LEC selection UMAPs.

Two current cohorts are shown without recomputing either embedding:
1. all QC-passed cells, highlighting the EC candidates carried into Step 06;
2. the EC-candidate embedding, highlighting the final LEC selection.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "new_result2" / "Fig01C_two_step_LEC_selection_UMAP"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_PNG = OUTPUT_DIR / "Fig01C_two_step_LEC_selection_UMAP.png"
FIGURE_PDF = OUTPUT_DIR / "Fig01C_two_step_LEC_selection_UMAP.pdf"
FIGURE_JPG = OUTPUT_DIR / "Fig01C_two_step_LEC_selection_UMAP.jpg"
SOURCE_CSV = OUTPUT_DIR / "Fig01C_source_data.csv.gz"
MANIFEST_CSV = OUTPUT_DIR / "Fig01C_input_manifest.csv"

COHORTS = [
    {
        "branch": "Aging",
        "cohort": "Dataset13 Q1 NF",
        "allcell": ROOT
        / "new_result2_aging/03/dataset13/Q1/dataset13_Q1_clustered.h5ad",
        "ec": ROOT
        / "new_result2_aging/06_EC_recluster/dataset13/Q1/dataset13_Q1_EC_reclustered.h5ad",
    },
    {
        "branch": "Disease",
        "cohort": "Dataset01 DCM",
        "allcell": ROOT
        / "new_result2_disease/03/dataset01/DCM/dataset01_DCM_clustered.h5ad",
        "ec": ROOT
        / "new_result2_disease/06_EC_recluster/dataset01/DCM/dataset01_DCM_EC_reclustered.h5ad",
    },
]

TEXT = "#202428"
SUBTEXT = "#5C646A"
BACKGROUND = "#CAD1D6"
EC_COLOR = "#3B82A0"
LEC_COLOR = "#C95C43"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_umap(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = ad.read_h5ad(path, backed="r")
    try:
        if "X_umap" not in data.obsm:
            raise KeyError(f"X_umap is absent from {path}")
        coordinates = np.asarray(data.obsm["X_umap"], dtype=float)
        if coordinates.shape != (data.n_obs, 2) or not np.isfinite(coordinates).all():
            raise ValueError(f"Invalid UMAP coordinates in {path}")
        obs = data.obs.copy()
        obs.index = obs.index.astype(str)
    finally:
        data.file.close()
    return coordinates, obs


def load_cohort(spec: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame]:
    all_path = Path(spec["allcell"])
    ec_path = Path(spec["ec"])
    all_xy, all_obs = read_umap(all_path)
    ec_xy, ec_obs = read_umap(ec_path)

    ec_ids = set(ec_obs.index)
    missing_ec = ec_ids.difference(all_obs.index)
    if missing_ec:
        raise ValueError(
            f"{len(missing_ec)} EC candidates are absent from the all-cell object: {all_path}"
        )
    if "selected_as_LEC" not in ec_obs:
        raise KeyError(f"selected_as_LEC is absent from {ec_path}")

    selected_ec = all_obs.index.isin(ec_ids)
    selected_lec = ec_obs["selected_as_LEC"].astype(bool).to_numpy()
    if selected_ec.sum() != len(ec_obs) or selected_lec.sum() == 0:
        raise ValueError(f"Unexpected EC/LEC selection counts for {spec['cohort']}")

    stage1 = pd.DataFrame(
        {
            "figure_panel": "1C",
            "branch": spec["branch"],
            "cohort": spec["cohort"],
            "stage": "Step 1: EC candidate selection",
            "cell_id": all_obs.index,
            "UMAP1": all_xy[:, 0],
            "UMAP2": all_xy[:, 1],
            "display_class": np.where(selected_ec, "EC candidate", "Other QC-passed cell"),
            "selected_for_next_stage": selected_ec,
            "source_file": all_path.relative_to(ROOT).as_posix(),
        }
    )
    stage2 = pd.DataFrame(
        {
            "figure_panel": "1C",
            "branch": spec["branch"],
            "cohort": spec["cohort"],
            "stage": "Step 2: final LEC selection",
            "cell_id": ec_obs.index,
            "UMAP1": ec_xy[:, 0],
            "UMAP2": ec_xy[:, 1],
            "display_class": np.where(selected_lec, "Final LEC", "Other EC candidate"),
            "selected_for_next_stage": selected_lec,
            "source_file": ec_path.relative_to(ROOT).as_posix(),
        }
    )
    payload: dict[str, object] = {
        **spec,
        "all_xy": all_xy,
        "selected_ec": selected_ec,
        "ec_xy": ec_xy,
        "selected_lec": selected_lec,
    }
    return payload, pd.concat([stage1, stage2], ignore_index=True)


def clean_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("")
    axis.set_ylabel("")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_aspect("equal", adjustable="datalim")


def draw_stage(
    axis: plt.Axes,
    coordinates: np.ndarray,
    selected: np.ndarray,
    selected_label: str,
    selected_color: str,
) -> None:
    n_cells = len(coordinates)
    point_size = 1.2 if n_cells > 100_000 else 2.0
    axis.scatter(
        coordinates[~selected, 0],
        coordinates[~selected, 1],
        s=point_size,
        c=BACKGROUND,
        alpha=0.58,
        linewidths=0,
        rasterized=True,
    )
    axis.scatter(
        coordinates[selected, 0],
        coordinates[selected, 1],
        s=point_size + 1.0,
        c=selected_color,
        alpha=0.90,
        linewidths=0,
        rasterized=True,
    )
    clean_axis(axis)
    axis.text(
        0.02,
        0.02,
        f"{selected_label}: {selected.sum():,} / {n_cells:,} cells",
        transform=axis.transAxes,
        fontsize=9.2,
        color=TEXT,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
    )


def write_manifest() -> None:
    records: list[dict[str, str]] = []
    for spec in COHORTS:
        records.extend(
            [
                {
                    "figure_panel": "1C step 1",
                    "branch": str(spec["branch"]),
                    "cohort": str(spec["cohort"]),
                    "input_file": Path(spec["allcell"]).relative_to(ROOT).as_posix(),
                    "used_fields": "obsm/X_umap;obs_names",
                    "purpose": "all-cell embedding and EC-candidate membership",
                },
                {
                    "figure_panel": "1C step 2",
                    "branch": str(spec["branch"]),
                    "cohort": str(spec["cohort"]),
                    "input_file": Path(spec["ec"]).relative_to(ROOT).as_posix(),
                    "used_fields": "obsm/X_umap;obs/selected_as_LEC;obs_names",
                    "purpose": "EC-candidate embedding and final LEC selection",
                },
            ]
        )
    pd.DataFrame(records).to_csv(MANIFEST_CSV, index=False)


def main() -> None:
    loaded: list[dict[str, object]] = []
    source_parts: list[pd.DataFrame] = []
    for spec in COHORTS:
        payload, source = load_cohort(spec)
        loaded.append(payload)
        source_parts.append(source)
    pd.concat(source_parts, ignore_index=True).to_csv(
        SOURCE_CSV, index=False, compression="gzip"
    )
    write_manifest()

    figure, axes = plt.subplots(2, 2, figsize=(12.6, 9.0), facecolor="white")
    plt.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.10, wspace=0.11, hspace=0.20)

    for row, payload in enumerate(loaded):
        draw_stage(
            axes[row, 0],
            payload["all_xy"],
            payload["selected_ec"],
            "EC candidates",
            EC_COLOR,
        )
        draw_stage(
            axes[row, 1],
            payload["ec_xy"],
            payload["selected_lec"],
            "Final LECs",
            LEC_COLOR,
        )
        axes[row, 0].text(
            -0.08,
            0.5,
            f"{payload['branch']}\n{payload['cohort']}",
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=TEXT,
        )

    axes[0, 0].set_title(
        "Step 1  |  Select EC candidates from all QC-passed cells",
        fontweight="bold",
        pad=13,
        color=TEXT,
    )
    axes[0, 1].set_title(
        "Step 2  |  Identify final LECs within EC candidates",
        fontweight="bold",
        pad=13,
        color=TEXT,
    )

    for row in range(2):
        left = axes[row, 0].get_position()
        right = axes[row, 1].get_position()
        arrow = FancyArrowPatch(
            (left.x1 + 0.006, (left.y0 + left.y1) / 2),
            (right.x0 - 0.006, (right.y0 + right.y1) / 2),
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#6E777D",
        )
        figure.add_artist(arrow)

    figure.text(0.025, 0.955, "C", fontsize=18, fontweight="bold", color=TEXT, va="top")
    figure.text(
        0.08,
        0.955,
        "Two-step selection of cardiac lymphatic endothelial cells",
        fontsize=19,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    figure.text(
        0.08,
        0.918,
        "One representative aging cohort and one representative disease cohort; embeddings and labels are current pipeline outputs.",
        fontsize=10.5,
        color=SUBTEXT,
        va="top",
    )
    figure.text(
        0.08,
        0.025,
        "Gray cells remain visible to show the selection boundary. No UMAP, clustering, annotation, or cell selection was recomputed for this panel.",
        fontsize=9.2,
        color=SUBTEXT,
        va="bottom",
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

    print(f"Figure 1C written to: {OUTPUT_DIR}")
    for payload in loaded:
        print(
            f"{payload['cohort']}: {int(np.sum(payload['selected_ec'])):,} EC candidates; "
            f"{int(np.sum(payload['selected_lec'])):,} final LECs"
        )


if __name__ == "__main__":
    main()
