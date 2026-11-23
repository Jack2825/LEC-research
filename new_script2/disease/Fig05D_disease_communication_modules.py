"""Create disease Figure 5D: dominant communication modules by disease."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "new_result2_disease" / "12_NicheNet_donor_pseudobulk"
OUTPUT_DIR = ROOT / "new_result2" / "disease" / "Fig05D_disease_communication_modules"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_PNG = OUTPUT_DIR / "Fig05D_disease_communication_modules.png"
FIGURE_PDF = OUTPUT_DIR / "Fig05D_disease_communication_modules.pdf"
FIGURE_JPG = OUTPUT_DIR / "Fig05D_disease_communication_modules.jpg"
SOURCE_CSV = OUTPUT_DIR / "Fig05D_source_data.csv"
MANIFEST_CSV = OUTPUT_DIR / "Fig05D_input_manifest.csv"


RUNS = {
    "DCM": {
        "folder": "dataset01_DCM_vs_NF",
        "group": "DCM",
        "dataset": "D01",
        "tier": "primary ≥25 LECs/donor",
        "expected_reference_donors": 12,
        "expected_disease_donors": 10,
    },
    "HCM": {
        "folder": "dataset01_HCM_vs_NF",
        "group": "HCM",
        "dataset": "D01",
        "tier": "primary ≥25 LECs/donor",
        "expected_reference_donors": 12,
        "expected_disease_donors": 15,
    },
    "DoxCM": {
        "folder": "dataset08_DoxCM_vs_NF",
        "group": "DoxCM",
        "dataset": "D08",
        "tier": "primary ≥25 LECs/donor",
        "expected_reference_donors": 4,
        "expected_disease_donors": 9,
    },
    "HFpEF": {
        "folder": "dataset11_HFpEF_vs_NF_min10",
        "group": "HFpEF",
        "dataset": "D11",
        "tier": "exploratory ≥10 LECs/donor",
        "expected_reference_donors": 6,
        "expected_disease_donors": 7,
        "sensitivity_folder": "dataset11_HFpEF_vs_NF_min15",
        "expected_sensitivity_reference_donors": 4,
        "expected_sensitivity_disease_donors": 5,
    },
}


# Exact representative axes chosen from current disease-prioritized rows.
# The template's DoxCM HMGB1-TLR example is not present in the current result;
# the supported rows are BGN-TLR1 and HMGB1-HAVCR2 and are labeled as such.
AXIS_SPECS = {
    "DCM": [
        ("VEGF–neuropilin maintenance", "EC1_cap", "VEGFC", "NRP2"),
        ("Semaphorin guidance", "Adip1", "SEMA3A", "NRP2"),
        ("BMP/activin maintenance", "EC7_endocardial", "BMP4", "ACVR2A"),
        ("TNF-family signaling", "LYVE1.IGF1.MP", "TNFSF13", "TNFRSF14"),
    ],
    "HCM": [
        ("HBEGF–ERBB signaling", "LYVE1.IGF1.MP", "HBEGF", "ERBB4"),
        ("Neuregulin–ERBB signaling", "EC7_endocardial", "NRG1", "ERBB4"),
        ("Laminin–integrin remodeling", "EC7_endocardial", "LAMA4", "ITGA9"),
        ("Collagen–integrin remodeling", "NC1_glial", "COL15A1", "ITGA1"),
    ],
    "DoxCM": [
        ("BMP/activin injury response", "EC7_endocardial", "BMP4", "ACVR2A"),
        ("MERTK injury clearance", "LYVE1.IGF1.MP", "GAS6", "MERTK"),
        ("TLR danger sensing", "EC7_endocardial", "BGN", "TLR1"),
        ("HMGB1 injury signaling", "vCM1", "HMGB1", "HAVCR2"),
        ("Endothelial adhesion", "EC6_ven", "SELE", "CEACAM1"),
    ],
    "HFpEF": [
        ("Collagen–integrin remodeling", "PC1_vent", "COL6A3", "ITGA2"),
        ("TGFβ/activin signaling", "EC5_art", "TGFB2", "ACVR1"),
        ("BMP-family signaling", "PC1_vent", "GDF11", "BMPR2"),
        ("WNT signaling", "LYVE1.IGF1.MP", "WNT5B", "LRP5"),
        ("NODAL/activin signaling", "vCM1", "NODAL", "ACVR2B"),
    ],
}


SENDER_LABELS = {
    "Adip1": "Adipocytes",
    "EC1_cap": "Capillary ECs",
    "EC5_art": "Arterial ECs",
    "EC6_ven": "Venous ECs",
    "EC7_endocardial": "Endocardial ECs",
    "LYVE1.IGF1.MP": "LYVE1+IGF1\nmacrophages",
    "NC1_glial": "Glial cells",
    "PC1_vent": "Ventricular\npericytes",
    "vCM1": "Ventricular CMs",
}

SENDER_COLORS = {
    "Adip1": "#E8D4AA",
    "EC1_cap": "#BFD7EA",
    "EC5_art": "#AFCDE2",
    "EC6_ven": "#B9D8E6",
    "EC7_endocardial": "#A8D5BA",
    "LYVE1.IGF1.MP": "#F1CF9B",
    "NC1_glial": "#C8D1D6",
    "PC1_vent": "#D7C4E8",
    "vCM1": "#EDB7B7",
}

PANEL_LETTERS = {"DCM": "E", "HCM": "F", "DoxCM": "G", "HFpEF": "H"}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

TEXT = "#202428"
SUBTEXT = "#5A6268"
GRID = "#DCE1E4"
EDGE = "#69747C"
PRIMARY = "#426F8E"
EXPLORATORY = "#A27743"


def require_columns(data: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(data.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def group_csv(folder: str) -> Path:
    return INPUT_ROOT / folder / "tables" / "12_MultiNicheNet_group_prioritization.csv"


def summary_csv(folder: str) -> Path:
    return INPUT_ROOT / folder / "12_run_summary.csv"


def read_completed_summary(disease: str, config: dict[str, object]) -> pd.DataFrame:
    path = summary_csv(str(config["folder"]))
    summary = pd.read_csv(path)
    require_columns(
        summary,
        {
            "disease",
            "n_reference_donors",
            "n_disease_donors",
            "validation_status",
            "multinichenet_completed",
            "status",
        },
        path.name,
    )
    if len(summary) != 1 or str(summary.iloc[0]["disease"]) != disease:
        raise ValueError(f"Unexpected run-summary identity for {disease}.")
    row = summary.iloc[0]
    if int(row["n_reference_donors"]) != int(config["expected_reference_donors"]):
        raise ValueError(f"Unexpected reference-donor count for {disease}.")
    if int(row["n_disease_donors"]) != int(config["expected_disease_donors"]):
        raise ValueError(f"Unexpected disease-donor count for {disease}.")
    if str(row["validation_status"]).lower() != "passed":
        raise ValueError(f"Validation did not pass for {disease}.")
    if str(row["multinichenet_completed"]).lower() not in {"true", "1"}:
        raise ValueError(f"MultiNicheNet is not complete for {disease}.")
    if str(row["status"]).lower() != "completed":
        raise ValueError(f"Run status is not completed for {disease}.")
    return summary


def read_disease_edges(folder: str, disease_group: str) -> pd.DataFrame:
    path = group_csv(folder)
    data = pd.read_csv(path, low_memory=False)
    require_columns(
        data,
        {
            "contrast",
            "group",
            "sender",
            "receiver",
            "ligand",
            "receptor",
            "prioritization_score",
            "max_scaled_activity",
            "scaled_pb_ligand",
            "scaled_pb_receptor",
            "fraction_expressing_ligand_receptor",
            "top_group",
        },
        path.name,
    )
    numeric = [
        "prioritization_score",
        "max_scaled_activity",
        "scaled_pb_ligand",
        "scaled_pb_receptor",
        "fraction_expressing_ligand_receptor",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=numeric)
    return data.loc[
        data["receiver"].eq("LEC_EC8_ln")
        & data["group"].eq(disease_group)
        & data["top_group"].eq(disease_group)
    ].copy()


def select_exact_edge(
    data: pd.DataFrame,
    sender: str,
    ligand: str,
    receptor: str,
    disease: str,
) -> pd.Series:
    matched = data.loc[
        data["sender"].eq(sender)
        & data["ligand"].eq(ligand)
        & data["receptor"].eq(receptor)
    ].sort_values("prioritization_score", ascending=False)
    if len(matched) != 1:
        raise ValueError(
            f"Expected one disease-prioritized row for {disease}: "
            f"{sender} -> {ligand}-{receptor}; found {len(matched)}"
        )
    return matched.iloc[0]


def select_sensitivity_edge(
    data: pd.DataFrame,
    sender: str,
    ligand: str,
    receptor: str,
) -> tuple[pd.Series, bool]:
    exact = data.loc[
        data["sender"].eq(sender)
        & data["ligand"].eq(ligand)
        & data["receptor"].eq(receptor)
    ].sort_values("prioritization_score", ascending=False)
    if len(exact) == 1:
        return exact.iloc[0], True
    if ligand != "NODAL":
        raise ValueError(f"HFpEF min15 exact axis is absent: {sender} {ligand}-{receptor}")
    module = data.loc[data["ligand"].eq(ligand)].sort_values(
        "prioritization_score", ascending=False
    )
    if module.empty:
        raise ValueError("HFpEF min15 NODAL module is absent.")
    # NODAL remains supported, but the prioritized receptor shifts from ACVR2B
    # at min10 to ACVR2A at min15.
    return module.iloc[0], False


def load_selected_axes() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    validation_records: list[dict[str, object]] = []
    min15_edges: pd.DataFrame | None = None

    for disease, config in RUNS.items():
        summary = read_completed_summary(disease, config)
        validation_records.append(
            {
                "disease": disease,
                "run_summary": summary_csv(str(config["folder"])),
                "summary": summary.iloc[0].to_dict(),
            }
        )
        edges = read_disease_edges(str(config["folder"]), str(config["group"]))

        if disease == "HFpEF":
            sensitivity_folder = str(config["sensitivity_folder"])
            sensitivity_summary = pd.read_csv(summary_csv(sensitivity_folder))
            row = sensitivity_summary.iloc[0]
            if (
                int(row["n_reference_donors"])
                != int(config["expected_sensitivity_reference_donors"])
                or int(row["n_disease_donors"])
                != int(config["expected_sensitivity_disease_donors"])
                or str(row["validation_status"]).lower() != "passed"
                or str(row["status"]).lower() != "completed"
            ):
                raise ValueError("HFpEF min15 sensitivity run failed validation.")
            validation_records.append(
                {
                    "disease": "HFpEF_min15_sensitivity",
                    "run_summary": summary_csv(sensitivity_folder),
                    "summary": row.to_dict(),
                }
            )
            min15_edges = read_disease_edges(sensitivity_folder, "HFpEF")

        for order, (module, sender, ligand, receptor) in enumerate(
            AXIS_SPECS[disease], start=1
        ):
            edge = select_exact_edge(edges, sender, ligand, receptor, disease)
            record = {
                "disease": disease,
                "dataset": config["dataset"],
                "analysis_tier": config["tier"],
                "module_order": order,
                "module": module,
                "sender": sender,
                "sender_label": SENDER_LABELS[sender].replace("\n", " "),
                "ligand": ligand,
                "receptor": receptor,
                "receiver": edge["receiver"],
                "contrast": edge["contrast"],
                "prioritization_score": float(edge["prioritization_score"]),
                "max_scaled_activity": float(edge["max_scaled_activity"]),
                "scaled_pb_ligand": float(edge["scaled_pb_ligand"]),
                "scaled_pb_receptor": float(edge["scaled_pb_receptor"]),
                "fraction_expressing_ligand_receptor": float(
                    edge["fraction_expressing_ligand_receptor"]
                ),
                "source_csv": group_csv(str(config["folder"])).relative_to(ROOT).as_posix(),
                "sensitivity_min15_sender": "",
                "sensitivity_min15_ligand": "",
                "sensitivity_min15_receptor": "",
                "sensitivity_min15_prioritization_score": np.nan,
                "sensitivity_min15_exact_axis": np.nan,
            }
            if disease == "HFpEF":
                assert min15_edges is not None
                sensitivity_edge, exact_axis = select_sensitivity_edge(
                    min15_edges, sender, ligand, receptor
                )
                record.update(
                    {
                        "sensitivity_min15_sender": sensitivity_edge["sender"],
                        "sensitivity_min15_ligand": sensitivity_edge["ligand"],
                        "sensitivity_min15_receptor": sensitivity_edge["receptor"],
                        "sensitivity_min15_prioritization_score": float(
                            sensitivity_edge["prioritization_score"]
                        ),
                        "sensitivity_min15_exact_axis": exact_axis,
                    }
                )
            records.append(record)

    selected = pd.DataFrame(records)
    if len(selected) != 18:
        raise ValueError(f"Expected 18 representative axes; found {len(selected)}.")
    selected["figure_panels"] = selected["disease"].map(
        lambda value: f"D;{PANEL_LETTERS[value]}"
    )
    selected["record_type"] = "representative_sender_ligand_LEC_axis"
    selected["selection_note"] = np.where(
        selected["disease"].eq("HFpEF"),
        "exploratory min10 axis with min15 sensitivity support",
        "primary >=25-LEC within-dataset disease-prioritized axis",
    )
    return selected, validation_records


def style_module_axis(axis: plt.Axes, show_xlabel: bool) -> None:
    axis.set_xlim(0.40, 1.01)
    axis.set_xticks([0.4, 0.6, 0.8, 1.0])
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=6)
    axis.tick_params(axis="x", colors="#525A60")
    axis.set_xlabel("Representative-edge priority score" if show_xlabel else "")


def draw_module_facet(
    axis: plt.Axes,
    disease: str,
    data: pd.DataFrame,
    show_xlabel: bool,
) -> None:
    subset = data.loc[data["disease"].eq(disease)].sort_values("module_order")
    y = np.arange(len(subset))[::-1]
    color = EXPLORATORY if disease == "HFpEF" else PRIMARY
    for position, (_, row) in zip(y, subset.iterrows(), strict=True):
        score = float(row["prioritization_score"])
        axis.hlines(position, 0.40, score, color=color, linewidth=1.45, alpha=0.65)
        axis.scatter(
            score,
            position,
            s=62,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.text(
            min(score + 0.016, 0.985),
            position,
            f"{score:.2f}",
            ha="left" if score < 0.965 else "right",
            va="center",
            fontsize=7.4,
            color=TEXT,
        )
        if disease == "HFpEF":
            sensitivity = float(row["sensitivity_min15_prioritization_score"])
            axis.plot(
                [score, sensitivity],
                [position, position],
                color="#9A8A70",
                linewidth=0.8,
                alpha=0.7,
            )
            axis.scatter(
                sensitivity,
                position,
                s=50,
                marker="D",
                facecolor="white",
                edgecolor=EXPLORATORY,
                linewidth=1.0,
                zorder=4,
            )

    axis.set_yticks(y, subset["module"])
    axis.set_ylim(-0.65, len(subset) - 0.35)
    style_module_axis(axis, show_xlabel)
    config = RUNS[disease]
    axis.set_title(
        disease,
        loc="left",
        fontweight="bold",
        color=TEXT,
        pad=11,
    )
    axis.text(
        0,
        1.015,
        f"{config['dataset']} · {config['tier']}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=SUBTEXT,
    )
    if disease == "HFpEF":
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=EXPLORATORY, markeredgecolor="white", label="min10"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=EXPLORATORY, label="min15"),
        ]
        axis.legend(
            handles=handles,
            loc="lower right",
            frameon=False,
            fontsize=7.0,
            ncol=2,
            handletextpad=0.3,
            columnspacing=0.8,
        )


def add_node(
    axis: plt.Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    label: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float,
    italic: bool = False,
) -> None:
    x, y = center
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.85,
        zorder=4,
    )
    axis.add_patch(box)
    axis.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="semibold",
        fontstyle="italic" if italic else "normal",
        color=TEXT,
        zorder=5,
        linespacing=1.02,
    )


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    linewidth: float,
    color: str,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=linewidth,
            color=color,
            alpha=0.70,
            shrinkA=1.2,
            shrinkB=1.2,
            zorder=2,
        )
    )


def draw_network_panel(axis: plt.Axes, disease: str, data: pd.DataFrame) -> None:
    subset = data.loc[data["disease"].eq(disease)].sort_values("module_order")
    n_rows = len(subset)
    y_positions = np.linspace(0.78, 0.18, n_rows)
    sender_x, ligand_x, receptor_x = 0.15, 0.51, 0.83

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(
        Rectangle(
            (0.735, 0.095),
            0.205,
            0.78,
            facecolor="#F0F6F9",
            edgecolor="#A9C1CF",
            linewidth=0.8,
            zorder=0,
        )
    )

    for y, (_, row) in zip(y_positions, subset.iterrows(), strict=True):
        score = float(row["prioritization_score"])
        width = 0.9 + 2.5 * (score - 0.4) / 0.6
        add_arrow(axis, (0.265, y), (0.422, y), width, EDGE)
        add_arrow(axis, (0.595, y), (0.744, y), width, EXPLORATORY if disease == "HFpEF" else PRIMARY)
        add_node(
            axis,
            (sender_x, y),
            0.23,
            0.085,
            SENDER_LABELS[str(row["sender"])],
            SENDER_COLORS[str(row["sender"])],
            "#667179",
            7.5,
        )
        add_node(
            axis,
            (ligand_x, y),
            0.155,
            0.074,
            str(row["ligand"]),
            "#F4DFC1",
            "#8A6841",
            8.0,
            italic=True,
        )
        receptor_label = str(row["receptor"])
        if disease == "HFpEF" and row["ligand"] == "NODAL":
            receptor_label = "ACVR2B\n(min15: ACVR2A)"
        add_node(
            axis,
            (receptor_x, y),
            0.16,
            0.078,
            receptor_label,
            "#DCEAF3",
            "#3D6A82",
            7.7 if "\n" not in receptor_label else 6.6,
            italic=True,
        )
        axis.text(
            0.955,
            y,
            f"{score:.2f}",
            ha="left",
            va="center",
            fontsize=7.2,
            color=SUBTEXT,
        )

    axis.text(sender_x, 0.94, "Sender", ha="center", va="top", fontweight="bold")
    axis.text(ligand_x, 0.94, "Ligand", ha="center", va="top", fontweight="bold")
    axis.text(receptor_x, 0.94, "LEC receptor", ha="center", va="top", fontweight="bold")

    config = RUNS[disease]
    panel_letter = PANEL_LETTERS[disease]
    axis.text(-0.055, 1.03, panel_letter, transform=axis.transAxes, fontsize=16, fontweight="bold", color=TEXT)
    axis.text(
        0,
        1.03,
        disease,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.2,
        fontweight="bold",
        color=TEXT,
    )
    if disease == "HFpEF":
        subtitle = "D11 · exploratory min10; min15 sensitivity shown for NODAL receptor shift"
    else:
        subtitle = (
            f"{config['dataset']} · {config['expected_reference_donors']} NF / "
            f"{config['expected_disease_donors']} {disease} donors · ≥25 LECs/donor"
        )
    axis.text(
        0,
        0.985,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=SUBTEXT,
    )


def write_manifest(validation: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for disease, config in RUNS.items():
        rows.append(
            {
                "figure_panels": "D",
                "input_csv": group_csv(str(config["folder"])).relative_to(ROOT).as_posix(),
                "used_fields": (
                    "contrast;group;sender;receiver;ligand;receptor;prioritization_score;"
                    "max_scaled_activity;scaled_pb_ligand;scaled_pb_receptor;"
                    "fraction_expressing_ligand_receptor;top_group"
                ),
                "use": f"exact disease-prioritized representative axes for {disease}",
            }
        )
        rows.append(
            {
                "figure_panels": "D provenance",
                "input_csv": summary_csv(str(config["folder"])).relative_to(ROOT).as_posix(),
                "used_fields": (
                    "analysis_id;disease;n_reference_donors;n_disease_donors;"
                    "validation_status;multinichenet_completed;status"
                ),
                "use": f"validate completed {disease} run and donor counts",
            }
        )
    hfpef_config = RUNS["HFpEF"]
    sensitivity_folder = str(hfpef_config["sensitivity_folder"])
    rows.extend(
        [
            {
                "figure_panels": "D sensitivity",
                "input_csv": group_csv(sensitivity_folder).relative_to(ROOT).as_posix(),
                "used_fields": "sender;receiver;ligand;receptor;prioritization_score;group;top_group",
                "use": "HFpEF min15 exact-axis or ligand-module sensitivity support",
            },
            {
                "figure_panels": "D sensitivity provenance",
                "input_csv": summary_csv(sensitivity_folder).relative_to(ROOT).as_posix(),
                "used_fields": (
                    "analysis_id;disease;receiver_min_cells;n_reference_donors;"
                    "n_disease_donors;validation_status;status"
                ),
                "use": "validate completed HFpEF min15 sensitivity run",
            },
            {
                "figure_panels": "D design",
                "input_csv": "Figure 5D communication-module specification",
                "used_fields": "disease modules; requested 3-5 representative signaling axes",
                "use": "focused module selection and D-to-H panel structure",
            },
        ]
    )
    pd.DataFrame(rows).to_csv(MANIFEST_CSV, index=False)


def main() -> None:
    selected, validation = load_selected_axes()
    selected.to_csv(SOURCE_CSV, index=False)
    write_manifest(validation)
    figure = plt.figure(figsize=(14.5, 8.8), facecolor="white")
    outer = GridSpec(
        1,
        1,
        figure=figure,
        # Reserve enough space for the longest module names in the left column.
        # The previous 0.09 margin clipped labels such as
        # "VEGF–neuropilin maintenance" and "BMP/activin injury response".
        left=0.14,
        right=0.975,
        top=0.76,
        bottom=0.15,
    )
    modules_grid = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=outer[0, 0],
        hspace=0.48,
        wspace=0.34,
    )
    for index, disease in enumerate(["DCM", "HCM", "DoxCM", "HFpEF"]):
        axis = figure.add_subplot(modules_grid[index // 2, index % 2])
        draw_module_facet(axis, disease, selected, show_xlabel=index >= 2)
    figure.text(0.025, 0.96, "D", fontsize=18, fontweight="bold", color=TEXT, va="top")
    figure.text(
        0.075,
        0.96,
        "Dominant extracellular communication modules by disease",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.075,
        0.90,
        "Point position is the within-run MultiNicheNet priority of the displayed representative axis; small cross-run differences are not effect sizes.",
        fontsize=8.8,
        color=SUBTEXT,
        va="top",
    )

    figure.text(
        0.075,
        0.030,
        (
            "DCM, HCM, and DoxCM use completed within-dataset donor-pseudobulk runs with ≥25 final LECs per donor. HFpEF is exploratory: "
            "filled circles/rows use min10 and open diamonds summarize min15 sensitivity.\n"
            "The NODAL module persists, but its prioritized receptor shifts from ACVR2B to ACVR2A. DoxCM supports BGN–TLR1 and HMGB1–HAVCR2, "
            "not an HMGB1–TLR edge.\nPriority scores rank hypotheses within each run; "
            "they do not establish physical binding or causality. Full ligand–receptor and ligand–target networks remain supplementary."
        ),
        ha="left",
        va="bottom",
        fontsize=8.1,
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

    print(f"Figure 5D written to: {OUTPUT_DIR}")
    for disease in ["DCM", "HCM", "DoxCM", "HFpEF"]:
        print(f"{disease} representative axes: {(selected['disease'] == disease).sum()}")
    print(f"Total representative axes: {len(selected)}")


if __name__ == "__main__":
    main()
