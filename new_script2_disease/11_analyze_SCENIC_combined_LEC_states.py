#!/usr/bin/env python3
r"""
Step 11: analyze pySCENIC regulon activity across combined disease-pipeline LEC states.

This script aligns the pySCENIC AUCell matrix to all current Step 08 disease
LECs and produces complementary state-specificity summaries:

1. Raw cell-level RSS (retained only as a group-size-sensitive reference).
2. Equal-cell balanced RSS: the same number of cells is sampled per state,
   standard RSS is recalculated repeatedly, and the median, 95% interval, and
   top-regulon frequency are reported.
3. Donor-balanced summaries: AUCell is first averaged within donor x state.
   Each donor then has equal weight in state means/z-scores. A donor-bootstrap
   RSS uses equal numbers of donor-state observations per state and reports the
   median, 95% interval, and top-regulon frequency.
4. Dataset-and-donor-balanced summaries: datasets present in every state are
   given equal weight, equal numbers of donor-state means are resampled within
   each state x dataset, and RSS/stability intervals are reported. This is the
   primary disease-pipeline result.

The recommended interpretation uses agreement among dataset/donor-bootstrap
RSS, dataset-balanced z-score, and resampling stability, with the all-dataset
donor analysis retained as a sensitivity result. Z-scores standardize each
regulon across states; they do not compare absolute activity between regulons.

Default inputs
--------------
new_result2_disease/08_combined_LEC_recluster/
    08_combined_final_LEC_reclustered.h5ad
new_result2_disease/10_SCENIC/pyscenic_output/
    auc_mtx.csv
    regulons.csv

Default output
--------------
new_result2_disease/11_SCENIC_analysis/

Run after Step 10B
------------------
    .\.venv\Scripts\python.exe `
      .\new_script2_disease\11_analyze_SCENIC_combined_LEC_states.py
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg", force=True)


SCRIPT_VERSION = "11_disease_combined_LEC_SCENIC_dataset_donor_balanced_2026_08_11"
DEFAULT_H5AD = (
    "new_result2_disease/08_combined_LEC_recluster/"
    "08_combined_final_LEC_reclustered.h5ad"
)
DEFAULT_PALETTE = (
    "new_result2_disease/08_combined_LEC_recluster/"
    "08_state_color_mapping.csv"
)
DEFAULT_PYSCENIC = "new_result2_disease/10_SCENIC/pyscenic_output"
DEFAULT_AUC = f"{DEFAULT_PYSCENIC}/auc_mtx.csv"
DEFAULT_REGULONS = f"{DEFAULT_PYSCENIC}/regulons.csv"
DEFAULT_OUTPUT = "new_result2_disease/11_SCENIC_analysis"
DEFAULT_IDENTITY_QC = (
    "new_result2_disease/09_LEC_state_GSEA/09_state_identity_marker_QC.csv"
)
EXPECTED_DATASETS = {"dataset01", "dataset08", "dataset11"}
EXPECTED_CONDITIONS = {"DCM", "DOXCM", "HCM", "HFPEF", "NF"}
ALLOWED_REGIONS = {
    "dataset01": {"AP", "LV"},
    "dataset08": {"AP", "LV"},
    "dataset11": {"S", "SEPTUM"},
}


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create raw, equal-cell balanced, and donor-balanced pySCENIC "
            "regulon summaries for combined disease-pipeline LEC states."
        )
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--auc_csv", default=DEFAULT_AUC)
    parser.add_argument("--regulons_csv", default=DEFAULT_REGULONS)
    parser.add_argument("--state_palette_csv", default=DEFAULT_PALETTE)
    parser.add_argument("--identity_qc_csv", default=DEFAULT_IDENTITY_QC)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--state_key", default="LEC_state")
    parser.add_argument("--donor_key", default="donor_key")
    parser.add_argument("--dataset_key", default="dataset_folder")
    parser.add_argument(
        "--equal_cell_repetitions",
        type=int,
        default=200,
        help="Repeated equal-cell standard RSS calculations.",
    )
    parser.add_argument(
        "--cells_per_state",
        type=int,
        default=0,
        help="0 automatically uses the smallest state in the current input.",
    )
    parser.add_argument(
        "--donor_bootstrap_repetitions",
        type=int,
        default=500,
        help="Equal-donor bootstrap repetitions for donor-level RSS and CIs.",
    )
    parser.add_argument(
        "--donors_per_state",
        type=int,
        default=0,
        help="0 uses the smallest donor count across states in the current input.",
    )
    parser.add_argument(
        "--dataset_donor_bootstrap_repetitions",
        type=int,
        default=500,
        help="Repeated RSS with equal donor sampling inside each common dataset/state.",
    )
    parser.add_argument(
        "--donors_per_state_dataset",
        type=int,
        default=0,
        help="0 uses the smallest donor count across common state x dataset groups.",
    )
    parser.add_argument(
        "--common_datasets",
        nargs="*",
        default=None,
        help=(
            "Datasets required in every state for strict balancing. Default: "
            "automatic intersection across all states."
        ),
    )
    parser.add_argument("--top_regulons_per_state", type=int, default=8)
    parser.add_argument("--figure_regulons_per_state", type=int, default=4)
    parser.add_argument("--max_regulons_in_summary_figure", type=int, default=40)
    parser.add_argument("--umap_regulons", type=int, default=7)
    parser.add_argument("--active_quantile", type=float, default=0.90)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument(
        "--write_h5ad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a new h5ad with aligned AUCell values in obsm.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing primary state-regulon summary.",
    )
    return parser.parse_args()


def natural_key(value: object) -> tuple:
    pieces = re.split(r"(\d+(?:\.\d+)?)", str(value))
    result = []
    for piece in pieces:
        if not piece:
            continue
        try:
            result.append((0, float(piece)))
        except ValueError:
            result.append((1, piece.lower()))
    return tuple(result)


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def clean_regulon_name(value: object) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"Regulon\((.+)\)", text)
    return match.group(1) if match else text


def tf_from_regulon(value: object) -> str:
    return re.sub(r"\([+-]\)$", "", clean_regulon_name(value)).strip()


def regulon_sign(value: object) -> str:
    match = re.search(r"\(([+-])\)$", clean_regulon_name(value))
    return match.group(1) if match else ""


def matrix_to_long(matrix: pd.DataFrame, value_name: str) -> pd.DataFrame:
    frame = matrix.copy()
    frame.index.name = "LEC_state"
    return frame.reset_index().melt(
        id_vars="LEC_state", var_name="regulon", value_name=value_name
    )


def zscore_regulons_across_states(matrix: pd.DataFrame) -> pd.DataFrame:
    means = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)
    z = matrix.subtract(means, axis=1).divide(std.replace(0, np.nan), axis=1)
    return z.fillna(0.0)


def load_palette(path: Path, states: list[str]) -> dict[str, str]:
    fallback = [
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    ]
    palette: dict[str, str] = {}
    if path.exists():
        frame = pd.read_csv(path)
        if {"LEC_state", "color_hex"}.issubset(frame.columns):
            palette = dict(zip(frame["LEC_state"].astype(str), frame["color_hex"]))
    for index, state in enumerate(states):
        palette.setdefault(state, fallback[index % len(fallback)])
    return palette


def read_auc_matrix(path: Path, expected_cells: pd.Index) -> tuple[pd.DataFrame, dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"AUCell matrix not found: {path}")
    raw = pd.read_csv(path, index_col=0)
    raw.index = raw.index.astype(str).str.strip()
    raw.columns = pd.Index([clean_regulon_name(value) for value in raw.columns])
    expected = pd.Index(expected_cells.astype(str))
    row_overlap = int(raw.index.isin(expected).sum())
    column_overlap = int(pd.Index(raw.columns.astype(str)).isin(expected).sum())
    if column_overlap > row_overlap:
        raw = raw.T
        raw.index = raw.index.astype(str).str.strip()
        raw.columns = pd.Index([clean_regulon_name(value) for value in raw.columns])
        row_overlap = int(raw.index.isin(expected).sum())
    if raw.index.duplicated().any():
        raise ValueError("AUCell matrix contains duplicate cell IDs.")
    if raw.columns.duplicated().any():
        raw = raw.T.groupby(level=0, sort=False).mean().T
    missing = expected.difference(raw.index)
    extra = raw.index.difference(expected)
    if len(missing):
        raise ValueError(
            f"AUCell is missing {len(missing)} Step 08 cells. "
            f"Examples: {missing[:5].tolist()}"
        )
    auc = raw.reindex(expected).apply(pd.to_numeric, errors="coerce")
    if auc.isna().any().any():
        raise ValueError("AUCell matrix contains nonnumeric or missing values.")
    if (auc.to_numpy() < 0).any():
        raise ValueError("AUCell matrix contains negative values.")
    nonconstant = auc.std(axis=0, ddof=0).gt(0)
    dropped_constant = int((~nonconstant).sum())
    if dropped_constant:
        warnings.warn(f"Dropping {dropped_constant} constant regulons from state analysis.")
        auc = auc.loc[:, nonconstant]
    report = {
        "n_expected_cells": int(len(expected)),
        "n_auc_cells_before_alignment": int(len(raw)),
        "n_extra_auc_cells": int(len(extra)),
        "n_aligned_cells": int(len(auc)),
        "n_regulons_nonconstant": int(auc.shape[1]),
        "n_regulons_constant_dropped": dropped_constant,
    }
    return auc, report


def parse_target_genes(value: object) -> list[tuple[str, float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        result = []
        if isinstance(parsed, (list, tuple)):
            for entry in parsed:
                if isinstance(entry, (list, tuple)) and len(entry) >= 1:
                    gene = str(entry[0]).strip()
                    weight = float(entry[1]) if len(entry) > 1 else np.nan
                    if gene:
                        result.append((gene, weight))
                elif isinstance(entry, str) and entry.strip():
                    result.append((entry.strip(), np.nan))
        if result:
            return result
    except Exception:
        pass
    matches = re.findall(
        r"['\"]([^'\"]+)['\"]\s*,\s*([-+0-9.eE]+)", text
    )
    return [(gene, float(weight)) for gene, weight in matches]


def parse_regulon_targets(
    path: Path, auc_regulons: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Regulon motif table not found: {path}")
    table = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    target_column = next(
        (column for column in table.columns if str(column[-1]).strip() == "TargetGenes"),
        None,
    )
    context_column = next(
        (column for column in table.columns if str(column[-1]).strip() == "Context"),
        None,
    )
    if target_column is None:
        raise KeyError(
            f"Could not locate TargetGenes in pySCENIC regulons table: {path}"
        )

    records = []
    for index, row in table.iterrows():
        tf = str(index[0]).strip()
        motif = str(index[1]).strip()
        context = str(row[context_column]) if context_column is not None else ""
        context_lower = context.lower()
        sign = "-" if "repress" in context_lower or "negative" in context_lower else "+"
        regulon = f"{tf}({sign})"
        for target, weight in parse_target_genes(row[target_column]):
            records.append(
                {
                    "regulon": regulon,
                    "TF": tf,
                    "sign": sign,
                    "target_gene": target,
                    "target_weight": weight,
                    "motif_id": motif,
                    "context": context,
                }
            )
    detailed = pd.DataFrame(records)
    if detailed.empty:
        raise ValueError(f"No target genes could be parsed from {path}")
    aggregated = (
        detailed.groupby(["regulon", "TF", "sign", "target_gene"], observed=True)
        .agg(
            target_weight_max=("target_weight", "max"),
            n_motif_records=("motif_id", "nunique"),
            motif_ids=("motif_id", lambda values: ";".join(sorted(set(map(str, values))))),
            contexts=("context", lambda values: ";".join(sorted(set(map(str, values))))),
        )
        .reset_index()
    )
    summary = (
        aggregated.groupby(["regulon", "TF", "sign"], observed=True)
        .agg(
            n_target_genes=("target_gene", "nunique"),
            target_genes=("target_gene", lambda values: ";".join(sorted(set(values)))),
            n_motifs=("motif_ids", lambda values: len(set(";".join(values).split(";")))),
        )
        .reset_index()
    )
    expected = set(map(clean_regulon_name, auc_regulons))
    parsed = set(summary["regulon"])
    coverage = len(expected.intersection(parsed)) / max(len(expected), 1)
    if coverage < 0.80:
        raise ValueError(
            f"Only {coverage:.1%} of AUCell regulons matched parsed target regulons. "
            "The regulons.csv structure may be incompatible."
        )
    return aggregated, summary


def state_support_table(
    obs: pd.DataFrame,
    state_key: str,
    donor_key: str,
    dataset_key: str,
    states: list[str],
) -> pd.DataFrame:
    rows = []
    for state in states:
        group = obs.loc[normalized_text(obs[state_key]).eq(state)]
        donor_counts = normalized_text(group[donor_key]).value_counts()
        dataset_counts = normalized_text(group[dataset_key]).value_counts()
        n_cells = int(len(group))
        top_fraction = float(donor_counts.iloc[0] / n_cells)
        n_donors = int(len(donor_counts))
        flags = []
        if top_fraction > 0.5:
            flags.append("CAUTION_single_donor_over_50pct")
        if n_donors < 10:
            flags.append("CAUTION_fewer_than_10_donors")
        rows.append(
            {
                "LEC_state": state,
                "n_cells": n_cells,
                "n_donors": n_donors,
                "n_datasets": (
                    int(group[dataset_key].nunique()) if dataset_key in group else np.nan
                ),
                "top_donor": donor_counts.index[0],
                "top_donor_n_cells": int(donor_counts.iloc[0]),
                "top_donor_fraction": top_fraction,
                "top_dataset": dataset_counts.index[0],
                "top_dataset_n_cells": int(dataset_counts.iloc[0]),
                "top_dataset_fraction": float(dataset_counts.iloc[0] / n_cells),
                "donor_dominance_flag_over_50pct": top_fraction > 0.5,
                "low_donor_support_flag_under_10": n_donors < 10,
                "interpretation_flag": ";".join(flags) if flags else "OK",
            }
        )
    return pd.DataFrame(rows)


def calculate_rss_array(
    auc_values: np.ndarray, labels: np.ndarray, states: Sequence[str], chunk: int = 64
) -> np.ndarray:
    values = np.clip(np.asarray(auc_values, dtype=np.float64), 0, None)
    sums = values.sum(axis=0)
    valid = sums > 0
    probabilities = np.zeros_like(values)
    probabilities[:, valid] = values[:, valid] / sums[valid]
    result = np.full((len(states), values.shape[1]), np.nan, dtype=float)
    label_values = np.asarray(labels).astype(str)
    for state_index, state in enumerate(states):
        membership = label_values == str(state)
        n_group = int(membership.sum())
        if n_group == 0:
            continue
        q = membership.astype(np.float64) / n_group
        for start in range(0, values.shape[1], chunk):
            end = min(start + chunk, values.shape[1])
            p = probabilities[:, start:end]
            midpoint = 0.5 * (p + q[:, None])
            with np.errstate(divide="ignore", invalid="ignore"):
                kl_p = np.where(p > 0, p * np.log2(p / midpoint), 0.0)
                q_col = q[:, None]
                kl_q = np.where(q_col > 0, q_col * np.log2(q_col / midpoint), 0.0)
            js_divergence = 0.5 * (
                np.nansum(kl_p, axis=0) + np.nansum(kl_q, axis=0)
            )
            result[state_index, start:end] = 1.0 - np.sqrt(
                np.clip(js_divergence, 0.0, 1.0)
            )
    return result


def summarize_repeated_rss(
    arrays: list[np.ndarray],
    top_hits: np.ndarray,
    states: list[str],
    regulons: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stack = np.stack(arrays, axis=0)
    median = pd.DataFrame(np.nanmedian(stack, axis=0), index=states, columns=regulons)
    low = pd.DataFrame(np.nanpercentile(stack, 2.5, axis=0), index=states, columns=regulons)
    high = pd.DataFrame(np.nanpercentile(stack, 97.5, axis=0), index=states, columns=regulons)
    frequency = pd.DataFrame(top_hits / len(arrays), index=states, columns=regulons)
    return median, low, high, frequency


def repeated_equal_cell_rss(
    auc: pd.DataFrame,
    labels: pd.Series,
    states: list[str],
    repetitions: int,
    requested_target: int,
    top_n: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    label_values = labels.reindex(auc.index).astype(str).to_numpy()
    counts = {state: int(np.sum(label_values == state)) for state in states}
    target = min(counts.values()) if requested_target <= 0 else min(requested_target, min(counts.values()))
    if target < 10:
        raise ValueError(f"Equal-cell RSS target is too small: {target}")
    rng = np.random.default_rng(seed)
    arrays = []
    top_hits = np.zeros((len(states), auc.shape[1]), dtype=np.int64)
    values = auc.to_numpy(dtype=float)
    for repetition in range(repetitions):
        positions = np.concatenate(
            [
                rng.choice(np.where(label_values == state)[0], size=target, replace=False)
                for state in states
            ]
        )
        sampled_labels = label_values[positions]
        rss = calculate_rss_array(values[positions], sampled_labels, states)
        arrays.append(rss)
        for state_index in range(len(states)):
            best = np.argsort(rss[state_index])[-min(top_n, auc.shape[1]):]
            top_hits[state_index, best] += 1
    return (*summarize_repeated_rss(arrays, top_hits, states, auc.columns), target)


def donor_state_activity(
    auc: pd.DataFrame,
    obs: pd.DataFrame,
    state_key: str,
    donor_key: str,
    dataset_key: str,
) -> pd.DataFrame:
    frame = auc.copy()
    frame.insert(0, "LEC_state", normalized_text(obs.loc[auc.index, state_key]).to_numpy())
    frame.insert(0, "donor_key", normalized_text(obs.loc[auc.index, donor_key]).to_numpy())
    frame.insert(0, "dataset_folder", normalized_text(obs.loc[auc.index, dataset_key]).to_numpy())
    means = (
        frame.groupby(
            ["dataset_folder", "donor_key", "LEC_state"],
            observed=True,
            sort=False,
        )
        .mean(numeric_only=True)
        .reset_index()
    )
    counts = (
        frame.groupby(
            ["dataset_folder", "donor_key", "LEC_state"],
            observed=True,
            sort=False,
        )
        .size()
        .rename("n_cells_in_donor_state")
        .reset_index()
    )
    return means.merge(
        counts,
        on=["dataset_folder", "donor_key", "LEC_state"],
        how="left",
        validate="one_to_one",
    )


def donor_balanced_means_and_ci(
    donor_state: pd.DataFrame,
    states: list[str],
    regulons: pd.Index,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    means = []
    lows = []
    highs = []
    for state in states:
        values = donor_state.loc[
            donor_state["LEC_state"].astype(str).eq(state), regulons
        ].to_numpy(dtype=float)
        if len(values) == 0:
            raise ValueError(f"No donor-state observations for {state}")
        mean = values.mean(axis=0)
        draw_indices = rng.integers(0, len(values), size=(repetitions, len(values)))
        bootstrap_means = values[draw_indices].mean(axis=1)
        means.append(mean)
        lows.append(np.percentile(bootstrap_means, 2.5, axis=0))
        highs.append(np.percentile(bootstrap_means, 97.5, axis=0))
    return (
        pd.DataFrame(means, index=states, columns=regulons),
        pd.DataFrame(lows, index=states, columns=regulons),
        pd.DataFrame(highs, index=states, columns=regulons),
    )


def donor_bootstrap_rss(
    donor_state: pd.DataFrame,
    states: list[str],
    regulons: pd.Index,
    repetitions: int,
    requested_target: int,
    top_n: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    labels = donor_state["LEC_state"].astype(str).to_numpy()
    values = donor_state.loc[:, regulons].to_numpy(dtype=float)
    counts = {state: int(np.sum(labels == state)) for state in states}
    target = min(counts.values()) if requested_target <= 0 else min(requested_target, min(counts.values()))
    if target < 5:
        raise ValueError(f"Donor-bootstrap RSS target is too small: {target}")
    rng = np.random.default_rng(seed)
    arrays = []
    top_hits = np.zeros((len(states), len(regulons)), dtype=np.int64)
    for repetition in range(repetitions):
        positions = np.concatenate(
            [
                rng.choice(np.where(labels == state)[0], size=target, replace=True)
                for state in states
            ]
        )
        sampled_labels = labels[positions]
        rss = calculate_rss_array(values[positions], sampled_labels, states)
        arrays.append(rss)
        for state_index in range(len(states)):
            best = np.argsort(rss[state_index])[-min(top_n, len(regulons)):]
            top_hits[state_index, best] += 1
    return (*summarize_repeated_rss(arrays, top_hits, states, regulons), target)


def state_dataset_support_table(
    obs: pd.DataFrame,
    state_key: str,
    donor_key: str,
    dataset_key: str,
    states: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "LEC_state": normalized_text(obs[state_key]),
            "dataset_folder": normalized_text(obs[dataset_key]),
            "donor_key": normalized_text(obs[donor_key]),
        },
        index=obs.index,
    )
    support = (
        frame.groupby(["LEC_state", "dataset_folder"], observed=True)
        .agg(n_cells=("donor_key", "size"), n_donors=("donor_key", "nunique"))
        .reset_index()
    )
    full = pd.MultiIndex.from_product(
        [states, sorted(frame["dataset_folder"].unique())],
        names=["LEC_state", "dataset_folder"],
    ).to_frame(index=False)
    return full.merge(
        support,
        on=["LEC_state", "dataset_folder"],
        how="left",
    ).fillna({"n_cells": 0, "n_donors": 0})


def select_common_datasets(
    dataset_support: pd.DataFrame,
    states: list[str],
    requested: Optional[list[str]],
) -> list[str]:
    available = {
        state: set(
            dataset_support.loc[
                dataset_support["LEC_state"].eq(state)
                & dataset_support["n_cells"].gt(0),
                "dataset_folder",
            ].astype(str)
        )
        for state in states
    }
    automatic = sorted(set.intersection(*(available[state] for state in states)))
    if requested:
        selected = sorted(dict.fromkeys(str(value).strip() for value in requested))
        missing = {
            state: sorted(set(selected).difference(available[state]))
            for state in states
            if set(selected).difference(available[state])
        }
        if missing:
            raise ValueError(
                "Requested common datasets are absent from some states: "
                + json.dumps(missing, sort_keys=True)
            )
    else:
        selected = automatic
    if len(selected) < 2:
        raise ValueError(
            f"Strict dataset balance requires at least two common datasets; found {selected}."
        )
    return selected


def dataset_balanced_donor_mean(
    donor_state: pd.DataFrame,
    states: list[str],
    datasets: list[str],
    regulons: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = donor_state.loc[
        donor_state["dataset_folder"].astype(str).isin(datasets)
    ].copy()
    state_dataset = (
        selected.groupby(
            ["LEC_state", "dataset_folder"], observed=True, sort=False
        )[list(regulons)]
        .mean()
        .reset_index()
    )
    expected = len(states) * len(datasets)
    if len(state_dataset) != expected:
        raise ValueError(
            f"Expected {expected} state-dataset means, found {len(state_dataset)}."
        )
    mean = (
        state_dataset.groupby("LEC_state", observed=True)[list(regulons)]
        .mean()
        .reindex(states)
    )
    return mean, state_dataset


def dataset_donor_bootstrap(
    donor_state: pd.DataFrame,
    states: list[str],
    datasets: list[str],
    regulons: pd.Index,
    repetitions: int,
    requested_target: int,
    top_n: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    int,
]:
    selected = donor_state.loc[
        donor_state["dataset_folder"].astype(str).isin(datasets)
    ].copy()
    counts = selected.groupby(
        ["LEC_state", "dataset_folder"], observed=True
    ).size()
    minimum = int(counts.min())
    target = minimum if requested_target <= 0 else min(requested_target, minimum)
    if target < 3:
        raise ValueError(f"Dataset/donor bootstrap target is too small: {target}")
    rng = np.random.default_rng(seed)
    rss_arrays: list[np.ndarray] = []
    mean_arrays: list[np.ndarray] = []
    top_hits = np.zeros((len(states), len(regulons)), dtype=np.int64)
    for _ in range(repetitions):
        sampled_values = []
        sampled_labels = []
        state_means = []
        for state in states:
            state_blocks = []
            for dataset in datasets:
                block = selected.loc[
                    selected["LEC_state"].astype(str).eq(state)
                    & selected["dataset_folder"].astype(str).eq(dataset),
                    regulons,
                ].to_numpy(dtype=float)
                chosen = rng.integers(0, len(block), size=target)
                sampled = block[chosen]
                state_blocks.append(sampled)
                sampled_values.append(sampled)
                sampled_labels.extend([state] * target)
            state_means.append(np.vstack(state_blocks).mean(axis=0))
        values = np.vstack(sampled_values)
        labels = np.asarray(sampled_labels)
        rss = calculate_rss_array(values, labels, states)
        rss_arrays.append(rss)
        mean_arrays.append(np.vstack(state_means))
        for state_index in range(len(states)):
            best = np.argsort(rss[state_index])[-min(top_n, len(regulons)):]
            top_hits[state_index, best] += 1
    rss_median, rss_low, rss_high, rss_frequency = summarize_repeated_rss(
        rss_arrays, top_hits, states, regulons
    )
    means = np.stack(mean_arrays, axis=0)
    mean_low = pd.DataFrame(
        np.nanpercentile(means, 2.5, axis=0), index=states, columns=regulons
    )
    mean_high = pd.DataFrame(
        np.nanpercentile(means, 97.5, axis=0), index=states, columns=regulons
    )
    return (
        mean_low,
        mean_high,
        rss_median,
        rss_low,
        rss_high,
        rss_frequency,
        target,
    )


def active_percent_by_state(
    auc: pd.DataFrame, labels: pd.Series, states: list[str], quantile: float
) -> tuple[pd.Series, pd.DataFrame]:
    thresholds = auc.quantile(quantile, axis=0)
    active = auc.ge(thresholds, axis=1)
    rows = []
    aligned_labels = labels.reindex(auc.index).astype(str)
    for state in states:
        rows.append(active.loc[aligned_labels.eq(state)].mean(axis=0) * 100.0)
    return thresholds, pd.DataFrame(rows, index=states, columns=auc.columns)


def make_consensus_table(
    states: list[str],
    support: pd.DataFrame,
    regulon_summary: pd.DataFrame,
    raw_rss: pd.DataFrame,
    cell_mean: pd.DataFrame,
    cell_z: pd.DataFrame,
    active_percent: pd.DataFrame,
    cell_rss: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    donor_mean: pd.DataFrame,
    donor_mean_low: pd.DataFrame,
    donor_mean_high: pd.DataFrame,
    donor_z: pd.DataFrame,
    donor_rss: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    dataset_mean: pd.DataFrame,
    dataset_mean_low: pd.DataFrame,
    dataset_mean_high: pd.DataFrame,
    dataset_z: pd.DataFrame,
    dataset_rss: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    palette: dict[str, str],
) -> pd.DataFrame:
    matrices = [
        (raw_rss, "raw_cell_RSS"),
        (cell_mean, "cell_mean_AUCell"),
        (cell_z, "cell_mean_AUCell_zscore"),
        (active_percent, "active_cell_percent_visualization_threshold"),
        (cell_rss[0], "equal_cell_RSS_median"),
        (cell_rss[1], "equal_cell_RSS_q025"),
        (cell_rss[2], "equal_cell_RSS_q975"),
        (cell_rss[3], "equal_cell_RSS_top_frequency"),
        (donor_mean, "donor_balanced_mean_AUCell"),
        (donor_mean_low, "donor_balanced_mean_AUCell_q025"),
        (donor_mean_high, "donor_balanced_mean_AUCell_q975"),
        (donor_z, "donor_balanced_mean_AUCell_zscore"),
        (donor_rss[0], "donor_bootstrap_RSS_median"),
        (donor_rss[1], "donor_bootstrap_RSS_q025"),
        (donor_rss[2], "donor_bootstrap_RSS_q975"),
        (donor_rss[3], "donor_bootstrap_RSS_top_frequency"),
        (dataset_mean, "dataset_donor_balanced_mean_AUCell"),
        (dataset_mean_low, "dataset_donor_balanced_mean_AUCell_q025"),
        (dataset_mean_high, "dataset_donor_balanced_mean_AUCell_q975"),
        (dataset_z, "dataset_donor_balanced_mean_AUCell_zscore"),
        (dataset_rss[0], "dataset_donor_bootstrap_RSS_median"),
        (dataset_rss[1], "dataset_donor_bootstrap_RSS_q025"),
        (dataset_rss[2], "dataset_donor_bootstrap_RSS_q975"),
        (dataset_rss[3], "dataset_donor_bootstrap_RSS_top_frequency"),
    ]
    result = matrix_to_long(*matrices[0])
    for matrix, name in matrices[1:]:
        result = result.merge(matrix_to_long(matrix, name), on=["LEC_state", "regulon"])
    result.insert(1, "state_color", result["LEC_state"].map(palette))
    result["TF"] = result["regulon"].map(tf_from_regulon)
    result["regulon_sign"] = result["regulon"].map(regulon_sign)
    result = result.merge(regulon_summary, on=["regulon", "TF", "regulon_sign"], how="left")
    support_columns = [
        "LEC_state", "n_cells", "n_donors", "n_datasets", "top_donor",
        "top_donor_fraction", "top_dataset", "top_dataset_fraction",
        "interpretation_flag", "identity_marker_review_flag",
        "identity_marker_note", "cardiomyocyte_marker_hits",
    ]
    support_columns = [column for column in support_columns if column in support]
    result = result.merge(support[support_columns], on="LEC_state", how="left")

    result["donor_RSS_percentile_within_state"] = result.groupby(
        "LEC_state", observed=True
    )["donor_bootstrap_RSS_median"].rank(pct=True, method="average")
    result["donor_zscore_percentile_within_state"] = result.groupby(
        "LEC_state", observed=True
    )["donor_balanced_mean_AUCell_zscore"].rank(pct=True, method="average")
    result["dataset_donor_RSS_percentile_within_state"] = result.groupby(
        "LEC_state", observed=True
    )["dataset_donor_bootstrap_RSS_median"].rank(pct=True, method="average")
    result["dataset_donor_zscore_percentile_within_state"] = result.groupby(
        "LEC_state", observed=True
    )["dataset_donor_balanced_mean_AUCell_zscore"].rank(
        pct=True, method="average"
    )
    result["consensus_specificity_score"] = result[
        [
            "dataset_donor_RSS_percentile_within_state",
            "dataset_donor_zscore_percentile_within_state",
            "dataset_donor_bootstrap_RSS_top_frequency",
        ]
    ].mean(axis=1)
    result["consensus_rank_within_state"] = result.groupby(
        "LEC_state", observed=True
    )["consensus_specificity_score"].rank(ascending=False, method="first").astype(int)
    result["robust_candidate_flag"] = (
        result["dataset_donor_balanced_mean_AUCell_zscore"].gt(0)
        & result["dataset_donor_RSS_percentile_within_state"].ge(0.90)
        & result["dataset_donor_bootstrap_RSS_top_frequency"].ge(0.25)
        & result["donor_balanced_mean_AUCell_zscore"].gt(0)
    )
    state_order = {state: index for index, state in enumerate(states)}
    result["_state_order"] = result["LEC_state"].map(state_order)
    return result.sort_values(
        ["_state_order", "consensus_rank_within_state"], kind="mergesort"
    ).drop(columns="_state_order")


def select_figure_regulons(
    consensus: pd.DataFrame, states: list[str], per_state: int, maximum: int
) -> list[str]:
    ranked = {
        state: consensus.loc[consensus["LEC_state"].eq(state)]
        .sort_values("consensus_rank_within_state")["regulon"]
        .tolist()
        for state in states
    }
    selected: list[str] = []
    for rank in range(per_state):
        for state in states:
            if rank < len(ranked[state]) and ranked[state][rank] not in selected:
                selected.append(ranked[state][rank])
    if len(selected) > maximum:
        best = (
            consensus.loc[consensus["regulon"].isin(selected)]
            .groupby("regulon", observed=True)["consensus_specificity_score"]
            .max()
            .sort_values(ascending=False)
        )
        selected = best.head(maximum).index.tolist()
    return selected


def style_state_ticks(ax, states: list[str], palette: dict[str, str], axis: str) -> None:
    ticks = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for tick, state in zip(ticks, states):
        tick.set_color(palette[state])
        tick.set_fontweight("bold")


def save_state_support_figure(
    support: pd.DataFrame, states: list[str], palette: dict[str, str], path: Path, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    frame = support.set_index("LEC_state").reindex(states)
    x = np.arange(len(states))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, column, title in [
        (axes[0], "n_cells", "LEC cells per state"),
        (axes[1], "n_donors", "Contributing donors per state"),
    ]:
        ax.bar(x, frame[column], color=[palette[state] for state in states], edgecolor="#333333")
        labels = [
            f"{state}*" if bool(frame.loc[state, "donor_dominance_flag_over_50pct"]) else state
            for state in states
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        style_state_ticks(ax, states, palette, "x")
        ax.set_title(title)
        ax.set_ylabel(column.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SCENIC state-support QC (* = one donor contributes >50% of cells)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(
    matrix: pd.DataFrame,
    states: list[str],
    regulons: list[str],
    palette: dict[str, str],
    path: Path,
    title: str,
    colorbar_label: str,
    dpi: int,
    center_zero: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    values = matrix.reindex(index=states, columns=regulons).to_numpy(dtype=float)
    width = max(13, 0.34 * len(regulons) + 5)
    fig, ax = plt.subplots(figsize=(width, 7.5))
    if center_zero:
        extent = max(float(np.nanmax(np.abs(values))), 0.1)
        image = ax.imshow(
            values, aspect="auto", cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent),
        )
    else:
        image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(regulons)))
    ax.set_xticklabels(regulons, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states)
    style_state_ticks(ax, states, palette, "y")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_consensus_dotplot(
    consensus: pd.DataFrame,
    states: list[str],
    regulons: list[str],
    palette: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    selected = consensus.loc[consensus["regulon"].isin(regulons)].copy()
    state_pos = {state: index for index, state in enumerate(states)}
    regulon_order = list(reversed(regulons))
    regulon_pos = {regulon: index for index, regulon in enumerate(regulon_order)}
    selected["x"] = selected["LEC_state"].map(state_pos)
    selected["y"] = selected["regulon"].map(regulon_pos)
    rss = selected["dataset_donor_bootstrap_RSS_median"].to_numpy(float)
    rss_min, rss_max = float(np.nanmin(rss)), float(np.nanmax(rss))
    scaled = (rss - rss_min) / (rss_max - rss_min) if rss_max > rss_min else np.ones_like(rss)
    sizes = 15 + 230 * scaled
    z = selected["dataset_donor_balanced_mean_AUCell_zscore"].to_numpy(float)
    extent = max(float(np.nanmax(np.abs(z))), 0.1)
    stable = selected["dataset_donor_bootstrap_RSS_top_frequency"].ge(0.50)
    fig, ax = plt.subplots(
        figsize=(max(13, 0.65 * len(states) + 5), max(8, 0.33 * len(regulons) + 3))
    )
    scatter = ax.scatter(
        selected["x"], selected["y"], s=sizes, c=z, cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent),
        edgecolors=np.where(stable, "#111111", "#bbbbbb"),
        linewidths=np.where(stable, 0.9, 0.3), alpha=0.9,
    )
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, rotation=45, ha="right")
    style_state_ticks(ax, states, palette, "x")
    ax.set_yticks(range(len(regulon_order)))
    ax.set_yticklabels(regulon_order, fontsize=8)
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)
    ax.set_title(
        "Dataset- and donor-balanced SCENIC consensus\n"
        "Color = balanced z-score; size = balanced bootstrap RSS; "
        "dark outline = top-regulon frequency ≥50%"
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    colorbar.set_label("Dataset/donor-balanced mean AUCell z-score")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_umap_figure(
    adata: ad.AnnData,
    auc: pd.DataFrame,
    states: list[str],
    palette: dict[str, str],
    regulons: list[str],
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    if "X_umap" not in adata.obsm:
        warnings.warn("X_umap is absent; skipping SCENIC UMAP summary.")
        return
    coords = np.asarray(adata.obsm["X_umap"])
    n_panels = 1 + len(regulons)
    ncols = 4
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    state_values = normalized_text(adata.obs["LEC_state"]).to_numpy()
    for state in states:
        mask = state_values == state
        axes[0].scatter(
            coords[mask, 0], coords[mask, 1], s=9, color=palette[state],
            alpha=0.85, linewidths=0, label=state,
        )
    axes[0].set_title("Combined LEC states")
    axes[0].legend(markerscale=1.8, fontsize=7, ncol=2, frameon=False)
    for ax, regulon in zip(axes[1:], regulons):
        values = auc[regulon].to_numpy(float)
        low, high = np.quantile(values, [0.01, 0.995])
        clipped = np.clip(values, low, high)
        points = ax.scatter(
            coords[:, 0], coords[:, 1], c=clipped, s=9, cmap="magma",
            linewidths=0, alpha=0.9,
        )
        ax.set_title(regulon)
        fig.colorbar(points, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes[n_panels:]:
        ax.axis("off")
    for ax in axes[:n_panels]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.spines[:].set_visible(False)
    fig.suptitle("SCENIC regulon activity on the Step 08 embedding", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if (
        args.equal_cell_repetitions < 10
        or args.donor_bootstrap_repetitions < 10
        or args.dataset_donor_bootstrap_repetitions < 10
    ):
        raise ValueError("Use at least 10 repetitions for each balanced analysis.")
    if not 0 < args.active_quantile < 1:
        raise ValueError("active_quantile must be between 0 and 1.")

    project_dir = find_project_dir()
    h5ad_path = resolve_path(args.h5ad, project_dir)
    auc_path = resolve_path(args.auc_csv, project_dir)
    regulons_path = resolve_path(args.regulons_csv, project_dir)
    palette_path = resolve_path(args.state_palette_csv, project_dir)
    identity_qc_path = resolve_path(args.identity_qc_csv, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    primary_output = tables_dir / "11_SCENIC_state_regulon_full_summary.csv"
    if primary_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Primary output already exists: {primary_output}\n"
            "Use --overwrite only when replacement is intended."
        )
    for label, path in {
        "Step 08 h5ad": h5ad_path,
        "AUCell CSV": auc_path,
        "regulons CSV": regulons_path,
    }.items():
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {label}: {path}")
    tables_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading Step 08 h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    step08_combine_info = dict(adata.uns.get("step08_combine_info", {}))
    step08_embedding_info = dict(adata.uns.get("combined_LEC_embedding", {}))
    step08_resolution = float(
        step08_embedding_info.get(
            "leiden_resolution",
            step08_combine_info.get("leiden_resolution", np.nan),
        )
    )
    adata.obs_names = adata.obs_names.astype(str)
    for column in [args.state_key, args.donor_key, args.dataset_key]:
        if column not in adata.obs:
            raise KeyError(f"Required obs column is absent: {column}")
    dataset_values = normalized_text(adata.obs[args.dataset_key]).str.lower()
    datasets_found = set(dataset_values.unique())
    if datasets_found != EXPECTED_DATASETS:
        raise ValueError(
            f"Expected datasets {sorted(EXPECTED_DATASETS)}, found {sorted(datasets_found)}."
        )
    condition_column = next(
        (column for column in ["condition", "condition_merged", "disease"] if column in adata.obs),
        None,
    )
    if condition_column is None:
        raise KeyError("No condition column is available for disease-pipeline validation.")
    conditions_found = set(
        normalized_text(adata.obs[condition_column]).str.upper().unique()
    )
    if conditions_found != EXPECTED_CONDITIONS:
        raise ValueError(
            f"Expected conditions {sorted(EXPECTED_CONDITIONS)}, "
            f"found {sorted(conditions_found)}."
        )
    if "region_standardized_analysis" not in adata.obs:
        raise KeyError("region_standardized_analysis is required for tissue validation.")
    region_values = normalized_text(
        adata.obs["region_standardized_analysis"]
    ).str.upper()
    for dataset in sorted(datasets_found):
        unexpected = set(
            region_values.loc[dataset_values.eq(dataset)].unique()
        ).difference(ALLOWED_REGIONS[dataset])
        if unexpected:
            raise ValueError(
                f"{dataset} has disallowed regions: {sorted(unexpected)}"
            )
    states = sorted(normalized_text(adata.obs[args.state_key]).unique(), key=natural_key)
    palette = load_palette(palette_path, states)
    support = state_support_table(
        adata.obs, args.state_key, args.donor_key, args.dataset_key, states
    )
    if identity_qc_path.exists():
        identity = pd.read_csv(identity_qc_path)
        identity_columns = [
            column for column in [
                "LEC_state", "identity_marker_review_flag",
                "identity_marker_note", "cardiomyocyte_marker_hits",
                "n_cardiomyocyte_marker_hits",
            ]
            if column in identity
        ]
        if "LEC_state" not in identity_columns:
            raise KeyError(f"LEC_state is absent from identity QC: {identity_qc_path}")
        support = support.merge(
            identity[identity_columns],
            on="LEC_state",
            how="left",
            validate="one_to_one",
        )
        if "identity_marker_review_flag" in support:
            identity_values = support["identity_marker_review_flag"]
            identity_flag = (
                identity_values.fillna(False).astype(bool)
                if pd.api.types.is_bool_dtype(identity_values)
                else normalized_text(identity_values).str.lower().isin(
                    {"true", "1", "yes", "y"}
                )
            )
            support["interpretation_flag"] = np.where(
                identity_flag,
                np.where(
                    support["interpretation_flag"].eq("OK"),
                    "REVIEW_cardiomyocyte_like_top_markers",
                    support["interpretation_flag"].astype(str)
                    + ";REVIEW_cardiomyocyte_like_top_markers",
                ),
                support["interpretation_flag"],
            )
    else:
        warnings.warn(
            f"Optional Step 09 identity QC is absent: {identity_qc_path}"
        )
    dataset_support = state_dataset_support_table(
        adata.obs, args.state_key, args.donor_key, args.dataset_key, states
    )
    common_datasets = select_common_datasets(
        dataset_support, states, args.common_datasets
    )
    dataset_support["used_for_strict_dataset_balance"] = dataset_support[
        "dataset_folder"
    ].isin(common_datasets)
    print(
        "Datasets shared by every state and used for strict balancing: "
        + ", ".join(common_datasets)
    )

    print(f"Reading AUCell matrix: {auc_path}")
    auc, alignment_report = read_auc_matrix(auc_path, adata.obs_names)
    print(f"Aligned AUCell: {auc.shape[0]:,} cells x {auc.shape[1]:,} regulons")
    targets, regulon_summary = parse_regulon_targets(regulons_path, auc.columns)
    regulon_summary = regulon_summary.rename(columns={"sign": "regulon_sign"})
    all_regulons = pd.DataFrame(
        {
            "regulon": auc.columns,
            "TF": [tf_from_regulon(value) for value in auc.columns],
            "regulon_sign": [regulon_sign(value) for value in auc.columns],
            "mean_AUCell_all_cells": auc.mean(axis=0).to_numpy(),
            "median_AUCell_all_cells": auc.median(axis=0).to_numpy(),
            "sd_AUCell_all_cells": auc.std(axis=0).to_numpy(),
        }
    ).merge(regulon_summary, on=["regulon", "TF", "regulon_sign"], how="left")

    labels = normalized_text(adata.obs.loc[auc.index, args.state_key])
    raw_rss = pd.DataFrame(
        calculate_rss_array(auc.to_numpy(), labels.to_numpy(), states),
        index=states,
        columns=auc.columns,
    )
    cell_mean = pd.DataFrame(
        [auc.loc[labels.eq(state)].mean(axis=0) for state in states],
        index=states,
        columns=auc.columns,
    )
    cell_z = zscore_regulons_across_states(cell_mean)
    thresholds, active_percent = active_percent_by_state(
        auc, labels, states, args.active_quantile
    )

    print(
        f"Calculating equal-cell RSS ({args.equal_cell_repetitions} repetitions)..."
    )
    cell_rss_median, cell_rss_low, cell_rss_high, cell_rss_frequency, cell_target = (
        repeated_equal_cell_rss(
            auc,
            labels,
            states,
            args.equal_cell_repetitions,
            args.cells_per_state,
            args.top_regulons_per_state,
            args.random_seed,
        )
    )

    donor_state = donor_state_activity(
        auc, adata.obs, args.state_key, args.donor_key, args.dataset_key
    )
    donor_mean, donor_mean_low, donor_mean_high = donor_balanced_means_and_ci(
        donor_state,
        states,
        auc.columns,
        args.donor_bootstrap_repetitions,
        args.random_seed + 1,
    )
    donor_z = zscore_regulons_across_states(donor_mean)
    print(
        f"Calculating donor-bootstrap RSS ({args.donor_bootstrap_repetitions} repetitions)..."
    )
    donor_rss_median, donor_rss_low, donor_rss_high, donor_rss_frequency, donor_target = (
        donor_bootstrap_rss(
            donor_state,
            states,
            auc.columns,
            args.donor_bootstrap_repetitions,
            args.donors_per_state,
            args.top_regulons_per_state,
            args.random_seed + 2,
        )
    )

    dataset_mean, state_dataset_mean = dataset_balanced_donor_mean(
        donor_state, states, common_datasets, auc.columns
    )
    dataset_z = zscore_regulons_across_states(dataset_mean)
    print(
        "Calculating dataset/donor-bootstrap RSS "
        f"({args.dataset_donor_bootstrap_repetitions} repetitions)..."
    )
    (
        dataset_mean_low,
        dataset_mean_high,
        dataset_rss_median,
        dataset_rss_low,
        dataset_rss_high,
        dataset_rss_frequency,
        dataset_donor_target,
    ) = dataset_donor_bootstrap(
        donor_state,
        states,
        common_datasets,
        auc.columns,
        args.dataset_donor_bootstrap_repetitions,
        args.donors_per_state_dataset,
        args.top_regulons_per_state,
        args.random_seed + 3,
    )

    consensus = make_consensus_table(
        states,
        support,
        all_regulons,
        raw_rss,
        cell_mean,
        cell_z,
        active_percent,
        (cell_rss_median, cell_rss_low, cell_rss_high, cell_rss_frequency),
        donor_mean,
        donor_mean_low,
        donor_mean_high,
        donor_z,
        (donor_rss_median, donor_rss_low, donor_rss_high, donor_rss_frequency),
        dataset_mean,
        dataset_mean_low,
        dataset_mean_high,
        dataset_z,
        (
            dataset_rss_median,
            dataset_rss_low,
            dataset_rss_high,
            dataset_rss_frequency,
        ),
        palette,
    )
    top_candidates = (
        consensus.sort_values(["LEC_state", "consensus_rank_within_state"])
        .groupby("LEC_state", sort=False, observed=True)
        .head(args.top_regulons_per_state)
        .copy()
    )

    auc.to_csv(tables_dir / "11_AUCell_aligned.csv.gz", compression="gzip")
    targets.to_csv(tables_dir / "11_regulon_targets_long.csv", index=False)
    all_regulons.to_csv(tables_dir / "11_regulon_summary.csv", index=False)
    support.to_csv(tables_dir / "11_state_support_QC.csv", index=False)
    dataset_support.to_csv(
        tables_dir / "11_state_dataset_support_QC.csv", index=False
    )
    donor_state.to_csv(
        tables_dir / "11_donor_state_mean_AUCell.csv.gz", index=False, compression="gzip"
    )
    state_dataset_mean.to_csv(
        tables_dir / "11_state_dataset_donor_mean_AUCell.csv.gz",
        index=False,
        compression="gzip",
    )
    thresholds.rename("AUCell_visualization_threshold").to_csv(
        tables_dir / "11_regulon_activity_visualization_thresholds.csv"
    )
    for filename, matrix in [
        ("11_raw_cell_RSS_group_size_sensitive.csv", raw_rss),
        ("11_cell_mean_AUCell_by_state.csv", cell_mean),
        ("11_cell_mean_AUCell_zscore_by_state.csv", cell_z),
        ("11_active_cell_percent_by_state.csv", active_percent),
        ("11_equal_cell_balanced_RSS_median.csv", cell_rss_median),
        ("11_equal_cell_balanced_RSS_q025.csv", cell_rss_low),
        ("11_equal_cell_balanced_RSS_q975.csv", cell_rss_high),
        ("11_equal_cell_balanced_RSS_top_frequency.csv", cell_rss_frequency),
        ("11_donor_balanced_mean_AUCell.csv", donor_mean),
        ("11_donor_balanced_mean_AUCell_q025.csv", donor_mean_low),
        ("11_donor_balanced_mean_AUCell_q975.csv", donor_mean_high),
        ("11_donor_balanced_mean_AUCell_zscore.csv", donor_z),
        ("11_donor_bootstrap_RSS_median.csv", donor_rss_median),
        ("11_donor_bootstrap_RSS_q025.csv", donor_rss_low),
        ("11_donor_bootstrap_RSS_q975.csv", donor_rss_high),
        ("11_donor_bootstrap_RSS_top_frequency.csv", donor_rss_frequency),
        ("11_dataset_donor_balanced_mean_AUCell.csv", dataset_mean),
        ("11_dataset_donor_balanced_mean_AUCell_q025.csv", dataset_mean_low),
        ("11_dataset_donor_balanced_mean_AUCell_q975.csv", dataset_mean_high),
        ("11_dataset_donor_balanced_mean_AUCell_zscore.csv", dataset_z),
        ("11_dataset_donor_bootstrap_RSS_median.csv", dataset_rss_median),
        ("11_dataset_donor_bootstrap_RSS_q025.csv", dataset_rss_low),
        ("11_dataset_donor_bootstrap_RSS_q975.csv", dataset_rss_high),
        ("11_dataset_donor_bootstrap_RSS_top_frequency.csv", dataset_rss_frequency),
    ]:
        matrix.to_csv(tables_dir / filename)
    consensus.to_csv(primary_output, index=False)
    top_candidates.to_csv(
        tables_dir / "11_SCENIC_top_regulons_per_state.csv", index=False
    )

    if args.write_h5ad:
        aligned = adata[auc.index].copy()
        aligned.obsm["X_pyscenic_auc"] = auc.to_numpy(dtype=np.float32)
        aligned.uns["pyscenic_regulon_names"] = np.asarray(auc.columns, dtype=str)
        aligned.uns["pyscenic_auc_source"] = str(auc_path)
        aligned.uns["pyscenic_analysis_script"] = SCRIPT_VERSION
        aligned.write_h5ad(
            output_root / "11_combined_LEC_with_SCENIC_AUCell.h5ad",
            compression="gzip",
        )

    selected_regulons = select_figure_regulons(
        consensus,
        states,
        args.figure_regulons_per_state,
        args.max_regulons_in_summary_figure,
    )
    figure_rows = []
    if not args.skip_figures:
        figure_specs = [
            (
                "11_SCENIC_state_cell_donor_QC.png",
                "State cell and donor counts using Step 08 state colors",
                lambda path: save_state_support_figure(
                    support, states, palette, path, args.dpi
                ),
            ),
            (
                "11_dataset_donor_bootstrap_RSS_heatmap.png",
                "Dataset/donor-bootstrap balanced RSS median",
                lambda path: save_heatmap(
                    dataset_rss_median, states, selected_regulons, palette, path,
                    "Dataset/donor-bootstrap balanced regulon specificity",
                    "Median balanced-bootstrap RSS", args.dpi, False,
                ),
            ),
            (
                "11_dataset_donor_balanced_zscore_heatmap.png",
                "Dataset/donor-balanced mean AUCell z-score",
                lambda path: save_heatmap(
                    dataset_z, states, selected_regulons, palette, path,
                    "Dataset/donor-balanced regulon activity",
                    "Balanced mean AUCell z-score across states", args.dpi, True,
                ),
            ),
            (
                "11_SCENIC_consensus_dotplot.png",
                "Consensus of dataset/donor-balanced z-score, RSS, and stability",
                lambda path: save_consensus_dotplot(
                    consensus, states, selected_regulons, palette, path, args.dpi
                ),
            ),
        ]
        for filename, description, function in figure_specs:
            path = figures_dir / filename
            function(path)
            figure_rows.append({"figure": filename, "description": description})
        umap_regs = (
            top_candidates.sort_values("consensus_specificity_score", ascending=False)[
                "regulon"
            ]
            .drop_duplicates()
            .head(args.umap_regulons)
            .tolist()
        )
        umap_path = figures_dir / "11_SCENIC_top_regulon_UMAPs.png"
        save_umap_figure(adata, auc, states, palette, umap_regs, umap_path, args.dpi)
        if umap_path.exists():
            figure_rows.append(
                {
                    "figure": umap_path.name,
                    "description": "Selected regulon AUCell activity on Step 08 UMAP",
                }
            )
    pd.DataFrame(figure_rows, columns=["figure", "description"]).to_csv(
        output_root / "11_figure_manifest.csv", index=False
    )

    resampling_qc = pd.DataFrame(
        [
            {
                "analysis": "equal_cell_balanced_RSS",
                "repetitions": args.equal_cell_repetitions,
                "observations_per_state_per_repetition": cell_target,
                "sampling_unit": "cell_without_replacement",
                "purpose": "correct_state_cell_count_imbalance",
            },
            {
                "analysis": "donor_bootstrap_RSS",
                "repetitions": args.donor_bootstrap_repetitions,
                "observations_per_state_per_repetition": donor_target,
                "sampling_unit": "donor_state_mean_with_replacement",
                "purpose": "equal_donor_weight_and_donor_uncertainty",
            },
            {
                "analysis": "dataset_donor_bootstrap_RSS_primary",
                "repetitions": args.dataset_donor_bootstrap_repetitions,
                "observations_per_state_dataset_per_repetition": dataset_donor_target,
                "datasets": ";".join(common_datasets),
                "sampling_unit": "donor_state_mean_with_replacement_within_dataset",
                "purpose": "equal_dataset_and_donor_weight_for_state_regulon_specificity",
            },
        ]
    )
    resampling_qc.to_csv(tables_dir / "11_balancing_and_resampling_QC.csv", index=False)
    pd.DataFrame([alignment_report]).to_csv(
        tables_dir / "11_AUCell_alignment_QC.csv", index=False
    )

    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "h5ad": str(h5ad_path),
        "auc_csv": str(auc_path),
        "regulons_csv": str(regulons_path),
        "identity_qc_csv": str(identity_qc_path),
        "output_root": str(output_root),
        "state_key": args.state_key,
        "donor_key": args.donor_key,
        "dataset_key": args.dataset_key,
        "states": states,
        "common_datasets_for_primary_balance": common_datasets,
        "step08_script_version": str(
            step08_combine_info.get("script_version", "unknown")
        ),
        "step08_leiden_resolution": step08_resolution,
        "step08_excluded_donors": list(
            step08_combine_info.get("excluded_donors", [])
        ),
        "n_cells": int(len(auc)),
        "n_regulons": int(auc.shape[1]),
        "equal_cell_repetitions": args.equal_cell_repetitions,
        "equal_cell_target": cell_target,
        "donor_bootstrap_repetitions": args.donor_bootstrap_repetitions,
        "donor_bootstrap_target": donor_target,
        "dataset_donor_bootstrap_repetitions": args.dataset_donor_bootstrap_repetitions,
        "donors_per_state_dataset_bootstrap_target": dataset_donor_target,
        "active_quantile_visualization_only": args.active_quantile,
        "random_seed": args.random_seed,
        "write_h5ad": args.write_h5ad,
        "figures_enabled": not args.skip_figures,
        "alignment_report": alignment_report,
    }
    (output_root / "11_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    cautions = support.loc[support["interpretation_flag"].ne("OK"), "LEC_state"].tolist()
    notes = f"""Step 11 SCENIC interpretation notes
===================================

Primary interpretation
----------------------
Use agreement between dataset_donor_bootstrap_RSS_median,
dataset_donor_balanced_mean_AUCell_zscore, and
dataset_donor_bootstrap_RSS_top_frequency. The all-dataset donor-balanced
results remain a sensitivity analysis.
The consensus score is the unweighted mean of their three within-state
percentile/stability values and is intended for prioritization, not hypothesis
testing.

Step 08 source
--------------
Step 08 script: {step08_combine_info.get('script_version', 'unknown')}
Leiden resolution: {step08_resolution:g}
State labels: {', '.join(states)}
These labels were read from the current Step 08 h5ad.

Balancing
---------
Equal-cell RSS: {cell_target} cells/state x {args.equal_cell_repetitions} repetitions.
Donor-bootstrap RSS: {donor_target} donor-state means/state x
{args.donor_bootstrap_repetitions} repetitions, sampled with replacement.
Primary strict dataset/donor balance: {dataset_donor_target} donor-state
means per state x dataset from {', '.join(common_datasets)} x
{args.dataset_donor_bootstrap_repetitions} repetitions. Dataset11 is excluded
from this strict comparison when it is absent from any state; it remains in
the all-dataset donor-balanced sensitivity results.

Z-score
-------
Each regulon is standardized across the {len(states)} LEC states in this run. A positive z-score means
that the regulon is relatively more active in that state than in other states.
It does not mean that different regulons have directly comparable activity.

Cautions
--------
States requiring donor-support caution: {', '.join(cautions) if cautions else 'none'}.
Raw cell RSS is exported only for reference because unequal state sizes and
donor contributions can affect it. Active-cell thresholds are visualization
thresholds, not experimentally validated biological cutoffs.
SCENIC expression and AUCell values are not Harmony-corrected. The primary
resampling controls donor and shared-dataset composition instead of altering
gene-gene correlations. This analysis describes regulon specificity of states;
it is not a donor-level test of disease-condition effects.
"""
    (output_root / "11_method_and_interpretation_notes.txt").write_text(
        notes, encoding="utf-8"
    )

    print("\nStep 11 SCENIC analysis complete.")
    print(f"Full state-regulon summary: {primary_output}")
    print(f"Equal-cell RSS target: {cell_target} cells/state")
    print(f"Donor-bootstrap RSS target: {donor_target} donor-state means/state")
    print(
        "Dataset/donor-bootstrap target: "
        f"{dataset_donor_target} donors/state/dataset across "
        + ", ".join(common_datasets)
    )
    if cautions:
        print("Interpret with donor caution: " + ", ".join(cautions))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
