#!/usr/bin/env python3
r"""
Step 09: preranked GSEA for the combined NF LEC states from Step 08.

This script characterizes each LEC state by comparing that state with all
other combined LECs. It uses the all-gene Wilcoxon rankings already produced
by Step 08 from unscaled ``adata.raw`` expression. Harmony coordinates are
never used for differential ranking or GSEA.

Four local MSigDB 2026.1.Hs collections are analyzed:

* Hallmark
* GO Biological Process (GO:BP)
* KEGG Medicus
* Reactome

The GMT files are the GSEA inputs. Matching official MSigDB JSON files provide
``exactSource`` metadata so the output retains GO IDs, Reactome stable IDs,
and KEGG pathway IDs. When no external biological ID exists (normally
Hallmark), ``biological_id_or_name`` falls back to the pathway name.

Default input
-------------
new_result2_aging/08_combined_LEC_recluster/dataset13/
    08_combined_final_LEC_reclustered.h5ad
    08_state_marker_ranking_all_genes.csv.gz
    08_state_donor_project_QC.csv
    08_state_color_mapping.csv

resources/gene_sets/
    h.all.v2026.1.Hs.symbols.gmt + matching .json
    c5.go.bp.v2026.1.Hs.symbols.gmt + matching .json
    c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt + matching .json
    c2.cp.reactome.v2026.1.Hs.symbols.gmt + matching .json

Default output
--------------
new_result2_aging/09_LEC_state_GSEA/dataset13/

Main tables
-----------
09_pathway_inventory_full.csv
    One row per pathway, including stable ID, URLs, all member genes, overlap
    with the ranked genes, and whether it passed the GSEA size filter.

09_GSEA_all_pathways_full.csv
    One row per state x pathway, including all enrichment statistics. Pathways
    excluded by size are retained with an explicit status and empty statistics.

09_GSEA_significant_FDR05.csv
    Convenient subset passing the chosen within-run FDR threshold.

09_GSEA_top_pathways_per_state.csv
    A compact positive/negative summary for review and plotting.

Important interpretation
------------------------
This is a descriptive STATE-vs-rest analysis, not a test of aging. A later
age-effect analysis should rank genes with donors as the biological replicates
(preferably state-specific pseudobulk) before GSEA. States with low donor
support or >50% contribution from one donor are kept but explicitly flagged.

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_aging\09_GSEA_combined_LEC_states.py

For a quick pipeline check only (not final results):
    .\.venv\Scripts\python.exe `
      .\new_script2_aging\09_GSEA_combined_LEC_states.py `
      --collections Hallmark --states state0 --permutation_num 20 `
      --output_root outputs/aging_GSEA_smoke --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, Optional

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg", force=True)


SCRIPT_VERSION = "09_dataset13_NF_combined_LEC_state_GSEA_resolution_aware_2026_08_07"
DATASET = "dataset13"
REQUIRED_DISEASE = "NF"
ALLOWED_REGIONS = {"AP", "LV"}

DEFAULT_STEP08_ROOT = "new_result2_aging/08_combined_LEC_recluster/dataset13"
DEFAULT_H5AD = f"{DEFAULT_STEP08_ROOT}/08_combined_final_LEC_reclustered.h5ad"
DEFAULT_RANKING = f"{DEFAULT_STEP08_ROOT}/08_state_marker_ranking_all_genes.csv.gz"
DEFAULT_STATE_QC = f"{DEFAULT_STEP08_ROOT}/08_state_donor_project_QC.csv"
DEFAULT_STATE_PALETTE = f"{DEFAULT_STEP08_ROOT}/08_state_color_mapping.csv"
DEFAULT_GENE_SETS_DIR = "resources/gene_sets"
DEFAULT_OUTPUT_ROOT = "new_result2_aging/09_LEC_state_GSEA/dataset13"

COLLECTION_SPECS = {
    "Hallmark": {
        "gmt": "h.all.v2026.1.Hs.symbols.gmt",
        "json": "h.all.v2026.1.Hs.json",
        "prefix": "HALLMARK_",
        "id_type": "pathway_name",
    },
    "GO_BP": {
        "gmt": "c5.go.bp.v2026.1.Hs.symbols.gmt",
        "json": "c5.go.bp.v2026.1.Hs.json",
        "prefix": "GOBP_",
        "id_type": "GO_ID",
    },
    "KEGG_Medicus": {
        "gmt": "c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt",
        "json": "c2.cp.kegg_medicus.v2026.1.Hs.json",
        "prefix": "KEGG_MEDICUS_",
        "id_type": "KEGG_pathway_ID",
    },
    "Reactome": {
        "gmt": "c2.cp.reactome.v2026.1.Hs.symbols.gmt",
        "json": "c2.cp.reactome.v2026.1.Hs.json",
        "prefix": "REACTOME_",
        "id_type": "Reactome_stable_ID",
    },
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
            "Run Hallmark, GO:BP, KEGG Medicus, and Reactome preranked GSEA "
            "for each combined NF LEC state versus all other LECs."
        )
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--ranking_csv", default=DEFAULT_RANKING)
    parser.add_argument("--state_qc_csv", default=DEFAULT_STATE_QC)
    parser.add_argument("--state_palette_csv", default=DEFAULT_STATE_PALETTE)
    parser.add_argument("--gene_sets_dir", default=DEFAULT_GENE_SETS_DIR)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--state_key", default="LEC_state")
    parser.add_argument(
        "--states",
        nargs="*",
        default=None,
        help="Optional subset such as --states state0 state1. Default: all states.",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        choices=list(COLLECTION_SPECS),
        default=list(COLLECTION_SPECS),
    )
    parser.add_argument(
        "--ranking_metric",
        choices=["scores", "logfoldchanges", "signed_neglog10_padj"],
        default="scores",
        help=(
            "Metric from the Step 08 Wilcoxon table. Scores are recommended "
            "because they retain effect direction and ranking strength."
        ),
    )
    parser.add_argument("--min_size", type=int, default=10)
    parser.add_argument("--max_size", type=int, default=500)
    parser.add_argument(
        "--permutation_num",
        type=int,
        default=1000,
        help="Use 1000 for final results; small values are only for smoke tests.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fdr_threshold", type=float, default=0.05)
    parser.add_argument(
        "--top_per_direction",
        type=int,
        default=10,
        help="Top positive and negative terms saved per state and collection.",
    )
    parser.add_argument(
        "--top_pathways_figure",
        type=int,
        default=18,
        help="Maximum pathways shown in each collection-level dot plot.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing primary Step 09 outputs.",
    )
    return parser.parse_args()


def natural_key(value: object) -> tuple:
    pieces = re.split(r"(\d+(?:\.\d+)?)", str(value))
    out: list[tuple[int, object]] = []
    for piece in pieces:
        if not piece:
            continue
        try:
            out.append((0, float(piece)))
        except ValueError:
            out.append((1, piece.lower()))
    return tuple(out)


def normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def first_existing(columns: Iterable[str], frame: pd.DataFrame) -> Optional[str]:
    return next((column for column in columns if column in frame.columns), None)


def parse_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return normalized_text(values).str.lower().isin({"true", "1", "yes", "y"})


def read_step08_metadata(
    h5ad_path: Path, state_key: str
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Step 08 h5ad not found: {h5ad_path}")
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = backed.obs.copy()
        if state_key not in obs.columns:
            raise KeyError(
                f"State key '{state_key}' not found in {h5ad_path}. "
                f"Available obs columns: {list(obs.columns)}"
            )
        states = sorted(normalized_text(obs[state_key]).unique(), key=natural_key)
        raw_present = backed.raw is not None
        raw_n_vars = int(backed.raw.shape[1]) if raw_present else 0
        info = {
            "n_cells": int(backed.n_obs),
            "n_genes_X": int(backed.n_vars),
            "raw_present": bool(raw_present),
            "n_genes_raw": raw_n_vars,
            "n_states": int(len(states)),
        }
        combine_info = dict(backed.uns.get("step08_combine_info", {}))
        embedding_info = dict(backed.uns.get("combined_LEC_embedding", {}))
        info["step08_script_version"] = str(
            combine_info.get("script_version", "unknown")
        )
        info["leiden_resolution"] = float(
            embedding_info.get(
                "leiden_resolution", combine_info.get("leiden_resolution", np.nan)
            )
        )
        info["step08_excluded_donors"] = list(
            combine_info.get("excluded_donors", [])
        )
    finally:
        try:
            backed.file.close()
        except Exception:
            pass

    disease_col = first_existing(["disease", "aging_disease_filter"], obs)
    if disease_col:
        disease = set(normalized_text(obs[disease_col]).str.upper().unique())
        if disease != {REQUIRED_DISEASE}:
            raise ValueError(
                f"Step 09 requires NF only, but {disease_col} contains {sorted(disease)}."
            )
    else:
        warnings.warn("No NF disease-filter column was available for validation.")

    region_col = first_existing(
        ["aging_standardized_region", "Standardized_region"], obs
    )
    if region_col:
        regions = set(normalized_text(obs[region_col]).str.upper().unique())
        unexpected = regions.difference(ALLOWED_REGIONS)
        if unexpected:
            raise ValueError(
                f"Step 09 requires AP/LV only, but {region_col} also contains "
                f"{sorted(unexpected)}."
            )
        info["regions"] = sorted(regions)
    else:
        warnings.warn("No standardized-region column was available for AP/LV validation.")
        info["regions"] = []

    if not info["raw_present"]:
        warnings.warn(
            "adata.raw is absent. Step 09 uses the saved Step 08 ranking CSV, but "
            "the source expression cannot be independently verified from this h5ad."
        )
    return obs, states, info


def select_states(
    requested: Optional[list[str]], available: list[str]
) -> list[str]:
    if not requested:
        return available
    requested_clean = [str(value).strip() for value in requested]
    missing = sorted(set(requested_clean).difference(available), key=natural_key)
    if missing:
        raise ValueError(
            f"Requested states are absent: {missing}. Available states: {available}"
        )
    return sorted(dict.fromkeys(requested_clean), key=natural_key)


def load_state_support(
    obs: pd.DataFrame,
    state_key: str,
    selected_states: list[str],
    state_qc_path: Path,
) -> pd.DataFrame:
    if state_qc_path.exists():
        support = pd.read_csv(state_qc_path)
        if "LEC_state" not in support.columns:
            raise KeyError(f"LEC_state is absent from {state_qc_path}")
    else:
        warnings.warn(
            f"State QC table not found; recalculating basic support from obs: {state_qc_path}"
        )
        if "donor_key" not in obs.columns:
            raise KeyError("donor_key is required to recalculate state support.")
        rows = []
        for state, group in obs.groupby(state_key, observed=True):
            donor_counts = normalized_text(group["donor_key"]).value_counts()
            n_cells = int(len(group))
            rows.append(
                {
                    "LEC_state": str(state),
                    "n_cells": n_cells,
                    "n_donors": int(group["donor_key"].nunique()),
                    "top_donor": donor_counts.index[0],
                    "top_donor_n_cells": int(donor_counts.iloc[0]),
                    "top_donor_fraction": float(donor_counts.iloc[0] / n_cells),
                }
            )
        support = pd.DataFrame(rows)

    support["LEC_state"] = normalized_text(support["LEC_state"])
    support = support.loc[support["LEC_state"].isin(selected_states)].copy()
    missing = sorted(set(selected_states).difference(support["LEC_state"]), key=natural_key)
    if missing:
        raise ValueError(f"State QC information is missing for: {missing}")

    for column in ["n_cells", "n_donors", "top_donor_n_cells", "top_donor_fraction"]:
        if column in support.columns:
            support[column] = pd.to_numeric(support[column], errors="coerce")
    if "top_donor_fraction" not in support.columns:
        support["top_donor_fraction"] = (
            support["top_donor_n_cells"] / support["n_cells"]
        )
    support["donor_dominance_flag_over_50pct"] = support[
        "top_donor_fraction"
    ].gt(0.5)
    support["low_donor_support_flag_under_10"] = support["n_donors"].lt(10)

    def interpretation_flag(row: pd.Series) -> str:
        flags = []
        if bool(row["donor_dominance_flag_over_50pct"]):
            flags.append("CAUTION_single_donor_over_50pct")
        if bool(row["low_donor_support_flag_under_10"]):
            flags.append("CAUTION_fewer_than_10_donors")
        return ";".join(flags) if flags else "OK"

    support["interpretation_flag"] = support.apply(interpretation_flag, axis=1)
    support["state_order"] = support["LEC_state"].map(
        {state: index for index, state in enumerate(selected_states)}
    )
    return support.sort_values("state_order").drop(columns="state_order")


def load_state_palette(
    palette_path: Path, selected_states: list[str]
) -> dict[str, str]:
    fallback = [
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    ]
    palette: dict[str, str] = {}
    if palette_path.exists():
        frame = pd.read_csv(palette_path)
        if {"LEC_state", "color_hex"}.issubset(frame.columns):
            palette = dict(zip(frame["LEC_state"].astype(str), frame["color_hex"]))
    for index, state in enumerate(selected_states):
        palette.setdefault(state, fallback[index % len(fallback)])
    return palette


def load_rankings(
    ranking_path: Path,
    selected_states: list[str],
    ranking_metric: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    if not ranking_path.exists():
        raise FileNotFoundError(f"Step 08 all-gene ranking not found: {ranking_path}")
    ranking = pd.read_csv(ranking_path)
    required = {"LEC_state", "names", "scores", "logfoldchanges", "pvals_adj"}
    missing_columns = required.difference(ranking.columns)
    if missing_columns:
        raise KeyError(
            f"Missing required columns in {ranking_path}: {sorted(missing_columns)}"
        )
    ranking["LEC_state"] = normalized_text(ranking["LEC_state"])
    ranking["names"] = normalized_text(ranking["names"])
    ranking = ranking.loc[ranking["LEC_state"].isin(selected_states)].copy()

    if ranking_metric == "signed_neglog10_padj":
        logfc = pd.to_numeric(ranking["logfoldchanges"], errors="coerce")
        padj = pd.to_numeric(ranking["pvals_adj"], errors="coerce").clip(
            lower=np.finfo(float).tiny, upper=1.0
        )
        ranking["gsea_rank_metric_original"] = np.sign(logfc) * -np.log10(padj)
    else:
        ranking["gsea_rank_metric_original"] = pd.to_numeric(
            ranking[ranking_metric], errors="coerce"
        )

    by_state: dict[str, pd.DataFrame] = {}
    qc_rows = []
    adjusted_rows = []
    for state in selected_states:
        frame = ranking.loc[
            ranking["LEC_state"].eq(state),
            ["names", "gsea_rank_metric_original", "logfoldchanges", "pvals_adj"],
        ].copy()
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        frame = frame.loc[frame["names"].ne("")]
        if frame["names"].duplicated().any():
            frame = (
                frame.groupby("names", as_index=False, observed=True)
                .agg(
                    gsea_rank_metric_original=("gsea_rank_metric_original", "mean"),
                    logfoldchanges=("logfoldchanges", "mean"),
                    pvals_adj=("pvals_adj", "min"),
                )
            )
        frame["logfoldchanges"] = pd.to_numeric(
            frame["logfoldchanges"], errors="coerce"
        ).fillna(0.0)
        frame["pvals_adj"] = pd.to_numeric(
            frame["pvals_adj"], errors="coerce"
        ).fillna(1.0)
        frame = frame.sort_values(
            ["gsea_rank_metric_original", "logfoldchanges", "pvals_adj", "names"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        if len(frame) < 100:
            raise ValueError(f"Only {len(frame)} ranked genes remain for {state}.")

        # GSEApy warns that exact ties are otherwise ordered arbitrarily. Break
        # them deterministically without changing the ordering of any distinct
        # primary Wilcoxon score. Within a tie, logFC, adjusted p, and gene name
        # (the stable sort above) determine the order.
        original = frame["gsea_rank_metric_original"].to_numpy(dtype=float)
        adjusted = original.copy()
        unique_values = np.unique(original)[::-1]
        value_position = {value: index for index, value in enumerate(unique_values)}
        for value, index_values in frame.groupby(
            "gsea_rank_metric_original", sort=False, observed=True
        ).groups.items():
            positions = np.asarray(list(index_values), dtype=int)
            if len(positions) < 2:
                continue
            unique_index = value_position[float(value)]
            gaps = []
            if unique_index > 0:
                gaps.append(float(unique_values[unique_index - 1] - value))
            if unique_index + 1 < len(unique_values):
                gaps.append(float(value - unique_values[unique_index + 1]))
            positive_gaps = [gap for gap in gaps if gap > 0 and np.isfinite(gap)]
            if positive_gaps:
                margin = 0.25 * min(positive_gaps)
            else:
                margin = max(abs(float(value)), 1.0) * 1e-9
            adjusted[positions] = float(value) + np.linspace(
                margin, -margin, len(positions)
            )
        frame["gsea_rank_metric"] = adjusted
        frame = frame.sort_values(
            ["gsea_rank_metric", "names"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        duplicate_scores = int(
            frame["gsea_rank_metric_original"].duplicated(keep=False).sum()
        )
        remaining_ties = int(frame["gsea_rank_metric"].duplicated(keep=False).sum())
        if remaining_ties:
            raise ValueError(
                f"Deterministic tie-breaking left {remaining_ties} tied genes in {state}."
            )
        max_perturbation = float(
            np.max(
                np.abs(
                    frame["gsea_rank_metric"].to_numpy()
                    - frame["gsea_rank_metric_original"].to_numpy()
                )
            )
        )
        qc_rows.append(
            {
                "LEC_state": state,
                "n_ranked_genes": int(len(frame)),
                "n_positive_metric": int(frame["gsea_rank_metric"].gt(0).sum()),
                "n_negative_metric": int(frame["gsea_rank_metric"].lt(0).sum()),
                "n_zero_metric": int(frame["gsea_rank_metric"].eq(0).sum()),
                "n_genes_with_tied_original_metric": duplicate_scores,
                "n_genes_with_tied_final_metric": remaining_ties,
                "maximum_absolute_tie_break_perturbation": max_perturbation,
                "rank_metric_min": float(frame["gsea_rank_metric"].min()),
                "rank_metric_max": float(frame["gsea_rank_metric"].max()),
            }
        )
        adjusted_rows.append(
            frame[
                ["names", "gsea_rank_metric_original", "gsea_rank_metric"]
            ].assign(LEC_state=state)
        )
        by_state[state] = frame[["names", "gsea_rank_metric"]].copy()

    adjusted_table = pd.concat(adjusted_rows, ignore_index=True)
    ranking = ranking.drop(columns=["gsea_rank_metric_original"], errors="ignore").merge(
        adjusted_table,
        on=["LEC_state", "names"],
        how="left",
        validate="one_to_one",
    )
    return ranking, by_state, pd.DataFrame(qc_rows)


def read_gmt(path: Path) -> dict[str, dict[str, object]]:
    pathways: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Malformed GMT line {line_number} in {path}")
            name, description, *genes = fields
            name = name.strip()
            if not name:
                raise ValueError(f"Blank pathway name at GMT line {line_number} in {path}")
            if name in pathways:
                raise ValueError(f"Duplicate pathway name in {path}: {name}")
            clean_genes = list(dict.fromkeys(gene.strip() for gene in genes if gene.strip()))
            pathways[name] = {"description": description.strip(), "genes": clean_genes}
    return pathways


def display_pathway_name(pathway_name: str, prefix: str) -> str:
    text = pathway_name[len(prefix):] if pathway_name.startswith(prefix) else pathway_name
    return text.replace("_", " ").strip()


def build_pathway_inventory(
    gene_sets_dir: Path,
    collections: list[str],
    ranked_genes: set[str],
    min_size: int,
    max_size: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[str]]]]:
    rows = []
    gsea_sets: dict[str, dict[str, list[str]]] = {}
    for collection in collections:
        spec = COLLECTION_SPECS[collection]
        gmt_path = gene_sets_dir / str(spec["gmt"])
        json_path = gene_sets_dir / str(spec["json"])
        if not gmt_path.exists():
            raise FileNotFoundError(f"GMT resource not found: {gmt_path}")
        if not json_path.exists():
            raise FileNotFoundError(
                f"MSigDB metadata JSON not found: {json_path}. It is required "
                "to preserve stable biological pathway IDs."
            )
        pathways = read_gmt(gmt_path)
        with json_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        gsea_sets[collection] = {
            name: list(details["genes"]) for name, details in pathways.items()
        }
        for pathway_name, details in pathways.items():
            meta = metadata.get(pathway_name, {})
            exact_source = str(meta.get("exactSource", "") or "").strip()
            biological_id = exact_source if exact_source else pathway_name
            id_fallback = not bool(exact_source)
            id_type = "pathway_name" if id_fallback else str(spec["id_type"])
            genes = list(details["genes"])
            overlap = [gene for gene in genes if gene in ranked_genes]
            n_overlap = len(overlap)
            if n_overlap < min_size:
                eligibility = "excluded_below_min_size"
            elif n_overlap > max_size:
                eligibility = "excluded_above_max_size"
            else:
                eligibility = "eligible"
            display_name = display_pathway_name(pathway_name, str(spec["prefix"]))
            pathway_label = (
                display_name
                if id_fallback
                else f"{biological_id} | {display_name}"
            )
            rows.append(
                {
                    "collection": collection,
                    "pathway_name": pathway_name,
                    "pathway_display_name": display_name,
                    "biological_id": exact_source,
                    "biological_id_type": id_type,
                    "biological_id_or_name": biological_id,
                    "used_pathway_name_fallback": id_fallback,
                    "pathway_label": pathway_label,
                    "systematic_name": str(meta.get("systematicName", "") or ""),
                    "msigdb_collection": str(meta.get("collection", "") or ""),
                    "pmid": str(meta.get("pmid", "") or ""),
                    "msigdb_url": str(meta.get("msigdbURL", "") or details["description"]),
                    "external_details_url": str(meta.get("externalDetailsURL", "") or ""),
                    "gmt_description": str(details["description"]),
                    "source_gmt": str(gmt_path),
                    "source_metadata_json": str(json_path),
                    "n_genes_gmt": int(len(genes)),
                    "n_genes_in_rank": int(n_overlap),
                    "rank_overlap_fraction": (
                        float(n_overlap / len(genes)) if genes else np.nan
                    ),
                    "size_filter_status": eligibility,
                    "genes_gmt": ";".join(genes),
                    "genes_in_rank": ";".join(overlap),
                }
            )

        missing_meta = set(pathways).difference(metadata)
        if missing_meta:
            warnings.warn(
                f"{collection}: {len(missing_meta)} GMT pathways lack JSON metadata; "
                "their pathway names will be used as ID fallback."
            )

    inventory = pd.DataFrame(rows)
    collection_order = {name: index for index, name in enumerate(collections)}
    inventory["collection_order"] = inventory["collection"].map(collection_order)
    inventory = inventory.sort_values(
        ["collection_order", "pathway_name"], kind="mergesort"
    ).drop(columns="collection_order")
    return inventory.reset_index(drop=True), gsea_sets


def parse_fraction(value: object) -> tuple[float, float]:
    text = str(value).strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2))


def parse_percent(value: object) -> float:
    text = str(value).strip().rstrip("%")
    try:
        return float(text)
    except ValueError:
        return np.nan


def normalize_gseapy_results(result: pd.DataFrame) -> pd.DataFrame:
    if result is None or result.empty:
        return pd.DataFrame()
    frame = result.copy().reset_index(drop=True)
    rename = {
        "Name": "gseapy_analysis_name",
        "Term": "pathway_name",
        "ES": "ES",
        "NES": "NES",
        "NOM p-val": "nominal_p_value",
        "FDR q-val": "FDR_q_value",
        "FWER p-val": "FWER_p_value",
        "Tag %": "tag_fraction_raw",
        "Gene %": "gene_percent_raw",
        "Lead_genes": "leading_edge_genes",
    }
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame})
    if "pathway_name" not in frame.columns:
        raise KeyError(f"GSEApy result lacks a pathway/Term column: {list(frame.columns)}")
    for column in ["ES", "NES", "nominal_p_value", "FDR_q_value", "FWER_p_value"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = np.nan
    if "tag_fraction_raw" in frame.columns:
        fractions = frame["tag_fraction_raw"].map(parse_fraction)
        frame["tag_n_genes"] = [value[0] for value in fractions]
        frame["tag_gene_set_denominator"] = [value[1] for value in fractions]
    else:
        frame["tag_fraction_raw"] = ""
        frame["tag_n_genes"] = np.nan
        frame["tag_gene_set_denominator"] = np.nan
    if "gene_percent_raw" in frame.columns:
        frame["gene_percent"] = frame["gene_percent_raw"].map(parse_percent)
    else:
        frame["gene_percent_raw"] = ""
        frame["gene_percent"] = np.nan
    if "leading_edge_genes" not in frame.columns:
        frame["leading_edge_genes"] = ""
    frame["leading_edge_genes"] = frame["leading_edge_genes"].fillna("").astype(str)
    frame["n_leading_edge_genes"] = frame["leading_edge_genes"].map(
        lambda value: len([gene for gene in value.split(";") if gene])
    )
    return frame


def run_all_gsea(
    rankings: dict[str, pd.DataFrame],
    selected_states: list[str],
    collections: list[str],
    inventory: pd.DataFrame,
    gene_sets: dict[str, dict[str, list[str]]],
    state_support: pd.DataFrame,
    state_palette: dict[str, str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import gseapy as gp
    except ImportError as error:
        raise ImportError(
            "gseapy is required. Run this script with the project .venv."
        ) from error

    support_index = state_support.set_index("LEC_state")
    all_frames = []
    run_rows = []
    total_runs = len(selected_states) * len(collections)
    run_number = 0
    for state in selected_states:
        rank_frame = rankings[state]
        for collection in collections:
            run_number += 1
            print(f"[{run_number}/{total_runs}] {state} vs rest | {collection}")
            collection_inventory = inventory.loc[
                inventory["collection"].eq(collection)
            ].drop(columns=["genes_gmt", "genes_in_rank"])
            eligible_names = set(
                collection_inventory.loc[
                    collection_inventory["size_filter_status"].eq("eligible"),
                    "pathway_name",
                ]
            )
            eligible_sets = {
                name: genes
                for name, genes in gene_sets[collection].items()
                if name in eligible_names
            }
            started = time.perf_counter()
            prerank = gp.prerank(
                rnk=rank_frame,
                gene_sets=eligible_sets,
                min_size=args.min_size,
                max_size=args.max_size,
                permutation_num=args.permutation_num,
                weight=1.0,
                ascending=False,
                threads=args.threads,
                seed=args.seed,
                outdir=None,
                no_plot=True,
                verbose=False,
            )
            normalized = normalize_gseapy_results(prerank.res2d)
            elapsed = time.perf_counter() - started

            merged = collection_inventory.copy()
            if not normalized.empty:
                duplicate_terms = normalized["pathway_name"].duplicated().sum()
                if duplicate_terms:
                    raise ValueError(
                        f"GSEApy returned {duplicate_terms} duplicate pathways for "
                        f"{state} / {collection}."
                    )
                merged = merged.merge(normalized, on="pathway_name", how="left")
            else:
                for column in [
                    "gseapy_analysis_name", "ES", "NES", "nominal_p_value",
                    "FDR_q_value", "FWER_p_value", "tag_fraction_raw",
                    "tag_n_genes", "tag_gene_set_denominator", "gene_percent_raw",
                    "gene_percent", "leading_edge_genes", "n_leading_edge_genes",
                ]:
                    merged[column] = np.nan

            merged.insert(0, "LEC_state", state)
            merged.insert(1, "state_color", state_palette[state])
            support = support_index.loc[state]
            for output_column, source_column in [
                ("n_state_cells", "n_cells"),
                ("n_state_donors", "n_donors"),
                ("top_donor", "top_donor"),
                ("top_donor_fraction", "top_donor_fraction"),
                ("donor_dominance_flag_over_50pct", "donor_dominance_flag_over_50pct"),
                ("low_donor_support_flag_under_10", "low_donor_support_flag_under_10"),
                ("interpretation_flag", "interpretation_flag"),
            ]:
                merged[output_column] = support.get(source_column, np.nan)
            merged["contrast"] = f"{state}_vs_all_other_LEC_states"
            merged["gsea_status"] = np.where(
                merged["size_filter_status"].ne("eligible"),
                merged["size_filter_status"],
                np.where(merged["NES"].notna(), "tested", "eligible_no_GSEA_result"),
            )
            merged["rank_method"] = "Step08_cell_level_Wilcoxon_state_vs_rest"
            merged["ranking_metric"] = args.ranking_metric
            merged["permutation_num"] = args.permutation_num
            merged["min_size"] = args.min_size
            merged["max_size"] = args.max_size
            merged["seed"] = args.seed
            merged["FDR_threshold"] = args.fdr_threshold
            merged["significant_at_FDR_threshold"] = (
                merged["gsea_status"].eq("tested")
                & pd.to_numeric(merged["FDR_q_value"], errors="coerce").le(
                    args.fdr_threshold
                )
            )
            merged["enrichment_direction"] = np.select(
                [merged["NES"].gt(0), merged["NES"].lt(0)],
                ["positive_in_state", "negative_in_state"],
                default="not_tested_or_zero",
            )
            all_frames.append(merged)

            tested = merged.loc[merged["gsea_status"].eq("tested")]
            significant = tested.loc[tested["significant_at_FDR_threshold"]]
            run_rows.append(
                {
                    "LEC_state": state,
                    "collection": collection,
                    "n_ranked_genes": int(len(rank_frame)),
                    "n_pathways_total": int(len(merged)),
                    "n_pathways_eligible": int(len(eligible_sets)),
                    "n_pathways_tested": int(len(tested)),
                    "n_significant_FDR": int(len(significant)),
                    "n_significant_positive": int(significant["NES"].gt(0).sum()),
                    "n_significant_negative": int(significant["NES"].lt(0).sum()),
                    "elapsed_seconds": round(elapsed, 3),
                }
            )

    return pd.concat(all_frames, ignore_index=True), pd.DataFrame(run_rows)


def make_top_pathway_table(
    tested_results: pd.DataFrame, top_per_direction: int
) -> pd.DataFrame:
    frames = []
    for (_, _), group in tested_results.groupby(
        ["LEC_state", "collection"], sort=False, observed=True
    ):
        for direction, sign in [("positive_in_state", 1), ("negative_in_state", -1)]:
            subset = group.loc[np.sign(group["NES"]).eq(sign)].copy()
            subset = subset.sort_values(
                ["FDR_q_value", "nominal_p_value", "NES"],
                ascending=[True, True, sign < 0],
                kind="mergesort",
            ).head(top_per_direction)
            subset["top_direction"] = direction
            subset["top_direction_rank"] = np.arange(1, len(subset) + 1)
            frames.append(subset)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_collection_summary(
    full_results: pd.DataFrame,
    selected_states: list[str],
    collections: list[str],
) -> pd.DataFrame:
    state_order = {state: index for index, state in enumerate(selected_states)}
    collection_order = {name: index for index, name in enumerate(collections)}
    rows = []
    for (state, collection), group in full_results.groupby(
        ["LEC_state", "collection"], observed=True, sort=False
    ):
        tested = group.loc[group["gsea_status"].eq("tested")]
        significant = tested.loc[tested["significant_at_FDR_threshold"]]
        rows.append(
            {
                "LEC_state": state,
                "collection": collection,
                "n_pathways_total": int(len(group)),
                "n_pathways_tested": int(len(tested)),
                "n_significant_FDR": int(len(significant)),
                "n_significant_positive": int(significant["NES"].gt(0).sum()),
                "n_significant_negative": int(significant["NES"].lt(0).sum()),
                "best_FDR_q_value": (
                    float(tested["FDR_q_value"].min()) if len(tested) else np.nan
                ),
                "state_order": state_order[state],
                "collection_order": collection_order[collection],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["state_order", "collection_order"])
        .drop(columns=["state_order", "collection_order"])
    )


def state_plot_label(state: str, support_index: pd.DataFrame) -> str:
    dominant = bool(support_index.loc[state, "donor_dominance_flag_over_50pct"])
    return f"{state}*" if dominant else state


def save_significant_count_figure(
    summary: pd.DataFrame,
    selected_states: list[str],
    collections: list[str],
    state_palette: dict[str, str],
    state_support: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    ncols = 2
    nrows = math.ceil(len(collections) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    support_index = state_support.set_index("LEC_state")
    for ax, collection in zip(axes_flat, collections):
        frame = (
            summary.loc[summary["collection"].eq(collection)]
            .set_index("LEC_state")
            .reindex(selected_states)
        )
        x = np.arange(len(selected_states))
        ax.bar(
            x,
            frame["n_significant_FDR"].fillna(0),
            color=[state_palette[state] for state in selected_states],
            edgecolor="#333333",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        labels = [state_plot_label(state, support_index) for state in selected_states]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        for tick, state in zip(ax.get_xticklabels(), selected_states):
            tick.set_color(state_palette[state])
            tick.set_fontweight("bold")
        ax.set_title(collection.replace("_", " "))
        ax.set_ylabel("Pathways with FDR q ≤ threshold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes_flat[len(collections):]:
        ax.axis("off")
    fig.suptitle(
        "Significant pathways per combined LEC state\n"
        "State colors match Step 08; * = one donor contributes >50% of state cells",
        y=1.01,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def select_pathways_for_dotplot(
    collection_results: pd.DataFrame, max_pathways: int
) -> list[str]:
    tested = collection_results.loc[
        collection_results["gsea_status"].eq("tested")
    ].copy()
    if tested.empty:
        return []
    pathway_summary = (
        tested.groupby("pathway_name", observed=True)
        .agg(
            best_FDR=("FDR_q_value", "min"),
            max_abs_NES=("NES", lambda values: np.nanmax(np.abs(values))),
            significant=("significant_at_FDR_threshold", "max"),
        )
        .reset_index()
    )
    pathway_summary = pathway_summary.sort_values(
        ["significant", "best_FDR", "max_abs_NES"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    return pathway_summary.head(max_pathways)["pathway_name"].tolist()


def save_collection_dotplot(
    full_results: pd.DataFrame,
    collection: str,
    selected_states: list[str],
    state_palette: dict[str, str],
    state_support: pd.DataFrame,
    max_pathways: int,
    fdr_threshold: float,
    path: Path,
    dpi: int,
) -> bool:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    collection_results = full_results.loc[
        full_results["collection"].eq(collection)
    ].copy()
    pathways = select_pathways_for_dotplot(collection_results, max_pathways)
    if not pathways:
        return False
    selected = collection_results.loc[
        collection_results["pathway_name"].isin(pathways)
        & collection_results["gsea_status"].eq("tested")
    ].copy()
    label_map = (
        selected.drop_duplicates("pathway_name")
        .set_index("pathway_name")["pathway_label"]
        .to_dict()
    )
    pathway_order = list(reversed(pathways))
    state_pos = {state: index for index, state in enumerate(selected_states)}
    pathway_pos = {name: index for index, name in enumerate(pathway_order)}
    selected["x"] = selected["LEC_state"].map(state_pos)
    selected["y"] = selected["pathway_name"].map(pathway_pos)
    selected["minus_log10_FDR"] = -np.log10(
        pd.to_numeric(selected["FDR_q_value"], errors="coerce").clip(lower=1e-300)
    )
    selected["point_size"] = 16 + 30 * selected["minus_log10_FDR"].clip(upper=10)
    max_abs_nes = float(np.nanmax(np.abs(selected["NES"])))
    max_abs_nes = max(max_abs_nes, 0.1)

    height = max(6.0, 0.43 * len(pathway_order) + 2.8)
    width = max(12.0, 0.72 * len(selected_states) + 6.5)
    fig, ax = plt.subplots(figsize=(width, height))
    scatter = ax.scatter(
        selected["x"],
        selected["y"],
        s=selected["point_size"],
        c=selected["NES"],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-max_abs_nes, vcenter=0, vmax=max_abs_nes),
        edgecolors=np.where(
            selected["FDR_q_value"].le(fdr_threshold), "#111111", "#bbbbbb"
        ),
        linewidths=np.where(selected["FDR_q_value"].le(fdr_threshold), 0.8, 0.3),
        alpha=0.9,
    )
    support_index = state_support.set_index("LEC_state")
    ax.set_xticks(range(len(selected_states)))
    ax.set_xticklabels(
        [state_plot_label(state, support_index) for state in selected_states],
        rotation=45,
        ha="right",
    )
    for tick, state in zip(ax.get_xticklabels(), selected_states):
        tick.set_color(state_palette[state])
        tick.set_fontweight("bold")
    ax.set_yticks(range(len(pathway_order)))
    ax.set_yticklabels([label_map[name] for name in pathway_order], fontsize=8)
    ax.set_xlim(-0.6, len(selected_states) - 0.4)
    ax.set_ylim(-0.7, len(pathway_order) - 0.3)
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)
    ax.set_title(
        f"{collection.replace('_', ' ')}: top state-enriched pathways\n"
        "Color = NES; size = -log10(FDR); dark outline = passes FDR threshold; "
        "* = donor-dominant state"
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    colorbar.set_label("Normalized enrichment score (NES)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def write_method_notes(
    path: Path,
    args: argparse.Namespace,
    state_support: pd.DataFrame,
    input_info: dict[str, object],
) -> None:
    cautions = state_support.loc[
        state_support["interpretation_flag"].ne("OK"),
        ["LEC_state", "n_cells", "n_donors", "top_donor_fraction", "interpretation_flag"],
    ]
    caution_text = (
        cautions.to_csv(index=False).strip()
        if not cautions.empty
        else "No donor-support warnings."
    )
    text = f"""Step 09 GSEA interpretation notes
=================================

Question answered
-----------------
Which pathways distinguish each combined NF LEC state from all other LEC states?

Step 08 source
--------------
Step 08 script: {input_info.get('step08_script_version', 'unknown')}
Leiden resolution: {input_info.get('leiden_resolution', 'unknown')}
Number of states: {input_info.get('n_states', len(state_support))}
State labels are read from the current Step 08 h5ad and are not reused from an
older resolution.

Ranking
-------
Source: Step 08 all-gene Wilcoxon state-vs-rest ranking from unscaled adata.raw.
Ranking metric: {args.ranking_metric}
Exact ties are ordered reproducibly by log fold-change, adjusted p-value, and
gene name, then separated by a sub-gap numerical perturbation that cannot
cross any distinct original ranking score. The ranking QC table records its
maximum size.
Harmony was used only for clustering/embedding in Step 08 and is not used here.

Collections
-----------
Hallmark, GO Biological Process, KEGG Medicus, and Reactome from local
MSigDB v2026.1.Hs GMT files. Matching official JSON metadata supplies exact
source identifiers. If exactSource is empty, biological_id_or_name uses the
pathway name.

GSEA settings
-------------
Permutations: {args.permutation_num}
Minimum overlapping genes: {args.min_size}
Maximum overlapping genes: {args.max_size}
Seed: {args.seed}
FDR threshold: {args.fdr_threshold}
FDR values are calculated within each state/collection GSEA run.

Direction
---------
Positive NES: enriched toward genes higher in the named state.
Negative NES: enriched toward genes higher in the other LEC states.

Donor caveat
------------
The underlying marker ranking treats cells as observations, so these results
are descriptive state annotations, not donor-level inferential tests. Do not
interpret a pathway as an aging effect from this analysis. A future aging
analysis should use donors as biological replicates, preferably state-specific
pseudobulk rankings followed by GSEA.

States requiring extra caution
------------------------------
{caution_text}
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.min_size < 1 or args.max_size < args.min_size:
        raise ValueError("Require 1 <= min_size <= max_size.")
    if args.permutation_num < 1:
        raise ValueError("permutation_num must be at least 1.")
    if not 0 < args.fdr_threshold <= 1:
        raise ValueError("fdr_threshold must be in (0, 1].")

    project_dir = find_project_dir()
    h5ad_path = resolve_path(args.h5ad, project_dir)
    ranking_path = resolve_path(args.ranking_csv, project_dir)
    state_qc_path = resolve_path(args.state_qc_csv, project_dir)
    palette_path = resolve_path(args.state_palette_csv, project_dir)
    gene_sets_dir = resolve_path(args.gene_sets_dir, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    figures_dir = output_root / "figures"
    primary_output = output_root / "09_GSEA_all_pathways_full.csv"
    if primary_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Primary output already exists: {primary_output}\n"
            "Use --overwrite only when you intend to replace Step 09 results."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project directory: {project_dir}")
    print(f"Step 08 h5ad: {h5ad_path}")
    obs, available_states, input_info = read_step08_metadata(h5ad_path, args.state_key)
    selected_states = select_states(args.states, available_states)
    print(f"Selected states ({len(selected_states)}): {', '.join(selected_states)}")
    print(
        f"Validated {input_info['n_cells']:,} NF AP/LV LECs; "
        f"raw genes={input_info['n_genes_raw']:,}."
    )

    state_support = load_state_support(
        obs, args.state_key, selected_states, state_qc_path
    )
    state_palette = load_state_palette(palette_path, selected_states)
    ranking_full, rankings, ranking_qc = load_rankings(
        ranking_path, selected_states, args.ranking_metric
    )
    ranked_gene_sets = [set(frame["names"]) for frame in rankings.values()]
    ranked_genes = set.intersection(*ranked_gene_sets)
    if any(genes != ranked_genes for genes in ranked_gene_sets):
        warnings.warn(
            "Ranked gene universes differ between states; pathway overlap uses "
            "their intersection while each GSEA run uses its complete state ranking."
        )
    print(f"Ranked gene universe shared by states: {len(ranked_genes):,}")

    inventory, gene_sets = build_pathway_inventory(
        gene_sets_dir,
        args.collections,
        ranked_genes,
        args.min_size,
        args.max_size,
    )
    print(
        f"Loaded {len(inventory):,} pathways; "
        f"{inventory['size_filter_status'].eq('eligible').sum():,} pass the size filter."
    )

    inventory.to_csv(output_root / "09_pathway_inventory_full.csv", index=False)
    state_support.to_csv(output_root / "09_state_support_QC.csv", index=False)
    ranking_qc.to_csv(output_root / "09_state_ranking_QC.csv", index=False)
    ranking_export_columns = [
        column
        for column in [
            "LEC_state", "rank", "names", "scores", "logfoldchanges", "pvals",
            "pvals_adj", "pct_nz_group", "pct_nz_reference",
            "gsea_rank_metric_original", "gsea_rank_metric",
        ]
        if column in ranking_full.columns
    ]
    ranking_full[ranking_export_columns].to_csv(
        output_root / "09_state_ranking_used.csv.gz", index=False, compression="gzip"
    )

    full_results, run_qc = run_all_gsea(
        rankings,
        selected_states,
        args.collections,
        inventory,
        gene_sets,
        state_support,
        state_palette,
        args,
    )
    state_order = {state: index for index, state in enumerate(selected_states)}
    collection_order = {
        collection: index for index, collection in enumerate(args.collections)
    }
    full_results["_state_order"] = full_results["LEC_state"].map(state_order)
    full_results["_collection_order"] = full_results["collection"].map(
        collection_order
    )
    full_results = full_results.sort_values(
        ["_state_order", "_collection_order", "FDR_q_value", "pathway_name"],
        na_position="last",
        kind="mergesort",
    ).drop(columns=["_state_order", "_collection_order"])
    full_results.to_csv(primary_output, index=False)

    tested = full_results.loc[full_results["gsea_status"].eq("tested")].copy()
    tested.to_csv(output_root / "09_GSEA_tested_pathways.csv", index=False)
    significant = tested.loc[tested["significant_at_FDR_threshold"]].copy()
    significant.to_csv(output_root / "09_GSEA_significant_FDR05.csv", index=False)
    top = make_top_pathway_table(tested, args.top_per_direction)
    top.to_csv(output_root / "09_GSEA_top_pathways_per_state.csv", index=False)
    summary = make_collection_summary(full_results, selected_states, args.collections)
    summary.to_csv(output_root / "09_GSEA_state_collection_summary.csv", index=False)
    run_qc.to_csv(output_root / "09_GSEA_run_QC.csv", index=False)

    figure_rows = []
    if not args.skip_figures:
        count_path = figures_dir / "09_significant_pathway_counts_by_state.png"
        save_significant_count_figure(
            summary,
            selected_states,
            args.collections,
            state_palette,
            state_support,
            count_path,
            args.dpi,
        )
        figure_rows.append(
            {
                "figure": count_path.name,
                "description": "FDR-significant pathway count per state and collection",
            }
        )
        for collection in args.collections:
            path = figures_dir / f"09_{collection}_top_pathways_dotplot.png"
            created = save_collection_dotplot(
                full_results,
                collection,
                selected_states,
                state_palette,
                state_support,
                args.top_pathways_figure,
                args.fdr_threshold,
                path,
                args.dpi,
            )
            if created:
                figure_rows.append(
                    {
                        "figure": path.name,
                        "description": (
                            f"Top {collection} pathways across combined LEC states"
                        ),
                    }
                )
    pd.DataFrame(figure_rows, columns=["figure", "description"]).to_csv(
        output_root / "09_figure_manifest.csv", index=False
    )

    write_method_notes(
        output_root / "09_method_and_interpretation_notes.txt",
        args,
        state_support,
        input_info,
    )
    config = {
        "script_version": SCRIPT_VERSION,
        "project_dir": str(project_dir),
        "h5ad": str(h5ad_path),
        "ranking_csv": str(ranking_path),
        "state_qc_csv": str(state_qc_path),
        "state_palette_csv": str(palette_path),
        "gene_sets_dir": str(gene_sets_dir),
        "output_root": str(output_root),
        "state_key": args.state_key,
        "states": selected_states,
        "collections": args.collections,
        "ranking_metric": args.ranking_metric,
        "rank_method": "Step08_cell_level_Wilcoxon_state_vs_rest",
        "min_size": args.min_size,
        "max_size": args.max_size,
        "permutation_num": args.permutation_num,
        "threads": args.threads,
        "seed": args.seed,
        "fdr_threshold": args.fdr_threshold,
        "top_per_direction": args.top_per_direction,
        "top_pathways_figure": args.top_pathways_figure,
        "figures_enabled": not args.skip_figures,
        "input_info": input_info,
    }
    (output_root / "09_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nStep 09 complete.")
    print(f"Full state x pathway table: {primary_output}")
    print(f"Tested pathways: {len(tested):,}")
    print(
        f"Significant rows at FDR <= {args.fdr_threshold}: {len(significant):,}"
    )
    cautions = state_support.loc[
        state_support["interpretation_flag"].ne("OK"), "LEC_state"
    ].tolist()
    if cautions:
        print(
            "Interpret with extra donor caution: " + ", ".join(cautions)
        )
    print(
        "This state-vs-rest result supports state annotation; it does not test "
        "age-quartile effects."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
