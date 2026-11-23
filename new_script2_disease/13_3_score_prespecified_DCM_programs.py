#!/usr/bin/env python3
r"""
Step 13.3: prespecified DCM program validation in GSE271676 GeoMx AOIs.

Step 13.2 selected primary LV vessel AOIs in which at least two of CCL21,
LYVE1, PDPN, FLT4, and PROX1 were above the AOI-specific LOQ.  This step tests
the mentor-specified disease programs in those selected AOIs without
pretending that GeoMx AOIs are single cells or purified LECs.

The primary biological replicate is PID and AOIs from the same donor are first
averaged.  The selected cohort contains only two Control donors and therefore
does not meet the prespecified minimum of three donors per group.  All group
effects are consequently descriptive: no permutation P values, bootstrap
intervals, or AOI mixed-model P values are produced.

Program scores use log2(primary-cohort Q3 CPM + 1).  Each gene is robustly
standardized using donor means, then gene z scores are averaged within a
program.  SEMA3D and PDGFC are documented but excluded because they failed the
authors' 1% probe-QC rule.  CCL21 is both a selection marker and a homeostatic
endpoint, so its endpoint result is explicitly flagged as selection-overlapping.

These analyses validate LEC-like expression programs in spatially indexed,
CD31-enriched vessel AOIs.  They do not prove a purified LEC state, regulon
activity, ligand-receptor directionality, or causality.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import warnings
import zlib
from collections import OrderedDict
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind
from statsmodels.regression.mixed_linear_model import MixedLM


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_VERSION = "13_3_GSE271676_two_of_five_descriptive_DCM_programs_2026_08_28"
DEFAULT_INPUT = (
    "new_result2_spatial_disease/13_2_disease_LEC_reference/"
    "13_2_GSE271676_two_of_five_LEC_selected.h5ad"
)
DEFAULT_TARGET_REGISTRY = (
    "new_result2_spatial_disease/13_1_spatial_disease_QC/"
    "13_1_DCM_target_registry.csv"
)
DEFAULT_OUTPUT = (
    "new_result2_spatial_disease/13_3_prespecified_DCM_program_validation"
)
EXPRESSION_LAYER = "log2_primary_vessel_upper_quartile_CPM_plus1"

PRIMARY_PROGRAMS = OrderedDict(
    [
        ("homeostatic_LEC", ["MMRN1", "CCL21"]),
        ("disease_expanded_State0_like", ["ITGA9", "NRP2", "NPNT", "SEMA3D"]),
        ("disease_expanded_State3_like", ["PLSCR4", "CRIM1", "COLEC12", "PDGFC"]),
        ("activated_vascular_State5_like", ["FLT1", "ENG", "EPAS1"]),
    ]
)
EXPECTED_DIRECTIONS = {
    "homeostatic_LEC": "decrease_in_DCMP",
    "disease_expanded_State0_like": "increase_in_DCMP",
    "disease_expanded_State3_like": "increase_in_DCMP",
    "activated_vascular_State5_like": "increase_in_DCMP",
}
SECONDARY_NICHE_PANELS = OrderedDict(
    [
        ("BMP4_ligand_context", ["BMP4"]),
        ("BMPR_receptor_context", ["BMPR1A", "BMPR2"]),
        ("neuropilin_receptor_context", ["NRP1", "NRP2"]),
    ]
)
PROGRAM_COLORS = {
    "homeostatic_LEC": "#0072B2",
    "disease_expanded_State0_like": "#D55E00",
    "disease_expanded_State3_like": "#CC79A7",
    "activated_vascular_State5_like": "#009E73",
}
GROUP_ORDER = ["Control", "DCMP"]


def find_project_dir() -> Path:
    """Use this release root; never discover an adjacent historical project."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test prespecified DCM programs in GSE271676 GeoMx vessel AOIs."
    )
    parser.add_argument("--input_h5ad", default=DEFAULT_INPUT)
    parser.add_argument("--target_registry", default=DEFAULT_TARGET_REGISTRY)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum_donors_per_group", type=int, default=3)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--skip_h5ad", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype("string").fillna("").str.lower().isin({"true", "1", "yes", "y"})


def dense_layer(adata: ad.AnnData, layer: str) -> pd.DataFrame:
    if layer not in adata.layers:
        raise KeyError(f"Required layer is absent: {layer}")
    matrix = adata.layers[layer]
    values = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    if values.shape != adata.shape:
        raise ValueError(f"Layer {layer} has shape {values.shape}, expected {adata.shape}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Layer {layer} contains invalid values")
    return pd.DataFrame(
        values.astype(float),
        index=pd.Index(adata.obs_names.astype(str), name="aoi_id"),
        columns=pd.Index(adata.var_names.astype(str), name="gene"),
    )


def donor_balanced_robust_z(
    expression: pd.DataFrame, donor: pd.Series, clip: float = 5.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize AOIs using center/scale estimated from equal-weight donor means."""
    donor = donor.reindex(expression.index).astype(str)
    donor_mean = expression.groupby(donor, observed=True).mean()
    center = donor_mean.median(axis=0)
    mad = donor_mean.sub(center, axis=1).abs().median(axis=0)
    robust_scale = mad / 0.6744897501960817
    fallback_sd = donor_mean.std(axis=0, ddof=1)
    scale = robust_scale.where(robust_scale > 0, fallback_sd)
    scale = scale.where(scale > 0, 1.0).fillna(1.0)
    z = expression.sub(center, axis=1).div(scale, axis=1).clip(-clip, clip)
    parameters = pd.DataFrame(
        {
            "gene": expression.columns,
            "donor_balanced_median": center.reindex(expression.columns).to_numpy(),
            "donor_balanced_MAD_scale": robust_scale.reindex(expression.columns).to_numpy(),
            "scale_used": scale.reindex(expression.columns).to_numpy(),
            "scale_fell_back_to_SD_or_one": (
                robust_scale.reindex(expression.columns).fillna(0).to_numpy() <= 0
            ),
        }
    )
    return z, parameters


def module_score(z: pd.DataFrame, genes: list[str]) -> tuple[pd.Series, list[str]]:
    used = [gene for gene in genes if gene in z.columns]
    if not used:
        raise ValueError(f"No usable genes for panel: {genes}")
    return z[used].mean(axis=1), used


def bh_fdr(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[order] = np.minimum(adjusted, 1.0)
    return result


def hedges_g(control: np.ndarray, dcmp: np.ndarray) -> float:
    n0, n1 = len(control), len(dcmp)
    if n0 < 2 or n1 < 2:
        return float("nan")
    pooled_num = (n0 - 1) * np.var(control, ddof=1) + (n1 - 1) * np.var(dcmp, ddof=1)
    pooled_den = n0 + n1 - 2
    if pooled_den <= 0 or pooled_num <= 0:
        return float("nan")
    pooled_sd = math.sqrt(pooled_num / pooled_den)
    correction = 1.0 - 3.0 / (4.0 * (n0 + n1) - 9.0)
    return float(correction * (np.mean(dcmp) - np.mean(control)) / pooled_sd)


def exact_label_permutation(
    values: np.ndarray, labels: np.ndarray, expected_direction: str
) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=str)
    n_dcmp = int(np.sum(labels == "DCMP"))
    observed = float(values[labels == "DCMP"].mean() - values[labels == "Control"].mean())
    differences = []
    indices = np.arange(len(values))
    for dcmp_indices in itertools.combinations(indices, n_dcmp):
        case = np.zeros(len(values), dtype=bool)
        case[list(dcmp_indices)] = True
        differences.append(float(values[case].mean() - values[~case].mean()))
    permuted = np.asarray(differences)
    tolerance = 1e-12
    p_two = float(np.mean(np.abs(permuted) >= abs(observed) - tolerance))
    if expected_direction == "increase_in_DCMP":
        p_direction = float(np.mean(permuted >= observed - tolerance))
    elif expected_direction == "decrease_in_DCMP":
        p_direction = float(np.mean(permuted <= observed + tolerance))
    else:
        p_direction = float("nan")
    return p_two, p_direction, len(permuted)


def bootstrap_mean_difference(
    control: np.ndarray, dcmp: np.ndarray, iterations: int, seed: int
) -> tuple[float, float]:
    if iterations <= 0 or len(control) == 0 or len(dcmp) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)
    for index in range(iterations):
        c = rng.choice(control, size=len(control), replace=True)
        d = rng.choice(dcmp, size=len(dcmp), replace=True)
        differences[index] = d.mean() - c.mean()
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def one_test(
    donor_frame: pd.DataFrame,
    value_column: str,
    expected_direction: str,
    minimum_donors: int,
    bootstrap_iterations: int,
    seed: int,
    allow_inference: bool = True,
) -> dict[str, object]:
    frame = donor_frame[["PID", "Clinical_phenotype_LV", value_column]].dropna().copy()
    control = frame.loc[frame["Clinical_phenotype_LV"].eq("Control"), value_column].to_numpy(float)
    dcmp = frame.loc[frame["Clinical_phenotype_LV"].eq("DCMP"), value_column].to_numpy(float)
    difference = float(np.mean(dcmp) - np.mean(control)) if len(control) and len(dcmp) else float("nan")
    threshold_met = len(control) >= minimum_donors and len(dcmp) >= minimum_donors
    direction_concordant = (
        difference < 0 if expected_direction == "decrease_in_DCMP" else difference > 0
        if expected_direction == "increase_in_DCMP" else np.nan
    )
    result: dict[str, object] = {
        "n_Control_donors": len(control),
        "n_DCMP_donors": len(dcmp),
        "mean_Control": float(np.mean(control)) if len(control) else np.nan,
        "mean_DCMP": float(np.mean(dcmp)) if len(dcmp) else np.nan,
        "difference_DCMP_minus_Control": difference,
        "hedges_g": hedges_g(control, dcmp),
        "expected_direction": expected_direction,
        "direction_concordant": direction_concordant,
        "minimum_donor_threshold_met": threshold_met,
        "inference_status": (
            "eligible_exact_donor_inference" if threshold_met and allow_inference
            else "descriptive_only"
        ),
    }
    if not (threshold_met and allow_inference):
        result.update(
            {
                "bootstrap_95CI_low": np.nan,
                "bootstrap_95CI_high": np.nan,
                "exact_permutation_p_two_sided": np.nan,
                "exact_permutation_p_expected_direction": np.nan,
                "n_exact_label_permutations": np.nan,
                "Welch_t_p_two_sided": np.nan,
                "Mann_Whitney_p_two_sided": np.nan,
            }
        )
        return result
    low, high = bootstrap_mean_difference(
        control, dcmp, bootstrap_iterations, seed
    )
    p_two, p_direction, n_permutations = exact_label_permutation(
        frame[value_column].to_numpy(float),
        frame["Clinical_phenotype_LV"].to_numpy(str),
        expected_direction,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        welch = ttest_ind(dcmp, control, equal_var=False, nan_policy="omit")
        mann_whitney = mannwhitneyu(dcmp, control, alternative="two-sided", method="auto")
    result.update(
        {
            "bootstrap_95CI_low": low,
            "bootstrap_95CI_high": high,
            "exact_permutation_p_two_sided": p_two,
            "exact_permutation_p_expected_direction": p_direction,
            "n_exact_label_permutations": n_permutations,
            "Welch_t_p_two_sided": float(welch.pvalue),
            "Mann_Whitney_p_two_sided": float(mann_whitney.pvalue),
        }
    )
    return result


def donor_table(aoi_scores: pd.DataFrame, score_columns: list[str]) -> pd.DataFrame:
    aggregations: dict[str, str] = {column: "mean" for column in score_columns}
    for column in [
        "LEC_marker_score_five",
        "n_LEC_markers_above_LOQ",
    ]:
        if column in aoi_scores.columns:
            aggregations[column] = "mean"
    result = (
        aoi_scores.groupby(["PID", "Clinical_phenotype_LV"], observed=True)
        .agg(aggregations)
        .reset_index()
    )
    counts = aoi_scores.groupby("PID", observed=True).size().rename("n_AOIs").reset_index()
    return result.merge(counts, on="PID", how="left")


def sensitivity_masks(obs: pd.DataFrame) -> OrderedDict[str, pd.Series]:
    strict_controls = obs["Clinical_phenotype_LV"].eq("DCMP") | as_bool(
        obs["analysis_strict_noncardiac_control"]
    )
    return OrderedDict(
        [
            ("primary_two_of_five", pd.Series(True, index=obs.index)),
            ("strict_anatomical_origin_LV", as_bool(obs["analysis_strict_origin_LV_vessel"])),
            ("strict_noncardiac_controls", strict_controls),
            ("pre_LVAD_only", as_bool(obs["analysis_pre_LVAD_sensitivity"])),
            ("exclude_author_candidate_EC_outliers", ~as_bool(obs["author_code_candidate_EC_outlier"])),
            ("stringent_three_of_five", as_bool(obs["LEC_stringent_three_of_five"])),
        ]
    )


def mixed_model_sensitivity(
    frame: pd.DataFrame, score_column: str
) -> dict[str, object]:
    model_frame = frame[
        ["PID", "Clinical_phenotype_LV", "TMA.block", score_column]
    ].dropna().copy()
    model_frame["DCMP"] = model_frame["Clinical_phenotype_LV"].eq("DCMP").astype(float)
    block = pd.get_dummies(
        model_frame["TMA.block"].astype(str), prefix="TMA_block", drop_first=True, dtype=float
    )
    exog = pd.concat(
        [
            pd.Series(1.0, index=model_frame.index, name="Intercept"),
            model_frame[["DCMP"]],
            block,
        ],
        axis=1,
    )
    result: dict[str, object] = {
        "program": score_column.replace("program_score__", ""),
        "n_AOIs": len(model_frame),
        "n_donors": model_frame["PID"].nunique(),
        "fixed_effects": ";".join(exog.columns),
        "model": "AOI_score ~ DCMP + TMA_block + (1|PID)",
    }
    donor_counts = model_frame.groupby("Clinical_phenotype_LV", observed=True)["PID"].nunique()
    if any(int(donor_counts.get(group, 0)) < 3 for group in GROUP_ORDER):
        result.update(
            {
                "status": "not_fit_insufficient_independent_donors",
                "DCMP_coefficient": np.nan,
                "DCMP_SE": np.nan,
                "DCMP_p": np.nan,
                "random_intercept_variance": np.nan,
                "cluster_robust_OLS_model": "not_fit_insufficient_independent_donors",
                "cluster_robust_OLS_DCMP_coefficient": np.nan,
                "cluster_robust_OLS_DCMP_SE": np.nan,
                "cluster_robust_OLS_DCMP_p": np.nan,
                "warnings": "At least three independent donors per group are required.",
            }
        )
        return result
    if np.linalg.matrix_rank(exog.to_numpy(float)) < exog.shape[1]:
        result.update({"status": "not_fit_rank_deficient", "DCMP_coefficient": np.nan, "DCMP_SE": np.nan, "DCMP_p": np.nan})
        return result
    try:
        cluster_fit = sm.OLS(
            model_frame[score_column].astype(float), exog.astype(float)
        ).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": model_frame["PID"].astype(str),
                "use_correction": True,
            },
        )
        result.update(
            {
                "cluster_robust_OLS_model": "AOI_score ~ DCMP + TMA_block; SE clustered by PID",
                "cluster_robust_OLS_DCMP_coefficient": float(cluster_fit.params.get("DCMP", np.nan)),
                "cluster_robust_OLS_DCMP_SE": float(cluster_fit.bse.get("DCMP", np.nan)),
                "cluster_robust_OLS_DCMP_p": float(cluster_fit.pvalues.get("DCMP", np.nan)),
            }
        )
    except Exception as exc:
        result.update(
            {
                "cluster_robust_OLS_model": "fit_failed",
                "cluster_robust_OLS_DCMP_coefficient": np.nan,
                "cluster_robust_OLS_DCMP_SE": np.nan,
                "cluster_robust_OLS_DCMP_p": np.nan,
                "cluster_robust_OLS_warning": f"{type(exc).__name__}: {exc}",
            }
        )
    try:
        fit = None
        caught_messages: list[str] = []
        failures: list[str] = []
        for method in ["lbfgs", "powell"]:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    candidate = MixedLM(
                        model_frame[score_column].astype(float),
                        exog.astype(float),
                        groups=model_frame["PID"].astype(str),
                    ).fit(reml=True, method=method, maxiter=2000, disp=False)
                caught_messages.extend(str(item.message) for item in caught)
                if bool(candidate.converged):
                    fit = candidate
                    fit_method = method
                    break
                failures.append(f"{method}:not_converged")
            except Exception as exc:
                failures.append(f"{method}:{type(exc).__name__}:{exc}")
        if fit is None:
            raise RuntimeError("; ".join(failures))
        random_variance = float(fit.cov_re.iloc[0, 0])
        boundary = random_variance < 1e-6
        result.update(
            {
                "status": (
                    "converged_boundary_random_variance_interpret_cautiously"
                    if boundary else "converged"
                ),
                "mixed_model_fit_method": fit_method,
                "DCMP_coefficient": float(fit.params.get("DCMP", np.nan)),
                "DCMP_SE": float(fit.bse.get("DCMP", np.nan)),
                "DCMP_p": float(fit.pvalues.get("DCMP", np.nan)),
                "random_intercept_variance": random_variance,
                "warnings": " | ".join(sorted(set(caught_messages + failures))),
            }
        )
    except Exception as exc:  # model failure is reported, never fatal to primary analysis
        result.update(
            {
                "status": "fit_failed",
                "DCMP_coefficient": np.nan,
                "DCMP_SE": np.nan,
                "DCMP_p": np.nan,
                "random_intercept_variance": np.nan,
                "warnings": f"{type(exc).__name__}: {exc}",
            }
        )
    return result


def plot_donor_scores(
    donor_scores: pd.DataFrame, program_columns: dict[str, str], output: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    rng = np.random.default_rng(1303)
    for axis, (program, column) in zip(axes.flat, program_columns.items()):
        arrays = [
            donor_scores.loc[donor_scores["Clinical_phenotype_LV"].eq(group), column].to_numpy(float)
            for group in GROUP_ORDER
        ]
        axis.boxplot(arrays, positions=[0, 1], widths=0.5, showfliers=False, patch_artist=True,
                     boxprops={"facecolor": "#ECECEC", "edgecolor": "#555555"},
                     medianprops={"color": "black", "linewidth": 1.5})
        for position, (group, values) in enumerate(zip(GROUP_ORDER, arrays)):
            jitter = rng.uniform(-0.09, 0.09, len(values))
            axis.scatter(position + jitter, values, s=48, color=PROGRAM_COLORS[program],
                         edgecolor="white", linewidth=0.6, alpha=0.9, zorder=3)
        axis.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
        axis.set_xticks([0, 1], GROUP_ORDER)
        axis.set_ylabel("Donor-balanced program score")
        axis.set_title(program.replace("_", " "), fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("GSE271676 LV vessel AOIs: equal-weight donor program scores", fontsize=13)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_effects(tests: pd.DataFrame, output: Path, dpi: int) -> None:
    frame = tests.set_index("program").loc[list(PRIMARY_PROGRAMS)].reset_index()
    y = np.arange(len(frame))[::-1]
    fig, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for position, row in zip(y, frame.itertuples(index=False)):
        color = PROGRAM_COLORS[row.program]
        if np.isfinite(row.bootstrap_95CI_low) and np.isfinite(row.bootstrap_95CI_high):
            axis.plot(
                [row.bootstrap_95CI_low, row.bootstrap_95CI_high],
                [position, position],
                color=color,
                linewidth=2,
            )
        axis.scatter(row.difference_DCMP_minus_Control, position, s=70, color=color, edgecolor="white", zorder=3)
        axis.text(
            frame["difference_DCMP_minus_Control"].max() + 0.08,
            position,
            f"descriptive only; {row.n_Control_donors} vs {row.n_DCMP_donors} donors",
            va="center", fontsize=8,
        )
    axis.axvline(0, color="black", linewidth=0.9)
    axis.set_yticks(y, [name.replace("_", " ") for name in frame["program"]])
    axis.set_xlabel("Mean difference: DCMP - Control (program-score units)")
    axis.set_title("Descriptive donor-level DCM effects (inference not permitted)")
    axis.spines[["top", "right"]].set_visible(False)
    right = float(frame["difference_DCMP_minus_Control"].max())
    left = float(frame["difference_DCMP_minus_Control"].min())
    axis.set_xlim(left - 0.15, right + max(0.9, (right - left) * 0.75))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_target_heatmap(
    donor_gene_z: pd.DataFrame, donor_meta: pd.DataFrame, genes: list[str], output: Path, dpi: int
) -> None:
    order = donor_meta.sort_values(["Clinical_phenotype_LV", "PID"], key=lambda s: s.map({"Control": 0, "DCMP": 1}).fillna(s) if s.name == "Clinical_phenotype_LV" else s).index
    matrix = donor_gene_z.reindex(index=order, columns=genes)
    fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    image = axis.imshow(matrix.T.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    axis.set_yticks(np.arange(len(genes)), genes)
    labels = [f"{pid}\n{donor_meta.loc[pid, 'Clinical_phenotype_LV']}" for pid in matrix.index]
    axis.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    axis.set_title("Mentor target expression (donor means; robust gene z scores)")
    fig.colorbar(image, ax=axis, shrink=0.75, label="Donor-balanced gene z")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_heatmap(tests: pd.DataFrame, output: Path, dpi: int) -> None:
    matrix = tests.pivot(index="analysis_set", columns="program", values="difference_DCMP_minus_Control")
    matrix = matrix.reindex(columns=list(PRIMARY_PROGRAMS))
    fig, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(matrix.to_numpy())))
    vmax = max(vmax, 0.1)
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axis.set_xticks(np.arange(matrix.shape[1]), [x.replace("_", " ") for x in matrix.columns], rotation=35, ha="right")
    axis.set_yticks(np.arange(matrix.shape[0]), [x.replace("_", " ") for x in matrix.index])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                          color="white" if abs(value) > 0.55 * vmax else "black")
    axis.set_title("DCMP - Control program effects across prespecified cohorts")
    fig.colorbar(image, ax=axis, shrink=0.8, label="Program-score difference")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    project_dir = find_project_dir()
    input_h5ad = resolve_path(args.input_h5ad, project_dir)
    target_registry_path = resolve_path(args.target_registry, project_dir)
    output_root = resolve_path(args.output_root, project_dir)
    validate_file(input_h5ad, "Step 13.2 H5AD")
    validate_file(target_registry_path, "Step 13.1 target registry")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is nonempty; rerun with --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    figures_dir.mkdir(exist_ok=True)

    adata = ad.read_h5ad(input_h5ad)
    required_obs = [
        "PID", "Clinical_phenotype_LV", "TMA.block",
        "analysis_strict_origin_LV_vessel", "analysis_strict_noncardiac_control",
        "analysis_pre_LVAD_sensitivity", "author_code_candidate_EC_outlier",
        "LEC_selected_two_of_five", "LEC_stringent_three_of_five",
        "n_LEC_markers_above_LOQ", "LEC_markers_above_LOQ",
        "LEC_marker_score_five", "LEC_selection_tier",
        "CCL21_selection_endpoint_overlap",
    ]
    missing_obs = [column for column in required_obs if column not in adata.obs]
    if missing_obs:
        raise KeyError(f"Input H5AD lacks required obs columns: {missing_obs}")
    phenotype = adata.obs["Clinical_phenotype_LV"].astype(str)
    if set(phenotype) - {"Control", "DCMP"}:
        raise ValueError("Input contains phenotypes outside Control/DCMP")
    donor_conditions = adata.obs.assign(_condition=phenotype).groupby("PID", observed=True)["_condition"].nunique()
    if (donor_conditions > 1).any():
        raise ValueError("At least one PID occurs in both disease groups")

    expression = dense_layer(adata, EXPRESSION_LAYER)
    gene_z, z_parameters = donor_balanced_robust_z(expression, adata.obs["PID"])
    target_registry = pd.read_csv(target_registry_path)

    program_columns: dict[str, str] = {}
    registry_rows: list[dict[str, object]] = []
    all_panels = OrderedDict([*PRIMARY_PROGRAMS.items(), *SECONDARY_NICHE_PANELS.items()])
    for program, planned in all_panels.items():
        score, used = module_score(gene_z, planned)
        column = f"program_score__{program}"
        adata.obs[column] = score.reindex(adata.obs_names).to_numpy()
        program_columns[program] = column
        excluded = [gene for gene in planned if gene not in used]
        reasons = {}
        for gene in excluded:
            row = target_registry.loc[target_registry["gene"].eq(gene)]
            reasons[gene] = row.iloc[0]["primary_analysis_status"] if not row.empty else "not_in_post_QC_matrix"
        registry_rows.append(
            {
                "program": program,
                "analysis_tier": "primary_DCM_program" if program in PRIMARY_PROGRAMS else "secondary_niche_context",
                "expected_direction": EXPECTED_DIRECTIONS.get(program, "not_prespecified"),
                "planned_genes": ";".join(planned),
                "genes_used": ";".join(used),
                "n_planned_genes": len(planned),
                "n_genes_used": len(used),
                "excluded_genes": ";".join(excluded),
                "exclusion_reasons": ";".join(f"{gene}:{reason}" for gene, reason in reasons.items()),
                "score_method": "mean donor-balanced robust gene z; z clipped to [-5,5]",
            }
        )
    program_registry = pd.DataFrame(registry_rows)

    primary_score_columns = {program: program_columns[program] for program in PRIMARY_PROGRAMS}
    all_score_columns = list(program_columns.values())
    aoi_export_columns = list(dict.fromkeys(required_obs + [
        "aoi_id", "Scan.name", "ROI..label.", "Segment..Name..Label.", "After_LVAD",
        "Origin_Structure", "control_definition", *all_score_columns,
    ]))
    aoi_export_columns = [column for column in aoi_export_columns if column in adata.obs]
    aoi_scores = adata.obs[aoi_export_columns].copy()
    aoi_scores.insert(0, "obs_id", adata.obs_names.astype(str))
    if "aoi_id" not in aoi_scores:
        aoi_scores["aoi_id"] = adata.obs_names.astype(str)

    donor_scores = donor_table(aoi_scores, all_score_columns)
    donor_meta = aoi_scores.groupby("PID", observed=True)["Clinical_phenotype_LV"].first().to_frame()
    donor_expression = expression.assign(PID=adata.obs["PID"].astype(str).to_numpy()).groupby("PID", observed=True).mean()
    retained_primary_genes = list(dict.fromkeys(
        gene for genes in PRIMARY_PROGRAMS.values() for gene in genes if gene in expression.columns
    ))
    donor_gene_z = gene_z.assign(PID=adata.obs["PID"].astype(str).to_numpy()).groupby("PID", observed=True).mean()
    donor_gene_export = donor_meta.join(donor_expression[retained_primary_genes].add_prefix("log2_Q3CPM_plus1__")).join(
        donor_gene_z[retained_primary_genes].add_prefix("robust_z__")
    ).reset_index()

    primary_rows = []
    for program, column in primary_score_columns.items():
        seed = args.seed + zlib.crc32(program.encode("utf-8"))
        row = one_test(
            donor_scores, column, EXPECTED_DIRECTIONS[program],
            args.minimum_donors_per_group, args.bootstrap_iterations, seed,
        )
        primary_rows.append({"program": program, "score_column": column, **row})
    primary_tests = pd.DataFrame(primary_rows)
    primary_tests["exact_permutation_FDR_two_sided"] = bh_fdr(
        primary_tests["exact_permutation_p_two_sided"]
    )
    primary_tests["exact_permutation_FDR_expected_direction"] = bh_fdr(
        primary_tests["exact_permutation_p_expected_direction"]
    )
    primary_tests["primary_evidence_flag"] = np.select(
        [
            ~primary_tests["minimum_donor_threshold_met"]
            & primary_tests["direction_concordant"].eq(True),
            ~primary_tests["minimum_donor_threshold_met"]
            & primary_tests["direction_concordant"].eq(False),
            primary_tests["direction_concordant"].eq(True)
            & primary_tests["exact_permutation_FDR_two_sided"].le(0.05),
            primary_tests["direction_concordant"].eq(True),
        ],
        [
            "descriptive_expected_direction_insufficient_donors",
            "descriptive_opposite_direction_insufficient_donors",
            "direction_concordant_and_FDR_lt_0.05",
            "direction_concordant_not_FDR_lt_0.05",
        ],
        default="direction_not_concordant_or_not_estimable",
    )

    sensitivity_rows = []
    masks = sensitivity_masks(adata.obs)
    for analysis_set, mask in masks.items():
        subset_aoi = aoi_scores.loc[mask.to_numpy()].copy()
        subset_donor = donor_table(subset_aoi, list(primary_score_columns.values()))
        counts = subset_donor.groupby("Clinical_phenotype_LV", observed=True)["PID"].nunique()
        eligible = all(int(counts.get(group, 0)) >= args.minimum_donors_per_group for group in GROUP_ORDER)
        for program, column in primary_score_columns.items():
            seed = args.seed + zlib.crc32(f"{analysis_set}|{program}".encode("utf-8"))
            row = one_test(
                subset_donor, column, EXPECTED_DIRECTIONS[program],
                args.minimum_donors_per_group, args.bootstrap_iterations, seed,
                allow_inference=eligible,
            )
            sensitivity_rows.append(
                {"analysis_set": analysis_set, "program": program,
                 "n_AOIs": len(subset_aoi), **row}
            )
    sensitivity_tests = pd.DataFrame(sensitivity_rows)
    sensitivity_tests["exact_permutation_FDR_within_analysis_set"] = (
        sensitivity_tests.groupby("analysis_set", observed=True)["exact_permutation_p_two_sided"]
        .transform(bh_fdr)
    )

    gene_rows = []
    gene_donor_frame = donor_meta.reset_index().merge(
        donor_gene_z[retained_primary_genes].reset_index(), on="PID", how="left"
    )
    for program, planned in PRIMARY_PROGRAMS.items():
        for gene in planned:
            if gene not in retained_primary_genes:
                continue
            seed = args.seed + zlib.crc32(gene.encode("utf-8"))
            row = one_test(
                gene_donor_frame, gene, EXPECTED_DIRECTIONS[program],
                args.minimum_donors_per_group, args.bootstrap_iterations, seed,
            )
            gene_rows.append({"program": program, "gene": gene, **row})
    gene_tests = pd.DataFrame(gene_rows)
    gene_tests["exact_permutation_FDR_two_sided_all_genes"] = bh_fdr(
        gene_tests["exact_permutation_p_two_sided"]
    )

    niche_rows = []
    for program in SECONDARY_NICHE_PANELS:
        column = program_columns[program]
        seed = args.seed + zlib.crc32(program.encode("utf-8"))
        row = one_test(
            donor_scores, column, "not_prespecified",
            args.minimum_donors_per_group, args.bootstrap_iterations, seed,
        )
        niche_rows.append({"niche_panel": program, **row})
    niche_tests = pd.DataFrame(niche_rows)
    niche_tests["exact_permutation_FDR_two_sided"] = bh_fdr(
        niche_tests["exact_permutation_p_two_sided"]
    )

    association_rows = []
    niche_names = list(SECONDARY_NICHE_PANELS)
    association_pairs = []
    for niche in niche_names:
        for program in PRIMARY_PROGRAMS:
            association_pairs.append((niche, program))
    association_pairs.extend(
        [("BMP4_ligand_context", "BMPR_receptor_context"),
         ("BMP4_ligand_context", "neuropilin_receptor_context")]
    )
    for source, target in association_pairs:
        x = donor_scores[program_columns[source]].to_numpy(float)
        y = donor_scores[program_columns[target]].to_numpy(float)
        correlation = spearmanr(x, y, nan_policy="omit")
        association_rows.append(
            {
                "source_score": source, "target_score": target,
                "unit": "PID_equal_weight_donor_mean", "n_donors": len(donor_scores),
                "Spearman_rho": float(correlation.statistic),
                "Spearman_p_two_sided": np.nan,
                "inference_status": "descriptive_only_insufficient_donors",
                "interpretation_limit": "descriptive within-selected-AOI co-expression; not directional ligand-receptor proof",
            }
        )
    associations = pd.DataFrame(association_rows)
    associations["Spearman_FDR_all_pairs"] = bh_fdr(associations["Spearman_p_two_sided"])

    mixed_rows = [
        mixed_model_sensitivity(aoi_scores, column)
        for column in primary_score_columns.values()
    ]
    mixed_models = pd.DataFrame(mixed_rows)
    mixed_models["BH_FDR_across_programs"] = bh_fdr(mixed_models["DCMP_p"])
    mixed_models["cluster_robust_OLS_BH_FDR_across_programs"] = bh_fdr(
        mixed_models["cluster_robust_OLS_DCMP_p"]
    )

    program_registry.to_csv(output_root / "13_3_program_registry.csv", index=False)
    z_parameters.loc[z_parameters["gene"].isin(set(retained_primary_genes + [g for x in SECONDARY_NICHE_PANELS.values() for g in x]))].to_csv(
        output_root / "13_3_target_gene_standardization_parameters.csv", index=False
    )
    aoi_scores.to_csv(output_root / "13_3_AOI_program_scores.csv", index=False)
    donor_scores.to_csv(output_root / "13_3_donor_program_scores.csv", index=False)
    donor_gene_export.to_csv(output_root / "13_3_donor_target_expression.csv", index=False)
    primary_tests.to_csv(output_root / "13_3_primary_program_tests.csv", index=False)
    sensitivity_tests.to_csv(output_root / "13_3_sensitivity_program_tests.csv", index=False)
    gene_tests.to_csv(output_root / "13_3_gene_level_donor_tests.csv", index=False)
    niche_tests.to_csv(output_root / "13_3_secondary_niche_group_tests.csv", index=False)
    associations.to_csv(output_root / "13_3_niche_program_associations.csv", index=False)
    mixed_models.to_csv(output_root / "13_3_AOI_mixed_model_sensitivity.csv", index=False)

    if not args.skip_h5ad:
        adata.uns["step13_3_config_json"] = json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "expression_layer": EXPRESSION_LAYER,
                "primary_unit": "PID",
                "programs": PRIMARY_PROGRAMS,
                "expected_directions": EXPECTED_DIRECTIONS,
            }
        )
        adata.write_h5ad(output_root / "13_3_GSE271676_primary_DCM_program_scores.h5ad", compression="gzip")

    figure_manifest = []
    if not args.skip_figures:
        figure_jobs = [
            ("13_3_donor_program_scores.png", lambda path: plot_donor_scores(donor_scores, primary_score_columns, path, args.dpi), "Primary visualization: each point is one donor."),
            ("13_3_primary_program_effects.png", lambda path: plot_effects(primary_tests, path, args.dpi), "Descriptive DCMP-Control donor effects; no interval or P value because the Control group has two donors."),
            ("13_3_donor_target_heatmap.png", lambda path: plot_target_heatmap(donor_gene_z, donor_meta, retained_primary_genes, path, args.dpi), "Donor-level retained mentor targets; descriptive gene pattern."),
            ("13_3_sensitivity_effect_consistency.png", lambda path: plot_sensitivity_heatmap(sensitivity_tests, path, args.dpi), "Effect direction across prespecified cohort definitions."),
        ]
        for filename, function, role in figure_jobs:
            path = figures_dir / filename
            function(path)
            figure_manifest.append({"figure": str(path.relative_to(output_root)), "role": role})
    pd.DataFrame(figure_manifest).to_csv(output_root / "13_3_figure_manifest.csv", index=False)

    counts = donor_scores.groupby("Clinical_phenotype_LV", observed=True).agg(
        n_donors=("PID", "nunique"), n_AOIs=("n_AOIs", "sum")
    )
    summary_rows: list[dict[str, object]] = [
        {"metric": "script_version", "value": SCRIPT_VERSION},
        {"metric": "input_AOIs", "value": adata.n_obs},
        {"metric": "input_post_probe_QC_genes", "value": adata.n_vars},
        {"metric": "Control_donors", "value": int(counts.loc["Control", "n_donors"])},
        {"metric": "Control_AOIs", "value": int(counts.loc["Control", "n_AOIs"])},
        {"metric": "DCMP_donors", "value": int(counts.loc["DCMP", "n_donors"])},
        {"metric": "DCMP_AOIs", "value": int(counts.loc["DCMP", "n_AOIs"])},
        {"metric": "primary_inference_unit", "value": "PID"},
        {"metric": "primary_exact_permutations", "value": "not_performed"},
        {"metric": "formal_donor_inference_performed", "value": False},
        {"metric": "why_no_formal_donor_inference", "value": "only 2 selected Control donors; prespecified minimum is 3 per group"},
        {"metric": "primary_programs_direction_concordant", "value": int(primary_tests["direction_concordant"].sum())},
        {"metric": "primary_programs_FDR_lt_0.05", "value": int(primary_tests["exact_permutation_FDR_two_sided"].lt(0.05).sum())},
        {"metric": "genes_excluded_by_author_probe_QC", "value": "SEMA3D;PDGFC"},
        {"metric": "LEC_selection_rule", "value": "at least 2 of CCL21,LYVE1,PDPN,FLT4,PROX1 above AOI-specific LOQ"},
        {"metric": "CCL21_selection_endpoint_overlap", "value": True},
    ]
    for row in primary_tests.itertuples(index=False):
        summary_rows.extend(
            [
                {"metric": f"{row.program}__difference_DCMP_minus_Control", "value": row.difference_DCMP_minus_Control},
                {"metric": f"{row.program}__exact_p_two_sided", "value": row.exact_permutation_p_two_sided},
                {"metric": f"{row.program}__FDR_two_sided", "value": row.exact_permutation_FDR_two_sided},
                {"metric": f"{row.program}__primary_evidence_flag", "value": row.primary_evidence_flag},
            ]
        )
    pd.DataFrame(summary_rows).to_csv(output_root / "13_3_run_summary.csv", index=False)

    run_config = {
        "script_version": SCRIPT_VERSION,
        "input_h5ad": str(input_h5ad),
        "target_registry": str(target_registry_path),
        "output_root": str(output_root),
        "expression_layer": EXPRESSION_LAYER,
        "normalization_inherited_from_step13_2": "log2(primary-cohort upper-quartile CPM + 1)",
        "primary_unit": "PID equal-weight donor mean",
        "minimum_donors_per_group": args.minimum_donors_per_group,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "primary_programs": PRIMARY_PROGRAMS,
        "expected_directions": EXPECTED_DIRECTIONS,
        "LEC_selection_rule_inherited_from_step13_2": "at least 2 of CCL21,LYVE1,PDPN,FLT4,PROX1 above AOI-specific LOQ",
        "formal_inference": "not performed because only 2 selected Control donors; minimum is 3 per group",
        "multiple_testing": "not applicable; no inferential P values were computed",
    }
    (output_root / "13_3_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    notes = f"""Step 13.3 method and interpretation notes
=========================================

Primary question
----------------
Do selected LV vessel AOIs from DCMP donors show the mentor-specified shift
away from the MMRN1/CCL21 homeostatic program and toward State0-like,
State3-like, and activated-vascular programs?

Selection and scoring
---------------------
- Step 13.2 required at least two of CCL21, LYVE1, PDPN, FLT4, and PROX1 above
  each AOI's LOQ. No additional endothelial-expression gate was used.
- Input: {adata.n_obs} selected GeoMx AOIs
  ({int(counts.loc['Control', 'n_AOIs'])} Control, {int(counts.loc['DCMP', 'n_AOIs'])} DCMP)
  from {len(donor_scores)} independent donors.
- Expression: {EXPRESSION_LAYER} inherited from Step 13.2.
- Each gene was median/MAD standardized using equal-weight donor means so
  donors with repeated AOIs did not control the reference distribution.
- Program score: arithmetic mean of retained member-gene z scores.
- SEMA3D and PDGFC were excluded because they failed the authors' 1% probe-QC
  filter; they were not imputed.
- CCL21 contributes to selection and to the homeostatic endpoint. Therefore,
  CCL21 and the MMRN1/CCL21 score are not selection-independent validation.

Statistics
----------
- Biological replicate: PID ({int(counts.loc['Control', 'n_donors'])} Control,
  {int(counts.loc['DCMP', 'n_donors'])} DCMP), not AOI.
- Repeated AOIs were averaged within donor.
- The Control group has fewer than the prespecified minimum of
  {args.minimum_donors_per_group} independent donors. Consequently, every
  program, gene, sensitivity, niche, and association result is descriptive.
- No permutation P value, bootstrap interval, FDR, mixed-model P value, or
  positive validation grade is allowed for this selected cohort.

Sensitivity analyses
--------------------
- Strict anatomical LV uses the original anatomy metadata.
- Strict controls use the original noncardiac/autopsy control definition.
- Pre-LVAD removes after-LVAD material.
- Candidate endothelial outliers from the authors' codes are only a named
  sensitivity, not an automatic primary exclusion.
- The stringent sensitivity requires at least three of the same five LEC
  markers above LOQ. Every sensitivity remains descriptive.

Niche and TF boundary
---------------------
BMP4, BMPR1A/BMPR2, and NRP1/NRP2 scores and correlations are descriptive
co-expression only; they do not establish sender-to-receiver direction,
pathway activity, or causality. EPAS1 expression is a secondary target and
does not validate SCENIC regulon activity.

Interpretation boundary
-----------------------
Use the phrase 'LEC-marker-selected LV vessel AOIs.' Do not call these AOIs
individual or purified LECs. Step 13.4 must grade expected versus opposite
directions as descriptive and must not claim formal DCM validation.
"""
    (output_root / "13_3_method_and_interpretation_notes.txt").write_text(notes, encoding="utf-8")

    print(f"Step 13.3 complete: {output_root}")
    print(primary_tests[[
        "program", "difference_DCMP_minus_Control", "direction_concordant",
        "exact_permutation_p_two_sided", "exact_permutation_FDR_two_sided",
        "primary_evidence_flag",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
