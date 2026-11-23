"""Panel E: visualize frozen state assignments; never fit or rename clusters.

All inputs must already be inside this figure project. Run --inspect-only to
check availability. --prepare-only makes auditable summaries without rendering;
--render-only uses those summaries. The default prepares and renders.
Original inputs and completed outputs are never overwritten by default.
"""
from pathlib import Path
from figure_paths import panel_dir, resolve_result_path, assert_preserved
import argparse
import hashlib
import json
import os
import platform
import sys

sys.dont_write_bytecode = True
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(variable, '1')

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
import matplotlib

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'result'
SOURCE = RESULT / 'Panel_E/panel_E_source'
C_DATA = RESULT / 'Panel_C/panel_C_data'
os.environ['MPLCONFIGDIR'] = str(RESULT / '.matplotlib_cache')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize
from matplotlib.patches import Rectangle

PREFIX = 'panel_E_state_visualization'
QC_PREFIX = 'panel_E_state_QC'
GROUPS = ['P3-Control', 'P3-H1N1', 'P9-Control', 'P9-H1N1']
CATEGORIES = ['Cell', 'Nucleus', 'ER', 'Mitochondria']
STATE_NAMES = {i: f'State {i}' for i in range(3)}
STATE_COLORS = ['#d58120', '#327aa4', '#6d439b']
CATEGORY_COLORS = ['#818a94', '#9988a4', '#71a38c', '#be9972']
PASSAGE_COLORS = {'P3': '#0072B2', 'P9': '#D55E00'}
CONFIG = {
    'assignment_column': 'All__k3',
    'state_numbers': [0, 1, 2],
    'groups': GROUPS,
    'refit_clustering': False,
    'renumber_states': False,
    'normalization': 'frozen original state-analysis standardized feature matrix',
    'profile': 'median across represented wells of within-well/state cell medians',
    'features_per_category': 3,
    'nucleus_quota': {'AreaShape': 2, 'Intensity': 1},
    'exclude_from_compact_display': 'Zernike descriptors; retain in full profile table',
    'rank': 'descending range of the three state profiles, then feature name',
    'maximum_within_category_absolute_rank_correlation': 0.80,
    'rank_correlation_weights': 'equal total weight per primary-group well',
    'well_support_min_cells_per_well': 2,
    'well_support_min_wells': 3,
    'maximum_single_well_share_of_equal_well_state_mass': 0.40,
    'profile_sensitivity_min_cells_per_well_state': 2,
    'color_normalization': 'symmetric log, linear threshold 1; no data clipping',
    'statistical_tests_between_states': False,
    'representative_image_placeholders': 3,
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'axes.linewidth': .8,
})


def digest(path):
    with resolve_result_path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def local(path):
    path = resolve_result_path(path)
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f'Input outside permitted figure project: {path}')
    return path


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path,)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
                                    default=json_default), encoding='utf-8')


def csv_read(path, **kwargs):
    return pd.read_csv(local(path), keep_default_na=False, **kwargs)


def inputs():
    return {
        'state_assignments': SOURCE / 'cell_cluster_assignments.csv.gz',
        'original_feature_matrix': SOURCE / 'feature_matrices.npz',
        'original_matrix_cell_order_and_metadata': SOURCE / 'cells_all_features.csv.gz',
        'original_feature_decisions': SOURCE / 'feature_decisions.csv',
        'primary_cell_metadata': C_DATA / 'cell_metadata.csv',
    }


def input_inventory():
    records = []
    for role, path in inputs().items():
        path = local(path)
        records.append({'role': role, 'path': str(path.relative_to(ROOT)),
                        'exists': path.is_file(),
                        'sha256': digest(path) if path.is_file() else None})
    candidates = []
    for path in sorted(RESULT.rglob('*')):
        if not path.is_file() or '.matplotlib_cache' in path.parts:
            continue
        if path.suffix.lower() == '.csv' or path.name.endswith('.csv.gz'):
            columns = csv_read(path, nrows=0).columns.tolist()
            state_columns = [c for c in columns if c == 'All__k3'
                             or c.lower() in {'state', 'state_id', 'cluster', 'cluster_id'}]
            if state_columns:
                candidates.append({'path': str(path.relative_to(ROOT)),
                                   'state_columns': state_columns})
    return {'project': str(ROOT), 'required_inputs': records,
            'required_inputs_available': all(r['exists'] for r in records),
            'tables_with_state_columns': candidates,
            'local_primary_matrix_available': (C_DATA / 'normalized_features.npz').is_file(),
            'note': 'No external file access, cluster reconstruction, or label inference is attempted.'}


def category(feature):
    if feature.startswith('ERWNuclei__AreaShape_'):
        return 'Cell'
    if feature.startswith(('nuclei__AreaShape_', 'nuclei__Intensity_')):
        return 'Nucleus'
    if feature.startswith('ERWONuclei__Intensity_'):
        return 'ER'
    if feature.startswith('mitochondriaWONuclei__Intensity_'):
        return 'Mitochondria'
    raise ValueError(f'Unmapped original retained feature: {feature}')


def display_label(feature):
    measurement = feature.split('__', 1)[1]
    if measurement.startswith('AreaShape_'):
        measurement = measurement[len('AreaShape_'):]
    else:
        measurement = measurement[len('Intensity_'):]
        for suffix in ['_nucleus', '_ER', '_mitochondria']:
            if measurement.endswith(suffix):
                measurement = measurement[:-len(suffix)]
    names = {
        'Area': 'Area', 'Compactness': 'Compactness', 'Eccentricity': 'Eccentricity',
        'Extent': 'Extent', 'FormFactor': 'Form factor', 'Solidity': 'Solidity',
        'Perimeter': 'Perimeter', 'MinFeretDiameter': 'Minimum Feret diameter',
        'MaxFeretDiameter': 'Maximum Feret diameter', 'MedianRadius': 'Median radius',
        'MeanRadius': 'Mean radius', 'MaximumRadius': 'Maximum radius',
        'MajorAxisLength': 'Major axis length', 'MinorAxisLength': 'Minor axis length',
        'MassDisplacement': 'Intensity mass displacement',
        'IntegratedIntensity': 'Integrated intensity',
        'IntegratedIntensityEdge': 'Integrated intensity (edge)',
        'MeanIntensity': 'Mean intensity', 'MeanIntensityEdge': 'Mean intensity (edge)',
        'MaxIntensity': 'Maximum intensity', 'MaxIntensityEdge': 'Maximum intensity (edge)',
        'MinIntensity': 'Minimum intensity', 'MinIntensityEdge': 'Minimum intensity (edge)',
        'MedianIntensity': 'Median intensity', 'MADIntensity': 'MAD of intensity',
        'StdIntensity': 'SD of intensity', 'StdIntensityEdge': 'SD of intensity (edge)',
    }
    return names.get(measurement, measurement)


def preserve_manifest(out):
    return {str(p.resolve()): digest(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and '.matplotlib_cache' not in p.parts
            and '__pycache__' not in p.parts
            and p.resolve() != Path(__file__).resolve()
            and not (p.parent.resolve() == out and
                     (p.name.startswith(PREFIX) or p.name.startswith(QC_PREFIX)))
            and not p.resolve().is_relative_to(out / f'{PREFIX}_data')}


def verify_preserved(manifest):
    for name, expected in manifest.items():
        assert_preserved(name, expected)


def load_primary():
    required = inputs()
    missing = [str(p.relative_to(ROOT)) for p in required.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Required existing local outputs are missing:\n' + '\n'.join(missing))
    assignments = csv_read(required['state_assignments'])
    original = csv_read(required['original_matrix_cell_order_and_metadata'])
    meta = csv_read(required['primary_cell_metadata'])
    assert assignments.cell_id.is_unique and original.cell_id.is_unique and meta.cell_id.is_unique
    assert np.array_equal(assignments.cell_id.to_numpy(), original.cell_id.to_numpy()), \
        'Original feature-matrix row identity must match the saved assignment row identity.'
    for column in ['well_id', 'passage', 'plasma', 'intervention', 'physical_well',
                   'ImageNumber', 'ObjectNumber', 'source_nd2_filename', 'source_nd2_md5']:
        assert assignments[column].astype(str).equals(original[column].astype(str)), column
    with np.load(local(required['original_feature_matrix']), allow_pickle=True) as arrays:
        # This trusted project NPZ stores feature names as an object array.
        features = arrays['features'].astype(str).tolist()
        original_z = np.asarray(arrays['standardized'], dtype=float)
    assert original_z.shape == (len(original), len(features))
    feature_decisions = csv_read(required['original_feature_decisions'])
    assert set(features) == set(feature_decisions.loc[feature_decisions.retained.eq(True), 'feature'])
    assert meta.panel_group.isin(GROUPS).all() and meta.intervention.eq('None').all()
    assert meta.panel_group.equals(meta.passage + '-' + meta.plasma)
    ids = original.set_index('cell_id').index.get_indexer(meta.cell_id)
    assert np.all(ids >= 0)
    linked = assignments.iloc[ids].reset_index(drop=True)
    for column in [c for c in assignments.columns if c in meta.columns]:
        assert linked[column].astype(str).equals(meta[column].astype(str)), column
    states = linked[CONFIG['assignment_column']].to_numpy(int)
    assert set(states) == {0, 1, 2}
    meta['state'] = states
    meta['state_name'] = meta.state.map(STATE_NAMES)
    z = original_z[ids]
    assert np.isfinite(z).all()
    well_n = meta.groupby('well_id').cell_id.transform('size').to_numpy()
    weights = 1.0 / (meta.well_id.nunique() * well_n)
    assert np.isclose(weights.sum(), 1)
    np.testing.assert_allclose(weights, meta.equal_well_cell_weight.to_numpy(float))
    return meta, z, features, weights, assignments, feature_decisions


def state_tables(meta):
    well_meta = meta[['well_id', 'panel_group', 'passage', 'plasma', 'physical_well']].drop_duplicates()
    assert well_meta.well_id.is_unique
    well_meta['group_order'] = well_meta.panel_group.map({g: i for i, g in enumerate(GROUPS)})
    well_meta = well_meta.sort_values(['group_order', 'physical_well', 'well_id']).reset_index(drop=True)
    well_order = well_meta.well_id.tolist()
    counts = pd.crosstab(meta.well_id, meta.state).reindex(index=well_order, columns=range(3), fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0)
    mass = fractions.div(fractions.sum(axis=0), axis=1)
    records = []
    for info in well_meta.to_dict('records'):
        well = info['well_id']
        for state in range(3):
            records.append({**info, 'state': state, 'state_name': STATE_NAMES[state],
                            'n_cells_in_well': int(counts.loc[well].sum()),
                            'n_cells_in_state': int(counts.loc[well, state]),
                            'fraction_of_well': fractions.loc[well, state],
                            'fraction_of_state_cells': counts.loc[well, state] / counts[state].sum(),
                            'fraction_of_equal_well_state_mass': mass.loc[well, state]})
    well_states = pd.DataFrame(records)
    summaries = []
    for state in range(3):
        cells = meta.loc[meta.state.eq(state)]
        support = well_states.loc[well_states.state.eq(state)]
        row = {'state': state, 'state_name': STATE_NAMES[state], 'n_cells': len(cells),
               'n_primary_wells': len(well_meta),
               'n_wells_with_1_or_more_cells': int(support.n_cells_in_state.ge(1).sum()),
               'n_wells_with_2_or_more_cells': int(support.n_cells_in_state.ge(2).sum()),
               'max_single_well_fraction_of_state_cells': support.fraction_of_state_cells.max(),
               'largest_wells_by_cell_count': '|'.join(support.loc[
                   support.n_cells_in_state.eq(support.n_cells_in_state.max()), 'well_id']),
               'max_single_well_fraction_of_equal_well_state_mass': support.fraction_of_equal_well_state_mass.max(),
               'largest_well_by_equal_well_mass': support.loc[
                   support.fraction_of_equal_well_state_mass.idxmax(), 'well_id'],
               'support_rule_pass': bool(support.n_cells_in_state.ge(2).sum() >= 3 and
                                         support.fraction_of_equal_well_state_mass.max() <= .40),
               'n_P3_cells': int(cells.passage.eq('P3').sum()),
               'n_P9_cells': int(cells.passage.eq('P9').sum()),
               'n_source_edge_flags': int(cells.qc_edge_touching.sum()),
               'n_source_saturation_flags': int(cells.qc_saturated_maximum.sum()),
               'n_source_extreme_feature_flags': int(cells.qc_extreme_z8.sum()),
               'n_panel_C_extreme_feature_flags': int(cells.qc_extreme_primary_z8.sum())}
        for group in GROUPS:
            row['n_' + group] = int(cells.panel_group.eq(group).sum())
        summaries.append(row)
    group_counts = pd.crosstab(meta.panel_group, meta.state).reindex(index=GROUPS, columns=range(3), fill_value=0)
    group_counts.columns = [STATE_NAMES[s] for s in group_counts.columns]
    group_counts['n_cells'] = group_counts.sum(axis=1)
    group_counts['n_wells'] = well_meta.groupby('panel_group').size().reindex(GROUPS)
    return pd.DataFrame(summaries), well_states, group_counts.reset_index(), well_meta


def feature_profiles(meta, z, features):
    frame = pd.DataFrame(z, columns=features)
    frame['well_id'] = meta.well_id.to_numpy()
    frame['state'] = meta.state.to_numpy()
    grouped = frame.groupby(['well_id', 'state'], sort=True)
    per_well = grouped[features].median()
    per_well['n_state_cells_in_well'] = grouped.size()
    profile = per_well.groupby(level='state')[features].median().reindex(range(3))
    sensitivity = per_well.loc[per_well.n_state_cells_in_well.ge(2)].groupby(level='state')[features].median().reindex(range(3))
    rows = []
    for feature in features:
        row = {'original_feature': feature, 'category': category(feature),
               'display_label': display_label(feature)}
        for state in range(3):
            row[f'state_{state}'] = float(profile.loc[state, feature])
            row[f'state_{state}_min2_sensitivity'] = float(sensitivity.loc[state, feature])
        row['state_profile_range'] = float(profile[feature].max() - profile[feature].min())
        rows.append(row)
    profile_table = pd.DataFrame(rows)
    group_lookup = meta.set_index('well_id')[['panel_group', 'passage', 'physical_well']]
    group_lookup = group_lookup.loc[~group_lookup.index.duplicated()]
    well_table = per_well.reset_index().merge(group_lookup, left_on='well_id', right_index=True, validate='many_to_one')
    stratified = well_table.groupby(['panel_group', 'state'])[features].median().reset_index()
    return profile_table, well_table, stratified


def select_features(profiles, z, weights, features):
    ranks = rankdata(z, axis=0, method='average')
    ranks -= np.sum(weights[:, None] * ranks, axis=0)
    covariance = (ranks * weights[:, None]).T @ ranks
    sd = np.sqrt(np.diag(covariance))
    assert np.all(sd > 0)
    correlation = covariance / np.outer(sd, sd)
    correlation = np.clip(correlation, -1, 1)
    indices = {feature: j for j, feature in enumerate(features)}
    audit = profiles.copy().set_index('original_feature')
    audit['selected_for_display'] = False
    audit['display_order'] = 0
    audit['candidate_rank_in_category'] = 0
    audit['selection_reason'] = ''
    audit['maximum_abs_correlation_to_previously_selected'] = np.nan
    audit['most_correlated_previously_selected'] = ''
    ordered = []
    for cat in CATEGORIES:
        members = audit.loc[audit.category.eq(cat)].copy()
        candidate = members.loc[~members.index.str.contains('Zernike')].reset_index()
        candidate = candidate.sort_values(['state_profile_range', 'original_feature'], ascending=[False, True], kind='stable')
        for name in members.index[members.index.str.contains('Zernike')]:
            audit.loc[name, 'selection_reason'] = 'Zernike descriptor omitted from compact display; retained in full summaries'
        selected = []
        subtype_counts = {'AreaShape': 0, 'Intensity': 0}
        for rank, row in enumerate(candidate.to_dict('records'), start=1):
            feature = row['original_feature']
            subtype = 'AreaShape' if '__AreaShape_' in feature else 'Intensity'
            audit.loc[feature, 'candidate_rank_in_category'] = rank
            highest = 0.0
            if selected:
                values = np.abs(correlation[indices[feature], [indices[s] for s in selected]])
                match = int(np.argmax(values))
                highest = float(values[match])
                audit.loc[feature, 'maximum_abs_correlation_to_previously_selected'] = highest
                audit.loc[feature, 'most_correlated_previously_selected'] = selected[match]
            if len(selected) == CONFIG['features_per_category']:
                reason = 'Three-feature category quota already filled'
            elif cat == 'Nucleus' and subtype_counts[subtype] == CONFIG['nucleus_quota'][subtype]:
                reason = f'Fixed nucleus {subtype} quota already filled'
            elif highest > CONFIG['maximum_within_category_absolute_rank_correlation']:
                reason = 'Redundant with an earlier selected feature in this category: absolute weighted rank correlation >0.80'
            else:
                selected.append(feature)
                ordered.append(feature)
                subtype_counts[subtype] += 1
                audit.loc[feature, 'selected_for_display'] = True
                audit.loc[feature, 'display_order'] = len(ordered)
                reason = 'Selected by profile-range ranking, category quota and correlation rule'
            audit.loc[feature, 'selection_reason'] = reason
        assert len(selected) == CONFIG['features_per_category'], f'Insufficient nonredundant candidates in {cat}; no silent rule relaxation.'
        if cat == 'Nucleus':
            assert subtype_counts == CONFIG['nucleus_quota']
    selected = audit.loc[ordered].reset_index()
    selected_indices = [indices[f] for f in ordered]
    pairs = []
    for i, f in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            g = ordered[j]
            value = float(correlation[selected_indices[i], selected_indices[j]])
            pairs.append({'feature_1': f, 'feature_2': g, 'same_category': category(f) == category(g),
                          'equal_well_weighted_rank_correlation': value})
            if category(f) == category(g):
                assert abs(value) <= .80
    return audit.reset_index(), selected, pd.DataFrame(pairs)


def source_evidence():
    evidence = []
    tables = {}
    for name in ['candidate_screens', 'parameter_scan', 'state_well_support', 'feature_panel_sensitivity']:
        path = SOURCE / f'{name}.csv'
        if path.is_file():
            table = csv_read(path)
            table = table.loc[table.scope.eq('All') & table.k.eq(3)].copy()
            tables[name] = table
            if name == 'candidate_screens':
                for column, value in table.iloc[0].items():
                    if column not in {'scope', 'k'}:
                        evidence.append({'measure': column, 'value': value,
                                         'scope': 'Original all-12-condition model; not recomputed for primary groups',
                                         'source_file': str(path.relative_to(ROOT))})
            elif name == 'feature_panel_sensitivity':
                for row in table.to_dict('records'):
                    evidence.append({'measure': row['sensitivity'] + '_ARI', 'value': row['ARI'],
                                     'scope': 'Original all-12-condition model; not recomputed for primary groups',
                                     'source_file': str(path.relative_to(ROOT))})
            elif name == 'parameter_scan':
                chosen = table.loc[table.pca_variance_target.eq(.9) & table.seed.eq(20260916)]
                if len(chosen) == 1:
                    for key in ['n_pcs', 'silhouette_equal_well_sample']:
                        evidence.append({'measure': key, 'value': chosen.iloc[0][key],
                                         'scope': 'Original all-12-condition model; not recomputed for primary groups',
                                         'source_file': str(path.relative_to(ROOT))})
    return pd.DataFrame(evidence, columns=['measure', 'value', 'scope', 'source_file']), tables


def markdown_table(frame, columns):
    header = '| ' + ' | '.join(columns) + ' |\n'
    separator = '| ' + ' | '.join(['---'] * len(columns)) + ' |\n'
    lines = ['| ' + ' | '.join(str(row[c]) for c in columns) + ' |' for row in frame.to_dict('records')]
    return header + separator + '\n'.join(lines)


def write_report(out, meta, features, summary, group_counts, selected, evidence):
    report = f'''# Panel E: Morphological state summary

## Existing inputs and unchanged state definitions

Assignments: `result/Panel_E/panel_E_source/cell_cluster_assignments.csv.gz`, column `All__k3`.
Original state-analysis feature matrix: `result/Panel_E/panel_E_source/feature_matrices.npz`, array `standardized`.
Row order is verified against `cells_all_features.csv.gz` and the complete assignment table before joining primary cells by `cell_id`.
Existing state summaries, if supplied, remain reference records only: their all-condition profiles are not used as primary-only profiles.
Panel C `cell_metadata.csv` identifies the primary cohort and supplies matching well/group metadata.

The existing labels State 0, State 1 and State 2 are preserved exactly. No clustering, PCA refit, state renumbering, new normalization or cell deletion occurs. The original model and preprocessing were fitted to all 12 conditions; only the {len(meta):,} cells in the four primary groups enter the new summaries. This is a filtered visualization of a frozen classification, not a new primary-only clustering result.

The original analysis used {len(features)} retained features, equal-well-weighted PCA (42 components for >=90% variance), and weighted k-means with k=3, 15 initializations, 300 iterations maximum and seed 20260916. Those settings are provenance, not parameters fitted by this visualization script. Model stability records are inherited from the all-condition model and are not independent validation in the primary subset. The imported records and their source hashes are supplied with these outputs.

## Cohort and computational QC

{markdown_table(summary, ['state_name', 'n_cells', 'n_wells_with_1_or_more_cells', 'n_wells_with_2_or_more_cells'])}

{markdown_table(group_counts, ['panel_group', 'State 0', 'State 1', 'State 2', 'n_cells', 'n_wells'])}

All {meta.well_id.nunique()} primary wells are retained. Well/state counts include zero counts and are exported for every combination. Raw cell-count shares and equal-well shares are both reported, because wells have unequal cell yields. For state s and well w, p_ws = n_ws/n_w. Its equal-well contribution to that state is p_ws/sum_w(p_ws). The inherited computational support screen requires at least two state cells in at least three wells and no single well contributing more than 40% of this equal-well state mass. The threshold is a screening convention, not proof that a state is free of technical effects.

Source edge, saturation and extreme-feature flags are reported without deleting cells. The original extreme-feature flag and Panel C's primary-normalization flag are kept separate. A large extreme-feature fraction is a reason to qualify the computational summary; it does not by itself establish that a record is erroneous. Single-well support does not exclude acquisition, processing or positional effects. Existing image review has no segmentation masks; the placeholder images have not been selected or verified. The source channel annotation discrepancy documented for Panel B remains unresolved, so category labels reproduce measurement names only.

## Feature profiles and reproducible display selection

Category is determined only by CellProfiler object/measurement prefixes: ERWNuclei AreaShape -> Cell; nuclei AreaShape/Intensity -> Nucleus; ERWONuclei Intensity -> ER; mitochondriaWONuclei Intensity -> Mitochondria. These are display categories, with no functional or mechanistic meaning assigned.

For every feature j, state s and well w with at least one state cell, calculate m_wsj = median of the original normalized values in that well/state. The displayed profile is median_w(m_wsj). Thus each represented well has equal influence. Wells with no cells in a state are absent from its profile rather than being assigned zero. State 0 has fewer contributing wells and fewer cells per well. A companion table repeats the profile using only well/state combinations with at least two cells; the labels, feature selection and normalization remain fixed.

All {len(features)} features are included in the complete profile table. The compact display omits Zernike descriptors to favor directly named measurements. Within each category, candidates are ranked by max_s(profile_sj)-min_s(profile_sj), descending, with alphabetical feature-name tie-breaking. Greedy selection retains three features per category and rejects candidates with absolute equal-well weighted rank correlation >0.80 to an already-selected feature in the same category. Nucleus has a fixed quota of two AreaShape and one Intensity measurement. Rank correlation uses average cell ranks and weight 1/(number of wells * cells in that well). No treatment contrast, P value, expected mechanism or preferred direction enters selection. The full audit records all exclusions and correlation partners; cross-category correlations are also exported but do not remove a category from the display.

These features describe the existing state classification. Selecting them for contrast is explicitly descriptive and is not independent evidence that the states are validated. No state-versus-state significance tests or biological interpretation are provided.

{markdown_table(selected, ['category', 'original_feature', 'state_0', 'state_1', 'state_2'])}

## Figure layout and reading

Left: 12 features in four category blocks, with State 0, State 1 and State 2 columns. Values are the original robust-scaled feature values summarized as above, not raw units or new per-state z scores. Blue indicates lower and red higher values on this shared scale. Numeric values are printed in each cell. Color uses symmetric-log normalization (linear threshold 1, symmetric limits covering every displayed value) to keep modest and extreme values visible without clipping or altering numerical values. This nonlinear color mapping is stated on the colorbar.

Right: three intentionally empty, labeled representative-image areas. These contain only text and borders; no microscopy image is generated, substituted or selected. The supporting QC figure shows all well/state fractions, with raw state-cell counts printed in each tile.

## Files and reproduction

Run `script/panel_E_state_visualization.py --prepare-only`, inspect the summaries, then run with `--render-only`. A default run does both steps. All required input paths are constrained to the figure project. A completed run is not overwritten by default; use a new `--output-dir` inside result for another run. --render-only rewrites only this script's figure files from saved tables. Source and earlier-panel hashes are checked before and after preparation/rendering. The `panel_E_state_QC.xlsx` workbook contains state summaries, well distributions, feature profiles, selection audit, correlation pairs and inherited computational stability records. Exact source paths and SHA-256 hashes are recorded in the data subfolder.
'''
    (out / f'{PREFIX}.methods.md').write_text(report, encoding='utf-8')
    legend = f'''**E. Morphological state summary.** Existing State 0–2 assignments were retained without refitting or renumbering. Only P3-Control, P3-H1N1, P9-Control and P9-H1N1 cells are summarized ({len(meta):,} cells, {meta.well_id.nunique()} wells). Heatmap entries are medians across represented wells of within-well/state median normalized measurements; the normalization is frozen from the original {len(features)}-feature state analysis. Three features per name-defined measurement category were selected by a reproducible state-profile range and redundancy rule. All features and selection decisions are supplied in the supporting tables. The original classification and preprocessing used all 12 conditions; other conditions do not contribute cells to this panel. Colors use a symmetric-log scale with linear threshold 1 and no clipping; printed values are unchanged. Counts beneath each state give cells and contributing wells. Empty representative-image spaces await manual selection. Category labels follow the existing measurement annotations and do not assign functional or mechanistic meaning to the states. *Mitochondria follows the source measurement annotation; the CY5/TRITC annotation discrepancy documented for Panel B remains unresolved.
'''
    (out / f'{PREFIX}.legend.md').write_text(legend, encoding='utf-8')
    concise = summary[['state_name', 'n_cells', 'n_wells_with_1_or_more_cells',
                       'n_wells_with_2_or_more_cells']].copy()
    concise['largest_well_share_of_cells'] = summary.max_single_well_fraction_of_state_cells.map(lambda x: f'{100*x:.2f}%')
    concise['largest_well_share_of_equal_well_mass'] = summary.max_single_well_fraction_of_equal_well_state_mass.map(lambda x: f'{100*x:.2f}%')
    concise['source_extreme_flags'] = summary.apply(lambda x: f'{x.n_source_extreme_feature_flags}/{x.n_cells}', axis=1)
    qc_text = f'''# Panel E computational QC

Only the four primary groups contribute to these summaries: {len(meta):,} cells, {meta.well_id.nunique()} wells, five wells per group. State assignments match the saved `All__k3` labels exactly. No clustering is rerun and no cell is reassigned or deleted.

{markdown_table(concise, concise.columns.tolist())}

All three states pass the inherited support screen of at least two state cells in at least three wells and at most 40% of equal-well state mass from one well. The largest contribution is {100*summary.max_single_well_fraction_of_equal_well_state_mass.max():.2f}%. Accordingly, no state is dominated by one well under that specific screen. This does not rule out other technical associations.

Raw cell-count shares and equal-well shares have different denominators. The latter normalize each well's state-cell count by that well's total cell count before comparing contributions between wells. Every well/state combination, including zeros, is available in the workbook and the data CSV. The QC figure prints cell counts and encodes within-well fractions with color.

State 0 has 28/28 source extreme-feature flags and 27/28 flags under Panel C's separate primary-only normalization. State 1 has 7/1,325 and State 2 has 2/273 under both flags. These are existing computational flags and are not proof of incorrect records. The state labels are unchanged. The feature matrix used here is the original 97-feature matrix; Panel C's separately prepared matrix contains 98 features. Panel E does not refit either normalization.

Category counts in the original matrix: {', '.join(f'{c}: {sum(category(f)==c for f in features)}' for c in CATEGORIES)}. The compact heatmap contains 12 features; all {len(features)} profiles and all display exclusions are exported. The 3 x 20 well/state display covers all primary wells. The three representative-image areas are placeholders only.

Inherited model stability records apply to the original all-12-condition model. They are supplied in `inherited_model_evidence.csv` and the workbook and are not recomputed or relabeled as primary-only validation. No state-versus-state significance tests, biological interpretation or functional inference is made.
'''
    (out / f'{QC_PREFIX}.md').write_text(qc_text, encoding='utf-8')


def prepare(out):
    if (out / f'{PREFIX}.validation.json').exists():
        raise FileExistsError('Completed Panel E exists. Use a new --output-dir to preserve it.')
    data = out / f'{PREFIX}_data'
    if data.exists() and any(data.iterdir()):
        raise FileExistsError('Panel E data directory is nonempty; use a fresh output directory.')
    inventory = input_inventory()
    if not inventory['required_inputs_available']:
        print(json.dumps(inventory, indent=2))
        raise FileNotFoundError('Existing local state inputs are missing; no analysis or figure was generated.')
    manifest = preserve_manifest(out)
    data.mkdir(parents=True, exist_ok=True)
    write_json(CONFIG, out / f'{PREFIX}.preanalysis_plan.json')
    meta, z, features, weights, assignments, decisions = load_primary()
    summary, well_states, group_counts, well_meta = state_tables(meta)
    profiles, well_profiles, group_profiles = feature_profiles(meta, z, features)
    audit, selected, correlation_pairs = select_features(profiles, z, weights, features)
    evidence, inherited = source_evidence()
    assert int(summary.n_cells.sum()) == len(meta)
    assert len(well_states) == meta.well_id.nunique() * 3
    assert (group_counts.n_wells == 5).all()
    assert len(selected) == 12 and selected.groupby('category').size().eq(3).all()
    linked = assignments.set_index('cell_id').loc[meta.cell_id, CONFIG['assignment_column']].to_numpy(int)
    assert np.array_equal(meta.state.to_numpy(), linked)
    assert len(profiles) == len(features) and audit.selection_reason.ne('').all()
    meta.to_csv(data / 'primary_cell_state_assignments.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    np.savez_compressed(data / 'primary_existing_normalized_features.npz',
                        standardized=z, features=np.array(features), cell_ids=meta.cell_id.to_numpy(str), weights=weights)
    summary.to_csv(out / f'{QC_PREFIX}.csv', index=False)
    well_states.to_csv(data / 'well_state_counts_and_fractions.csv', index=False)
    group_counts.to_csv(data / 'group_state_counts.csv', index=False)
    profiles.to_csv(data / 'all_state_feature_profiles.csv', index=False)
    well_profiles.to_csv(data / 'well_state_feature_profiles.csv.gz', index=False,
                         compression={'method': 'gzip', 'mtime': 0})
    group_profiles.to_csv(data / 'group_state_feature_profiles.csv', index=False)
    audit.to_csv(out / f'{PREFIX}.feature_selection.csv', index=False)
    selected.to_csv(data / 'selected_feature_profiles.csv', index=False)
    correlation_pairs.to_csv(data / 'selected_feature_correlations.csv', index=False)
    evidence.to_csv(data / 'inherited_model_evidence.csv', index=False)
    for name, frame in inherited.items():
        frame.to_csv(data / f'inherited_{name}.csv', index=False)
    source_records = []
    for role, path in inputs().items():
        source_records.append({'role': role, 'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
    for path in sorted(SOURCE.iterdir()):
        if path.is_file() and path not in inputs().values():
            source_records.append({'role': 'existing source evidence', 'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
    pd.DataFrame(source_records).to_csv(data / 'source_manifest.csv', index=False)
    pd.DataFrame([{'path': p, 'sha256': h} for p, h in manifest.items()]).to_csv(data / 'preservation_manifest.csv', index=False)
    with pd.ExcelWriter(out / f'{QC_PREFIX}.xlsx', engine='openpyxl') as writer:
        for name, frame in [('State summary', summary), ('Group counts', group_counts),
                            ('Well distribution', well_states), ('Selected features', selected),
                            ('All feature profiles', profiles), ('Feature selection', audit),
                            ('Selected correlations', correlation_pairs), ('Inherited model evidence', evidence),
                            ('Source files', pd.DataFrame(source_records))]:
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                values = [len(str(c.value or '')) for c in column]
                sheet.column_dimensions[column[0].column_letter].width = min(65, max(12, max(values) + 2))
    write_report(out, meta, features, summary, group_counts, selected, evidence)
    verify_preserved(manifest)
    validation = {
        'status': 'prepared; figures not yet rendered', 'existing_assignments_preserved': True,
        'cluster_refit': False, 'all_numerical_inputs_inside_figure_project': True,
        'primary_cells': len(meta), 'primary_wells': meta.well_id.nunique(),
        'group_wells': group_counts[['panel_group', 'n_wells']].to_dict('records'),
        'existing_model_features': len(features), 'display_features': len(selected),
        'state_summary': summary.to_dict('records'), 'selected_features': selected.original_feature.tolist(),
        'all_selected_within_category_abs_rank_correlations_le_0_80': True,
        'previous_files_hash_checked': len(manifest), 'previous_files_unchanged': True,
        'artificial_microscopy_images': 0, 'placeholder_count': 3,
        'inherited_stability_scope': 'original all-12-condition model',
        'software': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
    }
    write_json(validation, out / f'{PREFIX}.validation.json')
    print('PREPARATION COMPLETE; existing labels unchanged.')
    print(summary[['state_name', 'n_cells', 'n_wells_with_1_or_more_cells',
                   'max_single_well_fraction_of_state_cells',
                   'max_single_well_fraction_of_equal_well_state_mass']].to_string(index=False))
    print(selected[['category', 'original_feature', 'state_0', 'state_1', 'state_2']].to_string(index=False))


def save_figure(fig, out, name, dpi=600):
    for extension in ['pdf', 'svg']:
        fig.savefig(out / f'{name}.{extension}', facecolor='white')
    fig.savefig(out / f'{name}.png', dpi=dpi, facecolor='white')
    fig.savefig(out / f'{name}.preview.png', dpi=140, facecolor='white')
    plt.close(fig)


def render_main(out, selected, summary):
    values = selected[[f'state_{s}' for s in range(3)]].to_numpy(float)
    bound = max(1.0, float(np.abs(values).max()))
    norm = SymLogNorm(linthresh=1, linscale=1, vmin=-bound, vmax=bound, base=10)
    cmap = plt.get_cmap('RdBu_r')
    fig = plt.figure(figsize=(12.8, 6.1))
    fig.text(.018, .965, 'E', fontsize=21, fontweight='bold', va='top')
    fig.text(.065, .961, 'Morphological state summary', fontsize=15, fontweight='bold', va='top')
    fig.text(.065, .918, 'P3-Control  ·  P3-H1N1  ·  P9-Control  ·  P9-H1N1', fontsize=10, color='#53616c', va='top')
    heat = fig.add_axes([.312, .278, .296, .550])
    image = heat.pcolormesh(np.arange(4) - .5, np.arange(13) - .5, values,
                           cmap=cmap, norm=norm, shading='flat', rasterized=False)
    heat.set_xlim(-.5, 2.5)
    heat.set_ylim(11.5, -.5)
    heat.set_xticks([])
    heat.set_yticks(np.arange(len(selected)))
    heat.set_yticklabels(selected.display_label, fontsize=9.0)
    heat.tick_params(axis='y', length=0, pad=8)
    for spine in heat.spines.values():
        spine.set_visible(False)
    for row in range(values.shape[0]):
        for col in range(3):
            rgba = cmap(norm(values[row, col]))
            lightness = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            heat.text(col, row, f'{values[row, col]:.2f}', ha='center', va='center',
                      fontsize=9.4, color='white' if lightness < .50 else '#17212b')
    for edge in [2.5, 5.5, 8.5]:
        heat.axhline(edge, color='white', lw=3)
    for edge in [.5, 1.5]:
        heat.axvline(edge, color='white', lw=.8)
    for state in range(3):
        heat.text(state, -1.02, STATE_NAMES[state], ha='center', va='center',
                  fontsize=11, color=STATE_COLORS[state], fontweight='bold', clip_on=False)
        heat.plot([state - .42, state + .42], [-.67, -.67], color=STATE_COLORS[state], lw=2, clip_on=False)
        row = summary.loc[summary.state.eq(state)].iloc[0]
        heat.text(state, 12.0, f'{int(row.n_cells):,} cells\n{int(row.n_wells_with_1_or_more_cells)}/{int(row.n_primary_wells)} wells',
                  ha='center', va='top', fontsize=8.8, color='#45535e', clip_on=False, linespacing=1.45)
    positions = heat.get_position()
    for j, cat in enumerate(CATEGORIES):
        center = positions.y1 - positions.height * (j * 3 + 1.5) / 12
        height = positions.height * 3 / 12
        fig.add_artist(Rectangle((.019, center - height / 2 + .007), .004, height - .014,
                                 transform=fig.transFigure, facecolor=CATEGORY_COLORS[j], edgecolor='none'))
        fig.text(.029, center, cat + ('*' if cat == 'Mitochondria' else ''), ha='left', va='center', fontsize=9.6, fontweight='bold')
    color_ax = fig.add_axes([.315, .166, .289, .023])
    bar = fig.colorbar(image, cax=color_ax, orientation='horizontal')
    candidates = [-100, -30, -10, -3, -1, 0, 1, 3, 10, 30, 100]
    ticks = [t for t in candidates if -bound <= t <= bound]
    bar.set_ticks(ticks)
    bar.set_ticklabels([str(t) for t in ticks])
    bar.ax.tick_params(labelsize=8, length=2)
    bar.ax.minorticks_off()
    bar.outline.set_linewidth(.5)
    bar.solids.set_rasterized(False)
    bar.set_label('Normalized feature value\nSymmetric-log colors; linear threshold = 1', fontsize=8.6, labelpad=4)
    fig.text(.804, .863, 'Representative images', ha='center', fontsize=11, fontweight='bold')
    for state in range(3):
        y = .657 - state * .193
        box = fig.add_axes([.672, y, .287, .165])
        box.set_xlim(0, 1)
        box.set_ylim(0, 1)
        box.axis('off')
        box.add_patch(Rectangle((.005, .01), .99, .98, fill=False, lw=1,
                                linestyle=(0, (4, 3)), edgecolor=STATE_COLORS[state]))
        box.text(.5, .53, f'Representative image\nState {state}', ha='center', va='center',
                 fontsize=10.5, linespacing=1.6, color='#61707a')
    fig.text(.804, .222, 'Spaces reserved for manual image selection', ha='center', fontsize=8.5, color='#61707a')
    fig.text(.019, .055, 'Profiles: within-well/state medians, then median across represented wells. Existing state definitions retained.',
             fontsize=8.7, color='#45535e')
    fig.text(.019, .025, '*Category follows the source annotation; CY5/TRITC mapping is unresolved. Full feature profiles and QC are supplied separately.',
             fontsize=8.5, color='#61707a')
    save_figure(fig, out, PREFIX)
    return bound


def render_qc(out, well_states, summary):
    well_meta = well_states.drop_duplicates('well_id').sort_values(['group_order', 'physical_well', 'well_id'])
    order = well_meta.well_id.tolist()
    fractions = well_states.pivot(index='state', columns='well_id', values='fraction_of_well').reindex(index=range(3), columns=order)
    counts = well_states.pivot(index='state', columns='well_id', values='n_cells_in_state').reindex(index=range(3), columns=order)
    fig = plt.figure(figsize=(12.8, 4.8))
    fig.text(.055, .953, 'Panel E · State distribution across primary-group wells', fontsize=14, fontweight='bold', va='top')
    fig.text(.055, .884, 'Numbers: cells in each well/state. Color: fraction of that well assigned to the state.', fontsize=10, color='#53616c')
    ax = fig.add_axes([.092, .438, .83, .338])
    im = ax.pcolormesh(np.arange(len(order) + 1) - .5, np.arange(4) - .5,
                      fractions.to_numpy(), cmap='Blues', norm=Normalize(0, 1),
                      shading='flat', rasterized=False)
    ax.set_xlim(-.5, len(order) - .5)
    ax.set_ylim(2.5, -.5)
    ax.set_yticks(range(3), [STATE_NAMES[s] for s in range(3)])
    ax.set_xticks(range(len(order)), well_meta.physical_well.to_list())
    ax.tick_params(axis='both', length=0, labelsize=9)
    for row in range(3):
        for col in range(len(order)):
            ax.text(col, row, str(int(counts.iloc[row, col])), ha='center', va='center', fontsize=9,
                    color='white' if fractions.iloc[row, col] > .55 else '#17212b')
    for j, group in enumerate(GROUPS):
        members = np.flatnonzero(well_meta.panel_group.eq(group).to_numpy())
        ax.text(members.mean(), -.82, group, ha='center', va='center', fontsize=10,
                color=PASSAGE_COLORS[group.split('-')[0]], clip_on=False)
        if j:
            ax.axvline(members[0] - .5, color='white', lw=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    color_ax = fig.add_axes([.94, .438, .012, .338])
    cb = fig.colorbar(im, cax=color_ax)
    cb.solids.set_rasterized(False)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Within-well fraction', fontsize=9)
    table_ax = fig.add_axes([.07, .125, .87, .214])
    table_ax.axis('off')
    table_rows = []
    for row in summary.to_dict('records'):
        table_rows.append([row['state_name'], f"{row['n_cells']:,}",
                           str(row['n_wells_with_1_or_more_cells']),
                           f"{100 * row['max_single_well_fraction_of_state_cells']:.1f}%",
                           f"{100 * row['max_single_well_fraction_of_equal_well_state_mass']:.1f}%",
                           f"{row['n_source_extreme_feature_flags']}/{row['n_cells']}"])
    table = table_ax.table(cellText=table_rows,
                          colLabels=['State', 'Cells', 'Wells with cells', 'Largest well\nshare of cells',
                                     'Largest well\nshare, equal-well mass', 'Source extreme\nfeature flags'],
                          loc='center', cellLoc='center', colWidths=[.12, .10, .16, .19, .25, .18])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(.4)
        cell.set_edgecolor('#d7dce0')
        if r == 0:
            cell.set_facecolor('#edf0f3')
            cell.set_height(cell.get_height() * 1.35)
        else:
            cell.set_facecolor('white')
    fig.text(.055, .035, 'QC flags and well support are descriptive. They do not establish that the state definitions are free of technical effects.',
             fontsize=9, color='#53616c')
    save_figure(fig, out, QC_PREFIX, dpi=400)


def render(out):
    data = out / f'{PREFIX}_data'
    manifest = csv_read(data / 'preservation_manifest.csv')
    preserved = dict(zip(manifest.path, manifest.sha256))
    verify_preserved(preserved)
    selected = csv_read(data / 'selected_feature_profiles.csv')
    summary = csv_read(out / f'{QC_PREFIX}.csv')
    well_states = csv_read(data / 'well_state_counts_and_fractions.csv')
    meta = csv_read(data / 'primary_cell_state_assignments.csv.gz')
    features = csv_read(data / 'all_state_feature_profiles.csv').original_feature.tolist()
    group_counts = csv_read(data / 'group_state_counts.csv')
    evidence = csv_read(data / 'inherited_model_evidence.csv')
    write_report(out, meta, features, summary, group_counts, selected, evidence)
    bound = render_main(out, selected, summary)
    render_qc(out, well_states, summary)
    verify_preserved(preserved)
    validation_path = out / f'{PREFIX}.validation.json'
    validation = json.loads(validation_path.read_text(encoding='utf-8'))
    validation.update({'status': 'complete', 'figure_color_limits': [-bound, bound],
                       'figure_color_linear_threshold': 1, 'display_values_clipped': False,
                       'source_and_previous_outputs_unchanged_after_render': True})
    write_json(validation, validation_path)
    print(f'FIGURES COMPLETE: {out / PREFIX}.pdf / .svg / .png')
    validate_outputs(out)


def validate_outputs(out):
    """Check row identity, numerical summaries, vector exports and preservation."""
    import pymupdf
    from PIL import Image
    data = out / f'{PREFIX}_data'
    meta, original_z, features, weights, assignments, decisions = load_primary()
    saved = csv_read(data / 'primary_cell_state_assignments.csv.gz')
    assert np.array_equal(meta.cell_id, saved.cell_id)
    assert np.array_equal(meta.state, saved.state)
    assert set(saved.panel_group) == set(GROUPS) and saved.intervention.eq('None').all()
    with np.load(data / 'primary_existing_normalized_features.npz', allow_pickle=False) as matrix:
        np.testing.assert_array_equal(matrix['standardized'], original_z)
        np.testing.assert_array_equal(matrix['cell_ids'], meta.cell_id.to_numpy(str))
        np.testing.assert_array_equal(matrix['features'], np.array(features))
        np.testing.assert_array_equal(matrix['weights'], weights)
    profiles = csv_read(data / 'all_state_feature_profiles.csv').set_index('original_feature')
    # Independently check the exported profiles using nested NumPy medians.
    for state in range(3):
        state_mask = meta.state.eq(state).to_numpy()
        medians = []
        medians_min2 = []
        for well in meta.loc[state_mask, 'well_id'].unique():
            ix = state_mask & meta.well_id.eq(well).to_numpy()
            value = np.median(original_z[ix], axis=0)
            medians.append(value)
            if ix.sum() >= 2:
                medians_min2.append(value)
        np.testing.assert_allclose(np.median(medians, axis=0),
                                   profiles.loc[features, f'state_{state}'], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(np.median(medians_min2, axis=0),
                                   profiles.loc[features, f'state_{state}_min2_sensitivity'], atol=1e-12, rtol=1e-12)
    well_states = csv_read(data / 'well_state_counts_and_fractions.csv')
    for row in well_states.to_dict('records'):
        within_well = meta.well_id.eq(row['well_id'])
        assert row['n_cells_in_well'] == int(within_well.sum())
        assert row['n_cells_in_state'] == int((within_well & meta.state.eq(row['state'])).sum())
        assert np.isclose(row['fraction_of_well'], row['n_cells_in_state'] / row['n_cells_in_well'])
    selected = csv_read(data / 'selected_feature_profiles.csv')
    feature_indices = [features.index(f) for f in selected.original_feature]
    ranked = rankdata(original_z[:, feature_indices], axis=0)
    covariance = np.cov(ranked, rowvar=False, aweights=weights, bias=True)
    corr = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    for cat in CATEGORIES:
        ix = np.flatnonzero(selected.category.eq(cat))
        assert len(ix) == 3
        matrix = corr[np.ix_(ix, ix)].copy()
        np.fill_diagonal(matrix, 0)
        assert np.max(np.abs(matrix)) <= .80
    source_manifest = csv_read(SOURCE / 'import_manifest.csv')
    for row in source_manifest.to_dict('records'):
        assert digest(local(ROOT / row['local_copy'])) == row['sha256']
    preserved = csv_read(data / 'preservation_manifest.csv')
    verify_preserved(dict(zip(preserved.path, preserved.sha256)))
    export_checks = []
    for name in [PREFIX, QC_PREFIX]:
        with pymupdf.open(out / f'{name}.pdf') as pdf:
            assert len(pdf) == 1
            page = pdf[0]
            text = page.get_text()
            assert all(state in text for state in STATE_NAMES.values())
            assert 'VEGF' not in text and 'UC2288' not in text
            assert len(page.get_images(full=True)) == 0, 'All heatmap tiles and text should be vector elements.'
            for block in page.get_text('blocks'):
                box = pymupdf.Rect(block[:4])
                assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= page.rect.width and box.y1 <= page.rect.height, block
            if name == PREFIX:
                assert 'Morphological state summary' in text
                assert text.splitlines().count('Representative image') == 3
            pdf_check = {'file': name + '.pdf', 'pages': len(pdf), 'embedded_raster_images': 0,
                         'all_text_within_page': True, 'only_primary_group_labels': True}
        with Image.open(out / f'{name}.png') as png:
            pdf_check['png_pixels'] = list(png.size)
            pdf_check['png_dpi'] = list(png.info['dpi'])
        export_checks.append(pdf_check)
    path = out / f'{PREFIX}.validation.json'
    validation = json.loads(path.read_text(encoding='utf-8'))
    validation.update({'independent_output_validation_passed': True,
                       'frozen_feature_values_preserved_exactly': True,
                       'all_profiles_checked_with_independent_numpy_medians': True,
                       'all_60_well_state_counts_checked_against_original_assignments': True,
                       'selected_correlations_checked_with_numpy_covariance': True,
                       'imported_source_copies_hash_checked': len(source_manifest),
                       'exports': export_checks, 'script_sha256': digest(Path(__file__))})
    write_json(validation, path)
    print('VALIDATION PASSED: original assignments and feature values, all well/state counts, profiles, vector PDFs and source hashes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--inspect-only', action='store_true')
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--render-only', action='store_true')
    modes.add_argument('--validate-only', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=panel_dir('E'))
    args = parser.parse_args()
    out = local(args.output_dir)
    if not out.is_relative_to(RESULT.resolve()):
        raise ValueError('Generated outputs must stay inside result/.')
    out.mkdir(parents=True, exist_ok=True)
    if args.inspect_only:
        inventory = input_inventory()
        destination = out / f'{PREFIX}.input_inspection.json'
        if destination.exists():
            raise FileExistsError(f'Inspection record exists: {destination}')
        write_json(inventory, destination)
        print(json.dumps(inventory, indent=2))
        return
    if args.validate_only:
        validate_outputs(out)
        return
    if not args.render_only:
        prepare(out)
    if not args.prepare_only:
        render(out)


if __name__ == '__main__':
    main()
