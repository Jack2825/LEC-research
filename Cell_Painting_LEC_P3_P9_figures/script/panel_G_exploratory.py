"""Exploratory Panel G from existing Cell Painting computational data only.

Reuse Panel C's frozen normalization/PCA/RMS metric, Panel D's fixed families,
and saved State 0-2 assignments. No feature search, new responder threshold,
clustering, experiment design or tube-assay analysis is performed.

--prepare-only writes auditable tables before plotting; --render-only plots
those tables. Use a fresh --output-dir inside result/ for another analysis.
"""
from pathlib import Path
from figure_paths import panel_dir, resolve_result_path, assert_preserved
import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
sys.dont_write_bytecode = True
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(variable, '1')
import numpy as np
import pandas as pd
import matplotlib

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'result'
os.environ['MPLCONFIGDIR'] = str(RESULT / '.matplotlib_cache')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

PASSAGES = ['P3', 'P9']
INTERVENTIONS = ['VEGF-C', 'UC2288']
CONDITIONS = ['Control', 'H1N1', 'VEGF-C', 'H1N1+VEGF-C', 'UC2288', 'H1N1+UC2288']
FAMILIES = ['Global', 'Whole-cell morphology', 'Nuclear morphology / intensity', 'ER', 'Mitochondria']
SHORT_FAMILIES = ['Global', 'Cell', 'Nucleus', 'ER', 'Mitochondria*']
COLORS = {'P3': '#0072B2', 'P9': '#D55E00'}
MARKERS = {'None': '^', 'VEGF-C': 's', 'UC2288': 'D'}
STATE_COLORS = ['#d58120', '#327aa4', '#6d439b']
SOURCES = {
    'existing_all_condition_cells': RESULT / 'Panel_E/panel_E_source/cells_all_features.csv.gz',
    'existing_state_assignments': RESULT / 'Panel_E/panel_E_source/cell_cluster_assignments.csv.gz',
    'primary_normalization_and_features': RESULT / 'Panel_C/panel_C_data/normalized_features.npz',
    'primary_feature_QC_and_transforms': RESULT / 'Panel_C/panel_C_feature_QC.csv',
    'frozen_primary_PCA': RESULT / 'Panel_C/panel_C_data/pca_model.npz',
    'existing_primary_PCA_well_centroids': RESULT / 'Panel_C/panel_C_data/pca_well_centroids.csv',
    'existing_primary_displacements': RESULT / 'Panel_C/panel_C.displacement.csv',
    'predefined_feature_families': RESULT / 'Panel_D/panel_D.feature_families.csv',
    'existing_primary_state_wells': RESULT / 'Panel_F/panel_F_well_summary.csv',
}
CONFIG = {
    'scope': 'Exploratory Cell Painting only; separate from tube formation',
    'main_comparisons': [f'{p}: H1N1+{i} minus H1N1' for p in PASSAGES for i in INTERVENTIONS],
    'supplementary_comparisons': [f'{p}: {i} minus Control' for p in PASSAGES for i in INTERVENTIONS],
    'normalization_and_PCA': 'frozen from the four-primary-group Panel C fit; applied without refitting',
    'features': 'all 98 retained Panel C measurements; no new selection',
    'well_profile': 'arithmetic mean of normalized cell features within each well',
    'main_metric': 'RMS difference between equal-well mean feature vectors',
    'secondary_metrics': 'same RMS metric in four predefined families, plus mean-well State 0/1/2 fraction differences',
    'permutation': 'all 252 allocations of 10 whole wells to two groups of five; frozen representation',
    'permutation_interpretation': 'exhaustive enumeration conditional on the chosen representation and well-exchangeability assumption',
    'FDR_families': {'H1N1_exploratory_comparisons': 32, 'Control_context_comparisons': 32},
    'FDR_method': 'Benjamini-Hochberg within each declared family',
    'well_consistency': 'omit each of the 10 comparison wells; RMS range and cosine agreement with the full contrast',
    'range_is_confidence_interval': False, 'new_responder_definition': False,
    'refit_clustering': False, 'mechanistic_state_labels': False,
    'main_display': 'two centered well-PCA views plus four global RMS comparisons',
}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .8,
                     'savefig.facecolor': 'white', 'xtick.labelsize': 9, 'ytick.labelsize': 9})


def sha(path):
    with resolve_result_path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path, **kwargs):
    path = resolve_result_path(path)
    assert path.is_relative_to(ROOT.resolve()), 'Input outside the permitted figure project.'
    return pd.read_csv(path, keep_default_na=False, **kwargs)


def dump(value, path):
    def cast(item):
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, Path): return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=cast), encoding='utf-8')


def bh(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind='stable')
    adjusted = values[order] * len(values) / np.arange(1, len(values)+1)
    result = np.empty_like(values)
    result[order] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    return result


def preserve(out):
    return {str(p.resolve()): sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and '.matplotlib_cache' not in p.parts and '__pycache__' not in p.parts
            and p.resolve() != Path(__file__).resolve()
            and not (p.parent.resolve() == out and p.name.startswith('panel_G'))
            and not p.resolve().is_relative_to(out / 'panel_G_data')
            and not p.resolve().is_relative_to(out / 'panel_G_supplement')}


def verify_preserved(manifest):
    for path, expected in manifest.items():
        assert_preserved(path, expected)


def condition_label(plasma, intervention):
    if intervention == 'None': return plasma
    return intervention if plasma == 'Control' else f'{plasma}+{intervention}'


def load_existing():
    cells = read(SOURCES['existing_all_condition_cells'])
    labels = read(SOURCES['existing_state_assignments'])
    assert cells.cell_id.is_unique and labels.cell_id.is_unique
    linked = labels.set_index('cell_id').loc[cells.cell_id].reset_index()
    for column in ['cell_id', 'well_id', 'passage', 'plasma', 'intervention', 'physical_well',
                   'ImageNumber', 'ObjectNumber', 'source_nd2_filename', 'source_nd2_md5']:
        assert cells[column].astype(str).reset_index(drop=True).equals(linked[column].astype(str)), column
    cells['state'] = linked.All__k3.to_numpy(int)
    assert set(cells.state) == {0, 1, 2}
    with np.load(SOURCES['primary_normalization_and_features'], allow_pickle=False) as a:
        features = a['features'].astype(str).tolist()
        centers, scales = a['centers'].copy(), a['scales'].copy()
        primary_ids, primary_z = a['cell_ids'].astype(str), a['standardized'].copy()
    assert len(features) == 98
    qc = read(SOURCES['primary_feature_QC_and_transforms']).set_index('original_feature').loc[features]
    assert qc.retained.eq(True).all()
    raw = cells[features].to_numpy(float)
    assert np.isfinite(raw).all(), 'Unexpected nonfinite input; do not silently drop measurements.'
    transformed = raw.copy()
    for j, row in enumerate(qc.itertuples()):
        if row.transformation == 'linear':
            pass
        elif row.transformation.startswith('log1p('):
            divisor = float(row.transform_divisor)
            assert divisor > 0 and np.all(raw[:, j] >= 0)
            transformed[:, j] = np.log1p(raw[:, j] / divisor)
        else:
            raise ValueError('Unrecognized frozen transform: ' + row.transformation)
    z = (transformed - centers) / scales
    assert np.isfinite(z).all()
    primary_indices = cells.set_index('cell_id').index.get_indexer(primary_ids)
    assert np.all(primary_indices >= 0)
    np.testing.assert_allclose(z[primary_indices], primary_z, atol=1e-10, rtol=1e-12)
    with np.load(SOURCES['frozen_primary_PCA'], allow_pickle=False) as a:
        np.testing.assert_array_equal(a['features'].astype(str), np.array(features))
        components, pc_center, variance = a['components'].copy(), a['center'].copy(), a['variance'].copy()
    pc = (z - pc_center) @ components
    family_map = read(SOURCES['predefined_feature_families']).set_index('original_feature').loc[features]
    family_indices = [np.arange(len(features))] + [np.flatnonzero(family_map.family.eq(f).to_numpy()) for f in FAMILIES[1:]]
    assert [len(ix) for ix in family_indices] == [98, 40, 43, 5, 10]
    metadata = ['cell_id', 'well_id', 'passage', 'plasma', 'intervention', 'physical_well',
                'source_nd2_filename', 'qc_edge_touching', 'qc_saturated_maximum', 'qc_extreme_z8', 'state']
    meta = cells[metadata].copy()
    meta['condition'] = [condition_label(p, i) for p, i in zip(meta.plasma, meta.intervention)]
    meta['display_group'] = meta.passage + '-' + meta.condition
    info = meta.drop(columns=['cell_id', 'state', 'qc_edge_touching', 'qc_saturated_maximum', 'qc_extreme_z8']).drop_duplicates('well_id').copy()
    info['condition_order'] = info.condition.map({c: i for i, c in enumerate(CONDITIONS)})
    info = info.sort_values(['passage', 'condition_order', 'physical_well']).reset_index(drop=True)
    assert len(info) == 60 and info.groupby('display_group').size().eq(5).all()
    grouped = {well: np.flatnonzero(meta.well_id.eq(well).to_numpy()) for well in info.well_id}
    means = np.stack([z[grouped[well]].mean(axis=0) for well in info.well_id])
    pc_means = np.stack([pc[grouped[well]].mean(axis=0) for well in info.well_id])
    info['n_cells'] = [len(grouped[w]) for w in info.well_id]
    for state in range(3):
        info[f'state_{state}_cells'] = [int(meta.iloc[grouped[w]].state.eq(state).sum()) for w in info.well_id]
        info[f'state_{state}_fraction'] = info[f'state_{state}_cells'] / info.n_cells
    for qc_flag in ['qc_edge_touching', 'qc_saturated_maximum', 'qc_extreme_z8']:
        info['n_' + qc_flag] = [int(meta.iloc[grouped[w]][qc_flag].sum()) for w in info.well_id]
    info = pd.concat([info, pd.DataFrame(pc_means, columns=[f'PC{j+1}' for j in range(pc_means.shape[1])])], axis=1)
    old_pc = read(SOURCES['existing_primary_PCA_well_centroids']).set_index('well_id')
    for well in old_pc.index:
        current = info.set_index('well_id').loc[well]
        np.testing.assert_allclose(current[[f'PC{j+1}' for j in range(98)]].to_numpy(float),
                                   old_pc.loc[well, [f'PC{j+1}' for j in range(98)]].to_numpy(float), atol=1e-10)
    prior_state = read(SOURCES['existing_primary_state_wells']).set_index('well_id')
    current = info.set_index('well_id')
    for state in range(3):
        np.testing.assert_allclose(current.loc[prior_state.index, f'state_{state}_fraction'],
                                   prior_state[f'state_{state}_fraction'], atol=1e-15)
    for passage in PASSAGES:
        reference = info.passage.eq(passage) & info.plasma.eq('H1N1') & info.intervention.eq('None')
        for j in [1, 2]:
            center = info.loc[reference, f'PC{j}'].mean()
            info.loc[info.passage.eq(passage), f'PC{j}_relative_to_H1N1_mean'] = info.loc[info.passage.eq(passage), f'PC{j}'] - center
    old_d = read(SOURCES['existing_primary_displacements']).set_index('passage')
    for passage in PASSAGES:
        a = info.passage.eq(passage) & info.plasma.eq('Control') & info.intervention.eq('None')
        b = info.passage.eq(passage) & info.plasma.eq('H1N1') & info.intervention.eq('None')
        value = np.sqrt(np.mean((means[b].mean(axis=0) - means[a].mean(axis=0))**2))
        np.testing.assert_allclose(value, old_d.loc[passage, 'RMS_displacement'], atol=1e-12)
    return meta, info, z, means, features, family_indices, variance


def analyze(info, means, features, family_indices):
    stats, permutations, omissions, vectors = [], [], [], []
    state_values = info[[f'state_{s}_fraction' for s in range(3)]].to_numpy(float)
    for plasma in ['H1N1', 'Control']:
        family_name = 'H1N1_exploratory_comparisons' if plasma == 'H1N1' else 'Control_context_comparisons'
        for passage in PASSAGES:
            for intervention in INTERVENTIONS:
                comparison = f'{passage}__{plasma}__{intervention}'
                a = np.flatnonzero((info.passage.eq(passage) & info.plasma.eq(plasma) & info.intervention.eq('None')).to_numpy())
                b = np.flatnonzero((info.passage.eq(passage) & info.plasma.eq(plasma) & info.intervention.eq(intervention)).to_numpy())
                assert len(a) == len(b) == 5
                ix = np.r_[a, b]
                y, state_y = means[ix], state_values[ix]
                delta = y[5:].mean(axis=0) - y[:5].mean(axis=0)
                state_delta = 100 * (state_y[5:].mean(axis=0) - state_y[:5].mean(axis=0))
                observed = np.r_[[np.sqrt(np.mean(delta[f]**2)) for f in family_indices], state_delta]
                null = []
                for draw, selected in enumerate(itertools.combinations(range(10), 5)):
                    is_b = np.zeros(10, bool); is_b[list(selected)] = True
                    d = y[is_b].mean(axis=0) - y[~is_b].mean(axis=0)
                    ds = 100 * (state_y[is_b].mean(axis=0) - state_y[~is_b].mean(axis=0))
                    values = np.r_[[np.sqrt(np.mean(d[f]**2)) for f in family_indices], ds]
                    null.append(values)
                    permutations.append({'comparison': comparison, 'allocation': draw,
                                         'hypothetical_target_well_positions': '|'.join(map(str, selected)),
                                         **{f'metric_{j}': value for j, value in enumerate(values)}})
                null = np.asarray(null)
                np.testing.assert_allclose(null[-1], observed, atol=1e-12)
                p = np.mean(np.abs(null) >= np.abs(observed)[None, :] - 1e-12*np.maximum(1, np.abs(observed))[None, :], axis=0)
                loo = []
                for omitted in range(10):
                    aa = np.arange(5); bb = np.arange(5, 10)
                    aa = aa[aa != omitted]; bb = bb[bb != omitted]
                    d = y[bb].mean(axis=0) - y[aa].mean(axis=0)
                    ds = 100 * (state_y[bb].mean(axis=0) - state_y[aa].mean(axis=0))
                    vals = np.r_[[np.sqrt(np.mean(d[f]**2)) for f in family_indices], ds]
                    cosines = []
                    for f in family_indices:
                        denom = np.linalg.norm(d[f]) * np.linalg.norm(delta[f])
                        cosines.append(float(np.dot(d[f], delta[f]) / denom) if denom > 1e-15 else np.nan)
                    loo.append(vals)
                    for j, value in enumerate(vals):
                        omissions.append({'comparison': comparison, 'omitted_well': info.iloc[ix[omitted]].well_id,
                                          'omitted_group': 'reference' if omitted < 5 else 'target',
                                          'metric_index': j, 'estimate': value,
                                          'cosine_to_full_response_vector': cosines[j] if j < 5 else np.nan})
                loo = np.asarray(loo)
                for feature, value in zip(features, delta):
                    vectors.append({'comparison': comparison, 'feature': feature, 'target_minus_reference': value})
                for j, value in enumerate(observed):
                    metric = FAMILIES[j] if j < 5 else f'State {j-5}'
                    relevant = [r for r in omissions if r['comparison'] == comparison and r['metric_index'] == j]
                    most = int(np.argmax(np.abs(loo[:, j] - value)))
                    row = {'comparison': comparison, 'passage': passage, 'plasma': plasma, 'intervention': intervention,
                           'reference_condition': condition_label(plasma, 'None'),
                           'target_condition': condition_label(plasma, intervention),
                           'analysis_role': 'main_G_global' if plasma == 'H1N1' and j == 0 else 'supplementary',
                           'metric': metric, 'metric_index': j, 'n_features': len(family_indices[j]) if j < 5 else 0,
                           'estimate': float(value), 'unit': 'normalized feature RMS units' if j < 5 else 'percentage points',
                           'estimate_definition': 'RMS of target-minus-reference mean-well feature vector' if j < 5 else 'target-minus-reference mean-well state fraction',
                           'n_reference_wells': 5, 'n_target_wells': 5,
                           'n_reference_cells': int(info.iloc[a].n_cells.sum()), 'n_target_cells': int(info.iloc[b].n_cells.sum()),
                           'p_permutation': p[j], 'permutation_allocations': 252, 'FDR_family': family_name,
                           'LOO_min': float(loo[:, j].min()), 'LOO_max': float(loo[:, j].max()),
                           'max_absolute_LOO_change': float(np.max(np.abs(loo[:, j] - value))),
                           'most_influential_well': info.iloc[ix[most]].well_id,
                           'LOO_min_vector_cosine': min(r['cosine_to_full_response_vector'] for r in relevant) if j < 5 else np.nan,
                           'LOO_state_effect_same_sign_fraction': float(np.mean(np.sign(loo[:, j]) == np.sign(value))) if j >= 5 else np.nan,
                           'LOO_range_is_confidence_interval': False}
                    if j >= 5:
                        row['reference_mean_state_fraction'] = float(state_y[:5, j-5].mean())
                        row['target_mean_state_fraction'] = float(state_y[5:, j-5].mean())
                    else:
                        row['reference_mean_state_fraction'] = np.nan
                        row['target_mean_state_fraction'] = np.nan
                    stats.append(row)
                print('Completed fixed comparisons and well checks:', comparison, flush=True)
    statistics = pd.DataFrame(stats)
    statistics['q_BH'] = np.nan
    for name, group in statistics.groupby('FDR_family'):
        assert len(group) == CONFIG['FDR_families'][name] == 32
        statistics.loc[group.index, 'q_BH'] = bh(group.p_permutation)
    return statistics, pd.DataFrame(permutations), pd.DataFrame(omissions), pd.DataFrame(vectors)


def prepare(out):
    data = out / 'panel_G_data'
    supplement = out / 'panel_G_supplement'
    if (out / 'panel_G.validation.json').exists() or (data.exists() and any(data.iterdir())):
        raise FileExistsError('Panel G data exists; use a new --output-dir.')
    preserved = preserve(out)
    data.mkdir(parents=True, exist_ok=True)
    supplement.mkdir(parents=True, exist_ok=True)
    dump(CONFIG, out / 'panel_G.preanalysis_plan.json')
    meta, info, z, means, features, family_indices, variance = load_existing()
    statistics, nulls, omissions, vectors = analyze(info, means, features, family_indices)
    counts = info.groupby(['passage', 'plasma', 'intervention', 'condition'], sort=False).agg(
        wells=('well_id', 'size'), cells=('n_cells', 'sum'), minimum_cells_per_well=('n_cells', 'min'),
        maximum_cells_per_well=('n_cells', 'max')).reset_index()
    info.to_csv(data / 'well_summaries_and_frozen_PCA.csv', index=False)
    meta.to_csv(data / 'cell_metadata_and_existing_states.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    np.savez_compressed(data / 'frozen_normalized_features.npz', standardized=z, features=np.array(features), cell_ids=meta.cell_id.to_numpy(str))
    well_profiles = pd.DataFrame(means, columns=features)
    well_profiles.insert(0, 'well_id', info.well_id)
    well_profiles.to_csv(data / 'well_mean_feature_profiles.csv', index=False)
    counts.to_csv(data / 'condition_counts.csv', index=False)
    statistics.to_csv(out / 'panel_G_statistics.csv', index=False)
    omissions.to_csv(data / 'leave_one_well_out.csv', index=False)
    nulls.to_csv(data / 'permutation_reference.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    vectors.to_csv(data / 'all_feature_contrast_vectors.csv', index=False)
    dump({'variance_ratio': variance.tolist(), 'fit_cohort': 'four primary groups; parameters frozen from Panel C'}, data / 'PCA_display_metadata.json')
    sources = pd.DataFrame([{'role': role, 'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for role, path in SOURCES.items()])
    sources.to_csv(data / 'source_manifest.csv', index=False)
    pd.DataFrame([{'path': p, 'sha256': h} for p, h in preserved.items()]).to_csv(data / 'preservation_manifest.csv', index=False)
    with pd.ExcelWriter(out / 'panel_G_statistics.xlsx', engine='openpyxl') as writer:
        for name, frame in [('Main G global', statistics.loc[statistics.analysis_role.eq('main_G_global')]),
                            ('H1N1 exploratory comparisons', statistics.loc[statistics.plasma.eq('H1N1')]),
                            ('Control context comparisons', statistics.loc[statistics.plasma.eq('Control')]),
                            ('All condition counts', counts), ('All well summaries', info),
                            ('Omit-one-well sensitivity', omissions), ('Source files', sources)]:
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]; sheet.freeze_panes = 'A2'; sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(65, max(12, max(len(str(c.value or '')) for c in column)+2))
    write_reports(out, counts, statistics, variance)
    verify_preserved(preserved)
    dump({'status': 'prepared; figures not yet rendered', 'all_cells': len(meta), 'all_wells': len(info),
          'main_G_cells': int(info.loc[info.plasma.eq('H1N1'), 'n_cells'].sum()),
          'main_G_wells': int(info.plasma.eq('H1N1').sum()), 'features': len(features),
          'normalization_and_PCA_refitted': False, 'state_assignments_changed': False,
          'primary_C_feature_values_PCA_and_displacement_reproduced': True,
          'primary_F_state_fractions_reproduced': True, 'new_responder_definition': False,
          'statistics_rows': len(statistics), 'primary_exploratory_FDR_tests': 32,
          'supplementary_Control_context_FDR_tests': 32, 'previous_files_hash_checked': len(preserved),
          'previous_files_unchanged': True, 'all_analysis_inputs_inside_figure_project': True,
          'software': {'python': platform.python_version(), 'numpy': np.__version__,
                       'pandas': pd.__version__, 'matplotlib': matplotlib.__version__}}, out / 'panel_G.validation.json')
    print('MAIN G:', flush=True)
    print(statistics.loc[statistics.analysis_role.eq('main_G_global'), ['comparison', 'estimate', 'p_permutation', 'q_BH', 'LOO_min', 'LOO_max', 'LOO_min_vector_cosine', 'most_influential_well']].to_string(index=False), flush=True)


def table_md(frame):
    columns = list(frame.columns)
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.to_dict('records'):
        lines.append('| '+' | '.join(str(row[c]) for c in columns)+' |')
    return '\n'.join(lines)


def write_reports(out, counts, statistics, variance):
    main_rows = statistics.loc[statistics.analysis_role.eq('main_G_global')].copy()
    readable = main_rows[['comparison', 'estimate', 'p_permutation', 'q_BH', 'LOO_min', 'LOO_max', 'LOO_min_vector_cosine']].copy()
    for col in readable.columns[1:]: readable[col] = readable[col].map(lambda x: f'{float(x):.6g}')
    available = counts.copy()
    text = f'''# Panel G: exploratory VEGF-C / UC2288 modulation

## Scope and existing inputs

This is an **exploratory Cell Painting analysis** of existing metadata groups. These perturbations are not presented as part of the separate tube-formation experiment. The primary analyses in Panels C-F remain unchanged. Their global tests did not establish an FDR-significant primary morphological response, so this panel does not claim restoration, rescue, or a confirmed modification of a validated primary phenotype.

All inputs are inside `Cell_Painting_LEC_P3_P9_figures`. Cell measurements and existing All__k3 assignments are taken from the byte-identical computational output copies in `result/Panel_E/panel_E_source/`. The 98 retained features, 93 log transforms/5 linear transforms, transform divisors, centers and scales are frozen from Panel C. All new groups are mapped into that same feature space without recalculating any preprocessing parameters or selecting new features. The primary cells' transformed feature values, PCA well coordinates, RMS displacements, and Panel F state fractions are reproduced as checks.

The original three-state model used its own 97-feature all-condition representation, as documented in Panel E. Its saved labels are reused exactly here; the 98-feature response representation does not refit, rename or reassign those states. No responder classification is available in this figure project, and none is created.

## Conditions, cell counts and prespecified comparisons

{table_md(available)}

All conditions have five wells. Main comparisons, separately within P3 and P9, are H1N1+VEGF-C minus H1N1 and H1N1+UC2288 minus H1N1. These are four independent-group comparisons, not P3 versus P9. Wells are not paired by image number or plate position. The main PCA views contain the 30 H1N1-background wells (2,404 cells).

Supplementary context comparisons are VEGF-C minus Control and UC2288 minus Control, separately in P3/P9. These use the existing perturbation-only groups and are reported in a separate multiplicity family. They supply context for changes that may also occur without H1N1; pairwise results alone do not establish H1N1-specific modification. No new treatment conditions or biological procedures are introduced.

## Reused metrics and well-level inference

For each well w, let m_wj be the mean of its cells' frozen normalized values for feature j. Every well contributes equally to its group's mean, regardless of the number of cells. For each comparison, d_j = mean_target_wells(m_wj) - mean_reference_wells(m_wj). The global displacement is the Panel C definition:

    D = sqrt(sum_j(d_j**2) / 98).

The same RMS metric is calculated in the Panel D measurement families: Whole-cell morphology (40), Nuclear morphology/intensity (43), ER (5), and Mitochondria (10). These families are determined from existing CellProfiler names and are not selected using exploratory effects. No individual-feature significance screen is performed. All 98 contrast coordinates are exported for traceability.

For State 0/1/2, the reused Panel F endpoint is n_state_cells/n_analyzed_cells within each well. The contrast is 100 times the target-minus-reference difference of the five-well mean fractions, reported in percentage points. The source labels retain their neutral computational names; no functional or mechanistic label is assigned.

For each comparison, all choose(10,5)=252 allocations of the ten whole-well profiles are enumerated with five target and five reference wells. RMS tests count permuted distances at least as large as the observed distance; state-fraction tests use the absolute signed difference. The fraction of all allocations meeting that criterion is the reported P value. Exhaustive enumeration uses no Monte Carlo approximation or +1 correction. Original cell and well assignments are unchanged in the observed-data exports; hypothetical allocations are stored only in the permutation-reference file.

BH correction covers **32 prespecified H1N1-background tests**: four comparisons times five RMS metrics and three state-occupancy metrics. The 32 Control-background context tests form a separate supplementary family. Panel G's displayed global q values come from the full 32-test H1N1 family, not just the four plotted results. These corrections do not claim experiment-wide FDR control across previous panels or across both families combined.

The permutation enumeration is conditional on the frozen representation and an assumption of exchangeable independent wells. The original H1N1 reference wells contributed to Panel C preprocessing; this is not validation in a wholly independent cohort. Frozen preprocessing is deliberately retained for comparability rather than refitted to exploratory groups. Together with fixed plate layouts, small replicate numbers and unverified donor/pool identity, this limits confirmatory interpretation. The numerical P values are exhaustive conditional permutation calculations, not a guarantee against these design or preprocessing limitations.

## Replicate consistency and displayed results

Every comparison is recomputed ten times, omitting one well at a time while keeping the feature representation fixed. The line beside each global RMS point shows the minimum and maximum of these ten estimates. These ranges are **not confidence intervals** and a positive lower bound is not evidence of a nonzero effect: distances are nonnegative and have a sampling-noise floor.

For each multivariate contrast, the cosine between every omitted-well contrast vector and the full contrast vector is also reported. A cosine near 1 indicates similar direction; low or negative values indicate changing patterns. This is a sensitivity description, not a validation threshold or biological conclusion. For state fractions, the supplementary table reports how often the signed mean difference retains its direction after omitting a well. The most influential omitted well is recorded for every metric.

{table_md(readable)}

## Figure reading and Supplementary Figure S1

Panel G has two small well-level PCA views (P3 and P9) and one four-row global RMS summary. The PCA transformation is exactly the Panel C fit: PC1 and PC2 explained {100*variance[0]:.2f}% and {100*variance[1]:.2f}% of variance in the primary fitting cohort, not necessarily the exploratory cohort. To show within-passage changes, both coordinates are translated by the same-passage H1N1 mean. Axes and scale are shared between passages. Each small symbol is one well; larger open symbols are equal-well group means. Arrows connect the H1N1 group mean to each exploratory group mean. Only the display uses two PCs; the RMS statistics use all 98 features. Visual separation on the two PCs is not the statistical test.

The prespecified main display is global RMS for all four requested comparisons, regardless of P value or consistency. Families or individual features are not promoted into Panel G after inspecting which appears strongest. The primary-study passage colors are retained: P3 blue and P9 orange; shapes distinguish the existing metadata conditions.

The `panel_G_supplement/` folder contains S1-ready detail figures: all five RMS summaries and all three state-fraction changes for the four H1N1 comparisons; corresponding Control-background comparisons; and individual-well State 0/1/2 proportions for all 12 groups. `S1_exploratory_details.pdf` combines those figures without rasterizing the vector plots. The full statistics workbook, well summaries, exact permutation references, omission results and full feature contrast vectors remain available. Family labels follow source annotations; the existing CY5/TRITC annotation discrepancy remains unresolved. These results do not support a new organelle mechanism or functional claim.

## Reproduction and preservation

Run `script/panel_G_exploratory.py --prepare-only`, inspect the tables, then `--render-only`. Default execution performs both steps. A completed analysis is not overwritten by default; provide a new `--output-dir` inside result for another run. --render-only updates this panel's figures and text from its saved tables. SHA-256 checks preserve all input files and previous panels. No source microscopy is modified and no synthetic microscopy is generated.
'''
    (out / 'panel_G.methods.md').write_text(text, encoding='utf-8')
    inspection = f'''# Panel G prefigure report

1. All exploratory conditions have five wells. Counts are below.
2. Reused outputs: Panel C frozen 98-feature normalization/PCA and RMS metric; Panel D feature families; Panels E-F existing State 0/1/2 labels and within-well fractions. No responder classification is available or created.
3. Main comparisons: P3 and P9 H1N1+VEGF-C versus same-passage H1N1, and H1N1+UC2288 versus same-passage H1N1.
4. Replicate consistency is reported by omit-one-well RMS ranges and contrast-vector cosine agreement; no replicate is removed from the main calculation.
5. Compact display: two centered well-PCA views plus four global RMS estimates, omit-one-well ranges and BH q values. Clearly labeled Exploratory / Cell Painting.
6. S1: all compartment RMS results, all three state-occupancy differences, perturbation-only Control context comparisons, and all 12 conditions' individual-well state proportions.

{table_md(available)}

{table_md(readable)}
'''
    (out / 'panel_G.inspection.md').write_text(inspection, encoding='utf-8')
    (out / 'panel_G_statistics.md').write_text('# Exploratory Panel G global comparisons\n\n'+table_md(readable)+'\n\nFull tables contain all 32 H1N1-background tests and 32 separate Control-context tests. Omit-one-well ranges are not confidence intervals.\n', encoding='utf-8')
    legend = f'''**G. Exploratory VEGF-C / UC2288 modulation in the separate Cell Painting experiment.** For each passage, existing H1N1+VEGF-C and H1N1+UC2288 groups are compared with H1N1 alone (five wells per group). Left: existing well profiles projected into the frozen Panel C PCA and centered on the same-passage H1N1 mean. Small symbols denote wells; larger open symbols are group means. PC1/PC2 variance labels refer to the primary fitting cohort. Right: RMS difference between equal-well group means across all 98 frozen normalized features. Lines show the ten omit-one-well estimates' range, not confidence intervals. P values enumerate all 252 whole-well allocations conditional on the fixed representation; displayed BH q values cover 32 prespecified exploratory H1N1-background tests. Full compartment and existing-state summaries, Control-background context comparisons and replicate checks are supplied for Supplementary Figure S1. These exploratory differences do not establish H1N1-specific restoration, a confirmed mechanism, or participation of these perturbations in the tube-formation experiment.
'''
    (out / 'panel_G.legend.md').write_text(legend, encoding='utf-8')
    (out / 'panel_G_supplement/S1.legend.md').write_text('''**Supplementary Figure S1. Exploratory Cell Painting detail.** First two pages show all predefined RMS feature-family differences and existing State 0/1/2 mean-well fraction differences for H1N1-background and Control-background comparisons, respectively. Numeric values show normalized-feature RMS displacement or signed percentage-point differences; q values use BH within the declared 32-test family for that background. Shared color ranges are used across the two pages for each metric type. The third page shows individual wells for all 12 existing metadata groups; short bars are equal-well means. Cell and well counts, exact conditional permutation P values and omit-one-well sensitivity results are in the statistics workbook. The original state labels and feature preprocessing are unchanged. *Mitochondria follows the existing source annotation; CY5/TRITC annotation remains unresolved. These are exploratory computational summaries of a separate Cell Painting experiment, without inferred mechanisms or cell functions.
''', encoding='utf-8')


def save_figure(fig, folder, name, dpi=600):
    for suffix in ['pdf', 'svg']: fig.savefig(folder / f'{name}.{suffix}', facecolor='white')
    fig.savefig(folder / f'{name}.png', dpi=dpi, facecolor='white')
    fig.savefig(folder / f'{name}.preview.png', dpi=140, facecolor='white')
    plt.close(fig)


def render_main(out, info, statistics, variance):
    wells = info.loc[info.plasma.eq('H1N1')].copy()
    main = statistics.loc[statistics.analysis_role.eq('main_G_global')].reset_index(drop=True)
    coords = wells[['PC1_relative_to_H1N1_mean', 'PC2_relative_to_H1N1_mean']].to_numpy(float)
    bound = max(.25, float(np.abs(coords).max()) * 1.19)
    fig = plt.figure(figsize=(13.4, 5.3))
    fig.text(.016, .958, 'G', fontsize=21, fontweight='bold', va='top')
    fig.text(.061, .954, 'Exploratory VEGF-C / UC2288 modulation', fontsize=15, fontweight='bold', va='top')
    fig.text(.061, .893, 'Cell Painting · 5 wells per group · frozen primary-study feature space', fontsize=10, color='#53616c', va='top')
    for pi, passage in enumerate(PASSAGES):
        ax = fig.add_axes([.066 + pi * .260, .294, .226, .488])
        ax.set_aspect('equal', adjustable='box')
        ax.axhline(0, color='#b8c0c6', lw=.7, ls='--', zorder=0)
        ax.axvline(0, color='#b8c0c6', lw=.7, ls='--', zorder=0)
        for intervention in ['None', *INTERVENTIONS]:
            sub = wells.loc[wells.passage.eq(passage) & wells.intervention.eq(intervention)]
            x = sub.PC1_relative_to_H1N1_mean.to_numpy(float)
            y = sub.PC2_relative_to_H1N1_mean.to_numpy(float)
            ax.scatter(x, y, c=COLORS[passage], marker=MARKERS[intervention], s=37,
                       alpha=.87, edgecolors='white', linewidths=.5, zorder=3)
            if intervention != 'None':
                ax.annotate('', xy=(x.mean(), y.mean()), xytext=(0, 0),
                            arrowprops={'arrowstyle': '-|>', 'color': COLORS[passage], 'lw': 1.2, 'alpha': .65}, zorder=2)
            ax.scatter([x.mean()], [y.mean()], facecolors='none', edgecolors=COLORS[passage],
                       marker=MARKERS[intervention], s=112, linewidths=1.5, zorder=4)
        ax.set_xlim(-bound, bound); ax.set_ylim(-bound, bound)
        ax.set_title(passage, color=COLORS[passage], fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel(f'Centered PC1 ({100*variance[0]:.1f}% in primary fit)', fontsize=8.8)
        if pi == 0: ax.set_ylabel(f'Centered PC2 ({100*variance[1]:.1f}% in primary fit)', fontsize=8.8)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([], [], marker=MARKERS[i], color='#45535e', linestyle='None', markersize=6,
                      label='H1N1' if i == 'None' else 'H1N1+'+i) for i in ['None', *INTERVENTIONS]]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.060, .141), ncol=3, frameon=False,
               fontsize=9, handletextpad=.4, columnspacing=1.05)
    forest = fig.add_axes([.723, .319, .166, .437])
    upper = float(main.LOO_max.max()) * 1.13
    forest.set_xlim(0, upper)
    forest.set_ylim(3.65, -.7)
    forest.set_yticks(range(4), [f"{r.passage}: +{r.intervention}" for r in main.itertuples()], fontsize=9.2)
    for y, row in main.iterrows():
        forest.plot([row.LOO_min, row.LOO_max], [y, y], color=COLORS[row.passage], lw=2)
        forest.plot([row.LOO_min, row.LOO_min], [y-.055, y+.055], color=COLORS[row.passage], lw=1)
        forest.plot([row.LOO_max, row.LOO_max], [y-.055, y+.055], color=COLORS[row.passage], lw=1)
        forest.scatter([row.estimate], [y], s=49, marker='D', color=COLORS[row.passage], zorder=3,
                       edgecolors='white', linewidths=.5)
        forest.text(1.11, y, f'{row.q_BH:.3f}', transform=forest.get_yaxis_transform(),
                    va='center', ha='center', fontsize=9.2)
    forest.text(1.11, -.67, 'BH q', transform=forest.get_yaxis_transform(), ha='center', fontsize=9, fontweight='bold')
    forest.set_title('Global displacement', fontsize=11.5, fontweight='bold', pad=20)
    forest.set_xlabel('RMS difference from H1N1\n(all 98 features)', fontsize=9.2, labelpad=9)
    forest.spines[['top', 'right', 'left']].set_visible(False)
    forest.tick_params(axis='y', length=0, pad=10)
    forest.axhline(1.5, color='#dce1e5', lw=.7)
    fig.text(.061, .083, 'Small symbols: wells; open symbols: group means. PCA is centered on the same-passage H1N1 mean.', fontsize=8.7, color='#53616c')
    fig.text(.061, .039, 'RMS lines: omit-one-well ranges, not confidence intervals. q: BH across 32 exploratory H1N1-background tests.', fontsize=8.7, color='#53616c')
    save_figure(fig, out, 'panel_G')


def render_family_detail(folder, statistics, plasma):
    data = statistics.loc[statistics.plasma.eq(plasma)]
    comparisons = [f'{p}__{plasma}__{i}' for p in PASSAGES for i in INTERVENTIONS]
    family = np.array([[data.loc[data.comparison.eq(c) & data.metric.eq(f), 'estimate'].iloc[0] for c in comparisons] for f in FAMILIES])
    state = np.array([[data.loc[data.comparison.eq(c) & data.metric.eq(f'State {s}'), 'estimate'].iloc[0] for c in comparisons] for s in range(3)])
    qf = np.array([[data.loc[data.comparison.eq(c) & data.metric.eq(f), 'q_BH'].iloc[0] for c in comparisons] for f in FAMILIES])
    qs = np.array([[data.loc[data.comparison.eq(c) & data.metric.eq(f'State {s}'), 'q_BH'].iloc[0] for c in comparisons] for s in range(3)])
    vmax = float(statistics.loc[statistics.metric_index.lt(5), 'estimate'].max())
    state_bound = max(1., float(statistics.loc[statistics.metric_index.ge(5), 'estimate'].abs().max()))
    fig = plt.figure(figsize=(12.6, 5.6))
    fig.text(.025, .957, f'S1 · Exploratory {plasma}-background comparisons', fontsize=15, fontweight='bold', va='top')
    fig.text(.025, .900, f'Existing Cell Painting groups: perturbation minus {plasma}; five wells per group', fontsize=10, color='#53616c')
    for side, values, qs_here, names, title, color in [
        (0, family, qf, SHORT_FAMILIES, 'Predefined feature families', 'Blues'),
        (1, state, qs, [f'State {s}' for s in range(3)], 'Existing state occupancy', 'RdBu_r')]:
        ax = fig.add_axes([.13 if side == 0 else .655, .284, .337 if side == 0 else .286, .506])
        norm = matplotlib.colors.Normalize(0, vmax) if side == 0 else TwoSlopeNorm(vmin=-state_bound, vcenter=0, vmax=state_bound)
        mesh = ax.pcolormesh(np.arange(5)-.5, np.arange(len(values)+1)-.5, values, cmap=color, norm=norm,
                             edgecolors='white', linewidth=.8, rasterized=False)
        ax.set_xlim(-.5, 3.5); ax.set_ylim(len(values)-.5, -.5)
        ax.set_xticks(range(4), [f'{p}\n+{i}' for p in PASSAGES for i in INTERVENTIONS], fontsize=8.6)
        ax.set_yticks(range(len(names)), names, fontsize=9.2)
        ax.tick_params(length=0, pad=7)
        ax.set_title(title, fontsize=11.5, fontweight='bold', pad=14)
        cmap = plt.get_cmap(color)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                rgba = cmap(norm(values[i, j])); light = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                number = f'{values[i,j]:.3f}' if side == 0 else f'{values[i,j]:+.2f}'
                ax.text(j, i, number+f'\nq={qs_here[i,j]:.3f}', ha='center', va='center', fontsize=8.4,
                        color='white' if light < .50 else '#17212b', linespacing=1.4)
        for spine in ax.spines.values(): spine.set_visible(False)
        cbax = fig.add_axes([.142 if side == 0 else .664, .150, .310 if side == 0 else .266, .022])
        bar = fig.colorbar(mesh, cax=cbax, orientation='horizontal'); bar.solids.set_rasterized(False)
        bar.ax.tick_params(labelsize=8)
        bar.set_label('Normalized feature RMS difference' if side == 0 else 'Mean-well fraction difference (percentage points)', fontsize=8.4, labelpad=5)
    fig.text(.025, .047, 'q values: BH across 32 tests for this background. No individual-feature significance screen. All results are exploratory.', fontsize=8.5, color='#53616c')
    fig.text(.025, .017, '*Mitochondria follows the source measurement annotation; CY5/TRITC annotation is unresolved.', fontsize=8.3, color='#61707a')
    name = 'S1_H1N1_comparisons' if plasma == 'H1N1' else 'S1_Control_context_comparisons'
    save_figure(fig, folder, name, dpi=400)


def render_state_detail(folder, info):
    fig = plt.figure(figsize=(13.5, 7.6))
    fig.text(.026, .969, 'S1 · Exploratory state occupancy across all existing Cell Painting groups', fontsize=15, fontweight='bold', va='top')
    fig.text(.026, .923, 'Each point is one well; short bars are equal-well means. Existing State 0–2 assignments are unchanged.', fontsize=10, color='#53616c')
    for pi, passage in enumerate(PASSAGES):
        for state in range(3):
            ax = fig.add_axes([.074 + state*.314, .595 if pi == 0 else .165, .251, .250])
            for ci, condition in enumerate(CONDITIONS):
                sub = info.loc[info.passage.eq(passage) & info.condition.eq(condition)].sort_values('physical_well')
                y = sub[f'state_{state}_fraction'].to_numpy()
                intervention = sub.intervention.iloc[0]
                marker = 'o' if condition == 'Control' else MARKERS[intervention]
                ax.scatter(ci + np.linspace(-.17, .17, len(sub)), y, c=COLORS[passage], marker=marker,
                           s=25, edgecolors='white', linewidths=.4, zorder=3)
                ax.plot([ci-.24, ci+.24], [y.mean(), y.mean()], color='#27333b', lw=1.4, zorder=2)
            ax.set_xlim(-.5, 5.5)
            maximum = min(1., float(info[f'state_{state}_fraction'].max()) * 1.12) if state == 0 else 1.
            ax.set_ylim(-.035*maximum, maximum*1.02)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            ax.set_xticks(range(6), CONDITIONS, rotation=35, ha='right', fontsize=8)
            ax.tick_params(axis='x', length=0)
            ax.set_title(f'{passage} · State {state}', color=COLORS[passage], fontsize=11, fontweight='bold')
            ax.set_ylabel('Cells in state (%)', fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#e5e9ec', lw=.5, zorder=0)
    fig.text(.026, .035, 'All 60 wells are shown. Separate treatment-only and H1N1-background contrasts are provided in the statistics workbook.', fontsize=9, color='#53616c')
    save_figure(fig, folder, 'S1_state_occupancy', dpi=400)


def render(out):
    import pymupdf
    data = out / 'panel_G_data'; supplement = out / 'panel_G_supplement'
    preserved = read(data / 'preservation_manifest.csv')
    verify_preserved(dict(zip(preserved.path, preserved.sha256)))
    info = read(data / 'well_summaries_and_frozen_PCA.csv')
    statistics = read(out / 'panel_G_statistics.csv')
    variance = np.array(json.loads((data / 'PCA_display_metadata.json').read_text())['variance_ratio'])
    counts = read(data / 'condition_counts.csv')
    write_reports(out, counts, statistics, variance)
    render_main(out, info, statistics, variance)
    render_family_detail(supplement, statistics, 'H1N1')
    render_family_detail(supplement, statistics, 'Control')
    render_state_detail(supplement, info)
    with pymupdf.open() as combined:
        for name in ['S1_H1N1_comparisons', 'S1_Control_context_comparisons', 'S1_state_occupancy']:
            with pymupdf.open(supplement / f'{name}.pdf') as source: combined.insert_pdf(source)
        combined.save(supplement / 'S1_exploratory_details.pdf', garbage=4, deflate=True)
    verify_preserved(dict(zip(preserved.path, preserved.sha256)))
    validate(out)
    print('FIGURES COMPLETE:', out / 'panel_G.pdf')


def validate(out):
    import pymupdf
    from PIL import Image
    data = out / 'panel_G_data'
    info = read(data / 'well_summaries_and_frozen_PCA.csv')
    profiles = read(data / 'well_mean_feature_profiles.csv').set_index('well_id')
    stats = read(out / 'panel_G_statistics.csv')
    vectors = read(data / 'all_feature_contrast_vectors.csv')
    family = read(SOURCES['predefined_feature_families']).set_index('original_feature')
    features = list(profiles.columns)
    assert len(features) == 98 and len(info) == 60 and info.n_cells.sum() == 4866
    assert info.groupby(['passage', 'plasma', 'intervention']).size().eq(5).all()
    counts = info[[f'state_{s}_cells' for s in range(3)]].sum(axis=1)
    np.testing.assert_array_equal(counts, info.n_cells)
    original_states = read(SOURCES['existing_state_assignments']).set_index('cell_id').All__k3
    local_meta = read(data / 'cell_metadata_and_existing_states.csv.gz')
    np.testing.assert_array_equal(original_states.loc[local_meta.cell_id].to_numpy(int), local_meta.state.to_numpy(int))
    for comparison, group in stats.groupby('comparison', sort=False):
        row = group.iloc[0]
        ref = info.passage.eq(row.passage) & info.plasma.eq(row.plasma) & info.intervention.eq('None')
        target = info.passage.eq(row.passage) & info.plasma.eq(row.plasma) & info.intervention.eq(row.intervention)
        delta = profiles.loc[info.loc[target, 'well_id']].mean(axis=0) - profiles.loc[info.loc[ref, 'well_id']].mean(axis=0)
        saved_vector = vectors.loc[vectors.comparison.eq(comparison)].set_index('feature').loc[features, 'target_minus_reference']
        np.testing.assert_allclose(delta, saved_vector, atol=1e-12)
        for item in group.itertuples():
            if item.metric_index < 5:
                subset = features if item.metric == 'Global' else [f for f in features if family.loc[f, 'family'] == item.metric]
                value = float(np.sqrt(np.mean(delta.loc[subset].to_numpy()**2)))
            else:
                col = f'state_{item.metric_index-5}_fraction'
                value = 100 * (info.loc[target, col].mean() - info.loc[ref, col].mean())
            np.testing.assert_allclose(value, item.estimate, atol=1e-12)
    nulls = read(data / 'permutation_reference.csv.gz')
    for row in stats.itertuples():
        values = nulls.loc[nulls.comparison.eq(row.comparison), f'metric_{row.metric_index}'].to_numpy(float)
        assert len(values) == 252
        p = np.mean(np.abs(values) >= abs(row.estimate) - 1e-12*max(1, abs(row.estimate)))
        np.testing.assert_allclose(p, row.p_permutation, atol=1e-15)
    for _, group in stats.groupby('FDR_family'):
        assert len(group) == 32
        np.testing.assert_allclose(bh(group.p_permutation.to_numpy()), group.q_BH, atol=1e-15)
    omitted = read(data / 'leave_one_well_out.csv')
    assert len(omitted) == 8*8*10
    preserved = read(data / 'preservation_manifest.csv')
    verify_preserved(dict(zip(preserved.path, preserved.sha256)))
    outputs = []
    paths = [out / 'panel_G.pdf'] + [out / 'panel_G_supplement' / f'{n}.pdf' for n in
            ['S1_H1N1_comparisons', 'S1_Control_context_comparisons', 'S1_state_occupancy']]
    for path in paths:
        with pymupdf.open(path) as doc:
            assert len(doc) == 1
            page = doc[0]
            assert 'Exploratory' in page.get_text()
            assert len(page.get_images(full=True)) == 0
            for block in page.get_text('blocks'):
                rect = pymupdf.Rect(block[:4])
                assert rect.x0 >= 0 and rect.y0 >= 0 and rect.x1 <= page.rect.width and rect.y1 <= page.rect.height, (str(path), block)
            outputs.append({'file': str(path.relative_to(out)), 'pages': 1, 'vector_without_raster_images': True, 'all_text_within_page': True})
    with pymupdf.open(out / 'panel_G_supplement/S1_exploratory_details.pdf') as doc: assert len(doc) == 3
    with Image.open(out / 'panel_G.png') as png:
        pixels, dpi = list(png.size), list(png.info['dpi'])
    path = out / 'panel_G.validation.json'
    validation = json.loads(path.read_text())
    validation.update({'status': 'complete', 'all_state_labels_verified_against_original_assignments': True,
                       'all_64_estimates_recomputed_from_well_tables': True,
                       'all_64_permutation_P_values_and_BH_q_values_verified': True,
                       'source_and_previous_outputs_unchanged_after_render': True,
                       'main_figure_well_points': 30, 'supplement_state_well_points': 180,
                       'exports': outputs, 'main_png_pixels': pixels, 'main_png_dpi': dpi,
                       'script_sha256': sha(Path(__file__))})
    dump(validation, path)
    print('VALIDATION PASSED: frozen primary results, 64 well-level statistics, original states, vector exports and preserved inputs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=panel_dir('G'))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--render-only', action='store_true')
    modes.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    assert out.is_relative_to(RESULT.resolve()), 'Outputs must remain within result/.'
    out.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        validate(out); return
    if not args.render_only: prepare(out)
    if not args.prepare_only: render(out)


if __name__ == '__main__':
    main()
