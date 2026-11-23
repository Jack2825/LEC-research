"""Panel F: visualize existing primary-group well-level state proportions.

No response scores, response thresholds, clustering solutions or significance
tests are created. Existing cell assignments and saved well fractions are checked
against each other before visualization. All inputs stay in this figure project.

Run --prepare-only for inspection and QC tables, then --render-only for figures.
Use a fresh --output-dir within result/ to preserve a completed analysis.
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
import numpy as np
import pandas as pd
import matplotlib

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'result'
os.environ['MPLCONFIGDIR'] = str(RESULT / '.matplotlib_cache')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PREFIX = 'panel_F_visualization'
GROUPS = ['P3-Control', 'P3-H1N1', 'P9-Control', 'P9-H1N1']
COLORS = {'P3': '#0072B2', 'P9': '#D55E00'}
MARKERS = {'Control': 'o', 'H1N1': '^'}
SOURCES = {
    'existing_cell_state_assignments': RESULT / 'Panel_E/panel_E_state_visualization_data/primary_cell_state_assignments.csv.gz',
    'existing_well_state_proportions': RESULT / 'Panel_E/panel_E_state_visualization_data/well_state_counts_and_fractions.csv',
    'existing_state_QC': RESULT / 'Panel_E/panel_E_state_QC.csv',
}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .8,
                     'savefig.facecolor': 'white', 'xtick.labelsize': 9, 'ytick.labelsize': 9})


def sha(path):
    with resolve_result_path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path, **kwargs):
    path = resolve_result_path(path)
    assert path.is_relative_to(ROOT.resolve()), 'Input outside the allowed figure project.'
    return pd.read_csv(path, keep_default_na=False, **kwargs)


def dump(value, path):
    def cast(item):
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, Path): return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=cast), encoding='utf-8')


def manifest_existing(out):
    return {str(p.resolve()): sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and '.matplotlib_cache' not in p.parts and '__pycache__' not in p.parts
            and p.resolve() != Path(__file__).resolve()
            and not (p.parent.resolve() == out and p.name.startswith('panel_F'))
            and not p.resolve().is_relative_to(out / 'panel_F_visualization_data')}


def check_preservation(manifest):
    for name, expected in manifest.items():
        assert_preserved(name, expected)


def inspect_metrics():
    """Record existing derived metrics; never convert other summaries to responders."""
    records = []
    for path in sorted(RESULT.rglob('*')):
        if not path.is_file() or '.matplotlib_cache' in path.parts or path.name.startswith('panel_F'):
            continue
        if 'panel_F_visualization_data' in path.parts:
            continue
        if path.suffix == '.csv' or path.name.endswith('.csv.gz'):
            columns = read(path, nrows=0).columns.tolist()
            matches = [c for c in columns if any(word in c.lower() for word in ['responder', 'response_score', 'distance', 'mahalanobis', 'rms'])]
            if matches:
                records.append({'path': str(path.relative_to(ROOT)), 'format': 'CSV',
                                'matching_columns': '|'.join(matches),
                                'row_identity_columns': '|'.join(c for c in columns if c in ['cell_id', 'well_id', 'passage', 'omitted_well']),
                                'existing_well_responder_classification': any('responder' in c.lower() for c in matches)})
        elif path.suffix == '.xlsx':
            with pd.ExcelFile(path, engine='openpyxl') as book:
                for sheet in book.sheet_names:
                    columns = pd.read_excel(book, sheet_name=sheet, nrows=0).columns.astype(str).tolist()
                    matches = [c for c in columns if any(word in c.lower() for word in ['responder', 'response_score', 'distance', 'mahalanobis', 'rms'])]
                    if matches:
                        records.append({'path': str(path.relative_to(ROOT)) + ' :: ' + sheet, 'format': 'Excel',
                                        'matching_columns': '|'.join(matches),
                                        'row_identity_columns': '|'.join(c for c in columns if c in ['cell_id', 'well_id', 'passage', 'omitted_well']),
                                        'existing_well_responder_classification': any('responder' in c.lower() for c in matches)})
    result = pd.DataFrame(records, columns=['path', 'format', 'matching_columns', 'row_identity_columns', 'existing_well_responder_classification'])
    # The inspected project has no saved responder classification. A future new
    # source must be inspected rather than silently assigned a definition here.
    if len(result) and result.existing_well_responder_classification.any():
        raise ValueError('A responder-related source is now present. Inspect its existing definition before extending this state-only visualization.')
    return result


def prepare_tables():
    cells = read(SOURCES['existing_cell_state_assignments'])
    saved = read(SOURCES['existing_well_state_proportions'])
    assert cells.cell_id.is_unique and cells.intervention.eq('None').all()
    assert set(cells.panel_group) == set(GROUPS)
    assert set(cells.state) == {0, 1, 2}
    assert set(saved.panel_group) == set(GROUPS)
    assert saved.set_index(['well_id', 'state']).index.is_unique
    columns = ['well_id', 'physical_well', 'passage', 'plasma', 'panel_group']
    wells = cells[columns].drop_duplicates().copy()
    assert wells.well_id.is_unique
    wells['group_order'] = wells.panel_group.map({g: i for i, g in enumerate(GROUPS)})
    wells = wells.sort_values(['group_order', 'physical_well', 'well_id']).reset_index(drop=True)
    indexed = saved.set_index(['well_id', 'state'])
    counts = pd.crosstab(cells.well_id, cells.state).reindex(index=wells.well_id, columns=range(3), fill_value=0)
    wells['n_cells'] = counts.sum(axis=1).to_numpy(int)
    for state in range(3):
        state_rows = indexed.loc[[(well, state) for well in wells.well_id]]
        np.testing.assert_array_equal(state_rows.n_cells_in_state.to_numpy(), counts[state].to_numpy())
        np.testing.assert_array_equal(state_rows.n_cells_in_well.to_numpy(), wells.n_cells.to_numpy())
        fractions = state_rows.fraction_of_well.to_numpy(float)
        np.testing.assert_allclose(fractions, counts[state].to_numpy() / wells.n_cells.to_numpy(), atol=1e-15)
        wells[f'state_{state}_cells'] = counts[state].to_numpy(int)
        wells[f'state_{state}_fraction'] = fractions
    np.testing.assert_allclose(wells[[f'state_{s}_fraction' for s in range(3)]].sum(axis=1), 1)
    assert len(cells) == 1626 and len(wells) == 20 and wells.groupby('panel_group').size().eq(5).all()
    groups, omission = [], []
    for group in GROUPS:
        subset = wells.loc[wells.panel_group.eq(group)].copy()
        for state in range(3):
            values = subset[f'state_{state}_fraction']
            state_counts = subset[f'state_{state}_cells']
            mean = float(values.mean())
            state_total = int(state_counts.sum())
            mass = values / values.sum() if values.sum() > 0 else values * 0
            omitted_means = []
            for index, row in subset.iterrows():
                remaining_mean = float(values.drop(index).mean())
                omitted_means.append(remaining_mean)
                omission.append({'panel_group': group, 'state': state, 'state_name': f'State {state}',
                                 'omitted_well': row.well_id, 'omitted_physical_well': row.physical_well,
                                 'mean_fraction_all_5_wells': mean,
                                 'mean_fraction_remaining_4_wells': remaining_mean,
                                 'change_in_mean_percentage_points': 100 * (remaining_mean - mean)})
            influence = np.abs(np.asarray(omitted_means) - mean)
            max_index = int(np.argmax(influence))
            row = {'panel_group': group, 'state': state, 'state_name': f'State {state}',
                   'n_wells': len(subset), 'total_cells': int(subset.n_cells.sum()),
                   'state_cells': state_total, 'wells_with_state_cells': int(state_counts.gt(0).sum()),
                   'mean_well_fraction': mean, 'minimum_well_fraction': float(values.min()),
                   'maximum_well_fraction': float(values.max()),
                   'largest_single_well_share_of_group_state_cells': float(state_counts.max() / state_total) if state_total else 0.,
                   'largest_cell_count_wells': '|'.join(subset.loc[state_counts.eq(state_counts.max()), 'well_id']),
                   'largest_single_well_share_of_group_state_proportion_sum': float(mass.max()),
                   'largest_proportion_contribution_well': subset.loc[mass.idxmax(), 'well_id'],
                   'omit_one_well_mean_min': min(omitted_means), 'omit_one_well_mean_max': max(omitted_means),
                   'largest_absolute_change_in_mean_pp': float(100 * influence[max_index]),
                   'most_influential_omitted_well': subset.iloc[max_index].well_id}
            groups.append(row)
    return cells, wells, pd.DataFrame(groups), pd.DataFrame(omission)


def md_table(frame):
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in frame.to_dict('records'):
        lines.append('| ' + ' | '.join(str(row[c]) for c in columns) + ' |')
    return '\n'.join(lines)


def reports(out, wells, groups, metrics):
    display = wells[['panel_group', 'physical_well', 'n_cells']].rename(columns={'panel_group': 'Group', 'physical_well': 'Well', 'n_cells': 'Cells'})
    for state in range(3):
        display[f'State {state} (%)'] = wells[f'state_{state}_fraction'].map(lambda x: f'{100*x:.2f}')
    p9_rare = groups.loc[groups.panel_group.eq('P9-H1N1') & groups.state.eq(0)].iloc[0]
    text = f'''# Panel F: State abundance and response-summary visualization

## Existing inputs and response-metric inspection

Cell-level assignments: `result/Panel_E/panel_E_state_visualization_data/primary_cell_state_assignments.csv.gz`, columns `cell_id`, `well_id`, `panel_group`, `state` and `state_name`. The labels are the existing State 0/1/2 assignments, inherited from `All__k3`; clustering is not rerun or relabeled.

Existing well fractions: `result/Panel_E/panel_E_state_visualization_data/well_state_counts_and_fractions.csv`. These saved fractions are used directly and independently checked against the existing cell assignments and counts. All inputs are inside the figure project. Files and hashes are listed in the data subfolder.

No existing cell-level responder label, responder score or well-level responder fraction is present inside the permitted `Cell_Painting_LEC_P3_P9_figures` folder. Therefore Panel F contains **state proportions only**. No responder criterion, score or threshold is created. This inspection does not assert that such outputs are absent from other study folders outside the permitted input scope.

Other derived response summaries do exist: `panel_C.displacement.csv` contains passage-level RMS displacement between the equal-well H1N1 and Control feature profiles. Its omission and permutation tables contain alternative group-level displacements. `panel_D_statistics.compartments.csv` and the related omission table contain group/family-level response and interaction RMS summaries. These are group-level comparisons, not a classification of individual cells or a response fraction assigned to each well. Their existing values and definitions are unchanged and are not converted into a responder classification. The metric inventory records matching columns in CSV and Excel outputs. Existing PCA coordinate arrays also contain no responder labels.

The earlier configuration file includes a nominal tail-quantile setting, but a configuration setting alone is not an existing classification. It is not applied to create responders here.

## Well and cell counts; existing proportions

The four groups each contain five wells: 20 wells and 1,626 cells total. P3-Control has 124 cells, P3-H1N1 162, P9-Control 693 and P9-H1N1 647. Each well contributes one plotted point regardless of its cell count. No cells or wells are removed.

{md_table(display)}

## Descriptive computational QC

The plotted fraction is n_cells_in_state / n_analyzed_cells_in_the_same_well. The short group bar is the arithmetic mean of the five well fractions; cells from different wells are not pooled to define this bar. All three state fractions sum to one in each well. Different state panels have explicitly labeled percent axes; State 0 uses a smaller vertical range to make its low fractions visible.

The workbook reports the concentration of each state within each group, including the largest single-well share of the state's cell count and of the sum of the five well fractions. It also shows the minimum and maximum group mean after omitting each of the five wells in turn. These are descriptive arithmetic QC summaries, **not confidence intervals or significance tests**. They do not define new response criteria or alter any state assignment.

One clear concentration is P9-H1N1 State 0: four of its five state cells are in well B04 (80% of that group's State 0 cells). That well contributes {100*p9_rare.largest_single_well_share_of_group_state_proportion_sum:.2f}% of the group's summed State 0 well fractions. The group's equal-well State 0 mean is {100*p9_rare.mean_well_fraction:.3f}%; omitting B04 changes it to {100*p9_rare.omit_one_well_mean_min:.3f}%. This is a computational concentration summary, not a biological interpretation. Whole-cohort state support reported for Panel E does not by itself establish balanced support within every group.

Wells are shown as the descriptive replicate unit. Separate donor or plasma-pool identities are not established by these metadata. Only the existing exported cells are represented. The existing source state QC remains unchanged; the plot does not establish that the computational states are free of technical effects.

## Figure layout and statistical scope

Three aligned plots show State 0, State 1 and State 2 across P3-Control, P3-H1N1, P9-Control and P9-H1N1. Every well is shown. Passage colors match Panels C-D: P3 blue (#0072B2), P9 orange (#D55E00). Control is a circle and H1N1 a triangle. Horizontal jitter is fixed by the sorted physical-well order and only separates overlapping points; it has no numerical meaning. Short horizontal bars indicate equal-well group means. State labels remain unchanged.

No new statistical tests are performed and no P values, significance stars, or statistical-significance claims are added. The data shown are state proportions only; there is no responder subplot. No functional, biological or mechanistic meaning is assigned to a state or pattern.

## Outputs and reproduction

Run `script/panel_F_visualization.py --prepare-only` to create the source inspection, 20-row well table, group QC and omission summaries. Then run `--render-only` to create the PDF/SVG and 600-dpi PNG. The default run performs both steps. Use a new `--output-dir` inside result for a new analysis; completed analyses are not overwritten by default. --render-only updates this panel's figures and documentation from the saved tables.

`panel_F_well_summary.xlsx` contains well fractions/counts, group-level concentration QC, omit-one-well group means, the existing-metric inventory and source paths. The corresponding CSV tables preserve full numerical precision. Source and previous-panel SHA-256 hashes are checked before and after figure generation. The vector PDF contains no embedded microscopy images.
'''
    (out / f'{PREFIX}.methods.md').write_text(text, encoding='utf-8')
    inspection = f'''# Panel F prefigure inspection

- State assignment file: `result/Panel_E/panel_E_state_visualization_data/primary_cell_state_assignments.csv.gz`.
- Existing well proportions: `result/Panel_E/panel_E_state_visualization_data/well_state_counts_and_fractions.csv`.
- Existing responder score/label/fraction available at cell or well level: **No**.
- Other available summaries: Panel C group-level RMS displacement and Panel D group/family-level response/interaction RMS. They do not supply an existing per-cell responder classification.
- Wells per group: five; 20 total. Cell counts and all state proportions follow below.
- Layout: three aligned state-proportion plots, four groups each, five individual-well points and an equal-well mean bar per group; P3 blue/P9 orange and Control circles/H1N1 triangles.
- No new response definition, threshold, clustering or significance test.

{md_table(display)}
'''
    (out / f'{PREFIX}.inspection.md').write_text(inspection, encoding='utf-8')
    legend = '''**F. State abundance and response-summary visualization.** Existing State 0–2 assignments are summarized for the four primary groups. Each point is one well; short horizontal bars show the arithmetic mean of five well fractions per group. Fractions use each well's own analyzed-cell count as the denominator. P3 is blue and P9 orange; circles denote Control and triangles H1N1. State 0 uses a smaller labeled y-axis range. Only state proportions are shown because no existing cell-level responder classification or well-level responder fraction is available in the permitted project outputs. No new response metric, threshold, clustering or significance test was created. Group and single-well concentration summaries are supplied in the QC workbook. The labels are computational states and carry no assigned biological or functional interpretation.
'''
    (out / f'{PREFIX}.legend.md').write_text(legend, encoding='utf-8')


def prepare(out):
    data = out / f'{PREFIX}_data'
    if (out / f'{PREFIX}.validation.json').exists() or (data.exists() and any(data.iterdir())):
        raise FileExistsError('Completed Panel F data exists; use a new --output-dir.')
    preserved = manifest_existing(out)
    metrics = inspect_metrics()
    cells, wells, groups, omission = prepare_tables()
    data.mkdir(parents=True, exist_ok=True)
    wells.to_csv(out / 'panel_F_well_summary.csv', index=False)
    groups.to_csv(data / 'group_state_QC.csv', index=False)
    omission.to_csv(data / 'omit_one_well_group_means.csv', index=False)
    metrics.to_csv(data / 'existing_response_metric_inventory.csv', index=False)
    source_files = pd.DataFrame([{'role': role, 'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for role, path in SOURCES.items()])
    source_files.to_csv(data / 'source_manifest.csv', index=False)
    pd.DataFrame([{'path': p, 'sha256': h} for p, h in preserved.items()]).to_csv(data / 'preservation_manifest.csv', index=False)
    with pd.ExcelWriter(out / 'panel_F_well_summary.xlsx', engine='openpyxl') as writer:
        for name, frame in [('Well summary', wells), ('Group state QC', groups), ('Omit one well', omission),
                            ('Existing response metrics', metrics), ('Source files', source_files)]:
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(65, max(12, max(len(str(c.value or '')) for c in column)+2))
    reports(out, wells, groups, metrics)
    check_preservation(preserved)
    dump({'status': 'prepared; figure not rendered', 'cells': len(cells), 'wells': len(wells),
          'wells_per_group': wells.groupby('panel_group').size().to_dict(),
          'existing_assignments_unchanged': True, 'existing_well_fractions_verified': True,
          'new_response_score_or_threshold_created': False, 'new_statistical_tests': False,
          'eligible_existing_responder_classification_available': False,
          'panel_content': 'existing state proportions only', 'all_inputs_inside_figure_project': True,
          'previous_files_hash_checked': len(preserved), 'previous_files_unchanged': True,
          'software': {'python': platform.python_version(), 'numpy': np.__version__,
                       'pandas': pd.__version__, 'matplotlib': matplotlib.__version__}}, out / f'{PREFIX}.validation.json')
    print('PREPARED: existing state proportions only; no new responder definition or significance test.')
    print(wells.to_string(index=False))
    print(groups[['panel_group', 'state_name', 'mean_well_fraction', 'largest_single_well_share_of_group_state_cells',
                  'omit_one_well_mean_min', 'omit_one_well_mean_max']].to_string(index=False))


def render(out):
    data = out / f'{PREFIX}_data'
    preserved = read(data / 'preservation_manifest.csv')
    preserved = dict(zip(preserved.path, preserved.sha256))
    check_preservation(preserved)
    wells = read(out / 'panel_F_well_summary.csv')
    groups = read(data / 'group_state_QC.csv')
    metrics = read(data / 'existing_response_metric_inventory.csv')
    reports(out, wells, groups, metrics)
    fig = plt.figure(figsize=(12.8, 4.8))
    fig.text(.017, .963, 'F', fontsize=21, fontweight='bold', va='top')
    fig.text(.061, .954, 'State abundance and response-summary visualization', fontsize=15, fontweight='bold', va='top')
    fig.text(.061, .891, 'Existing state proportions · 5 wells per group · 1,626 cells in total', fontsize=10, color='#53616c', va='top')
    positions = [0., 1., 2.8, 3.8]
    axes = []
    for state in range(3):
        ax = fig.add_axes([.070 + state * .320, .266, .250, .510])
        axes.append(ax)
        for gi, group in enumerate(GROUPS):
            sub = wells.loc[wells.panel_group.eq(group)].sort_values('physical_well')
            passage, plasma = group.split('-', 1)
            y = sub[f'state_{state}_fraction'].to_numpy(float)
            jitter = np.linspace(-.18, .18, len(sub))
            ax.scatter(positions[gi] + jitter, y, s=43, c=COLORS[passage], marker=MARKERS[plasma],
                       edgecolors='white', linewidths=.55, zorder=3)
            ax.plot([positions[gi]-.27, positions[gi]+.27], [y.mean(), y.mean()],
                    color='#27333b', linewidth=1.75, zorder=2)
        ax.set_xlim(-.52, 4.32)
        if state == 0:
            ax.set_ylim(-.006, .123)
            ax.set_yticks([0, .03, .06, .09, .12])
        else:
            ax.set_ylim(-.04, 1.03)
            ax.set_yticks([0, .25, .50, .75, 1.0])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xticks(positions, ['Control', 'H1N1', 'Control', 'H1N1'])
        ax.tick_params(axis='x', length=0, pad=8)
        ax.tick_params(axis='y', labelsize=9)
        ax.set_title(f'State {state}', fontsize=12, pad=13, fontweight='bold')
        ax.set_ylabel('Cells in state (%)', fontsize=10, labelpad=6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e6e9ec', linewidth=.6, zorder=0)
        for passage, center in [('P3', .5), ('P9', 3.3)]:
            ax.text(center, -.196, passage, transform=ax.get_xaxis_transform(), ha='center',
                    va='center', color=COLORS[passage], fontsize=11, fontweight='bold', clip_on=False)
            ax.plot([center-.7, center+.7], [-.144, -.144], transform=ax.get_xaxis_transform(),
                    color=COLORS[passage], linewidth=1, clip_on=False)
    fig.text(.062, .070, 'Each point = one well. Short bars = equal-well group means. State 0 has a smaller y-axis range.',
             fontsize=9.3, color='#45535e')
    fig.text(.062, .028, 'State proportions only. Existing cell classifications are unchanged.',
             fontsize=8.8, color='#61707a')
    for suffix in ['pdf', 'svg']:
        fig.savefig(out / f'{PREFIX}.{suffix}', facecolor='white')
    fig.savefig(out / f'{PREFIX}.png', dpi=600, facecolor='white')
    fig.savefig(out / f'{PREFIX}.preview.png', dpi=140, facecolor='white')
    plt.close(fig)
    check_preservation(preserved)
    validate(out)
    print('FIGURE COMPLETE:', out / f'{PREFIX}.pdf')


def validate(out):
    import pymupdf
    from PIL import Image
    cells, original, group_original, omitted_original = prepare_tables()
    saved = read(out / 'panel_F_well_summary.csv')
    assert saved.well_id.equals(original.well_id)
    for state in range(3):
        np.testing.assert_array_equal(saved[f'state_{state}_cells'], original[f'state_{state}_cells'])
        np.testing.assert_allclose(saved[f'state_{state}_fraction'], original[f'state_{state}_fraction'], atol=1e-15, rtol=0)
    assert saved.groupby('panel_group').size().eq(5).all()
    assert saved.n_cells.sum() == 1626
    assert not any('responder' in c or 'response_score' in c for c in saved.columns)
    data = out / f'{PREFIX}_data'
    group_saved = read(data / 'group_state_QC.csv')
    np.testing.assert_allclose(group_saved.mean_well_fraction, group_original.mean_well_fraction, atol=1e-15)
    omitted = read(data / 'omit_one_well_group_means.csv')
    np.testing.assert_allclose(omitted.mean_fraction_remaining_4_wells, omitted_original.mean_fraction_remaining_4_wells, atol=1e-15)
    preserved = read(data / 'preservation_manifest.csv')
    check_preservation(dict(zip(preserved.path, preserved.sha256)))
    path = out / f'{PREFIX}.validation.json'
    validation = json.loads(path.read_text(encoding='utf-8'))
    with pymupdf.open(out / f'{PREFIX}.pdf') as document:
        assert len(document) == 1
        page = document[0]
        text = page.get_text()
        assert all(f'State {s}' in text for s in range(3))
        assert 'VEGF' not in text and 'UC2288' not in text
        assert len(page.get_images(full=True)) == 0
        for block in page.get_text('blocks'):
            rectangle = pymupdf.Rect(block[:4])
            assert rectangle.x0 >= 0 and rectangle.y0 >= 0 and rectangle.x1 <= page.rect.width and rectangle.y1 <= page.rect.height
    with Image.open(out / f'{PREFIX}.png') as png:
        dimensions = list(png.size)
        dpi = list(png.info['dpi'])
    validation.update({'status': 'complete', 'saved_fractions_checked_against_existing_cell_assignments': True,
                       'all_20_wells_displayed_in_each_state_plot': True, 'number_of_plotted_well_points': 60,
                       'source_and_previous_outputs_unchanged_after_render': True,
                       'vector_PDF_no_embedded_raster_images': True, 'all_PDF_text_within_page': True,
                       'png_pixels': dimensions, 'png_dpi': dpi, 'script_sha256': sha(Path(__file__))})
    dump(validation, path)
    print('VALIDATION PASSED: original labels/fractions, 20 wells, descriptive summaries, vector PDF and source preservation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=panel_dir('F'))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--render-only', action='store_true')
    modes.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    assert out.is_relative_to(RESULT.resolve()), 'Generated outputs must stay inside result/.'
    out.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        validate(out)
        return
    if not args.render_only:
        prepare(out)
    if not args.prepare_only:
        render(out)


if __name__ == '__main__':
    main()
