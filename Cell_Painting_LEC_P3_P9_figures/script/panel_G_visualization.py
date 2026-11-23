"""Descriptive Panel G using already-generated local computational outputs.

Read existing well PCA coordinates and existing global RMS summaries. No new
score, threshold, feature selection, clustering or statistical test is fitted.
Run --prepare-only before --render-only, or run once without either flag.
Source files and earlier figures are preserved by SHA-256 verification.
"""
from pathlib import Path
from figure_paths import panel_dir, resolve_result_path, assert_preserved
import argparse
import hashlib
import json
import os
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
from matplotlib.lines import Line2D

PREFIX = 'panel_G_visualization'
PASSAGES = ['P3', 'P9']
CONDITIONS = ['H1N1', 'H1N1+VEGF-C', 'H1N1+UC2288']
INTERVENTIONS = ['VEGF-C', 'UC2288']
COLORS = {'P3': '#0072B2', 'P9': '#D55E00'}
MARKERS = {'H1N1': '^', 'H1N1+VEGF-C': 's', 'H1N1+UC2288': 'D'}
SOURCES = {
    'existing_well_coordinates_and_group_metadata': RESULT / 'Panel_G/panel_G_data/well_summaries_and_frozen_PCA.csv',
    'existing_aggregate_comparison_values': RESULT / 'Panel_G/panel_G_statistics.csv',
    'existing_well_feature_profiles_for_identity_check': RESULT / 'Panel_G/panel_G_data/well_mean_feature_profiles.csv',
    'existing_PCA_display_metadata': RESULT / 'Panel_G/panel_G_data/PCA_display_metadata.json',
}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.linewidth': .8, 'savefig.facecolor': 'white',
                     'xtick.labelsize': 9, 'ytick.labelsize': 9})


def sha(path):
    with resolve_result_path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path, **kwargs):
    path = resolve_result_path(path)
    assert path.is_relative_to(ROOT.resolve()), 'Input is outside the permitted figure folder.'
    return pd.read_csv(path, keep_default_na=False, **kwargs)


def dump(value, path):
    def cast(item):
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, Path): return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=cast), encoding='utf-8')


def preservation(out):
    return {str(p.resolve()): sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and '.matplotlib_cache' not in p.parts and '__pycache__' not in p.parts
            and p.resolve() != Path(__file__).resolve()
            and not (p.parent.resolve() == out and (p.name.startswith(PREFIX) or p.name.startswith('panel_G_summary')))
            and not p.resolve().is_relative_to(out / f'{PREFIX}_data')}


def verify_preserved(manifest):
    for name, expected in manifest.items():
        assert_preserved(name, expected)


def load_existing():
    full_wells = read(SOURCES['existing_well_coordinates_and_group_metadata'])
    wells = full_wells.loc[full_wells.plasma.eq('H1N1') & full_wells.condition.isin(CONDITIONS)].copy()
    columns = ['well_id', 'physical_well', 'passage', 'plasma', 'intervention', 'condition',
               'display_group', 'n_cells', 'PC1', 'PC2', 'PC1_relative_to_H1N1_mean', 'PC2_relative_to_H1N1_mean']
    wells = wells[columns].sort_values(['passage', 'condition', 'physical_well']).reset_index(drop=True)
    assert len(wells) == 30 and wells.well_id.is_unique
    assert wells.groupby(['passage', 'condition']).size().eq(5).all()
    assert wells.n_cells.sum() == 2404
    statistics = read(SOURCES['existing_aggregate_comparison_values'])
    order = [f'{p}__H1N1__{i}' for p in PASSAGES for i in INTERVENTIONS]
    summary = statistics.loc[statistics.metric.eq('Global') & statistics.comparison.isin(order)].copy()
    summary = summary.set_index('comparison').loc[order].reset_index()
    columns = ['comparison', 'passage', 'reference_condition', 'target_condition', 'intervention',
               'n_reference_wells', 'n_target_wells', 'n_reference_cells', 'n_target_cells',
               'metric', 'n_features', 'estimate', 'unit', 'estimate_definition', 'LOO_min', 'LOO_max',
               'most_influential_well', 'LOO_min_vector_cosine', 'LOO_range_is_confidence_interval']
    summary = summary[columns]
    summary['LOO_min_vector_cosine'] = pd.to_numeric(summary.LOO_min_vector_cosine, errors='raise')
    assert len(summary) == 4 and summary.n_features.eq(98).all()
    profiles = read(SOURCES['existing_well_feature_profiles_for_identity_check']).set_index('well_id')
    assert profiles.shape[1] == 98
    for row in summary.itertuples():
        ref = wells.loc[wells.passage.eq(row.passage) & wells.condition.eq(row.reference_condition)]
        target = wells.loc[wells.passage.eq(row.passage) & wells.condition.eq(row.target_condition)]
        assert len(ref) == row.n_reference_wells == 5
        assert len(target) == row.n_target_wells == 5
        assert ref.n_cells.sum() == row.n_reference_cells and target.n_cells.sum() == row.n_target_cells
        # Reproduce the already-defined RMS value only as an identity check.
        difference = profiles.loc[target.well_id].mean(axis=0) - profiles.loc[ref.well_id].mean(axis=0)
        reproduced = np.sqrt(np.mean(difference.to_numpy(float)**2))
        np.testing.assert_allclose(reproduced, row.estimate, atol=1e-12, rtol=1e-12)
    for passage in PASSAGES:
        sub = wells.loc[wells.passage.eq(passage)]
        ref = sub.loc[sub.condition.eq('H1N1')]
        for pc in ['PC1', 'PC2']:
            np.testing.assert_allclose(sub[pc] - ref[pc].mean(), sub[f'{pc}_relative_to_H1N1_mean'], atol=1e-12)
    counts = wells.groupby(['passage', 'condition'], sort=False).agg(wells=('well_id', 'size'), cells=('n_cells', 'sum')).reset_index()
    variance = np.array(json.loads(SOURCES['existing_PCA_display_metadata'].read_text(encoding='utf-8'))['variance_ratio'])
    inventory = pd.DataFrame([
        {'metric': 'Global RMS displacement', 'available': True, 'displayed': True,
         'source': 'result/Panel_G/panel_G_statistics.csv (Global rows)',
         'existing_definition': 'Panel C: root mean square across all 98 coordinates of target-minus-reference equal-well group mean feature vectors'},
        {'metric': 'Well-level PCA coordinates', 'available': True, 'displayed': True,
         'source': 'result/Panel_G/panel_G_data/well_summaries_and_frozen_PCA.csv',
         'existing_definition': 'Panel C frozen PCA applied to existing well profiles; saved coordinates centered on same-passage H1N1 mean are reused'},
        {'metric': 'Predefined feature-family RMS summaries', 'available': True, 'displayed': False,
         'source': 'result/Panel_G/panel_G_statistics.csv (Cell, Nucleus, ER and Mitochondria family rows)',
         'existing_definition': 'Panel D name-defined families; same existing RMS definition applied within each family'},
        {'metric': 'Existing State 0/1/2 proportions', 'available': True, 'displayed': False,
         'source': 'result/Panel_G/panel_G_data/well_summaries_and_frozen_PCA.csv; original labels in result/Panel_E/panel_E_source/cell_cluster_assignments.csv.gz',
         'existing_definition': 'Panels E-F: existing state-cell count divided by analyzed-cell count in each well'},
        {'metric': 'Existing responder fraction', 'available': False, 'displayed': False,
         'source': 'Not available in the permitted figure folder',
         'existing_definition': 'No new definition or threshold is introduced'},
    ])
    return wells, summary, counts, variance, inventory


def md_table(frame):
    columns = list(frame.columns)
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    lines += ['| '+' | '.join(str(row[c]) for c in columns)+' |' for row in frame.to_dict('records')]
    return '\n'.join(lines)


def write_reports(out, summary, counts, variance):
    compact = summary[['passage', 'target_condition', 'estimate', 'LOO_min', 'LOO_max']].copy()
    for column in ['estimate', 'LOO_min', 'LOO_max']:
        compact[column] = compact[column].map(lambda x: f'{x:.6g}')
    report = f'''# Panel G: Exploratory Cell Painting comparison summary

## Existing files and groups

`result/Panel_G/panel_G_data/well_summaries_and_frozen_PCA.csv` contains the well identities, group metadata, cell counts, saved well PCA coordinates and existing state proportions. The corresponding original computational cell/state tables remain in `result/Panel_E/panel_E_source/`. `result/Panel_G/panel_G_statistics.csv` contains the already-generated aggregate comparison values. Only the four specified H1N1-background comparisons enter this figure. This is a Cell Painting figure, separate from the tube-formation experiment.

{md_table(counts)}

The figure shows 30 unique wells and 2,404 analyzed cells across six existing groups. Each condition has five wells. H1N1 is the shared reference for both comparisons within its passage. Group names are metadata labels; no source assignment is changed.

## Available existing summaries and compact selection

The local outputs contain global RMS displacement, predefined feature-family RMS summaries, saved PCA well coordinates, and State 0/1/2 proportions. No existing responder classification is available in the permitted figure folder. The metric inventory records source files and definitions.

For a compact display, this figure uses **one aggregate comparison metric: global RMS displacement**, with existing well PCA coordinates as the replicate view. Family summaries and state proportions remain in their existing source outputs and are not added to this figure. Individual features are not ranked or selected. No new score, threshold, preprocessing fit, clustering result or statistical test is created.

## Existing definitions and reproducibility

The RMS definition comes from Panel C and is reproduced in the existing Panel G comparison table. If m_A and m_B are the equal-well mean normalized feature vectors for the reference and comparison groups, then D = sqrt(mean_j((m_Bj-m_Aj)^2)) over all 98 retained features. This summary is a nonnegative magnitude. The figure reads D from the saved table and checks it against the existing well feature profiles; it does not redefine the metric or change the values.

The PCA coordinates come from the frozen Panel C representation. They are the saved well means of cell scores. The previously saved translation subtracts each passage's H1N1 mean for display; it does not refit PCA. PC1/PC2 variance labels ({100*variance[0]:.2f}% and {100*variance[1]:.2f}%) refer to the primary fitting cohort. The same axes are used for P3/P9. No baseline passage comparison is represented by the translated origins.

The existing omit-one-well minima and maxima are copied from the same comparison table. Lines in the figure show those saved sensitivity ranges, **not confidence intervals**. Their definition and numeric values are unchanged. The current visualization does not add P values, significance annotations, or new statistical conclusions.

{md_table(compact)}

## Proposed and generated layout

Two small PCA views show all individual wells in P3 and P9, respectively. H1N1 is a triangle, H1N1+VEGF-C a square, and H1N1+UC2288 a diamond. P3 is blue and P9 orange, matching the earlier panels. Open symbols show equal-well group means; arrows connect the reference mean to the comparison mean. Shared H1N1 wells are plotted once per passage. The adjacent summary shows the four existing global RMS values and saved omit-one-well ranges. Numeric RMS labels are display-rounded to three decimals; CSV/Excel tables retain full stored precision.

The title explicitly says Exploratory. The figure is a descriptive comparison of existing computational summaries, with no biological interpretation.

## Files and checks

`panel_G_summary.csv` has four comparison rows. `panel_G_summary.wells.csv` contains the 30 displayed well records. `panel_G_summary.xlsx` includes the comparison summary, well coordinates, group counts, metric inventory and source manifest. Existing source figures and statistics, including previous Panel G outputs, are preserved.

Run `script/panel_G_visualization.py --prepare-only` for source inspection and summary tables, then `--render-only` for the PDF/SVG and 600-dpi PNG. Default execution performs both steps. A new output directory under result can be supplied for a separate run. Source and previous-output hashes are verified. No source images or data are overwritten.
'''
    (out / f'{PREFIX}.methods.md').write_text(report, encoding='utf-8')
    inspection = f'''# Panel G prefigure inspection

1. Relevant groups and well values: `result/Panel_G/panel_G_data/well_summaries_and_frozen_PCA.csv`; aggregate comparisons: `result/Panel_G/panel_G_statistics.csv`.
2. Five wells in each of the six displayed groups; counts below.
3. Existing metrics: global RMS, predefined family RMS, well PCA coordinates, State 0/1/2 proportions. No existing responder fraction is available locally.
4. Definitions: Panel C global RMS/PCA, Panel D feature-family mapping, Panels E-F existing state proportions.
5. Reproduction: all four RMS values match the existing well-profile calculation to numerical precision; saved PCA coordinates and group metadata are reused unchanged.
6. Layout: two well-level PCA views and one four-row existing global RMS summary, explicitly Exploratory. No additional significance annotations.

{md_table(counts)}

{md_table(compact)}
'''
    (out / f'{PREFIX}.inspection.md').write_text(inspection, encoding='utf-8')
    (out / 'panel_G_summary.md').write_text('# Existing Panel G comparison values\n\n'+md_table(compact)+'\n\nRanges are existing omit-one-well sensitivity ranges, not confidence intervals. Full-precision values and displayed well records are supplied in CSV/Excel.\n', encoding='utf-8')
    legend = '''**G. Exploratory Cell Painting comparison summary.** Existing H1N1, H1N1+VEGF-C and H1N1+UC2288 groups are displayed separately for P3/P9 (five wells per group). Left: saved well-level coordinates in the frozen primary-study PCA, centered on the same-passage H1N1 mean. Each small symbol is one well; open symbols indicate equal-well group means. Arrows connect reference and comparison means. Right: existing global RMS differences between group mean feature profiles across all 98 retained normalized measurements. Lines show previously calculated omit-one-well ranges and are not confidence intervals. This descriptive figure uses the existing definitions and values without creating a new score, threshold or significance test. It represents the Cell Painting dataset only and is separate from the tube-formation experiment.
'''
    (out / f'{PREFIX}.legend.md').write_text(legend, encoding='utf-8')


def prepare(out):
    data = out / f'{PREFIX}_data'
    if (out / f'{PREFIX}.validation.json').exists() or (data.exists() and any(data.iterdir())):
        raise FileExistsError('Completed visualization exists; use a new --output-dir.')
    preserved = preservation(out)
    wells, summary, counts, variance, inventory = load_existing()
    data.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / 'panel_G_summary.csv', index=False)
    wells.to_csv(out / 'panel_G_summary.wells.csv', index=False)
    counts.to_csv(data / 'group_counts.csv', index=False)
    inventory.to_csv(data / 'existing_metric_inventory.csv', index=False)
    sources = pd.DataFrame([{'role': role, 'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for role, path in SOURCES.items()])
    sources.to_csv(data / 'source_manifest.csv', index=False)
    pd.DataFrame([{'path': p, 'sha256': h} for p, h in preserved.items()]).to_csv(data / 'preservation_manifest.csv', index=False)
    with pd.ExcelWriter(out / 'panel_G_summary.xlsx', engine='openpyxl') as writer:
        for name, frame in [('Comparison summary', summary), ('Displayed wells', wells), ('Group counts', counts),
                            ('Existing metric inventory', inventory), ('Source files', sources)]:
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]; sheet.freeze_panes = 'A2'; sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(65, max(12, max(len(str(c.value or '')) for c in column)+2))
    write_reports(out, summary, counts, variance)
    verify_preserved(preserved)
    dump({'status': 'prepared; figure not rendered', 'displayed_wells': 30, 'displayed_cells': 2404,
          'existing_comparisons': 4, 'wells_per_group': 5, 'source_files_unchanged': True,
          'previous_files_hash_checked': len(preserved), 'new_score_or_threshold': False,
          'new_statistical_tests': False, 'feature_selection': False, 'PCA_refit': False,
          'existing_RMS_values_reproduced_from_saved_well_profiles': True,
          'existing_centered_PCA_coordinates_verified': True}, out / f'{PREFIX}.validation.json')
    print('PREPARED: 30 existing well records and 4 existing RMS summaries; source files preserved.')


def render(out):
    data = out / f'{PREFIX}_data'
    preserved = read(data / 'preservation_manifest.csv')
    verify_preserved(dict(zip(preserved.path, preserved.sha256)))
    wells = read(out / 'panel_G_summary.wells.csv')
    summary = read(out / 'panel_G_summary.csv')
    counts = read(data / 'group_counts.csv')
    variance = np.array(json.loads(SOURCES['existing_PCA_display_metadata'].read_text())['variance_ratio'])
    write_reports(out, summary, counts, variance)
    bound = max(.25, float(np.abs(wells[['PC1_relative_to_H1N1_mean', 'PC2_relative_to_H1N1_mean']].to_numpy()).max()) * 1.19)
    fig = plt.figure(figsize=(13.4, 5.3))
    fig.text(.016, .958, 'G', fontsize=21, fontweight='bold', va='top')
    fig.text(.061, .954, 'Exploratory Cell Painting comparison summary', fontsize=15, fontweight='bold', va='top')
    fig.text(.061, .893, 'Existing computational summaries · 5 wells per group', fontsize=10, color='#53616c', va='top')
    for pi, passage in enumerate(PASSAGES):
        ax = fig.add_axes([.066 + pi*.260, .294, .226, .488])
        ax.set_aspect('equal', adjustable='box')
        ax.axhline(0, color='#b8c0c6', ls='--', lw=.7, zorder=0)
        ax.axvline(0, color='#b8c0c6', ls='--', lw=.7, zorder=0)
        for condition in CONDITIONS:
            sub = wells.loc[wells.passage.eq(passage) & wells.condition.eq(condition)]
            x = sub.PC1_relative_to_H1N1_mean.to_numpy(float)
            y = sub.PC2_relative_to_H1N1_mean.to_numpy(float)
            assert len(x) == 5
            ax.scatter(x, y, c=COLORS[passage], marker=MARKERS[condition], s=37,
                       alpha=.87, edgecolors='white', linewidths=.5, zorder=3)
            if condition != 'H1N1':
                ax.annotate('', xy=(x.mean(), y.mean()), xytext=(0, 0),
                            arrowprops={'arrowstyle': '-|>', 'color': COLORS[passage], 'lw': 1.2, 'alpha': .65}, zorder=2)
            ax.scatter([x.mean()], [y.mean()], facecolors='none', edgecolors=COLORS[passage],
                       marker=MARKERS[condition], s=112, linewidths=1.5, zorder=4)
        ax.set_xlim(-bound, bound); ax.set_ylim(-bound, bound)
        ax.set_title(passage, color=COLORS[passage], fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel(f'Centered PC1 ({100*variance[0]:.1f}% in primary fit)', fontsize=8.8)
        if pi == 0: ax.set_ylabel(f'Centered PC2 ({100*variance[1]:.1f}% in primary fit)', fontsize=8.8)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([], [], marker=MARKERS[c], color='#45535e', linestyle='None', markersize=6, label=c) for c in CONDITIONS]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.060, .141), ncol=3, frameon=False,
               fontsize=9, handletextpad=.4, columnspacing=1.05)
    ax = fig.add_axes([.723, .319, .166, .437])
    ax.set_xlim(0, float(summary.LOO_max.max()) * 1.13); ax.set_ylim(3.65, -.7)
    ax.set_yticks(range(4), [f'{r.passage}: +{r.intervention}' for r in summary.itertuples()], fontsize=9.2)
    for y, row in summary.iterrows():
        ax.plot([row.LOO_min, row.LOO_max], [y, y], color=COLORS[row.passage], lw=2)
        for x in [row.LOO_min, row.LOO_max]: ax.plot([x, x], [y-.055, y+.055], color=COLORS[row.passage], lw=1)
        ax.scatter([row.estimate], [y], s=49, marker='D', c=COLORS[row.passage], edgecolors='white', linewidths=.5, zorder=3)
        ax.text(1.11, y, f'{row.estimate:.3f}', transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=9.2)
    ax.text(1.11, -.67, 'RMS', transform=ax.get_yaxis_transform(), ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Existing global difference', fontsize=11.5, fontweight='bold', pad=20)
    ax.set_xlabel('RMS difference from H1N1\n(all 98 features)', fontsize=9.2, labelpad=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=10)
    ax.axhline(1.5, color='#dce1e5', lw=.7)
    fig.text(.061, .083, 'Small symbols: individual wells; open symbols: equal-well group means. Center: same-passage H1N1 mean.', fontsize=8.7, color='#53616c')
    fig.text(.061, .039, 'Lines show existing omit-one-well ranges, not confidence intervals. Descriptive Cell Painting comparisons only.', fontsize=8.7, color='#53616c')
    for suffix in ['pdf', 'svg']: fig.savefig(out / f'{PREFIX}.{suffix}', facecolor='white')
    fig.savefig(out / f'{PREFIX}.png', dpi=600, facecolor='white')
    fig.savefig(out / f'{PREFIX}.preview.png', dpi=140, facecolor='white')
    plt.close(fig)
    verify_preserved(dict(zip(preserved.path, preserved.sha256)))
    validate(out)
    print('DESCRIPTIVE FIGURE COMPLETE:', out / f'{PREFIX}.pdf')


def validate(out):
    import pymupdf
    from PIL import Image
    wells, summary, _, _, _ = load_existing()
    saved_wells = read(out / 'panel_G_summary.wells.csv')
    saved = read(out / 'panel_G_summary.csv')
    pd.testing.assert_frame_equal(wells, saved_wells, check_exact=False, atol=1e-12, rtol=1e-12)
    for column in ['estimate', 'LOO_min', 'LOO_max']:
        np.testing.assert_array_equal(saved[column].to_numpy(), summary[column].to_numpy())
    assert saved.comparison.equals(summary.comparison)
    assert not any('p_value' in c or 'q_BH' in c or 'p_permutation' in c for c in saved.columns)
    preserved = read(out / f'{PREFIX}_data/preservation_manifest.csv')
    verify_preserved(dict(zip(preserved.path, preserved.sha256)))
    with pymupdf.open(out / f'{PREFIX}.pdf') as doc:
        assert len(doc) == 1
        page = doc[0]; text = page.get_text()
        assert 'Exploratory Cell Painting comparison summary' in text
        assert all(c in text for c in CONDITIONS)
        assert 'BH q' not in text
        assert len(page.get_images(full=True)) == 0
        for block in page.get_text('blocks'):
            rect = pymupdf.Rect(block[:4])
            assert rect.x0 >= 0 and rect.y0 >= 0 and rect.x1 <= page.rect.width and rect.y1 <= page.rect.height
    with Image.open(out / f'{PREFIX}.png') as png:
        pixels, dpi = list(png.size), list(png.info['dpi'])
    path = out / f'{PREFIX}.validation.json'
    validation = json.loads(path.read_text())
    validation.update({'status': 'complete', 'existing_numeric_summaries_preserved_exactly': True,
                       'existing_well_coordinates_and_group_assignments_verified': True,
                       'source_and_previous_outputs_unchanged_after_render': True,
                       'individual_well_points': 30, 'vector_PDF_no_raster_images': True,
                       'all_PDF_text_within_page': True, 'png_pixels': pixels, 'png_dpi': dpi,
                       'script_sha256': sha(Path(__file__))})
    dump(validation, path)
    print('VALIDATION PASSED: existing numeric values, groups, 30 well coordinates, vector PDF and unchanged source files.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=panel_dir('G'))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--render-only', action='store_true')
    modes.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    assert out.is_relative_to(RESULT.resolve()), 'Outputs must remain inside result/.'
    out.mkdir(parents=True, exist_ok=True)
    if args.validate_only: validate(out); return
    if not args.render_only: prepare(out)
    if not args.prepare_only: render(out)


if __name__ == '__main__':
    main()
