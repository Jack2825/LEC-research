"""Copy the existing computational outputs authorized for Panel E.

This import is separate from visualization; it never modifies upstream files.
Only existing derived tables, matrices, and provenance records are copied.
No raw microscopy files or raw CellProfiler exports are copied.
"""
from pathlib import Path
from figure_paths import panel_dir, resolve_result_path, assert_preserved
import csv
import hashlib
import shutil

PROJECT = Path(__file__).resolve().parents[1]
STUDY = PROJECT.parent / 'cell_painting_LEC_P3_P9_expanded_20260916'
DESTINATION = panel_dir('E') / 'panel_E_source'
FILES = {
    'results/03_states/cell_cluster_assignments.csv.gz': 'cell_cluster_assignments.csv.gz',
    'results/01_data_qc/feature_matrices.npz': 'feature_matrices.npz',
    'results/01_data_qc/cells_all_features.csv.gz': 'cells_all_features.csv.gz',
    'results/01_data_qc/feature_decisions.csv': 'feature_decisions.csv',
    'results/03_states/state_feature_profiles.csv': 'state_feature_profiles.csv',
    'results/03_states/candidate_screens.csv': 'candidate_screens.csv',
    'results/03_states/parameter_scan.csv': 'parameter_scan.csv',
    'results/03_states/state_well_support.csv': 'state_well_support.csv',
    'results/03_states/feature_panel_sensitivity.csv': 'feature_panel_sensitivity.csv',
    'results/01_data_qc/image_review_summary.json': 'image_review_summary.json',
    'config/analysis.json': 'analysis.json',
}


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    sources = [(STUDY / relative, DESTINATION / name) for relative, name in FILES.items()]
    for source, target in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists() and sha(source) != sha(target):
            raise FileExistsError(f'Different local source copy already exists: {target}')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    records = []
    for source, target in sources:
        before = sha(source)
        if not target.exists():
            shutil.copy2(source, target)
        assert before == sha(source) == sha(target)
        records.append({'original_path': str(source.resolve()),
                        'local_copy': str(target.relative_to(PROJECT)),
                        'sha256': before, 'bytes': source.stat().st_size,
                        'original_unchanged': True, 'copy_byte_identical': True})
    manifest = DESTINATION / 'import_manifest.csv'
    if not manifest.exists():
        with manifest.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f'Copied/verified {len(records)} existing output files; {sum(r["bytes"] for r in records):,} bytes.')
    print(f'Local source folder: {DESTINATION}')
    print('All upstream source files are unchanged; all copies are byte-identical.')


if __name__ == '__main__':
    main()
