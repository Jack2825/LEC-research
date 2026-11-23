"""Resolve and verify the declared inputs for current publication stages."""
from pathlib import Path
import csv
import hashlib

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('figures', 'formatting', 'tables', 'data_s8', 'calibration', 'representative')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def inside(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'Path escapes its declared root: {relative}')
    return target

def records(group):
    if group not in GROUPS:
        raise ValueError(f'Unknown input group: {group}')
    with (ROOT/'provenance/publication_input_manifest.csv').open(encoding='utf-8', newline='') as stream:
        return [row for row in csv.DictReader(stream) if row['group'] == group]

def check(group, hashes=True):
    missing, mismatched = [], []
    rows = records(group)
    for row in rows:
        path = inside(ROOT, row['path'])
        if not path.is_file():
            missing.append(row['path'])
        elif path.stat().st_size != int(row['bytes']) or (hashes and sha(path) != row['sha256']):
            mismatched.append(row['path'])
    return dict(group=group, declared=len(rows), missing=missing, checksum_mismatches=mismatched,
                status='FAIL' if missing or mismatched else 'PASS', sha256_checked=hashes)

def require_inputs(group):
    report = check(group)
    if report['status'] != 'PASS':
        details = '\n'.join((report['missing'] + report['checksum_mismatches'])[:8])
        raise FileNotFoundError(
            f"{group}: {len(report['missing'])} missing and {len(report['checksum_mismatches'])} mismatched inputs.\n"
            f"{details}\nSee provenance/publication_input_manifest.csv and docs/DATA_ACCESS.md."
        )
    return report
