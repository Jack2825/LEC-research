"""Paths for the isolated publication revision; source projects are read-only."""
from pathlib import Path
import csv
import hashlib
import json

REVISION = Path(__file__).resolve().parents[1]
WORKSPACE = REVISION.parent
RESULT = REVISION / 'result'
SOURCES = REVISION / 'inputs'
CP = SOURCES / 'Cell_Painting_LEC_P3_P9_figures/result'
OLD = SOURCES / 'new_result2'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def destination(path):
    path = Path(path).resolve()
    if not path.is_relative_to(REVISION.resolve()):
        raise ValueError(f'Output outside the new revision folder: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    destination(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with destination(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))
