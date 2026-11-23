"""Panel folders and compatibility with pre-organization provenance paths.

Result files and historical manifests are preserved byte-for-byte. Only audited
script path edits may differ from the script hashes in those old manifests.
"""
from pathlib import Path
import hashlib
import json
import re

PROJECT = Path(__file__).resolve().parents[1]
RESULT = PROJECT / 'result'


def panel_dir(letter):
    letter = letter.upper()
    if letter not in 'ABCDEFG' or len(letter) != 1:
        raise ValueError(f'Unknown panel: {letter}')
    return RESULT / f'Panel_{letter}'


def panel_for_name(name):
    if name in {'assets', 'source_material'}:
        return 'A'
    match = re.match(r'panel_([a-g])(?:[_.]|$)', name, re.IGNORECASE)
    return match.group(1).upper() if match else None


def resolve_result_path(value):
    path = Path(value)
    if not path.is_absolute() and path.parts and path.parts[0].lower() == 'result':
        path = PROJECT / path
    path = path.resolve()
    if not path.is_relative_to(RESULT.resolve()):
        return path
    parts = path.relative_to(RESULT.resolve()).parts
    if not parts or re.fullmatch(r'Panel_[A-G]', parts[0]):
        return path
    letter = panel_for_name(parts[0])
    return panel_dir(letter).joinpath(*parts) if letter else path


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def assert_preserved(value, expected):
    """Check original data hashes, with explicit audited script-edit exceptions."""
    path = resolve_result_path(value)
    actual = file_sha(path)
    if actual == expected:
        return
    ledger = RESULT / 'panel_folder_script_updates.json'
    if path.parent == PROJECT / 'script' and ledger.is_file():
        records = json.loads(ledger.read_text(encoding='utf-8'))
        for row in records:
            if (path.name == row['script'] and expected == row['before_sha256']
                    and actual == row['after_sha256']):
                backup = (PROJECT / row['backup']).resolve()
                if backup.is_relative_to(PROJECT / 'script') and file_sha(backup) == expected:
                    return
    raise AssertionError(f'File differs from its recorded hash: {path}')
