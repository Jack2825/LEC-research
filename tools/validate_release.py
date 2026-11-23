"""Validate the code bundle without importing scientific packages or running analyses."""
from pathlib import Path
import argparse, ast, csv, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', action='store_true', help='Also require author-supplied publication metadata.')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    errors = []
    files = [p for p in ROOT.rglob('*.py') if not any(
        part in {'.venv', '.venv_scenic', '_validation'} for part in p.parts)]
    roots_checked = 0
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except SyntaxError as exc:
            errors.append(str(exc)); continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in {'find_project_dir', 'find_project_root', 'project_root', 'find_root'}:
                # Execute only the isolated path function, never module-level analysis code.
                namespace = {'Path': Path, '__file__': str(path)}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
                if namespace[node.name]() != ROOT:
                    errors.append(f'Incorrect repository root: {path.relative_to(ROOT)}')
                roots_checked += 1
    with (ROOT/'provenance/code_manifest.csv').open(encoding='utf-8', newline='') as f:
        code = list(csv.DictReader(f))
    for row in code:
        path = ROOT/row['release_path']
        if not path.is_file() or sha(path) != row['release_sha256']:
            errors.append(f'Code checksum mismatch: {row["release_path"]}')
        if '_before_' in row['release_path'] or row['release_path'].startswith('new_scripts/'):
            errors.append(f'Historical script included: {row["release_path"]}')
    for name in ['packaged_asset_manifest.csv', 'support_code_manifest.csv']:
        with (ROOT/'provenance'/name).open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                path = (ROOT/row['release_path']).resolve()
                if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != row['release_sha256']:
                    errors.append(f'Bundled asset/helper checksum mismatch: {row["release_path"]}')
    from release_inputs import GROUPS, records, inside
    declared = {}
    for group in GROUPS:
        for row in records(group):
            path = inside(ROOT, row['path'])
            if row['path'] in declared and declared[row['path']] != row['sha256']:
                errors.append(f'Conflicting input hashes: {row["path"]}')
            declared[row['path']] = row['sha256']
            if row['included'] == 'yes' and (not path.is_file() or sha(path) != row['sha256']):
                errors.append(f'Included publication input mismatch: {row["path"]}')
    with (ROOT/'provenance/excluded_scripts.csv').open(encoding='utf-8', newline='') as f:
        excluded = list(csv.DictReader(f))
    for row in excluded:
        if (ROOT/row['excluded_script']).is_file():
            errors.append(f'Excluded script exists: {row["excluded_script"]}')
    workflow = json.loads((ROOT/'config/workflow.json').read_text(encoding='utf-8'))
    ids = [r['id'] for r in workflow]
    if len(ids) != len(set(ids)):
        errors.append('Duplicate workflow step IDs')
    for step in workflow:
        if not (ROOT/step['script']).is_file():
            errors.append(f'Missing workflow script: {step["script"]}')
    metadata = json.loads((ROOT/'config/publication_metadata.json').read_text(encoding='utf-8'))
    pending = [key for key, value in metadata.items() if not value]
    if args.release:
        errors.extend(f'Publication metadata pending: {key}' for key in pending)
    oversized = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
                 if p.is_file() and '_validation' not in p.parts and '.venv' not in p.parts
                 and '.venv_scenic' not in p.parts and p.stat().st_size >= 100*1024*1024]
    errors.extend(f'Oversized bundled file: {name}' for name in oversized)
    report = {'status': 'FAIL' if errors else 'PASS', 'curated_source_scripts': len(code),
              'excluded_source_scripts': len(excluded), 'root_functions_checked': roots_checked,
              'syntax_checked': len(files), 'workflow_steps': len(workflow),
              'publication_input_groups': list(GROUPS),
              'distinct_declared_publication_inputs': len(declared),
              'pending_publication_metadata': pending, 'errors': errors,
              'scope': 'Bundle syntax, checksums, root isolation and declared workflow; not a full scientific rerun'}
    print(json.dumps(report, indent=2))
    if args.json:
        output = args.json.resolve()
        if not output.is_relative_to(ROOT):
            raise ValueError('Validation report must be inside this repository.')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return int(bool(errors))

if __name__ == '__main__':
    raise SystemExit(main())
