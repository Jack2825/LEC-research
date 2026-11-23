"""Check declared external inputs; optionally verify recorded SHA-256 checksums."""
from pathlib import Path
import argparse, csv, hashlib, json
from release_inputs import GROUPS, check

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['analysis', *GROUPS], default='analysis')
    parser.add_argument('--sha256', action='store_true')
    args = parser.parse_args()
    if args.group != 'analysis':
        report = check(args.group, hashes=args.sha256)
        print(json.dumps(report, indent=2))
        return int(report['status'] != 'PASS')
    name = 'required_assets.csv' if args.group == 'analysis' else 'figure_input_manifest.csv'
    with (ROOT/'provenance'/name).open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    missing, mismatched, ok = [], [], 0
    for row in records:
        path = ROOT/row['path']
        if not path.is_file():
            missing.append(row['path']); continue
        if args.sha256 and row['sha256']:
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != row['sha256']:
                mismatched.append(row['path']); continue
        ok += 1
    print(json.dumps({'group': args.group, 'present_and_checked': ok, 'missing': missing,
                      'checksum_mismatches': mismatched}, indent=2))
    return int(bool(missing or mismatched))

if __name__ == '__main__':
    raise SystemExit(main())
