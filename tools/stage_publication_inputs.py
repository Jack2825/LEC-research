"""Copy only declared inputs from a local study project into this checkout.

The default prints a plan. Add --copy to stage verified copies. The source
project is read-only. Existing destinations must already match their hashes.
For deposited data, restore the paths in the manifest instead of using this tool.
"""
from pathlib import Path
import argparse
import json
import shutil
from release_inputs import ROOT, GROUPS, inside, records, sha

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--from-project', type=Path, required=True)
    parser.add_argument('--copy', action='store_true')
    args = parser.parse_args()
    source_root = args.from_project.resolve()
    if source_root == ROOT:
        parser.error('The source project must differ from this checkout.')
    rows = records(args.group)
    pending = []
    for row in rows:
        target = inside(ROOT, row['path'])
        if target.exists():
            if not target.is_file() or sha(target) != row['sha256']:
                raise ValueError(f'Existing destination differs; nothing overwritten: {target}')
            continue
        if row['included'] == 'yes':
            raise FileNotFoundError(f'Bundled input missing; restore it from the code release: {target}')
        source = inside(source_root, row['source'])
        if not source.is_file() or sha(source) != row['sha256']:
            raise ValueError(f'Source absent or different from recorded revision: {source}')
        pending.append((source, target, row))
    # Check all files before starting any copies.
    if args.copy:
        for source, target, row in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha(target) != row['sha256']:
                raise ValueError(f'Copy checksum mismatch: {target}')
    print(json.dumps(dict(group=args.group, action='copied' if args.copy else 'plan only',
                         files=len(pending), bytes=sum(int(r['bytes']) for _,_,r in pending),
                         destination=str(ROOT)), indent=2))

if __name__ == '__main__':
    main()
