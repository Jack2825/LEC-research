"""List or execute declared analysis stages, always from this repository root."""
from pathlib import Path
import argparse, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=['aging','disease','cell_painting','cell_painting_primary','geomx','publication'])
    parser.add_argument('--step', help='Exact step ID from --list')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Run selected stages; otherwise only show commands.')
    parser.add_argument('--rscript', default='Rscript')
    args = parser.parse_args()
    stages = json.loads((ROOT/'config/workflow.json').read_text(encoding='utf-8'))
    chosen = [s for s in stages if (not args.branch or s['branch']==args.branch) and (not args.step or s['id']==args.step)]
    if not chosen:
        parser.error('No matching stages.')
    if args.execute and not (args.step or args.branch):
        parser.error('Choose --branch or --step to execute.')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', LEC_PROJECT_ROOT=str(ROOT))
    for stage in chosen:
        script = ROOT/stage['script']
        if stage['engine']=='python':
            command = [sys.executable, '-B', '-X', 'utf8', str(script), *stage.get('args', [])]
        elif stage['engine']=='Rmd':
            command = [args.rscript, str(ROOT/'tools/render_report.R'), str(script)]
        else:
            command = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script)]
        print(stage['id']+': '+subprocess.list2cmdline(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
