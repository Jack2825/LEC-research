"""Reproduce the November 9 final figures, or their current formatting stage.

Final assembly uses declared frozen artwork and the exact supplied tube JPEG.
Formatting is a separate upstream stage for the fifteen figures carried forward.
It does not rerun expression analysis or replace supplied experimental plots.
"""
import argparse
import subprocess
import sys
from release_inputs import ROOT, require_inputs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formatting', action='store_true', help='Run the current upstream figure/S7/graphical-abstract formatting instead of final assembly.')
    parser.add_argument('--verify-native-images', action='store_true', help='Recheck S7 acquisition calibration against original ND2; requires --formatting.')
    args = parser.parse_args()
    if args.verify_native_images and not args.formatting:
        parser.error('--verify-native-images requires --formatting')
    require_inputs('formatting' if args.formatting else 'figures')
    if args.formatting:
        folder = ROOT/'CVR_figure_revision_20261103/script'
        jobs = [('prepare.py', []), ('format_figures.py', ['--verify-native-images'] if args.verify_native_images else []),
                ('format_graphical_abstract.py', [])]
    else:
        folder = ROOT/'CVR_figure_revision_20261109/script'
        jobs = [('use_exact_tube_picture.py', [])]
    for name, extra in jobs:
        subprocess.run([sys.executable, '-B', '-X', 'utf8', str(folder/name), *extra], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
