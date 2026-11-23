"""Locate the original Arial fonts without bundling licensed font files."""
from pathlib import Path
import os

def arial(bold=False):
    variable = 'LEC_ARIAL_BOLD' if bold else 'LEC_ARIAL_REGULAR'
    filename = 'arialbd.ttf' if bold else 'arial.ttf'
    macname = 'Arial Bold.ttf' if bold else 'Arial.ttf'
    if os.environ.get(variable):
        path = Path(os.environ[variable]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'{variable} does not point to a font file: {path}')
        return str(path)
    candidates = [Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'/filename,
                  Path('/Library/Fonts')/macname,
                  Path('/System/Library/Fonts/Supplemental')/macname,
                  Path('/usr/share/fonts/truetype/msttcorefonts')/filename,
                  Path('/usr/share/fonts/truetype/msttcorefonts')/macname.replace(' ', '_')]
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f'Original Arial font required. Set {variable} to a licensed local {filename}; no metric-changing fallback is selected.')
