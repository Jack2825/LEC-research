"""Embed the complete user-supplied JPEG in Figure 4A, without cropping.

All writes stay in this new November 9 folder. Earlier revisions stay read-only.
The supplied JPEG is embedded as its original, byte-identical JPEG stream.
The existing design schematic is retained. The first row grows by 100 PDF points;
the remaining main-figure panels are translated down together at unchanged scale.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys

import numpy as np
from PIL import Image
import pymupdf as fitz

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0,str(ROOT/'tools'))
from release_inputs import require_inputs
PREVIOUS = BASE / 'inputs/previous'
OUT = BASE / 'result'
SOURCE = OUT / 'source'
MAIN = OUT / 'main/Fig04'
REVIEW = OUT / 'review'
INPUT_JPEG = BASE / 'inputs/20261105-updated one.jpg'
CUT_Y = 210
DOWN = 100


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path, record):
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')


def snapshot(src, dst):
    if not dst.exists():
        shutil.copy2(src, dst)
    assert sha(src) == sha(dst)


def arr(page, clip=None):
    pm = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
    return np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, 3)


def compare(reference, revised, allowed):
    a, b = arr(reference), arr(revised)
    assert a.shape == b.shape
    keep = np.ones(a.shape[:2], dtype=bool)
    r = fitz.Rect(allowed) + (-2, -2, 2, 2)
    x0, y0, x1, y1 = [int(v*2) for v in r]
    keep[max(0,y0):y1+1, max(0,x0):x1+1] = False
    difference = np.max(np.abs(a.astype(np.int16)-b.astype(np.int16)), axis=2)
    changed = int(((difference > 3) & keep).sum())
    assert changed == 0, changed
    return dict(pixels_compared_outside_panel_A=int(keep.sum()),
                changed_pixels_outside_panel_A=changed)


def place_intact(page, bbox, raw):
    xref = page.insert_image(bbox, stream=raw, keep_proportion=True)
    assert page.parent.xref_stream_raw(xref) == raw
    info = [i for i in page.get_image_info(xrefs=True) if i['xref'] == xref]
    assert len(info) == 1
    assert (info[0]['width'], info[0]['height']) == (2695, 1828)
    actual = fitz.Rect(info[0]['bbox'])
    assert max(abs(a-b) for a,b in zip(actual, bbox)) < .001
    return dict(xref=xref, bbox=list(actual), width_px=2695, height_px=1828,
                full_JPEG_stream_sha256=hashlib.sha256(raw).hexdigest(),
                original_JPEG_bytes_preserved=True, cropped=False,
                resampled=False, labels_redrawn=False, plots_redrawn=False,
                nominal_effective_DPI=2695/actual.width*72)


def build_panel_a(raw):
    source = fitz.open(SOURCE / 'Fig04A_previous.pdf')
    reference = fitz.open()
    rp = reference.new_page(width=source[0].rect.width, height=530)
    rp.show_pdf_page(source[0].rect, source, 0)
    result = fitz.open()
    page = result.new_page(width=rp.rect.width, height=rp.rect.height)
    page.show_pdf_page(page.rect, reference, 0)
    # Cover only the previous image/plot region. Keep title, A, and schematic.
    cover = fitz.Rect(230, 68, page.rect.width, page.rect.height)
    page.draw_rect(cover, color=None, fill=(1, 1, 1))
    target = fitz.Rect(235, 70, 872, 70 + (872-235)*1828/2695)
    record = place_intact(page, target, raw)
    audit = compare(rp, page, cover)
    target_path = MAIN / 'panels/Fig04A.pdf'
    result.set_metadata({'title': 'Figure 4A', 'subject': 'Complete supplied tube JPEG; original design schematic retained'})
    result.save(target_path, garbage=4, deflate=True)
    result.close()
    source.close()
    reference.close()
    return record, audit, target


def build_whole(raw, native_rect):
    source = fitz.open(SOURCE / 'Fig04_previous.pdf')
    native = fitz.open(SOURCE / 'Fig04A_previous.pdf')
    # Recover the existing native-A-to-whole-figure transform from the same photo.
    def first_photo(page):
        rows = [i for i in page.get_image_info() if (i['width'],i['height']) == (852,639)]
        assert len(rows) == 4
        return fitz.Rect(sorted(rows, key=lambda i: (round(i['bbox'][1],2),i['bbox'][0]))[0]['bbox'])
    a, b = first_photo(native[0]), first_photo(source[0])
    scale = b.width/a.width
    dx, dy = b.x0-scale*a.x0, b.y0-scale*a.y0
    target = fitz.Rect(*(v*scale+(dx if i%2==0 else dy) for i,v in enumerate(native_rect)))
    assert target.y1 < CUT_Y+DOWN-10
    reference = fitz.open()
    old = source[0]
    rp = reference.new_page(width=old.rect.width, height=old.rect.height+DOWN)
    rp.show_pdf_page(fitz.Rect(0,0,old.rect.width,CUT_Y), source, 0,
                     clip=fitz.Rect(0,0,old.rect.width,CUT_Y))
    rp.show_pdf_page(fitz.Rect(0,CUT_Y+DOWN,old.rect.width,old.rect.height+DOWN), source, 0,
                     clip=fitz.Rect(0,CUT_Y,old.rect.width,old.rect.height))
    result = fitz.open()
    page = result.new_page(width=rp.rect.width,height=rp.rect.height)
    page.show_pdf_page(page.rect,reference,0)
    cover = fitz.Rect(128.5,61,469.5,CUT_Y+DOWN)
    page.draw_rect(cover, color=None, fill=(1,1,1))
    record = place_intact(page,target,raw)
    audit = compare(rp,page,cover)
    # Retain the original drawing stream verbatim inside the imported PDF form.
    # Comparing against the identically placed reference avoids tiny rasterizer
    # rounding changes caused by embedding a page with fractional-point height.
    original_stream = old.read_contents()
    assert any(result.xref_stream(row[0]) == original_stream
               for row in page.get_xobjects())
    lower_before = arr(old,fitz.Rect(0,CUT_Y,old.rect.width,old.rect.height))
    lower_after = arr(page,fitz.Rect(0,CUT_Y+DOWN,old.rect.width,old.rect.height+DOWN))
    assert lower_before.shape == lower_after.shape
    delta = np.max(np.abs(lower_before.astype(np.int16)-lower_after.astype(np.int16)),axis=2)
    changed = int((delta>3).sum())
    lower_reference = arr(rp,fitz.Rect(0,CUT_Y+DOWN,old.rect.width,old.rect.height+DOWN))
    assert np.array_equal(lower_reference, lower_after)
    audit.update(lower_panels_translation_pt=DOWN, lower_panels_scale_factor=1,
                 original_drawing_stream_retained_verbatim=True,
                 changed_lower_panel_pixels_vs_unmodified_translation_reference=0,
                 rasterizer_differences_vs_direct_original_render=changed,
                 rasterizer_max_channel_difference=int(delta.max()),
                 panel_B_preserved_at_original_position=True)
    result.set_metadata({'title': 'Figure 4', 'subject': 'Complete supplied tube image in A; B-G scientific artwork unchanged'})
    result.save(MAIN/'Fig04.pdf',garbage=4,deflate=True)
    result.close()
    reference.close()
    native.close()
    source.close()
    return record,audit


def finish_outputs():
    for name in ['Fig04.pdf','panels/Fig04A.pdf']:
        with fitz.open(MAIN/name) as doc:
            factor=1600/doc[0].rect.width
            doc[0].get_pixmap(matrix=fitz.Matrix(factor,factor),alpha=False).save((MAIN/name).with_suffix('.preview.png'))
    shutil.copy2(MAIN/'Fig04.pdf',OUT/'figures/Fig04.pdf')
    with fitz.open() as book:
        for name in [f'Fig04{c}.pdf' for c in 'ABCDEFG']:
            with fitz.open(MAIN/'panels'/name) as d:
                book.insert_pdf(d)
        book.save(MAIN/'Fig04_individual_panels.pdf',garbage=4,deflate=True)
    with fitz.open() as book:
        for name in [f'Fig{i:02d}.pdf' for i in range(1,8)]+['Graphic_Abstract.pdf']:
            with fitz.open(OUT/'figures'/name) as d:
                book.insert_pdf(d)
        book.save(REVIEW/'All_main_figures.pdf',garbage=4,deflate=True)
    with fitz.open() as book:
        for name in ['Fig04.pdf','panels/Fig04A.pdf']:
            page=book.new_page(width=1600,height=1100)
            for x,root,label in [(20,PREVIOUS/'result/main/Fig04','November 5'),(810,MAIN,'November 9')]:
                page.insert_text((x,25),f'{label}: {Path(name).stem}',fontsize=14)
                with fitz.open(root/name) as d:
                    page.show_pdf_page(fitz.Rect(x,45,x+770,1080),d,0)
        book.save(REVIEW/'Figure4_before_after.pdf',garbage=4,deflate=True)


def legend():
    old=(SOURCE/'Fig04_legend_previous.md').read_text(encoding='utf-8')
    start=old.index('A: ')
    stop=old.index('B: ',start)
    text=old[:start]+('A: Separate P3/P9 Control/H1N1 tube-formation experiment. '
        'The existing design schematic is retained. The complete supplied '
        '20261105-updated one.jpg is embedded intact, including all four representative '
        'images, both quantification plots, group labels, statistical annotations '
        '(** for P3; ns for P9), and the supplied 1 mm bar on the P3-Control image. '
        'The image is placed proportionally without cropping, rearranging its components, '
        'redrawing plots, or recompressing its JPEG stream. No numerical tube source '
        'table or test/replicate definition was supplied. ')+old[stop:]
    (MAIN/'Fig04_legend.md').write_text(text,encoding='utf-8')
    shutil.copy2(MAIN/'Fig04_legend.md',OUT/'documents/Fig04_legend_updated_20261109.md')


def main():
    require_inputs('figures')
    for p in [SOURCE,MAIN/'panels',REVIEW,OUT/'figures',OUT/'documents']:
        p.mkdir(parents=True,exist_ok=True)
    pairs=[(INPUT_JPEG,SOURCE/INPUT_JPEG.name),
        (PREVIOUS/'result/main/Fig04/Fig04.pdf',SOURCE/'Fig04_previous.pdf'),
        (PREVIOUS/'result/main/Fig04/panels/Fig04A.pdf',SOURCE/'Fig04A_previous.pdf'),
        (PREVIOUS/'result/main/Fig04/Fig04_legend.md',SOURCE/'Fig04_legend_previous.md')]
    before={str(p):sha(p) for p,_ in pairs}
    for src,dst in pairs:
        snapshot(src,dst)
    for p in (PREVIOUS/'result/figures').glob('*.pdf'):
        before[str(p)]=sha(p)
        shutil.copy2(p,OUT/'figures'/p.name)
    for p in (PREVIOUS/'result/main/Fig04/panels').glob('Fig04[BCDEFG].*'):
        before[str(p)]=sha(p)
        shutil.copy2(p,MAIN/'panels'/p.name)
    for name in ['Fig04B_calibration.json','Fig04B_source_field_comparison.csv']:
        shutil.copy2(PREVIOUS/'result/review'/name,REVIEW/name)
    scale_note=(PREVIOUS/'result/documents/Scale_bar_legend_update.md').read_text(encoding='utf-8')
    (OUT/'documents/Scale_bar_legend_update.md').write_text(
        scale_note.replace('Fig04_legend_updated_20261105.md','Fig04_legend_updated_20261109.md'),
        encoding='utf-8')
    raw=(SOURCE/INPUT_JPEG.name).read_bytes()
    with Image.open(SOURCE/INPUT_JPEG.name) as im:
        assert im.size==(2695,1828)
    a,ac,rect=build_panel_a(raw)
    whole,wc=build_whole(raw,rect)
    finish_outputs()
    legend()
    rows=[]
    for p in sorted((OUT/'figures').glob('*.pdf')):
        same=sha(p)==sha(PREVIOUS/'result/figures'/p.name)
        assert same==(p.name!='Fig04.pdf')
        rows.append(dict(file=p.name,sha256=sha(p),unchanged_from_20261105=same))
    assert len(rows)==16
    for letter in 'BCDEFG':
        assert sha(MAIN/f'panels/Fig04{letter}.pdf')==sha(PREVIOUS/f'result/main/Fig04/panels/Fig04{letter}.pdf')
    assert all(sha(p)==s for p,s in before.items())
    # Check the saved PDF streams, not just the unsaved in-memory document.
    for path in [MAIN/'Fig04.pdf',MAIN/'panels/Fig04A.pdf']:
        with fitz.open(path) as d:
            exact=[i for i in d[0].get_image_info(xrefs=True)
                   if (i['width'],i['height'])==(2695,1828)
                   and d.xref_stream_raw(i['xref'])==raw]
            assert len(exact)==1
    with (REVIEW/'final_figure_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save_json(REVIEW/'validation.json',dict(originals_unchanged=True,source_hashes=before,
        intact_JPEG_SHA256=sha(SOURCE/INPUT_JPEG.name),panel_A=a,whole_figure=whole,
        panel_A_audit=ac,whole_figure_audit=wc,unchanged_other_figures=15,
        unchanged_individual_panels=list('BCDEFG'),
        scale_bars_panel_B='Unchanged: white bars inside, 100 µm labels below images',
        no_new_numerical_analysis=True))
    (OUT/'README.md').write_text('''# Exact supplied Figure 4A picture — November 9, 2026

Use `figures/` for all 16 figure PDFs. Only Figure 4 changed; the other 15 are exact copies of the November 5 version.

Figure 4A now contains the ENTIRE supplied `20261105-updated one.jpg`, including its images, plots, labels and 1 mm bar, as one image. The original JPEG bytes are embedded unchanged. No cropping, reconstruction, intensity adjustment or JPEG recompression was used. The existing experimental-design schematic is retained separately alongside it.

Panel A has more vertical space. In the whole figure, panels C–G move down together by 100 PDF points at their original scale. Figure 4B retains its white 100 µm bars inside the images and black labels below. Individual panels B–G are byte-identical to November 5.

Files:
- `main/Fig04/Fig04.pdf`: revised whole figure.
- `main/Fig04/panels/Fig04A.pdf`: revised separate A.
- `main/Fig04/Fig04_individual_panels.pdf`: seven panel pages.
- `documents/Fig04_legend_updated_20261109.md`: current Figure 4 legend.
- `review/Figure4_before_after.pdf`: visual comparison.
- `review/validation.json`: exact JPEG and artwork-preservation checks.
- `source/`: unchanged supplied image and prior figure inputs.

All original images, data, Word files, and previous revision folders remain unchanged. This update does not change or resolve other deferred figure issues.

Reproduce with the configured Python environment `.venv/Scripts/python.exe -B -X utf8 CVR_figure_revision_20261109/script/use_exact_tube_picture.py`. All generated files stay in this folder. PNGs are previews; PDFs retain the intact source JPEG and other vector artwork.
''',encoding='utf-8')
    print('Complete: exact full JPEG in Figure 4A; B–G preserved; all 16 figures packaged.',flush=True)


if __name__=='__main__':
    main()
