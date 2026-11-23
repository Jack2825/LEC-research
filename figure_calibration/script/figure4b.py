"""Verify Figure 4B acquisition calibration and add the current 100-micrometre bars.
The superseded tube-photo cropping/replacement functions are intentionally excluded.
"""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, sys
import h5py, nd2, numpy as np
from PIL import Image
import pymupdf as fitz
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parent
sys.path.insert(0,str(ROOT/'tools'))
from release_fonts import arial
from release_inputs import require_inputs
SOURCE=BASE/'inputs'
REVIEW=BASE/'result/review'
FONT=arial()

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def calibrate():
    """Confirm the existing CP image index against all 180 acquired fields."""
    root = ROOT / 'Data/vivo'
    cp = SOURCE / 'CellProfiler.cpproj'
    with h5py.File(cp, 'r') as f:
        run = next(iter(f['Measurements'].values()))
        imgs = run['Image']
        assert int(imgs['ImageNumber/data'][1]) == 2
        filename = imgs['FileName_DAPI/data'][1].decode()
        position = int(imgs['Series_DAPI/data'][1])
    candidates = list(root.rglob(filename))
    assert len(candidates) == 1 and position == 1
    matched = candidates[0]
    with Image.open(root / 'CorrDAPI0002.tiff') as im:
        target = np.asarray(im)[::16, ::16].astype(float).ravel()
    target -= target.mean()
    target /= np.linalg.norm(target)
    search = []
    for p in sorted(root.rglob('*.nd2')):
        with nd2.ND2File(p) as f:
            ci = next(i for i, c in enumerate(f.metadata.channels)
                      if 'DAPI' in c.channel.name)
            for j in range(f.sizes.get('P', 1)):
                a = f.read_frame(j)[ci, ::16, ::16].astype(float).ravel()
                a -= a.mean()
                a /= np.linalg.norm(a)
                search.append(dict(path=str(p), position_index=j,
                                   nuclear_correlation=float(a @ target)))
    search.sort(key=lambda r: r['nuclear_correlation'], reverse=True)
    assert Path(search[0]['path']) == matched
    assert search[0]['position_index'] == position
    assert search[0]['nuclear_correlation'] > .99
    assert search[1]['nuclear_correlation'] < .2
    csv_write(REVIEW / 'Fig04B_source_field_comparison.csv', search)
    checks = []
    with nd2.ND2File(matched) as f:
        um = f.voxel_size().x
        assert abs(um - f.voxel_size().y) < 1e-12
        assert abs(um - .324081161957359) < 1e-12
        assert f.sizes['X'] == f.sizes['Y'] == 2048
        raw = f.read_frame(position)
        for i, channel in enumerate(['CY5', 'TRITC', 'FITC', 'DAPI']):
            assert channel in f.metadata.channels[i].channel.name
            with Image.open(root / f'Corr{channel}0002.tiff') as im:
                corr = np.asarray(im).astype(float)
            tile_corr = []
            for y in range(0, 2048, 128):
                for x in range(0, 2048, 128):
                    a = raw[i, y:y+128, x:x+128].ravel()
                    b = corr[y:y+128, x:x+128].ravel()
                    tile_corr.append(float(np.corrcoef(a, b)[0, 1]))
            median = float(np.median(tile_corr))
            assert median > .99
            checks.append(dict(channel=channel, local_correlation_median=median,
                               corrected_tiff_sha256=sha(root / f'Corr{channel}0002.tiff'),
                               display_tiff_sha256=sha(root / f'Corr{channel}0002_final.tif')))
        calibrated_axes = [list(c.volume.axesCalibrated)
                           for c in f.frame_metadata(position).channels]
        assert all(v[:2] == [True, True] for v in calibrated_axes)
    record = dict(source_nd2=str(matched), source_sha256=sha(matched),
                  cellprofiler_project_sha256=sha(cp), cellprofiler_image_number=2,
                  field_index_zero_based=position, field_ordinal=2,
                  acquisition_size_px=[2048, 2048], um_per_pixel=um,
                  full_field_width_um=2048*um, bar_length_um=100,
                  bar_length_original_pixels=100/um,
                  comparison_fields=len(search), top_two_matches=search[:2],
                  channels=checks, calibrated_axes=calibrated_axes,
                  method='CP index plus pixel-pattern matching; original ND2 physical calibration')
    dump(REVIEW / 'Fig04B_calibration.json', record)
    return record


def add_bars(page, calibration):
    images = sorted([x for x in page.get_image_info() if x['width'] == x['height'] == 2048],
                    key=lambda r: r['bbox'][0])
    assert len(images) == 4
    records = []
    page.insert_font(fontname='ScaleArial', fontfile=FONT)
    font = fitz.Font(fontfile=FONT)
    for name, img in zip(['Merge', 'Nucleus', 'ER', 'Mitochondria'], images):
        box = fitz.Rect(img['bbox'])
        length = 100 / calibration['full_field_width_um'] * box.width
        margin = 5
        x0 = box.x0 + margin
        x1 = x0 + length
        y = box.y1 - margin
        size = 8
        label = '100 µm'
        width = font.text_length(label, fontsize=size)
        # Match the supplied tube-image placement: bar near the lower left.
        # The label is black on the white margin BELOW the image, with a clear gap.
        tx = max(box.x0 + 1, (x0 + x1)/2 - width/2)
        ty = box.y1 + 12
        bar_rect = fitz.Rect(x0-1, y-1, x1+1, y+1)
        label_rect = fitz.Rect(tx, ty-size, tx+width, ty+3)
        assert box.contains(bar_rect)
        assert label_rect.y0 >= box.y1 + 4
        assert label_rect.x0 >= box.x0 and label_rect.x1 <= box.x1
        mask = (bar_rect | label_rect) + (-1, -1, 1, 1)
        page.insert_text((tx, ty), label, fontname='ScaleArial', fontsize=size, color=(0, 0, 0))
        page.draw_line((x0, y), (x1, y), color=(1, 1, 1), width=2)
        records.append(dict(kind='calibrated_scale_bar', channel_label=name,
                            image_bbox=list(box), rect=list(mask), bar_xyxy=[x0, y, x1, y],
                            label_bbox=list(label_rect), label_origin=[tx, ty],
                            style='White bar inside lower left; black label below image',
                            length_um=100, font_size_pt=size,
                            um_per_original_pixel=calibration['um_per_pixel']))
    return records


def pixel_audit(old_path, new_path, edits):
    with fitz.open(old_path) as old, fitz.open(new_path) as new:
        assert old[0].rect == new[0].rect
        a = old[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        b = new[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        aa = np.frombuffer(a.samples, np.uint8).reshape(a.height, a.width, 3)
        bb = np.frombuffer(b.samples, np.uint8).reshape(b.height, b.width, 3)
        mask = np.ones(aa.shape[:2], bool)
        for e in edits:
            r = fitz.Rect(e['rect']) + (-1, -1, 1, 1)
            x0, y0 = np.floor(np.array([r.x0, r.y0])*2).astype(int)
            x1, y1 = np.ceil(np.array([r.x1, r.y1])*2).astype(int)
            mask[max(0, y0):y1+1, max(0, x0):x1+1] = False
        delta = np.max(np.abs(aa.astype(np.int16)-bb.astype(np.int16)), axis=2)
        changed = int(np.sum((delta > 3) & mask))
        assert changed == 0, (new_path, changed)
        old_b = [i['digest'].hex() for i in old[0].get_image_info(hashes=True)
                 if i['width'] == i['height'] == 2048]
        new_b = [i['digest'].hex() for i in new[0].get_image_info(hashes=True)
                 if i['width'] == i['height'] == 2048]
        assert old_b == new_b
        # No text replacement; the only new text is the scale label.
        before = Counter(old[0].get_text().split())
        after = Counter(new[0].get_text().split())
        assert not before - after
        return dict(file=str(new_path.relative_to(BASE)),
                    checked_pixels=int(mask.sum()), changed_pixels_outside_edit_regions=changed,
                    page_geometry_unchanged=True, original_text_retained=True,
                    cell_painting_image_payloads_unchanged=True,
                    quantification_and_other_panels_unchanged=True)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,help='Optional unbarred Figure 4B PDF; the calibrated output stays in this module result folder.')
    args=parser.parse_args()
    require_inputs('calibration')
    REVIEW.mkdir(parents=True,exist_ok=True)
    record=calibrate()
    if args.panel:
        target=BASE/'result/Fig04B_calibrated.pdf'
        with fitz.open(args.panel) as doc:
            if '100 µm' in doc[0].get_text():raise ValueError('Input already has scale labels.')
            edits=add_bars(doc[0],record)
            doc.save(target,garbage=4,deflate=True)
        dump(REVIEW/'pixel_audit.json',pixel_audit(args.panel,target,edits))
if __name__=='__main__':main()
