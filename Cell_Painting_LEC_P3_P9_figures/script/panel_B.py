"""Assemble Panel B using existing Fig. 2E channel artwork only.

Run with the workspace Python environment:
  python -B script/panel_B.py --inspect-only
  python -B script/panel_B.py --review-draft
  python -B script/panel_B.py --channel-labels fig2e

The last option is for use ONLY after the channel-label discrepancy is resolved.
No source image is changed. No quantitative analysis or biological procedures
are performed. A merge combines real source channels, with no synthetic pixels.
"""
from pathlib import Path
from figure_paths import panel_dir, resolve_result_path, assert_preserved
from io import BytesIO
import argparse
import csv
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_OUT = panel_dir('B')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'result' / '.matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
import pymupdf

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'image.composite_image': False, 'savefig.facecolor': 'white'})

PDF = ROOT / 'result/Panel_A/source_material/Fig2F_source.pdf'
DATA = WORKSPACE / 'Data/vivo'
STUDY = WORKSPACE / 'cell_painting_LEC_P3_P9_expanded_20260916'
# Explicit spatial mapping from the supplied PDF, independent of TIFF filenames.
CHANNELS = [
    ('DAPI', 'Nucleus', (137.2546, 207.3272, 248.3091, 318.1636)),
    ('FITC', 'ER', (21.1818, 73.3636, 132.2364, 184.4182)),
    ('CY5', 'Mitochondria', (21.1818, 207.3272, 132.2364, 318.1636)),
    ('TRITC', 'Plasma', (137.4727, 74.4546, 248.3090, 185.2909)),
]
GROUPS = ['Control', 'H1N1', 'VEGF-C', 'H1N1+VEGF-C', 'UC2288', 'H1N1+UC2288']


def sha(path):
    with resolve_result_path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source_paths():
    files = [PDF, DATA/'0002overlay_final.tif',
             STUDY/'results/01_data_qc/original_field_qc.csv',
             STUDY/'results/01_data_qc/plate_condition_mapping.csv']
    for channel, _, _ in CHANNELS:
        files += [DATA/f'Corr{channel}0002_final.tif', DATA/f'Corr{channel}0002.tiff']
    return files


def inspect(out):
    assets = out/'panel_B_assets'
    assets.mkdir(exist_ok=True)
    images, rows = {}, []
    with pymupdf.open(PDF) as doc:
        blocks = [b for b in doc[0].get_text('dict')['blocks'] if b['type'] == 1]
        for channel, label, expected in CHANNELS:
            block = min(blocks, key=lambda b: sum(abs(x-y) for x,y in zip(b['bbox'], expected)))
            if max(abs(x-y) for x,y in zip(block['bbox'], expected)) > .05:
                raise ValueError('Fig. 2E source geometry changed: inspect group/channel mapping.')
            path = DATA/f'Corr{channel}0002_final.tif'
            with Image.open(path) as im:
                if im.size != (2048,2048) or im.mode != 'RGB':
                    raise ValueError('Unexpected display-image geometry or color mode.')
                arr = np.asarray(im).copy()
                description = str(im.tag_v2.get(270, ''))
            images[channel] = arr
            image = Image.fromarray(arr)
            asset_path = assets/f'{channel}_source_display.png'
            image.save(asset_path)
            assert np.array_equal(arr, np.asarray(Image.open(asset_path)))
            embedded = Image.open(BytesIO(block['image'])).convert('RGB')
            reduced = image.resize(embedded.size, Image.Resampling.LANCZOS)
            a = np.asarray(embedded, dtype=float)
            b = np.asarray(reduced, dtype=float)
            corr = float(np.corrcoef(a.ravel(), b.ravel())[0,1])
            if corr < .98:
                raise ValueError('TIFF does not closely match the existing Fig. 2E artwork.')
            with Image.open(DATA/f'Corr{channel}0002.tiff') as raw:
                raw_mode = raw.mode
                raw_size = raw.size
            rows.append({'channel_filename_token':channel, 'fig2e_label':label,
                         'display_file':path.relative_to(WORKSPACE).as_posix(),
                         'source_sha256':sha(path), 'width':2048, 'height':2048,
                         'display_mode':'RGB', 'display_bits_per_component':8,
                         'raw_mode':raw_mode, 'raw_dimensions':str(raw_size),
                         'pdf_source_bbox':json.dumps(block['bbox']),
                         'pdf_embedded_dimensions':str(embedded.size),
                         'pdf_to_TIFF_RGB_correlation':corr,
                         'mean_absolute_RGB_difference':float(np.abs(a-b).mean()),
                         'TIFF_description':description.strip(),
                         'micrometres_per_pixel_verified':False,
                         'well_identity_verified':False,
                         'extracted_display_pixels_unchanged':True})
    with (out/'panel_B.source_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return images, rows


def box(ax, xy, width, height, text, size=10, bold=False):
    ax.add_patch(FancyBboxPatch(xy,width,height,
        boxstyle='round,pad=0.009,rounding_size=0.018',
        facecolor='#eaf0f5',edgecolor='#87939c',linewidth=.8))
    ax.text(xy[0]+width/2,xy[1]+height/2,text,ha='center',va='center',
            fontsize=size,fontweight='bold' if bold else 'normal',color='#152431')


def schematic(ax):
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    ax.text(.49,.97,'LECs',ha='center',va='center',fontsize=12,fontweight='bold')
    box(ax,(.58,.79),.16,.12,'P3',12,True)
    box(ax,(.80,.79),.16,.12,'P9',12,True)
    for y,label in zip(np.linspace(.716,.166,6),GROUPS):
        ax.text(.02,y,label,ha='left',va='center',fontsize=9.5,color='#152431')
        for x in [.66,.88]:
            ax.plot(x,y,'o',color='#626d76',markersize=5)
        ax.plot([.01,.98],[y-.053,y-.053],color='#e4e8eb',linewidth=.6)
    ax.text(.49,.035,'Six groups in each passage',ha='center',va='center',
            fontsize=9,color='#626d76')


def add_image(fig, arr, rect):
    ax = fig.add_axes(rect)
    ax.imshow(arr,interpolation='none',aspect='equal')
    ax.set_axis_off()


def render(out, images, mapping, draft):
    # These files already have a linear RGB display encoding. Retain their
    # existing intensity mapping: no autoscaling, gamma, contrast or sharpening.
    nucleus = images['DAPI']
    er = images['FITC']
    mito_source = 'CY5' if mapping == 'fig2e' else 'TRITC'
    mito = images[mito_source].copy()
    if mapping == 'expanded':
        # TRITC's supplied yellow LUT contains the same scalar values in R/G.
        # Remap only its color to red, retaining the exact scalar pixel values.
        assert np.array_equal(mito[:,:,0],mito[:,:,1])
        mito[:,:,1] = 0
    assert not nucleus[:,:,:2].any()
    assert not er[:,:,[0,2]].any()
    assert not mito[:,:,1:].any()
    merged = np.stack([mito[:,:,0],er[:,:,1],nucleus[:,:,2]],axis=-1)
    assert np.array_equal(merged[:,:,0],mito[:,:,0])
    assert np.array_equal(merged[:,:,1],er[:,:,1])
    assert np.array_equal(merged[:,:,2],nucleus[:,:,2])
    arrays = [('merged',merged),('nucleus',nucleus),('ER',er),('mitochondria',mito)]
    for name, arr in arrays:
        Image.fromarray(arr).save(out/'panel_B_assets'/f'{name}_display.png')
    fig = plt.figure(figsize=(12.2,4.6),facecolor='white')
    fig.text(.018,.969,'B',fontsize=21,fontweight='bold',va='top')
    fig.text(.054,.962,'Cell Painting design and representative images',
             fontsize=15,fontweight='bold',va='top')
    fig.text(.054,.889,'Separate discovery experiment',fontsize=10,color='#626d76')
    fig.text(.147,.803,'Experimental design',ha='center',fontsize=11,fontweight='bold')
    fig.text(.647,.803,'Representative image set (Fig. 2E)',ha='center',fontsize=11,fontweight='bold')
    schematic(fig.add_axes([.025,.135,.245,.600]))
    width = .160
    height = width*12.2/4.6
    for x,(_,arr),label in zip([.304,.479,.654,.829], arrays,
                              ['Merge','Nucleus','ER','Mitochondria']):
        fig.text(x+width/2,.719,label,ha='center',fontsize=11)
        add_image(fig,arr,[x,.256,width,height])
    fig.text(.647,.208,'Same field and spatial extent in all four views',
             ha='center',fontsize=9,color='#626d76')
    if draft:
        fig.text(.989,.065,'REVIEW DRAFT: Fig. 2E channel labels retained; CY5/TRITC annotation unresolved',
                 ha='right',fontsize=9,color='#855b24')
    fig.savefig(out/'panel_B.pdf')
    fig.savefig(out/'panel_B.svg')
    fig.savefig(out/'panel_B.png',dpi=600)
    fig.savefig(out/'panel_B.preview.png',dpi=180)
    plt.close(fig)
    return mito_source


def write_inspection(out, rows):
    report = '''# Project inspection and Panel B source assessment

## Existing project structure

- `Cell_Painting_Lec_P3_P9_figures/script/make_panel_A.py`: existing Panel A script.
- `Cell_Painting_Lec_P3_P9_figures/result/Panel_A/Panel_A.*`: completed Panel A; its source copy, individual image assets, legend and validation records are present.
- `cell_painting_LEC_P3_P9_expanded_20260916`: latest expanded study currently present in the workspace. Results are organized into data/QC, responses, states, embeddings, figures and validation folders.
- `Data/vivo/Flu P3` and `Flu P9`: 42 raw CSV exports each.
- Original ND2 folders: 30 files for each passage. Existing image-review records indicate three fields per ND2; exported measurements use the first field only.
- `Data/vivo`: corrected 16-bit channel TIFFs (0001–0003), four RGB display TIFFs (0002), and an existing overlay.

## Available quantitative material and units

The expanded study contains 4,866 cells across 60 wells and 12 conditions, five wells per condition. The primary four groups contain 1,626 cells across 20 wells: P3-Control 124, P3-H1N1 162, P9-Control 693, P9-H1N1 647 cells. Individual cells are not independent treatment replicates. Distinct donor identities are not documented; the prior analysis treated wells as different replicates at the user's request.

- `results/01_data_qc/cells_all_features.csv.gz`: 4,866 rows, 186 measurement columns plus metadata.
- `feature_decisions.csv`: 186 decisions, 139 eligible measurements, 97 retained features.
- `plate_condition_mapping.csv`, `well_qc.csv`, and `inputs/verified_cell_registry.csv`: passage, plasma, intervention, well identity and provenance.
- `results/02_responses/well_feature_summaries.csv.gz`: 60 wells × 139 features × three summaries (median, q10, q90).
- `well_response_metrics.csv`, `primary_response_tests.csv`, `compartment_response_tests.csv`, `feature_driver_tests.csv`: well-level distances/fractions, effect estimates, existing P values and BH-adjusted values.
- `results/03_states`: cell assignments, state profiles, well proportions and robustness results. Keep State 0/1/2 as exploratory numerical labels.
- `results/04_embeddings`: PCA coordinates/loadings/variance and 18 UMAP settings. Existing representations include exploratory groups; simply filtering plotted points would not constitute a primary-only fit.
- `results/05_figures`: 16 analysis figure stems, plus 12 image-review pages covering 48 selected cells. Those review crops were selected for technical review, not automatically as representative state cells.

## Panel support

| Panel | Existing sources | Remaining assembly/analysis requirement |
|---|---|---|
| A | `2E-F.pdf`, existing `Panel_A.*` | Already assembled; keep tube assay separate. Original plot artwork retained; no raw tube numeric table identified. |
| B | Fig. 2E portion of `2E-F.pdf`; matching `Corr…0002_final.tif` images; plate-condition mapping | Resolve CY5/TRITC label discrepancy. Use one field only, no numerical-result plot from Fig. 2E. |
| C | Complete feature table, PCA coordinates/loadings/variance | Primary four-group PCA and well-level displacement; document fit population. |
| D | Well feature summaries and predefined feature families | Explicit response difference: (P9 H1N1 − P9 Control) − (P3 H1N1 − P3 Control), using wells. Existing within-passage comparisons alone are insufficient. |
| E | State profiles, assignments and stability tables | Neutral state labels; document all-condition origin of existing state definitions. Select no mechanistic labels. Representative cells remain a separate provenance requirement. |
| F | Well responder fractions and state proportions | Show only four primary groups with individual wells. |
| G | Well/compartment response tables and existing perturbation contrasts | Exploratory H1N1 versus H1N1+VEGF-C or H1N1+UC2288 only, with explicit multiplicity context. |

## Panel B image assessment

The PDF contains four 512 × 512 JPEG images. The matching TIFF display images are 2048 × 2048, RGB, 8 bits per component. Corresponding corrected single-channel TIFFs are 16 bit. They share image dimensions, orientation and field content. The full field is retained; no crop, rotation, geometric registration, sharpening, nonlinear contrast adjustment or cell selection is performed.

| Existing Fig. 2E label | Matching TIFF | PDF/TIFF RGB correlation after display resizing |
|---|---|---|
'''
    for row in rows:
        report += f"| {row['fig2e_label']} | `{row['display_file']}` | {row['pdf_to_TIFF_RGB_correlation']:.6f} |\n"
    report += '''
These comparisons support the artwork-to-TIFF correspondence; compression/downsampling prevents claiming pixel identity between the PDF JPEG and original TIFF. Lossless extracted PNGs preserve the TIFF RGB pixels exactly.

The current Fig. 2E labels CY5/red as Mitochondria and TRITC/yellow as Plasma. The expanded analysis's `original_field_qc.csv` labels TRITC as mitochondria and CY5 as unmeasured. This discrepancy is unresolved. It is not corrected by changing numerical tables or silently renaming channels.

The TIFFs contain no verified micrometre-per-pixel calibration (unit/resolution fields are absent or unitless). An exact ND2/well link for this field is not established, so calibration is not borrowed from another image. No scale bar or passage/treatment assignment is invented. Equal image dimensions and layout preserve relative spatial scale.

The representative field was inherited from the supplied Fig. 2E, rather than selected using treatment effects. It illustrates channel appearance and does not establish a typical population phenotype. The existing `0002overlay_final.tif` is not reused: it does not reproduce the three displayed channels with matching component values. The new merge is an exact RGB combination of the same displayed nucleus, ER and chosen mitochondria planes. This is a composite of existing microscopy, not generated microscopy. The Plasma channel is omitted.

Proposed layout: left, six existing condition labels represented in both P3 and P9; right, equally sized Merge, Nucleus, ER and Mitochondria views. Fonts and neutral schematic styling match Panel A. The panel identifies Cell Painting as a separate discovery experiment.
'''
    (out/'panel_B.inspection.md').write_text(report,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only',action='store_true')
    parser.add_argument('--review-draft',action='store_true')
    parser.add_argument('--channel-labels',choices=['fig2e','expanded'])
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(DEFAULT_OUT.resolve()):
        raise ValueError('All generated outputs must remain inside result/.')
    out.mkdir(parents=True,exist_ok=True)
    if not args.inspect_only and not args.review_draft and not args.channel_labels:
        parser.error('Choose --inspect-only, --review-draft, or a confirmed --channel-labels mapping.')
    if not args.inspect_only and (out/'panel_B.pdf').exists():
        raise FileExistsError('Preserving existing Panel B. Choose a new --output-dir within result/.')
    sources = source_paths()
    before = {p:sha(p) for p in sources}
    panel_a = {p:sha(p) for p in panel_dir('A').glob('Panel_A*') if p.is_file()}
    images, rows = inspect(out)
    write_inspection(out,rows)
    if not args.inspect_only:
        mapping = args.channel_labels or 'fig2e'
        channel = render(out,images,mapping,args.review_draft)
        legend = (
            '# Panel B — Cell Painting design and representative images\n\n'
            'Left: the existing P3/P9 Cell Painting groups (Control, H1N1, VEGF-C, '
            'H1N1+VEGF-C, UC2288 and H1N1+UC2288). Each dot indicates that a group '
            'is present in that passage; dots are not replicate counts. This is a '
            'separate experiment from the tube-formation assay. Right: the existing '
            'Fig. 2E field shown as a three-channel merge and separate nucleus, ER '
            'and mitochondria views. Channel names are existing annotations.\n\n'
            f'Displayed mapping: DAPI → Nucleus; FITC → ER; {channel} → Mitochondria. '
            'The identical full 2048 × 2048 field is shown at the same spatial scale '
            'in all views. Existing display intensities are retained and the merge '
            'uses exactly the same scalar pixel values as the separate views. '
            'There is no autoscaling, gamma adjustment, sharpening or local editing. '
            'No outcome statistics or group-specific conclusions are shown. '
            'The field has no independently verified passage/well identity. '
            'No scale bar is shown because a physical calibration for this specific '
            'field could not be established.\n\n'
        )
        if mapping == 'expanded':
            legend += 'The supplied TRITC display is remapped from yellow to red without changing its scalar intensities; the separate view and merge use the same remapping.\n\n'
        if args.review_draft:
            legend += '**Review draft:** the CY5/TRITC annotation discrepancy between Fig. 2E and the expanded analysis is unresolved. Fig. 2E labels are retained provisionally. This draft is not a confirmed channel assignment or evidence of a mitochondrial phenotype.\n\n'
        legend += 'Source files, hashes, dimensions and artwork matches are documented in `panel_B.source_manifest.csv` and `panel_B.validation.json`; project inventory and panel mapping are in `panel_B.inspection.md`.\n'
        (out/'panel_B.legend.md').write_text(legend,encoding='utf-8')
    assert all(sha(p)==digest for p,digest in before.items()),'Source file changed'
    assert all(sha(p)==digest for p,digest in panel_a.items()),'Existing Panel A changed'
    checks = {'source_files_unchanged':True,'panel_A_unchanged':True,
              'display_TIFF_pixels_preserved_in_assets':True,
              'source_field_dimensions':[2048,2048], 'crop':'full field',
              'physical_scale_verified':False,'scale_bar_added':False,
              'representative_field_well_identity_verified':False,
              'statistics_computed':False,'biological_group_assignments_changed':False,
              'mode':'inspection' if args.inspect_only else ('review draft' if args.review_draft else 'confirmed channel mapping'),
              'channel_mapping':args.channel_labels or ('fig2e provisional' if args.review_draft else 'unresolved'),
              'source_sha256':{p.relative_to(WORKSPACE).as_posix():h for p,h in before.items()}}
    (out/'panel_B.validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in checks.items() if k!='source_sha256'},indent=2))


if __name__ == '__main__':
    main()
