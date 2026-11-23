"""Package existing provenance-filtered measurements and verify Table S8.

No clustering, feature fitting, statistical testing, image processing or group
reassignment is performed. Outputs are confined to this new dated folder.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import math
import numbers
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
OUT = ROOT / 'result'
AUDIT = OUT / 'verification'
sys.path.insert(0,str(WORK/'tools'))
from release_inputs import require_inputs
REV = ROOT / 'inputs/base'
TABLES = REV / 'result/supplementary/Tables'
RAWZIP = REV / 'result/supplementary/Data/Data_S10_original_Cell_Painting_database.zip'
EXP = WORK / 'cell_painting_LEC_P3_P9_expanded_20260916'
REGISTRY = EXP / 'inputs/verified_cell_registry.csv'
PROV = ROOT / 'inputs/duplicate_reconciliation'
S8NAME = 'Table_S8_Cell_Painting_figure_analysis_collection.xlsx'
S8ZIP = 'Table_S8_Cell_Painting_analysis_collection.zip'
D10NAME = 'Data_S10_Cell_Painting_measurements_duplicate_rows_removed.zip'
OBJECTS = ['ERWNuclei', 'ERWONuclei', 'mitochondriaWNuclei', 'mitochondriaWONuclei', 'nuclei']


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.read_bytes()).hexdigest()


def csvbytes(rows, fields=None):
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fieldnames=fields or list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode('utf-8-sig')


def records(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline='')))


def exact_records(data):
    """Return parsed records and their original byte segments, including newlines."""
    bom = b'\xef\xbb\xbf' if data.startswith(b'\xef\xbb\xbf') else b''
    txt = data.decode('utf-8-sig')
    lines = txt.splitlines(keepends=True)
    rd = csv.reader(io.StringIO(txt, newline=''))
    header = next(rd)
    at = rd.line_num
    prefix = bom + ''.join(lines[:at]).encode('utf-8')
    out = []
    for row in rd:
        end = rd.line_num
        block = ''.join(lines[at:end]).encode('utf-8')
        assert len(row) == len(header)
        out.append((dict(zip(header, row)), block))
        at = end
    assert prefix + b''.join(v for _,v in out) == data
    return header, prefix, out


def readframe(data, name='source.csv'):
    if name.endswith('.gz'):
        data = gzip.decompress(data)
    return pd.read_csv(io.BytesIO(data), keep_default_na=False, float_precision='round_trip')


def frames_match(a, b):
    if a.columns.tolist() != b.columns.tolist() or a.shape != b.shape:
        return False
    for av,bv in zip(a.to_numpy(object).flat, b.to_numpy(object).flat):
        if (av is None or av == '') and (bv is None or bv == ''):
            continue
        if isinstance(av, numbers.Number) and isinstance(bv, numbers.Number):
            if not math.isclose(float(av),float(bv),rel_tol=1e-12,abs_tol=1e-14):
                return False
        elif av != bv:
            return False
    return True


def table_frame(sheet):
    vals = list(sheet.iter_rows(values_only=True))
    return pd.DataFrame(vals[1:], columns=vals[0]).fillna('')


def source_path_arc(raw, sheet):
    raw = raw.replace('\\', '/')
    p = REV/raw if raw.startswith('result/') else WORK/raw
    arc = raw.split('/result/',1)[-1] if raw.startswith('Cell_Painting_') else (
        ('Tube_artwork/' if sheet == 'Tube_artwork_sources' else 'Representative_images/') + p.name)
    return p, arc


def verify_table(reg, exclusions):
    book = openpyxl.load_workbook(TABLES/S8NAME, read_only=True, data_only=True)
    index = table_frame(book['Source_Index'])
    external_index = pd.read_csv(TABLES/'Table_S8_source_index.csv', keep_default_na=False)
    assert frames_match(index, external_index)
    frames, audit = {}, []
    with zipfile.ZipFile(TABLES/S8ZIP) as z:
        assert z.testzip() is None
        for row in index.to_dict('records'):
            sheet = row['worksheet']
            path, arc = source_path_arc(row['original_file'], sheet)
            archived = z.read(arc)
            assert sha(archived) == row['sha256'], sheet
            expected = readframe(archived, arc)
            actual = table_frame(book[sheet])
            if sheet == 'Tube_artwork_sources':
                # This provenance sheet intentionally stores PDF page labels as text.
                expected['source_page'] = expected['source_page'].astype(str)
            assert expected.shape == (int(row['rows']),int(row['columns'])), sheet
            assert frames_match(expected, actual), 'Worksheet values mismatch: '+sheet
            current = path.read_bytes()
            same = sha(current) == row['sha256']
            note = 'Byte-identical current source'
            if not same:
                # Known image-QC source was sorted after packaging; compare by its keys.
                assert sheet == 'Image_display_QC', sheet
                live = readframe(current)
                a = expected.sort_values(['cell_id','channel']).reset_index(drop=True)
                b = live.sort_values(['cell_id','channel']).reset_index(drop=True)
                assert a.equals(b), 'Changed image QC content'
                note = 'Current CSV has different row order; all keyed values exactly identical. Archived hash matches index.'
            frames[sheet] = expected
            audit.append(dict(worksheet=sheet,rows=len(actual),columns=len(actual.columns),
                              archive_sha256_matches=True,worksheet_values_match=True,
                              live_source_sha256_matches=same,source_note=note))
            print('Table S8 verified:', sheet, flush=True)
    book.close()
    meta = frames['Cell_metadata_states']
    primary = frames['Primary_normalized_cells']
    assert len(meta)==4866 and meta.cell_id.is_unique
    assert set(meta.cell_id)==set(reg.cell_id)
    assert not set(meta.cell_id).intersection(exclusions.remove_cell_id)
    primary_reg = reg[(reg.intervention=='None') & reg.plasma.isin(['Control','H1N1'])]
    assert len(primary)==1626 and primary.cell_id.is_unique
    assert set(primary.cell_id)==set(primary_reg.cell_id)
    assert primary.well_id.nunique()==20 and len(primary.columns)-3==98
    for c in ['well_id','passage','plasma','intervention','physical_well','source_nd2_filename']:
        a = meta.set_index('cell_id')[c].sort_index()
        b = reg.set_index('cell_id')[c].sort_index()
        assert a.equals(b), c
    well = frames['Well_map_and_summaries'].set_index('well_id')
    assert len(well)==60 and well.index.is_unique
    counts = meta.groupby('well_id').size()
    assert np.array_equal(well.loc[counts.index,'n_cells'],counts)
    for state in range(3):
        cnt=meta[meta.state==state].groupby('well_id').size().reindex(well.index,fill_value=0)
        assert np.array_equal(well[f'state_{state}_cells'],cnt)
        assert np.allclose(well[f'state_{state}_fraction'],cnt/well.n_cells,rtol=1e-12,atol=1e-14)
    pw = frames['Primary_state_well_fractions'].set_index('well_id')
    for c in ['n_cells']+[f'state_{s}_{v}' for s in range(3) for v in ['cells','fraction']]:
        assert np.allclose(pw[c],well.loc[pw.index,c],rtol=1e-12,atol=1e-14)
    gp=frames['Group_counts']
    for r in gp.itertuples():
        m=meta[(meta.passage==r.passage)&(meta.plasma==r.plasma)&(meta.intervention==r.intervention)]
        assert len(m)==r.cells and m.well_id.nunique()==r.wells==5
    primstates = meta[meta.cell_id.isin(primary.cell_id)]
    assert primstates.state.value_counts().to_dict()=={1:1325,2:273,0:28}
    for r in frames['Representative_images'].itertuples():
        assert r.cell_id in set(primary.cell_id)
        assert int(meta.set_index('cell_id').loc[r.cell_id,'state'])==r.state
    stats = frames['Feature_response_statistics']
    assert np.allclose(stats.interaction_P9_minus_P3,stats.delta_P9-stats.delta_P3,rtol=1e-12,atol=1e-14)
    for label,df,cols in [('features',stats,['p_interaction','q_BH']),
                         ('exploratory',frames['Exploratory_statistics'],['p_permutation','q_BH'])]:
        for c in cols:
            x=pd.to_numeric(df[c],errors='coerce').dropna()
            assert x.between(0,1).all(),label+' '+c
    (AUDIT/'Table_S8_worksheet_checks.csv').write_bytes(csvbytes(audit))
    return frames, audit


def data_s10(reg, exclusion):
    # Map the historical sheet names using the verified registry, never infer from effects.
    folder_map = reg[['passage','source_sheet','raw_export_folder']].drop_duplicates()
    assert not folder_map.duplicated(['passage','source_sheet']).any()
    lookup = {(r.passage,r.source_sheet):r.raw_export_folder for r in folder_map.itertuples()}
    regkeys = {(r.passage,r.raw_export_folder,int(r.ImageNumber),int(r.ObjectNumber)) for r in reg.itertuples()}
    removekeys = {('P9',lookup[('P9',r.remove_sheet)],int(r.remove_ImageNumber),int(r.ObjectNumber)) for r in exclusion.itertuples()}
    assert len(regkeys)==4866 and len(removekeys)==577 and not regkeys.intersection(removekeys)
    objects, images, exported, manifest = {}, {}, {}, []
    with zipfile.ZipFile(RAWZIP) as z:
        assert z.testzip() is None
        for n in z.namelist():
            if not n.startswith('raw_exports/') or not n.endswith('.csv'):
                continue
            _,passpart,folder,filename=n.split('/')
            passage=passpart.split()[-1]
            data=z.read(n)
            original=WORK/'Data/vivo'/passpart/folder/filename
            assert sha(data)==sha(original),n
            header,prefix,rows=exact_records(data)
            obj=Path(filename).stem
            kept=[]
            for row,block in rows:
                if obj in OBJECTS:
                    key='ObjectNumber' if obj=='nuclei' else 'Parent_nuclei'
                    k=(passage,folder,int(row['ImageNumber']),int(row[key]))
                    assert (obj,k) not in objects
                    objects[(obj,k)]=row
                    assert k in regkeys or k in removekeys,('Unclassified record',obj,k)
                    if k in regkeys:kept.append((row,block))
                elif obj=='Image':
                    ik=(passage,folder,int(row['ImageNumber']))
                    images[ik]=row
                    if any(k[:3]==ik for k in regkeys):kept.append((row,block))
                else:
                    assert obj=='Experiment'
                    kept.append((row,block))
            out=prefix+b''.join(block for _,block in kept)
            arc='corrected_exports/'+passpart+'/'+folder+'/'+filename
            exported[arc]=out
            manifest.append(dict(passage=passage,folder=folder,file=filename,archive_path=arc,
                original_archive_path=n,original_rows=len(rows),retained_rows=len(kept),removed_rows=len(rows)-len(kept),
                columns=len(header),original_sha256=sha(data),corrected_sha256=sha(out),
                retained_row_bytes_unchanged=True,original_file_unchanged=True))
    assert len(manifest)==84
    assert sum(r['removed_rows'] for r in manifest if r['file']=='nuclei.csv')==577
    assert sum(r['retained_rows'] for r in manifest if r['file']=='nuclei.csv')==4866
    assert sum(r['removed_rows'] for r in manifest if r['file']=='Image.csv')==5
    # The identical cell is retained at its verified source folder; check all 186 exported measurements.
    pairs=[]
    for r in exclusion.itertuples():
        rk=('P9',lookup[('P9',r.remove_sheet)],int(r.remove_ImageNumber),int(r.ObjectNumber))
        kk=('P9',lookup[('P9',r.keep_sheet)],int(r.keep_ImageNumber),int(r.ObjectNumber))
        assert kk in regkeys
        assert reg.set_index('cell_id').loc[r.keep_cell_id,'physical_well']==r.physical_well
        count=0
        for obj in OBJECTS:
            a,b=objects[(obj,rk)],objects[(obj,kk)]
            features=[c for c in a if c.startswith(('AreaShape_','Intensity_','Location_'))]
            assert all(a[c]==b[c] for c in features),('Nonidentical measurement pair',r.remove_cell_id,obj)
            count+=len(features)
        assert count==186
        pairs.append(dict(remove_cell_id=r.remove_cell_id,remove_folder=rk[1],remove_ImageNumber=rk[2],
                          keep_cell_id=r.keep_cell_id,keep_folder=kk[1],keep_ImageNumber=kk[2],
                          ObjectNumber=int(r.ObjectNumber),physical_well=r.physical_well,
                          verified_identical_measurements=count,exclusion_reason='Previously verified duplicate in wrong source folder'))
    # Validate retained Image.csv counts without altering historical processing fields.
    for r in reg[['passage','raw_export_folder','ImageNumber','physical_well','source_nd2_filename']].drop_duplicates().itertuples():
        ik=(r.passage,r.raw_export_folder,int(r.ImageNumber))
        im=images[ik]
        filenames=[v for c,v in im.items() if c.startswith('FileName_')]
        assert filenames==[r.source_nd2_filename]
        assert r.source_nd2_filename.startswith('Well'+r.physical_well+'_')
        assert r.physical_well[0]==r.raw_export_folder[-1] if r.passage=='P3' else int(r.physical_well[1:])==int(r.raw_export_folder.split()[-1])
        n=sum(k[:3]==ik for k in regkeys)
        assert all(int(float(im['Count_'+obj]))==n for obj in OBJECTS)
    # A convenient one-cell-per-row table, untransformed; source feature tokens are preserved.
    metadata=['cell_id','well_id','passage','plasma','intervention','source_sheet','raw_export_folder',
              'ImageNumber','ObjectNumber','physical_well','source_nd2_filename','source_nd2_md5']
    merged=[]
    for r in reg.to_dict('records'):
        k=(r['passage'],r['raw_export_folder'],int(r['ImageNumber']),int(r['ObjectNumber']))
        row={c:r[c] for c in metadata}
        for obj in OBJECTS:
            raw=objects[(obj,k)]
            row.update({obj+'__'+c:v for c,v in raw.items() if c.startswith(('AreaShape_','Intensity_','Location_'))})
        merged.append(row)
    matrix=pd.DataFrame(merged)
    features=[c for c in matrix if '__' in c]
    assert len(features)==186
    existing=pd.read_csv(EXP/'results/01_data_qc/cells_all_features.csv.gz',keep_default_na=False,float_precision='round_trip').set_index('cell_id')
    numeric=matrix[features].astype(float).to_numpy()
    expected=existing.loc[matrix.cell_id,features].to_numpy(float)
    assert np.array_equal(numeric,expected),'Retained measurements do not exactly match existing analysis input'
    assert not matrix.duplicated(features).any()
    assert not matrix.duplicated(['passage','source_nd2_md5','ObjectNumber']).any()
    counts=reg.groupby(['passage','plasma','intervention'],sort=False).agg(cells=('cell_id','size'),wells=('well_id','nunique')).reset_index()
    assert counts.wells.eq(5).all()
    wellmap=reg.groupby(['passage','plasma','intervention','well_id','physical_well','raw_export_folder','ImageNumber','source_nd2_filename'],sort=False).size().rename('cells').reset_index()
    manifest_bytes=csvbytes(manifest)
    (AUDIT/'Data_S10_file_manifest.csv').write_bytes(manifest_bytes)
    (AUDIT/'Data_S10_exclusion_verification.csv').write_bytes(csvbytes(pairs))
    summary=dict(original_cell_records=5443,excluded_wrong_folder_cell_records=577,retained_cells=4866,
                 retained_P3_cells=912,retained_P9_cells=3954,retained_wells=60,primary_cells=1626,primary_wells=20,
                 measurement_columns=186,identical_complete_profiles_remaining=0,
                 excluded_image_entries=5,retained_image_entries=60,
                 raw_measurement_values_transformed=False,retained_row_bytes_unchanged=True,
                 retained_measurements_exactly_match_existing_expanded_analysis=True,
                 excluded_pairs_identical_across_all_186_measurements=True,
                 unchanged_export_files=sum(r['original_sha256']==r['corrected_sha256'] for r in manifest),
                 filtered_export_files=sum(r['original_sha256']!=r['corrected_sha256'] for r in manifest))
    readme=f'''Data S10 — provenance-filtered original-scale Cell Painting measurements
Prepared 8 October 2026 from the existing verified exclusion decisions.

This is the requested corrected measurement release, not the untouched original
archive. Exactly 577 previously identified wrong-folder duplicate cell records
were excluded. Their valid counterparts were retained. No new filtering rule,
normalization, clustering, statistical test or group reassignment was applied.

RETAINED: 4,866 cells (P3 912; P9 3,954), 60 wells, five wells per condition.
Primary subset: 1,626 cells, 20 wells. Joined measurements: 186 original-scale
columns, before feature QC; 98 and 97 are downstream analysis feature counts.

corrected_exports/: the original five object tables and Image/Experiment tables
per export folder, with only previously excluded records removed. Retained row
bytes, feature names, values, coordinates, ObjectNumber and ImageNumber remain
unchanged. There can be gaps in ImageNumber. These IDs are local to the source
folder, not globally unique. Image.csv loses the five wrong-folder image entries;
Experiment.csv and historical processing fields (e.g. Group_Length) are unchanged.
There are 577 exclusions in each of five related object tables, representing
577 cells, NOT 2,885 independent cells. No other cells were removed.

cells_measurements_untransformed.csv.gz: one cell per row, keyed/metadata-labelled
join of 186 exported measurements. Values are unchanged and match the existing
expanded analysis input exactly. Location features are included as raw source
measurements; they are not all analysis features.

metadata/: existing group assignments, retained registry and well map.
provenance/: record-by-record removal/retention mapping and exact original versus
corrected file hashes. Removed measurement rows are NOT included in this release;
the audit log keeps identifiers/evidence so exclusions remain traceable.
Data_S10_database_index.xlsx: corrected-file catalogue and count/exclusion summary.

The original unfiltered Data_S10_original_Cell_Painting_database.zip is preserved
unchanged at its existing project location (SHA-256 {sha(RAWZIP)}).
ND2 microscopy is not included. This packaging does not resolve acquisition
channel annotation, segmentation or donor-independence limitations.
The matching Table S8 contains the existing processed figure-analysis outputs.
'''
    book=openpyxl.Workbook()
    book.remove(book.active)
    for title,rows in [('README',[{'item':k,'value':str(v)} for k,v in summary.items()]),
                       ('Corrected_export_index',manifest),('Existing_group_counts',counts.to_dict('records')),
                       ('Retained_well_map',wellmap.to_dict('records')),('Excluded_record_map',pairs)]:
        s=book.create_sheet(title);s.append(list(rows[0]))
        for row in rows:s.append(list(row.values()))
        s.freeze_panes='A2';s.auto_filter.ref=s.dimensions
        for cell in s[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='284B63')
    mem=io.BytesIO();book.save(mem)
    with zipfile.ZipFile(OUT/D10NAME,'w',zipfile.ZIP_DEFLATED) as z:
        for path,data in exported.items():z.writestr(path,data)
        z.writestr('README.txt',readme)
        z.writestr('Data_S10_database_index.xlsx',mem.getvalue())
        z.writestr('cells_measurements_untransformed.csv.gz',gzip.compress(csvbytes(merged),mtime=0))
        z.writestr('metadata/retained_cell_registry.csv',REGISTRY.read_bytes())
        z.writestr('metadata/condition_counts.csv',csvbytes(counts.to_dict('records')))
        z.writestr('metadata/retained_well_map.csv',csvbytes(wellmap.to_dict('records')))
        z.writestr('provenance/original_and_corrected_file_manifest.csv',manifest_bytes)
        z.writestr('provenance/excluded_duplicate_record_map.csv',csvbytes(pairs))
        z.writestr('provenance/original_exclusion_evidence.csv',(PROV/'cell_row_provenance.csv').read_bytes())
        z.writestr('provenance/original_deletion_plan_by_well.csv',(PROV/'deletion_plan_by_well.csv').read_bytes())
        z.writestr('provenance/correction_summary.json',json.dumps(summary,indent=2))
    with zipfile.ZipFile(OUT/D10NAME) as z:
        assert z.testzip() is None
        for r in manifest:assert sha(z.read(r['archive_path']))==r['corrected_sha256']
    return summary


def copy_table_with_documentation():
    # Only two explanatory README strings change; every scientific worksheet XML
    # and all remaining XLSX package parts are preserved byte for byte.
    changes={
        'Original CellProfiler CSV exports are packaged separately in Data S10.':
            'Data_S10_Cell_Painting_measurements_duplicate_rows_removed.zip contains original-scale CellProfiler exports after exclusion of 577 previously verified wrong-folder duplicate cell records. All retained measurement values are unchanged; 4,866 cells remain. The untouched original archive is preserved separately.',
        'Original CSV/GZIP/NPZ bytes are in the companion analysis-collection ZIP; Excel uses its normal numeric precision.':
            'Original CSV/GZIP/NPZ bytes are preserved in supporting/Table_S8_Cell_Painting_analysis_collection.zip; Excel uses its normal numeric precision. The Data S10 exclusion list matches the existing retained cells in this workbook. No analysis was rerun.'}
    with zipfile.ZipFile(TABLES/S8NAME) as src,zipfile.ZipFile(OUT/S8NAME,'w',zipfile.ZIP_DEFLATED) as dest:
        for item in src.infolist():
            data=src.read(item.filename)
            if item.filename=='xl/worksheets/sheet1.xml':
                for old,new in changes.items():
                    assert data.count(old.encode())==1
                    data=data.replace(old.encode(),new.encode())
                ET.fromstring(data)
            dest.writestr(item,data)
    with zipfile.ZipFile(TABLES/S8NAME) as a,zipfile.ZipFile(OUT/S8NAME) as b:
        assert b.testzip() is None
        assert a.namelist()==b.namelist()
        for n in a.namelist():
            if n!='xl/worksheets/sheet1.xml':assert a.read(n)==b.read(n),n
    supporting=OUT/'supporting';supporting.mkdir(exist_ok=True)
    shutil.copy2(TABLES/S8ZIP,supporting/S8ZIP)
    shutil.copy2(TABLES/'Table_S8_source_index.csv',supporting/'Table_S8_source_index.csv')
    assert sha(TABLES/S8ZIP)==sha(supporting/S8ZIP)
    (AUDIT/'Table_S8_documentation_changes.json').write_text(json.dumps(changes,indent=2),encoding='utf-8')


def main():
    require_inputs('tables')
    OUT.mkdir(parents=True,exist_ok=True);AUDIT.mkdir(exist_ok=True)
    originals=[RAWZIP,REGISTRY,PROV/'cell_row_provenance.csv',TABLES/S8NAME,TABLES/S8ZIP,TABLES/'Table_S8_source_index.csv']
    before={str(p):sha(p) for p in originals}
    reg=pd.read_csv(REGISTRY,keep_default_na=False,float_precision='round_trip')
    exc=pd.read_csv(PROV/'cell_row_provenance.csv',keep_default_na=False)
    frames,tableaudit=verify_table(reg,exc)
    summary=data_s10(reg,exc)
    copy_table_with_documentation()
    assert before=={str(p):sha(p) for p in originals}
    validation=dict(date='2026-10-08',data_s10=summary,Table_S8_data_sheets_verified=len(tableaudit),
                    Table_S8_total_sheets=37,Table_S8_all_scientific_worksheet_XML_unchanged=True,
                    Table_S8_primary_cell_ids_match_corrected_registry=True,
                    Table_S8_all_cell_ids_match_corrected_registry=True,
                    Table_S8_wrong_folder_cell_ids_present=0,
                    source_files_unchanged=True,original_file_hashes=before,
                    data_s10_sha256=sha(OUT/D10NAME),table_s8_sha256=sha(OUT/S8NAME),
                    analysis_rerun=False,measurement_values_changed=False)
    (AUDIT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    note=f'''# Verified Data S10 and Table S8 — 8 October 2026

Use these two main files:

- [Data S10: measurements with wrong-folder duplicate records removed]({D10NAME})
- [Table S8: verified figure-analysis workbook]({S8NAME})

Data S10 now excludes exactly the previously identified 577 wrong-folder cell
records and retains their correct counterparts. The remaining 4,866 cells across
60 wells include 1,626 primary-group cells across 20 wells. All retained original
measurement row bytes are unchanged. No normalization or new cell/feature QC was
applied. This is a provenance-filtered measurement release, not the untouched raw
archive. The earlier original and combined-index ZIPs still contain the excluded
records; do not substitute those for this corrected release.

Table S8 passed comparison of all 35 data/source worksheets against its archived
CSV sources and their recorded checksums, with normal Excel floating-point
rounding allowed (relative 1e-12, absolute 1e-14). The 4,866 all-condition and
1,626 primary cell IDs match the retained registry; none of the 577 excluded IDs
is present. Counts, well/state proportions, metadata, response-interaction
arithmetic and representative-cell states were checked. Its scientific worksheet
XML is unchanged. Only two README descriptions now name and explain the matching
corrected Data S10 and the location of the exact-precision companion ZIP.

One live image-display QC CSV was reordered after packaging. Its values are
exactly identical after matching cell_id and channel; the archived bytes and
workbook remain consistent with the original source index. No numerical repair
was necessary.

[The unchanged Table S8 companion ZIP](supporting/{S8ZIP}) preserves exact-precision
CSV/GZIP/NPZ data and image-provenance records. Keep it with the workbook when
sharing the complete collection. Historical README files inside that archive
describe the original packaging; this top-level README describes this release.

Verification is of existing data consistency, not clearance of every manuscript
issue. CY5/TRITC compartment annotation remains unresolved. The tube-assay source
is artwork only; Table S8 does not provide numerical tube measurements or missing
test/replicate details. Table S8 correctly documents these limitations. No claim
of segmentation validation or verified plasma-donor independence is added.

Suggested replacement Data S10 description for Supplemental-CVR-v3 (not edited):

> Data S10. Cell Painting CellProfiler measurement database after removal of
> provenance-confirmed wrong-folder duplicate records
> ({D10NAME}). The archive contains original-scale measurements for 4,866 cells
> across 60 wells after exclusion of 577 duplicate/misassigned cell records;
> retained measurement values are unchanged. A one-cell-per-row measurement
> matrix, source index, file-integrity manifest and exclusion/retention audit trail
> are included. Processed figure-analysis outputs are supplied in Table S8.

The methods sentence saying “raw exports in Data S10 remain unchanged” also
needs to distinguish unchanged retained VALUES from removal of the 577 RECORDS.
The untouched original archive remains preserved separately in the project.
No Word document has been changed in this task.

[Verification details](verification/validation.json) ·
[Worksheet checks](verification/Table_S8_worksheet_checks.csv) ·
[File/exclusion counts](verification/Data_S10_file_manifest.csv)
'''
    (OUT/'README.md').write_text(note,encoding='utf-8')
    (OUT/'PACKAGE_NOTES.md').write_text('# Data S10 and Table S8\n\n[Open the verified release](result/README.md).\n\nScripts are in `script/`; deliverables and verification records are in `result/`. Earlier project files are untouched.\n',encoding='utf-8')
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':main()
