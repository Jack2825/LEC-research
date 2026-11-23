"""Prepare current artwork from declared snapshots; never copy historical code."""
from pathlib import Path
import csv, hashlib, json, sys
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parent
sys.path.insert(0,str(ROOT/'tools'))
from release_inputs import require_inputs
OUT=BASE/'result'
SOURCE=BASE/'inputs'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare_calibration():
    """Reproduce the four existing crop extents from original review coordinates."""
    import pandas as pd
    import numpy as np
    import nd2
    expanded=ROOT/'cell_painting_LEC_P3_P9_expanded_20260916/results/01_data_qc'
    registry=pd.read_csv(expanded/'image_review_cell_registry.csv',keep_default_na=False)
    registry=registry.loc[registry.passage.eq('P3')&registry.plasma.eq('Control')&registry.intervention.eq('None')]
    cells=pd.read_csv(expanded/'cells_all_features.csv.gz',usecols=['cell_id','nuclei__Location_Center_X','nuclei__Location_Center_Y']).set_index('cell_id')
    iq=pd.read_csv(expanded/'image_qc.csv').set_index('well_id')
    records=[]
    for filename,g in registry.groupby('source_nd2_filename',sort=False):
        paths=list((ROOT/'Data/vivo/2025.2.11 Mice FLU LE cellpainting P3').rglob(filename));assert len(paths)==1
        path=paths[0]
        with nd2.ND2File(path) as d:
            um=d.voxel_size().x
            assert d.sizes['X']==2048 and d.sizes['Y']==2048
        for row in g.to_dict('records'):
            x,y=np.rint(cells.loc[row['cell_id']].to_numpy()).astype(int)
            x0,x1=max(0,x-160),min(2048,x+160);y0,y1=max(0,y-160),min(2048,y+160)
            row.update(source_nd2=path.relative_to(ROOT).as_posix(),source_sha256=sha(path),frame=int(iq.loc[row['well_id'],'series']),
                 um_per_pixel=um,crop_x0=x0,crop_y0=y0,crop_width=x1-x0,crop_height=y1-y0,target_x=x-x0,target_y=y-y0)
            records.append(row)
    dest=OUT/'calibration';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(dest/'S7A_calibrated_crop_records.csv',index=False)
    return expanded


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-calibration',action='store_true',help='Recompute the existing crop calibration from original ND2 and saved review coordinates.')
    args=parser.parse_args()
    require_inputs('formatting')
    if args.refresh_calibration:prepare_calibration()
    OUT.mkdir(parents=True,exist_ok=True)
    print('Frozen formatting inputs verified. Historical scripts are not required.')
if __name__=='__main__':main()
