"""Primary-only, label-blind preprocessing for Panel C.

Run from the workspace with its scientific Python environment:
  .venv/Scripts/python.exe -B Cell_Painting_Lec_P3_P9_figures/script/panel_C_preprocessing.py
Then run panel_C_pca.py. Original inputs and completed panels are read-only.
To reproduce without overwriting outputs, pass a new --output-dir under result/.
"""
from pathlib import Path
from figure_paths import panel_dir, resolve_result_path, assert_preserved
import argparse
import hashlib
import json
import os
import sys
sys.dont_write_bytecode = True
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(variable,'1')
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RESULT = ROOT/'result'
SOURCE = WORKSPACE/'cell_painting_LEC_P3_P9_expanded_20260916/results/01_data_qc'
CONFIG = {'missing_fraction_max':.01, 'near_constant_modal_fraction':.99,
          'transformed_sd_min':1e-12, 'correlation_abs_max':.90,
          'seed':20260918, 'fdr_threshold':.05}


def sha(path):
    with resolve_result_path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def dump(value,path):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def output_dir(value):
    out=Path(value).resolve()
    if not out.is_relative_to(RESULT.resolve()):
        raise ValueError('All outputs must be inside the project result folder.')
    out.mkdir(parents=True,exist_ok=True)
    return out


def weights(meta):
    count=meta.groupby('well_id').well_id.transform('size').to_numpy(float)
    w=1/count
    return w/w.sum()


def wq(x,w,q):
    order=np.argsort(x,kind='stable')
    cumulative=np.cumsum(w[order])
    k=min(np.searchsorted(cumulative,q*cumulative[-1]),len(order)-1)
    return float(x[order[k]])


def robust_scale(x,w):
    center=wq(x,w,.5)
    scale=1.4826*wq(np.abs(x-center),w,.5)
    method='1.4826 * weighted MAD'
    if scale<=CONFIG['transformed_sd_min']:
        scale=(wq(x,w,.75)-wq(x,w,.25))/1.349
        method='weighted IQR / 1.349 fallback'
    if scale<=CONFIG['transformed_sd_min']:
        scale=float(np.sqrt(np.average((x-np.average(x,weights=w))**2,weights=w)))
        method='weighted SD fallback'
    if scale<=CONFIG['transformed_sd_min']:
        raise ValueError('Negligible spread must be excluded before scaling.')
    return center,scale,method


def family(f):
    if f.startswith('ERWNuclei__AreaShape_'):return 'Cell shape (ER-derived mask)'
    if f.startswith('nuclei__AreaShape_'):return 'Nuclear shape'
    if f.startswith('ERWONuclei__Intensity_'):return 'ER intensity (source annotation)'
    if f.startswith('mitochondriaWONuclei__Intensity_'):return 'mitochondriaWONuclei intensity (source annotation)'
    if f.startswith('nuclei__Intensity_'):return 'Nuclear intensity'
    return 'Technical / position / metadata'


def priority(f):
    # Inherited fixed hierarchy favors direct geometric/intensity measurements
    # over higher-order descriptors. It does not use treatment effects or P values.
    preferred=['Area','Eccentricity','Compactness','Solidity','Extent','FormFactor',
               'Perimeter','MeanIntensity','MedianIntensity','StdIntensity',
               'MADIntensity','MaxIntensity','IntegratedIntensity']
    short=f.split('__',1)[1]
    order=next((i for i,s in enumerate(preferred)
                if short=='AreaShape_'+s or short.startswith('Intensity_'+s+'_')),50)
    return (100 if 'Zernike' in f else order,f)


def previous_panel_hashes():
    paths=[]
    for p in RESULT.rglob('*'):
        if p.is_file() and (p.name.startswith(('Panel_A','panel_B'))
                           or p.parent.name in ['assets','panel_B_assets','source_material']):
            paths.append(p)
    return {str(p.resolve()):sha(p) for p in sorted(set(paths))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=panel_dir('C'))
    args=parser.parse_args(); out=output_dir(args.output_dir)
    if (out/'panel_C_feature_QC.csv').exists() or (out/'panel_C_data').exists():
        raise FileExistsError('Preserving existing Panel C preprocessing; choose a new --output-dir.')
    data=out/'panel_C_data'; data.mkdir()
    old_panels=previous_panel_hashes()
    inputs=[SOURCE/name for name in ['cells_all_features.csv.gz','feature_decisions.csv',
                                     'feature_matrices.npz','well_qc.csv','plate_condition_mapping.csv']]
    source_hashes={str(p.resolve()):sha(p) for p in inputs}
    cells=pd.read_csv(inputs[0],keep_default_na=False,low_memory=False,float_precision='round_trip')
    prior=pd.read_csv(inputs[1],keep_default_na=False).set_index('feature')
    raw_features=list(prior.index)
    assert set(raw_features)=={c for c in cells if '__' in c}
    with np.load(inputs[2],allow_pickle=False) as existing:
        existing_shapes={k:list(existing[k].shape) for k in existing.files}
        assert len(cells)==len(existing['standardized'])
        assert np.isfinite(existing['standardized']).all()
    keep=(cells.passage.isin(['P3','P9']) & cells.plasma.isin(['Control','H1N1'])
          & cells.intervention.eq('None'))
    primary=cells.loc[keep].reset_index(drop=True)
    assert len(primary)==1626 and primary.well_id.nunique()==20
    assert primary.qc_valid.eq(True).all()
    assert not primary.cell_id.duplicated().any()
    assert not primary.duplicated(['passage','source_nd2_md5','ObjectNumber']).any()
    meta=primary[[c for c in primary if c not in raw_features]].copy()
    meta['panel_group']=meta.passage+'-'+meta.plasma
    group_wells=meta.groupby('panel_group').well_id.nunique()
    assert group_wells.eq(5).all() and len(group_wells)==4
    w=weights(meta)
    assert np.allclose(pd.Series(w).groupby(meta.well_id).sum(),1/20)
    rows=[]; transformed={}; numeric={}
    for f in primary:
        record={'original_feature':f,'family':family(f),'source_role':'measurement' if f in raw_features else 'metadata/QC',
                'retained':False,'status':'excluded','reason':'','n_nonfinite':0,
                'cell_missing_fraction':0.,'equal_well_missing_fraction':0.,
                'n_unique_finite':None,'modal_cell_fraction':None,'modal_equal_well_fraction':None,
                'transformation':'not applied','transform_divisor':None,'imputation_value':None,
                'imputed_cell_count':0,'center':None,'scale':None,'scale_method':'not applied',
                'transformed_weighted_SD':None,'redundant_with':'','redundant_correlation':None,
                'prior_expanded_retained':bool(prior.loc[f,'retained']) if f in prior.index else False,
                'prior_expanded_reason':str(prior.loc[f,'reason']) if f in prior.index else 'metadata, not a feature'}
        if f not in raw_features:
            record['reason']='identifier, filename, physical/source metadata, or QC flag; excluded from numerical feature space'
            rows.append(record); continue
        x=pd.to_numeric(primary[f],errors='coerce').to_numpy(float)
        finite=np.isfinite(x)
        record.update(n_nonfinite=int((~finite).sum()),cell_missing_fraction=float((~finite).mean()),
                      equal_well_missing_fraction=float(w[~finite].sum()))
        if finite.any():
            values,inverse,counts=np.unique(x[finite],return_inverse=True,return_counts=True)
            record.update(n_unique_finite=len(values),modal_cell_fraction=float(counts.max()/finite.sum()),
                          modal_equal_well_fraction=float(np.bincount(inverse,weights=w[finite]).max()/w[finite].sum()))
        if '__Location_' in f or any(s in f for s in ['BoundingBoxMaximum_','BoundingBoxMinimum_','AreaShape_Center_']):
            record['reason']='absolute position / image coordinates; retained only as source metadata'
        elif f.endswith('Orientation'):
            record['reason']='absolute orientation; no rotation-invariant interpretation specified'
        elif not finite.any():
            record['reason']='invalid: no finite numeric values'
        elif max(record['cell_missing_fraction'],record['equal_well_missing_fraction'])>CONFIG['missing_fraction_max']:
            record['reason']='more than 1% missing/nonfinite by cell count or equal-well weight'
        elif record['n_unique_finite']<=1:
            record['reason']='zero variance'
        elif max(record['modal_cell_fraction'],record['modal_equal_well_fraction'])>=CONFIG['near_constant_modal_fraction']:
            record['reason']='near-zero variance: at least 99% identical by cell count or equal-well weight'
        else:
            x=x.copy()
            if not finite.all():
                replacement=wq(x[finite],w[finite],.5)
                x[~finite]=replacement
                record.update(imputation_value=replacement,imputed_cell_count=int((~finite).sum()))
            bounded=any(f.endswith(s) for s in ['Eccentricity','Solidity','Extent','FormFactor'])
            if (x>=0).all() and not bounded:
                positive=x>0
                divisor=wq(x[positive],w[positive],.5) if positive.any() else 1.
                tx=np.log1p(x/divisor)
                record.update(transformation='log1p(x / equal-well positive median)',transform_divisor=divisor)
            else:
                tx=x.copy(); record.update(transformation='linear',transform_divisor=1.)
            sd=float(np.sqrt(np.average((tx-np.average(tx,weights=w))**2,weights=w)))
            record['transformed_weighted_SD']=sd
            if not np.isfinite(tx).all():
                record['reason']='invalid: transformation produced nonfinite values'
            elif sd<=CONFIG['transformed_sd_min']:
                record['reason']='near-zero transformed variance: equal-well SD <= 1e-12'
            else:
                center,scale,method=robust_scale(tx,w)
                record.update(center=center,scale=scale,scale_method=method)
                transformed[f]=tx; numeric[f]=x
        rows.append(record)
    lookup={r['original_feature']:r for r in rows}
    candidates=sorted(transformed,key=priority)
    t=np.column_stack([transformed[f] for f in candidates])
    # Weighted Pearson correlation of average ranks, matching the inherited
    # redundancy criterion. Each well has equal total weight in the covariance.
    ranks=rankdata(t,axis=0)
    ranks-=np.average(ranks,axis=0,weights=w)
    covariance=(ranks*w[:,None]).T@ranks
    correlation=covariance/np.sqrt(np.outer(np.diag(covariance),np.diag(covariance)))
    retained=[]
    for j,f in enumerate(candidates):
        related=[k for k in retained if abs(correlation[j,k])>CONFIG['correlation_abs_max']]
        if related:
            k=max(related,key=lambda k:abs(correlation[j,k]))
            lookup[f].update(reason='redundant: absolute equal-well weighted rank correlation > 0.90',
                             redundant_with=candidates[k],redundant_correlation=float(correlation[j,k]))
        else:
            retained.append(j)
            lookup[f].update(retained=True,status='retained',reason='retained after primary-only QC and redundancy filtering')
    features=[candidates[j] for j in retained]
    center=np.array([lookup[f]['center'] for f in features])
    scale=np.array([lookup[f]['scale'] for f in features])
    z=(t[:,retained]-center)/scale
    assert np.isfinite(z).all()
    assert not primary[raw_features].duplicated().any()
    meta['qc_extreme_primary_z8']=(np.abs(z)>8).any(axis=1)
    meta['equal_well_cell_weight']=w
    primary.to_csv(data/'primary_cells_original_measurements.csv.gz',index=False)
    meta.to_csv(data/'cell_metadata.csv',index=False)
    normalized=pd.concat([meta[['cell_id','well_id','panel_group']],pd.DataFrame(z,columns=features)],axis=1)
    normalized.to_csv(data/'normalized_features.csv.gz',index=False)
    np.savez_compressed(data/'normalized_features.npz',standardized=z,transformed=t[:,retained],
                        features=np.array(features),cell_ids=meta.cell_id.to_numpy(str),
                        weights=w,centers=center,scales=scale)
    qc=pd.DataFrame(rows)
    assert len(qc)==len(cells.columns) and qc.original_feature.is_unique
    qc.to_csv(out/'panel_C_feature_QC.csv',index=False)
    group_summary=meta.groupby(['passage','plasma','panel_group'],sort=True).agg(
        cells=('cell_id','size'),wells=('well_id','nunique')).reset_index()
    group_summary.to_csv(data/'group_counts.csv',index=False)
    summary={'source_cells':len(cells),'source_wells':int(cells.well_id.nunique()),
             'existing_normalized_matrix_shapes':existing_shapes,
             'existing_normalization_reused':False,'reason_for_refitting':'Existing matrices were fitted on all 12 conditions; Panel C is restricted to four primary groups.',
             'primary_cells':len(meta),'primary_wells':int(meta.well_id.nunique()),
             'raw_measurements':len(raw_features),'metadata_columns_excluded':len(cells.columns)-len(raw_features),
             'eligible_before_redundancy':len(candidates),'retained_features':len(features),
             'feature_counts_by_family':qc.loc[qc.retained,'family'].value_counts().to_dict(),
             'QC_decision_counts':qc.reason.value_counts().to_dict(),
             'scale_methods':qc.loc[qc.retained,'scale_method'].value_counts().to_dict(),
             'imputed_values':int(qc.imputed_cell_count.sum()),'additional_cells_removed':0,
             'edge_cells_retained':int(meta.qc_edge_touching.sum()),
             'primary_extreme_z8_cells_retained':int(meta.qc_extreme_primary_z8.sum()),
             'primary_groups':group_summary.to_dict('records'),'config':CONFIG,
             'independence':'Wells treated as independent at user request; donor/pool identities not independently documented.',
             'channel_annotation':'CY5/TRITC biological annotations are unresolved between Fig. 2E and the expanded analysis. Numerical source features are retained unchanged; no organelle mechanism is inferred.'}
    dump(summary,out/'panel_C_preprocessing.json')
    source_rows=[{'source_file':p,'sha256':h,'role':'existing analysis input'} for p,h in source_hashes.items()]
    source_rows += [{'source_file':p,'sha256':h,'role':'preserved earlier panel'} for p,h in old_panels.items()]
    pd.DataFrame(source_rows).to_csv(data/'source_and_preservation_manifest.csv',index=False)
    assert all(sha(p)==h for p,h in source_hashes.items())
    assert all(sha(p)==h for p,h in old_panels.items())
    text=(f'# Panel C feature QC\n\nThe audit covers all {len(qc)} input columns: '
          f'{len(raw_features)} measurements and {len(cells.columns)-len(raw_features)} metadata/QC fields. '
          f'It retains {len(features)} features from {len(candidates)} eligible candidates. '
          'Every original column, decision, exclusion reason, transformation, fitted scale and redundancy partner is in `panel_C_feature_QC.csv`.\n\n'
          'Preprocessing is refitted once using only the four primary groups, with each of 20 wells contributing equal total weight. '
          'No feature decision uses H1N1-versus-Control effects, P values or visual separation. '
          'Technical fields are retained as metadata where needed, never as PCA variables.\n\n'
          'Fixed rules: >1% missing/nonfinite (cell fraction or equal-well fraction); zero variance; >=99% identical values; '
          'transformed weighted SD <=1e-12; and redundancy at absolute weighted rank correlation >0.90. '
          'The deterministic hierarchy favors named direct measurements before Zernike descriptors, with alphabetical tie-breaking.\n\n'
          'Unbounded nonnegative values use log1p(x / equal-well positive median); bounded shape ratios remain linear. '
          'All groups share the same weighted median center and 1.4826 weighted MAD scale, with IQR/1.349 then weighted SD fallbacks. '
          f'{int(qc.imputed_cell_count.sum())} values were imputed. No clipping, winsorization, group-specific scaling or additional cell removal was used.\n\n'
          'The original 12-condition normalized matrix exists but is not reused. The primary-only raw subset, normalized matrix, parameters and input hashes are saved in `panel_C_data/`.\n')
    (out/'panel_C_feature_QC.md').write_text(text,encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
