from core import *
import re

def priority(f):
    useful=['Area','Eccentricity','Compactness','Solidity','Extent','FormFactor','Perimeter',
            'MeanIntensity','MedianIntensity','StdIntensity','MADIntensity','MaxIntensity','IntegratedIntensity']
    short=f.split('__')[1]
    score=next((i for i,s in enumerate(useful) if short=='AreaShape_'+s or short.startswith('Intensity_'+s+'_')),50)
    return (100 if 'Zernike' in f else score, f)

def main():
    registry=pd.read_csv(ROOT/'inputs/verified_cell_registry.csv',keep_default_na=False,float_precision='round_trip')
    oldhash=pd.read_csv(ROOT/'inputs/verified_raw_export_hashes.csv')
    inventory=pd.read_csv(ROOT/'inputs/verified_image_inventory.csv',keep_default_na=False)
    parts=[]; sources=[]; checks=[]; images=[]
    objects=['ERWNuclei','ERWONuclei','mitochondriaWNuclei','mitochondriaWONuclei','nuclei']
    rawroot=(ROOT/CFG['raw_exports_root']).resolve()
    for (passage,folder),reg in registry.groupby(['passage','raw_export_folder'],sort=False):
        base=rawroot/f'Flu {passage}'/folder
        for path in sorted(base.glob('*.csv')):
            expected=oldhash.loc[oldhash.passage.eq(passage)&oldhash.folder.eq(folder)&oldhash.filename.eq(path.name),'sha256'].item()
            digest=sha(path);assert digest==expected,str(path)
            sources.append(dict(path=os.path.relpath(path,ROOT),bytes=path.stat().st_size,sha256=digest))
        image=pd.read_csv(base/'Image.csv',keep_default_na=False)
        img=image.set_index('ImageNumber')
        sub=reg[META].copy()
        for obj in objects:
            raw=pd.read_csv(base/(obj+'.csv'),float_precision='round_trip')
            key='ObjectNumber' if obj=='nuclei' else 'Parent_nuclei'
            assert not raw.duplicated(['ImageNumber',key]).any()
            cols=[c for c in raw if c.startswith(('AreaShape_','Intensity_','Location_'))]
            source=raw[['ImageNumber',key]+cols].rename(columns={key:'ObjectNumber',**{c:obj+'__'+c for c in cols}})
            sub=sub.merge(source,on=['ImageNumber','ObjectNumber'],how='left',validate='one_to_one')
            assert np.array_equal(raw.groupby('ImageNumber').size().sort_index(),image.set_index('ImageNumber')[f'Count_{obj}'].sort_index())
            checks.append(dict(passage=passage,folder=folder,object=obj,rows=len(raw),retained_rows=len(sub),parent_join_unique=True,image_counts_match=True))
        for well,group in reg.groupby('well_id'):
            r=group.iloc[0]; im=img.loc[r.ImageNumber]
            fn=next(c for c in image if c.startswith('FileName_'))
            assert im[fn]==r.source_nd2_filename
            coord=re.match(r'Well([A-P]\d{2})_',im[fn]).group(1)
            assert coord==r.physical_well
            assert (coord[0]==folder[-1]) if passage=='P3' else (int(coord[1:])==int(folder.split()[-1]))
            errors=sum(float(im[c])!=0 for c in image if c.startswith('ModuleError_'))
            assert errors==0
            images.append(dict(well_id=well,passage=passage,physical_well=coord,n_cells=len(group),
                               width=int(im[next(c for c in image if c.startswith('Width_'))]),height=int(im[next(c for c in image if c.startswith('Height_'))]),
                               series=int(im[next(c for c in image if c.startswith('Series_'))]),frame=int(im[next(c for c in image if c.startswith('Frame_'))]),module_errors=errors,
                               nucleus_threshold=im['Threshold_FinalThreshold_nuclei'],
                               ER_threshold=im['Threshold_FinalThreshold_ERWNuclei'],
                               mitochondria_threshold=im['Threshold_FinalThreshold_mitochondriaWNuclei']))
        parts.append(sub)
    cells=pd.concat(parts,ignore_index=True).set_index('cell_id').loc[registry.cell_id].reset_index()
    measurement=[c for c in cells if '__' in c]
    assert len(measurement)==186
    assert len(cells)==4866 and cells.well_id.nunique()==60
    assert not cells.duplicated(measurement).any()
    assert not cells.duplicated(['passage','source_nd2_md5','ObjectNumber']).any()
    # Explicitly verify the 13 legacy values against the new all-feature join.
    verification=[]
    for c in registry:
        if '__' not in c:continue
        prefix,measure=c.split('__')
        obj=('ERWNuclei' if measure.startswith('AreaShape') else 'ERWONuclei') if prefix=='ER_Cell' else ('nuclei' if prefix=='Nucleus' else 'mitochondriaWONuclei')
        source=obj+'__'+measure
        ok=np.isclose(cells[source],registry[c],rtol=1e-12,atol=1e-14)
        assert ok.all(),c
        verification.append(dict(legacy_feature=c,raw_feature=source,matching_rows=int(ok.sum())))
    wellimg=pd.DataFrame(images).set_index('well_id')
    edge=np.zeros(len(cells),bool)
    for obj in ['ERWNuclei','nuclei']:
        for axis,dimension in [('X','width'),('Y','height')]:
            edge |= cells[f'{obj}__AreaShape_BoundingBoxMinimum_{axis}'].le(0).to_numpy()
            edge |= cells[f'{obj}__AreaShape_BoundingBoxMaximum_{axis}'].ge(cells.well_id.map(wellimg[dimension])-1).to_numpy()
    cells['qc_edge_touching']=edge
    maxima=[c for c in measurement if '__Intensity_MaxIntensity_' in c]
    cells['qc_saturated_maximum']=cells[maxima].ge(.999).any(axis=1)
    w=weights(cells);rows=[]; transformed={}
    for f in measurement:
        x=cells[f].to_numpy(float);finite=np.isfinite(x)
        rec=dict(feature=f,family=family(f),n_missing=int((~finite).sum()),n_unique=int(pd.Series(x).nunique()),
                 max_identical_fraction=float(pd.Series(x).value_counts(normalize=True).max()),
                 transformation='',transform_divisor=1.,reason='',retained=False,redundant_with='',correlation=np.nan)
        if '__Location_' in f or any(t in f for t in ['BoundingBoxMaximum_','BoundingBoxMinimum_','AreaShape_Center_']):rec['reason']='absolute position; QC only'
        elif f.endswith('Orientation'):rec['reason']='absolute orientation; no rotation-invariant biological interpretation specified'
        elif 1-finite.mean()>CFG['missing_feature_fraction_max']:rec['reason']='more than 1% missing'
        elif rec['max_identical_fraction']>=CFG['near_constant_fraction']:rec['reason']='at least 99% identical'
        else:
            if not finite.all():x[~finite]=wq(x[finite],w[finite],.5)
            bounded=any(f.endswith(s) for s in ['Eccentricity','Solidity','Extent','FormFactor'])
            if (x>=0).all() and not bounded:
                pos=x>0;div=wq(x[pos],w[pos],.5) if pos.any() else 1
                tx=np.log1p(x/div);rec.update(transformation='log1p(x / positive weighted median)',transform_divisor=float(div))
            else:tx=x.copy();rec['transformation']='linear'
            c,s=robust_scale(tx[:,None],w)
            rec.update(center=float(c[0]),scale=float(s[0]))
            if np.ptp(tx)<1e-12:rec['reason']='zero spread'
            else:transformed[f]=tx
        rows.append(rec)
    candidates=sorted(transformed,key=priority)
    t=np.column_stack([transformed[f] for f in candidates]);center,scale=robust_scale(t,w);z=(t-center)/scale
    ranks=rankdata(t,axis=0);ranks-=np.average(ranks,axis=0,weights=w)
    cov=(ranks*w[:,None]).T@ranks
    corr=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
    keep=[];lookup={r['feature']:r for r in rows}
    for j,f in enumerate(candidates):
        related=[k for k in keep if abs(corr[j,k])>CFG['spearman_cutoff']]
        if related:
            k=max(related,key=lambda k:abs(corr[j,k]));lookup[f].update(reason='redundant weighted Spearman',redundant_with=candidates[k],correlation=float(corr[j,k]))
        else:keep.append(j);lookup[f].update(retained=True,reason='retained')
    features=[candidates[j] for j in keep]
    cells['qc_extreme_z8']=(abs(z[:,keep])>8).any(axis=1)
    cells['qc_valid']=np.isfinite(cells[measurement].to_numpy(float)).all(1)&cells['nuclei__AreaShape_Area'].gt(0)&cells['ERWNuclei__AreaShape_Area'].gt(0)
    assert cells.qc_valid.all(),'Invalid cells require explicit reconstruction'
    save(cells,'01_data_qc/cells_all_features.csv.gz')
    save(pd.DataFrame(rows),'01_data_qc/feature_decisions.csv')
    save(pd.DataFrame(sources),'01_data_qc/raw_input_manifest.csv')
    save(pd.DataFrame(checks),'01_data_qc/object_join_checks.csv')
    save(pd.DataFrame(verification),'01_data_qc/legacy_feature_verification.csv')
    save(pd.DataFrame(images),'01_data_qc/image_qc.csv')
    qc=cells.groupby(['well_id','passage','plasma','intervention','physical_well'],sort=False).agg(n_cells=('cell_id','size'),edge_fraction=('qc_edge_touching','mean'),saturation_fraction=('qc_saturated_maximum','mean'),extreme_fraction=('qc_extreme_z8','mean')).reset_index()
    save(qc,'01_data_qc/well_qc.csv')
    np.savez_compressed(ROOT/'results/01_data_qc/feature_matrices.npz',transformed=t[:,keep],standardized=z[:,keep],features=np.array(features),all_transformed=t,all_standardized=z,all_features=np.array(candidates))
    result=dict(cells=len(cells),wells=cells.well_id.nunique(),measurements=len(measurement),shape_intensity_named=sum('__AreaShape_' in f or '__Intensity_' in f for f in measurement),eligible_features=len(candidates),retained_features=len(features),features_by_family=pd.Series([family(f) for f in features]).value_counts().to_dict(),edge_cells=int(edge.sum()),saturation_cells=int(cells.qc_saturated_maximum.sum()),extreme_cells=int(cells.qc_extreme_z8.sum()),P3_cells=int(cells.passage.eq('P3').sum()),P9_cells=int(cells.passage.eq('P9').sum()),excluded_wrong_folder_rows=577,missing_planned_P9_wells=['G03','G04','G05','G06','G07','G08'],independence=CFG['well_independence'])
    dump(result,'01_data_qc/summary.json');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
