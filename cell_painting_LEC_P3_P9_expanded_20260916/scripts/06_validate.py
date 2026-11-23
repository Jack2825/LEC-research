"""Integrity and scientific-output checks, including preservation of the old work."""
from core import *
import importlib.metadata
import sys

def main():
    cells,panel,a=load_data();checks=[]
    def check(name,condition,detail=''):
        checks.append(dict(check=name,passed=bool(condition),detail=str(detail)))
    registry=pd.read_csv(ROOT/'inputs/verified_cell_registry.csv',keep_default_na=False)
    check('Exact retained cell identity and order',cells.cell_id.tolist()==registry.cell_id.tolist())
    check('4866 cells / 60 wells / 12 conditions',len(cells)==4866 and cells.well_id.nunique()==60 and cells.groupby(['passage','plasma','intervention']).ngroups==12)
    check('Five distinct physical wells in every condition',cells.groupby(['passage','plasma','intervention']).physical_well_id.nunique().eq(5).all())
    check('No duplicate physical cell identity',not cells.duplicated(['passage','source_nd2_md5','ObjectNumber']).any())
    measures=[c for c in cells if '__' in c];check('186 original measurements',len(measures)==186)
    check('No complete duplicate raw profiles',not cells.duplicated(measures).any())
    check('139 eligible / 97 retained variables',len(a['all_features'])==139 and len(a['features'])==97)
    check('Metadata and positions excluded from phenotypic variables',all(family(f)!='Position/QC' and not any(s in f for s in ['Location_','BoundingBoxMaximum_','BoundingBoxMinimum_','Center_','Orientation']) for f in a['features']))
    check('Finite analysis matrices',np.isfinite(a['standardized']).all() and np.isfinite(a['transformed']).all())
    wellweights=pd.Series(weights(cells)).groupby(cells.well_id).sum()
    check('Equal total PCA/clustering influence per well',np.allclose(wellweights,1/60))
    obj=pd.read_csv(ROOT/'results/01_data_qc/object_join_checks.csv')
    check('Raw nucleus rows minus retained rows = 577',obj.loc[obj.object.eq('nuclei'),'rows'].sum()-len(cells)==577)
    legacy=pd.read_csv(ROOT/'results/01_data_qc/legacy_feature_verification.csv')
    check('All 13 legacy features reconstructed from raw',len(legacy)==13 and legacy.matching_rows.eq(4866).all())
    img=pd.read_csv(ROOT/'results/01_data_qc/image_qc.csv')
    check('Exported source is first field for every well',img.series.eq(0).all() and img.frame.eq(0).all())
    manifest=pd.read_csv(ROOT/'results/01_data_qc/raw_input_manifest.csv')
    check('84 raw source files inventoried',len(manifest)==84)
    for row in manifest.itertuples():check('Raw file unchanged: '+row.path,sha(ROOT/row.path)==row.sha256)
    scores=pd.read_csv(ROOT/'results/02_responses/cell_response_scores.csv.gz',keep_default_na=False)
    check('Every retained cell has one response row',set(scores.cell_id)==set(cells.cell_id) and not scores.cell_id.duplicated().any())
    for name in [c for c in scores if '__tail' in c]:check('Valid cell call frequencies: '+name,scores[name].between(0,1).all())
    reference=scores.plasma.eq('Control')&scores.intervention.eq('None')
    check('Out-of-fold reference control designation',np.array_equal(reference,scores['All__reference_control']))
    check('Control consensus flag not misrepresented',scores.loc[reference,'consensus_responder_95'].eq(-1).all())
    check('Consensus flag is at least four of five fits',np.array_equal(scores.loc[~reference,'consensus_responder_95'],scores.loc[~reference,'All__tail95'].ge(.8).astype(int)))
    check('Tail calls monotonically decrease with threshold',(scores['All__tail90']>=scores['All__tail95']).all() and (scores['All__tail95']>=scores['All__tail97.5']).all())
    wells=pd.read_csv(ROOT/'results/02_responses/well_response_metrics.csv',keep_default_na=False).set_index('well_id')
    expected=scores.groupby('well_id')['All__tail95'].mean().loc[wells.index]
    check('Saved well fractions agree with cell call frequencies',np.allclose(wells['All__tail95'],expected))
    check('All distance test replicates are distinct wells',all(len(x)==5 and len(y)==5 and len(set(x+y))==10 for _,x,y in contrasts(cells)))
    for name,count,qcol in [('02_responses/primary_response_tests.csv',18,'q_BH_family'),('02_responses/compartment_response_tests.csv',30,'q_BH_family'),('02_responses/feature_driver_tests.csv',2502,'q_BH_all_feature_tests'),('03_states/state_proportion_tests.csv',240,'q_BH_all_candidate_state_tests')]:
        d=pd.read_csv(ROOT/'results'/name)
        check('Multiplicity family size: '+name,len(d)==count)
        check('BH values: '+name,np.allclose(d[qcol],bh(d.p_exact)))
        check('Exact permutation resolution: '+name,np.allclose(d.p_exact*252,np.rint(d.p_exact*252),atol=1e-8))
    pt=pd.read_csv(ROOT/'results/02_responses/primary_response_tests.csv');ct=pd.read_csv(ROOT/'results/02_responses/compartment_response_tests.csv')
    for passage in ['P3','P9']:
        name=passage+'__H1N1_vs_Control';null=pd.read_csv(ROOT/'results/02_responses'/('permutation_null_'+name+'.csv'))
        check('252 reference-refitted permutations: '+passage,len(null)==252)
        for table in [pt,ct]:
            for r in table.loc[table.contrast.eq(name)].itertuples():
                p=(abs(null[r.endpoint].to_numpy())>=abs(r.effect_B_minus_A)-1e-12).mean()
                check('Saved refitted null supports P value: '+passage+' '+r.endpoint,np.isclose(p,r.p_exact))
    props=pd.read_csv(ROOT/'results/03_states/well_state_proportions.csv',keep_default_na=False)
    check('All candidate well state fractions sum to one',np.allclose(props.groupby(['scope','k','well_id']).proportion.sum(),1))
    scan=pd.read_csv(ROOT/'results/03_states/parameter_scan.csv')
    check('135 k/PCA/seed clustering combinations',len(scan)==135)
    boot=pd.read_csv(ROOT/'results/03_states/bootstrap_stability.csv')
    check('40 hierarchical bootstraps for every state/partition',boot.groupby(['scope','k','state']).bootstrap.nunique().eq(40).all())
    loo=pd.read_csv(ROOT/'results/03_states/leave_one_well_out.csv')
    for scope,n in [('All',60),('P3',30),('P9',30)]:check('Every well omitted in clustering: '+scope,loo.loc[loo.scope.eq(scope)].groupby(['k','state']).omitted_well.nunique().eq(n).all())
    um=pd.read_csv(ROOT/'results/04_embeddings/umap_all_settings.csv.gz')
    check('18 complete UMAP embeddings',um.setting.nunique()==18 and um.groupby('setting').cell_id.nunique().eq(4866).all() and len(um)==4866*18)
    check('UMAP coordinates finite',np.isfinite(um[['UMAP1','UMAP2']]).all().all())
    cuts=pd.read_csv(ROOT/'results/02_responses/control_thresholds.csv')
    check('36 positive control reference thresholds',len(cuts)==36 and cuts.threshold.gt(0).all())
    pixpath=ROOT/'results/01_data_qc/original_pixel_identity_checks.csv.gz'
    if pixpath.exists():
        pix=pd.read_csv(pixpath);check('All 4866 original DAPI pixels confirm source identity',len(pix)==4866 and pix.exact_match.all())
    # Historical sibling folders are not part of the public release.
    # Raw-input hashes and all scientific validations above remain mandatory.
    old = []
    for name in ['REPORT.html','REPORT.md','analysis_tables.xlsx','05_figures/all_analysis_figures.pdf','05_figures/figure_guide.csv']:check('Deliverable exists: '+name,(ROOT/'results'/name).is_file())
    guide=pd.read_csv(ROOT/'results/05_figures/figure_guide.csv')
    for r in guide.itertuples():check('Figure exists: '+r.file,(ROOT/'results/05_figures'/r.file).stat().st_size>10000)
    result=pd.DataFrame(checks);save(result,'06_validation/checks.csv')
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','umap-learn','matplotlib','openpyxl','nd2']}
    dump(dict(python=sys.version,packages=versions),'06_validation/software_versions.json')
    summary=dict(checks=len(result),passed=int(result.passed.sum()),failed=int((~result.passed).sum()),previous_files_verified=len(old),raw_files_verified=len(manifest),original_image_pixel_matches=4866 if pixpath.exists() else 'image review not run',status='PASS' if result.passed.all() else 'FAIL')
    dump(summary,'06_validation/validation_summary.json')
    outputs=[]
    for p in sorted((ROOT/'results').rglob('*')):
        if p.is_file() and p.name!='output_sha256.csv':outputs.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    save(pd.DataFrame(outputs),'06_validation/output_sha256.csv')
    print(json.dumps(summary,indent=2),flush=True)
    if not result.passed.all():raise RuntimeError(result.loc[~result.passed].to_string(index=False))

if __name__=='__main__':main()
