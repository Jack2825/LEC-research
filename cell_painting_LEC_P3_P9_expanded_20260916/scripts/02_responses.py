"""Control-referenced single-cell distances; tests use independent well labels."""
from core import *
from scipy.linalg import inv

def reference_scores(z,meta,refs,shrink=.2,diagonal=False):
    """Leave-one-reference-well-out fit; no control cell scores itself."""
    z=np.asarray(z,float); ids=meta.well_id.to_numpy(); refs=list(refs)
    means=[];seconds=[]
    for well in refs:
        a=z[ids==well]; assert len(a)>1
        means.append(a.mean(0)); seconds.append(a.T@a/len(a))
    means=np.array(means);seconds=np.array(seconds);dist=[]
    for fold in range(len(refs)):
        other=np.arange(len(refs))!=fold
        mu=means[other].mean(0);cov=seconds[other].mean(0)-np.outer(mu,mu)
        floor=max(float(np.trace(cov)/len(mu))*1e-8,1e-10)
        if diagonal: cov=np.diag(np.diag(cov)+floor)
        else:cov=(1-shrink)*cov+shrink*np.eye(len(mu))*np.trace(cov)/len(mu)+np.eye(len(mu))*floor
        precision=inv(cov,check_finite=False)
        v=z-mu;dist.append(np.sqrt(np.maximum(np.einsum('ij,ij->i',v@precision,v),0)/z.shape[1]))
    dist=np.array(dist); held=np.full(len(z),np.nan)
    for j,well in enumerate(refs):held[ids==well]=dist[j,ids==well]
    rmask=np.isfinite(held);rw=weights(meta.loc[rmask]);thresholds={q:float(wq(held[rmask],rw,q)) for q in CFG['tail_quantiles']}
    assert all(v>0 for v in thresholds.values())
    mean=dist.mean(0);mean[rmask]=held[rmask]
    out={'distance':mean/thresholds[.95]}
    for q,cut in thresholds.items():
        v=(dist>cut).mean(0);v[rmask]=(held[rmask]>cut).astype(float)
        out[f'tail{q*100:g}']=v
    out['reference_control']=rmask
    return out,thresholds,dist

def summarize(meta,scores,prefix):
    d=pd.DataFrame({'well_id':meta.well_id.to_numpy(),**scores})
    rows=[]
    for well,g in d.groupby('well_id',sort=False):
        r={'well_id':well,prefix+'__median':g.distance.median(),prefix+'__q90':g.distance.quantile(.9),prefix+'__mean':g.distance.mean()}
        r.update({prefix+'__'+c:g[c].mean() for c in g if c.startswith('tail')});rows.append(r)
    return pd.DataFrame(rows).set_index('well_id')

def all_scores(z,meta,features,refs,groups=True,shrink=.2,diagonal=False):
    sets={'All':np.arange(len(features))}
    if groups:sets.update({f:np.array([j for j,c in enumerate(features) if family(c)==f]) for f in FAMILIES})
    wells=[];cell=meta[META+['qc_edge_touching','qc_saturated_maximum','qc_extreme_z8']].copy();cuts=[]
    for f,ix in sets.items():
        if not len(ix):continue
        s,t,d=reference_scores(z[:,ix],meta,refs,shrink,diagonal)
        wells.append(summarize(meta,s,f))
        for name,val in s.items():cell[f+'__'+name]=val
        for q,value in t.items():cuts.append(dict(passage=meta.passage.iloc[0],family=f,quantile=q,threshold=value,reference_wells=len(refs),features=len(ix)))
    out=pd.concat(wells,axis=1)
    wm=meta.drop_duplicates('well_id').set_index('well_id')[['passage','plasma','intervention','physical_well']]
    return wm.join(out),cell,pd.DataFrame(cuts)

def test_rows(wells,cells,endpoints,permutations):
    out=[]
    for name,a,b in contrasts(cells):
        aa=wells.loc[a,endpoints].to_numpy();bb=wells.loc[b,endpoints].to_numpy()
        p=permutations.get(name)
        if p is None:p=exact_p(aa,bb)
        ci=conditional_ci(aa,bb)
        for j,ep in enumerate(endpoints):
            consistency,lo,hi=loo_direction(aa[:,j],bb[:,j])
            out.append(dict(contrast=name,endpoint=ep,n_A=len(a),n_B=len(b),mean_A=aa[:,j].mean(),mean_B=bb[:,j].mean(),effect_B_minus_A=bb[:,j].mean()-aa[:,j].mean(),ci_low_conditional=ci[0,j],ci_high_conditional=ci[1,j],p_exact=p[j],leave_one_well_direction_fraction=consistency,leave_one_well_effect_min=lo,leave_one_well_effect_max=hi,reference_refitted_in_permutation='H1N1_vs_Control' in name))
    result=pd.DataFrame(out);result['q_BH_family']=bh(result.p_exact)
    return result

def feature_tests(cells,arrays):
    f=arrays['all_features'].tolist();z=arrays['all_standardized']
    pieces=[]
    for q,label in [(.1,'q10'),(.5,'median'),(.9,'q90')]:
        d=pd.DataFrame(z,columns=f).assign(well_id=cells.well_id.to_numpy()).groupby('well_id',sort=False).quantile(q)
        d.columns=[c+'::'+label for c in d];pieces.append(d)
    wf=pd.concat(pieces,axis=1);rows=[]
    for name,a,b in contrasts(cells):
        aa=wf.loc[a].to_numpy();bb=wf.loc[b].to_numpy();p=exact_p(aa,bb)
        for j,ep in enumerate(wf):
            feat,stat=ep.split('::');frac,lo,hi=loo_direction(aa[:,j],bb[:,j])
            rows.append(dict(contrast=name,feature=feat,family=family(feat),summary=stat,effect_in_robust_z=bb[:,j].mean()-aa[:,j].mean(),p_exact=p[j],LOO_direction_fraction=frac,LOO_effect_min=lo,LOO_effect_max=hi))
    out=pd.DataFrame(rows);out['q_BH_all_feature_tests']=bh(out.p_exact)
    save(out,'02_responses/feature_driver_tests.csv')
    save(wf.reset_index(),'02_responses/well_feature_summaries.csv.gz')
    return out

def main():
    cells,panel,arrays=load_data();z=arrays['standardized'];features=arrays['features'].tolist()
    wellparts=[];cellparts=[];cutparts=[]
    for passage in ['P3','P9']:
        mask=cells.passage.eq(passage);m=cells.loc[mask].reset_index(drop=True)
        refs=m.loc[m.plasma.eq('Control')&m.intervention.eq('None'),'well_id'].unique()
        w,c,t=all_scores(z[mask],m,features,refs)
        wellparts.append(w);cellparts.append(c);cutparts.append(t)
    wells=pd.concat(wellparts);scored=pd.concat(cellparts,ignore_index=True)
    scored['consensus_responder_95']=np.where(scored['All__reference_control'],-1,(scored['All__tail95']>=.8).astype(int))
    save(wells.reset_index(),'02_responses/well_response_metrics.csv')
    save(scored,'02_responses/cell_response_scores.csv.gz')
    save(pd.concat(cutparts),'02_responses/control_thresholds.csv')
    primary=['All__tail95','All__median','All__q90'];compartment=[f+'__tail95' for f in FAMILIES]
    eps=primary+compartment;perm={}
    # Reference rebuilding prevents label-dependent control fitting from invalidating permutations.
    for name,a,b in contrasts(cells):
        if 'H1N1_vs_Control' not in name:continue
        ids=a+b;mask=cells.well_id.isin(ids);m=cells.loc[mask].reset_index(drop=True);x=z[mask]
        observed=wells.loc[b,eps].mean().to_numpy()-wells.loc[a,eps].mean().to_numpy();null=[]
        for i,comb in enumerate(itertools.combinations(ids,5)):
            other=[j for j in ids if j not in comb]
            w,_,_=all_scores(x,m,features,comb)
            null.append(w.loc[other,eps].mean().to_numpy()-w.loc[list(comb),eps].mean().to_numpy())
            if i%63==0:print('Reference-refitted permutation',name,i,'/252',flush=True)
        null=np.array(null);perm[name]=(abs(null)>=abs(observed)-1e-12).mean(0)
        save(pd.DataFrame(null,columns=eps),'02_responses/permutation_null_'+name+'.csv')
    pt=test_rows(wells,cells,primary,{k:v[:3] for k,v in perm.items()});ct=test_rows(wells,cells,compartment,{k:v[3:] for k,v in perm.items()})
    save(pt,'02_responses/primary_response_tests.csv');save(ct,'02_responses/compartment_response_tests.csv')
    print('Primary inference finished',flush=True)
    driver=feature_tests(cells,arrays)
    sensitivity=[];sensitivity_wells=[]
    for passage in ['P3','P9']:
        mask=cells.passage.eq(passage);m=cells.loc[mask].reset_index(drop=True);x=z[mask]
        refs=m.loc[m.plasma.eq('Control')&m.intervention.eq('None'),'well_id'].unique().tolist()
        specifications=[('shrinkage_0.1',.1,False,None,None),('shrinkage_0.5',.5,False,None,None),('diagonal_distance',.2,True,None,None),('exclude_edge_saturated',.2,False,'qc',None),('without_Zernike',.2,False,'no_Zernike',None),('family_balanced',.2,False,'balance',None)]
        specifications += [('leave_reference_'+r,.2,False,None,r) for r in refs]
        for spec,shrink,diag,filterkind,omit in specifications:
            xx=x.copy();mm=m.copy();ff=features.copy();rr=[r for r in refs if r!=omit]
            if filterkind=='qc':
                keep=~(mm.qc_edge_touching|mm.qc_saturated_maximum);xx=xx[keep];mm=mm.loc[keep].reset_index(drop=True)
            elif filterkind=='no_Zernike':
                keep=np.array(['Zernike' not in f for f in ff]);xx=xx[:,keep];ff=list(np.array(ff)[keep])
            elif filterkind=='balance':
                counts=pd.Series([family(f) for f in ff]).value_counts();xx=xx/np.sqrt([counts[family(f)] for f in ff])
            ww,_,_=all_scores(xx,mm,ff,rr,groups=True,shrink=shrink,diagonal=diag)
            if omit:ww=ww.drop(index=omit)
            sw=ww.reset_index();sw['specification']=spec;sensitivity_wells.append(sw)
            for name,a,b in contrasts(mm):
                if not name.startswith(passage):continue
                a=[v for v in a if v in ww.index];b=[v for v in b if v in ww.index]
                for ep in [f+'__'+metric for f in ['All']+FAMILIES for metric in ['tail90','tail95','tail97.5','median','q90']]:
                    effect=ww.loc[b,ep].mean()-ww.loc[a,ep].mean()
                    primaryeffect=wells.loc[b,ep].mean()-wells.loc[a,ep].mean()
                    sensitivity.append(dict(contrast=name,specification=spec,endpoint=ep,effect_B_minus_A=effect,primary_effect_same_wells=primaryeffect,same_direction=bool(np.sign(effect)==np.sign(primaryeffect))))
        print('Sensitivity finished',passage,flush=True)
    sens=pd.DataFrame(sensitivity);save(sens,'02_responses/response_sensitivity.csv');save(pd.concat(sensitivity_wells),'02_responses/sensitivity_well_metrics.csv')
    # Descriptive responder signatures: not independent validation of response definitions.
    signatures=[]
    index={v:i for i,v in enumerate(cells.cell_id)}
    for well,g in scored.loc[~scored['All__reference_control']].groupby('well_id'):
        respond=g.consensus_responder_95.eq(1);other=g['All__tail95'].eq(0)
        if respond.sum()<2 or other.sum()<2:continue
        a=np.array([index[v] for v in g.loc[respond,'cell_id']]);b=np.array([index[v] for v in g.loc[other,'cell_id']])
        difference=np.median(z[a],axis=0)-np.median(z[b],axis=0)
        for f,d in zip(features,difference):signatures.append(dict(well_id=well,passage=g.passage.iloc[0],plasma=g.plasma.iloc[0],intervention=g.intervention.iloc[0],feature=f,family=family(f),n_responders=len(a),n_nonresponders=len(b),responder_minus_nonresponder_z=d))
    save(pd.DataFrame(signatures),'02_responses/descriptive_responder_signatures.csv')
    summary=dict(primary_tests=len(pt),primary_significant=int(pt.q_BH_family.lt(.05).sum()),compartment_tests=len(ct),compartment_significant=int(ct.q_BH_family.lt(.05).sum()),feature_tests=len(driver),feature_significant=int(driver.q_BH_all_feature_tests.lt(.05).sum()),nonreference_consensus_responders=int(scored.consensus_responder_95.eq(1).sum()),confidence_interval_scope='conditional on fitted reference/feature representation; reference re-fitting assessed separately',primary_results=pt.to_dict('records'))
    dump(summary,'02_responses/summary.json');print('Response analysis complete',summary['primary_significant'],'primary tests pass FDR',flush=True)

if __name__=='__main__':main()
