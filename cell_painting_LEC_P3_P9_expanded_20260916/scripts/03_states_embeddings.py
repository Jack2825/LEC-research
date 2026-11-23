"""Unsupervised states in PCA space; UMAP is a separate visualization sweep."""
from core import *
from sklearn.cluster import KMeans,HDBSCAN
from sklearn.metrics import adjusted_rand_score,silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
import umap

def rep(t,w,target=.9):
    c,s=robust_scale(t,w);z=(t-c)/s;pc,ratio,loading,mu=weighted_pca(z,w)
    return pc[:,:n_pcs(ratio,target)],ratio,loading,z

def km(x,k,w,seed=SEED):
    use=w>0
    model=KMeans(n_clusters=k,n_init=15,random_state=seed,max_iter=300).fit(x[use],sample_weight=w[use]*len(w))
    return model.predict(x)

def jaccards(a,b,k):
    return [max(np.sum((a==j)&(b==l))/max(1,np.sum((a==j)|(b==l))) for l in np.unique(b)) for j in range(k)]

def balanced_indices(m,rng):
    return np.concatenate([rng.choice(ix,size=min(16,len(ix)),replace=False) for ix in m.groupby('well_id').indices.values()])

def state_support(labels,m,k):
    frac=pd.crosstab(m.well_id,labels).reindex(columns=range(k),fill_value=0)
    props=frac.div(frac.sum(1),axis=0)
    rows=[]
    for j in range(k):
        per=frac[j];weighted=props[j];maxmass=float(weighted.max()/weighted.sum())
        rows.append(dict(state=j,cells=int(per.sum()),wells_with_2_cells=int(per.ge(2).sum()),max_well_fraction_of_weighted_state=maxmass,largest_well=weighted.idxmax(),support_pass=bool(per.ge(2).sum()>=3 and maxmass<=.4)))
    return rows,props

def main():
    cells,panel,arrays=load_data();t=arrays['transformed'];features=arrays['features'].tolist()
    fits=[];checks=[];support=[];bootrows=[];loorows=[];sensrows=[];propsrows=[];labelsout=cells[META].copy();densityrows=[];pcparts=[];loadparts=[]
    primarypcs={};scopeids={};base_labels={}
    for scope in ['All','P3','P9']:
        use=np.ones(len(cells),bool) if scope=='All' else cells.passage.eq(scope).to_numpy()
        m=cells.loc[use].reset_index(drop=True);tt=t[use];w=weights(m);rng=np.random.default_rng(SEED)
        scopeids[scope]=np.flatnonzero(use)
        pc,ratio,load,z=rep(tt,w,.95);n90=n_pcs(ratio,.9);basepc=pc[:,:n90];primarypcs[scope]=basepc
        pcf=m[META].copy()
        for j in range(pc.shape[1]):pcf[f'PC{j+1}']=pc[:,j]
        pcf['scope']=scope;pcparts.append(pcf)
        for j in range(min(10,load.shape[1])):
            for f,v in zip(features,load[:,j]):loadparts.append(dict(scope=scope,PC=j+1,feature=f,family=family(f),loading=v,variance_ratio=ratio[j]))
        bix=balanced_indices(m,rng)
        baselines={k:km(basepc,k,w) for k in CFG['cluster_k']};base_labels[scope]=baselines
        for k,labels in baselines.items():
            col=f'{scope}__k{k}';labelsout[col]=-99;labelsout.loc[use,col]=labels
            rows,proportions=state_support(labels,m,k)
            for r in rows:r.update(scope=scope,k=k);support.append(r)
            for well,row in proportions.iterrows():
                info=m.loc[m.well_id.eq(well)].iloc[0]
                for state,value in row.items():propsrows.append(dict(scope=scope,k=k,state=state,well_id=well,passage=info.passage,plasma=info.plasma,intervention=info.intervention,proportion=value,n_cells=int(m.well_id.eq(well).sum())))
            for target in CFG['cluster_pca_variances']:
                xp=pc[:,:n_pcs(ratio,target)]
                for seed in [SEED,SEED+1,SEED+2]:
                    lab=km(xp,k,w,seed)
                    sil=silhouette_score(xp[bix],lab[bix]) if len(np.unique(lab[bix]))>1 else np.nan
                    fits.append(dict(scope=scope,k=k,pca_variance_target=target,n_pcs=xp.shape[1],seed=seed,silhouette_equal_well_sample=sil,ARI_to_primary=adjusted_rand_score(labels,lab),min_state_jaccard=min(jaccards(labels,lab,k))))
        print('Cluster parameter sweep',scope,'PCs',n90,flush=True)
        # Reuse each resampled representation across all k; do not select favorable resamples.
        ids=m.well_id.to_numpy();groups=m.groupby('well_id').indices
        strata=m.drop_duplicates('well_id').groupby(['passage','plasma','intervention']).well_id.apply(list).tolist()
        for b in range(CFG['cluster_bootstraps']):
            counts=np.zeros(len(m))
            for stratum in strata:
                for well in rng.choice(stratum,len(stratum),replace=True):
                    ix=groups[well];draw=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)))
                    counts[ix]+=draw/len(ix)
            bw=counts/counts.sum();bp,_,_,_=rep(tt,bw)
            for k,base in baselines.items():
                lab=km(bp,k,bw,SEED+b)
                js=jaccards(base,lab,k);ari=adjusted_rand_score(base,lab)
                unseen=counts==0
                for state,value in enumerate(js):bootrows.append(dict(scope=scope,k=k,bootstrap=b,state=state,jaccard=value,ARI_all_cells=ari,ARI_unseen_cells=adjusted_rand_score(base[unseen],lab[unseen]) if unseen.sum()>10 else np.nan))
            if b%10==0:print('Hierarchical cluster bootstrap',scope,b,flush=True)
        for index,well in enumerate(groups):
            lw=w.copy();lw[ids==well]=0;lw/=lw.sum();lp,_,_,_=rep(tt,lw)
            for k,base in baselines.items():
                lab=km(lp,k,lw);shared=ids!=well;js=jaccards(base[shared],lab[shared],k)
                for state,value in enumerate(js):loorows.append(dict(scope=scope,k=k,omitted_well=well,state=state,jaccard=value,ARI_shared_cells=adjusted_rand_score(base[shared],lab[shared]),ARI_all_cells=adjusted_rand_score(base,lab)))
            if index%15==0:print('Leave-one-well-out clustering',scope,index,flush=True)
        for name in ['morphology_only','without_nuclear_intensity','exclude_edge_saturated','without_Zernike','family_balanced']:
            keep=np.ones(len(features),bool);sw=w.copy();sensitivity_t=tt.copy()
            if name=='morphology_only':keep=np.array(['AreaShape_' in f for f in features])
            if name=='without_nuclear_intensity':keep=np.array([family(f)!='Nuclear intensity' for f in features])
            if name=='without_Zernike':keep=np.array(['Zernike' not in f for f in features])
            if name=='exclude_edge_saturated':sw[(m.qc_edge_touching|m.qc_saturated_maximum).to_numpy()]=0;sw/=sw.sum()
            sp,sr,sl,sz=rep(sensitivity_t[:,keep],sw)
            if name=='family_balanced':
                cnt=pd.Series([family(f) for f in features]).value_counts();sz=sz/np.sqrt([cnt[family(f)] for f in features]);sp,sr,_,_=weighted_pca(sz,sw);sp=sp[:,:n_pcs(sr,.9)]
            for k,base in baselines.items():
                lab=km(sp,k,sw);shared=sw>0
                sensrows.append(dict(scope=scope,k=k,sensitivity=name,ARI=adjusted_rand_score(base[shared],lab[shared]),min_state_jaccard=min(jaccards(base[shared],lab[shared],k))))
        for size in [30,60,120]:
            for samples in [5,15]:
                lab=HDBSCAN(min_cluster_size=size,min_samples=samples,n_jobs=1,copy=True).fit_predict(basepc)
                col=f'{scope}__HDBSCAN_size{size}_samples{samples}';labelsout[col]=-99;labelsout.loc[use,col]=lab
                valid=lab>=0;ncluster=len(np.unique(lab[valid]));sil=np.nan
                ix=bix[valid[bix]]
                if ncluster>=2 and len(ix)>ncluster and len(np.unique(lab[ix]))>=2:sil=silhouette_score(basepc[ix],lab[ix])
                densityrows.append(dict(scope=scope,min_cluster_size=size,min_samples=samples,n_clusters=ncluster,unassigned_fraction=float((~valid).mean()),silhouette_assigned_balanced_sample=sil))
        print('Clustering finished',scope,flush=True)
    fits=pd.DataFrame(fits);boot=pd.DataFrame(bootrows);loo=pd.DataFrame(loorows);supp=pd.DataFrame(support);sens=pd.DataFrame(sensrows);props=pd.DataFrame(propsrows)
    screens=[]
    for scope in ['All','P3','P9']:
        for k in CFG['cluster_k']:
            bb=boot.loc[boot.scope.eq(scope)&boot.k.eq(k)];ll=loo.loc[loo.scope.eq(scope)&loo.k.eq(k)];ff=fits.loc[fits.scope.eq(scope)&fits.k.eq(k)];ss=supp.loc[supp.scope.eq(scope)&supp.k.eq(k)]
            row=dict(scope=scope,k=k,bootstrap_median_ARI=bb.groupby('bootstrap').ARI_all_cells.first().median(),bootstrap_p10_ARI=bb.groupby('bootstrap').ARI_all_cells.first().quantile(.1),bootstrap_min_state_median_jaccard=bb.groupby('state').jaccard.median().min(),LOO_median_ARI=ll.groupby('omitted_well').ARI_shared_cells.first().median(),LOO_min_ARI=ll.groupby('omitted_well').ARI_shared_cells.first().min(),LOO_min_state_median_jaccard=ll.groupby('state').jaccard.median().min(),parameter_min_ARI=ff.ARI_to_primary.min(),all_states_well_supported=ss.support_pass.all())
            row['resampling_screen_pass']=bool(row['bootstrap_median_ARI']>=.8 and row['bootstrap_min_state_median_jaccard']>=.75 and row['LOO_median_ARI']>=.8 and row['LOO_min_ARI']>=.6 and row['LOO_min_state_median_jaccard']>=.75 and row['parameter_min_ARI']>=.8 and row['all_states_well_supported'])
            screens.append(row)
    screen=pd.DataFrame(screens)
    tests=[]
    for (scope,k,state),g in props.groupby(['scope','k','state']):
        v=g.set_index('well_id').proportion
        for name,a,b in contrasts(cells):
            if not all(w in v.index for w in a+b):continue
            aa=v.loc[a].to_numpy();bb=v.loc[b].to_numpy();frac,lo,hi=loo_direction(aa,bb)
            tests.append(dict(scope=scope,k=k,state=state,contrast=name,mean_A=aa.mean(),mean_B=bb.mean(),effect_B_minus_A=bb.mean()-aa.mean(),p_exact=float(exact_p(aa,bb)[0]),LOO_direction_fraction=frac))
    tests=pd.DataFrame(tests);tests['q_BH_all_candidate_state_tests']=bh(tests.p_exact)
    for df,name in [(fits,'parameter_scan'),(boot,'bootstrap_stability'),(loo,'leave_one_well_out'),(supp,'state_well_support'),(sens,'feature_panel_sensitivity'),(props,'well_state_proportions'),(screen,'candidate_screens'),(tests,'state_proportion_tests'),(pd.DataFrame(densityrows),'density_clustering_scan'),(pd.DataFrame(loadparts),'pca_loadings')]:save(df,'03_states/'+name+'.csv')
    save(labelsout,'03_states/cell_cluster_assignments.csv.gz');save(pd.concat(pcparts),'04_embeddings/pca_coordinates.csv.gz')
    # State profiles are descriptive and are not independent validation of their defining features.
    profiles=[]
    for scope in ['All','P3','P9']:
        ix=scopeids[scope];m=cells.iloc[ix].reset_index(drop=True);w=weights(m)
        for k,lab in base_labels[scope].items():
            for state in range(k):
                use=lab==state
                vals=wq(arrays['standardized'][ix][use],w[use],.5)
                for f,value in zip(features,vals):profiles.append(dict(scope=scope,k=k,state=state,feature=f,family=family(f),weighted_median_global_z=value))
    save(pd.DataFrame(profiles),'03_states/state_feature_profiles.csv')
    print('Starting UMAP sweep (visualization only)',flush=True)
    pc=primarypcs['All'];rng=np.random.default_rng(SEED);sample=balanced_indices(cells,rng)
    original_nn=NearestNeighbors(n_neighbors=16).fit(pc).kneighbors(pc[sample],return_distance=False)[:,1:]
    base_nn={};embeddings=[];metrics=[]
    for neighbors in CFG['umap_neighbors']:
        for mindist in CFG['umap_min_dist']:
            for seed in CFG['umap_seeds']:
                emb=umap.UMAP(n_neighbors=neighbors,min_dist=mindist,random_state=seed,n_jobs=1,n_epochs=350).fit_transform(pc)
                key=f'n{neighbors}_d{mindist}_s{seed}'
                df=pd.DataFrame({'cell_id':cells.cell_id,'setting':key,'UMAP1':emb[:,0],'UMAP2':emb[:,1]});embeddings.append(df)
                en=NearestNeighbors(n_neighbors=16).fit(emb).kneighbors(emb[sample],return_distance=False)[:,1:]
                overlap=np.mean([len(set(a)&set(b))/15 for a,b in zip(original_nn,en)])
                basekey=(neighbors,mindist)
                if basekey not in base_nn:base_nn[basekey]=en
                seedover=np.mean([len(set(a)&set(b))/len(set(a)|set(b)) for a,b in zip(base_nn[basekey],en)])
                metrics.append(dict(setting=key,n_neighbors=neighbors,min_dist=mindist,seed=seed,neighbor_recall_15=float(overlap),trustworthiness_balanced_sample=float(trustworthiness(pc[sample],emb[sample],n_neighbors=15)),seed_neighbor_jaccard=float(seedover)))
                print('UMAP',key,'neighbor recall',round(overlap,3),flush=True)
    save(pd.concat(embeddings),'04_embeddings/umap_all_settings.csv.gz');save(pd.DataFrame(metrics),'04_embeddings/umap_quality.csv')
    dump(dict(candidate_partitions=len(screen),passing_resampling_screen=int(screen.resampling_screen_pass.sum()),candidate_screens=screen.to_dict('records'),state_proportion_tests=len(tests),significant_state_proportion_tests=int(tests.q_BH_all_candidate_state_tests.lt(.05).sum()),umap_settings=len(metrics),all_states_exploratory=True),'03_states/summary.json')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
