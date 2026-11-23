"""Well-weighted PCA and within-passage H1N1 displacement for Panel C.

Run panel_C_preprocessing.py first, then this script with the same --output-dir.
Inference enumerates all 252 allocations of ten whole wells to groups of five.
Observed biological assignments are never changed in saved cell/well metadata.
"""
from pathlib import Path
from figure_paths import panel_dir, resolve_result_path, assert_preserved
import argparse
import itertools
import json
import os
import platform
import sys
sys.dont_write_bytecode=True
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(variable,'1')
from panel_C_preprocessing import ROOT,RESULT,CONFIG,sha,dump,output_dir
os.environ['MPLCONFIGDIR']=str(RESULT/'.matplotlib_cache')
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pymupdf

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'pdf.fonttype':42,'svg.fonttype':'none',
                     'savefig.facecolor':'white','axes.linewidth':.8,
                     'xtick.labelsize':9,'ytick.labelsize':9})
COLORS={'P3':'#0072B2','P9':'#D55E00'}
MARKERS={'Control':'o','H1N1':'^'}
GROUPS=['P3-Control','P3-H1N1','P9-Control','P9-H1N1']


def pca(z,w):
    center=np.average(z,axis=0,weights=w)
    centered=z-center
    covariance=(centered*w[:,None]).T@centered
    values,components=np.linalg.eigh(covariance)
    order=np.argsort(values)[::-1]
    values=np.maximum(values[order],0)
    components=components[:,order]
    # Deterministic signs, independent of biological group labels.
    anchors=np.argmax(np.abs(components),axis=0)
    signs=np.sign(components[anchors,np.arange(len(anchors))])
    components*=np.where(signs==0,1,signs)
    return centered@components,values/values.sum(),components,center


def bh(values):
    values=np.asarray(values,float)
    order=np.argsort(values,kind='stable')
    ordered=values[order]*len(values)/np.arange(1,len(values)+1)
    q=np.minimum.accumulate(ordered[::-1])[::-1].clip(0,1)
    result=np.empty_like(values);result[order]=q
    return result


def axes_style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(width=.8,length=3)


def scatter_groups(ax,frame,size,alpha=1,edge=False):
    for group in GROUPS:
        passage,treatment=group.split('-',1)
        d=frame.loc[frame.panel_group.eq(group)]
        ax.scatter(d.PC1,d.PC2,s=size,marker=MARKERS[treatment],
                   c=COLORS[passage],alpha=alpha,
                   edgecolors='#17212b' if edge else 'none',
                   linewidths=.55 if edge else 0,
                   rasterized=not edge,zorder=3 if edge else 2)


def response_arrows(ax,wells):
    for passage in ['P3','P9']:
        means=wells.loc[wells.passage.eq(passage)].groupby('plasma')[['PC1','PC2']].mean()
        if means.empty:
            continue
        assert set(means.index)=={'Control','H1N1'}
        ax.annotate('',xy=means.loc['H1N1'],xytext=means.loc['Control'],
                    arrowprops={'arrowstyle':'-|>','color':COLORS[passage],
                                'lw':1.7,'mutation_scale':11,'shrinkA':0,'shrinkB':0},zorder=4)


def render(out,cells,wells,variance,metrics,null,n_features):
    fig=plt.figure(figsize=(12.2,4.6),facecolor='white')
    fig.text(.018,.969,'C',fontsize=21,fontweight='bold',va='top')
    fig.text(.054,.962,'Unbiased global morphological landscape',fontsize=15,fontweight='bold',va='top')
    handles=[]
    for group in GROUPS:
        passage,treatment=group.split('-',1)
        handles.append(Line2D([],[],marker=MARKERS[treatment],linestyle='',
                              markerfacecolor=COLORS[passage],markeredgecolor='#17212b',
                              markeredgewidth=.5,markersize=6,label=group))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.900),ncol=4,
               frameon=False,fontsize=10,handletextpad=.5,columnspacing=2.2)
    ax=fig.add_axes([.066,.225,.321,.505])
    aw=fig.add_axes([.463,.225,.226,.505])
    ad=fig.add_axes([.775,.225,.207,.505])
    ax.set_title('Single-cell PCA',fontsize=11,fontweight='bold',pad=15)
    aw.set_title('Well centroids · same PCA',fontsize=11,fontweight='bold',pad=15)
    ad.set_title('H1N1 displacement',fontsize=11,fontweight='bold',pad=15)
    scatter_groups(ax,cells,8,.34)
    scatter_groups(aw,wells,44,1,True)
    response_arrows(aw,wells)
    for target in [ax,aw]:
        target.set_xlabel(f'PC1 ({variance[0]*100:.1f}%)',labelpad=5)
        target.set_ylabel(f'PC2 ({variance[1]*100:.1f}%)',labelpad=4)
        target.margins(.09)
        axes_style(target)
    # This enlarged well-centroid view uses the same scores, not a separate PCA.
    xmax=0
    for y,passage in [(1,'P3'),(0,'P9')]:
        row=metrics.loc[metrics.passage.eq(passage)].iloc[0]
        reference=null.loc[null.passage.eq(passage),'RMS_displacement'].to_numpy()
        parts=ad.violinplot([reference],positions=[y],orientation='horizontal',
                           widths=.29,showmeans=False,showmedians=False,showextrema=False)
        for body in parts['bodies']:
            body.set_facecolor('#c4c9ce');body.set_edgecolor('none');body.set_alpha(.6)
        ad.plot([row.LOO_min,row.LOO_max],[y,y],color=COLORS[passage],lw=2,zorder=3)
        ad.plot([row.LOO_min,row.LOO_min],[y-.045,y+.045],color=COLORS[passage],lw=1)
        ad.plot([row.LOO_max,row.LOO_max],[y-.045,y+.045],color=COLORS[passage],lw=1)
        ad.scatter([row.RMS_displacement],[y],s=55,marker='D',c=COLORS[passage],
                   edgecolors='white',linewidths=.6,zorder=5)
        ad.text(.0,y+.32,f'D={row.RMS_displacement:.3f}',transform=ad.get_yaxis_transform(),fontsize=9.5)
        ad.text(.0,y-.36,f'P={row.p_exact:.3f}; q={row.q_BH_two_tests:.3f}',
                transform=ad.get_yaxis_transform(),fontsize=9)
        xmax=max(xmax,float(reference.max()),row.LOO_max,row.RMS_displacement)
    ad.set(xlim=(0,xmax*1.10),ylim=(-.60,1.60),yticks=[1,0],yticklabels=['P3','P9'])
    ad.set_xlabel('RMS difference\n(robust-scaled units)',labelpad=5,fontsize=9.5)
    ad.locator_params(axis='x',nbins=4)
    axes_style(ad)
    fig.text(.066,.064,f'{n_features} features · 1,626 cells · 20 wells (5 per group)',fontsize=9,color='#626d76')
    fig.text(.463,.064,'Each point is one well; arrows: Control → H1N1',fontsize=8.1,color='#626d76')
    fig.text(.775,.095,'Grey: permuted-label reference\nBars: leave-one-well-out range',fontsize=8.1,color='#626d76',va='top')
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'panel_C.{ext}',dpi=600)
    fig.savefig(out/'panel_C.preview.png',dpi=180)
    plt.close(fig)
    # A traceable supplement exposes every physical well without crowding Panel C.
    fig,axes=plt.subplots(1,2,figsize=(9.0,4.0))
    for ax,passage in zip(axes,['P3','P9']):
        d=wells.loc[wells.passage.eq(passage)]
        scatter_groups(ax,d,70,1,True)
        response_arrows(ax,d)
        for row in d.itertuples():
            # Separate the two adjacent P3 labels without moving their data points.
            offset=(-6,6) if (passage,row.physical_well)==('P3','B06') else (4,5)
            align='right' if offset[0]<0 else 'left'
            ax.annotate(row.physical_well,(row.PC1,row.PC2),xytext=offset,
                        textcoords='offset points',fontsize=8,ha=align)
        ax.set_title(passage+' · individual well centroids',fontsize=11,fontweight='bold')
        ax.set_xlabel(f'PC1 ({variance[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({variance[1]*100:.1f}%)')
        ax.margins(.25);axes_style(ax)
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Primary-only PCA: well identity and within-passage response',fontsize=12)
    fig.tight_layout(rect=(0,.09,1,.94))
    fig.savefig(out/'panel_C.well_centroids.pdf')
    fig.savefig(out/'panel_C.well_centroids.png',dpi=300)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=panel_dir('C'))
    parser.add_argument('--render-only',action='store_true',help='Regenerate only this run\'s derived figure artwork.')
    parser.add_argument('--resume-incomplete',action='store_true',help='Resume a run that has no completion/validation record.')
    args=parser.parse_args();out=output_dir(args.output_dir);data=out/'panel_C_data'
    pre=json.loads((out/'panel_C_preprocessing.json').read_text(encoding='utf-8'))
    if args.render_only:
        numeric_paths=[data/'pca_cell_scores.csv.gz',data/'pca_well_centroids.csv',
                       data/'pca_variance.csv',out/'panel_C.displacement.csv',
                       data/'displacement_permutation_null.csv']
        before={p:sha(p) for p in numeric_paths}
        render(out,pd.read_csv(data/'pca_cell_scores.csv.gz',keep_default_na=False),
               pd.read_csv(data/'pca_well_centroids.csv',keep_default_na=False),
               pd.read_csv(data/'pca_variance.csv').variance_fraction.to_numpy(),
               pd.read_csv(out/'panel_C.displacement.csv'),
               pd.read_csv(data/'displacement_permutation_null.csv'),pre['retained_features'])
        assert all(sha(p)==h for p,h in before.items())
        validation_path=out/'panel_C.validation.json'
        if validation_path.exists():
            validation=json.loads(validation_path.read_text(encoding='utf-8'))
            validation['script_sha256']={name:sha(ROOT/'script'/name) for name in ['panel_C_preprocessing.py','panel_C_pca.py']}
            validation['figure_regenerated_without_numerical_changes']=True
            dump(validation,validation_path)
        return
    if (out/'panel_C.validation.json').exists():
        raise FileExistsError('Preserving completed Panel C. Use a new preprocessing/output directory.')
    if ((out/'panel_C.pdf').exists() or (out/'panel_C.displacement.csv').exists()) and not args.resume_incomplete:
        raise FileExistsError('Preserving completed Panel C. Use a new preprocessing/output directory.')
    preservation=pd.read_csv(data/'source_and_preservation_manifest.csv',keep_default_na=False)
    for row in preservation.itertuples():
        assert_preserved(row.source_file, row.sha256)
    matrix_hash=sha(data/'normalized_features.npz')
    meta=pd.read_csv(data/'cell_metadata.csv',keep_default_na=False)
    with np.load(data/'normalized_features.npz',allow_pickle=False) as a:
        z=a['standardized'];features=a['features'].tolist();w=a['weights']
        assert np.array_equal(a['cell_ids'],meta.cell_id.to_numpy(str))
    assert meta.intervention.eq('None').all()
    assert set(meta.panel_group)==set(GROUPS)
    pc,variance,loadings,center=pca(z,w)
    assert np.allclose(loadings.T@loadings,np.eye(len(features)),atol=1e-10)
    assert np.allclose(np.average(pc,axis=0,weights=w),0,atol=1e-10)
    assert np.allclose(pc@loadings.T+center,z,atol=1e-10)
    pc_names=[f'PC{i+1}' for i in range(len(features))]
    pc_cells=pd.concat([meta,pd.DataFrame(pc,columns=pc_names)],axis=1)
    pc_cells.to_csv(data/'pca_cell_scores.csv.gz',index=False)
    well_meta=meta.groupby('well_id',sort=True).agg(
        passage=('passage','first'),plasma=('plasma','first'),intervention=('intervention','first'),
        panel_group=('panel_group','first'),physical_well=('physical_well','first'),
        n_cells=('cell_id','size')).reset_index()
    z_frame=pd.DataFrame(z,columns=features)
    z_frame['well_id']=meta.well_id
    well_z=z_frame.groupby('well_id',sort=True)[features].mean().loc[well_meta.well_id]
    well_pcs=pc_cells.groupby('well_id',sort=True)[pc_names].mean().loc[well_meta.well_id]
    well_pc_frame=pd.concat([well_meta,well_pcs.reset_index(drop=True)],axis=1)
    well_pc_frame.to_csv(data/'pca_well_centroids.csv',index=False)
    pd.concat([well_meta,well_z.reset_index(drop=True)],axis=1).to_csv(data/'well_feature_centroids.csv',index=False)
    pd.DataFrame({'PC':pc_names,'variance_fraction':variance,'cumulative_variance':np.cumsum(variance)}).to_csv(data/'pca_variance.csv',index=False)
    load_table=pd.DataFrame(loadings,index=features,columns=pc_names).rename_axis('feature').reset_index()
    load_table.to_csv(data/'pca_loadings.csv',index=False)
    np.savez_compressed(data/'pca_model.npz',components=loadings,center=center,variance=variance,features=np.array(features))
    rows=[];null_rows=[];loo_rows=[];responses=pd.DataFrame({'feature':features})
    for passage in ['P3','P9']:
        subset=well_meta.loc[well_meta.passage.eq(passage)].reset_index(drop=True)
        x=well_z.loc[subset.well_id].to_numpy()
        control=subset.plasma.eq('Control').to_numpy();challenge=~control
        assert control.sum()==5 and challenge.sum()==5 and len(x)==10
        delta=x[challenge].mean(axis=0)-x[control].mean(axis=0)
        observed=float(np.sqrt(np.mean(delta**2)))
        responses[passage+'_H1N1_minus_Control']=delta
        # Inference uses full feature space, not the displayed PC1/PC2 projection.
        projected=well_pcs.loc[subset.well_id].to_numpy()
        pc_delta=projected[challenge].mean(axis=0)-projected[control].mean(axis=0)
        assert np.isclose(observed,np.sqrt(np.mean(pc_delta**2)),rtol=1e-10,atol=1e-10)
        assignments=list(itertools.combinations(range(10),5))
        null=[]
        for number,allocation in enumerate(assignments):
            null_control=np.zeros(10,bool);null_control[list(allocation)]=True
            contrast=x[~null_control].mean(axis=0)-x[null_control].mean(axis=0)
            distance=float(np.sqrt(np.mean(contrast**2)))
            null.append(distance)
            null_rows.append({'passage':passage,'allocation_index':number,
                              'RMS_displacement':distance,'null_reference_only':True,
                              'control_well_indices_under_null':','.join(map(str,allocation))})
        p_exact=float(np.mean(np.asarray(null)>=observed-1e-12*max(1,observed)))
        omitted_distances=[]
        for i,well in enumerate(subset.well_id):
            c=control.copy();h=challenge.copy();c[i]=False;h[i]=False
            d=x[h].mean(axis=0)-x[c].mean(axis=0)
            distance=float(np.sqrt(np.mean(d**2)))
            omitted_distances.append(distance)
            denominator=np.linalg.norm(d)*np.linalg.norm(delta)
            cosine=float(d@delta/denominator) if denominator>0 else 0.
            loo_rows.append({'passage':passage,'omitted_well':well,
                             'omitted_group':subset.iloc[i].panel_group,
                             'RMS_displacement':distance,'response_vector_cosine_to_full':cosine,
                             'normalization_refitted':False})
        well_meta.loc[well_meta.passage.eq(passage)].to_csv(data/f'permutation_well_order_{passage}.csv',index=False)
        rows.append({'passage':passage,'contrast':passage+'-H1N1 relative to '+passage+'-Control',
                     'n_Control_wells':5,'n_H1N1_wells':5,'n_features':len(features),
                     'RMS_displacement':observed,'LOO_min':float(min(omitted_distances)),
                     'LOO_max':float(max(omitted_distances)),'p_exact':p_exact,
                     'permutation_allocations':len(null),
                     'null_median':float(np.median(null)),
                     'null_q025':float(np.quantile(null,.025)),
                     'null_q975':float(np.quantile(null,.975)),
                     'LOO_range_is_confidence_interval':False})
    metrics=pd.DataFrame(rows)
    metrics['q_BH_two_tests']=bh(metrics.p_exact)
    metrics.to_csv(out/'panel_C.displacement.csv',index=False)
    null=pd.DataFrame(null_rows);null.to_csv(data/'displacement_permutation_null.csv',index=False)
    pd.DataFrame(loo_rows).to_csv(data/'displacement_leave_one_well_out.csv',index=False)
    responses.to_csv(data/'within_passage_response_vectors.csv',index=False)
    n90=int(np.searchsorted(np.cumsum(variance),.9)+1)
    render(out,pc_cells,well_pc_frame,variance,metrics,null,len(features))
    assert sha(data/'normalized_features.npz')==matrix_hash
    for row in preservation.itertuples():
        assert_preserved(row.source_file, row.sha256)
    with pymupdf.open(out/'panel_C.pdf') as doc:
        assert len(doc)==1
        text=doc[0].get_text()
        assert all(g in text for g in GROUPS)
        assert 'VEGF' not in text and 'UC2288' not in text
    summary={'cells':len(meta),'wells':len(well_meta),'features':len(features),
             'PC1_variance_fraction':float(variance[0]),'PC2_variance_fraction':float(variance[1]),
             'PC1_PC2_total_variance_fraction':float(variance[:2].sum()),'PCs_for_90pct_variance':n90,
             'displacement':metrics.to_dict('records'),
             'primary_only_preprocessing_and_PCA':True,'equal_total_weight_per_well':True,
             'no_source_or_previous_panel_changes':True,'additional_cells_removed':0,
             'full_PC_RMS_equals_full_feature_RMS':True,'PCA_reconstruction_verified':True,
             'cell_rows_are_not_treatment_replicates':True,'feature_level_tests_performed':False,
             'normalization_matrix_sha256':matrix_hash,
             'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                         'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
             'script_sha256':{name:sha(ROOT/'script'/name) for name in ['panel_C_preprocessing.py','panel_C_pca.py']}}
    dump(summary,out/'panel_C.validation.json')
    methods=f'''# Panel C — primary-only morphological landscape

## Inputs, population and QC

Source: `cell_painting_LEC_P3_P9_expanded_20260916/results/01_data_qc/cells_all_features.csv.gz`. The existing normalized matrix contains all 12 conditions and is not reused. Panel C uses only intervention=None and plasma=Control/H1N1, with P3/P9 labels preserved: 1,626 cells, 20 wells, five wells per primary group. Source hashes and preservation checks are recorded in `panel_C_data/source_and_preservation_manifest.csv`.

The audit covers every input column: 17 metadata/QC columns are excluded from the numerical feature space. Of 186 measurement columns, 43 absolute coordinates, two orientations and two nearly constant Euler-number features are excluded. There are 139 remaining candidates; fixed redundancy filtering excludes 41, leaving {len(features)} features. No invalid or missing measurements occur, and no cells are additionally deleted. All 90 edge-flagged cells are retained. There are {pre['primary_extreme_z8_cells_retained']} cells flagged by the new primary-only |z|>8 rule; the flag is not a deletion criterion.

Rules were specified before PCA or displacement analysis: >1% missing/nonfinite by cell fraction or equal-well fraction; >=99% identical values; zero variance or transformed weighted SD <=1e-12; and |equal-well weighted rank correlation|>0.90. Average ranks are computed across the primary cells, and their correlation covariance gives each well equal total weight. The fixed feature hierarchy is copied from the earlier study, favoring direct measurements before higher-order Zernike descriptors. No H1N1 effect, significance, or visual separation enters feature selection. Full decisions, paired redundancy features, correlations and fitted parameters are in `panel_C_feature_QC.csv`.

## Common normalization and PCA

Each cell in well w receives weight 1/(20 × n_w). Nonnegative unbounded measurements use log1p(x / weighted positive median); bounded shape ratios remain linear. Features share one primary-cohort weighted median center and 1.4826 × weighted MAD scale across all four groups. IQR/1.349 and weighted SD are fallback rules; all {len(features)} retained features in this run use MAD. No group-specific normalization, clipping, outlier removal or separation optimization is applied.

PCA diagonalizes the weighted covariance of the normalized single-cell matrix after subtracting its weighted mean. Signs are fixed by requiring each component's largest absolute loading to be positive. All wells have equal total influence; cell yields therefore do not determine replicate weights. PC1 explains {100*variance[0]:.2f}% and PC2 {100*variance[1]:.2f}% ({100*variance[:2].sum():.2f}% together). {n90} components are needed for >=90% variance. The figure uses only PC1/PC2 as a display; the full spectrum, all scores and loadings are exported. Well centroids are averages of their cells' scores from this same PCA, not a separate fit. Mean-group arrows are Control → H1N1 within passage. The well view has enlarged axis ranges to expose replicate spread.

## H1N1 phenotypic displacement

For every well, calculate its mean vector across all {len(features)} normalized features. For each passage separately, average the five Control well vectors and the five H1N1 well vectors with equal weight. Let delta = mean(H1N1 wells) − mean(Control wells). The reported distance is:

    D = sqrt(sum(delta_j**2) / number_of_features)

This is RMS displacement in common robust-scaled feature units. It measures magnitude, not benefit/harm, and is not a P3-versus-P9 baseline comparison. It uses the full feature space; its equality to RMS displacement in all PCA coordinates is checked. The metric avoids estimating an unstable high-dimensional inverse covariance from five wells. It is nonnegative and has a positive sampling-noise floor, so D>0 alone is not statistical evidence of a response.

For each passage, enumerate all choose(10,5)=252 allocations of the ten whole-well profiles into groups of five. The P value is the fraction whose displacement is at least as large as observed. This is a nondirectional multivariate test; both response directions are included. No Monte Carlo approximation is used, so no +1 correction is needed. The original assignments remain unchanged in the observed data. Preprocessing is not refitted under allocation because it is label-blind and depends on the same wells in every allocation. BH correction is applied to these two prespecified within-passage tests only; it is not a claim of experiment-wide FDR control across other panels.

Grey violin shapes display the exact permuted-label reference distributions. Coloured diamonds are observed D. The coloured bars show the minimum and maximum after omitting each of the ten wells in turn, keeping preprocessing fixed. They are sensitivity ranges, NOT 95% confidence intervals. A conventional bootstrap interval for a nonnegative norm could misleadingly exclude zero even under a null effect; inference instead uses the permutation reference. All allocations, omit-one-well distances and full response vectors are saved.

## Results

| Passage | RMS displacement | Leave-one-well-out range | Exact P | BH q (2 tests) |
|---|---:|---:|---:|---:|
'''
    for row in metrics.itertuples():
        methods+=f'| {row.passage} | {row.RMS_displacement:.4f} | {row.LOO_min:.4f}–{row.LOO_max:.4f} | {row.p_exact:.6f} | {row.q_BH_two_tests:.6f} |\n'
    significant=metrics.loc[metrics.q_BH_two_tests.lt(.05),'passage'].tolist()
    methods+=('\nNeither within-passage displacement test passes BH q<0.05. This is inconclusive evidence, not proof of no change.\n'
              if not significant else '\nPassages meeting the two-test BH q<0.05 threshold: '+', '.join(significant)+'.\n')
    methods+='''
## Interpretation limits and figure reading

PCA is descriptive: separation may reflect passage, technical factors or cell heterogeneity. It is not itself a test of treatment effects. The two displacement tests are not an interaction test; comparing their magnitudes or significance levels does not establish a difference between the P3 and P9 H1N1 responses. That requires the explicit response comparison requested for Panel D.

Wells are treated as independent at the user's request, but donor/pool identities have not been independently verified. Passage is confounded with acquisition date and processing settings. Treatments occupy fixed plate rows/columns, so technical position effects can violate exchangeability of well labels. Exact enumeration does not remove these design limitations. Only the exported first field per well enters this analysis. Cell counts are much lower for P3. The analysis has only five wells per primary group; inferential precision is limited.

The Fig. 2E CY5/TRITC channel-label discrepancy remains unresolved. Panel C retains source numerical feature identities without changing assignments and makes no organelle-specific mechanistic claim. Displaying all retained measurements does not certify the biological annotation of a channel.

Outputs: `panel_C.pdf/svg/png`, preview; a separate labelled-well plot; displacement CSV; complete feature QC CSV/Markdown; normalized cell matrix; well profiles; PCA scores/loadings/spectrum; permutation and sensitivity tables; source manifests; validation and software versions. No UMAP was generated because it is optional and PCA is the primary evidence requested here.

Reproduce with the workspace interpreter, first `script/panel_C_preprocessing.py`, then `script/panel_C_pca.py`. For a new run, give both scripts the same new `--output-dir` inside `result/`; default execution refuses to overwrite completed Panel C results.
'''
    (out/'panel_C.methods.md').write_text(methods,encoding='utf-8')
    legend=(f'**C. Unbiased global morphological landscape.** PCA of {len(features)} quality-filtered, commonly normalized features from 1,626 cells in 20 wells. '
            'Only P3-Control, P3-H1N1, P9-Control and P9-H1N1 are included. Blue denotes P3 and orange P9; circles denote Control and triangles H1N1. '
            'Left: individual cells, with each well given equal total weight during PCA fitting. Middle: individual well centroids in the same PCA, shown at enlarged axis ranges; arrows connect Control to H1N1 group centroids. '
            'Right: within-passage H1N1-versus-Control RMS displacement of mean well profiles in the full normalized feature space. Diamonds indicate observed displacement; bars indicate the leave-one-well-out range, not a confidence interval. '
            'Grey distributions show all 252 five-versus-five well-label allocations per passage. P values are exact allocation-test values; q values use BH correction across the two passage-specific tests. '
            'These comparisons do not test the passage-by-H1N1 interaction. The donor and plate-position limitations are documented in the methods.\n')
    (out/'panel_C.legend.md').write_text(legend,encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
