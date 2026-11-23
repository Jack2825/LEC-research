"""Panel D: compare the two within-passage H1N1 responses in existing data.

Input: primary-only normalized matrix and metadata created for Panel C.
Run with the workspace interpreter; no source or previous panel is overwritten:
  .venv/Scripts/python.exe -B Cell_Painting_Lec_P3_P9_figures/script/panel_D_response_analysis.py
For a new run, provide a new --output-dir inside the project's result folder.
"""
from pathlib import Path
from figure_paths import panel_dir, resolve_result_path, assert_preserved
import argparse
import hashlib
import json
import os
import platform
import sys
sys.dont_write_bytecode=True
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(variable,'1')
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t
import matplotlib
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'result'
C_DATA=RESULT/'Panel_C/panel_C_data'
os.environ['MPLCONFIGDIR']=str(RESULT/'.matplotlib_cache')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pymupdf

CONFIG={'seed':20260918,'wild_bootstrap_draws':19999,'bootstrap_batch_size':512,
        'primary_summary':'mean','sensitivity_summaries':['median','q90'],
        'primary_feature_tests':98,'primary_omnibus_tests':5,'fdr':.05,
        'interaction_direction':'(P9-H1N1 - P9-Control) - (P3-H1N1 - P3-Control)',
        'variance_tolerance':1e-24}
FAMILIES=['Whole-cell morphology','Nuclear morphology / intensity','ER','Mitochondria']
GROUPS=['P3-Control','P3-H1N1','P9-Control','P9-H1N1']
FAMILY_COLORS=['#818a94','#9988a4','#71a38c','#be9972']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                     'svg.fonttype':'none','savefig.facecolor':'white','axes.linewidth':.8,
                     'xtick.labelsize':9,'ytick.labelsize':9})


def sha(path):
    with resolve_result_path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(value,path):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def bh(values):
    values=np.asarray(values,float);result=np.full(values.shape,np.nan)
    finite=np.flatnonzero(np.isfinite(values));order=finite[np.argsort(values[finite],kind='stable')]
    if len(order):
        adjusted=values[order]*len(order)/np.arange(1,len(order)+1)
        result[order]=np.minimum.accumulate(adjusted[::-1])[::-1].clip(0,1)
    return result


def family(feature):
    if feature.startswith('ERWNuclei__AreaShape_'):return FAMILIES[0]
    if feature.startswith(('nuclei__AreaShape_','nuclei__Intensity_')):return FAMILIES[1]
    if feature.startswith('ERWONuclei__Intensity_'):return FAMILIES[2]
    if feature.startswith('mitochondriaWONuclei__Intensity_'):return FAMILIES[3]
    raise ValueError('Unmapped retained feature: '+feature)


def preserve_existing(out):
    return {str(p.resolve()):sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and '.matplotlib_cache' not in p.parts and '__pycache__' not in p.parts
            and p.resolve()!=Path(__file__).resolve()
            and not (p.parent.resolve()==out and p.name.startswith('panel_D'))
            and not p.resolve().is_relative_to(out/'panel_D_data')}


def verify_preserved(manifest):
    for name,digest in manifest.items():
        assert_preserved(name, digest)


def group_parts(y,well_meta):
    indices=[np.flatnonzero(well_meta.panel_group.eq(g).to_numpy()) for g in GROUPS]
    counts=np.array([len(ix) for ix in indices])
    means=np.stack([y[ix].mean(axis=0) for ix in indices])
    variances=np.stack([y[ix].var(axis=0,ddof=1) for ix in indices])
    return indices,counts,means,variances


def contrast_estimates(y,well_meta):
    indices,n,means,variances=group_parts(y,well_meta)
    delta3=means[1]-means[0];delta9=means[3]-means[2]
    interaction=delta9-delta3
    terms=variances/n[:,None]
    se2=terms.sum(axis=0)
    valid=se2>CONFIG['variance_tolerance']
    se=np.sqrt(se2)
    denominator=(terms**2/(n[:,None]-1)).sum(axis=0)
    df=np.divide(se2**2,denominator,out=np.full_like(se2,np.nan),where=denominator>0)
    statistic=np.divide(interaction,se,out=np.full_like(se,np.nan),where=valid)
    p=2*student_t.sf(np.abs(statistic),df)
    critical=student_t.ppf(.975,df)
    return {'delta3':delta3,'delta9':delta9,'interaction':interaction,'group_means':means,
            'se':se,'se2':se2,'df':df,'t':statistic,'p':p,'valid':valid,
            'ci_low':interaction-critical*se,'ci_high':interaction+critical*se,
            'group_indices':indices,'group_counts':n,'group_variances':variances}


def wild_reference(y,well_meta,est,family_indices,seed):
    passage=well_meta.passage.eq('P9').to_numpy(float)
    treatment=well_meta.plasma.eq('H1N1').to_numpy(float)
    full=np.column_stack([np.ones(len(y)),passage,treatment,passage*treatment])
    reduced=full[:,:3]
    beta=np.linalg.lstsq(full,y,rcond=None)[0]
    np.testing.assert_allclose(beta[3],est['interaction'],atol=1e-12,rtol=1e-10)
    full_inverse=np.linalg.pinv(full)
    full_leverage=np.einsum('ij,ji->i',full,full_inverse)
    residual_full=y-full@beta
    hc2_se2=np.sum(full_inverse[3,:,None]**2*residual_full**2/(1-full_leverage[:,None]),axis=0)
    np.testing.assert_allclose(hc2_se2,est['se2'],atol=1e-12,rtol=1e-10)
    fitted0=reduced@np.linalg.lstsq(reduced,y,rcond=None)[0]
    leverage=np.einsum('ij,ji->i',reduced,np.linalg.pinv(reduced))
    assert np.allclose(leverage,.15)
    residual=(y-fitted0)/np.sqrt(1-leverage[:,None])
    # The contrast is orthogonal to the reduced-model main effects.
    coeff=np.zeros(len(y))
    for sign,ix in zip([1,-1,-1,1],est['group_indices']):coeff[ix]=sign/len(ix)
    assert np.allclose(coeff@reduced,0,atol=1e-14)
    assert np.allclose(coeff@fitted0,0,atol=1e-12)
    groups=[np.arange(y.shape[1])]+family_indices
    observed=np.array([np.mean(est['t'][ix]**2) for ix in groups])
    count=CONFIG['wild_bootstrap_draws'];rng=np.random.default_rng(seed)
    references=np.empty((count,len(groups)));zero_variance_values=0
    for start in range(0,count,CONFIG['bootstrap_batch_size']):
        size=min(CONFIG['bootstrap_batch_size'],count-start)
        signs=rng.integers(0,2,size=(size,len(y)),dtype=np.int8)*2-1
        error=signs[:,:,None]*residual[None,:,:]
        effect=np.einsum('i,bif->bf',coeff,error,optimize=True)
        variances=sum(error[:,ix,:].var(axis=1,ddof=1)/len(ix) for ix in est['group_indices'])
        positive=variances>CONFIG['variance_tolerance']
        zero_variance_values+=int((~positive).sum())
        squared_t=np.divide(effect**2,variances,out=np.full_like(effect,np.inf),where=positive)
        squared_t[(~positive)&(np.abs(effect)<=1e-12)]=0
        for j,ix in enumerate(groups):references[start:start+size,j]=squared_t[:,ix].mean(axis=1)
    exceed=(references>=observed[None,:]-1e-12*np.maximum(1,observed)[None,:]).sum(axis=0)
    p=(exceed+1)/(count+1)
    return observed,references,p,zero_variance_values


def analyse_summary(y,well_meta,features,summary,seed):
    est=contrast_estimates(y,well_meta)
    if not est['valid'].all():
        raise ValueError('A feature has no estimable between-well variance; document before proceeding: '+
                         ', '.join(np.array(features)[~est['valid']]))
    family_indices=[np.array([i for i,f in enumerate(features) if family(f)==fam]) for fam in FAMILIES]
    loo=[]
    for i in range(len(well_meta)):
        use=np.arange(len(well_meta))!=i
        loo.append(contrast_estimates(y[use],well_meta.loc[use].reset_index(drop=True))['interaction'])
    loo=np.stack(loo)
    primary=summary=='mean'
    features_table=pd.DataFrame({'summary':summary,'analysis_role':'primary' if primary else 'sensitivity',
        'original_feature':features,'family':[family(f) for f in features],
        'P3_Control_mean_well_profile':est['group_means'][0],
        'P3_H1N1_mean_well_profile':est['group_means'][1],
        'P9_Control_mean_well_profile':est['group_means'][2],
        'P9_H1N1_mean_well_profile':est['group_means'][3],
        'delta_P3':est['delta3'],'delta_P9':est['delta9'],
        'interaction_P9_minus_P3':est['interaction'],
        'interaction_SE':est['se'],'Welch_Satterthwaite_df':est['df'],
        'interaction_t':est['t'],'p_interaction':est['p'],
        'interaction_CI95_low_unadjusted':est['ci_low'],
        'interaction_CI95_high_unadjusted':est['ci_high'],
        'LOO_interaction_min':loo.min(axis=0),'LOO_interaction_max':loo.max(axis=0),
        'LOO_sign_agreement_fraction':(np.sign(loo)==np.sign(est['interaction'])[None,:]).mean(axis=0),
        'n_P3_Control':5,'n_P3_H1N1':5,'n_P9_Control':5,'n_P9_H1N1':5,
        'effect_units':'common robust-scaled feature units',
        'p_method':'two-sided Welch-Satterthwaite contrast; model-based approximation',
        'p_is_distribution_free_exact':False})
    observed,reference,probabilities,zeros=wild_reference(y,well_meta,est,family_indices,seed)
    omnibus=[];loo_rows=[]
    for j,(name,indices) in enumerate(zip(['Global']+FAMILIES,[np.arange(len(features))]+family_indices)):
        d3=est['delta3'][indices];d9=est['delta9'][indices];di=est['interaction'][indices]
        norms=np.sqrt(np.mean(loo[:,indices]**2,axis=1))
        denominator=np.linalg.norm(d3)*np.linalg.norm(d9)
        cosine=float(d3@d9/denominator) if denominator>0 else np.nan
        omnibus.append({'summary':summary,'analysis_role':'primary' if primary else 'sensitivity',
                        'family':name,'n_features':len(indices),
                        'P3_response_RMS':float(np.sqrt(np.mean(d3**2))),
                        'P9_response_RMS':float(np.sqrt(np.mean(d9**2))),
                        'interaction_RMS':float(np.sqrt(np.mean(di**2))),
                        'response_vector_cosine':cosine,
                        'interaction_RMS_LOO_min':float(norms.min()),
                        'interaction_RMS_LOO_max':float(norms.max()),
                        'mean_squared_studentized_interaction':observed[j],
                        'p_interaction':probabilities[j],
                        'p_Monte_Carlo_SE_approx':float(np.sqrt(probabilities[j]*(1-probabilities[j])/(CONFIG['wild_bootstrap_draws']+1))),
                        'wild_bootstrap_draws':CONFIG['wild_bootstrap_draws'],'seed':seed,
                        'p_method':'restricted HC2 wild bootstrap; Monte Carlo approximation',
                        'p_is_distribution_free_exact':False,
                        'LOO_range_is_confidence_interval':False})
        for well,value in zip(well_meta.well_id,norms):
            loo_rows.append({'summary':summary,'family':name,'omitted_well':well,
                             'interaction_RMS':value,'preprocessing_refitted':False})
    null=pd.DataFrame(reference,columns=['Global']+FAMILIES)
    null.insert(0,'draw',np.arange(len(null)));null.insert(0,'summary',summary)
    return features_table,pd.DataFrame(omnibus),pd.DataFrame(loo_rows),null,zeros


def format_p(p):
    return f'{p:.4f}' if p<.01 else f'{p:.3f}'


def render_main(out,feature_table,omnibus,summary='mean'):
    f=feature_table.loc[feature_table.summary.eq(summary)]
    o=omnibus.loc[omnibus.summary.eq(summary)].set_index('family')
    effects=f[['delta_P3','delta_P9','interaction_P9_minus_P3']].to_numpy().T
    # A single limit covers the primary and both prespecified sensitivity plots.
    maximum=float(np.max(np.abs(feature_table[['delta_P3','delta_P9','interaction_P9_minus_P3']].to_numpy())))
    norm=TwoSlopeNorm(vmin=-maximum,vcenter=0,vmax=maximum)
    fig=plt.figure(figsize=(12.2,4.8),facecolor='white')
    fig.text(.018,.970,'D',fontsize=21,fontweight='bold',va='top')
    fig.text(.054,.963,'Passage-dependent H1N1 response',fontsize=15,fontweight='bold',va='top')
    fig.text(.054,.885,'ΔP3 = P3-H1N1 − P3-Control     |     ΔP9 = P9-H1N1 − P9-Control',fontsize=10)
    fig.text(.054,.833,'Response difference: I = ΔP9 − ΔP3',fontsize=10,fontweight='bold')
    centers=[]
    for j,fam in enumerate(FAMILIES):
        x=.112+j*.137
        ax=fig.add_axes([x,.325,.127,.330])
        part=f.loc[f.family.eq(fam),['delta_P3','delta_P9','interaction_P9_minus_P3']].to_numpy().T
        art=ax.imshow(part,cmap='RdBu_r',norm=norm,aspect='auto',interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([0,1,2],labels=['ΔP3','ΔP9','I = ΔP9 − ΔP3'] if j==0 else ['','',''])
        ax.tick_params(length=0,labelsize=9)
        for edge in ax.spines.values():edge.set_linewidth(.6);edge.set_color('#aaaaaa')
        titles=['Whole-cell\nmorphology','Nuclear morphology\n/ intensity','ER','Mitochondria*']
        ax.set_title(titles[j]+f'\n({part.shape[1]} features)',fontsize=9.5,pad=9)
        for y in [.5,1.5]:ax.axhline(y,color='white',linewidth=1)
        centers.append(x+.0635)
    color_ax=fig.add_axes([.176,.204,.390,.024])
    bar=fig.colorbar(art,cax=color_ax,orientation='horizontal')
    bar.set_label('Signed response / response difference (robust-scaled units)',fontsize=8.5,labelpad=4)
    bar.ax.tick_params(labelsize=8,length=2)
    fig.text(.112,.274,'Every retained feature is shown; columns are grouped by family.',fontsize=8.5,color='#626d76')
    ar=fig.add_axes([.786,.285,.123,.395])
    names=['Global']+FAMILIES
    labels=['Global','Whole-cell','Nuclear','ER','Mitochondria*']
    max_x=0
    for j,name in enumerate(names):
        y=4-j;row=o.loc[name]
        color='#152431' if j==0 else FAMILY_COLORS[j-1]
        ar.plot([row.interaction_RMS_LOO_min,row.interaction_RMS_LOO_max],[y,y],color=color,lw=1.5)
        ar.scatter([row.interaction_RMS],[y],marker='D',s=36,c=color,zorder=3)
        ar.text(1.06,y,f'q={format_p(row.q_BH)}',transform=ar.get_yaxis_transform(),fontsize=8,va='center')
        max_x=max(max_x,row.interaction_RMS_LOO_max,row.interaction_RMS)
    ar.set(xlim=(0,max_x*1.13),ylim=(-.55,4.55),yticks=range(4,-1,-1),yticklabels=labels)
    ar.set_xlabel('RMS of I',fontsize=9)
    ar.set_title('Difference between\nresponses',fontsize=10,fontweight='bold',pad=13)
    ar.spines[['top','right']].set_visible(False);ar.tick_params(axis='y',length=0,labelsize=8.5)
    ar.locator_params(axis='x',nbins=3)
    correction='5 primary tests' if summary=='mean' else '10 sensitivity tests'
    fig.text(.737,.158,'Bars: leave-one-well-out range\nq: interaction test, BH across '+correction,
             fontsize=8,color='#626d76',va='top')
    note='Primary: mean profile per well · 5 wells per group' if summary=='mean' else 'Sensitivity: '+('median' if summary=='median' else '90th percentile')+' profile per well · 5 wells per group'
    fig.text(.054,.072,note,fontsize=9,color='#626d76')
    fig.text(.054,.025,'*Mitochondria follows the exported feature annotation; CY5/TRITC channel identity remains unresolved.',fontsize=8,color='#626d76')
    stem='panel_D' if summary=='mean' else 'panel_D.sensitivity_'+summary
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'{stem}.{ext}',dpi=600)
    fig.savefig(out/f'{stem}.preview.png',dpi=180)
    plt.close(fig)


def render_full_features(out,feature_table):
    f=feature_table.loc[feature_table.summary.eq('mean')].reset_index(drop=True)
    values=f[['delta_P3','delta_P9','interaction_P9_minus_P3']].to_numpy()
    limit=float(np.max(np.abs(feature_table[['delta_P3','delta_P9','interaction_P9_minus_P3']].to_numpy())))
    fig=plt.figure(figsize=(11.8,19.8),facecolor='white')
    ax=fig.add_axes([.53,.06,.235,.88])
    art=ax.imshow(values,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto',interpolation='none')
    ax.set_xticks([0,1,2],labels=['ΔP3','ΔP9','I = ΔP9 − ΔP3'])
    ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,labelsize=8)
    labels=[]
    for row in f.itertuples():
        feature=row.original_feature.split('__',1)[1]
        suffix=feature.removeprefix('AreaShape_').removeprefix('Intensity_')
        labels.append(suffix)
    ax.set_yticks(range(len(f)),labels=labels)
    for name in FAMILIES:
        indices=np.flatnonzero(f.family.eq(name).to_numpy())
        lo,hi=indices[0],indices[-1]
        ax.axhline(lo-.5,color='black',lw=.8)
        fig.text(.035,.94-((lo+hi)/2+.5)/len(f)*.88,
                 name.replace(' / ',' /\n')+('*' if name=='Mitochondria' else ''),
                 fontsize=10,fontweight='bold',va='center')
    for i,row in enumerate(f.itertuples()):
        ax.text(3.12,i,f'{row.p_interaction:.5g}',fontsize=7,va='center')
        ax.text(4.10,i,f'{row.q_BH:.5g}',fontsize=7,va='center')
    ax.text(3.12,-1.6,'P (interaction)',fontsize=8)
    ax.text(4.10,-1.6,'BH q (98)',fontsize=8)
    fig.text(.035,.977,'Panel D: all 98 primary feature responses',fontsize=15,fontweight='bold')
    fig.text(.035,.955,'Same ordering and values as the compact main heatmap; full CellProfiler names are in the statistics CSV.',fontsize=9)
    cax=fig.add_axes([.54,.027,.23,.009]);bar=fig.colorbar(art,cax=cax,orientation='horizontal')
    bar.set_label('Signed robust-scaled units',fontsize=8);bar.ax.tick_params(labelsize=7)
    fig.text(.035,.018,'*Source feature annotation; CY5/TRITC channel identity remains unresolved.',fontsize=8)
    fig.savefig(out/'panel_D.full_feature_heatmap.pdf')
    fig.savefig(out/'panel_D.full_feature_heatmap.png',dpi=200)
    plt.close(fig)


def write_report(out,feature_table,omnibus,mapping,validation):
    primary=omnibus.loc[omnibus.summary.eq('mean')]
    count=(feature_table.summary.eq('mean') & feature_table.q_BH.lt(.05)).sum()
    report='''# Panel D — Passage-dependent H1N1 response

## Scope and source files

Only P3-Control, P3-H1N1, P9-Control and P9-H1N1 enter this analysis. The source is `panel_C_data/normalized_features.npz`, with the linked `cell_metadata.csv`, `panel_C_feature_QC.csv`, and previous response vectors used for identity checks. The primary-only preprocessing from Panel C is reused unchanged: 98 features from 1,626 cells in 20 wells. There are five wells per group. No VEGF-C or UC2288 observations enter any fit, display, test or sensitivity analysis.

This is existing-data computational analysis. Observed measurements and group assignments are preserved. Bootstrap draws are null-reference calculations on fitted well residuals, not invented biological observations. No source image or existing Panel A/B/C output is overwritten.

## Family definitions fixed before testing

| Family | CellProfiler name rule | Retained features |
|---|---|---:|
| Whole-cell morphology | ERWNuclei__AreaShape_* | 40 |
| Nuclear morphology / intensity | nuclei__AreaShape_* or nuclei__Intensity_* | 43 (37 morphology, 6 intensity) |
| ER | ERWONuclei__Intensity_* | 5 |
| Mitochondria* | mitochondriaWONuclei__Intensity_* | 10 |

The whole-cell measurements come from the ER-derived secondary object mask. The ER and mitochondria families here contain intensity measurements; the export does not establish mitochondrial network morphology or organelle function. Family membership uses names only. Features are neither selected nor reordered using treatment effects, P values, or an expected mechanism. The original identifiers and deterministic display order are exported in `panel_D.feature_families.csv`.

*The mitochondria label follows the exported object/measurement names. Fig. 2E and the expanded pipeline disagree on CY5/TRITC channel annotation. This is unresolved; no numerical measurements are relabelled or corrected in this analysis. The label is marked in the figures and is not an independently validated biological annotation.

## Common normalization and well profiles

The Panel C representation was fitted using only these four groups. Each of 20 wells has equal total weight during common scaling. Unbounded nonnegative features use log1p(x / equal-well positive median); bounded shape ratios remain linear. The common center is the weighted median and the scale is 1.4826 weighted MAD. All 98 features use this scale in the current run. No group-specific scaling, additional filtering, clipping, or cell deletion occurs in Panel D. The 90 edge-flagged cells remain present.

The primary measurement y_wj is the arithmetic mean of normalized feature j across cells in well w. Each well contributes one vector, regardless of its cell count. Group means then average the five well vectors equally. Median and within-well 90th-percentile profiles are prespecified sensitivity summaries computed from the same normalized cells and same 98 features. They are not substituted for the primary mean analysis based on results.

## Response and interaction definitions

For each feature j:

    delta_P3,j = mean(P3-H1N1 wells)_j - mean(P3-Control wells)_j
    delta_P9,j = mean(P9-H1N1 wells)_j - mean(P9-Control wells)_j
    I_j       = delta_P9,j - delta_P3,j

I_j > 0 means the signed H1N1-associated change is more positive in P9; it does not necessarily mean the magnitude of the P9 response is larger. Opposite directions can produce a large interaction even when the two response norms are equal. These effects are common robust-scaled feature units, not raw-intensity fold changes.

The fitted feature-level model is:

    y_wj = beta0,j + betaP,j * 1(P9) + betaH,j * 1(H1N1)
           + betaPH,j * 1(P9) * 1(H1N1) + error_wj

Here betaH,j = delta_P3,j and betaPH,j = I_j. Baseline passage differences and a shared H1N1 response remain in the model. Numerical checks verify betaPH,j equals the four-group response contrast, and that mean-response vectors match the values already exported for Panel C. The null is I_j=0, allowing nonzero main effects.

## Feature-level inference

Groups are treated as independent and are allowed different variances. For the four groups g, let s_gj^2 be sample variance across the five well profiles. Then:

    SE_j^2 = sum_g(s_gj^2 / n_g)
    t_j = I_j / SE_j
    df_j = [sum_g(s_gj^2/n_g)]^2 /
           sum_g[(s_gj^2/n_g)^2 / (n_g-1)]

The two-sided P value uses the Student t distribution with this Welch–Satterthwaite degrees-of-freedom approximation. This standard error is also checked against the HC2 sandwich standard error of the saturated 2×2 model. The 95% interval is I_j ± t_(.975,df_j) × SE_j and is unadjusted for multiplicity, conditional on the fitted representation. Every feature has an estimable well-level variance in this run.

Full numerical P values are reported rather than significance stars. They are model-based approximations, not mathematically exact distribution-free randomization P values. Five wells per group and nuisance passage/common-response effects do not justify such an exactness claim. No feature-level P value is derived by treating individual cells as independent replicates.

## Family/global multivariate summaries and tests

For a family F containing m features, the descriptive response magnitudes are:

    RMS_P3(F) = sqrt(mean_j_in_F(delta_P3,j^2))
    RMS_P9(F) = sqrt(mean_j_in_F(delta_P9,j^2))
    RMS_I(F)  = sqrt(mean_j_in_F(I_j^2))

The global summary uses all 98 features with equal feature weight. Family-specific summaries prevent the larger shape panels from concealing the smaller intensity panels. A signed feature average is not used because it could cancel opposing response components. Response-vector cosine is also exported descriptively; its direction is uncertain when responses are small.

Omnibus testing uses Q_F = mean_j_in_F(t_j^2), the mean squared studentized interaction. Q accounts for marginal well-level uncertainty; the bootstrap reference retains cross-feature dependence. The displayed RMS_I is the interpretable effect magnitude; its omnibus P value is computed from Q, not from an assumption that the raw norm is unbiased or normally distributed.

A reduced model with intercept, passage and H1N1 main effects is fitted under zero interaction. Restricted residuals are adjusted by sqrt(1-h_ii) (HC2; h_ii=0.15 for this balanced design). In each of 19,999 draws, each entire well residual vector receives one shared +1/-1 Rademacher multiplier. The additive fitted values are retained. The interaction and its groupwise variance estimate are recomputed for every draw. Using the same sign across all features of a well preserves their joint structure.

The P value is (1 + number of null Q values at least as large as observed Q)/(19,999 + 1). The primary seed is 20260918; median and q90 sensitivities use the next two seeds. These are approximate restricted wild-bootstrap P values, not exact permutation tests. Monte Carlo standard errors and all Q reference values are saved. The treatment labels in the observed tables are never changed. Ordinary treatment-label shuffling would generally target a different null when a common H1N1 response is allowed.

Each plotted horizontal range omits one of the 20 wells in turn and recomputes RMS_I with the representation held fixed. These are influence/sensitivity ranges, not confidence intervals. Small positive norms alone are not evidence of an interaction because a nonnegative norm has a sampling-noise floor.

## Multiplicity fixed before fitting

- Primary feature interactions: BH across all 98 features together.
- Primary omnibus interactions: BH across Global plus the four families (five tests).
- Sensitivity feature interactions: BH across all 98 × 2 summaries = 196 tests together.
- Sensitivity omnibus interactions: BH across all five × two summaries = ten tests together.

The primary and sensitivity correction families are explicit and are not pooled into a claimed experiment-wide FDR guarantee. Conclusions are not selected using the smallest P value across summaries. Feature-level 95% intervals and omit-one-well ranges are not simultaneous confidence bounds. FDR threshold: q<0.05.

## Primary compartment/global results

| Family | Features | RMS response P3 | RMS response P9 | RMS interaction | Numerical P | BH q |
|---|---:|---:|---:|---:|---:|---:|
'''
    for row in primary.itertuples():
        report+=f'| {row.family} | {row.n_features} | {row.P3_response_RMS:.4f} | {row.P9_response_RMS:.4f} | {row.interaction_RMS:.4f} | {row.p_interaction:.6f} | {row.q_BH:.6f} |\n'
    passed=primary.loc[primary.q_BH.lt(.05),'family'].tolist()
    report+=f'\nPrimary feature interactions passing BH q<0.05: **{count}/98**. '
    report+=('No primary family/global interaction passes its five-test BH correction.\n' if not passed else 'Primary omnibus tests passing their five-test BH correction: '+', '.join(passed)+'.\n')
    sensitivity=omnibus.loc[omnibus.summary.ne('mean')]
    report+='\n## Prespecified distribution-sensitive results\n\n| Well summary | Family | RMS interaction | Numerical P | BH q (10) |\n|---|---|---:|---:|---:|\n'
    for row in sensitivity.itertuples():
        report+=f'| {row.summary} | {row.family} | {row.interaction_RMS:.4f} | {row.p_interaction:.6f} | {row.q_BH:.6f} |\n'
    report+='''
These comparisons assess the same response difference using the typical cell and upper-tail summaries of each well. They use the same dataset and provide sensitivity analysis, not independent validation. Upper-tail estimates are less precise in the low-cell-count P3 wells.

## How to read the figure

Each heatmap column is one retained feature, with all 98 features shown. The four family blocks have equal display width; their feature counts are printed, so column widths differ between blocks. Rows are delta_P3, delta_P9 and I=delta_P9−delta_P3. All blocks/rows, the primary plot, both sensitivity plots and the full-feature appendix share one symmetric color scale with zero at white; no effects are clipped. Red is positive and blue negative. The maximum absolute effect across the three prespecified summaries sets the common color bound; it is not a significance threshold. Features have a fixed name-based order and are not clustered or ranked by a desirable effect.

The right-hand summary shows RMS_I and omit-one-well ranges. Its q values test the interaction vectors and use the declared five-test family. Sensitivity figures use their separate ten-test q values. The full-height feature heatmap and statistics workbook identify every feature and provide numerical interaction P/q values and effect intervals.

## Limits

Wells are treated as independent as requested, but donor/plasma-pool identities have not been verified. Passage is confounded with acquisition date and processing settings; fixed plate rows/columns are confounded with conditions. A response interaction can therefore include technical passage-by-condition effects. Models and resampling cannot establish causality or eliminate those design limitations. Only the exported first field per well is included. Five wells per group limit variance estimation and precision; nonsignificance is not evidence that the responses are identical. The CY5/TRITC annotation issue also limits organelle-specific interpretation.

## Reproducibility and method references

`panel_D_response_analysis.py` reads existing normalized data, freezes the family map, writes the preanalysis plan, computes all prespecified summaries/tests, and generates plots. Source and previous-panel hashes, effect/standard-error identity checks, bootstrap seeds, all output tables and software versions are retained. No original biological data or source artwork is edited.

For a new run, supply a new `--output-dir` inside `result/`. Completed Panel D outputs are protected from default overwriting. `--render-only` updates derived artwork without recalculating or changing numerical tables.

Restricted wild-bootstrap background: [Davidson & Flachaire, 2008](https://www.sciencedirect.com/science/article/pii/S0304407608000833). Nuisance effects and exchangeability in multivariate imaging inference: [Winkler et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/). These sources motivate the inference framework; the exact project choices and limitations are stated above.
'''
    (out/'panel_D.methods.md').write_text(report,encoding='utf-8')
    legend='''**D. Passage-dependent H1N1 response.** Primary analysis of 98 commonly normalized CellProfiler measurements from five wells per group (P3-Control, P3-H1N1, P9-Control and P9-H1N1). The heatmaps show the equal-well H1N1-minus-Control response for P3 and P9 and their difference, I=delta_P9−delta_P3. Features are grouped by predefined CellProfiler name families; all retained features are displayed with one shared, zero-centered color scale. Red indicates positive and blue negative effects in robust-scaled units. The right-hand plot shows the root-mean-square interaction across each family and globally, with leave-one-well-out sensitivity ranges (not confidence intervals). Omnibus interaction q values use restricted wild-bootstrap P values and BH correction across the five primary tests. Numerical feature-level effects, Welch–Satterthwaite P values and BH q values across 98 tests are supplied in the statistics table. The comparison tests the difference between H1N1 responses, allowing baseline passage differences and a common H1N1 effect. *Mitochondria follows the exported measurement annotation; the CY5/TRITC discrepancy is unresolved. No mechanistic identity is inferred.\n'''
    (out/'panel_D.legend.md').write_text(legend,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=panel_dir('D'))
    parser.add_argument('--render-only',action='store_true')
    parser.add_argument('--resume-incomplete',action='store_true')
    args=parser.parse_args();out=args.output_dir.resolve()
    if not out.is_relative_to(RESULT.resolve()):raise ValueError('Outputs must remain inside result/.')
    out.mkdir(parents=True,exist_ok=True)
    if args.render_only:
        ft_path=out/'panel_D_statistics.csv';ot_path=out/'panel_D_statistics.compartments.csv'
        before={p:sha(p) for p in [ft_path,ot_path]}
        ft=pd.read_csv(ft_path,keep_default_na=False);ot=pd.read_csv(ot_path,keep_default_na=False)
        for summary in ['mean','median','q90']:render_main(out,ft,ot,summary)
        render_full_features(out,ft)
        verify_preserved(before)
        vpath=out/'panel_D.validation.json'
        if vpath.exists():
            v=json.loads(vpath.read_text(encoding='utf-8'));v['script_sha256']=sha(__file__)
            v['figure_regenerated_without_numerical_changes']=True
            v['common_heatmap_color_limit_all_summaries']=float(np.max(np.abs(ft[['delta_P3','delta_P9','interaction_P9_minus_P3']].to_numpy())))
            write_report(out,ft,ot,pd.read_csv(out/'panel_D.feature_families.csv'),v)
            dump(v,vpath)
        return
    if (out/'panel_D.validation.json').exists():raise FileExistsError('Preserving completed Panel D; choose a new output directory.')
    if (out/'panel_D_statistics.csv').exists() and not args.resume_incomplete:
        raise FileExistsError('Incomplete output exists. Explicit --resume-incomplete is required.')
    preserved=preserve_existing(out)
    data=out/'panel_D_data';data.mkdir(exist_ok=True)
    meta=pd.read_csv(C_DATA/'cell_metadata.csv',keep_default_na=False)
    qc=pd.read_csv(RESULT/'Panel_C/panel_C_feature_QC.csv',keep_default_na=False)
    with np.load(C_DATA/'normalized_features.npz',allow_pickle=False) as a:
        z=a['standardized'];features=a['features'].tolist()
        assert np.array_equal(a['cell_ids'],meta.cell_id.to_numpy(str))
    assert meta.intervention.eq('None').all() and set(meta.panel_group)==set(GROUPS)
    assert len(meta)==1626 and meta.well_id.nunique()==20 and len(features)==98
    assert set(features)==set(qc.loc[qc.retained,'original_feature'])
    order=sorted(range(len(features)),key=lambda j:(FAMILIES.index(family(features[j])),features[j]))
    z=z[:,order];features=[features[j] for j in order]
    mapping=pd.DataFrame({'original_feature':features,'family':[family(f) for f in features],
                          'display_order':np.arange(1,len(features)+1),
                          'classification_rule':'CellProfiler object prefix plus AreaShape/Intensity measurement prefix',
                          'selected_using_effects':False})
    assert mapping.groupby('family').size().reindex(FAMILIES).tolist()==[40,43,5,10]
    mapping.to_csv(out/'panel_D.feature_families.csv',index=False)
    well_meta=meta.groupby('well_id',sort=True).agg(passage=('passage','first'),plasma=('plasma','first'),
        intervention=('intervention','first'),panel_group=('panel_group','first'),
        physical_well=('physical_well','first'),n_cells=('cell_id','size')).reset_index()
    assert well_meta.groupby('panel_group').size().eq(5).all()
    well_meta.to_csv(data/'well_metadata.csv',index=False)
    plan={**CONFIG,'feature_counts':mapping.family.value_counts().to_dict(),
          'groups':well_meta.groupby('panel_group').size().to_dict(),'cells':1626,'wells':20,
          'null_model':'intercept + passage + H1N1, without passage:H1N1 interaction',
          'full_model':'intercept + passage + H1N1 + passage:H1N1',
          'multiple_testing_families':{'primary_features':98,'primary_omnibus':5,'sensitivity_features':196,'sensitivity_omnibus':10},
          'numerical_p_values_are_approximate':True,'normalization_reused_unchanged_from_panel_C':True,
          'source_channel_annotation_unresolved':True}
    dump(plan,out/'panel_D.preanalysis_plan.json')
    pd.DataFrame([{'source_or_preserved_file':p,'sha256':h} for p,h in preserved.items()]).to_csv(data/'preservation_manifest.csv',index=False)
    original_manifest=pd.read_csv(C_DATA/'source_and_preservation_manifest.csv',keep_default_na=False)
    for row in original_manifest.itertuples():assert_preserved(row.source_file, row.sha256)
    positions=[np.flatnonzero(meta.well_id.eq(well).to_numpy()) for well in well_meta.well_id]
    summaries={'mean':np.stack([z[ix].mean(axis=0) for ix in positions]),
               'median':np.stack([np.median(z[ix],axis=0) for ix in positions]),
               'q90':np.stack([np.quantile(z[ix],.9,axis=0,method='linear') for ix in positions])}
    prior_wells=pd.read_csv(C_DATA/'well_feature_centroids.csv',keep_default_na=False,float_precision='round_trip').set_index('well_id')
    np.testing.assert_allclose(summaries['mean'],prior_wells.loc[well_meta.well_id,features].to_numpy(),atol=1e-12,rtol=1e-10)
    parts=[];family_parts=[];loo_parts=[];null_parts=[];bootstrap_zero_counts={}
    for index,(summary,y) in enumerate(summaries.items()):
        print('Fitting',summary,'well profiles and 19999 restricted wild-bootstrap draws',flush=True)
        pd.concat([well_meta,pd.DataFrame(y,columns=features)],axis=1).to_csv(data/f'well_profiles_{summary}.csv',index=False)
        ft,ot,lt,nt,zeros=analyse_summary(y,well_meta,features,summary,CONFIG['seed']+index)
        parts.append(ft);family_parts.append(ot);loo_parts.append(lt);null_parts.append(nt)
        bootstrap_zero_counts[summary]=zeros
    ft=pd.concat(parts,ignore_index=True);ot=pd.concat(family_parts,ignore_index=True)
    for role,count_features,count_omnibus in [('primary',98,5),('sensitivity',196,10)]:
        ix=ft.analysis_role.eq(role);jx=ot.analysis_role.eq(role)
        assert ix.sum()==count_features and jx.sum()==count_omnibus
        ft.loc[ix,'q_BH']=bh(ft.loc[ix,'p_interaction'])
        ft.loc[ix,'FDR_family']=f'{role}: {count_features} feature interactions'
        ot.loc[jx,'q_BH']=bh(ot.loc[jx,'p_interaction'])
        ot.loc[jx,'FDR_family']=f'{role}: {count_omnibus} omnibus interactions'
    prior_response=pd.read_csv(C_DATA/'within_passage_response_vectors.csv').set_index('feature')
    mean=ft.loc[ft.summary.eq('mean')].set_index('original_feature')
    np.testing.assert_allclose(mean.delta_P3,prior_response.loc[mean.index,'P3_H1N1_minus_Control'],atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(mean.delta_P9,prior_response.loc[mean.index,'P9_H1N1_minus_Control'],atol=1e-12,rtol=1e-10)
    ft.to_csv(out/'panel_D_statistics.csv',index=False)
    ot.to_csv(out/'panel_D_statistics.compartments.csv',index=False)
    pd.concat(loo_parts,ignore_index=True).to_csv(data/'leave_one_well_out.csv',index=False)
    pd.concat(null_parts,ignore_index=True).to_csv(data/'wild_bootstrap_null_Q.csv.gz',index=False)
    with pd.ExcelWriter(out/'panel_D_statistics.xlsx',engine='openpyxl') as writer:
        ft.loc[ft.analysis_role.eq('primary')].to_excel(writer,sheet_name='Primary_features',index=False)
        ot.loc[ot.analysis_role.eq('primary')].to_excel(writer,sheet_name='Primary_compartments',index=False)
        ft.loc[ft.analysis_role.eq('sensitivity')].to_excel(writer,sheet_name='Sensitivity_features',index=False)
        ot.loc[ot.analysis_role.eq('sensitivity')].to_excel(writer,sheet_name='Sensitivity_compartments',index=False)
        mapping.to_excel(writer,sheet_name='Feature_families',index=False)
        well_meta.to_excel(writer,sheet_name='Well_metadata',index=False)
    for summary in summaries:render_main(out,ft,ot,summary)
    render_full_features(out,ft)
    verify_preserved(preserved)
    for row in original_manifest.itertuples():assert_preserved(row.source_file, row.sha256)
    with pymupdf.open(out/'panel_D.pdf') as doc:
        assert len(doc)==1
        text=doc[0].get_text()
        assert 'Passage-dependent H1N1 response' in text
        assert 'VEGF' not in text and 'UC2288' not in text
    validation={'primary_cells':1626,'wells':20,'wells_per_group':5,'features':98,
        'family_counts':mapping.family.value_counts().to_dict(),
        'model_interaction_equals_response_difference':True,
        'model_HC2_SE_equals_four_group_Welch_SE':True,
        'mean_profiles_and_responses_match_panel_C':True,
        'previous_project_files_preserved':len(preserved),
        'sources_and_panels_A_B_C_unchanged':True,
        'bootstrap_zero_variance_values':bootstrap_zero_counts,
        'primary_features_q_below_005':int((ft.analysis_role.eq('primary') & ft.q_BH.lt(.05)).sum()),
        'primary_omnibus_q_below_005':int((ot.analysis_role.eq('primary') & ot.q_BH.lt(.05)).sum()),
        'sensitivity_features_q_below_005':int((ft.analysis_role.eq('sensitivity') & ft.q_BH.lt(.05)).sum()),
        'sensitivity_omnibus_q_below_005':int((ot.analysis_role.eq('sensitivity') & ot.q_BH.lt(.05)).sum()),
        'primary_omnibus_results':ot.loc[ot.analysis_role.eq('primary')].to_dict('records'),
        'distribution_free_exact_p_values_claimed':False,'channel_annotation_unresolved':True,
        'script_sha256':sha(__file__),
        'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                    'scipy':scipy.__version__,'matplotlib':matplotlib.__version__}}
    write_report(out,ft,ot,mapping,validation)
    dump(validation,out/'panel_D.validation.json')
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':main()
