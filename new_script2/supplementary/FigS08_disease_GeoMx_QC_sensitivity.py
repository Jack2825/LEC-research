"""Build Supplementary Figure S8 from completed GeoMx analyses.

This display layer does not rerun normalization, probe QC, models, permutation
tests, or biological analyses. Run without arguments for panels A-I and the
combined figure, or use --panel A. Main Figure 6 is not modified.
"""
from pathlib import Path
import argparse
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'new_result2_spatial_disease'
QCD=BASE/'13_1_spatial_disease_QC'
LECD=BASE/'13_2_disease_LEC_reference'
PROGD=BASE/'13_3_prespecified_DCM_program_validation'
FINALD=BASE/'13_4_final_DCM_spatial_validation'
MAIN=ROOT/'new_result2/disease/Fig06_spatial_disease_BMP_ACVR2A/source_data'
OUT=ROOT/'new_result2/supplementary/FigS08_disease_GeoMx_QC_sensitivity'
MARKERS=['CCL21','LYVE1','PDPN','FLT4','PROX1']
HITS=['SMAD7','ID1','ID2','ENG','ID3']
SETS=['primary','strict_anatomical_LV','pre_LVAD','strict_noncardiac_controls','strict_controls_block1']
SET_LABELS=['Primary','Strict LV','Pre-LVAD','Strict controls','Strict controls\n+ block 1']
GROUP_ORDER=['Ligands','Receptors / co-receptors','SMAD / readout / feedback']
NAMES=dict(zip('ABCDEFGHI',['overall_dataset_AOI_QC','LV_vessel_AOI_rationale','LEC_like_AOI_criteria',
 'lymphatic_marker_coverage','probe_detection_quality','pathway_assay_coverage',
 'cohort_sensitivity_analyses','selected_component_robustness','ACVR2A_probe_limitation']))
TITLES=dict(zip('ABCDEFGHI',['Overall GeoMx dataset and AOI quality control','Rationale for selecting LV vessel AOIs',
 'Criteria for identifying LEC-like AOIs','Lymphatic-marker expression and coverage',
 'Probe detection and quality assessment','BMP/TGF-β/activin pathway assay coverage',
 'Strict-LV, pre-LVAD and control sensitivities','Robustness of selected pathway components',
 'Why ACVR2A could not be evaluated']))
NAVY,TEAL,CORAL,GREY='#2C5F8A','#2B8C87','#BA6556','#65727C'
CONTROL,DCM='#4C78A8','#E07A5F'
MANIFEST=[]


def track(path,panels,purpose):
    path=Path(path)
    MANIFEST.append(dict(panels=panels,source_file=path.relative_to(ROOT).as_posix(),purpose=purpose,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def read(path,panels,purpose):
    track(path,panels,purpose)
    t=pd.read_csv(path,low_memory=False)
    t['source_file']=Path(path).relative_to(ROOT).as_posix()
    return t


def build():
    for path,panels in [(QCD/'13_1_method_and_limitations.txt','A;B;E'),
      (LECD/'13_2_method_and_interpretation_notes.txt','B;C;D'),
      (PROGD/'13_3_method_and_interpretation_notes.txt','C;G;H'),
      (FINALD/'13_4_READ_ME_FINAL_VALIDATION.txt','H;I')]:
        track(path,panels,'Original methods and interpretation limitations')
    run=read(QCD/'13_1_run_summary.csv','A;B;E','Dataset, AOI, donor, probe-QC and validation counts')
    valid=read(QCD/'13_1_input_validation.csv','A;E','DCC and matrix-integrity audit')
    meta=read(QCD/'13_1_AOI_metadata_and_QC.csv','A;B;E','All 178 AOIs with clinical, segment and QC metadata')
    design=read(QCD/'13_1_design_summary.csv','A;B;G','Saved AOI/donor cohort definitions')
    primary=read(QCD/'13_1_primary_LV_vessel_Control_DCMP_design.csv','B;C;D','Primary 16 Control/DCMP LV vessel AOIs')
    evidence=read(LECD/'13_2_all_primary_AOI_two_of_five_LEC_evidence.csv','C','Marker-count selection evidence for all 16 primary AOIs')
    selected=read(LECD/'13_2_selected_AOI_LEC_evidence.csv','C;D','Seven LEC-marker-selected LV vessel AOIs')
    readiness=read(LECD/'13_2_LEC_selection_readiness.csv','C;H','Selected AOI and donor counts for all saved criteria')
    marker=read(QCD/'13_1_mentor_target_detection_by_AOI.csv.gz','D;I','Raw count, AOI-specific LOQ and normalized expression for targets')
    audit=read(MAIN/'Fig06_assay_audit.csv','E;F;I','Complete 47-gene assay audit and raw detection/QC availability')
    detection=read(MAIN/'Fig06C_pathway_detection.csv','F','Complete pathway detection/effect table for primary and selected cohorts')
    sens=read(MAIN/'Fig06F_pathway_sensitivity.csv','G;H','All 16 detection-eligible genes across five prespecified cohorts')
    decisions=read(FINALD/'13_4_FINAL_VALIDATION_DECISIONS.csv','H','Canonical validation constraints and recommended wording')
    rule_audit=read(FINALD/'13_4_evidence_rule_audit.csv','H','Original evidence checks for descriptive program assessment')

    assert len(meta)==178 and meta.PID.nunique()==44 and not meta.qc_hard_fail.any()
    assert len(primary)==16 and primary.PID.nunique()==11
    assert len(evidence)==16 and evidence.LEC_selected_two_of_five.sum()==7
    assert evidence.loc[evidence.LEC_selected_two_of_five,'PID'].nunique()==5
    assert set(sens.sensitivity_set)==set(SETS)
    assert not decisions.formal_validation_allowed.any()

    run_long=pd.concat([run.melt(var_name='metric',value_name='value').assign(record_type='run_summary'),
                        valid.melt(var_name='metric',value_name='value').assign(record_type='input_validation')],ignore_index=True)
    a=pd.concat([run_long,meta.assign(record_type='AOI_QC')],ignore_index=True)
    stages=[]
    stage_masks=[('All AOIs',pd.Series(True,index=meta.index)),('LV AOIs',meta.Ventricle.eq('LV')),
      ('LV vessel AOIs',meta.Ventricle.eq('LV')&meta['Segment.type'].eq('vessel')),
      ('Control/DCMP LV vessel',meta.analysis_primary_paper_LV_vessel)]
    for order,(stage,mask) in enumerate(stage_masks):
        stages.append(dict(stage_order=order,stage=stage,n_AOIs=int(mask.sum()),n_donors=meta.loc[mask,'PID'].nunique()))
    b=pd.concat([pd.DataFrame(stages).assign(record_type='selection_stage'),
                 design.assign(record_type='design_summary'),primary.assign(record_type='primary_AOI')],ignore_index=True)
    evidence=evidence.copy();evidence['display_AOI']=[f'AOI {i+1}' for i in range(len(evidence))]
    c=pd.concat([evidence.assign(record_type='primary_AOI_marker_count'),
                 readiness.assign(record_type='selection_readiness')],ignore_index=True)
    target=marker.loc[marker.analysis_primary_paper_LV_vessel&marker.gene.isin(MARKERS)].copy()
    selected_ids=set(selected.aoi_id)
    rows=[]
    for scope,mask in [('Primary Control',target.Clinical_phenotype_LV.eq('Control')),
                       ('Primary DCM',target.Clinical_phenotype_LV.eq('DCMP')),
                       ('Selected Control',target.aoi_id.isin(selected_ids)&target.Clinical_phenotype_LV.eq('Control')),
                       ('Selected DCM',target.aoi_id.isin(selected_ids)&target.Clinical_phenotype_LV.eq('DCMP'))]:
        sub=target.loc[mask]
        for gene,g in sub.groupby('gene'):
            rows.append(dict(scope=scope,gene=gene,n_AOIs=g.aoi_id.nunique(),n_AOIs_above_LOQ=int(g.above_LOQ.sum()),
                percent_AOIs_above_LOQ=100*g.above_LOQ.mean(),mean_log2_Q3_CPM_plus1=g.log2_upper_quartile_CPM_plus1.mean(),
                source_file=g.source_file.iloc[0]))
    d=pd.concat([pd.DataFrame(rows).assign(record_type='marker_summary'),
                 target.assign(record_type='primary_AOI_marker_detail')],ignore_index=True)
    audit=audit.copy();audit['raw_primary_percent']=100*audit.n_raw_primary_AOIs_above_LOQ/16
    e=pd.concat([audit.assign(record_type='pathway_probe_audit'),
                 meta[['aoi_id','Segment.type','Clinical_phenotype_LV','post_probe_QC_library_size','n_post_probe_QC_genes_above_LOQ',
                       'fraction_post_probe_QC_genes_above_LOQ','qc_status','source_file']].assign(record_type='AOI_detection_QC')],ignore_index=True)
    f=pd.concat([audit.assign(record_type='assay_audit'),detection.assign(record_type='cohort_detection')],ignore_index=True)
    h=sens.loc[sens.gene.isin(HITS)&sens.sensitivity_set.isin(SETS[:4])].copy()
    h['same_positive_direction']=h.DCMP_minus_Control_donor_mean_log2_Q3_CPM.gt(0)
    h['robustness_code']=np.select([h['FDR_lt_0.05'],h.same_positive_direction],[2,1],default=0)
    hh=pd.concat([h.assign(record_type='selected_component_sensitivity'),
                  rule_audit.assign(record_type='original_program_evidence_audit'),
                  decisions.assign(record_type='canonical_decision')],ignore_index=True)
    raw_path=ROOT/'resources/spatial_transcriptomic/GSE271676_RAW/InitialDataset.csv'
    track(raw_path,'I','Original raw GeoMx matrix; ACVR2A single-probe counts only')
    raw=pd.read_csv(raw_path,low_memory=False)
    raw_acv=raw.loc[raw.HUGOSymbol.eq('ACVR2A')]
    assert len(raw_acv)==1
    aoi_columns=[c for c in raw.columns if c in set(meta.aoi_id)]
    acv=pd.DataFrame({'aoi_id':aoi_columns,
                      'raw_initial_target_count':[raw_acv[c].iloc[0] for c in aoi_columns]})
    acv=acv.merge(meta[['aoi_id','PID','Clinical_phenotype_LV','Segment.type',
                        'analysis_primary_paper_LV_vessel','LOQ_per_probe_geomean_times_geoSD2']],on='aoi_id',how='left')
    acv=acv.rename(columns={'LOQ_per_probe_geomean_times_geoSD2':'target_LOQ'})
    acv['above_LOQ']=acv.raw_initial_target_count>acv.target_LOQ
    acv['count_to_LOQ_ratio']=acv.raw_initial_target_count/acv.target_LOQ
    acv['source_file']=raw_path.relative_to(ROOT).as_posix()
    assert len(acv)==178 and int(acv.above_LOQ.sum())==1
    assert int(acv.loc[acv.analysis_primary_paper_LV_vessel,'above_LOQ'].sum())==0
    assert int(audit.loc[audit.gene.eq('ACVR2A'),'n_raw_AOIs_above_LOQ_all_178'].iloc[0])==1
    i=pd.concat([acv.assign(record_type='ACVR2A_AOI_audit'),
                 audit.loc[audit.gene.isin(['ACVR2A','ACVR2B'])].assign(record_type='ACVR2_family_assay_audit')],ignore_index=True)
    return dict(A=a,B=b,C=c,D=d,E=e,F=f,G=sens,H=hh,I=i),dict(run=run.iloc[0],valid=valid.iloc[0],meta=meta,
        primary=primary,evidence=evidence,marker_rows=pd.DataFrame(rows),audit=audit,sens=sens,h=h,acv=acv)


def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':11,
      'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
      'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':GREY,
      'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})


def head(fig,l,sub):
    fig.text(.025,.98,l,fontsize=19,fontweight='bold',va='top')
    fig.text(.085,.977,TITLES[l],fontsize=14,fontweight='bold',va='top')
    fig.text(.085,.922,sub,fontsize=10,color=GREY,va='top')
def foot(fig,s):fig.text(.035,.025,s,fontsize=9,color=GREY,va='bottom',linespacing=1.35)
def clean(ax,axis='x'):ax.grid(axis=axis,color='#E3E7EA',lw=.6);ax.set_axisbelow(True)


def draw_a(fig,t,x):
    head(fig,'A','GSE271676: 178 segmented AOIs from 44 donors; all passed the saved AOI-level QC gate')
    cards=[('178','AOIs'),('44','donors'),('178 / 178','valid DCC files'),('18,676','endogenous probes'),('12,800','post-QC genes')]
    for k,(v,l) in enumerate(cards):
        ax=fig.add_axes([.055+k*.185,.68,.16,.14]);ax.axis('off');ax.add_patch(plt.Rectangle((0,0),1,1,facecolor='#F2F5F7',edgecolor='#B9C4CC'))
        ax.text(.5,.61,v,ha='center',va='center',fontsize=17,fontweight='bold',color=NAVY);ax.text(.5,.25,l,ha='center',fontsize=9,color=GREY)
    order=['vessel','Trp','fb','Full ROI'];m=x['meta'];data=[m.loc[m['Segment.type'].eq(g)] for g in order]
    for pos,col,label,trans in [([.10,.24,.37,.31],'post_probe_QC_library_size','Post-QC library size',lambda z:np.log10(z)),
                                ([.58,.24,.37,.31],'fraction_post_probe_QC_genes_above_LOQ','Genes above LOQ (%)',lambda z:100*z)]:
        ax=fig.add_axes(pos);vals=[trans(g[col].dropna()) for g in data]
        bp=ax.boxplot(vals,vert=False,patch_artist=True,showfliers=False,medianprops={'color':NAVY})
        for box in bp['boxes']:box.set_facecolor('#D9E4EA');box.set_edgecolor('#9AAAB5')
        ax.set_yticks(range(1,5),order);ax.set_xlabel('log10 '+label if 'library' in col else label);clean(ax)
    foot(fig,'AOIs are segmented regions, not cells. Vessel AOIs are CD31-enriched endothelial regions, not purified LECs.\nQC used raw DCC validation, negative-probe background, post-probe-QC library size and above-LOQ gene coverage.\nNo AOI hard failures occurred; repeated AOIs within a donor are not independent biological replicates.')


def draw_b(fig,t,x):
    head(fig,'B','Anatomical, segmentation and phenotype restrictions define the 16-AOI primary pathway cohort')
    s=t['B'].loc[t['B'].record_type.eq('selection_stage')]
    ax=fig.add_axes([.25,.27,.29,.55]);y=np.arange(len(s));ax.barh(y,s.n_AOIs,color=[GREY,'#A9BAC4',TEAL,NAVY],height=.55)
    ax.set_yticks(y,s.stage);ax.invert_yaxis();ax.set_xlabel('AOIs retained');clean(ax)
    for yi,r in enumerate(s.itertuples()):ax.text(r.n_AOIs+3,yi,f'{int(r.n_AOIs)} AOIs · {int(r.n_donors)} donors',va='center',fontsize=9)
    ax=fig.add_axes([.63,.35,.28,.38]);p=x['primary'];groups=['Control','DCMP'];xx=np.arange(2)
    a=[p.Clinical_phenotype_LV.eq(g).sum() for g in groups];d=[p.loc[p.Clinical_phenotype_LV.eq(g),'PID'].nunique() for g in groups]
    ax.bar(xx-.18,a,.36,color=[CONTROL,DCM],label='AOIs');ax.bar(xx+.18,d,.36,color=[CONTROL,DCM],alpha=.45,hatch='//',label='donors')
    ax.set_xticks(xx,['Control','DCM']);ax.set_ylabel('Count');ax.legend(frameon=False);clean(ax,'y')
    foot(fig,'Primary screen: Ventricle=LV, vessel segment, Clinical_phenotype_LV in Control/DCMP.\nThis matches cardiac anatomy, endothelial segmentation and the DCM contrast; it is not an expression-selected cohort.\nSix Control AOIs from four donors and ten DCM AOIs from seven donors permit donor-level testing in the pathway screen.')


def draw_c(fig,t,x):
    head(fig,'C','Prespecified rule: ≥2 of five canonical lymphatic markers above each AOI-specific LOQ')
    e=x['evidence'].sort_values(['Clinical_phenotype_LV','n_LEC_markers_above_LOQ','PID'],ascending=[True,False,True]).reset_index(drop=True)
    y=np.arange(16);ax=fig.add_axes([.30,.20,.45,.63])
    colors=[CONTROL if g=='Control' else DCM for g in e.Clinical_phenotype_LV]
    ax.hlines(y,0,e.n_LEC_markers_above_LOQ,color='#D7DEE2',lw=1.4);ax.scatter(e.n_LEC_markers_above_LOQ,y,c=colors,s=45,zorder=3)
    ax.axvline(2,color=TEAL,ls='--',lw=1.4);ax.set_xlim(-.15,5.2);ax.set_xticks(range(6));ax.set_ylim(15.6,-.6)
    labels=[f'{r.display_AOI} | {"Control" if r.Clinical_phenotype_LV=="Control" else "DCM"} | {r.PID}' for r in e.itertuples()]
    ax.set_yticks(y,labels);ax.set_xlabel('Markers above AOI-specific LOQ');clean(ax)
    for yi,r in enumerate(e.itertuples()):
        if r.LEC_selected_two_of_five:ax.text(5.12,yi,'selected',ha='right',va='center',fontsize=8,color=TEAL)
    fig.legend(handles=[Line2D([],[],marker='o',ls='none',color=CONTROL,label='Control'),Line2D([],[],marker='o',ls='none',color=DCM,label='DCM')],
               frameon=False,ncol=2,loc='center',bbox_to_anchor=(.54,.87))
    foot(fig,'Markers: CCL21, LYVE1, PDPN, FLT4 and PROX1. Detection means target count above the AOI-specific GeoMx LOQ.\nSeven of 16 AOIs were selected: four Control AOIs from two donors and three DCM AOIs from three donors.\nTwo markers in a mixed vessel AOI do not demonstrate co-expression in one cell; no formal selected-cohort test is permitted.')


def draw_d(fig,t,x):
    head(fig,'D','Dot area: percentage above LOQ; color: mean log2 upper-quartile CPM + 1')
    d=x['marker_rows'];scopes=['Primary Control','Primary DCM','Selected Control','Selected DCM']
    mat=d.set_index(['scope','gene']).loc[pd.MultiIndex.from_product([scopes,MARKERS])].reset_index()
    ax=fig.add_axes([.30,.25,.50,.55]);norm=Normalize(mat.mean_log2_Q3_CPM_plus1.min(),mat.mean_log2_Q3_CPM_plus1.max())
    im=None
    for yi,scope in enumerate(scopes):
        z=mat.loc[mat.scope.eq(scope)].set_index('gene').loc[MARKERS]
        im=ax.scatter(range(5),np.repeat(yi,5),s=3*z.percent_AOIs_above_LOQ,c=z.mean_log2_Q3_CPM_plus1,
                      cmap='Blues',norm=norm,edgecolors='#8799A5',lw=.5)
    ax.set_xticks(range(5),MARKERS);ax.set_yticks(range(4),[f'{s}\n(n={int(mat.loc[mat.scope.eq(s),"n_AOIs"].iloc[0])} AOIs)' for s in scopes]);ax.set_xlim(-.6,4.6);ax.set_ylim(3.6,-.6);ax.tick_params(length=0)
    cb=fig.colorbar(im,cax=fig.add_axes([.31,.13,.48,.018]),orientation='horizontal');cb.set_label('Mean log2 upper-quartile CPM + 1')
    hs=[plt.scatter([],[],s=3*v,color='#BED3DF',edgecolors=GREY,lw=.4) for v in [25,50,75,100]]
    fig.legend(hs,['25%','50%','75%','100%'],loc='center',bbox_to_anchor=(.55,.865),ncol=4,frameon=False,fontsize=9)
    foot(fig,'Detection uses raw target count > AOI-specific LOQ; expression is the mean normalized value including below-LOQ AOIs.\nSelected-group marker enrichment is expected because these five markers define selection; it is not independent validation.\nCCL21 also appears in the homeostatic endpoint, creating explicit selection–outcome overlap.')


def draw_e(fig,t,x):
    head(fig,'E','Raw detection is shown before the authors’ 1%-probe-QC expression filter')
    a=x['audit'];ax=fig.add_axes([.13,.24,.52,.58])
    keep=a.present_after_authors_1pct_probe_QC.astype(bool)
    ax.scatter(a.percent_raw_AOIs_above_LOQ_all_178,a.raw_primary_percent,c=np.where(keep,NAVY,'#B9BFC4'),s=35,alpha=.85)
    ax.axvline(1,color=CORAL,ls='--',lw=1);ax.set_xlabel('Raw AOIs above LOQ, all 178 (%)');ax.set_ylabel('Raw primary AOIs above LOQ (%)');clean(ax,'both')
    fig.legend(handles=[Line2D([],[],marker='o',ls='none',color=NAVY,label='Post-QC available'),
                        Line2D([],[],marker='o',ls='none',color='#B9BFC4',label='Post-QC unavailable')],
               frameon=False,ncol=2,loc='center',bbox_to_anchor=(.39,.85),fontsize=9)
    m=x['meta'];ax=fig.add_axes([.76,.31,.18,.43]);vals=[np.log10(m.loc[m['Segment.type'].eq(g),'post_probe_QC_library_size']) for g in ['vessel','Trp','fb','Full ROI']]
    ax.boxplot(vals,tick_labels=['vessel','Trp','fb','Full ROI'],showfliers=False,patch_artist=True,boxprops={'facecolor':'#D9E4EA'},medianprops={'color':NAVY})
    ax.tick_params(axis='x',rotation=45);ax.set_ylabel('log10 post-QC library');clean(ax,'y');ax.set_title('AOI library distributions',fontsize=10)
    fig.text(.69,.20,'Post-QC unavailable:\nBMP7, BMP10, ACVR1C, ACVR2A,\nACVR2B, BMPR1B and NOG',fontsize=9,color=GREY,linespacing=1.45)
    foot(fig,'Each left point is one of 47 prespecified pathway genes. Gray points failed the authors’ probe-QC expression matrix.\nThe dashed 1% line contextualizes sparse raw detection; probe retention is read from the supplied filtered matrix.\nUnavailable genes remain in the audit but are not reintroduced for expression inference.')


def draw_f(fig,t,x):
    head(fig,'F','Complete 47-gene audit: raw above-LOQ coverage and post-probe-QC availability')
    a=x['audit'];positions=[.08,.40,.72];widths=[.23,.25,.25]
    for pos,w,group in zip(positions,widths,GROUP_ORDER):
        g=a.loc[a.pathway_group.eq(group)].copy().sort_values('gene');n=len(g)
        ax=fig.add_axes([pos,.18,w,.66]);matrix=g[['percent_raw_AOIs_above_LOQ_all_178','raw_primary_percent']].to_numpy()
        im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100,aspect='auto',interpolation='nearest')
        ax.set_yticks(range(n),g.gene);ax.set_xticks([0,1],['All 178 AOIs','Primary 16'],rotation=25,ha='right');ax.tick_params(length=0)
        ax.set_title(group,fontsize=10,pad=12)
        for yi,r in enumerate(g.itertuples()):
            ax.text(1.72,yi,'Yes' if r.present_after_authors_1pct_probe_QC else 'No',ha='center',va='center',fontsize=7,
                    color=TEAL if r.present_after_authors_1pct_probe_QC else CORAL)
        ax.text(1.72,-1.4,'Post-QC',ha='center',fontsize=8);ax.set_xlim(-.5,2.05)
    cb=fig.colorbar(im,cax=fig.add_axes([.25,.105,.50,.018]),orientation='horizontal');cb.set_label('Raw AOIs above AOI-specific LOQ (%)')
    foot(fig,'Post-QC Yes: available in the authors’ 12,800-gene matrix; No: unavailable and excluded from expression testing.\nCoverage is assay detectability, not differential expression; all 47 predefined family components remain in the source CSV.\nThe focused main-figure heatmap is not repeated here.')


def draw_g(fig,t,x):
    head(fig,'G','All 16 detection-eligible pathway genes across five prespecified donor-level cohorts')
    s=x['sens'];genes=list(s.loc[s.sensitivity_set.eq('primary')].sort_values('DCMP_minus_Control_donor_mean_log2_Q3_CPM',ascending=False).gene)
    mat=s.pivot(index='gene',columns='sensitivity_set',values='DCMP_minus_Control_donor_mean_log2_Q3_CPM').reindex(index=genes,columns=SETS)
    q=s.pivot(index='gene',columns='sensitivity_set',values='BH_FDR_q').reindex(index=genes,columns=SETS)
    ax=fig.add_axes([.25,.18,.50,.65]);v=np.nanmax(abs(mat.to_numpy()));im=ax.imshow(mat,cmap='RdBu_r',vmin=-v,vmax=v,aspect='auto',interpolation='nearest')
    ax.set_yticks(range(len(genes)),genes);ax.set_xticks(range(5),SET_LABELS,rotation=35,ha='right');ax.tick_params(length=0)
    for i in range(len(genes)):
        for j in range(5):
            if q.iloc[i,j]<.05:ax.text(j,i,'•',ha='center',va='center',fontsize=16,color='white' if abs(mat.iloc[i,j])>.55*v else '#21313A')
    cb=fig.colorbar(im,cax=fig.add_axes([.80,.27,.018,.45]));cb.set_label('DCM − Control donor-mean\nlog2(Q3 CPM + 1)')
    foot(fig,'Dot: BH FDR <0.05 within that 16-gene cohort analysis. Biological replicate is donor; repeated AOIs are averaged.\nCohorts: primary 4/7 Control/DCM donors; strict LV 4/6; pre-LVAD 4/6; strict controls 3/7; block-1 strict controls 3/4.\nThe final cohort has limited permutation resolution. This sensitivity panel does not replace the prespecified primary analysis.')


def draw_h(fig,t,x):
    head(fig,'H','Direction and BH-FDR support in the four major cohorts used for the main-figure robustness label')
    h=x['h'];sets=SETS[:4]
    code=h.pivot(index='gene',columns='sensitivity_set',values='robustness_code').reindex(index=HITS,columns=sets)
    q=h.pivot(index='gene',columns='sensitivity_set',values='BH_FDR_q').reindex(index=HITS,columns=sets)
    ax=fig.add_axes([.28,.27,.58,.55]);ax.imshow(code,cmap=ListedColormap(['#D8DDE0','#F1DFC8','#C8E2DD']),vmin=0,vmax=2,aspect='auto')
    ax.set_yticks(range(5),HITS);ax.set_xticks(range(4),SET_LABELS[:4]);ax.tick_params(length=0)
    for i in range(5):
        for j in range(4):ax.text(j,i,f'q={q.iloc[i,j]:.3f}',ha='center',va='center',fontsize=9)
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True);ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',length=0)
    hs=[plt.Rectangle((0,0),1,1,color=c) for c in ['#C8E2DD','#F1DFC8','#D8DDE0']]
    fig.legend(hs,['Positive effect + q<0.05','Positive effect, q≥0.05','Opposite/zero effect'],loc='center',bbox_to_anchor=(.57,.17),ncol=3,frameon=False,fontsize=9)
    foot(fig,'All five selected components have positive DCM−Control effects across major cohorts; FDR support varies by component/set.\n“Robust across major sensitivities” requires q<0.05 in primary, strict LV, pre-LVAD and strict-control analyses.\nThis is the detailed source for that label, not a new composite score or separate validation experiment.')


def draw_i(fig,t,x):
    head(fig,'I','ACVR2A was assayed, but raw detection was below the authors’ expression-QC requirement')
    a=x['acv'].sort_values('count_to_LOQ_ratio').reset_index(drop=True);ax=fig.add_axes([.11,.25,.55,.55])
    colors=np.where(a.analysis_primary_paper_LV_vessel,NAVY,'#AEB9C0')
    ax.scatter(range(1,179),a.count_to_LOQ_ratio,c=colors,s=15,alpha=.8);ax.axhline(1,color=CORAL,ls='--',lw=1.2)
    ax.set_yscale('log');ax.set_xlabel('AOIs ranked by raw count / LOQ');ax.set_ylabel('ACVR2A raw count / AOI-specific LOQ');clean(ax,'both')
    r=x['audit'].loc[x['audit'].gene.eq('ACVR2A')].iloc[0]
    ax=fig.add_axes([.72,.30,.24,.45]);ax.axis('off')
    cards=[('1 / 178','all AOIs above LOQ'),('0 / 16','primary AOIs above LOQ'),('0.56%','raw detection rate'),('Excluded','authors’ 1%-probe-QC matrix')]
    for k,(v,l) in enumerate(cards):
        y=.87-k*.23;ax.text(.5,y,v,ha='center',fontsize=16,fontweight='bold',color=NAVY if k<3 else CORAL);ax.text(.5,y-.08,l,ha='center',fontsize=9,color=GREY)
    foot(fig,'Dashed line: raw count equals the AOI-specific LOQ. Blue: primary 16 LV vessel AOIs; gray: all other AOIs.\nACVR2A had one endogenous probe and was present in the raw panel, but it was absent from the authors’ post-QC matrix.\nRaw counts are used only to document assay failure. ACVR2A expression, group effects and receptor validation are not evaluated.')


DRAW=dict(zip('ABCDEFGHI',[draw_a,draw_b,draw_c,draw_d,draw_e,draw_f,draw_g,draw_h,draw_i]))
SIZES={**{x:(12,8) for x in 'ABCDEGHI'},'F':(12,10)}


def bounds(fig):
    fig.canvas.draw();ren=fig.canvas.get_renderer();bb=fig.bbox;bad=[];hidden=set()
    for ax in fig.axes:
        for axis,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            lo,hi=sorted(lim)
            for tick in [*axis.get_major_ticks(),*axis.get_minor_ticks()]:
                if tick.get_loc()<lo-1e-9 or tick.get_loc()>hi+1e-9:hidden.update([tick.label1,tick.label2])
    for a in fig.findobj(match=Text):
        if a in hidden or not a.get_visible() or not a.get_text().strip():continue
        b=a.get_window_extent(ren)
        if b.width>0 and b.height>0 and (b.x0<bb.x0-2 or b.y0<bb.y0-2 or b.x1>bb.x1+2 or b.y1>bb.y1+2):bad.append(a.get_text())
    if bad:raise RuntimeError('Text outside canvas: '+repr(bad))


def save(fig,path):
    bounds(fig)
    for ext in ['pdf','png','jpg']:
        fig.savefig(path.with_suffix('.'+ext),dpi=240,**({'pil_kwargs':{'quality':95}} if ext=='jpg' else {}))
    plt.close(fig)


LEGEND='''Figure S8. Disease GeoMx QC and sensitivity analyses supporting Figure 6.

Scope: completed GSE271676 GeoMx WTA analyses. AOIs are segmented regions, not cells; vessel AOIs are CD31-enriched endothelial regions, not purified LECs. Biological replicate is PID, with repeated AOIs averaged within donor where donor-level analyses were performed. No normalization, probe QC, models, permutations or biological analyses are rerun here.

A. Overall resource and AOI QC. All 178 DCC files passed gzip/header validation and aligned exactly to annotations and post-probe-QC counts. The raw resource contained 18,676 endogenous probes; the authors’ filtered matrix contained 12,800 genes. All 178 AOIs passed the saved primary and sensitivity AOI-QC status. Boxplots show post-probe-QC library size and above-LOQ gene fraction across vessel, transcriptomic-profiling (Trp), fibroblast and full-ROI segments. AOI-level QC does not make repeated AOIs independent donors.
B. Cohort rationale. Sequential restrictions identify LV anatomy, vessel segmentation and the paper-matched Control/DCMP contrast. The primary screen contains 16 AOIs: 6 Control AOIs from 4 donors and 10 DCM AOIs from 7 donors. This cohort is anatomy/segmentation/phenotype defined, not lymphatic-marker selected, and supports the donor-level focused pathway screen.
C. LEC-like AOI definition. An LV vessel AOI is selected if at least two of CCL21, LYVE1, PDPN, FLT4 and PROX1 exceed its AOI-specific LOQ. Seven of 16 primary AOIs pass: 4 Control AOIs from 2 donors and 3 DCM AOIs from 3 donors. The Control group fails the prespecified minimum of three donors; all selected-cohort program results are descriptive. Mixed AOIs do not establish marker co-expression in one LEC.
D. Lymphatic-marker coverage. Dot area gives the percentage of AOIs above LOQ; color is mean log2 upper-quartile CPM+1 including below-LOQ AOIs. Primary and selected Control/DCM groups are shown. Higher marker coverage in selected AOIs is expected by construction. CCL21 is both a selector and a homeostatic endpoint, so that endpoint is not selection-independent.
E. Probe detection and quality assessment. Each point is one of the 47 prespecified BMP/TGF-beta/activin family genes. Axes show raw above-LOQ detection in all 178 AOIs and the 16 primary AOIs. Gray labeled genes are absent from the authors’ 1%-probe-QC expression matrix and are not reintroduced. The right panel provides post-QC AOI-library context by segment.
F. Complete pathway assay coverage. All 47 prespecified ligands, receptors/co-receptors and SMAD/readout/feedback components are shown. Heatmap columns report raw above-LOQ coverage in all 178 and primary 16 AOIs. Check/cross marks report availability in the authors’ post-probe-QC matrix. Coverage is assay detectability, not differential expression; the biological main-figure heatmap is not repeated.
G. Complete saved sensitivity analysis for 16 detection-eligible genes across primary, strict anatomical LV, pre-LVAD, strict noncardiac-control and strict-control/block-1 cohorts. Color is the DCM-minus-Control donor-mean log2(Q3 CPM+1) difference; dots indicate within-analysis-set BH FDR <0.05. Donor counts are 4/7, 4/6, 4/6, 3/7 and 3/4 Control/DCM, respectively. The smallest cohort has limited exact-permutation resolution.
H. Detailed robustness matrix for the five pathway components selected for the main figure (SMAD7, ID1, ID2, ENG and ID3). Cells separate positive-direction effects with q<0.05, positive effects without FDR support, and opposite/zero effects across the four major cohorts. Robustness requires q<0.05 in all four. The matrix documents the main-figure robustness label and is not a new composite or validation experiment.
I. ACVR2A assay limitation. ACVR2A was present in the raw panel with one endogenous probe but exceeded its AOI-specific LOQ in only 1/178 AOIs (0.56%) and 0/16 primary AOIs. It was excluded from the authors’ 1%-probe-QC matrix. Raw counts are shown only to document unavailability and are not rescued for expression inference. Therefore ACVR2A expression, a DCM-Control effect, or direct receptor validation cannot be evaluated in this GeoMx dataset; the tissue results can assess only other detectable BMP/TGF-beta/activin pathway components.

The main Figure 6 pathway heatmap and donor-level biological panels are not duplicated. The seven LEC-marker-selected AOIs remain a descriptive contextual subset, whereas the 16-AOI primary LV-vessel cohort is used for the donor-level pathway screen. Each panel is exported as PDF, PNG and JPG with its source CSV. The input manifest records exact source paths, uses and hashes. Main-figure and upstream files are unchanged.
'''


def main(panel=None):
    style();OUT.mkdir(parents=True,exist_ok=True);t,x=build()
    letters=[panel] if panel else list('ABCDEFGHI')
    for l in letters:
        folder=OUT/f'FigS08{l}_{NAMES[l]}';folder.mkdir(exist_ok=True)
        t[l].to_csv(folder/f'FigS08{l}_source_data.csv',index=False)
        fig=plt.figure(figsize=SIZES[l]);DRAW[l](fig,t,x);save(fig,folder/f'FigS08{l}_{NAMES[l]}')
        print(f'Saved S8{l}: PDF, PNG, JPG and source CSV',flush=True)
    if not panel:
        fig=plt.figure(figsize=(36,26));subs=fig.subfigures(3,3,wspace=.025,hspace=.035)
        for l,sub in zip('ABCDEFGHI',subs.flat):DRAW[l](sub,t,x)
        save(fig,OUT/'FigS08_combined');print('Saved combined S8 in three formats',flush=True)
    pd.DataFrame(MANIFEST).drop_duplicates().to_csv(OUT/'FigS08_input_manifest.csv',index=False)
    (OUT/'FigS08_figure_legend.txt').write_text(LEGEND,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',choices=list('ABCDEFGHI'));main(p.parse_args().panel)
