"""Reproduce S6 from saved results only; no inference or resampling is rerun.

Run without arguments for all panels and the combined figure, or --panel A.
The six small panel entry scripts use this same audited implementation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'new_result2/supplementary/FigS06_extended_disease_SCENIC_MultiNicheNet'
SC = ROOT / 'new_result2_disease/11_SCENIC_analysis'
NN = ROOT / 'new_result2_disease/12_NicheNet_donor_pseudobulk'
DISEASES = ['DCM', 'HCM', 'DoxCM', 'HFpEF']
RUNS = dict(zip(DISEASES, ['dataset01_DCM_vs_NF', 'dataset01_HCM_vs_NF',
                         'dataset08_DoxCM_vs_NF', 'dataset11_HFpEF_vs_NF_min10']))
NAMES = dict(zip('ABCDEF', ['extended_regulons', 'regulon_robustness',
    'extended_ligand_rankings', 'sender_contributions', 'extended_LR_LT_links',
    'cross_disease_and_threshold_sensitivity']))
TITLES = dict(zip('ABCDEF', ['Extended disease-state regulons',
    'Regulon reproducibility across balancing schemes', 'Extended ligand activity rankings',
    'Sender representation beyond selected axes', 'Additional receptor and target links',
    'Cross-disease context and HFpEF sensitivity']))
NAVY, TEAL, UP, DOWN = '#2C5F8A', '#2B8C87', '#BA6556', '#467DA4'
GREY = '#65727C'
COHORT = 'DCM, HCM, DoxCM: ≥25 LECs/donor; HFpEF: exploratory ≥10. Matched NF uses the same cutoff.'
SENDERS = {'Adip1':'Adipocytes', 'EC1_cap':'Capillary ECs', 'EC5_art':'Arterial ECs',
 'EC6_ven':'Venous ECs', 'EC7_endocardial':'Endocardial ECs',
 'LYVE1.IGF1.MP':'LYVE1+IGF1+ macrophages', 'Mast':'Mast cells', 'NC1_glial':'Glial cells',
 'PC1_vent':'Ventricular pericytes', 'SMC1_basic':'SMCs', 'vCM1':'Ventricular cardiomyocytes',
 'CD4.T_naive':'Naive CD4+ T cells'}
MANIFEST = []


def track(path, panels, purpose):
    path = Path(path)
    MANIFEST.append(dict(panels=panels, source_csv=path.relative_to(ROOT).as_posix(),
                         purpose=purpose, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def read(path, panels, purpose):
    track(path, panels, purpose)
    return pd.read_csv(path)


def setup():
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':11, 'axes.titlesize':12, 'axes.labelsize':11, 'xtick.labelsize':10,
        'ytick.labelsize':10, 'pdf.fonttype':42, 'ps.fonttype':42,
        'figure.facecolor':'white', 'savefig.facecolor':'white', 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.edgecolor':GREY, 'axes.linewidth':.7})


def build_data():
    for name in ['11_run_config.json','11_method_and_interpretation_notes.txt']:
        track(SC/name,'A;B','Original SCENIC cohort, balancing, bootstrap and interpretation provenance')
    config=json.loads((SC/'11_run_config.json').read_text(encoding='utf-8'))
    assert config['common_datasets_for_primary_balance']==['dataset01','dataset08']
    assert config['donor_bootstrap_repetitions']==config['dataset_donor_bootstrap_repetitions']==500
    sc = read(SC/'tables/11_SCENIC_state_regulon_full_summary.csv', 'A;B',
              'Full saved SCENIC summary; includes both balancing schemes and existing intervals')
    main = read(ROOT/'new_result2/disease/Fig05A_disease_state_specific_regulons/Fig05A_source_data.csv',
                'A;B', 'Exclude state-regulon pairs already selected in main Figure 5A')
    mainb = read(ROOT/'new_result2/disease/Fig05B_disease_regulon_robustness/Fig05B_source_data.csv',
                 'A;B', 'Exclude state-regulon pairs already selected in main Figure 5B')
    pairs = set(zip(main.target_state, main.regulon)) | set(zip(mainb.LEC_state, mainb.regulon))
    sc['already_in_main'] = [(s,r) in pairs for s,r in zip(sc.LEC_state,sc.regulon)]
    for col in ['donor_bootstrap_RSS_top_frequency','dataset_donor_bootstrap_RSS_top_frequency']:
        assert np.allclose(sc.groupby('LEC_state')[col].sum(),8), 'Saved frequencies must represent a top-eight set'
    eligible = sc.loc[sc.robust_candidate_flag.eq(True) & ~sc.already_in_main]
    selected = eligible.sort_values(['LEC_state','consensus_rank_within_state','regulon']).groupby('LEC_state').head(2)
    assert len(selected) == 18, 'Expected two additional robust candidates for each of nine states'
    chosen = set(zip(selected.LEC_state,selected.regulon))
    sc['plotted'] = [(s,r) in chosen for s,r in zip(sc.LEC_state,sc.regulon)]
    sc['record_type'] = 'full_state_regulon_summary'
    sc['source_csv'] = 'new_result2_disease/11_SCENIC_analysis/tables/11_SCENIC_state_regulon_full_summary.csv'
    heat = []
    for order, row in enumerate(selected.itertuples()):
        for state in [f'state{i}' for i in range(9)]:
            hit = sc.loc[(sc.LEC_state==state)&(sc.regulon==row.regulon)].iloc[0]
            heat.append(dict(record_type='plotted_heatmap_cell', row_order=order,
                target_state=row.LEC_state, regulon=row.regulon, heatmap_state=state,
                zscore=hit.dataset_donor_balanced_mean_AUCell_zscore,
                plotted=True, source_csv=hit.source_csv))
    a = pd.concat([sc, pd.DataFrame(heat)], ignore_index=True)
    main_axes = read(ROOT/'new_result2/disease/Fig05D_disease_communication_modules/Fig05D_source_data.csv',
                     'C;E;F', 'Identify main-figure ligands and ligand-receptor pairs for exclusion')
    activities, edges, links, senders, ligand_context = [], [], [], [], []
    runs = read(NN/'12_run_summary.csv', 'C;D;E;F', 'Verify completed analyses, cutoffs and donor counts')
    for disease, run in RUNS.items():
        base = NN/run/'tables'
        runrow = runs.loc[runs.analysis_id.eq(run)].iloc[0]
        assert runrow.status == 'completed' and runrow.receiver_min_cells == (10 if disease=='HFpEF' else 25)
        meta = dict(disease=disease, receiver_min_LECs=int(runrow.receiver_min_cells),
                    disease_donors=int(runrow.n_disease_donors), NF_donors=int(runrow.n_reference_donors))
        main_d = main_axes.loc[main_axes.disease.eq(disease)]
        main_ligands = set(main_d.ligand)
        main_pairs = set(zip(main_d.ligand,main_d.receptor))
        for direction in ['up','down']:
            p = base/f'12_Sydney_ligand_activities_{direction}.csv'
            t = read(p, 'C', 'Complete ligand activities; rank extra ligands by corrected AUPR')
            t['already_in_main'] = t.test_ligand.isin(main_ligands)
            t['plotted'] = False
            ids = t.loc[~t.already_in_main].sort_values(['aupr_corrected','test_ligand'],ascending=[False,True]).head(5).index
            t.loc[ids,'plotted'] = True
            activities.append(t.assign(**meta, source_csv=p.relative_to(ROOT).as_posix()))
            for kind in ['receptor','target']:
                p = base/f'12_Sydney_ligand_{kind}_links_{direction}.csv'
                t = read(p, 'E', 'All available saved links; select two strongest additional links per direction and type')
                t = t.rename(columns={'from':'ligand','to':'partner','target':'partner'})
                t['already_in_main'] = [(l,r) in main_pairs if kind=='receptor' else False
                                        for l,r in zip(t.ligand,t.partner)]
                t['plotted'] = False
                ids = t.loc[~t.already_in_main].sort_values(['weight','ligand','partner'],ascending=[False,True,True]).drop_duplicates(['ligand','partner']).head(2).index
                t.loc[ids,'plotted'] = True
                links.append(t.assign(**meta, link_type=kind, source_csv=p.relative_to(ROOT).as_posix()))
        p = base/'12_MultiNicheNet_group_prioritization.csv'
        t = read(p, 'D;F', 'Complete priorities; restrict sender summary and ligand ranks to disease-prioritized LEC axes')
        t = t.assign(**meta, source_csv=p.relative_to(ROOT).as_posix())
        t['disease_prioritized'] = t.group.eq(disease)&t.top_group.eq(disease)&t.receiver.eq('LEC_EC8_ln')
        t['top50_for_sender_summary'] = False
        active = t.loc[t.disease_prioritized].sort_values(['prioritization_score','id'],ascending=[False,True])
        assert not active.id.duplicated().any()
        top = active.head(50)
        t.loc[top.index,'top50_for_sender_summary'] = True
        edges.append(t)
        elig = read(base/'12_celltype_eligibility.csv', 'D', 'Distinguish an ineligible sender from an eligible sender with no top-50 axes')
        mapping = read(base/'12_celltype_name_map.csv', 'D', 'Map original cell types to MultiNicheNet names')
        mapping = dict(zip(mapping.analysis_celltype,mapping.celltype_mnn))
        for row in elig.itertuples():
            if row.role=='receiver': continue
            sender = mapping.get(row.analysis_celltype,row.analysis_celltype.replace('+','.'))
            count = int(top.sender.eq(sender).sum())
            senders.append(dict(**meta, sender=sender, eligible=bool(row.eligible),
                top50_axis_count=count, total_selected_axes=len(top),
                share_percent=100*count/len(top) if row.eligible else np.nan,
                eligibility_rule=row.eligibility_rule, source_csv=p.relative_to(ROOT).as_posix()))
        ligand = active.groupby('ligand',as_index=False).prioritization_score.max()
        ligand['priority_percentile'] = ligand.prioritization_score.rank(method='average',pct=True)*100
        ligand['already_in_main'] = ligand.ligand.isin(main_ligands)
        ligand['selected_as_top3_extra'] = False
        ix = ligand.loc[~ligand.already_in_main].sort_values(['prioritization_score','ligand'],ascending=[False,True]).head(3).index
        ligand.loc[ix,'selected_as_top3_extra'] = True
        ligand_context.append(ligand.assign(**meta, source_csv=p.relative_to(ROOT).as_posix()))
    context = pd.concat(ligand_context,ignore_index=True)
    selected_ligands = list(dict.fromkeys(context.loc[context.selected_as_top3_extra].ligand))
    context['plotted'] = context.ligand.isin(selected_ligands)
    p = NN/'12_dataset11_HFpEF_MultiNicheNet_min10_min15_concordance.csv.gz'
    concord = read(p, 'F', 'Saved common-axis sensitivity comparison; plot HFpEF contrast only')
    concord['plotted'] = concord.group.eq('HFpEF') & concord.receiver.eq('LEC_EC8_ln')
    concord['source_csv'] = p.relative_to(ROOT).as_posix()
    details = pd.DataFrame(senders)
    assert np.allclose(details.groupby('disease').share_percent.sum(),100)
    d = pd.concat([pd.concat(edges).assign(record_type='complete_prioritization'),
                   details.assign(record_type='sender_top50_summary')],ignore_index=True)
    f = pd.concat([context.assign(record_type='ligand_context'),
                   concord.assign(record_type='HFpEF_threshold_concordance')],ignore_index=True)
    common=concord.loc[concord.plotted].dropna(subset=['prioritization_score_min10','prioritization_score_min15'])
    summary=dict(record_type='HFpEF_threshold_summary',disease='HFpEF',plotted=True,
                 common_axes=len(common),spearman_rho=spearmanr(common.prioritization_score_min10,
                 common.prioritization_score_min15).statistic,source_csv=p.relative_to(ROOT).as_posix())
    f=pd.concat([f,pd.DataFrame([summary])],ignore_index=True)
    return dict(A=a, B=sc.copy(), C=pd.concat(activities,ignore_index=True), D=d,
                E=pd.concat(links,ignore_index=True), F=f), selected, details, selected_ligands


def header(fig, letter, subtitle):
    fig.text(.025,.975,letter,fontsize=20,fontweight='bold',va='top')
    fig.text(.085,.972,TITLES[letter],fontsize=14,fontweight='bold',va='top')
    fig.text(.085,.923,subtitle,fontsize=10,color=GREY,va='top')


def foot(fig, text):
    fig.text(.04,.025,text,fontsize=9,color=GREY,va='bottom',linespacing=1.35)


def clean(ax):
    ax.grid(axis='x',color='#E6E9EB',lw=.6,zorder=0)
    ax.tick_params(length=3)
    ax.set_axisbelow(True)


def state_label(state):
    n=int(state.replace('state',''))
    return f'State {n}' + (' *' if n in [4,7,8] else '')


def disease_label(disease):
    return disease + (' · ≥10 LECs (exploratory)' if disease=='HFpEF' else ' · ≥25 LECs')


def draw_a(fig, data, selected, *_):
    header(fig,'A','Two additional consensus-ranked robust candidates per state; main-figure pairs excluded')
    ax=fig.add_axes([.27,.18,.64,.65])
    t=data['A'].query("record_type == 'plotted_heatmap_cell'")
    mat=t.pivot(index='row_order',columns='heatmap_state',values='zscore')
    im=ax.imshow(mat,cmap='RdBu_r',vmin=-2.5,vmax=2.5,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(9),[f'{i}' for i in range(9)])
    ax.set_xlabel('LEC state')
    ax.set_yticks(range(len(selected)),[f'{state_label(r.LEC_state)}  |  {r.regulon}' for r in selected.itertuples()])
    for i in np.arange(1.5,17,2): ax.axhline(i,color='white',lw=1.5)
    ax.tick_params(length=0)
    cb=fig.colorbar(im,cax=fig.add_axes([.93,.36,.018,.32]),extend='both')
    cb.set_label('AUCell z-score across states',fontsize=10)
    foot(fig,'Saved strict dataset/donor-balanced activity: datasets 01 and 08.\n* States 4/7: contractile-marker caution; State 8: donor-dominance caution.\nZ-scores describe relative regulon activity, not disease-condition effects.')


def draw_b(fig, data, selected, *_):
    header(fig,'B','Existing 500-replicate bootstrap summaries for the additional regulons in A')
    ax=fig.add_axes([.27,.22,.37,.60]); fr=fig.add_axes([.72,.22,.23,.60])
    y=np.arange(len(selected))
    for offset,prefix,color,label in [(-.14,'dataset_donor_bootstrap_RSS',NAVY,'Dataset + donor balance'),
                                     (.14,'donor_bootstrap_RSS',TEAL,'All-dataset donor balance')]:
        mid=selected[prefix+'_median'].to_numpy(); lo=selected[prefix+'_q025'].to_numpy();hi=selected[prefix+'_q975'].to_numpy()
        ax.errorbar(mid,y+offset,xerr=np.vstack([mid-lo,hi-mid]),fmt='o',color=color,ms=3.8,lw=1,capsize=1.6,label=label)
        fr.scatter(selected[prefix+'_top_frequency']*100,y+offset,s=18,color=color)
    ax.set_yticks(y,[f'{state_label(r.LEC_state)}  |  {r.regulon}' for r in selected.itertuples()])
    fr.set_yticks([])
    for a in [ax,fr]:a.set_ylim(len(y)-.5,-.5);clean(a)
    ax.set_xlabel('RSS median and 95% bootstrap interval');fr.set_xlabel('Top-8 frequency (%)');fr.set_xlim(-4,104);fr.set_xticks([0,50,100])
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.55,.105),frameon=False,ncol=2,fontsize=10)
    foot(fig,'Strict balance: datasets 01/08; all-dataset sensitivity also includes dataset 11.\nTop-8 frequency: fraction of saved replicates ranking the regulon among the top 8 for its state.\n* State cautions as in A. Intervals are not p-values; no new resampling was run.')


def direction_key(fig, y=.877):
    handles=[Line2D([0],[0],color=UP,marker='o',lw=0,label='Disease-upregulated LEC program'),
             Line2D([0],[0],color=DOWN,marker='o',lw=0,label='Disease-downregulated LEC program')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.53,y),ncol=2,frameon=False,fontsize=9)


def draw_c(fig, data, *_):
    header(fig,'C','Top five additional ligands per direction, ranked by corrected AUPR')
    direction_key(fig)
    for i,disease in enumerate(DISEASES):
        ax=fig.add_axes([.16+(i%2)*.48,.53-(i//2)*.35,.30,.26])
        t=data['C'].loc[data['C'].disease.eq(disease)&data['C'].plotted]
        t=t.sort_values(['direction','aupr_corrected'],ascending=[False,False])
        colors=[UP if x=='upregulated' else DOWN for x in t.direction]
        y=np.arange(len(t)); ax.barh(y,t.aupr_corrected,color=colors,height=.68)
        ax.set_yticks(y,t.test_ligand);ax.invert_yaxis();ax.set_title(disease_label(disease),fontsize=11,loc='left',pad=10)
        ax.set_xlim(0,max(t.aupr_corrected)*1.15);ax.set_xlabel('Corrected AUPR');clean(ax)
    foot(fig,COHORT+'\nDirection refers to the receiver gene program, not ligand up/downregulation.\nActivity is model-based ranking evidence; complete saved rankings are in the panel source CSV.')


def draw_d(fig, data, selected, details, *_):
    header(fig,'D','All sender types eligible in at least one disease; summary of the top 50 prioritized axes')
    included=sorted(details.loc[details.eligible].sender.unique(),key=lambda s:SENDERS.get(s,s))
    mat=details.pivot(index='sender',columns='disease',values='share_percent').reindex(index=included,columns=DISEASES)
    ax=fig.add_axes([.37,.23,.49,.59])
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#E4E7E9')
    im=ax.imshow(np.ma.masked_invalid(mat),cmap=cmap,vmin=0,vmax=100,aspect='auto',interpolation='nearest')
    ax.set_yticks(range(len(included)),[SENDERS.get(s,s) for s in included]);ax.set_xticks(range(4),['DCM','HCM','DoxCM','HFpEF*'])
    ax.tick_params(length=0)
    for row in range(len(mat)):
        for col in range(4):
            val=mat.iloc[row,col]
            ax.text(col,row,'—' if pd.isna(val) else f'{val:.0f}',ha='center',va='center',fontsize=10,color='white' if val>65 else '#25313A')
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,len(mat),1),minor=True)
    ax.grid(which='minor',color='white',lw=1);ax.tick_params(which='minor',length=0)
    cb=fig.colorbar(im,cax=fig.add_axes([.89,.36,.018,.32]));cb.set_label('Share of top-50 axes (%)')
    foot(fig,'Each column sums to 100%. Share counts sender–ligand–receptor axes, not cells or signaling strength.\nGray/—: sender not eligible or not tested; 0: eligible but no axis among the top 50.\n* HFpEF ≥10 LECs/donor; other diseases ≥25. Selection uses disease-prioritized axes only.')


def draw_e(fig, data, *_):
    header(fig,'E','Two strongest additional saved links per direction and type, for each disease')
    direction_key(fig)
    for i,disease in enumerate(DISEASES):
        for j,kind in enumerate(['receptor','target']):
            ax=fig.add_axes([.24+j*.47,.67-i*.155,.23,.092])
            t=data['E'].loc[data['E'].disease.eq(disease)&data['E'].link_type.eq(kind)&data['E'].plotted]
            t=t.sort_values(['direction','weight'],ascending=[False,False])
            y=np.arange(len(t));colors=[UP if x=='upregulated' else DOWN for x in t.direction]
            ax.hlines(y,0,t.weight,color=colors,lw=1.3);ax.scatter(t.weight,y,c=colors,s=24,zorder=3)
            ax.set_yticks(y,[f'{r.ligand} → {r.partner}' for r in t.itertuples()]);ax.invert_yaxis()
            ax.set_xlim(0,max(t.weight)*1.15);clean(ax)
            if i==0:ax.set_title('Ligand–receptor' if j==0 else 'Ligand–target',fontsize=12,pad=28)
            if j==0:fig.text(.045,.78-i*.155,disease_label(disease),fontsize=10,fontweight='bold')
            if i==3:ax.set_xlabel('Prior-network link weight' if j==0 else 'Regulatory-potential weight',fontsize=10)
    foot(fig,'Receptor links already selected in Fig. 5 are excluded. These are extended saved candidate-link tables.\nWeights are model/prior-network evidence, not measured interaction strength or validation.\nDirection refers to the LEC gene program; weight scales differ between receptor and target links.\n'+COHORT)


def draw_f(fig, data, selected, details, ligands):
    header(fig,'F','Additional ligand context across all diseases, with a saved cutoff sensitivity comparison')
    context=data['F'].query("record_type == 'ligand_context'")
    mat=context.pivot(index='ligand',columns='disease',values='priority_percentile').reindex(index=ligands,columns=DISEASES)
    ax=fig.add_axes([.12,.28,.33,.50]);right=fig.add_axes([.64,.30,.30,.46])
    cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#E4E7E9')
    im=ax.imshow(np.ma.masked_invalid(mat),vmin=0,vmax=100,cmap=cmap,aspect='auto',interpolation='nearest')
    ax.set_yticks(range(len(ligands)),ligands);ax.set_xticks(range(4),['DCM','HCM','DoxCM','HFpEF*']);ax.tick_params(length=0)
    ax.set_title('Within-disease ligand percentile',fontsize=11,pad=14)
    cb=fig.colorbar(im,cax=fig.add_axes([.12,.19,.33,.018]),orientation='horizontal');cb.set_label('Percentile of maximum axis priority',fontsize=10)
    t=data['F'].loc[data['F'].record_type.eq('HFpEF_threshold_concordance')&data['F'].plotted.eq(True)].dropna(subset=['prioritization_score_min10','prioritization_score_min15'])
    x=t.prioritization_score_min10;y=t.prioritization_score_min15
    right.scatter(x,y,s=5,color=NAVY,alpha=.20,rasterized=True,edgecolors='none')
    right.plot([0,1],[0,1],ls='--',lw=.8,color=GREY);right.set_xlim(0,1);right.set_ylim(0,1)
    right.set_xlabel('Priority at ≥10 LECs/donor');right.set_ylabel('Priority at ≥15 LECs/donor')
    right.set_title('HFpEF vs matched NF',fontsize=11,pad=14)
    rho=spearmanr(x,y).statistic
    right.text(.05,.96,f'{len(t):,} common axes\nSpearman ρ = {rho:.3f}',transform=right.transAxes,va='top',fontsize=10)
    foot(fig,'Left: union of the top 3 additional ligands per disease; all main-figure ligand selections excluded at selection.\nRanks are within-disease, not comparable effect sizes. Gray: no disease-prioritized qualifying axis.\nRight: all common HFpEF-contrast axes, not only top-ranked axes; shared donors mean this is not replication.\n* HFpEF exploratory ≥10, sensitivity ≥15; matched NF uses the same cutoff. Other diseases ≥25.')


DRAW = dict(zip('ABCDEF',[draw_a,draw_b,draw_c,draw_d,draw_e,draw_f]))
SIZES = {'A':(10,7.5),'B':(10,7.5),'C':(10,9),'D':(10,9),'E':(10,10),'F':(10,10)}


def check_bounds(fig):
    """Detect text outside the actual export canvas (not only axes)."""
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();box=fig.bbox
    failures=[]
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():continue
        b=artist.get_window_extent(renderer)
        if b.width<=0 or b.height<=0:continue
        # Hidden out-of-range tick labels are not drawn by their Axis.
        if artist.axes and artist not in artist.axes.texts and not artist.get_in_layout():continue
        if b.x0<box.x0-2 or b.y0<box.y0-2 or b.x1>box.x1+2 or b.y1>box.y1+2:
            failures.append(artist.get_text())
    if failures:raise RuntimeError('Text outside canvas: '+repr(failures))


def save(fig, path):
    check_bounds(fig)
    for ext in ['pdf','png','jpg']:
        kwargs={'pil_kwargs':{'quality':95}} if ext=='jpg' else {}
        fig.savefig(path.with_suffix('.'+ext),dpi=240,**kwargs)
    plt.close(fig)


LEGEND = '''Figure S6. Extended disease SCENIC and MultiNicheNet analyses supporting Figure 5.

A. Two additional robust state-regulon candidates per state, ranked by the existing consensus rank after excluding pairs shown in Fig. 5A/B. Color is saved dataset/donor-balanced AUCell standardized within each regulon across nine states; values beyond +/-2.5 are saturated and indicated by colorbar extensions. Eighteen rows are selected without requiring that different states have different regulons.
B. The same additional pairs with saved median RSS, 2.5th-97.5th percentile bootstrap intervals and top-set selection frequency. The original analysis used 500 repetitions. Strict dataset/donor balance samples nine donor-state means per state per dataset from datasets 01 and 08; all-dataset donor balance samples 23 donor-state means per state and also includes dataset 11. These are the saved atlas-level SCENIC results, NOT analyses refitted after a >=25-LEC donor filter. They characterize state specificity, not disease-condition differences. States 4/7 carry cardiomyocyte-like marker cautions; State 8 carries a donor-dominance caution. No new resampling was performed. Intervals are not hypothesis tests.
C. Five highest corrected-AUPR additional ligands for each disease-upregulated/downregulated LEC gene program, excluding each disease's main Fig. 5 ligand selections. Corrected AUPR is the saved NicheNet activity metric. Program direction does not imply that the ligand itself is differentially expressed. Complete activity rankings are included in the panel CSV.
D. Percentage of the top 50 disease-prioritized sender-ligand-receptor axes attributed to each sender. Filters: receiver LEC_EC8_ln, group=disease, top_group=disease. Sort by descending prioritization score, ties by id. All sender types eligible in at least one disease are displayed. Missing/ineligible senders are gray, eligible zero contributions are numerical zero; columns sum to 100%. This describes representation in a ranked candidate set, not cell abundance or signaling strength. Full priorities, including NF-side rows, and eligibility-derived summaries are included in the panel CSV.
E. Two largest saved network weights per disease, direction and link type. Main-figure ligand-receptor pairs are excluded; ligand-target links extend the main figure, which did not display targets. Selection is restricted to the available saved Sydney-equivalent link tables, not the entire interaction database. Receptor weights are prior-network evidence; target weights are regulatory-potential evidence. They are not differential-expression values, measured communication strengths or experimental validation. Direction identifies the receiver gene program. All available saved links are included in the panel CSV.
F. Left: union of the top three additional disease-prioritized ligands per disease, excluding that disease's main-figure ligands during selection. For each displayed ligand, color is its within-disease percentile (average-rank percentile, 0-100) of the maximum prioritization score across qualifying axes; a selected ligand may already occur in another disease's main figure. Gray denotes no qualifying disease-prioritized axis. This is descriptive cross-disease context, not a pooled contrast: diseases and datasets are partly confounded and percentile ranks are not comparable biological effect sizes. Right: all common HFpEF-vs-NF LEC axes from the saved >=10/15 concordance CSV, irrespective of top-group assignment. Spearman correlation is descriptive; thresholds share donors and do not constitute independent replication.

MultiNicheNet cohorts (C-F): DCM 10 vs 12 NF, HCM 15 vs 12 NF, DoxCM 9 vs 4 NF at >=25 LECs/donor. HFpEF exploratory >=10: 7 vs 6 NF; sensitivity >=15: 5 vs 4 NF. Both disease and matched NF donors use the indicated receiver-cell threshold. Sender eligibility is analysis-specific and is recorded in the original eligibility CSVs. HFpEF relaxed-cutoff results remain exploratory.

All outputs are regenerated from current saved CSV results. No raw counts, models, significance tests, bootstraps or main-figure files are changed. The input manifest records exact relative paths, uses and SHA-256 hashes. Panel CSVs contain selected and extended records with explicit record_type/plotted flags where applicable. Native PDF text is embedded; only the dense sensitivity scatter is rasterized for file size.
'''


def main(panel=None):
    setup();OUT.mkdir(parents=True,exist_ok=True)
    data, selected, details, ligands=build_data()
    letters=[panel] if panel else list('ABCDEF')
    for letter in letters:
        dest=OUT/f'FigS06{letter}_{NAMES[letter]}';dest.mkdir(exist_ok=True)
        data[letter].to_csv(dest/f'FigS06{letter}_source_data.csv',index=False)
        fig=plt.figure(figsize=SIZES[letter]);DRAW[letter](fig,data,selected,details,ligands)
        save(fig,dest/f'FigS06{letter}_{NAMES[letter]}')
        print(f'Saved panel {letter}: PDF, PNG, JPG and source CSV',flush=True)
    if not panel:
        fig=plt.figure(figsize=(20.8,27.3))
        subs=fig.subfigures(3,2,height_ratios=[7.5,9,10],wspace=.04,hspace=.03)
        for letter,sub in zip('ABCDEF',subs.flat):DRAW[letter](sub,data,selected,details,ligands)
        save(fig,OUT/'FigS06_combined')
        print('Saved combined figure: PDF, PNG, JPG',flush=True)
    pd.DataFrame(MANIFEST).drop_duplicates().to_csv(OUT/'FigS06_input_manifest.csv',index=False)
    (OUT/'FigS06_figure_legend.txt').write_text(LEGEND,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--panel',choices=list('ABCDEF'))
    main(parser.parse_args().panel)
