from core import *
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
CONDITIONS=[('Control','None'),('H1N1','None'),('Control','VEGF-C'),('H1N1','VEGF-C'),('Control','UC2288'),('H1N1','UC2288')]
COLORS=['#65738a','#c04440','#4e98ad','#ef954c','#577952','#985da4']
SCOL=['#d58120','#327aa4','#6d439b','#a9617d','#68874b','#777777']
FIGURES=[];PDF=None

def read(rel):return pd.read_csv(ROOT/'results'/rel,keep_default_na=False,low_memory=False)
def cj(p,i):return CONDITIONS.index((p,i))
def short(f):return f.replace('ERWNuclei__AreaShape_','Cell: ').replace('nuclei__AreaShape_','Nucleus: ').replace('ERWONuclei__Intensity_','ER: ').replace('mitochondriaWONuclei__Intensity_','Mito: ').replace('nuclei__Intensity_','Nucleus intensity: ').replace('_mitochondria','').replace('_nucleus','').replace('_ER','')
def contrast_label(c):return c.replace('__',' | ').replace('H1N1_vs_Control','H1N1 - Control').replace('VEGFC_in_H1N1','VEGF-C under H1N1').replace('UC2288_in_H1N1','UC2288 under H1N1')
def finish(fig,name,title,caption):
    path=ROOT/'results/05_figures'/name
    fig.savefig(path.with_suffix('.png'),dpi=170,bbox_inches='tight');fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    PDF.savefig(fig,bbox_inches='tight');plt.close(fig)
    FIGURES.append(dict(file=name+'.png',title=title,caption=caption))

def dotplot(ax,d,column,percent=True,annotate=False):
    for j,(p,i) in enumerate(CONDITIONS):
        g=d.loc[d.plasma.eq(p)&d.intervention.eq(i)].sort_values('physical_well')
        vals=g[column].to_numpy(float);off=np.linspace(-.15,.15,len(g))
        ax.scatter(vals,np.full(len(vals),j)+off,color=COLORS[j],s=35,zorder=3)
        ax.plot([vals.mean()]*2,[j-.27,j+.27],color='black',lw=2)
        if annotate:
            for value,y,well in zip(vals,j+off,g.physical_well):ax.annotate(well,(value,y),xytext=(4,0),textcoords='offset points',fontsize=7)
    ax.set_yticks(range(6),[p+' / '+('no intervention' if i=='None' else i) for p,i in CONDITIONS]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    if percent:ax.xaxis.set_major_formatter(PercentFormatter(1))

def scatter_groups(ax,xy,cells,full=True):
    for passage in ['P3','P9']:
        for j,(plasma,intervention) in enumerate(CONDITIONS):
            mask=cells.passage.eq(passage)&cells.plasma.eq(plasma)&cells.intervention.eq(intervention)
            ax.scatter(xy[mask,0],xy[mask,1],s=7,alpha=.55,color=COLORS[j],marker='o' if passage=='P3' else '^',rasterized=True)

def group_legend(fig):
    handles=[Line2D([0],[0],marker='o' if passage=='P3' else '^',ls='',color=COLORS[j],label=f'{passage} {p} / {i}') for passage in ['P3','P9'] for j,(p,i) in enumerate(CONDITIONS)]
    fig.legend(handles=handles,loc='outside lower center',ncol=4,fontsize=8,frameon=False)

def make_figures(cells,panel,arrays):
    global PDF
    wells=read('02_responses/well_response_metrics.csv');scored=read('02_responses/cell_response_scores.csv.gz').set_index('cell_id').loc[cells.cell_id].reset_index()
    qc=read('01_data_qc/well_qc.csv');screens=read('03_states/candidate_screens.csv');labels=read('03_states/cell_cluster_assignments.csv.gz').set_index('cell_id').loc[cells.cell_id]
    pca=read('04_embeddings/pca_coordinates.csv.gz');pca=pca.loc[pca.scope.eq('All')].set_index('cell_id').loc[cells.cell_id]
    um=read('04_embeddings/umap_all_settings.csv.gz');setting=f'n30_d0.5_s{SEED}';baseum=um.loc[um.setting.eq(setting)].set_index('cell_id').loc[cells.cell_id]
    xy=baseum[['UMAP1','UMAP2']].to_numpy();pcs=pca[['PC1','PC2']].to_numpy()
    spectrum=[]
    for scope in ['All','P3','P9']:
        sel=np.ones(len(cells),bool) if scope=='All' else cells.passage.eq(scope).to_numpy()
        wt=weights(cells.loc[sel]);tt=arrays['transformed'][sel];center,scale=robust_scale(tt,wt)
        _,variance,_,_=weighted_pca((tt-center)/scale,wt)
        for j,value in enumerate(variance):spectrum.append(dict(scope=scope,PC=j+1,variance_ratio=value,cumulative_variance=variance[:j+1].sum()))
    save(pd.DataFrame(spectrum),'04_embeddings/pca_variance_spectrum.csv')
    PDF=PdfPages(ROOT/'results/05_figures/all_analysis_figures.pdf')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,p in zip(axes,['P3','P9']):dotplot(ax,qc[qc.passage.eq(p)],'n_cells',percent=False);ax.set_title(p);ax.set_xlabel('Cells per well (each dot = one well)')
    finish(fig,'01_cell_recovery','Cell recovery and experimental units','All 60 wells and 4,866 cells are retained. Each dot is one well; black lines are condition means. Different horizontal scales show the much smaller P3 cell yield. Counts are not evidence of treatment efficacy.')
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    scatter_groups(axes[0,0],pcs,cells);scatter_groups(axes[0,1],xy,cells)
    for j,coords in enumerate([pcs,xy]):
        for s in range(3):
            mask=labels['All__k3'].eq(s);axes[1,j].scatter(coords[mask,0],coords[mask,1],s=7,color=SCOL[s],alpha=.6,label=f'Candidate state {s}',rasterized=True)
        axes[1,j].legend(fontsize=8)
    for row in axes:
        row[0].set(xlabel='PC1',ylabel='PC2');row[1].set(xlabel='UMAP1',ylabel='UMAP2')
    axes[0,0].set_title('Weighted PCA: condition overlay');axes[0,1].set_title('Fixed UMAP: condition overlay');axes[1,0].set_title('PCA: exploratory k=3 partition');axes[1,1].set_title('Same UMAP: exploratory k=3 partition')
    group_legend(fig)
    finish(fig,'02_pca_umap_overview','PCA and UMAP in the expanded feature space','Geometry uses 97 features and 42 PCs explaining at least 90% of equal-well variance. Circles are P3; triangles are P9. The bottom panels show the common-space k=3 partition selected only by prespecified resampling screens. UMAP gaps are not statistical evidence of states or treatment effects.')
    fig,axes=plt.subplots(1,2,figsize=(16,7),layout='constrained')
    for ax,p in zip(axes,['P3','P9']):
        entries=cells.loc[cells.passage.eq(p)].drop_duplicates('well_id').sort_values('physical_well')
        colors=plt.cm.turbo(np.linspace(.03,.97,len(entries)))
        for color,(_,r) in zip(colors,entries.iterrows()):
            mask=cells.well_id.eq(r.well_id);ax.scatter(xy[mask,0],xy[mask,1],s=9,color=color,alpha=.7,label=r.physical_well,rasterized=True)
        ax.set(title=p+' wells in the common UMAP',xlabel='UMAP1',ylabel='UMAP2');ax.legend(title='Physical well',bbox_to_anchor=(1.01,1),loc='upper left',ncol=2,fontsize=7,frameon=False)
    finish(fig,'03_umap_by_well','Individual-well overlays','Both panels use the same UMAP coordinates. The legend identifies each physical well. Concentration of a region in one well is a technical/reproducibility concern; colors themselves do not establish biological states.')
    for seed in CFG['umap_seeds']:
        fig,axes=plt.subplots(2,3,figsize=(14,9),layout='constrained')
        for ax,(md,nn) in zip(axes.flat,itertools.product(CFG['umap_min_dist'],CFG['umap_neighbors'])):
            a=um.loc[um.setting.eq(f'n{nn}_d{md}_s{seed}')].set_index('cell_id').loc[cells.cell_id]
            scatter_groups(ax,a[['UMAP1','UMAP2']].to_numpy(),cells);ax.set(title=f'neighbors={nn}, min_dist={md}',xlabel='UMAP1',ylabel='UMAP2')
        group_legend(fig)
        finish(fig,f'04_umap_parameters_seed{seed}',f'UMAP parameter sweep, seed {seed}','Six parameter combinations for this seed; all 18 embeddings are saved. Apparent spacing, orientation and islands can change. No setting was selected to maximize condition separation; clustering and response tests do not use these 2-D coordinates.')
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    for ax,p in zip(axes,['P3','P9']):dotplot(ax,wells[wells.passage.eq(p)],'All__tail95');ax.set_title(p);ax.set_xlabel('Mean fraction exceeding held-out control 95th percentile')
    finish(fig,'05_minority_responder_fractions','Minority high-distance fractions','Dots show individual wells. Each treated-cell call is averaged across five models trained without one control well; controls use only their held-out model. Each passage has its own empirical control threshold. P9 UC2288 under H1N1 increases the mean from 4.08% to 8.30%, but the primary test has FDR q=0.286.')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,p in zip(axes,['P3','P9']):
        for j,(plasma,intervention) in enumerate(CONDITIONS):
            g=scored.loc[scored.passage.eq(p)&scored.plasma.eq(plasma)&scored.intervention.eq(intervention)]
            grid=np.linspace(.5,2,220);curves=[]
            for _,well in g.groupby('well_id'):
                v=well['All__distance'].to_numpy();survival=np.mean(v[:,None]>grid[None,:],axis=0);curves.append(survival);ax.plot(grid,survival,color=COLORS[j],lw=.6,alpha=.2)
            ax.plot(grid,np.mean(curves,axis=0),color=COLORS[j],lw=2,label=f'{plasma} / {intervention}')
        ax.axvline(1,ls='--',color='black',lw=1);ax.set(xlabel='Mean distance / control 95th-percentile distance',ylabel='Fraction of cells above distance',title=p,ylim=(0,.3));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(fontsize=7)
    finish(fig,'06_single_cell_distance_tails','Single-cell distance distributions','Thin lines show wells and thick lines are equal-well mean survival curves, zoomed to the upper 30% of the distribution. The dashed line is the control distance threshold. This plot uses each cell\'s mean distance; the primary responder metric averages threshold calls across models and is not exactly the same quantity.')
    fig,axes=plt.subplots(1,2,figsize=(15,5.5),layout='constrained')
    for ax,p in zip(axes,['P3','P9']):
        d=wells[wells.passage.eq(p)].groupby(['plasma','intervention'])[[f+'__tail95' for f in FAMILIES]].mean().reindex(CONDITIONS)
        im=ax.imshow(d.to_numpy()*100,vmin=0,vmax=25,cmap='magma');ax.set_yticks(range(6),[a+' / '+b for a,b in CONDITIONS]);ax.set_xticks(range(5),['Cell\nshape','Nuclear\nshape','ER\nintensity','Mito\nintensity','Nuclear\nintensity']);ax.set_title(p)
        for i in range(6):
            for j in range(5):ax.text(j,i,f'{d.iloc[i,j]*100:.1f}',ha='center',va='center',color='white' if d.iloc[i,j]<.14 else 'black',fontsize=9)
    fig.colorbar(im,ax=axes,label='Mean high-distance fraction (%)',shrink=.7)
    finish(fig,'07_compartment_response_fractions','Compartment-specific deviations','Each entry is a mean across five wells using a separate distance model for that measurement family. Values are percentages, not changes from baseline. Full individual-well values are in well_response_metrics.csv. No compartment comparison passes its 30-test BH family.')
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    p9=wells.loc[wells.passage.eq('P9')&wells.plasma.eq('H1N1')]
    for ax,ep,title in zip(axes,['All__tail95','ER intensity__tail95'],['All 97 features','ER intensity features']):
        for j,iv in enumerate(['None','VEGF-C','UC2288']):
            d=p9[p9.intervention.eq(iv)].sort_values('physical_well');values=d[ep].to_numpy();ax.scatter(values,j+np.linspace(-.12,.12,5),s=45,color=COLORS[cj('H1N1',iv)])
            ax.plot([values.mean()]*2,[j-.25,j+.25],color='black',lw=2)
            for val,y,well in zip(values,j+np.linspace(-.12,.12,5),d.physical_well):ax.annotate(well,(val,y),xytext=(5,0),textcoords='offset points',fontsize=8)
        ax.set_yticks(range(3),['H1N1','H1N1 + VEGF-C','H1N1 + UC2288']);ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set(title=title,xlabel='Model-averaged responder fraction');ax.margins(x=.2)
    finish(fig,'08_P9_intervention_candidate','P9 intervention-associated candidate signal','Physical wells are labeled. UC2288 increases the average ER-tail fraction from 3.66% to 14.66%; all five UC2288 wells exceed the untreated H1N1 wells in this endpoint. Exact p=0.00794, BH q=0.238 across 30 compartment tests. This is a hypothesis for follow-up, not a confirmed effect or evidence of functional rescue.')
    sensitivity=read('02_responses/response_sensitivity.csv')
    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    for ax,ep in zip(axes,['All__tail95','ER intensity__tail95']):
        d=sensitivity[sensitivity.endpoint.eq(ep)]
        for j,name in enumerate([v[0] for v in contrasts(cells)]):
            v=d.loc[d.contrast.eq(name),'effect_B_minus_A'].to_numpy(float)*100;ax.scatter(v,j+np.linspace(-.2,.2,len(v)),s=22,color='#326d93',alpha=.7)
        ax.set_yticks(range(6),[contrast_label(v[0]) for v in contrasts(cells)]);ax.invert_yaxis();ax.axvline(0,color='black',ls='--',lw=1);ax.set(title=ep.replace('__tail95',''),xlabel='Change in responder fraction (percentage points)')
    finish(fig,'09_response_sensitivity','Metric, feature and control-well sensitivity','Each point is one prespecified sensitivity: alternate shrinkage, diagonal distance, removal of edge objects, removal of Zernike descriptors, equal weighting of feature families, or reconstruction omitting one control well. These are effect-size checks, not independent replications or additional opportunities to select a significant test.')
    drivers=read('02_responses/feature_driver_tests.csv');simple=drivers[~drivers.feature.str.contains('Zernike')].copy()
    simple['strength']=abs(simple.effect_in_robust_z);chosen=[]
    for fam in FAMILIES:
        sub=simple.loc[simple.family.eq(fam)].groupby(['feature','summary']).strength.mean().sort_values(ascending=False).head(3)
        chosen += list(sub.index)
    mat=simple.set_index(['feature','summary','contrast']).effect_in_robust_z.unstack('contrast').reindex(pd.MultiIndex.from_tuples(chosen));order=[v[0] for v in contrasts(cells)];mat=mat[order]
    fig,ax=plt.subplots(figsize=(13,9),layout='constrained');limit=max(.6,float(abs(mat.to_numpy()).max()));im=ax.imshow(mat,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_yticks(range(len(mat)),[short(f)+' ['+s+']' for f,s in mat.index],fontsize=8);ax.set_xticks(range(6),[contrast_label(c).replace(' | ','\n').replace(' under ','\n') for c in order],fontsize=8);fig.colorbar(im,ax=ax,label='Difference of well summaries (global robust-z units)',shrink=.65)
    finish(fig,'10_feature_driver_effects','Interpretable feature changes, including distribution tails','The display selects three feature/summary pairs per family by average absolute effect, excludes Zernike names for readability, and includes median, lower-tail or upper-tail summaries as labeled. Red means higher in the challenged/intervention group. This is descriptive ranking: none of the 2,502 feature tests passes the combined BH correction. All features, including Zernike descriptors, remain in the result table.')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for p,color in zip(['All','P3','P9'],['#3b7a57','#3073a1','#c17b25']):
        d=screens[screens.scope.eq(p)];axes[0].plot(d.k,d.bootstrap_median_ARI,'o-',label=p,color=color);axes[1].plot(d.k,d.bootstrap_min_state_median_jaccard,'o-',label=p,color=color)
    for ax,threshold,title in zip(axes,[.8,.75],['Whole-partition bootstrap stability','Least-stable state in each partition']):ax.axhline(threshold,color='black',ls='--',lw=1);ax.set(xlabel='Candidate number of clusters',ylabel='Agreement (larger = more reproducible)',title=title,xticks=[2,3,4,5,6],ylim=(0,1.03));ax.legend(title='Analysis scope')
    finish(fig,'11_cluster_stability','Reproducibility of candidate partitions','ARI measures agreement of the entire partition; Jaccard evaluates each state separately so a stable majority cannot conceal an unstable minority. Only common-space k=3 passes all prespecified resampling, parameter and well-support screens. Passing these screens is not biological validation.')
    profiles=read('03_states/state_feature_profiles.csv');gp=profiles[profiles.scope.eq('All')&profiles.k.eq(3)&~profiles.feature.str.contains('Zernike')].pivot(index='feature',columns='state',values='weighted_median_global_z')
    pick=(gp.max(1)-gp.min(1)).sort_values(ascending=False).head(18).index;gp=gp.loc[pick]
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained');im=ax.imshow(gp,cmap='RdBu_r',vmin=-4,vmax=4,aspect='auto')
    ax.set_xticks(range(3),['Candidate state 0\n123 cells','Candidate state 1\n3,853 cells','Candidate state 2\n890 cells']);ax.set_yticks(range(len(gp)),[short(f) for f in gp.index],fontsize=8)
    for i in range(len(gp)):
        for j in range(3):ax.text(j,i,f'{gp.iloc[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if abs(gp.iloc[i,j])>2.3 else 'black')
    fig.colorbar(im,ax=ax,label='Median global robust z; color clipped at +/-4')
    finish(fig,'12_candidate_state_features','Features defining the exploratory three-state partition','State 0 has unusually high nuclear compactness/perimeter and low extent; 120 of its 123 cells have at least one |robust z|>8. States 1 and 2 differ in size/intensity and strongly track passage. Feature-family balancing changes this partition (ARI=0.726). Labels remain numerical; none is assigned a mechanistic identity.')
    props=read('03_states/well_state_proportions.csv');props=props[props.scope.eq('All')&props.k.eq(3)]
    fig,axes=plt.subplots(1,2,figsize=(15,11),layout='constrained')
    for ax,p in zip(axes,['P3','P9']):
        wm=cells[cells.passage.eq(p)].drop_duplicates('well_id').copy();wm['sortkey']=[cj(a,b) for a,b in zip(wm.plasma,wm.intervention)];wm=wm.sort_values(['sortkey','physical_well']);v=props.pivot(index='well_id',columns='state',values='proportion').loc[wm.well_id];left=np.zeros(len(v))
        for j in range(3):ax.barh(np.arange(len(v)),v[j],left=left,color=SCOL[j],label=f'State {j}');left+=v[j].to_numpy()
        ax.set_yticks(range(len(v)),[f'{r.physical_well} | {r.plasma} / {r.intervention}' for _,r in wm.iterrows()],fontsize=7);ax.invert_yaxis();ax.set(title=p,xlabel='Fraction of cells in each well',xlim=(0,1));ax.xaxis.set_major_formatter(PercentFormatter(1));ax.legend(ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.07))
    finish(fig,'13_candidate_state_proportions','Candidate state proportions in every well','Every bar is one physical well and sums to 100%. Both passages use the same state definitions. The strong P3/P9 composition difference is confounded with acquisition date and segmentation settings. No state-proportion contrast passes BH correction across the 240 candidate-state tests.')
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,p in zip(axes,['P3','P9']):
        m=cells.passage.eq(p)
        for s in range(3):
            select=m&labels['All__k3'].to_numpy().__eq__(s);ax.scatter(cells.loc[select,'nuclei__AreaShape_Area'],cells.loc[select,'nuclei__AreaShape_Compactness'],s=9,color=SCOL[s],alpha=.6,label=f'State {s}',rasterized=True)
        ax.set(xlabel='Nuclear area (pixels)',ylabel='Nuclear compactness (CellProfiler)',title=p,yscale='log');ax.legend()
    finish(fig,'14_nuclear_shape_QC','Nuclear morphology underlying the rare candidate state','State 0 occupies an extreme nuclear-shape region. Such measurements may reflect genuine unusual nuclei, segmentation problems, or both. Original-image review is supplied; exported segmentation masks are unavailable, so these cells cannot be certified as a biological state from feature stability alone.')
    PDF.close();save(pd.DataFrame(FIGURES),'05_figures/figure_guide.csv')

def table_md(df):
    cols=list(df.columns);rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    rows += ['| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def report(cells,panel,arrays):
    q=json.loads((ROOT/'results/01_data_qc/summary.json').read_text());r=json.loads((ROOT/'results/02_responses/summary.json').read_text());s=json.loads((ROOT/'results/03_states/summary.json').read_text())
    tests=read('02_responses/primary_response_tests.csv');ct=read('02_responses/compartment_response_tests.csv');sens=read('02_responses/response_sensitivity.csv');screens=read('03_states/candidate_screens.csv');w=read('02_responses/well_response_metrics.csv')
    comparisons=tests[tests.endpoint.eq('All__tail95')].copy();display=pd.DataFrame({'Comparison':comparisons.contrast.map(contrast_label),'Reference (%)':comparisons.mean_A.map(lambda x:f'{100*x:.2f}'),'Compared (%)':comparisons.mean_B.map(lambda x:f'{100*x:.2f}'),'Change (pp)':comparisons.effect_B_minus_A.map(lambda x:f'{100*x:+.2f}'),'Exact p':comparisons.p_exact.map(lambda x:f'{x:.4f}'),'BH q (18 tests)':comparisons.q_BH_family.map(lambda x:f'{x:.3f}')})
    er=ct.loc[ct.contrast.eq('P9__UC2288_in_H1N1')&ct.endpoint.eq('ER intensity__tail95')].iloc[0]
    ers=sens.loc[sens.contrast.eq('P9__UC2288_in_H1N1')&sens.endpoint.eq('ER intensity__tail95')]
    uq=read('04_embeddings/umap_quality.csv')
    image_review_done=(ROOT/'results/01_data_qc/image_review_summary.json').exists()
    mapping=cells.groupby(['passage','plasma','intervention'],sort=False).agg(physical_wells=('physical_well',lambda x:', '.join(sorted(set(x)))),source_folders=('raw_export_folder',lambda x:', '.join(sorted(set(x)))),n_cells=('cell_id','size')).reset_index()
    save(mapping,'01_data_qc/plate_condition_mapping.csv')
    blocks=[]
    def add(title,text):blocks.append((title,text))
    add('What was done and what is supported',f'''The expanded analysis uses {q['cells']:,} verified cells in {q['wells']} physical wells. It starts from all 186 exported measurements, including 155 named shape/intensity fields. Removing absolute coordinates/orientation and near-constant measurements leaves 139 eligible biological measurements; correlation filtering leaves **97 features** (previous analysis: 10). The raw exports contain no mitochondrial network morphology, texture, or actin/Golgi feature panel. Cell shape comes from the ER-derived secondary cell-region mask.

Continuous control-referenced distances and compartment tail fractions are the most useful current outputs for the user's minority-response question. They preserve per-cell variation and convert it to measurements for each well, without requiring cells to form discrete clusters. **No primary response, compartment, feature-driver, or state-proportion test passes its prespecified BH family at q<0.05.** This is inconclusive evidence, not proof that effects are absent.

The clearest follow-up candidate is **P9 H1N1 + UC2288**, particularly its ER-intensity tail. Its direction is stable to reasonable distance and reference changes, but it remains exploratory after multiplicity correction. Increasing distance from control means an altered phenotype; it does not establish damage, senescence, benefit, or rescue.''')
    add('Minority-response results',table_md(display)+f'''\n\nFractions are model-averaged calls using the held-out control 95th-percentile distance. Comparisons use independent wells, not cells, as replicates. The UC2288 ER endpoint changes from **{100*er.mean_A:.2f}% to {100*er.mean_B:.2f}%** (difference {100*er.effect_B_minus_A:.2f} percentage points; exact p={er.p_exact:.5f}; BH q={er.q_BH_family:.3f}). Across the 11 sensitivity specifications its ER-tail effect ranges from {100*ers.effect_B_minus_A.min():.2f} to {100*ers.effect_B_minus_A.max():.2f} percentage points. These checks use the same dataset and are not independent validation.

H1N1 alone does not show an increased overall tail in either passage in the primary analysis. The estimated tail is smaller, particularly in P3, with uncertain inference. Do not interpret lower distance as improved function. The 90% and 97.5% thresholds are supplied as sensitivity outputs, not substitutes chosen for a better P value.''')
    add('Reproducible states versus forced partitions',f'''A common-space **k=3** partition passes the prespecified resampling and well-support screens: bootstrap median ARI 0.914; weakest-state median Jaccard 0.760; leave-one-well-out median ARI 0.964, minimum 0.896. The expanded **k=2** partition is less reproducible (bootstrap ARI 0.458). Thus fewer clusters do not automatically perform better.

However, this three-state partition is **not a validated set of biological cell states**. State 0 contains 123 cells (72 P3, 51 P9) with markedly unusual nuclear compactness/perimeter; 120/123 carry an extreme-feature flag. States 1 and 2 contain 3,853 and 890 cells and strongly track passage. Morphology-only and equal-family-weight representations change the partition (ARI 0.565 and 0.726). Excluding edge cells has less impact (ARI 0.933), and omitting Zernike descriptors retains much of it (ARI 0.918).

None of the passage-specific partitions passes every screen. No candidate-state proportion test passes the combined 240-test BH family. HDBSCAN assigns all cells as unassigned for the tested PCA-space density settings; this is a negative robustness check, not proof that no rare state exists, because density estimation is difficult in 41–45 dimensions. All candidate solutions, including failed ones, are saved.''')
    add('Which measurements drive the phenotypes',f'''The retained panel includes 40 cell-shape, 38 nuclear-shape, 5 ER-intensity, 9 mitochondrial-intensity and 5 nuclear-intensity features. It includes Zernike shape descriptors, which describe shape mathematically rather than identifying a mechanism.

Each of the 139 eligible measurements was tested using its **within-well median, 10th percentile and 90th percentile**. This permits minority-tail changes without requiring a shifted mean. There are 139 × 3 × 6 = **2,502 tests**, corrected as one family; none passes q<0.05. The complete effect-size table remains useful for prioritizing reproducible, interpretable follow-up measurements. Descriptive responder-versus-nonresponder feature profiles are also supplied, but those cells were selected using the same features, so their signature is not independent validation.

The rare common-space candidate state is primarily a nuclear-shape phenotype. The P9 UC2288 candidate is most apparent in the ER-intensity distance family. Neither observation identifies a molecular pathway, and mitochondrial intensity alone cannot establish network fragmentation or mitochondrial function.''')
    add('Exact methods and thresholds', '''Transforms and feature choices are label-blind. Nonnegative unbounded features use log1p(x / equal-well positive median); bounded shape ratios remain linear. Scale by an equal-well median and 1.4826 MAD (IQR fallback). Exclude >1% missing features, >=99% identical features, absolute position/orientation, and greedily prune |equal-well weighted Spearman|>0.90, prioritizing interpretable variables. The feature_decisions.csv file records every included or excluded measurement. This leaves 97 variables; rare high-distance cells are not removed merely for being unusual.

Each passage is referenced to its own five Control/None wells. Five models each fit four controls and withhold one entire control well. A model uses equal-well means/covariances with 20% covariance shrinkage toward an isotropic target. Distance is sqrt((x-mu)' inverse(Sigma) (x-mu) / number_of_features). The empirical equal-well 95th percentile of held-out control distances is the main responder threshold; 90% and 97.5% are sensitivities. Noncontrol cells are evaluated with all five reference models, and their well's fraction is averaged over models. A separate >=4/5-model consensus flag is provided as an exploratory cell-review aid. Controls have one out-of-fold evaluation and no comparable five-model consensus flag.

No chi-square distance cutoff is assumed. The 95th percentile is a screening convention, not a proven biological boundary or a guaranteed external false-positive rate. Well bootstrap intervals use 5,000 resamples and are **conditional on the fitted feature/reference representation**, unadjusted for multiplicity. They omit reference-training uncertainty, which is investigated separately by omitting each reference well and reconstructing the models/thresholds.

Exact tests enumerate all 252 allocations of ten wells to two groups of five. Challenge-versus-reference tests rebuild models and thresholds under every allocation. Intervention comparisons use the separate Control/None reference and permute only the two compared groups. BH is applied separately to 18 primary response tests, 30 compartment tests, 2,502 feature tests, and 240 candidate-state tests. These are distinct prespecified families; they are not one global experiment-wide FDR guarantee. P<0.05 before correction is not treated as confirmation. Cells are never used as independent treatment replicates.

Weighted PCA and k-means give each well equal total influence. The scan tests k=2–6, 85/90/95% PCA variance and three seeds. Forty hierarchical well/cell bootstraps and all leave-one-well-out fits refit scaling and PCA; the feature identities/transforms/correlation panel stay frozen, so stability remains conditional on that selection. Required screens are median ARI>=0.80, weakest-state median Jaccard>=0.75, leave-one-well-out minimum ARI>=0.60, minimum cross-parameter ARI>=0.80, >=2 cells in >=3 wells per state, and no well supplying >40% of equal-well state mass. These are project screening choices, not universal definitions of a biological state.''')
    add('UMAP robustness',f'''All **18 UMAP settings** are saved: neighbors 15/30/60, min_dist 0.1/0.5, and three seeds. The main view is fixed at neighbors=30, min_dist=0.5 and seed={SEED}; it was not selected by appearance or treatment separation. All overlays share identical coordinates.

The 15-neighbor recall relative to the PCA space ranges from {uq.neighbor_recall_15.min():.3f} to {uq.neighbor_recall_15.max():.3f}. Balanced-sample trustworthiness ranges from {uq.trustworthiness_balanced_sample.min():.3f} to {uq.trustworthiness_balanced_sample.max():.3f}. This distinction matters: UMAP can place generally similar cells nearby without preserving their exact nearest-neighbor identities. Distances, clusters and treatment inference are computed outside 2-D UMAP; changing UMAP does not change those numerical results.''')
    add('Quality control and concerns',f'''All 60 physical wells in the 12 export folders are traced through the verified registry and all 84 raw CSV hashes match the prior source audit. Tables are joined by ImageNumber and Parent_nuclei, not by row order. Exactly the previously identified 577 wrong-folder copies remain excluded; no additional cells were deleted. All 13 legacy features match the rebuilt raw measurements, including the corrected P3 ER maximum. There are no duplicate complete retained 186-feature profiles.

There are **289 edge-touching cells**, **0 saturated cell maxima** at normalized intensity >=0.999, and **167 cells with at least one |robust z|>8**. These are flags, not evidence that each cell is technically wrong. The separate edge-exclusion analysis retains the primary observations broadly, but cannot establish segmentation accuracy.

All 60 original ND2 files were read without copying or modifying them. Nuclear maximum-intensity coordinates and values match the original DAPI pixels for **all 4,866 cells**. This establishes source identity; it does not certify the segmentation boundary. Forty-eight traceable high-distance/typical-cell crops are provided. Some reviewed high-distance cells have unusually large, lobulated or diffuse nuclear appearances and strong organelle-channel signals. Without segmentation masks, biological changes and segmentation/acquisition artifacts cannot be resolved conclusively. Channel display limits are shared within each field, not across passages.

The exports use **series 0 / frame 0: one field per well**, although each ND2 contains three positions. The other two fields were not added to this analysis. P9 still covers 30 of the 36 planned wells; G03–G08 are missing. P3 wells contain only 16–49 cells, making a 5% tail roughly one or two cells in many wells. Rare-population fractions therefore have limited precision.

Passage is confounded with imaging date, staining/acquisition conditions, cell yield, and different nuclear diameter filters (P3 43–150 pixels; P9 38–150 pixels). Within-passage reference analysis reduces dependence on the large baseline passage difference but cannot eliminate all technical effects. The five wells are treated as different replicates at the user's request; donor/sample identities were not independently documented. If wells share donors or plasma pools, inference must be revised to the true independent unit. No matched functional assay is available here.

Biological labels preserve the supplied spreadsheet/export mapping: P3 conditions occupy fixed plate rows and P9 conditions occupy fixed plate columns. The physical sources are verified, but a separate experimental plate-plan document was not independently checked in this run. Thus source identity does not prove which treatment was applied. The mapping is exported in plate_condition_mapping.csv. Treatment is also confounded with plate row/column; systematic position effects could mimic a reproducible treatment difference. The exact well-label tests assume exchangeability under the null and do not resolve this experimental-design limitation.''')
    add('Comparison with the earlier pipeline and practical interpretation','''The original and corrected projects remain untouched. Their pre-existing files are listed in results/00_preservation/prior_projects_sha256.csv and verified again at completion. The new project uses the same corrected cell identities, but **rebuilds the measurement matrix from the raw exports**; it does not reuse the 13-feature panel as its analysis input.

The most informative current panels are 05 (individual-well responder fractions), 08 (the P9 UC2288 candidate with individual wells visible), 09 (sensitivity of the effects), and 14 plus the image-review pages (nuclear-shape/QC concerns). The state heatmap and UMAP are supporting exploratory views. There is no evidence here sufficient to claim validated biological states, functional rescue, or a confirmed treatment effect.

A focused next experiment or analysis should review segmentation masks for the extreme nuclear-shape group, process all three fields with matched image-analysis settings, and test the ER-tail candidate in independent wells/experiments with donor metadata. These are follow-up recommendations; the present scripts did not resegment images or impute missing wells.''')
    add('Files and reproducibility','''The runnable stages are: 01_build_features.py (source join/QC); 02_responses.py (distances, minority fractions, feature drivers, statistics); 03_states_embeddings.py (states, resampling, PCA/UMAP sweeps); 04_image_review.py (optional original-image QC); 05_figures_report.py (figures/report); 06_validate.py (source/output/preservation checks). core.py contains shared helpers. run_analysis.ps1 executes the stages.

Required feature-analysis inputs are the four small registry/provenance files in inputs/ and the 84 CSV files in the original Data/vivo/Flu P3 and Flu P9 folders. Original ND2 images are read only by optional image review. No Excel workbook or old pipeline script is required at runtime. The copied verified_cell_registry.csv retains legacy columns for comparison, but only its identities/labels determine the new feature join. Outputs are separated into data/QC, continuous responses, states, embeddings, figures and validation folders.

Method sources: [single-cell heterogeneity in image-based profiling](https://www.nature.com/articles/s41467-019-10154-8), [SPACe](https://www.nature.com/articles/s41467-024-54264-4), [UMAP clustering cautions](https://umap-learn.readthedocs.io/en/latest/clustering.html), [covariance regularization](https://scikit-learn.org/stable/modules/covariance.html). The exact numerical choices are documented in docs/analysis_plan.md and config/analysis.json.''')
    if not image_review_done:
        import re
        blocks=[(title,re.sub(r'All 60 original ND2 files were read.*?across passages\.',
                  'Original-image review was skipped for this run. Source identity is checked against verified exported measurements; no new pixel-level or visual image checks were performed. Segmentation accuracy remains unverified.',body,flags=re.S)) if title=='Quality control and concerns' else (title,body) for title,body in blocks]
    md='# Expanded Cell Painting analysis — 2026-09-16\n\n'+ '\n\n'.join('## '+title+'\n\n'+body for title,body in blocks)
    md+='\n\n## Figure guide\n\n'+'\n\n'.join(f'### {r["title"]}\n\n![{r["title"]}](05_figures/{r["file"]})\n\n{r["caption"]}' for r in FIGURES)
    (ROOT/'results/REPORT.md').write_text(md,encoding='utf-8')
    # Render through the installed Markdown library when available; otherwise plain paragraphs remain readable.
    try:
        import markdown
        body=markdown.markdown(md,extensions=['tables'])
    except ImportError:
        import re
        body=''
        for title,txt in blocks:
            body+='<h2>'+html.escape(title)+'</h2>'
            for paragraph in txt.split('\n\n'):
                if paragraph.startswith('| '):body+=display.to_html(index=False,escape=True)
                else:
                    txt2=html.escape(paragraph);txt2=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',txt2);txt2=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<a href="\2">\1</a>',txt2)
                    body+='<p>'+txt2+'</p>'
        body+='<h2>Figure guide</h2>'
        for f in FIGURES:body+='<h3>'+html.escape(f['title'])+'</h3><a href="05_figures/'+f['file']+'"><img src="05_figures/'+f['file']+'"></a><p>'+html.escape(f['caption'])+'</p>'
    reviews=''.join('<li><a href="05_figures/image_review/'+p.name+'">'+p.stem.replace('_',' ')+'</a></li>' for p in sorted((ROOT/'results/05_figures/image_review').glob('*.png')))
    page='<!doctype html><html><head><meta charset="utf-8"><title>Expanded Cell Painting analysis</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1180px;margin:36px auto;padding:0 24px;color:#243447}h1,h2{color:#173f59}h2{border-bottom:1px solid #cbd5df;padding-top:25px}p{max-width:1000px}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:9px;border:1px solid #d7e0e8;text-align:left}th{background:#edf3f7}a{color:#166691}.banner{background:#eef5f7;padding:18px;border-left:5px solid #2e718a}li{margin:6px 0}</style></head><body><h1>Expanded Cell Painting analysis</h1><div class="banner">4,866 cells · 60 wells · 97 retained features · 18 UMAP settings<br>Exploratory phenotypes; no prespecified test family yields q&lt;0.05.</div><p><a href="05_figures/all_analysis_figures.pdf">All analysis figures (PDF)</a> · <a href="analysis_tables.xlsx">Analysis tables (Excel)</a> · <a href="06_validation/validation_summary.json">Validation</a></p>'+body+'<h2>Original-image review</h2><p>High-distance and typical cells from the same wells; source identities in the image-review registry. Orange circles mark nuclear centroids, not segmentation boundaries.</p><ul>'+reviews+'</ul></body></html>'
    (ROOT/'results/REPORT.html').write_text(page,encoding='utf-8')
    tables={'feature_decisions':panel,'well_QC':read('01_data_qc/well_qc.csv'),'well_responses':w,'primary_tests':tests,'compartment_tests':ct,'feature_driver_tests':read('02_responses/feature_driver_tests.csv'),'response_sensitivity':sens,'cluster_screens':screens,'cluster_parameters':read('03_states/parameter_scan.csv'),'cluster_feature_sensitivity':read('03_states/feature_panel_sensitivity.csv'),'state_proportions':read('03_states/well_state_proportions.csv'),'state_tests':read('03_states/state_proportion_tests.csv'),'UMAP_quality':uq,'figure_guide':pd.DataFrame(FIGURES)}
    with pd.ExcelWriter(ROOT/'results/analysis_tables.xlsx',engine='openpyxl') as writer:
        for name,d in tables.items():
            d.to_excel(writer,sheet_name=name,index=False);ws=writer.sheets[name];ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
            for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=min(60,max(14,max(len(str(c.value or '')) for c in list(col)[:40])+2))
    print('Report and',len(FIGURES),'analysis figures complete',flush=True)

def main():
    cells,panel,arrays=load_data();make_figures(cells,panel,arrays);report(cells,panel,arrays)

if __name__=='__main__':main()
