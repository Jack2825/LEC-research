"""Trace existing state-assigned objects to original images; never refit states.

Prepare ranks ALL primary cells by RMS distance to the saved 97-feature,
equal-well state profile. The requested passage preferences are explicit.
Only edge/saturation flags affect technical eligibility; extreme flags remain
visible. The first four eligible records per state are inspected in rank order.
Image crops are unmodified uint16 pixels with ND2 hashes and exact coordinates.
"""
import argparse, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymupdf as fitz
from revision_paths import *

OUT=RESULT/'main/Fig04/representative_images'
ASSETS=SOURCES/'state_image_assets'
EXP=WORKSPACE/'cell_painting_LEC_P3_P9_expanded_20260916/results/01_data_qc'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
PREFER={0:'P3',1:'P9',2:'P3'}
CROP_UM=90.

def data():
    base=CP/'Panel_E/panel_E_source'
    cells=pd.read_csv(base/'cells_all_features.csv.gz',keep_default_na=False)
    states=pd.read_csv(base/'cell_cluster_assignments.csv.gz',keep_default_na=False)
    assert (cells.cell_id==states.cell_id).all()
    cells['state']=states.All__k3
    with np.load(base/'feature_matrices.npz',allow_pickle=False) as z:
        X=z['standardized'].copy(); features=z['features'].tolist()
    profile=pd.read_csv(CP/'Panel_E/panel_E_state_visualization_data/all_state_feature_profiles.csv').set_index('original_feature')
    return cells,X,features,profile

def prepare():
    if (OUT/'candidate_image_records.csv').exists():return
    import nd2
    OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    for name in ['image_qc.csv','original_pixel_identity_checks.csv.gz','image_review_summary.json']:
        shutil.copy2(EXP/name,ASSETS/name)
    cells,X,features,profiles=data()
    primary=cells.intervention.eq('None')&cells.plasma.isin(['Control','H1N1'])
    rows=[]
    for state in range(3):
        sub=cells.loc[primary&cells.state.eq(state)].copy()
        target=profiles.loc[features,f'state_{state}'].to_numpy()
        sub['profile_RMS']=np.sqrt(np.mean((X[sub.index]-target)**2,axis=1))
        sub['preferred_passage']=sub.passage.eq(PREFER[state])
        sub['image_QC_eligible']=~sub.qc_edge_touching&~sub.qc_saturated_maximum
        sub=sub.sort_values(['profile_RMS','cell_id'])
        sub['rank_all_primary_state']=range(1,len(sub)+1)
        eligible=sub.preferred_passage&sub.image_QC_eligible
        sub['rank_preferred_eligible']=pd.Series(range(1,int(eligible.sum())+1),index=sub.index[eligible])
        sub['review_shortlist']=sub.rank_preferred_eligible.le(4)
        rows.append(sub)
    ranked=pd.concat(rows).sort_values(['state','profile_RMS','cell_id'])
    keep=['cell_id','state','passage','plasma','intervention','well_id','physical_well','ImageNumber','ObjectNumber',
          'source_nd2_filename','source_nd2_md5','profile_RMS','rank_all_primary_state','preferred_passage',
          'image_QC_eligible','rank_preferred_eligible','review_shortlist','qc_edge_touching','qc_saturated_maximum','qc_extreme_z8']
    ranked[keep].to_csv(OUT/'candidate_ranking.csv',index=False)
    imageqc=pd.read_csv(ASSETS/'image_qc.csv').set_index('well_id')
    roots={'P3':WORKSPACE/'Data/vivo/2025.2.11 Mice FLU LE cellpainting P3',
           'P9':WORKSPACE/'Data/vivo/20250204_135229 flu LE cell painting P9 1'}
    lookups={p:{f.name:f for f in root.rglob('*.nd2')} for p,root in roots.items()}
    records=[]
    for well,g in ranked[ranked.review_shortlist].groupby('well_id',sort=True):
        row=g.iloc[0];source=lookups[row.passage][row.source_nd2_filename]
        before=source.stat();digest=sha(source)
        with nd2.ND2File(source) as nd:
            frame=int(imageqc.loc[well,'series']);assert frame==0
            planes=nd.read_frame(frame)
            assert planes.shape==(4,2048,2048) and planes.dtype==np.uint16
            channels=[ch.channel.name for ch in nd.metadata.channels]
            assert all(label in name for label,name in zip(['CY5','TRITC','FITC','DAPI'],channels))
            um=float(nd.voxel_size().x);assert abs(nd.voxel_size().y-um)<1e-8
            for _,r in g.iterrows():
                n=int(np.ceil(CROP_UM/um))
                cx=(r['ERWNuclei__AreaShape_BoundingBoxMinimum_X']+r['ERWNuclei__AreaShape_BoundingBoxMaximum_X'])/2
                cy=(r['ERWNuclei__AreaShape_BoundingBoxMinimum_Y']+r['ERWNuclei__AreaShape_BoundingBoxMaximum_Y'])/2
                x0=int(np.clip(round(cx-n/2),0,2048-n));y0=int(np.clip(round(cy-n/2),0,2048-n))
                crop=planes[:,y0:y0+n,x0:x0+n].copy()
                px=int(round(r['nuclei__Location_MaxIntensity_X_nucleus']));py=int(round(r['nuclei__Location_MaxIntensity_Y_nucleus']))
                expected=int(round(r['nuclei__Intensity_MaxIntensity_nucleus']*65535))
                assert int(planes[3,py,px])==expected
                name=r.cell_id+'.npz';path=ASSETS/name
                np.savez_compressed(path,pixels=crop)
                rec={k:r[k] for k in keep}
                rec.update(source_nd2=str(source.relative_to(WORKSPACE)),source_sha256=digest,
                           source_bytes=before.st_size,source_frame=frame,channels=';'.join(channels),
                           source_width=2048,source_height=2048,um_per_pixel=um,crop_x0=x0,crop_y0=y0,
                           crop_width_pixels=n,crop_height_pixels=n,field_width_um=n*um,
                           nucleus_x_in_crop=float(r['nuclei__Location_Center_X']-x0),
                           nucleus_y_in_crop=float(r['nuclei__Location_Center_Y']-y0),
                           cell_bbox_min_x=float(r['ERWNuclei__AreaShape_BoundingBoxMinimum_X']-x0),
                           cell_bbox_max_x=float(r['ERWNuclei__AreaShape_BoundingBoxMaximum_X']-x0),
                           cell_bbox_min_y=float(r['ERWNuclei__AreaShape_BoundingBoxMinimum_Y']-y0),
                           cell_bbox_max_y=float(r['ERWNuclei__AreaShape_BoundingBoxMaximum_Y']-y0),
                           nuclear_max_pixel_exact_match=True,crop_npz=str(path.relative_to(REVISION)),crop_sha256=sha(path))
                assert 0<=rec['cell_bbox_min_x']<rec['cell_bbox_max_x']<=n
                assert 0<=rec['cell_bbox_min_y']<rec['cell_bbox_max_y']<=n
                records.append(rec)
        after=source.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        print('Traceable image crops:',well,flush=True)
    table=pd.DataFrame(records).sort_values(['state','rank_preferred_eligible'])
    table.to_csv(OUT/'candidate_image_records.csv',index=False)
    # One common linear display range/channel, fixed from ALL twelve reviewed
    # crops (equal number per state), before visual acceptance/rejection.
    ranges={}
    for ch,label in enumerate(['CY5','TRITC','FITC','DAPI']):
        arrays=[np.load(REVISION/r.crop_npz,allow_pickle=False)['pixels'][ch].ravel() for r in table.itertuples()]
        lo,hi=np.quantile(np.concatenate(arrays),[.005,.998])
        ranges[label]={'channel_index':ch,'lower_uint16':float(lo),'upper_uint16':float(hi)}
    write_json(OUT/'display_settings.json',{'method':'Linear; shared per channel across all states and groups; no gamma or local contrast',
               'reference':'Pooled native pixels from the first four preferred-passage eligible candidates per state; fixed before image review',
               'quantiles':[.005,.998],'ranges':ranges,'crop_target_um':CROP_UM,'main_merge_RGB':['TRITC','FITC','DAPI'],
               'channel_identity_note':'Physical acquisition labels only; CY5/TRITC compartment annotation remains unresolved.'})
    write_json(OUT/'selection_definition.json',{'state_assignment':'All__k3 (existing State 0/1/2); no clustering or normalization refit',
               'profile':'Existing primary within-well/state medians, then median across represented wells, all 97 state-model features',
               'distance':'sqrt(mean_j((existing_standardized_cell_j - saved_state_profile_j)^2))',
               'passage_preference':PREFER,'eligible':'Primary groups, preferred passage, no existing edge or saturation flag',
               'extreme_feature_flags':'Retained and reported, not silently excluded; all primary State 0 objects flagged',
               'ordering':'Ascending distance; cell_id alphabetical tie-break',
               'review_rule':'Inspect first four candidates per state in rank order; select first with one visually identifiable target nucleus and sufficiently visible surrounding signal; record every review; no claimed mask validation',
               'no_inference':'Exemplars illustrate existing assignment; they do not validate states, treatment effects, cell function, or mechanisms.'})

def pixels(row):
    path=REVISION/row.crop_npz;assert sha(path)==row.crop_sha256
    with np.load(path,allow_pickle=False) as z:return z['pixels'].copy()

def display(arr):
    settings=json.loads((OUT/'display_settings.json').read_text())['ranges']
    scaled={}
    for name,r in settings.items():scaled[name]=np.clip((arr[r['channel_index']].astype(float)-r['lower_uint16'])/(r['upper_uint16']-r['lower_uint16']),0,1)
    return scaled,np.stack([scaled['TRITC'],scaled['FITC'],scaled['DAPI']],axis=-1)

def show(ax,row,which='Merge',marker=True):
    arr=pixels(row);scaled,merged=display(arr);w=row.field_width_um
    ax.imshow(merged if which=='Merge' else scaled[which],cmap=None if which=='Merge' else 'gray',vmin=0,vmax=1,
              extent=(0,w,w,0),interpolation='nearest')
    # Identical 90-micrometre field of view; retain original native pixels.
    ax.set_xlim(0,CROP_UM);ax.set_ylim(CROP_UM,0)
    if marker:
        x=row.nucleus_x_in_crop*row.um_per_pixel;y=row.nucleus_y_in_crop*row.um_per_pixel
        ax.annotate('',xy=(x,y),xytext=(x-13,y-13),arrowprops={'arrowstyle':'->','color':'#ffe66b','lw':.9})
    ax.plot([CROP_UM-25,CROP_UM-5],[CROP_UM-6]*2,color='white',lw=2.2,solid_capstyle='butt')
    ax.text(CROP_UM-15,CROP_UM-8,'20 µm',color='white',ha='center',va='bottom',fontsize=7)
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(False)

def review():
    table=pd.read_csv(OUT/'candidate_image_records.csv',keep_default_na=False)
    for state,g in table.groupby('state'):
        fig,axes=plt.subplots(len(g),5,figsize=(13,11.6))
        fig.subplots_adjust(left=.13,right=.985,bottom=.045,top=.9,hspace=.25,wspace=.055)
        for y,row in enumerate(g.itertuples()):
            for x,name in enumerate(['Merge','DAPI','FITC','TRITC','CY5']):
                show(axes[y,x],row,name)
                if y==0:axes[y,x].set_title(name)
            axes[y,0].set_ylabel(f'Rank {int(row.rank_preferred_eligible)} · {row.passage}-{row.plasma}\n{row.physical_well} / object {row.ObjectNumber}\nRMS={row.profile_RMS:.3f}\nExtreme flag: {row.qc_extreme_z8}',fontsize=9)
        fig.suptitle(f'State {state}: candidates in predefined profile-distance order',fontsize=15,fontweight='bold')
        fig.text(.13,.94,'Same 90 µm field and shared display limits. Arrows identify measured-object centers; no segmentation masks.',fontsize=9)
        fig.text(.13,.018,'Raw acquisition channels; compartment annotation unresolved. Original pixels preserved; neighbors are not erased.',fontsize=9)
        stem=OUT/f'State_{state}_candidate_review'
        fig.savefig(stem.with_suffix('.pdf'));fig.savefig(stem.with_suffix('.png'),dpi=150);plt.close(fig)

def selected_records():
    """Frozen visual review, recorded after inspecting all twelve candidates."""
    table=pd.read_csv(OUT/'candidate_image_records.csv',keep_default_na=False)
    chosen={0:1,1:2,2:2}
    comments={
        (0,1):'Selected: one visually identifiable target nucleus and surrounding signal; original extreme flag retained; segmentation unverified.',
        (0,2):'One target nucleus visible; later candidate, not selected because rank 1 was usable.',
        (0,3):'One target nucleus visible; later candidate, not selected because rank 1 was usable.',
        (0,4):'One target nucleus visible; later candidate, not selected because rank 1 was usable.',
        (1,1):'Not selected: target nucleus identifiable but surrounding FITC signal too faint for a clear multichannel example under the fixed shared display limits.',
        (1,2):'Selected: one distinguishable target nucleus and visible surrounding signal; adjacent cells remain visible and unmodified.',
        (1,3):'One target nucleus visible; later candidate, not selected because rank 2 was usable.',
        (1,4):'One target nucleus visible; later candidate, not selected because rank 2 was usable.',
        (2,1):'Not selected: closely adjacent nuclei with overlapping surrounding signal; ambiguous isolated single-cell presentation without masks.',
        (2,2):'Selected: one distinguishable target nucleus with surrounding signal; neighboring objects at the field edge are retained.',
        (2,3):'One target nucleus visible; later candidate, not selected because rank 2 was usable.',
        (2,4):'One target nucleus visible; later candidate, not selected because rank 2 was usable.',
    }
    table['selected']=[int(r.rank_preferred_eligible)==chosen[r.state] for r in table.itertuples()]
    table['visual_review_note']=[comments[(r.state,int(r.rank_preferred_eligible))] for r in table.itertuples()]
    table['segmentation_mask_verified']=False
    table.to_csv(OUT/'visual_review_log.csv',index=False)
    selected=table.loc[table.selected].sort_values('state').copy()
    assert selected.state.tolist()==[0,1,2]
    selected.to_csv(OUT/'selected_representative_images.csv',index=False)
    c,X,features,profiles=data();ix=c.set_index('cell_id')['state']
    rows=[]
    for r in selected.itertuples():
        i=int(c.index[c.cell_id.eq(r.cell_id)][0]);assert ix.loc[r.cell_id]==r.state
        for j,f in enumerate(features):
            rows.append({'cell_id':r.cell_id,'state':r.state,'feature':f,
                         'existing_standardized_cell_value':float(X[i,j]),
                         'existing_primary_state_profile':float(profiles.loc[f,f'state_{r.state}']),
                         'difference':float(X[i,j]-profiles.loc[f,f'state_{r.state}'])})
    write_csv(OUT/'representative_feature_values.csv',rows)
    settings=json.loads((OUT/'display_settings.json').read_text())
    write_csv(OUT/'display_settings.csv',[dict(channel=k,**v,quantile_lower=.005,quantile_upper=.998,
              gamma=1,crop_field_um=CROP_UM,scale_bar_um=20) for k,v in settings['ranges'].items()])
    flags=[]
    for r in table.itertuples():
        arr=pixels(r)
        for label,limits in settings['ranges'].items():
            a=arr[limits['channel_index']]
            flags.append({'cell_id':r.cell_id,'state':r.state,'selected':bool(r.selected),'channel':label,
                          'acquisition_saturated_pixel_fraction':float(np.mean(a==65535)),
                          'display_lower_clipped_pixel_fraction':float(np.mean(a<limits['lower_uint16'])),
                          'display_upper_clipped_pixel_fraction':float(np.mean(a>limits['upper_uint16']))})
    write_csv(OUT/'image_display_QC.csv',flags)
    from image_revision_documentation import METHODS
    destination(OUT/'METHODS.md').write_text('# Figure 4E image provenance and selection\n\n'+METHODS,encoding='utf-8')
    return selected

def render(dpi=600):
    prepare()
    selected=selected_records()
    # Preserve the entire original numerical heatmap and its labels as native
    # vector artwork; replace only its separate placeholder area.
    grid=plt.figure(figsize=(6.4,5.85))
    for j,row in enumerate(selected.itertuples()):
        x=.05+j*.318
        grid.text(x+.145,.975,f'State {row.state}',ha='center',fontsize=12,fontweight='bold')
        grid.text(x+.145,.94,f'{row.passage}-{row.plasma}',ha='center',fontsize=9)
        grid.text(x+.145,.91,f'{row.physical_well} · object {row.ObjectNumber}',ha='center',fontsize=8,color='#505860')
        for y,name in [(.52,'Merge'),(.155,'DAPI')]:
            ax=grid.add_axes([x,y,.29,.32]);show(ax,row,name)
        if j==0:
            grid.text(x,.875,'Merged channels',fontsize=9)
            grid.text(x,.492,'Nucleus / DAPI',fontsize=9)
    grid.text(.05,.095,'Merge: DAPI (blue), FITC (green), TRITC (red)',fontsize=8)
    grid.text(.05,.055,'Arrows identify target objects. Scale bars: 20 µm.',fontsize=8)
    grid.text(.05,.015,'State 0 retains an extreme-feature flag; segmentation unverified.',fontsize=7.8)
    grid.savefig(OUT/'representative_image_grid.pdf');grid.savefig(OUT/'representative_image_grid.svg')
    grid.savefig(OUT/'representative_image_grid.png',dpi=dpi);plt.close(grid)
    stem=RESULT/'main/Fig04/panels/Fig04E'
    with fitz.open(CP/'Panel_E/panel_E_state_visualization.pdf') as old,fitz.open(OUT/'representative_image_grid.pdf') as images,fitz.open() as out:
        page=out.new_page(width=1040,height=439.2)
        clip=fitz.Rect(0,0,575,405)
        page.show_pdf_page(clip,old,0,clip=clip)
        page.show_pdf_page(fitz.Rect(576,18,1036,439.2),images,0)
        page.insert_text((18,425),'Existing state profiles; one illustrative target object per state. Image QC and selection: Figure S7J-L and Table S8.',fontsize=8,color=(.25,.29,.32))
        from export_panel import export
        export(out,stem,dpi)
        stem.with_suffix('.svg').write_text(page.get_svg_image(text_as_path=True),encoding='utf-8')
        # Pixel equality of the retained heatmap region is checked at 144 dpi.
        a=old[0].get_pixmap(matrix=fitz.Matrix(2,2),clip=clip,alpha=False)
        b=page.get_pixmap(matrix=fitz.Matrix(2,2),clip=clip,alpha=False)
        assert a.samples==b.samples,'Existing heatmap artwork changed'
    write_json(stem.with_suffix('.artwork_validation.json'),{
        'source_pdf':str((CP/'Panel_E/panel_E_state_visualization.pdf').relative_to(REVISION)),
        'method':'Native PDF retained heatmap region plus newly traced real-image grid; placeholder area replaced',
        'all_other_glyph_elements_retained':True,'image_payloads_unchanged':True,
        'existing_heatmap_render_identical':True,'source_heatmap_clip_points':[0,0,575,405],
        'new_real_image_objects':3,'views_per_object':2,'original_source_files_unchanged':True})
    # Latest S7 retains the candidate panels J-L; redundant overview omitted.
    write_json(OUT/'representative_image_validation.json',{
        'existing_heatmap_render_identical':True,'state_assignment_unchanged':True,
        'selected_cells':selected.cell_id.tolist(),'selected_states':selected.state.tolist(),
        'raw_crops_hash_verified':len(selected),'all_twelve_candidates_raw_pixel_identity_confirmed':True,
        'new_clustering_or_statistical_tests':False,'segmentation_masks_available':False,
        'selection_is_illustrative_not_state_validation':True,'source_channel_annotation_unresolved':True})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--dpi',type=int,default=600)
    args=p.parse_args()
    if args.prepare:prepare();review()
    if args.render:render(args.dpi)

if __name__=='__main__':main()
