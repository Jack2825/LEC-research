"""Read-only original-image QC and traceable review crops; never alter ND2 files."""
from core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import nd2

def main():
    cells,_,_=load_data();scores=pd.read_csv(ROOT/'results/02_responses/cell_response_scores.csv.gz',keep_default_na=False)
    cells=cells.merge(scores[['cell_id','All__distance','All__tail95','consensus_responder_95']],on='cell_id',validate='one_to_one')
    imageqc=pd.read_csv(ROOT/'results/01_data_qc/image_qc.csv').set_index('well_id')
    image_roots={'P3':WORK/'Data/vivo/2025.2.11 Mice FLU LE cellpainting P3','P9':WORK/'Data/vivo/20250204_135229 flu LE cell painting P9 1'}
    paths={}
    for passage,base in image_roots.items():
        for p in base.rglob('*.nd2'):
            key=(passage,p.name);assert key not in paths,key
            paths[key]=p
    selected=[]
    for group,g in cells.groupby(['passage','plasma','intervention'],sort=False):
        eligible=g.loc[~g.qc_edge_touching]
        tops=eligible.sort_values('All__distance',ascending=False).drop_duplicates('well_id').head(2)
        for _,top in tops.iterrows():
            same=eligible.loc[eligible.well_id.eq(top.well_id)&eligible.cell_id.ne(top.cell_id)]
            mid=same.iloc[np.argmin(abs(same['All__distance'].to_numpy()-same['All__distance'].median()))]
            for label,row in [('High distance',top),('Typical same well',mid)]:
                rec=row.to_dict();rec['review_selection']=label;selected.append(rec)
    selected=pd.DataFrame(selected);crops={};scales={};pixelrows=[];fieldrows=[];manifest=[]
    for index,(well,g) in enumerate(cells.groupby('well_id',sort=False)):
        row=g.iloc[0];path=paths[(row.passage,row.source_nd2_filename)]
        before=path.stat();digest=sha(path)
        with nd2.ND2File(path) as nd:
            assert dict(nd.sizes)['C']==4
            planes=nd.read_frame(int(imageqc.loc[well,'series']))
            assert planes.shape==(4,2048,2048)
            channels=[ch.channel.name for ch in nd.metadata.channels]
            assert ['CY5' in channels[0],'TRITC' in channels[1],'FITC' in channels[2],'DAPI' in channels[3]]==[True]*4
            for ch in range(4):
                a=planes[ch];lo,hi=np.quantile(a,[.01,.995]);scales[(well,ch)]=(lo,hi)
                fieldrows.append(dict(well_id=well,passage=row.passage,channel=channels[ch],pipeline_channel=['gram (unmeasured)','mitochondria','ER','nucleus'][ch],mean=float(a.mean()),median=float(np.median(a)),p99=float(np.quantile(a,.99)),saturated_fraction=float((a==65535).mean()),positions_in_ND2=int(nd.sizes['P']),exported_series=int(imageqc.loc[well,'series']),exported_frame=int(imageqc.loc[well,'frame'])))
            xx=np.rint(g['nuclei__Location_MaxIntensity_X_nucleus']).astype(int)
            yy=np.rint(g['nuclei__Location_MaxIntensity_Y_nucleus']).astype(int)
            expected=np.rint(g['nuclei__Intensity_MaxIntensity_nucleus']*65535).astype(int)
            actual=planes[3,yy,xx].astype(int)
            for cell,x,y,e,a in zip(g.cell_id,xx,yy,expected,actual):pixelrows.append(dict(cell_id=cell,well_id=well,x=int(x),y=int(y),exported_nuclear_max_uint16=int(e),original_pixel_uint16=int(a),exact_match=bool(e==a)))
            for _,r in selected.loc[selected.well_id.eq(well)].iterrows():
                x=int(round(r['nuclei__Location_Center_X']));y=int(round(r['nuclei__Location_Center_Y']));radius=160
                x0=max(0,x-radius);x1=min(2048,x+radius);y0=max(0,y-radius);y1=min(2048,y+radius)
                crops[r.cell_id]=(planes[:,y0:y1,x0:x1].copy(),x-x0,y-y0)
        assert path.stat().st_size==before.st_size and path.stat().st_mtime_ns==before.st_mtime_ns
        manifest.append(dict(path=os.path.relpath(path,ROOT),bytes=before.st_size,sha256=digest,role='optional original-image QC; not required for feature/statistical stages'))
        if index%15==0:print('Original-image QC',index,'/60',flush=True)
    out=ROOT/'results/05_figures/image_review';out.mkdir(exist_ok=True)
    for group,g in selected.groupby(['passage','plasma','intervention'],sort=False):
        fig,axes=plt.subplots(4,4,figsize=(12,12),layout='constrained')
        for i,(_,r) in enumerate(g.iterrows()):
            planes,x,y=crops[r.cell_id]
            for j,ch in enumerate([3,2,1,0]):
                ax=axes[i,j];lo,hi=scales[(r.well_id,ch)];ax.imshow(planes[ch],cmap='gray',vmin=lo,vmax=hi)
                ax.add_patch(Circle((x,y),7,fill=False,edgecolor='#ffad33',linewidth=.8));ax.set_xticks([]);ax.set_yticks([])
                if i==0:ax.set_title(['Nucleus / DAPI','ER / FITC','Mitochondria / TRITC','CY5 (no exported features)'][j],fontsize=10)
                if j==0:ax.set_ylabel(f'{r.physical_well}, object {r.ObjectNumber}\n{r.review_selection}\ndistance={r["All__distance"]:.2f}',fontsize=9)
        fig.suptitle(' | '.join(group)+'\nHigh-distance and typical cells in the same wells; first acquired field; no segmentation masks',fontsize=12)
        name='_'.join(group).replace('-','_')
        fig.savefig(out/(name+'.png'),dpi=140,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    pixel=pd.DataFrame(pixelrows);assert pixel.exact_match.all(),'Raw nuclear pixel identity mismatch'
    save(pixel,'01_data_qc/original_pixel_identity_checks.csv.gz');save(pd.DataFrame(fieldrows),'01_data_qc/original_field_qc.csv')
    save(pd.DataFrame(manifest),'01_data_qc/original_images_read_manifest.csv')
    keep=META+['review_selection','All__distance','All__tail95','qc_edge_touching','consensus_responder_95']
    save(selected[keep],'01_data_qc/image_review_cell_registry.csv')
    dump(dict(images_read=len(manifest),cell_pixel_checks=len(pixel),exact_matches=int(pixel.exact_match.sum()),review_cells=len(selected),review_pages=12,exported_fields_per_well=1,available_fields_per_ND2=3,segmentation_masks_available=False),'01_data_qc/image_review_summary.json')
    print('Image QC complete:',len(pixel),'matching nuclear pixel coordinates;',len(selected),'review crops',flush=True)

if __name__=='__main__':main()
