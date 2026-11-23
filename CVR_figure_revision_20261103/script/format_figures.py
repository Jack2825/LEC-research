"""Format approved artwork in place: header, panel letters, verified scale bars.

No plots are regenerated. No panel contents, scales, symbols, colours, image
pixels, panel order or panel placements are changed. All edits are PDF overlays
in explicitly audited rectangles. Outside those rectangles rendered pixels are
compared with the October 4 originals.
"""
from pathlib import Path
import argparse, csv, hashlib, json, re, shutil
import numpy as np
import pymupdf as fitz
from PIL import Image
from prepare import BASE, ROOT, OUT, SOURCE, sha, require_inputs

fitz.TOOLS.set_small_glyph_heights(True)
from release_fonts import arial
FONT=arial(bold=True)
REGULAR=arial()
PANEL_SIZE=18
HEADER=32
EDITS=[];CHECKS=[];INVENTORY=[]
FIGURES=OUT/'figures';REVIEW=OUT/'review'
for p in [FIGURES,REVIEW,OUT/'main',OUT/'supplementary']:
    p.mkdir(parents=True,exist_ok=True)

def spans(p):
    return [s for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]

def image_hashes(p):
    return sorted((x['width'],x['height'],x['digest'].hex()) for x in p.get_image_info(hashes=True))

def panel_records(p,expected,manual=None):
    found=[]
    for letter in expected:
        candidates=[s for s in spans(p) if s['text'].strip()==letter]
        if manual and letter in manual:
            found.append(dict(letter=letter,rect=list(manual[letter]),baseline=None))
        else:
            assert len(candidates)==1,(expected,letter,len(candidates))
            s=candidates[0]
            found.append(dict(letter=letter,rect=s['bbox'],baseline=s['origin'][1]))
    return found

def scale_bars(p,whole=False):
    """Bar lengths use physical crop width, not resampled PDF pixel width."""
    rows=list(csv.DictReader((SOURCE/'calibration/S7A_calibrated_crop_records.csv').open(encoding='utf-8-sig')))
    imgs=[r for r in p.get_image_info() if r['width']==281 and r['height'] in [219,281,252]]
    # The whole figure contains these same 16 images once, in its original A slot.
    assert len(imgs)==16,len(imgs)
    imgs=sorted(imgs,key=lambda x:(round(x['bbox'][1],1),x['bbox'][0]))
    records=[]
    for i,img in enumerate(imgs):
        source=rows[i//4];bbox=fitz.Rect(img['bbox'])
        physical_width=float(source['crop_width'])*float(source['um_per_pixel'])
        length=20/physical_width*bbox.width
        y=bbox.y1-10;x1=bbox.x1-10;x0=x1-length
        text='20 µm';size=8 if not whole else 7
        font=fitz.Font(fontfile=REGULAR);tw=font.text_length(text,fontsize=size)
        rect=fitz.Rect(min(x0,x1-length/2-tw/2)-2,y-size-7,max(x1,x1-length/2+tw/2)+2,y+3)
        records.append(dict(kind='scale_bar',image_index=i,cell_id=source['cell_id'],
          length_um=20,um_per_acquisition_pixel=float(source['um_per_pixel']),
          native_crop_width_px=int(source['crop_width']),pdf_image_width_px=img['width'],
          image_bbox=list(bbox),bar=[x0,y,x1,y],text_origin=[x1-length/2-tw/2,y-4],
          font_size=size,rect=list(rect)))
    return records

def compare(old,new,rects,label,page_number):
    # Compare at the identical PDF placement to remove subpixel rounding from the
    # page-height translation. This reference contains only untouched source artwork.
    reference=fitz.open();rp=reference.new_page(width=old.rect.width,height=old.rect.height+HEADER)
    rp.show_pdf_page(fitz.Rect(0,HEADER,old.rect.width,old.rect.height+HEADER),old.parent,old.number)
    a=rp.get_pixmap(matrix=fitz.Matrix(1,1),clip=fitz.Rect(0,HEADER,old.rect.width,HEADER+old.rect.height),alpha=False)
    b=new.get_pixmap(matrix=fitz.Matrix(1,1),clip=fitz.Rect(0,HEADER,old.rect.width,HEADER+old.rect.height),alpha=False)
    aa=np.frombuffer(a.samples,dtype=np.uint8).reshape(a.height,a.width,3)
    bb=np.frombuffer(b.samples,dtype=np.uint8).reshape(b.height,b.width,3)
    assert aa.shape==bb.shape,(label,aa.shape,bb.shape)
    keep=np.ones(aa.shape[:2],dtype=bool)
    for rect in rects:
        r=fitz.Rect(rect)+(-3,-3,3,3)
        keep[max(0,int(np.floor(r.y0))):min(a.height,int(np.ceil(r.y1))+1),max(0,int(np.floor(r.x0))):min(a.width,int(np.ceil(r.x1))+1)]=False
    delta=np.max(np.abs(aa.astype(np.int16)-bb.astype(np.int16)),axis=2)
    changed=int(np.sum((delta>3)&keep));checked=int(keep.sum())
    images_ok=image_hashes(old)==image_hashes(new)
    assert changed==0,(label,page_number,changed,int(delta[keep].max()))
    assert images_ok,(label,'image payload changed')
    new_text=new.get_text()
    rec=dict(figure=label,page=page_number,checked_pixels_outside_overlays=checked,
             changed_pixels_above_3_levels_outside_overlays=changed,
             image_payloads_and_dimensions_unchanged=images_ok,
             source_text_retained=all(s['text'] in new_text for s in spans(old)),
             source_width_pt=old.rect.width,source_height_pt=old.rect.height,
             page_shift_pt=HEADER,panel_letter_size_pt=PANEL_SIZE)
    CHECKS.append(rec)
    reference.close()

def format_pdf(src,dest,label,expected,manual=None,bar_pages=(),whole=False):
    source=fitz.open(src);result=fitz.open()
    for i,p in enumerate(source):
        letters=expected[i] if isinstance(expected,list) else expected
        labels=panel_records(p,letters,manual)
        bars=scale_bars(p,whole) if i in bar_pages else []
        out=result.new_page(width=p.rect.width,height=p.rect.height+HEADER)
        out.show_pdf_page(fitz.Rect(0,HEADER,p.rect.width,p.rect.height+HEADER),source,i)
        out.insert_font(fontname='ArialBold',fontfile=FONT)
        out.insert_font(fontname='Arial',fontfile=REGULAR)
        out.insert_text((16,23),label,fontname='ArialBold',fontsize=18,color=(0,0,0))
        masks=[]
        for r in labels:
            old_rect=fitz.Rect(r['rect'])+(-.7,-.7,.7,.7)
            out.draw_rect(old_rect+(0,HEADER,0,HEADER),fill=(1,1,1),color=None)
            # Same upper-left anchor as the old letter, with a common physical type size.
            x=max(5,fitz.Rect(r['rect']).x0-10);top=fitz.Rect(r['rect']).y0
            baseline=top+PANEL_SIZE*.78
            text_rect=fitz.Rect(x,baseline-PANEL_SIZE,x+18,baseline+PANEL_SIZE*.3)
            out.insert_text((x,baseline+HEADER),r['letter'],fontname='ArialBold',fontsize=PANEL_SIZE)
            union=old_rect|text_rect;masks.append(list(union))
            EDITS.append(dict(figure=label,page=i+1,kind='panel_letter_format',letter=r['letter'],rect=list(union),font_size=PANEL_SIZE))
        for r in bars:
            x0,y,x1,_=r['bar'];y+=HEADER
            out.draw_line((x0,y),(x1,y),color=(0,0,0),width=4)
            out.draw_line((x0,y),(x1,y),color=(1,1,1),width=2)
            tx,ty=r['text_origin'];ty+=HEADER
            # Black backing affects only the new scale-label area, not source pixels.
            size=r['font_size'];tw=fitz.Font(fontfile=REGULAR).text_length('20 µm',fontsize=size)
            out.draw_rect(fitz.Rect(tx-1,ty-size,tx+tw+1,ty+2),fill=(0,0,0),color=None)
            out.insert_text((tx,ty),'20 µm',fontname='Arial',fontsize=size,color=(1,1,1))
            masks.append(r['rect']);EDITS.append(dict(figure=label,page=i+1,**r))
        compare(p,out,masks,label,i+1)
    dest.parent.mkdir(parents=True,exist_ok=True)
    result.set_metadata({'title':label,'subject':'Original October 4 artwork; numbering, lettering and calibrated scale bars only'})
    result.save(dest,garbage=4,deflate=True)
    for i,p in enumerate(result):
        width=1200;factor=width/p.rect.width
        p.get_pixmap(matrix=fitz.Matrix(factor,factor),alpha=False).save(dest.with_name(dest.stem+('' if len(result)==1 else f'_page{i+1:02d}')+'.preview.png'))
    INVENTORY.append(dict(figure=label,source=src.relative_to(BASE).as_posix(),output=dest.relative_to(BASE).as_posix(),pages=len(result),sha256=sha(dest)))
    result.close();source.close();print('Formatted',label,flush=True)

def verify_calibration():
    import nd2
    rows=list(csv.DictReader((SOURCE/'calibration/S7A_calibrated_crop_records.csv').open(encoding='utf-8-sig')))
    checks=[]
    for source in sorted({r['source_nd2'] for r in rows}):
        records=[r for r in rows if r['source_nd2']==source]
        assert sha(ROOT/source)==records[0]['source_sha256']
        with nd2.ND2File(ROOT/source) as d:um=d.voxel_size().x
        assert all(abs(um-float(r['um_per_pixel']))<1e-12 for r in records)
        checks.append(dict(source=source,sha256=records[0]['source_sha256'],um_per_pixel=um,matching_acquisition=True))
    (REVIEW/'scale_bar_calibration_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-native-images',action='store_true')
    args=parser.parse_args()
    require_inputs('formatting')
    if args.verify_native_images:
        verify_calibration()
    else:
        (REVIEW/'scale_bar_calibration_checks.json').write_text(json.dumps({'native_images_verified_this_run':False,'basis':'Checksummed recorded acquisition calibration; use --verify-native-images to recheck ND2.'},indent=2),encoding='utf-8')
    manual={1:{'A':[212,18,222.8,29]},4:{'B':[480.7,15,490.4,27],
      'D':[16,540,37,564],'F':[16,1320,37,1344],'G':[16,1674,39,1698]}}
    for n,count in [(1,3),(2,4),(3,6),(4,7),(5,4),(6,8),(7,5)]:
        stem=f'Fig{n:02d}';src=SOURCE/f'main/{stem}/{stem}.pdf'
        target=OUT/f'main/{stem}/{stem}.pdf'
        format_pdf(src,target,f'Figure {n}','ABCDEFGHIJKL'[:count],manual.get(n))
        shutil.copy2(target,FIGURES/target.name)
    for n,count in [(1,5),(2,5),(3,5),(4,7),(5,6),(6,6),(8,9)]:
        folder=next((SOURCE/'supplementary').glob(f'FigS{n:02d}_*'))
        if n in [6,8]:src=folder/f'FigS{n:02d}_combined.pdf'
        else:
            options=[p for p in folder.glob('*.pdf') if re.match(r'FigS\d\d_',p.name)]
            assert len(options)==1,options
            src=options[0]
        target=OUT/f'supplementary/FigS{n:02d}/FigS{n:02d}.pdf'
        format_pdf(src,target,f'Figure S{n}','ABCDEFGHIJKL'[:count])
        shutil.copy2(target,FIGURES/target.name)
    folder=SOURCE/'supplementary/FigS07_Cell_Painting'
    target=OUT/'supplementary/FigS07/FigS07.pdf'
    format_pdf(folder/'FigS07_Cell_Painting.pdf',target,'Figure S7',list('ABCDEFGHIJKL'),bar_pages=(0,))
    shutil.copy2(target,FIGURES/target.name)
    format_pdf(folder/'FigS07_Cell_Painting_whole.pdf',OUT/'supplementary/FigS07/FigS07_Cell_Painting_whole.pdf','Figure S7','ABCDEFGHIJKL',bar_pages=(0,),whole=True)
    with fitz.open(target) as d:
        for i,p in enumerate(d):
            out=fitz.open();out.insert_pdf(d,from_page=i,to_page=i)
            folder=OUT/f'supplementary/FigS07/panels';folder.mkdir(exist_ok=True)
            out.save(folder/f'FigS07{chr(65+i)}.pdf',garbage=4,deflate=True);out.close()
    # Original full-size main panels, with the same limited formatting changes.
    for src in sorted((SOURCE/'panels').glob('*.pdf')):
        n=int(src.stem[3:5]);letter=src.stem[-1]
        # Some approved panels use vector-outline letters; locate them separately.
        with fitz.open(src) as d:
            candidates=[s for s in spans(d[0]) if s['text'].strip()==letter]
        native_manual=None
        if len(candidates)!=1:
            with fitz.open(src) as d:
                glyphs=[z['rect'] for z in d[0].get_drawings() if z['rect'].x0<45 and z['rect'].y0<50 and 5<z['rect'].width<30 and 8<z['rect'].height<30]
                assert len(glyphs)==1,(src.name,glyphs)
                native_manual={letter:list(glyphs[0])}
        format_pdf(src,OUT/f'main/Fig{n:02d}/panels/{src.name}',f'Figure {n}',letter,native_manual)
    for n in [1,2,3,4,5,6,8]:
        folder=next((SOURCE/'supplementary').glob(f'FigS{n:02d}_*'))
        for src in sorted(folder.rglob('*.pdf')):
            match=re.match(r'FigS\d\d([A-Z])(?:_|\.)',src.name)
            if not match:continue
            format_pdf(src,OUT/f'supplementary/FigS{n:02d}/panels/{src.name}',f'Figure S{n}',match[1])
    for name,data in [('overlay_changes.json',EDITS),('artwork_preservation_checks.json',CHECKS),('figure_manifest.json',INVENTORY)]:
        (REVIEW/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('All checked artwork is unchanged outside explicit formatting overlays.',flush=True)

if __name__=='__main__':main()
