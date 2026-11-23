"""Typeset the existing two-column/eight-card abstract at CVR's specified size.

Uses the original archived code's CONTENT and the three already-approved
October 4 text replacements. No word, card, connector or scientific label is
added or removed. Only font sizes, wrapping and spacing are adjusted.
"""
from pathlib import Path
from collections import Counter
import ast, json, re, xml.etree.ElementTree as ET
import pymupdf as fitz
from prepare import BASE, ROOT, OUT, SOURCE, require_inputs
from release_fonts import arial

W,H=180/25.4*72,110/25.4*72
REG=fitz.Font(fontfile=arial())
BOLD=fitz.Font(fontfile=arial(bold=True))

def rgb(s):return tuple(int(s[i:i+2],16)/255 for i in [1,3,5])

def wrap(text,width,font,size):
    result=[]
    for para in text.split('\n'):
        line=''
        for word in para.split():
            trial=(line+' '+word).strip()
            if font.text_length(trial,fontsize=size)>width and line:
                result.append(line);line=word
            else:line=trial
        result.append(line)
    assert all(font.text_length(s,fontsize=size)<=width+.01 for s in result),(text,width,result)
    return result

def main():
    require_inputs('formatting')
    content=json.loads((ROOT/'config/graphical_abstract_content.json').read_text(encoding='utf-8'))
    for r in content:
        if r['content_order']==7:
            r['title']='Altered adaptive inflammatory\nresponsiveness'
            r['body']='Influenza challenge: divergent functional and\norganelle responses in young versus aged LECs'
        if r['content_order']==9:
            r['body']='Independent spatial transcriptomics provides tissue-level support for selected disease-associated LEC programs'
    title='Integrated model of cardiac LEC remodeling in aging and cardiovascular disease'
    categories=['MICROENVIRONMENTAL\nCUES','TRANSCRIPTIONAL\nREGULATION','LEC STATE\nREMODELING','FUNCTIONAL\nCONSEQUENCES']
    headings=['PHYSIOLOGICAL AGING','CARDIOVASCULAR DISEASE']
    # Character inventory from the approved SVG proves the exact text is retained,
    # including outline glyphs that are not extractable as PDF live text.
    svg=ET.fromstring((SOURCE/'main/Graphic_Abstract/Graphic_Abstract.svg').read_text(encoding='utf-8'))
    source_text=''.join(e.get('data-text','') for e in svg.iter())
    source_text+=''.join(''.join(e.itertext()) for e in svg.iter() if e.tag.endswith('}text'))
    expected='GRAPHIC ABSTRACT'+title+''.join(categories)+''.join(headings)+''.join(r['title']+r['body']+r['badge'] for r in content)
    # The embedded Arial font maps the same visible hyphen/semicolon glyphs to
    # soft hyphen/Greek question mark during PDF extraction; normalise aliases.
    normalize=lambda s:Counter(c for c in s.translate(str.maketrans({'\u00ad':'-','\u037e':';'})) if not c.isspace())
    assert normalize(source_text)==normalize(expected),(normalize(source_text)-normalize(expected),normalize(expected)-normalize(source_text))
    doc=fitz.open();p=doc.new_page(width=W,height=H)
    p.insert_font(fontname='Arial',fontfile=arial())
    p.insert_font(fontname='ArialBold',fontfile=arial(bold=True))
    recorded=[]
    def text(x,y,s,size=8,bold=False,color='#20252A',align='left'):
        font=BOLD if bold else REG
        if align=='center':x-=font.text_length(s,fontsize=size)/2
        if align=='right':x-=font.text_length(s,fontsize=size)
        assert x>=0 and x+font.text_length(s,fontsize=size)<=W+.01
        p.insert_text((x,y),s,fontname='ArialBold' if bold else 'Arial',fontsize=size,color=rgb(color))
        recorded.append(dict(text=s,size=size,x=x,y=y))
    text(8,10,'GRAPHIC ABSTRACT',8,True,'#59636C')
    title_lines=wrap(title,W-16,BOLD,9.5)
    for i,line in enumerate(title_lines):text(8,23+i*10,line,9.5,True)
    branch_y=34 if len(title_lines)==1 else 44
    coords=[(106,189,'#3973A9','#F4F8FB'),(305,W-311,'#C5544D','#FCF6F4')]
    for (x,width,color,fill),heading in zip(coords,headings):
        text(x+width/2,branch_y,heading,8.5,True,color,'center')
        p.draw_line((x,branch_y+4),(x+width,branch_y+4),color=rgb(color),width=.8)
    y=branch_y+10
    cards=[]
    for i,category in enumerate(categories):
        row=[]
        for r,(x,width,color,fill) in zip(content[2*i:2*i+2],coords):
            tw=width-15-(44 if r['badge'] else 0)
            titles=wrap(r['title'].replace('\n',' '),tw,BOLD,8)
            body=r['body'] if r['content_order']==5 else r['body'].replace('\n',' ')
            for start in ['Examples:','Prioritized pathways','SCENIC:','SCENIC supports','Shared remodeling','Improved cardiac']:
                body=body.replace(' '+start,'\n'+start)
            bodies=wrap(body,width-15,REG,8)
            title_slots=max(len(titles),2 if r['badge'] else 1)
            needed=6+8.2*title_slots+8.2*len(bodies)
            row.append((r,x,width,color,fill,titles,bodies,needed))
        height=max(z[-1] for z in row)
        lines=category.split('\n')
        for j,line in enumerate(lines):text(99,y+height/2+(j-(len(lines)-1)/2)*8.2+2.5,line,8,True,'#59636C','right')
        for r,x,width,color,fill,titles,bodies,_ in row:
            box=fitz.Rect(x,y,x+width,y+height)
            p.draw_rect(box,color=rgb(color),fill=rgb(fill),width=.6,radius=.04)
            p.draw_line((x+3,y+6),(x+3,y+height-6),color=rgb(color),width=1.3)
            ty=y+8
            for line in titles:text(x+8,ty,line,8,True,color);ty+=8.2
            ty=y+8+8.2*max(len(titles),2 if r['badge'] else 1)+2
            for line in bodies:text(x+8,ty,line,8,False);ty+=8.2
            if r['badge']:
                badge=fitz.Rect(x+width-43,y+4,x+width-4,y+16)
                p.draw_rect(badge,color=rgb(color),fill=(1,1,1),width=.5,radius=.06)
                text((badge.x0+badge.x1)/2,badge.y1-3,r['badge'],8,True,color,'center')
            cards.append(dict(order=r['content_order'],box=list(box),title=r['title'],body=r['body'],badge=r['badge']))
            if i<3:
                cx=x+width/2;bottom=y+height
                p.draw_line((cx,bottom),(cx,bottom+5),color=rgb(color),width=.6)
                p.draw_polyline([(cx-1.5,bottom+3.5),(cx,bottom+5),(cx+1.5,bottom+3.5)],color=rgb(color),width=.6)
        y+=height+5
    footer_lines=wrap(content[-1]['body'],W-124,BOLD,8)
    footer_height=7+len(footer_lines)*8.2
    assert y+footer_height<=H-3,(y,footer_height,H)
    p.draw_rect(fitz.Rect(106,y,W-6,y+footer_height),color=rgb('#71808B'),fill=rgb('#F3F6F7'),width=.6,radius=.06)
    for j,line in enumerate(footer_lines):text((106+W-6)/2,y+10+j*8.2,line,8,True,'#59636C','center')
    p.draw_line((300,branch_y+6),(300,y-2),color=rgb('#D8DEE2'),width=.5)
    assert normalize(p.get_text())==normalize(expected),(normalize(p.get_text())-normalize(expected),normalize(expected)-normalize(p.get_text()))
    dest=OUT/'main/Graphic_Abstract';dest.mkdir(parents=True,exist_ok=True)
    doc.set_metadata({'title':'Graphic Abstract','subject':'Same approved content and eight-card diagram; typography and physical dimensions only'})
    doc.save(dest/'Graphic_Abstract.pdf',garbage=4,deflate=True)
    p.get_pixmap(dpi=600).save(dest/'Graphic_Abstract.png')
    p.get_pixmap(dpi=170).save(dest/'Graphic_Abstract.preview.png')
    import shutil
    shutil.copy2(dest/'Graphic_Abstract.pdf',OUT/'figures/Graphic_Abstract.pdf')
    report=dict(width_mm=180,height_mm=110,min_font_pt=min(r['size'] for r in recorded),max_font_pt=max(r['size'] for r in recorded),
       exact_original_nonwhitespace_character_inventory_retained=True,existing_cards_retained=8,existing_stage_labels_retained=4,
       no_rewording=True,cards=cards,text=recorded)
    (OUT/'review/graphical_abstract_content_check.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Graphical abstract: original content, 180 × 110 mm, all text ≥8 pt.',flush=True)

if __name__=='__main__':main()
