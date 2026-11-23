"""Current representative-panel PDF/PNG export; no historical figure assembler."""
from revision_paths import *

def export(doc, stem, dpi=600):
    stem=destination(stem)
    doc.save(stem.with_suffix('.pdf'),garbage=4,deflate=True)
    page=doc[0]
    page.get_pixmap(dpi=dpi,alpha=False).save(stem.with_suffix('.png'))
    page.get_pixmap(dpi=95,alpha=False).save(stem.parent/(stem.name+'.preview.png'))
    print('Figure:',stem.relative_to(REVISION),flush=True)

