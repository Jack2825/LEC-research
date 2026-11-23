"""Shared numerical helpers. All output paths are inside this new project."""
from pathlib import Path
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import json
import hashlib
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
CFG = json.loads((ROOT/'config/analysis.json').read_text())
SEED = CFG['seed']
for _directory in ['01_data_qc','02_responses','03_states','04_embeddings','05_figures','06_validation']:
    (ROOT/'results'/_directory).mkdir(parents=True, exist_ok=True)
META = ['cell_id','well_id','passage','plasma','intervention','group','physical_well',
        'physical_well_id','raw_export_folder','ImageNumber','ObjectNumber','source_nd2_filename','source_nd2_md5']
FAMILIES = ['Cell shape','Nuclear shape','ER intensity','Mitochondrial intensity','Nuclear intensity']

def save(df, relative):
    path = ROOT/'results'/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path,index=False)

def dump(obj,relative):
    (ROOT/'results'/relative).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def weights(meta):
    w=1/meta.groupby('well_id').well_id.transform('size').to_numpy(float)
    return w/w.sum()

def wq(x,w,q):
    x=np.asarray(x,float); w=np.asarray(w,float)
    one=x.ndim==1
    if one:x=x[:,None]
    order=np.argsort(x,axis=0,kind='stable')
    v=np.take_along_axis(x,order,axis=0)
    cw=np.cumsum(w[order],axis=0)
    idx=(cw < q*cw[-1]).sum(axis=0).clip(0,len(x)-1)
    result=v[idx,np.arange(x.shape[1])]
    return result[0] if one else result

def robust_scale(x,w):
    c=wq(x,w,.5); s=1.4826*wq(abs(x-c),w,.5)
    fallback=(wq(x,w,.75)-wq(x,w,.25))/1.349
    s=np.where(s>1e-12,s,np.where(fallback>1e-12,fallback,1.))
    return c,s

def weighted_pca(x,w):
    c=np.average(x,axis=0,weights=w); z=x-c
    cov=(z*w[:,None]).T@z/w.sum()
    vals,vec=eigh(cov)
    vals=np.maximum(vals[::-1],0); vec=vec[:,::-1]
    return z@vec, vals/vals.sum(), vec, c

def n_pcs(ratio,target): return min(len(ratio),int(np.searchsorted(np.cumsum(ratio),target)+1))

def family(f):
    if f.startswith('ERWNuclei__AreaShape'):return 'Cell shape'
    if f.startswith('nuclei__AreaShape'):return 'Nuclear shape'
    if f.startswith('ERWONuclei__Intensity'):return 'ER intensity'
    if f.startswith('mitochondriaWONuclei__Intensity'):return 'Mitochondrial intensity'
    if f.startswith('nuclei__Intensity'):return 'Nuclear intensity'
    return 'Position/QC'

def bh(p):
    p=np.asarray(p,float); ans=np.full(len(p),np.nan); ok=np.flatnonzero(np.isfinite(p))
    order=ok[np.argsort(p[ok])]
    ans[order]=np.minimum.accumulate((p[order]*len(ok)/np.arange(1,len(ok)+1))[::-1])[::-1].clip(0,1)
    return ans

def contrasts(meta):
    out=[]
    for passage in ['P3','P9']:
        for name,a,b in [('H1N1_vs_Control',('Control','None'),('H1N1','None')),
                         ('VEGFC_in_H1N1',('H1N1','None'),('H1N1','VEGF-C')),
                         ('UC2288_in_H1N1',('H1N1','None'),('H1N1','UC2288'))]:
            select=lambda z:meta.loc[meta.passage.eq(passage)&meta.plasma.eq(z[0])&meta.intervention.eq(z[1]),'well_id'].unique().tolist()
            out.append((passage+'__'+name,select(a),select(b)))
    return out

def exact_p(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.ndim==1:a=a[:,None];b=b[:,None]
    z=np.concatenate([a,b]);obs=b.mean(0)-a.mean(0)
    combos=np.array(list(itertools.combinations(range(len(z)),len(a))))
    perm=(z.sum(0)-z[combos].sum(1))/len(b)-z[combos].mean(1)
    return (np.abs(perm)>=np.abs(obs)-1e-12).mean(0)

def conditional_ci(a,b,seed=SEED):
    rng=np.random.default_rng(seed); a=np.asarray(a);b=np.asarray(b)
    n=CFG['well_bootstraps']
    draws=b[rng.integers(len(b),size=(n,len(b)))].mean(1)-a[rng.integers(len(a),size=(n,len(a)))].mean(1)
    return np.quantile(draws,[.025,.975],axis=0)

def loo_direction(a,b):
    effect=np.mean(b)-np.mean(a)
    vals=[np.mean(b)-np.mean(np.delete(a,i)) for i in range(len(a))]
    vals += [np.mean(np.delete(b,i))-np.mean(a) for i in range(len(b))]
    return float(np.mean(np.sign(vals)==np.sign(effect))),float(min(vals)),float(max(vals))

def load_data():
    cells=pd.read_csv(ROOT/'results/01_data_qc/cells_all_features.csv.gz',keep_default_na=False,float_precision='round_trip')
    panel=pd.read_csv(ROOT/'results/01_data_qc/feature_decisions.csv',keep_default_na=False)
    arrays=np.load(ROOT/'results/01_data_qc/feature_matrices.npz')
    return cells,panel,arrays
