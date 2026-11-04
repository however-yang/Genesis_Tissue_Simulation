"""Localize the frozen step800 oracle discrepancy without changing its threshold."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
path=ROOT/'endoscopic_tissue/outputs/material-contact-next-20261031/additive/diagnostic-a2/additive-0800.bin'
with path.open('rb') as file:
    def get(dtype,count): return np.fromfile(file,dtype=dtype,count=count)
    _,n,top,coarse=get('<i4',4)
    mask=(get('<u2',n)>>7&1).astype(bool);h=get('<f4',6*n).reshape(n,6)
    parents,inverses=[None],[None]
    for level in range(top):
        count,size=get('<i4',2);parents.append(get('<i4',size));inverses.append(get('<f4',9*count).reshape(count,3,3))
    dense=get('<f4',coarse*coarse).reshape(coarse,coarse)
    r=get('<f4',3*n).reshape(n,3);gpu=get('<f4',3*n).reshape(n,3)


def fine(dtype):
    xx,yy,zz,xy,xz,yz=h.astype(dtype).T
    a,b,c=yy*zz-yz*yz,xz*yz-xy*zz,xy*yz-xz*yy
    d,e,f=xx*zz-xz*xz,xy*xz-xx*yz,xx*yy-xy*xy
    det=xx*a+xy*b+xz*c;valid=mask&(det>1e-25)
    out=np.zeros((n,3),dtype=dtype);rhs=r.astype(dtype)
    out[:,0]=a*rhs[:,0]+b*rhs[:,1]+c*rhs[:,2]
    out[:,1]=b*rhs[:,0]+d*rhs[:,1]+e*rhs[:,2]
    out[:,2]=c*rhs[:,0]+e*rhs[:,1]+f*rhs[:,2]
    out[valid]*=(dtype(1)/det[valid])[:,None];out*=dtype(.7);out[~valid]=0
    return out,det


restricted=[r.astype(np.float64)]
for level in range(1,top+1):
    p=parents[level];valid=p>=0
    restricted.append(np.stack([np.bincount(p[valid],weights=restricted[-1][valid,a],minlength=len(inverses[level])) for a in range(3)],axis=1))
correction=(dense.astype(np.float64)@restricted[-1].ravel()).reshape(-1,3)
for level in range(top-1,0,-1):
    local=np.einsum('nij,nj->ni',inverses[level].astype(np.float64),restricted[level])*.7
    p=parents[level+1];valid=p>=0;local[valid]+=correction[p[valid]];correction=local
coarse_part=np.zeros((n,3));valid=mask&(parents[1]>=0);coarse_part[valid]=correction[parents[1][valid]]
f64,det64=fine(np.float64);f32,det32=fine(np.float32);reference=f64+coarse_part
delta=gpu-reference;errors=np.linalg.norm(delta,axis=1);worst=np.argsort(errors)[-12:][::-1]
records=[]
for i in worst:
    xx,yy,zz,xy,xz,yz=map(float,h[i]);matrix=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
    records.append(dict(node=int(i),error=float(errors[i]),h=h[i].tolist(),eigenvalues=np.linalg.eigvalsh(matrix).tolist(),
                        condition=float(np.linalg.cond(matrix)),det64=float(det64[i]),det32=float(det32[i]),
                        gpu=gpu[i].tolist(),reference=reference[i].tolist(),fine64=f64[i].tolist(),fine32=f32[i].tolist(),coarse=coarse_part[i].tolist()))
result=dict(snapshot=str(path),relative_error=float(np.linalg.norm(delta)/np.linalg.norm(reference)),
            error_norm=float(np.linalg.norm(delta)),fine32_error_norm=float(np.linalg.norm(f32-f64)),
            gpu_vs_fine32_plus_coarse_error_norm=float(np.linalg.norm(gpu-(f32+coarse_part))),worst=records)
(WORK/'additive-error-localization.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({**{k:v for k,v in result.items() if k!='worst'},'worst':records[:3]},indent=2))
