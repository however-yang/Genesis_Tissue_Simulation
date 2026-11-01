import ctypes as ct,json,sys,hashlib
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;D=W.parents[1]/'endoscopic_tissue/outputs/interaction-fixes-20261031'
def native_structure(path):
 text=path.read_text(encoding='utf8');start=text.index('struct Symmetric3 {');end=text.index('\n};',start)+4
 return text[start:end]
assert native_structure(W/'solve_probe_actual.cu') == native_structure(W/'package/overrides/vertex-r18/vendor/cuda/solver/material/unified_vertex.cuh'), 'Probe differs from packaged native solver'
files=sorted((W/'package/evidence/numerics').glob('*.bin'))
assert len(files)==2,'Missing recorded matrix regression fixtures'
for fixture in json.loads((W/'package/evidence/numerics/manifest.json').read_text()):
 assert hashlib.sha256((W/'package/evidence/numerics'/fixture['file']).read_bytes()).hexdigest()==fixture['sha256'], 'Recorded numerical fixture changed'
rows=[];names=[]
for p in files:
 a=np.fromfile(p,'f4');rows.append(np.r_[a[27:33],a[24:27]]);names.append(str(p))
rng=np.random.default_rng(20261031)
# Orthogonally rotated SPD matrices, plus strongly anisotropic rank-one contact terms.
for condition in [1,10,100,1000,15000,100000,1000000]:
 for j in range(128):
  Q,_=np.linalg.qr(rng.normal(size=(3,3)));H=(Q@np.diag([1,np.sqrt(condition),condition])@Q.T).astype('f4');r=rng.normal(size=3).astype('f4')
  rows.append(np.r_[H[0,0],H[1,1],H[2,2],H[0,1],H[0,2],H[1,2],r]);names.append('synthetic-condition-'+str(condition))
x=np.ascontiguousarray(rows,dtype='f4');H=np.empty((len(x),3,3));H[:,0,0]=x[:,0];H[:,1,1]=x[:,1];H[:,2,2]=x[:,2];H[:,0,1]=H[:,1,0]=x[:,3];H[:,0,2]=H[:,2,0]=x[:,4];H[:,1,2]=H[:,2,1]=x[:,5];rhs=x[:,6:].astype(float);truth=np.linalg.solve(H,rhs[...,None])[...,0]
dll=ct.CDLL(str(W/'solve_probe_actual.dll'));fn=dll.RunSolveProbe;ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');fn.argtypes=[ptr,ptr,ct.c_int,ct.c_int];fn.restype=ct.c_int
out=[]
for mode,label in enumerate(['legacy-adjugate','float-ldlt','double-ldlt','guarded-double-ldlt','actual-native-vertex']):
 y=np.zeros((len(x),3),dtype='f4');code=fn(x,y,len(x),mode);assert code==0,code
 residual=np.linalg.norm(np.einsum('nij,nj->ni',H,y)+(-rhs),axis=1)/np.maximum(np.linalg.norm(rhs,axis=1),1e-30)
 relative=np.linalg.norm(y-truth,axis=1)/np.maximum(np.linalg.norm(truth,axis=1),1e-30)
 if mode==4:
  assert np.all(np.einsum('ni,ni->n',rhs,y)>0), 'Non-descent direction for SPD matrix'
  assert float(relative.max())<.005, ('Actual solver accuracy regression',float(relative.max()))
 items=[dict(name=n,relative_residual=float(residual[i]),relative_step_error=float(relative[i])) for i,n in enumerate(names)]
 out.append(dict(solver=label,finite=bool(np.isfinite(y).all()),recorded_cases=items[:len(files)],synthetic=[dict(name=n,n=int(np.sum(np.array(names)==n)),max_residual=float(max(residual[np.array(names)==n])),max_step_error=float(max(relative[np.array(names)==n]))) for n in sorted(set(names[len(files):]))]))
(W/'solve-probe-actual-results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
assert out[2]['finite'] and out[3]['finite']
for r in out[4]['recorded_cases']:assert r['relative_step_error']<1e-5 and r['relative_residual']<.005,r
