"""Actual GPU old/new solves on frozen cut-rim and seeded ill-conditioned SPD blocks."""
from pathlib import Path
import ctypes as ct
import json
import sys
import numpy as np

WORK=Path(__file__).resolve().parent
fixtures=json.loads((WORK/'diagonal-frozen-inputs.json').read_text(encoding='utf8'))
inputs=list(fixtures['rows'])
rng=np.random.default_rng(7310031)
for case in range(1024):
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    condition=10**rng.uniform(0,6)
    h=(q@np.diag([1,condition**-.5,1/condition])@q.T).astype(np.float32)
    inputs.append([h[0,0],h[1,1],h[2,2],h[0,1],h[0,2],h[1,2],*rng.uniform(-1,1,3)])
inputs.extend([[0]*6+[1,2,3],[-1,-1,-1,0,0,0,1,2,3]])
inputs=np.asarray(inputs,dtype=np.float32);out=np.empty((len(inputs),6),dtype=np.float32)
dll=ct.CDLL(str(Path(sys.argv[1]).resolve()));ptr=ct.POINTER(ct.c_float)
dll.LCS_ProbeAdditiveDiagonal.argtypes=(ct.c_int,ptr,ptr);dll.LCS_ProbeAdditiveDiagonal.restype=ct.c_int
assert dll.LCS_ProbeAdditiveDiagonal(len(inputs),inputs.ctypes.data_as(ptr),out.ctypes.data_as(ptr))==0
reference=[]
for row in inputs.astype(np.float64):
    xx,yy,zz,xy,xz,yz,*rhs=row;h=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
    reference.append(np.linalg.solve(h,rhs) if np.linalg.det(h)>1e-25 else np.zeros(3))
reference=np.asarray(reference)
denom=np.maximum(np.linalg.norm(reference,axis=1),1e-30)
old_error=np.linalg.norm(out[:,:3]-reference,axis=1)/denom
new_error=np.linalg.norm(out[:,3:]-reference,axis=1)/denom
assert np.isfinite(out).all() and np.max(old_error)>1e-4
assert np.max(new_error)<2e-6
result=dict(cases=len(inputs),frozen_blocks=len(fixtures['rows']),old_max_relative_error=float(old_error.max()),
            new_max_relative_error=float(new_error.max()),old_cases_above_1e_minus4=int((old_error>1e-4).sum()),
            note='GPU helper regression plus independent float64 solve; not a claim that every compiler context gives the same old rounding error.')
(WORK/'diagonal-regression.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
