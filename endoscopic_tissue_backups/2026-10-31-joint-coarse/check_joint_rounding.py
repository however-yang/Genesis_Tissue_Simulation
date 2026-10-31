"""Find a deterministic CUDA endpoint-rounding regression at the real path-check seam."""
import ctypes as ct
import json
from pathlib import Path
import sys
import numpy as np
rng=np.random.default_rng(310026)
rows=[]
for _ in range(12000):
    Q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(Q)<0:Q[:,0]*=-1
    F=Q@np.diag(rng.uniform(.7,1.3,3));new=F.copy();new[:,0]*=.15/np.linalg.det(F)
    origin=rng.uniform(-4,4,3)
    p=np.vstack([origin,origin+F.T]).astype(np.float32)
    trial=np.vstack([origin,origin+new.T]).astype(np.float32)
    d=(trial-p).astype(np.float32)
    rows.append(np.r_[p.ravel(),d.ravel(),1.,1.,.15])
inputs=np.array(rows,np.float32);outputs=np.empty((len(inputs),5),np.float32)
dll=ct.CDLL(str(Path(sys.argv[1]).resolve()));ptr=ct.POINTER(ct.c_float)
dll.LCS_ProbeJointRounding.argtypes=(ct.c_int,ptr,ptr);dll.LCS_ProbeJointRounding.restype=ct.c_int
assert dll.LCS_ProbeJointRounding(len(inputs),inputs.ctypes.data_as(ptr),outputs.ctypes.data_as(ptr))==0
counter=np.flatnonzero((outputs[:,0]==1)&(outputs[:,3]<outputs[:,4]))
assert len(counter)>0,'Regression did not exercise rounding mismatch'
assert not np.any((outputs[:,1]==1)&(outputs[:,3]<outputs[:,4]))
i=int(counter[0]);result=dict(cases=len(inputs),old_path_accepts_below_stored_floor=len(counter),stored_guard_accepts_below_floor=0,first_counterexample=dict(input=inputs[i].tolist(),path_endpoint=float(outputs[i,2]),stored_endpoint=float(outputs[i,3]),floor=float(outputs[i,4])),note='Frozen synthetic input at the same CUDA rounded-position/cubic-check seam; does not establish which compiler transformation caused each earlier live event.')
Path(sys.argv[2]).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='first_counterexample'},indent=2))
