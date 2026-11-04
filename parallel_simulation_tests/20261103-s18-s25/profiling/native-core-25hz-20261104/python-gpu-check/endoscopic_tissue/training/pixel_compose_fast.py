"""Fused CPU output materialization; no float contraction or approximation.

The public contract is CPU arrays (~252 MB at 1280x960). Fusing the gathers,
validation and stores avoids several equally large temporary arrays and another
GPU transfer. A future GPU-consumer contract should keep this data on device.
"""
import numpy as np
from numba import njit

@njit(cache=False,nogil=True)
def compose(flat,bary,triangles,node_ids,node_weights,offsets,vertex_valid):
    n=len(flat)
    ids=np.empty((n,24),np.int32);weights=np.empty((n,24),np.float32)
    offset=np.empty((n,3),np.float32);valid=np.empty(n,np.bool_)
    for row in range(n):
        tri=flat[row];good=tri>=0 and tri<len(triangles)
        if good:
            b0,b1,b2=bary[row]
            good=(np.isfinite(b0) and np.isfinite(b1) and np.isfinite(b2)
                  and b0>=-1e-7 and b1>=-1e-7 and b2>=-1e-7
                  and abs((b0+b1)+b2-1)<1e-6)
        if good:
            a,b,c=triangles[tri]
            good=vertex_valid[a] and vertex_valid[b] and vertex_valid[c]
        valid[row]=good
        if good:
            for v in range(3):
                vertex=triangles[tri,v]
                for k in range(8):
                    ids[row,v*8+k]=node_ids[vertex,k]
                    weights[row,v*8+k]=node_weights[vertex,k]*bary[row,v]
            for k in range(3):
                offset[row,k]=(offsets[a,k]*b0+offsets[b,k]*b1)+offsets[c,k]*b2
        else:
            for k in range(24):ids[row,k]=-1;weights[row,k]=0
            for k in range(3):offset[row,k]=0
    return ids,weights,offset,valid
