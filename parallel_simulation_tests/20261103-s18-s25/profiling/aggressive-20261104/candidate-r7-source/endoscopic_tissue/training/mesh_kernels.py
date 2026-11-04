"""Fresh per-frame preparation of the existing triangle soup (no layout cache)."""
import numpy as np
from numba import njit

@njit(nogil=True,cache=False)
def prepare_soup(vertices,normals,anchors,texture_cm):
    n=len(vertices);wall_count=0
    for t in range(n//3):
        if anchors[3*t,3]>.5:wall_count+=1
    counts=((n//3-wall_count)*3,wall_count*3)
    pos0=np.empty((counts[0],3),np.float32);pos1=np.empty((counts[1],3),np.float32)
    norm0=np.empty((counts[0],3),np.float32);norm1=np.empty((counts[1],3),np.float32)
    uv0=np.empty((counts[0],2),np.float32);uv1=np.empty((counts[1],2),np.float32)
    ids0=np.empty(counts[0],np.int32);ids1=np.empty(counts[1],np.int32)
    bounds=np.empty((2,2,3),np.float32)
    for part in range(2):
        bounds[part,0,:]=np.inf;bounds[part,1,:]=-np.inf
    offsets=np.zeros(2,np.int32)
    for t in range(n//3):
        a=3*t;b=a+1;c=a+2
        # Match float32 numpy subtract/cross and first-axis argmax ties.
        x0=anchors[b,0]-anchors[a,0];x1=anchors[b,1]-anchors[a,1];x2=anchors[b,2]-anchors[a,2]
        y0=anchors[c,0]-anchors[a,0];y1=anchors[c,1]-anchors[a,1];y2=anchors[c,2]-anchors[a,2]
        cross0=abs(x1*y2-x2*y1);cross1=abs(x2*y0-x0*y2);cross2=abs(x0*y1-x1*y0)
        axis=0;largest=cross0
        if cross1>largest:axis=1;largest=cross1
        if cross2>largest:axis=2
        u=1 if axis==0 else 0;v=1 if axis==2 else 2
        part=1 if anchors[a,3]>.5 else 0;start=offsets[part];offsets[part]+=3
        positions=pos1 if part else pos0;ns=norm1 if part else norm0;uv=uv1 if part else uv0;ids=ids1 if part else ids0
        for k in range(3):
            i=a+k;j=start+k;ids[j]=i
            positions[j,0]=vertices[i,0]*np.float32(.01);positions[j,1]=-vertices[i,2]*np.float32(.01);positions[j,2]=vertices[i,1]*np.float32(.01)
            ns[j,0]=normals[i,0];ns[j,1]=-normals[i,2];ns[j,2]=normals[i,1]
            uv[j,0]=anchors[i,u]/texture_cm;uv[j,1]=anchors[i,v]/texture_cm
            for dim in range(3):
                bounds[part,0,dim]=min(bounds[part,0,dim],positions[j,dim]);bounds[part,1,dim]=max(bounds[part,1,dim],positions[j,dim])
    return (pos0,norm0,uv0,ids0),(pos1,norm1,uv1,ids1),bounds

class ExactBoundsUpload:
    """Only bypass repeated min/max when bounds came from the same fresh vertices."""
    def __init__(self,jit):self.jit=jit;self.original=jit.update_buffer
    def __call__(self,node,buffer_name,data):
        primitive=node.mesh.primitives[0] if node.mesh is not None and len(node.mesh.primitives)==1 else None
        bounds=getattr(primitive,'_fresh_exact_bounds',None) if primitive is not None else None
        if buffer_name!='pos' or bounds is None:return self.original(node,buffer_name,data)
        self.jit._buffer[(primitive,buffer_name)]=data
        index=self.jit._primitive_index.get(primitive)
        if index is not None and len(data):
            self.jit.bounds_geom[index]=bounds
            if getattr(primitive,'_persistent_soup',False):self.jit.n_indices[index]=-len(data)
        self.jit._scene.bump_revision()
