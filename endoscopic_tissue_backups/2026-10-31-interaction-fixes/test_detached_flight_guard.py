from types import SimpleNamespace
import itertools,json
import numpy as np
from detached_flight_guard import audit_detached_flight
xyz=np.array(list(itertools.product(range(2),repeat=3)),dtype=np.float32)*.1
rest=np.vstack((xyz,xyz+(2,0,0))).astype(np.float32);neighbors=np.full((16,6),-1,np.int32)
for block in (0,8):
 for i,p in enumerate(xyz):
  for axis in range(3):
   for side in (-1,1):
    matches=np.flatnonzero(np.linalg.norm(xyz-(p+np.eye(3)[axis]*.1*side),axis=1)<1e-6)
    if len(matches):neighbors[block+i,2*axis+(side>0)]=block+matches[0]
grid=SimpleNamespace(active=np.ones(16,np.int32),inside=np.ones(16,np.int32),positions=rest,pinned=np.r_[1,np.zeros(15,np.int32)])
passed=[]
def check(name,nodes,grips=None,expected=True):
 obs=SimpleNamespace(nodes=nodes,neighbors=neighbors,grip_flags=np.zeros(16,np.int32) if grips is None else grips)
 allowed,detail=audit_detached_flight(grid,obs,np.linalg.norm(nodes-rest,axis=1)*10)
 assert allowed==expected,(name,allowed,detail)
 passed.append(dict(name=name,allowed=allowed,detail=detail))
x=rest.copy();x[8:]+=np.array((0,-10,0));check('Detached intact flight',x)
x=rest.copy();R=np.array(((0,-1,0),(1,0,0),(0,0,1)));x[8:]=(rest[8:]-rest[8:].mean(0))@R+np.array((2,-10,0));check('Detached rotated flight',x)
x=rest.copy();x[:8]+=np.array((0,-10,0));check('Attached body motion',x,expected=False)
x=rest.copy();x[8:]+=np.array((0,-10,0));g=np.zeros(16,np.int32);g[9]=1;check('Held piece motion',x,g,False)
x=rest.copy();x[8:]+=np.array((0,-10,0));x[8,0]+=30;check('Detached severe deformation',x,expected=False)
grid.inside[8:]=0;x=rest.copy();x[8:]+=np.array((0,-10,0));check('Support-only remnant excursion',x,expected=False)
grid.inside[8:]=1;old_rest=grid.positions.copy();grid.positions[8:,2]=0
x=grid.positions.copy();x[8:]+=np.array((0,-10,0));check('Planar material-labelled remnant',x,expected=False)
grid.positions[:]=old_rest;old_neighbors=neighbors.copy();neighbors[8:,1]= -1;neighbors[8:,0]= -1
x=rest.copy();x[8:]+=np.array((0,-10,0));check('Disconnected material sheets',x,expected=False)
neighbors[:]=old_neighbors
print(json.dumps(passed,indent=2))
