from pathlib import Path
import numpy as np
def load(path):
 with Path(path).open('rb') as f:
  h=np.fromfile(f,np.int32,10);assert h[0]==0x43555431; n,e=int(h[1]),int(h[2]);d=dict(header=h)
  for key,dt,shape in [('pos','f4',(n,3)),('rest','f4',(n,3)),('neighbors','i4',(n,6)),('inside','i4',(n,)),('level','u4',(n,)),('edges','i4',(e,4)),('pending','f4',(e,4)),('sweep','f4',(18,3))]:d[key]=np.fromfile(f,dt,int(np.prod(shape))).reshape(shape)
  assert not f.read(1)
 return d
def swept(x,y,a,b,side,now,op,reach):
 dx,dy=b[:2]-a[:2];r2=x*x+y*y;A=dx*dx+dy*dy;B=2*(a[0]*dx+a[1]*dy);C=a[0]*a[0]+a[1]*a[1]-r2;disc=B*B-4*A*C
 if A<1e-12 or disc<0:return False
 q=(-B+np.sqrt(disc))/(2*A)
 if q<0 or q>1:
  q=(-B-np.sqrt(disc))/(2*A)
  if q<0 or q>1:return False
 angle=side*(np.arctan2(y,x)-np.arctan2(a[1]+q*dy,a[0]+q*dx));return bool(angle>=now-reach/np.sqrt(r2) and angle<=op)
