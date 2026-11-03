"""CPU algebra witness for the endpoint-normal filter; never edits native state."""
from pathlib import Path
import hashlib,json
import numpy as np

def bary(q):
 a,b,c=q;u,v=b-a,c-a;w=-a
 aa,ab,bb=np.dot(u,u),np.dot(u,v),np.dot(v,v);aw,bw=np.dot(u,w),np.dot(v,w)
 den=aa*bb-ab*ab
 if den<1e-16:return None
 y=(bb*aw-ab*bw)/den;z=(aa*bw-ab*aw)/den
 return np.array([1-y-z,y,z])

def main():
 h=Path(__file__).resolve().parent;out=h/'ccd-endpoint-witness';out.mkdir(exist_ok=False)
 rng=np.random.default_rng(20401103);skin=.003;found=None
 # Batched CPU search, bounded iterations and memory. All vertex paths linear.
 for batch in range(400):
  old=rng.uniform(-.10,.10,(4096,3,3));new=old+rng.uniform(-.08,.08,old.shape)
  n0=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0]);n1=np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
  l0=np.linalg.norm(n0,axis=1);l1=np.linalg.norm(n1,axis=1)
  ok=(l0>1e-5)&(l1>1e-5);d0=np.einsum('ij,ij->i',-old[:,0],n0)/np.maximum(l0,1e-30);d1=np.einsum('ij,ij->i',-new[:,0],n1)/np.maximum(l1,1e-30)
  ok&=(d0>skin)&(d1>skin)&(np.einsum('ij,ij->i',n0,n1)/np.maximum(l0*l1,1e-30)>.5)
  mid=(old+new)*.5;nm=np.cross(mid[:,1]-mid[:,0],mid[:,2]-mid[:,0]);lm=np.linalg.norm(nm,axis=1);dm=np.einsum('ij,ij->i',-mid[:,0],nm)/np.maximum(lm,1e-30)
  candidates=np.flatnonzero(ok&(dm<-skin))
  for index in candidates:
   a=old[index];b=new[index];v=b-a
   # Exact degree-three oriented point-plane determinant via four sample values.
   times=np.array([0.,1/3,2/3,1.]);values=[]
   for t in times:
    q=a+t*v;values.append(np.dot(-q[0],np.cross(q[1]-q[0],q[2]-q[0])))
   coeff=np.polynomial.polynomial.polyfit(times,values,3);roots=np.polynomial.polynomial.polyroots(coeff)
   events=[]
   for root in roots:
    if abs(root.imag)>1e-8 or not 0<root.real<1:continue
    t=float(root.real);q=a+t*v;weights=bary(q)
    if weights is not None and min(weights)>1e-5:
     events.append(dict(t=t,barycentric=weights.tolist(),distance_cm=float(np.linalg.norm(weights@q))))
   if len(events)==2:
    found=dict(previous_triangle_cm=a.tolist(),current_triangle_cm=b.tolist(),point_previous_cm=[0.,0.,0.],point_current_cm=[0.,0.,0.],before_cm=float(d0[index]),now_cm=float(d1[index]),midpoint_cm=float(dm[index]),skin_cm=skin,events=events,max_vertex_motion_mm=float(np.linalg.norm(v,axis=1).max()*10),endpoint_filter_would_skip=bool(d1[index]>=skin),batch=batch,determinant_coefficients=coeff.tolist());break
  if found:break
 if found is None:
  (out/'result.json').write_text(json.dumps(dict(found=False,batches=400,scope='No witness found in bounded CPU search; no claim of correctness.')));print('No bounded witness');return
 source=h.parents[2]/'runs/a100-functional-fixes-20261102-r1/release-training-r3/source/vendor/cuda/contact/fragment_contact.cuh'
 result=dict(found=True,diagnostic_only=True,native_execution_tested=False,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),original_acceptance_unchanged=True,witness=found,scope='Algebraic counterexample to endpoint-only point/triangle candidate rejection for linearly moving vertices. This is not proof that this motion occurs in the accepted kidney trajectory; the full solver also has endpoint containment repair.')
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
