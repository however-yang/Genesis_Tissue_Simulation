"""Independent float64 solid-angle check of the experimental Taylor bound."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    rng=np.random.default_rng(1704);maximum_ratio=0.;used=0;rounding_disagreements=0
    for _ in range(10000):
        triangles=rng.normal(size=(24,3,3))*10**rng.uniform(-2,.5)+rng.uniform(-10,10,size=3)
        low=triangles.min(axis=(0,1));high=triangles.max(axis=(0,1));center=(low+high)/2
        radius=np.linalg.norm(high-low)*.50001+1e-6
        ray=rng.normal(size=3);ray/=np.linalg.norm(ray)
        point=center+ray*radius*10**rng.uniform(.001,2)
        area=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])/2
        total_area=np.linalg.norm(area,axis=1).sum()
        moment=np.einsum('ni,nj->ij',area,triangles.mean(axis=1)-center)
        r=center-point;distance=np.linalg.norm(r);nearest=distance-radius
        approximation=np.dot(area.sum(axis=0),r)/distance**3+np.trace(moment)/distance**3-3*r@moment@r/distance**5
        bound=3.0002*radius**2*total_area/nearest**4+2e-5*(abs(approximation)+total_area/distance**2*(1+(radius+abs(center).max())/distance))+1e-6
        v=triangles-point;length=np.linalg.norm(v,axis=2)
        determinant=np.einsum('ni,ni->n',v[:,0],np.cross(v[:,1],v[:,2]))
        denominator=length.prod(axis=1)+np.einsum('ni,ni->n',v[:,0],v[:,1])*length[:,2]+np.einsum('ni,ni->n',v[:,1],v[:,2])*length[:,0]+np.einsum('ni,ni->n',v[:,2],v[:,0])*length[:,1]
        exact=np.sum(2*np.arctan2(determinant,denominator))
        ratio=abs(exact-approximation)/bound;maximum_ratio=max(maximum_ratio,float(ratio))
        assert ratio<=1,'Taylor remainder bound violated'
        if bound<.02:
            lo=np.floor((approximation-bound)/(4*np.pi)+.5);hi=np.floor((approximation+bound)/(4*np.pi)+.5)
            if lo==hi:
                used+=1;rounding_disagreements+=int(lo!=np.floor(exact/(4*np.pi)+.5))
    assert used>1000 and rounding_disagreements==0
    result=dict(passed=True,random_patches=10000,certified_integer_cases=used,
                maximum_error_over_bound=maximum_ratio,rounding_disagreements=rounding_disagreements,
                scope='Independent float64 algebra check; actual float GPU cases are checked separately')
    (ROOT/'results/moment-bound-cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
