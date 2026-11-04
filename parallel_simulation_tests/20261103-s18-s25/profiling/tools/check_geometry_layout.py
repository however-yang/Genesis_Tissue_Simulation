import importlib.util,json,os,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'candidate-source'))
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import numpy as np
from dataclasses import replace
from threadpoolctl import threadpool_limits
from endoscopic_tissue.training.snapshot import decode_python
from endoscopic_tissue.instrument_geometry import load_scissors,scissors_body_depths
from endoscopic_tissue.training.render_layout import SurfaceLayoutCache
name='endoscopic_tissue._reference_geometry';spec=importlib.util.spec_from_file_location(name,r/'baseline-source/endoscopic_tissue/instrument_geometry.py');old=importlib.util.module_from_spec(spec);sys.modules[name]=old;spec.loader.exec_module(old)
state=decode_python((r/'results/baseline-profile/render-state.bin').read_bytes());surface=state['surface'];s=state['observed'].scissors;geometry=load_scissors();rng=np.random.default_rng(778)
result={'passed':False,'cases':0,'maximum_difference_cm':0,'threshold_disagreement':0};timings={}
with threadpool_limits(limits=1):
 cases=[(s.hinge,s.axis,s.normal,s.gap)]
 for i in range(50):
  q,_=np.linalg.qr(rng.normal(size=(3,3)));hinge=s.hinge+rng.normal(0,.2 if i<25 else 2,3)
  cases.append((hinge,q[:,0],q[:,1],rng.uniform(0,geometry.max_gap)))
 for args in cases:
  a=np.array(old.scissors_body_depths(surface.vertices,surface.triangles,geometry,*args));b=np.array(scissors_body_depths(surface.vertices,surface.triangles,geometry,*args))
  result['maximum_difference_cm']=max(result['maximum_difference_cm'],float(np.max(abs(a-b))))
  result['threshold_disagreement']+=int((max(a)>.02)!=(max(b)>.02));result['cases']+=1
  assert np.array_equal(a,b),(a,b)
 for name,fn in [('numpy',old.scissors_body_depths),('fused',scissors_body_depths)]:
  vals=[]
  for _ in range(30):
   start=time.perf_counter();fn(surface.vertices,surface.triangles,geometry,*cases[0]);vals.append((time.perf_counter()-start)*1000)
  timings[name]={'mean_ms':float(np.mean(vals)),'p95_ms':float(np.percentile(vals,95))}
 # Full cache invalidation: positions/normals may change, anchors and indices may not.
 axes=np.array([[1,2],[0,2],[0,1]]);cache=SurfaceLayoutCache();first=cache.get(surface,1.5,axes)
 assert cache.get(replace(surface,vertices=surface.vertices+.001),1.5,axes) is first
 anchors=surface.anchors.copy();anchors[0,0]+=.00001
 changed=replace(surface,anchors=anchors);assert cache.get(changed,1.5,axes) is not first
 triangles=changed.triangles.copy();triangles[[0,1]]=triangles[[1,0]];changed=replace(changed,triangles=triangles)
 before=cache.parts;assert cache.get(changed,1.5,axes) is not before
 before=cache.parts;assert cache.get(changed,1.6,axes) is not before
 for name,fn in [('recompute',lambda:SurfaceLayoutCache().get(surface,1.5,axes)),('cached',lambda:cache.get(surface,1.5,axes))]:
  fn();vals=[]
  for _ in range(30):
   start=time.perf_counter();fn();vals.append((time.perf_counter()-start)*1000)
  timings[name]={'mean_ms':float(np.mean(vals)),'p95_ms':float(np.percentile(vals,95))}
result.update(passed=True,timings=timings,cache_invalidation_passed=True)
(r/'results/geometry-layout-equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
