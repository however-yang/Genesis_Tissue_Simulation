import importlib.util,json,os,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'candidate-source'))
import numpy as np
from endoscopic_tissue.training.visual_labels import compose_pixel_supports
spec=importlib.util.spec_from_file_location('pixel_reference',r/'baseline-source/endoscopic_tissue/training/visual_labels.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rng=np.random.default_rng(787);bindings=dict(triangles=rng.integers(0,1000,(400,3),dtype=np.int32),node_ids=rng.integers(0,76050,(1000,8),dtype=np.int32),weights=rng.uniform(0,1,(1000,8)).astype(np.float32),offset_m=rng.normal(0,.001,(1000,3)).astype(np.float32),valid=rng.random(1000)>.02)
cases=0;timings={}
for shape in [(3,4),(73,97),(960,1280)]:
 index=rng.integers(-30,430,shape,dtype=np.int32);bary=rng.dirichlet(np.ones(3),size=shape).astype(np.float32)
 bary.reshape(-1,3)[:5]=[[np.nan,0,1],[np.inf,0,1],[-.1,.5,.6],[.3,.3,.3],[0,0,1]]
 a=old.compose_pixel_supports(index,bary,bindings);b=compose_pixel_supports(index,bary,bindings)
 for key in a:assert np.array_equal(a[key],b[key],equal_nan=True),key
 cases+=1
 if shape==(960,1280):
  del a,b
  for name,fn in [('numpy',old.compose_pixel_supports),('fused',compose_pixel_supports)]:
   values=[]
   for _ in range(8):
    start=time.perf_counter_ns();out=fn(index,bary,bindings);values.append((time.perf_counter_ns()-start)/1e6);del out
   timings[name]=dict(mean_ms=float(np.mean(values)),p50_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
result=dict(passed=True,cases=cases,bit_exact=True,timings=timings,note='Full-resolution synthetic visibility/material composition; not full-frame performance')
(r/'results/pixel-compose-equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
