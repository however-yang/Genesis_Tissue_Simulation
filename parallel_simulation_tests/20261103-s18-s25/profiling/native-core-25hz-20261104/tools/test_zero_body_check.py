import os,sys,json,time,hashlib
from pathlib import Path
os.sched_setaffinity(0,[30,31]);os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
p=Path(__file__).resolve().parents[1];sys.path.insert(0,str(p/'python-zero-check'))
import numpy as np
from threadpoolctl import threadpool_limits
from endoscopic_tissue import instrument_geometry as g
from endoscopic_tissue.training import zero_body_check as z
from endoscopic_tissue.training.snapshot import decode_python
from endoscopic_tissue.scenario import scissors_frame
geo=g.load_scissors();rng=np.random.default_rng(3617);results=[];cases=[]
for i in range(240):
 hinge=rng.uniform(-2,2,3).astype(np.float32);axis,normal=[np.asarray(v,dtype=np.float32) for v in scissors_frame(*rng.uniform(-45,45,3))];gap=float(rng.uniform(0,geo.max_gap));vertices=(hinge+rng.normal(size=(120,3))*rng.choice([.01,.1,1.,3.,20.])).astype(np.float32);tri=np.arange(120,dtype=np.int32).reshape(-1,3)
 # Near every box face, plus degenerate/skinny and far triangles.
 if i%3==0:vertices[:9]=hinge+np.array([[-4.,0,0],[-4,1e-8,0],[-4,0,1e-8]]*3,dtype=np.float32)
 cases.append((vertices,tri,geo,hinge,axis,normal,gap))
state_path=p/'results/query-v4-mode10-full-validation/failure-render-state.bin'
if state_path.exists():
 data=decode_python(state_path.read_bytes());s=data['surface'];o=data['observed'];s=s if hasattr(s,'vertices') else type('Surface',(),s);o=o if hasattr(o,'scissors') else type('Observed',(),o);pose=o.scissors;pose=pose if hasattr(pose,'hinge') else type('Pose',(),pose)
 cases.append((s.vertices,s.triangles,geo,pose.hinge,pose.axis,pose.normal,float(pose.gap)))
with threadpool_limits(limits=1):
 for i,args in enumerate(cases):
  z.enabled=False;reference=g.scissors_body_depths(*args);z.enabled=True;z.validate=True;actual=g.scissors_body_depths(*args)
  assert actual==reference,(i,actual,reference)
  results.append(dict(case=i,bit_exact=True,reference=list(map(float,reference))))
 args=cases[-1];g.scissors_body_depths(*args);repeats=180;timing={}
 for mode in [False,True,True,False]:
  z.enabled=mode;z.validate=False;start=time.perf_counter_ns()
  for _ in range(repeats):g.scissors_body_depths(*args)
  timing.setdefault('candidate' if mode else 'reference',[]).append((time.perf_counter_ns()-start)/1e6/repeats)
result=dict(passed=True,cases=len(cases),bit_exact_comparisons=len(results),statistics={k:getattr(z,k) for k in ['calls','clear_returns','fallbacks','validated_clear','mismatches']},timing_ms=timing,scope='diagnostic geometry only, exact positive/ambiguous fallback; not an end-to-end FPS claim',cases_results=results)
(p/'results/zero-body-check-cpu.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases_results'},indent=2))
