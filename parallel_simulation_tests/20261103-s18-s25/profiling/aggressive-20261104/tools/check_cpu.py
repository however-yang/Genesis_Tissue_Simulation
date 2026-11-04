import os,sys,json,time,importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1];os.sched_setaffinity(0,[25]);os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(p/'tmp'))
sys.path.insert(0,str(p/'candidate-source'))
import numpy as np
from endoscopic_tissue.training import instrument_collision as fast,grasp as grasp_fast
from endoscopic_tissue.instrument_geometry import load_grasper,load_scissors
from endoscopic_tissue.training.snapshot import decode_python
from endoscopic_tissue.scenario import scissors_settings

def reference(name):
 spec=importlib.util.spec_from_file_location('endoscopic_tissue.training._ref_'+name,p.parent/'candidate-v4-source/endoscopic_tissue/training'/str(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
ref=reference('instrument_collision');grasp_ref=reference('grasp');rng=np.random.default_rng(18);result=dict(passed=False)
box_errors=[];capsule_errors=[];classification=[]
for i in range(500):
 fa=np.linalg.qr(rng.normal(size=(3,3)))[0];fb=np.linalg.qr(rng.normal(size=(3,3)))[0]
 ca=rng.uniform(-.03,.03,3);cb=ca+rng.uniform(-.02,.02,3);ha=rng.uniform(.0001,.01,3);hb=rng.uniform(.0001,.01,3)
 if i<50:fb=fa.copy()
 if i<30:cb=ca.copy()
 if 50<=i<100:cb=ca+(ha[0]+hb[0]+rng.uniform(-1e-10,1e-10))*fa[0];fb=fa.copy()
 a=fast.Box(ca,fa,ha);b=fast.Box(cb,fb,hb)
 expected=ref.box_distance(a,b);actual=fast.box_distance(a,b);box_errors.append(abs(expected-actual));classification.append((expected<-.0002)==(actual<-.0002))
 start=rng.uniform(-.04,.04,3);end=start.copy() if i<30 else rng.uniform(-.04,.04,3);radius=rng.uniform(0,.004)
 expected=ref.capsule_box_distance(start,end,radius,a);actual=fast.capsule_box_distance(start,end,radius,a);capsule_errors.append(abs(expected-actual));classification.append((expected<-.0002)==(actual<-.0002))
 assert max(box_errors+capsule_errors)<1e-12 and all(classification)
state=decode_python((p.parent/'results/baseline-profile/render-state.bin').read_bytes());obs=state['observed'];settings=scissors_settings();g=load_grasper(settings.jaw_length,settings.jaw_width,settings.instrument_model);s=load_scissors()
kwargs=dict(grasper_hinge=obs.hinge,grasper_axis=settings.jaw_axis,grasper_normal=settings.jaw_normal,grasper_gap=obs.jaw_gap,scissors_hinge=obs.scissors.hinge,scissors_axis=obs.scissors.axis,scissors_normal=obs.scissors.normal,scissors_gap=obs.scissors.gap)
real_error=abs(ref.instrument_distance(g,s,**kwargs)-fast.instrument_distance(g,s,**kwargs));assert real_error<1e-12
bounds=np.asarray(g.bounds)*.01;tools=dict(grasper_axis=list(settings.jaw_axis),grasper_normal=list(settings.jaw_normal),grasper_gap_m=obs.jaw_gap*.01,grasper_hinge_m=(obs.hinge*.01).tolist())
for points in [obs.nodes*.01,rng.uniform(-.03,.03,(40000,3)),np.array([[np.nan,0,0],[np.inf,0,0],[-np.inf,0,0]])]:
 assert np.array_equal(grasp_ref.between_pads(points,tools,bounds,settings.jaw_length*.01),grasp_fast.between_pads(points,tools,bounds,settings.jaw_length*.01))
timings={}
for name,fn in [('distance_ref',lambda:ref.instrument_distance(g,s,**kwargs)),('distance_fast',lambda:fast.instrument_distance(g,s,**kwargs)),('grasp_ref',lambda:grasp_ref.between_pads(obs.nodes*.01,tools,bounds,settings.jaw_length*.01)),('grasp_fast',lambda:grasp_fast.between_pads(obs.nodes*.01,tools,bounds,settings.jaw_length*.01))]:
 values=[]
 for _ in range(60):
  t=time.perf_counter_ns();fn();values.append((time.perf_counter_ns()-t)/1e6)
 timings[name]=dict(mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)))
result.update(passed=True,boxes=500,capsules=500,box_max_error_m=max(box_errors),capsule_max_error_m=max(capsule_errors),threshold_classification_identical=all(classification),real_instrument_error_m=real_error,grasp_masks_bit_exact=True,timings=timings)
(p/'results/cpu-equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
