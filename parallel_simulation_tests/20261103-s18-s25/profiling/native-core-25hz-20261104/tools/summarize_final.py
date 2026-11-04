import os,json,csv,hashlib,collections
from pathlib import Path
os.sched_setaffinity(0,[30,31]);p=Path(__file__).resolve().parents[1]
import numpy as np
names=['readonly-abba-0-reference-a','readonly-abba-1-readonly-a','readonly-abba-2-readonly-b','readonly-abba-3-reference-b'];raw={n:json.loads((p/'results'/n/'result.json').read_text()) for n in names}
assert all(r['completed'] and r['quality_passed'] and len(r['frames'])==1410 for r in raw.values()),'Performance group contains a failed/incomplete run'
def describe(runs):
 fs=[f for r in runs for f in r['frames']];times=np.array([f['frame_ms'] for f in fs]);phases={k:float(np.mean([f['timings_ms'].get(k,0) for f in fs])) for k in ['native_step','quality_and_export','snapshot']};phases['render']=float(np.mean([f['render_ms'] for f in fs]));phases['other']=float(times.mean()-sum(phases.values()));samples=[]
 for r in runs:
  directory=p/'results'/next(k for k,v in raw.items() if v is r)
  samples += [s for s in map(json.loads,(directory/'gpu-monitor.jsonl').read_text().splitlines()) if r['measurement_start_unix_s']<=s['time']<=r['measurement_end_unix_s']]
 return dict(runs=len(runs),frames=len(fs),fps=float(1000/times.mean()),mean_ms=float(times.mean()),p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),p99_ms=float(np.percentile(times,99)),max_ms=float(times.max()),over_40ms=int((times>40).sum()),over_50ms=int((times>50).sum()),rolling20_min_hz=min(r['performance']['rolling20_min_hz'] for r in runs),thread_cpu_ms=float(np.mean([f['thread_cpu_ms'] for f in fs])),phases_ms=phases,phases_percent={k:100*v/float(times.mean()) for k,v in phases.items()},gpu_samples=len(samples),gpu_percent_mean=float(np.mean([s['gpu_percent'] for s in samples])),gpu_percent_peak=max(s['gpu_percent'] for s in samples),memory_mib_peak=max(s['used_mib'] for s in samples),temp_c_peak=max(s['temp_c'] for s in samples))
a=[raw[names[0]],raw[names[3]]];b=[raw[names[1]],raw[names[2]]];ref=describe(a);opt=describe(b)
stages=collections.defaultdict(lambda:collections.defaultdict(list))
for label,runs in [('reference',a),('readonly',b)]:
 for r in runs:
  for f in r['frames']:stages[f['stage']][label].append(f)
rows=[]
for stage,groups in stages.items():
 for label,fs in groups.items():
  times=np.array([f['frame_ms'] for f in fs]);row=dict(stage=stage,variant=label,frames=len(fs),fps=float(1000/times.mean()),mean_ms=float(times.mean()),p95_ms=float(np.percentile(times,95)),native_step_ms=float(np.mean([f['timings_ms']['native_step'] for f in fs])),quality_and_export_ms=float(np.mean([f['timings_ms']['quality_and_export'] for f in fs])),snapshot_ms=float(np.mean([f['timings_ms']['snapshot'] for f in fs])),render_ms=float(np.mean([f['render_ms'] for f in fs])));rows.append(row)
with (p/'results/stage-comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
validation=json.loads((p/'results/readonly-v8-full-rgbd-validation/result.json').read_text());sass=json.loads((p/'results/readonly-v8-sass.json').read_text())
result=dict(reference=ref,readonly=opt,observed_speedup=opt['fps']/ref['fps'],all_runs_complete_and_checked=True,full_validation_completed=validation['completed'] and validation['quality_passed'],validation=dict(native=validation['native_validation'],coarse=validation['coarse_bound_validation'],body=validation['gpu_body_statistics']),all_original_functions_unchanged=sass['all_original_functions_unchanged'],formal_physics_domain_approved=False,training_domain_approved=False,all_frames_within_25hz_budget=opt['max_ms']<=40,stability_observation=dict(p95_within_40ms=opt['p95_ms']<=40,worst20_at_least25hz=opt['rolling20_min_hz']>=25,worst20_at_least20hz=opt['rolling20_min_hz']>=20),scope='one worker, current 1280x960 RGBD CPU observation, all physics/checks preserved; no optional dense S25 labels, display/transport or disk recording',runs={n:{k:r.get(k) for k in ['completed','quality_passed','performance','gpu_body_statistics','measurement_start_unix_s','measurement_end_unix_s']} for n,r in raw.items()})
(p/'results/final-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
