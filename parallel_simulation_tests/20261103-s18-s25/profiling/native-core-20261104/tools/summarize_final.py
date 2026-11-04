"""Final measured statistics. Child timers are not added twice."""
import os,json,csv,statistics
from pathlib import Path
p=Path(__file__).resolve().parents[1];os.sched_setaffinity(0,[30,31]);os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(p/'tmp/matplotlib'))
import numpy as np
read=lambda name:json.loads((p/'results'/name/'result.json').read_text())
def aggregate(names):
 ds=[read(n) for n in names];frames=[f for d in ds for f in d['frames']];t=np.array([f['frame_ms'] for f in frames]);components={key:float(np.mean([f['timings_ms'][key] for f in frames])) for key in ['native_step','quality_and_export','snapshot']};components['vision']=float(np.mean([f['render_ms'] for f in frames]));components['other']=float(t.mean()-sum(components.values()));samples=[]
 for name,d in zip(names,ds):
  file=p/'results'/name/'gpu-monitor.jsonl'
  samples+=[r for r in [json.loads(s) for s in file.read_text().splitlines()] if d['measurement_start_unix_s']<=r['time']<=d['measurement_end_unix_s']]
 return dict(runs=names,frames=len(t),completed=all(d['completed'] for d in ds),local_checks_passed=all(d['quality_passed'] for d in ds),mean_ms=float(t.mean()),fps=float(1000/t.mean()),p50_ms=float(np.percentile(t,50)),p95_ms=float(np.percentile(t,95)),p99_ms=float(np.percentile(t,99)),max_ms=float(t.max()),over_50ms=int((t>50).sum()),rolling20_min_hz=min(d['performance']['rolling20_min_hz'] for d in ds),components_ms=components,component_percent={k:100*v/float(t.mean()) for k,v in components.items()},gpu_percent=statistics.mean(s['gpu_percent'] for s in samples),sampled_peak_memory_mib=max(s['used_mib'] for s in samples),host_cpu_percent=statistics.mean(s['host_cpu_percent'] for s in samples))
choice=json.loads((p/'results/recommended-candidate.json').read_text());before=aggregate(['all-queries-v2-original-a','all-queries-v2-original-b']);after=aggregate(['all-queries-v2-candidate-a','all-queries-v2-candidate-b']);pure_before=read('delivery-baseline-physics');pure_after=read('all-queries-v2-checked-physics');short=read('all-queries-v2-current-frame');snapshot=read('all-queries-v2-snapshot-roundtrip')
checks=before['completed'] and before['local_checks_passed'] and after['completed'] and after['local_checks_passed'] and short['passed'] and snapshot['passed'] and pure_after['completed'] and pure_after['quality_passed']
if choice['execution']['bound_cache']:
 checks=checks and read('all-queries-v2-full-validation')['coarse_bound_validation']['passed'] and read('all-queries-v2-full-validation')['native_validation']['passed']
else:
 checks=checks and read('conservative-captured-validation')['native_validation']['passed'] and read('conservative-captured-validation-repeat')['native_validation']['passed']
summary=dict(before=before,after=after,physics_before=pure_before['performance'],physics_after=pure_after['performance'],speedup=before['mean_ms']/after['mean_ms'],local_validation_passed=bool(checks),average20_passed=after['fps']>=20,p95_50ms_passed=after['p95_ms']<=50,rolling20_passed=after['rolling20_min_hz']>=20,all_frames_50ms_passed=after['over_50ms']==0,stable20_passed=after['p95_ms']<=50 and after['rolling20_min_hz']>=20,formal_physics_domain_approved=False,training_domain_approved=False,production_replaced=False,choice=choice)
(p/'results/final-summary.json').write_text(json.dumps(summary,indent=2));(p/'results/final-decision.json').write_text(json.dumps({k:v for k,v in summary.items() if k not in ['before','after','physics_before','physics_after']},indent=2))
rows=[]
for label,names in [('original',before['runs']),('optimized',after['runs'])]:
 fs=[f for name in names for f in read(name)['frames']]
 for stage in dict.fromkeys(f['stage'] for f in fs):
  ff=[f for f in fs if f['stage']==stage];tt=np.array([f['frame_ms'] for f in ff]);rows.append(dict(variant=label,stage=stage,frames=len(ff),mean_ms=float(tt.mean()),fps=float(1000/tt.mean()),p95_ms=float(np.percentile(tt,95)),native_ms=float(np.mean([f['timings_ms']['native_step'] for f in ff])),vision_ms=float(np.mean([f['render_ms'] for f in ff])),frames_with_new_cuts=sum(f['quality']['metrics']['new_cut_edges']>0 for f in ff)))
with (p/'results/final-stage-times.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
(p/'figures').mkdir(exist_ok=True);fig,ax=plt.subplots(figsize=(11,6.5));bottom=np.zeros(2)
for key,label,color in [('native_step','Native physics/contact/cutting + synchronized observation','#5366ad'),('quality_and_export','Quality checks + surface export','#e7a441'),('snapshot','State construction','#62a57c'),('vision','Full RGB-D + alignment + CPU return','#b56f9a'),('other','Control + policy observation + other','#9da4ac')]:
 values=[before['components_ms'][key],after['components_ms'][key]];ax.bar([0,1],values,bottom=bottom,label=label,color=color);bottom+=values
ax.scatter([0,1],[before['p95_ms'],after['p95_ms']],marker='D',color='black',label='P95 (pooled frames)',zorder=5);ax.axhline(50,color='#b83c3c',ls='--',label='20 Hz frame budget');ax.set_xticks([0,1],['Original kernels','Selected candidate']);ax.set_ylabel('Full RGB-D frame wall time (ms)');ax.set_ylim(0,max(before['p95_ms'],after['p95_ms'])+5);ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2);fig.tight_layout();fig.savefig(p/'figures/frame-budget.png',dpi=180);fig.savefig(p/'figures/frame-budget.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(13,4))
for name,label,color in [('all-queries-v2-original-a','Original','#8e99b7'),('all-queries-v2-candidate-a','Candidate','#b95941')]:
 d=read(name);ax.plot([f['frame_id'] for f in d['frames']],[f['frame_ms'] for f in d['frames']],lw=.7,alpha=.85,label=label,color=color)
ax.axhline(50,color='black',ls='--',lw=1,label='50 ms');ax.set_xlabel('Frame ID (same 1410-step command script)');ax.set_ylabel('Complete RGB-D frame (ms)');ax.legend();fig.tight_layout();fig.savefig(p/'figures/frame-times.png',dpi=180);fig.savefig(p/'figures/frame-times.svg');plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k not in ['choice','before','after']},indent=2))
