"""Aggregate the four unprofiled ABBA cases and plot retained raw frame times."""
import os
os.sched_setaffinity(0,[30,31])
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.update(MPLCONFIGDIR=str(ROOT/'tmp/matplotlib'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import json,csv,statistics
import numpy as np
from summarize import case
NAMES=['final-a1-v8','final-b1-runtime','final-b2-runtime','final-a2-v8']
data={n:json.loads((ROOT/'results'/n/'result.json').read_text()) for n in NAMES}
assert all(d['completed'] and d['quality_passed'] and len(d['frames'])==1410 for d in data.values())
cases=[case(n) for n in NAMES]

def aggregate(names):
 frames=[f for n in names for f in data[n]['frames']];times=np.array([f['frame_ms'] for f in frames])
 mean=lambda k:statistics.mean(f['timings_ms'].get(k,0) for f in frames)
 stages={}
 for stage in dict.fromkeys(f['stage'] for f in frames):
  subset=[f for f in frames if f['stage']==stage];t=np.array([f['frame_ms'] for f in subset]);stages[stage]=dict(frames=len(subset),mean_ms=float(t.mean()),fps=float(1000/t.mean()),p95_ms=float(np.percentile(t,95)),native_step_ms=statistics.mean(f['timings_ms']['native_step'] for f in subset))
 modules=dict(native_step=mean('native_step'),quality_including_surface=mean('quality_and_export'),snapshot=mean('snapshot'),render_rgbd=statistics.mean(f['render_ms'] for f in frames))
 modules['control_policy_other']=float(times.mean())-sum(modules.values())
 telemetry=[x['gpu'] for x in cases if x['name'] in names];ns=sum(x['samples'] for x in telemetry)
 gpu=dict(samples=ns,memory_peak_mib=max(x['memory_peak_mib'] for x in telemetry),utilization_mean_pct=sum(x['gpu_utilization_mean_pct']*x['samples'] for x in telemetry)/ns,memory_controller_utilization_mean_pct=sum(x['memory_controller_utilization_mean_pct']*x['samples'] for x in telemetry)/ns,power_mean_w=sum(x['power_mean_w']*x['samples'] for x in telemetry)/ns)
 q=[f['quality']['metrics'] for f in frames]
 quality=dict(frames=len(q),all_checked_metrics_passed=all(f['quality']['checked_metrics_passed'] for f in frames),min_volume_ratio=min(x['min_volume_ratio'] for x in q),max_invalid_tets=max(x['invalid_tets'] for x in q),max_healed_edges=max(x['healed_edges'] for x in q),max_unexpected_neighbor_changes=max(x['unexpected_neighbor_changes'] for x in q),total_new_cut_edges=sum(x['new_cut_edges'] for x in q),max_blade_body_overlap_m=max(x['blade_body_overlap_m'] for x in q),max_wrist_overlap_m=max(x['wrist_overlap_m'] for x in q),min_instrument_distance_m=min(x['instrument_min_distance_m'] for x in q))
 return dict(cases=names,performance=dict(frames=len(times),mean_ms=float(times.mean()),fps=float(1000/times.mean()),p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),p99_ms=float(np.percentile(times,99)),max_ms=float(times.max()),over_40ms=int((times>40).sum()),over_50ms=int((times>50).sum()),rolling20_min_hz=min(data[n]['performance']['rolling20_min_hz'] for n in names)),modules_ms=modules,stages=stages,gpu=gpu,quality=quality)
A=aggregate(['final-a1-v8','final-a2-v8']);B=aggregate(['final-b1-runtime','final-b2-runtime'])
result=dict(schema='runtime_abba_v1',baseline=A,candidate=B,cases=cases,fps_gain_pct=(B['performance']['fps']/A['performance']['fps']-1)*100,native_step_reduction_pct=(1-B['modules_ms']['native_step']/A['modules_ms']['native_step'])*100,stable_25hz_approved=False,all_20frame_windows_above_20hz=False,formal_physics_domain_approved=False,training_domain_approved=False,scope='single env, original wall-traction seed0 1410-step three-incision path, CPU-returned 1280x960 RGB-D plus policy observation; desktop/video transport and dense privileged labels excluded')
(ROOT/'results/final-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
with (ROOT/'results/final-metrics.csv').open('w') as stream:
 writer=csv.writer(stream);writer.writerow(['case','fps','mean_ms','p50_ms','p95_ms','p99_ms','slowest20_hz','native_ms','gpu_util_pct','peak_mib'])
 for x in cases:
  t=x['performance'];writer.writerow([x['name'],t['fps'],t['mean_ms'],t['p50_ms'],t['p95_ms'],t['p99_ms'],t['rolling20_min_hz'],x['parts_ms']['native_step'],x['gpu']['gpu_utilization_mean_pct'],x['gpu']['memory_peak_mib']])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,constrained_layout=True)
for axis,group,color,title in [(axes[0],A,'#777777','Frozen V8'),(axes[1],B,'#087f8c','Retained native runtime')]:
 for i,n in enumerate(group['cases']):
  times=[f['frame_ms'] for f in data[n]['frames']];axis.plot(times,lw=.65,alpha=.65,label=n)
 axis.axhline(40,color='#c53535',ls='--',lw=1,label='25 Hz / 40 ms');axis.axhline(50,color='#dc9c21',ls=':',lw=1,label='20 Hz / 50 ms');axis.set_ylim(20,85);axis.set_ylabel('Full frame (ms)');axis.set_title(title);axis.legend(loc='upper left',ncol=4,fontsize=8);axis.grid(alpha=.15)
axes[1].set_xlabel('Frame index (all 1410 frames retained)')
for extension in ['png','svg']:fig.savefig(FIG/('full-frame-latency.'+extension),dpi=160)
plt.close(fig)
stages=list(A['stages']);x=np.arange(len(stages));fig,axis=plt.subplots(figsize=(13,5),constrained_layout=True)
axis.bar(x-.19,[A['stages'][s]['mean_ms'] for s in stages],.38,label='Frozen V8',color='#858b91');axis.bar(x+.19,[B['stages'][s]['mean_ms'] for s in stages],.38,label='Retained runtime',color='#087f8c');axis.axhline(40,color='#c53535',ls='--',label='25 Hz budget');axis.set_xticks(x,stages,rotation=45,ha='right');axis.set_ylabel('Mean full frame (ms)');axis.legend();axis.grid(axis='y',alpha=.15)
for extension in ['png','svg']:fig.savefig(FIG/('stage-comparison.'+extension),dpi=160)
plt.close(fig)
fig,axis=plt.subplots(figsize=(9,3),constrained_layout=True);left=np.zeros(2)
for label,color in zip(A['modules_ms'],['#087f8c','#89b4bb','#b8cfcc','#e3ad69','#c4c4c4']):
 values=[A['modules_ms'][label],B['modules_ms'][label]];axis.barh(['V8','Runtime'],values,left=left,label=label,color=color);left+=values
axis.axvline(40,color='#c53535',ls='--');axis.set_xlabel('Disjoint mean wall-clock envelopes (ms)');axis.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=3,fontsize=8)
for extension in ['png','svg']:fig.savefig(FIG/('frame-budget.'+extension),dpi=160)
plt.close(fig)
print(json.dumps({key:result[key] for key in ['fps_gain_pct','native_step_reduction_pct']},indent=2));print(json.dumps({'baseline':A['performance'],'candidate':B['performance'],'candidate_modules':B['modules_ms'],'candidate_quality':B['quality']},indent=2))
