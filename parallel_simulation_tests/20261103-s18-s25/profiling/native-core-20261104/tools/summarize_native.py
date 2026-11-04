"""Read measured frames without adding child timers to parents."""
import csv,json,statistics,os
from pathlib import Path
p=Path(__file__).resolve().parents[1];os.sched_setaffinity(0,[30,31]);rows=[]
for file in sorted((p/'results').glob('*/result.json')):
 d=json.loads(file.read_text());perf=d.get('performance');name=file.parent.name
 if not perf:continue
 samples=[]
 mon=file.parent/'gpu-monitor.jsonl'
 if mon.exists():
  samples=[json.loads(x) for x in mon.read_text().splitlines() if x]
  samples=[x for x in samples if d.get('measurement_start_unix_s',0)<=x['time']<=d.get('measurement_end_unix_s',float('inf'))]
 avg=lambda key:statistics.mean([x[key] for x in samples if isinstance(x.get(key),(float,int))]) if samples and any(isinstance(x.get(key),(float,int)) for x in samples) else None
 frames=d['frames'];means={k:statistics.mean(f['timings_ms'][k] for f in frames) for k in frames[0]['timings_ms']}
 vision=statistics.mean(f['render_ms'] for f in frames);other=perf['mean_ms']-sum(means.get(k,0) for k in ['native_step','quality_and_export','snapshot'])-vision
 row=dict(name=name,completed=d.get('completed'),quality_passed=d.get('quality_passed'),profiled=d.get('profiled'),diagnostic=bool(d.get('native_validation') or d.get('validation_requested') or d.get('interrupted_native_validation') or d.get('workload_profiled') or d.get('topology_validation') or d.get('assembly_audit') or any(t in name for t in ['validation','workload','audit'])),rgbd=d['render']['fresh_rgbd_each_step'],labels=d.get('labels'),**perf,native_ms=means.get('native_step'),quality_ms=means.get('quality_and_export'),state_ms=means.get('snapshot'),vision_ms=vision,other_ms=other,gpu_percent=avg('gpu_percent'),peak_memory_mib=max((x['used_mib'] for x in samples),default=None),host_cpu_percent=avg('host_cpu_percent'),stable20_hz=perf['rolling20_min_hz']>=20,all_frames_50ms=perf['over_50ms']==0)
 rows.append(row)
(p/'results/performance-table.json').write_text(json.dumps(rows,indent=2))
with (p/'results/performance-table.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
lines=['|测试|完整|局部检查|诊断|RGB-D|Hz|P50 ms|P95 ms|最慢20帧 Hz|>50ms帧|GPU均值|峰值MiB|','|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 fmt=lambda x:'—' if x is None else f'{x:.2f}'
 lines.append('|'+ '|'.join([r['name'],str(r['completed']),str(r['quality_passed']),str(r['diagnostic'] or r['profiled']),str(r['rgbd']),fmt(r['fps']),fmt(r['p50_ms']),fmt(r['p95_ms']),fmt(r['rolling20_min_hz']),str(r['over_50ms']),fmt(r['gpu_percent']),fmt(r['peak_memory_mib'])])+'|')
(p/'results/performance-table.md').write_text('\n'.join(lines)+'\n')
print('Measured runs',len(rows))
