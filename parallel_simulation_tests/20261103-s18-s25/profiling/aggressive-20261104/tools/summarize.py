"""All runs remain in the data; partial/profiling/validation runs are labelled."""
import json,csv,statistics,math
from pathlib import Path
p=Path(__file__).resolve().parents[1];rows=[]
def extrema(values,fn):
 values=[x for x in values if x is not None and math.isfinite(x)]
 return fn(values) if values else None
for file in sorted((p/'results').glob('*/result.json')):
 d=json.loads(file.read_text());perf=d.get('performance',{})
 if 'fps' not in perf or not isinstance(d.get('frames'),list):continue
 frames=d['frames'];name=file.parent.name
 r=dict(name=name,source=d.get('source'),dll_sha256=d.get('dll_sha256'),profiled=d.get('profiled'),complete=d.get('completed'),quality_passed=d.get('quality_passed'),training_vision=d.get('training_vision'),render=d.get('render',{}).get('fresh_rgbd_each_step'),topology_validation=d.get('topology_validation',False),native_query_validation='native_validation' in d,error=d.get('error'),**perf)
 for key in ['gpu_topology','fused_mesh','parallel_frame','render_nogil','pbo','persistent_soup','static_visuals','direct_rgb','opencv_threads']:r[key]=d.get(key,False)
 for key in ['native_step','surface_export','quality_and_export','quality_cpu','same_frame_render','snapshot']:
  values=[f['timings_ms'].get(key,0) for f in frames];r[key+'_mean_ms']=statistics.mean(values)
  if key=='native_step':r['native_step_p95_ms']=sorted(values)[min(len(values)-1,int(.95*len(values)))]
 r['post_capture_mean_ms']=statistics.mean(f['render_ms'] for f in frames)
 r['phase_caveat']='quality_and_export includes same_frame_render when parallel_frame=True; overlapping components must not be added'
 metric=[f['quality']['metrics'] for f in frames]
 r.update(cut_edges_total=sum(m.get('new_cut_edges',0) for m in metric),invalid_tets_max=extrema([m.get('invalid_tets') for m in metric],max),healed_edges=sum(m.get('healed_edges',0) for m in metric),nonmanifold_edges_max=max(m.get('surface_topology',{}).get('positional_nonmanifold_edges',0) for m in metric),min_volume_ratio=extrema([m.get('min_volume_ratio') for m in metric],min),max_body_overlap_m=extrema([m.get('blade_body_overlap_m') for m in metric],max),failure_reasons=sorted({v for f in frames for v in f['quality']['numerical_failure_reasons']}),training_domain_approved=False)
 q=file.parent/'gpu-monitor.jsonl';samples=[json.loads(line) for line in q.read_text().splitlines()] if q.exists() else []
 lo=d.get('measurement_start_unix_s');hi=d.get('measurement_end_unix_s');samples=[s for s in samples if lo is not None and hi is not None and lo<=s['time']<=hi]
 r['measurement_resource_samples']=len(samples)
 if samples:r.update(gpu_utilization_mean_pct=statistics.mean(s['gpu_percent'] for s in samples),gpu_utilization_peak_pct=max(s['gpu_percent'] for s in samples),peak_vram_mib=max(s['used_mib'] for s in samples),host_cpu_mean_pct=statistics.mean(s['host_cpu_percent'] for s in samples))
 r['valid_full_performance_run']=bool(d.get('completed') and d.get('quality_passed') and not d.get('profiled') and not d.get('topology_validation') and 'native_validation' not in d)
 r['known_archive_regression_failed']=bool(d.get('gpu_topology') and d.get('dll_sha256')==json.loads((p/'results/r6-gpu-checks/result.json').read_text())['dll_sha256'])
 r['mean_20hz']=bool(r['valid_full_performance_run'] and perf['fps']>=20);r['mean_and_p95_20hz']=bool(r['mean_20hz'] and perf['p95_ms']<=50);r['all_frames_50ms']=bool(r['mean_20hz'] and perf['over_50ms']==0)
 rows.append(r)
(p/'results/performance-summary.json').write_text(json.dumps(rows,indent=2));fields=list(dict.fromkeys(k for row in rows for k in row))
with (p/'results/performance-summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
lines=['| 测试 | 完整/检查 | 平均 Hz | P50 ms | P95 ms | >50 ms 帧 | GPU % | 显存 MiB |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
 if r['profiled'] or not r['render'] or r['topology_validation'] or r['native_query_validation']:continue
 lines.append(f"| {r['name']} | {'完整，但新检查的快照回归失败，未采用' if r['known_archive_regression_failed'] else ('完成，已启用检查通过' if r['valid_full_performance_run'] else '失败/中止，不能验收')} | {r['fps']:.2f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['over_50ms']} | {r.get('gpu_utilization_mean_pct',0):.1f} | {r.get('peak_vram_mib',0):.0f} |")
(p/'results/performance-table.md').write_text('\n'.join(lines)+'\n')
for r in rows:
 if r['valid_full_performance_run'] and r['render']:print(r['name'],round(r['fps'],2),round(r['p95_ms'],2),'cut',r['cut_edges_total'])
