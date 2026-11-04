import json,pathlib,re,statistics,csv
r=pathlib.Path(__file__).resolve().parents[1];out=r/'results';rows=[]
paths=list((out/'comparison-v1').rglob('measurement/result.json'))+list(out.glob('*/result.json'))
for p in paths:
 d=json.loads(p.read_text());perf=d.get('performance')
 if not perf or 'fps' not in perf:continue
 name=p.parents[2].name if p.parent.name=='measurement' else p.parent.name
 resource=p.parent/'gpu-monitor.jsonl';samples=[json.loads(x) for x in resource.read_text().splitlines()] if resource.exists() else []
 row=dict(name=name,profiled=d.get('profiled'),completed=d.get('completed'),quality_passed=d.get('quality_passed'),labels=d.get('labels'),training_vision=d.get('training_vision',False),render=d['render']['fresh_rgbd_each_step'],**perf,
  gpu_percent_process_mean=statistics.mean(s['gpu_percent'] for s in samples) if samples else None,gpu_percent_process_max=max((s['gpu_percent'] for s in samples),default=None),gpu_mib_process_peak=max((s['used_mib'] for s in samples),default=None),resource_scope='2 s samples including initialization; no exclusive SM occupancy claim',error=d.get('error'))
 start=d.get('measurement_start_unix_s');end=d.get('measurement_end_unix_s')
 selected=[s for s in samples if start is not None and end is not None and start<=s['time']<=end]
 row.update(source=d.get('source'),gpu_aligned_depth=d.get('gpu_aligned_depth',False),steady_resource_samples=len(selected),gpu_percent_steady_mean=statistics.mean(s['gpu_percent'] for s in selected) if selected else None,gpu_mib_steady_peak=max((s['used_mib'] for s in selected),default=None))
 if d.get('frames'):
  for key in ['native_step','surface_export','quality_and_export','snapshot']:
   row[key+'_mean_ms']=statistics.mean(f['timings_ms'].get(key,0) for f in d['frames'])
  row['render_mean_ms']=statistics.mean(f['render_ms'] for f in d['frames'])
 rows.append(row)
(out/'performance-summary.json').write_text(json.dumps(rows,indent=2))
fields=list(dict.fromkeys(k for row in rows for k in row))
with (out/'performance-summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
d=json.loads((out/'analysis-nodes/summary.json').read_text());groups={}
cut=set('k_SeverLinks k_JoinComponents k_FlattenComponents k_InitComponents k_LevelScissorsColumns k_SpreadScissorsLevels k_CutScissorsLevels k_CommitPending k_SeverPieceSeams k_ColumnAxis'.split())
recon=set('k_ComputeComponentFP k_BuildCutTriangles k_ClearCutFP k_ExpandVertices k_IndexFaces k_FeaturePoints k_AccumulateCutFP k_NormalizeCutFP k_Stitch k_NormalsScatter k_NormalsNormalize k_ClearNormals k_AdvectIntactSurface k_LookupConnectivity k_CountIsolated k_RebuildIsectWorld k_CompactFaces k_ClearFaceTable'.split())
contact=set('k_FindSurface k_UnifiedManifold k_CacheBladeBackContacts k_SurfaceDepth k_ClassifyContactGroups k_CacheContacts k_UpdateSurfacePoints k_BuildContactGroups k_BindSurface k_SurfaceReaction'.split())
for row in d['kernels']:
 name=re.search(r'k_[A-Za-z_0-9]+',row['name']).group()
 group='cut and component topology' if name in cut else 'surface reconstruction' if name in recon else 'explicit contact search/manifold' if name in contact else 'coupled material/tool solve, guards, telemetry (remaining kernels)'
 g=groups.setdefault(group,dict(kernel_ms_per_frame=0,calls_per_frame=0,kernels=[]));g['kernel_ms_per_frame']+=row['sum_ms_per_frame'];g['calls_per_frame']+=row['calls_per_frame'];g['kernels'].append(name)
total=sum(g['kernel_ms_per_frame'] for g in groups.values())
for g in groups.values():g['share_of_summed_kernel_time_percent']=g['kernel_ms_per_frame']/total*100
(out/'kernel-groups.json').write_text(json.dumps(dict(groups=groups,note='Disjoint named-kernel buckets, not independent physical models. Coupled material/tool kernels also contain contact responses. Times are actual GPU durations, not API latency.'),indent=2))
for row in rows:
 if not row['profiled']:print(row['name'],round(row['fps'],3),round(row['p95_ms'],3),row['completed'],row['quality_passed'])
print({k:{kk:vv for kk,vv in v.items() if kk!='kernels'} for k,v in groups.items()})
