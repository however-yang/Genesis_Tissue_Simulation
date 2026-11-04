"""Analyze Nsight node traces without double counting CPU wait and GPU work.

Perfetto output keeps CPU thread IDs and CUDA stream/context lanes distinct.
Profiles carry instrumentation overhead and are not acceptance benchmarks.
"""
import argparse,bisect,csv,json,os,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('database',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-spans',type=Path);p.add_argument('--plot',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
c=sqlite3.connect('file:'+str(a.database)+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")};names={r['id']:r['value'] for r in c.execute('select * from StringIds')}
frames=[dict(r) for r in c.execute("select start,end from NVTX_EVENTS where text='frame' and end is not null order by start")]
assert frames,'No captured frame ranges'
lo=frames[0]['start'];hi=frames[-1]['end'];n=len(frames)
def read(table):
 return [dict(r) for r in c.execute('select * from '+table+' where start<? and end>? order by start',(hi,lo))] if table in tables else []
def union(v):
 total=0;end=-1
 for s,e in sorted(v):
  if e>end:total+=e-max(s,end);end=e
 return total
def clipped(v,s,e):return [(max(s,x['start']),min(e,x['end'])) for x in v if x['end']>s and x['start']<e]
K='CUPTI_ACTIVITY_KIND_KERNEL';C='CUPTI_ACTIVITY_KIND_MEMCPY';M='CUPTI_ACTIVITY_KIND_MEMSET';G='CUPTI_ACTIVITY_KIND_GRAPH_TRACE';GL='OPENGL_WORKLOAD'
data={t:read(t) for t in [K,C,M,G,GL,'CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER','OPENGL_API','NVTX_EVENTS']}
rows=[]
for i,f in enumerate(frames):
 s,e=f['start'],f['end'];iv={t:clipped(data[t],s,e) for t in [K,C,M,G,GL]};cuda=iv[K]+iv[C]+iv[M]
 rows.append(dict(frame=i,wall_ms=(e-s)/1e6,kernel_union_ms=union(iv[K])/1e6,memcpy_union_ms=union(iv[C])/1e6,memset_union_ms=union(iv[M])/1e6,cuda_union_ms=union(cuda)/1e6,graph_envelope_ms=union(iv[G])/1e6,opengl_union_ms=union(iv[GL])/1e6,gpu_union_ms=union(cuda+iv[GL])/1e6))
def summary(table,column):
 groups={}
 for x in data[table]:
  name=names.get(x.get(column),str(x.get(column)));r=groups.setdefault(name,dict(name=name,calls=0,sum_ns=0));r['calls']+=1;r['sum_ns']+=min(x['end'],hi)-max(x['start'],lo)
 return [dict(name=x['name'],calls=x['calls'],calls_per_frame=x['calls']/n,sum_ms_per_frame=x['sum_ns']/n/1e6) for x in sorted(groups.values(),key=lambda x:-x['sum_ns'])]
result=dict(schema='nsys_node_intervals_v2',database=str(a.database),captured_frames=n,mean={k:sum(r[k] for r in rows)/n for k in rows[0] if k!='frame'},frames=rows,kernels=summary(K,'demangledName'),cuda_api=summary('CUPTI_ACTIVITY_KIND_RUNTIME','nameId'),gl_api=summary('OPENGL_API','nameId'),gl_workloads=summary(GL,'nameId'),caveats=['CPU API times are wall-clock envelopes including prior GPU wait and possible driver spin. They cannot be added to actual GPU intervals.','Graph envelopes are reported separately; GPU union uses actual kernel/copy/memset node intervals, not graph envelopes.','On-CPU thread time includes native driver spin. No scheduler trace or hardware performance counters were available; CPU/GPU residuals are not proof of cache/bandwidth or GIL causes.','Profiled windows are instrumentation diagnostics, not unprofiled FPS acceptance.'])
copy_kinds={r['id']:r['label'] for r in c.execute('select id,label from ENUM_CUDA_MEMCPY_OPER')} if 'ENUM_CUDA_MEMCPY_OPER' in tables else {}
copygroups={}
for x in data[C]:
 r=copygroups.setdefault(copy_kinds.get(x['copyKind'],str(x['copyKind'])),dict(calls=0,bytes=0,gpu_ms=0));r['calls']+=1;r['bytes']+=x['bytes'];r['gpu_ms']+=(x['end']-x['start'])/1e6
result['copies']=copygroups
# Device coverage during CPU API envelopes is descriptive, not proof of a stall.
# In particular it must not be added to device time as another cost.
def merged_intervals(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    return merged
active=merged_intervals([(max(lo,x['start']),min(hi,x['end'])) for t in [K,C,M] for x in data[t]])
starts=[s for s,e in active];prefix=[0]
for start,end in active:prefix.append(prefix[-1]+end-start)
def covered_until(t):
    i=bisect.bisect_right(starts,t)-1
    return 0 if i<0 else prefix[i]+max(0,min(t,active[i][1])-active[i][0])
api_coverage={}
for x in data['CUPTI_ACTIVITY_KIND_RUNTIME']:
    start=max(lo,x['start']);end=min(hi,x['end']);name=names.get(x['nameId'],str(x['nameId']))
    item=api_coverage.setdefault(name,dict(calls=0,wall_ns=0,covered_cuda_ns=0));item['calls']+=1;item['wall_ns']+=end-start;item['covered_cuda_ns']+=covered_until(end)-covered_until(start)
result['cuda_api_device_overlap']=[dict(name=name,calls_per_frame=x['calls']/n,wall_ms_per_frame=x['wall_ns']/n/1e6,cuda_active_during_api_ms_per_frame=x['covered_cuda_ns']/n/1e6,no_cuda_activity_during_api_ms_per_frame=(x['wall_ns']-x['covered_cuda_ns'])/n/1e6) for name,x in sorted(api_coverage.items(),key=lambda kv:-kv[1]['wall_ns'])]
if a.cpu_spans:
 cpu=json.loads(a.cpu_spans.read_text());result['cpu_spans_per_frame']={key:{k:(v if k=='calls' else v/n) for k,v in values.items()} for key,values in cpu.items()}
(a.output/'summary.json').write_text(json.dumps(result,indent=2))
with (a.output/'frames.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
events=[]
def event(x,name,pid,tid):
 s=max(lo,x['start']);e=min(hi,x['end'])
 if e>s:events.append(dict(name=name,ph='X',pid=pid,tid=tid,ts=(s-lo)/1000,dur=(e-s)/1000))
for x in data['NVTX_EVENTS']:
 name=x.get('text') or names.get(x.get('textId'),'NVTX');event(x,name,1,'thread '+str(x.get('globalTid'))+' ranges')
for t in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER','OPENGL_API']:
 for x in data[t]:event(x,names.get(x.get('nameId'),t),1,'thread '+str(x.get('globalTid'))+' '+t)
for t in [K,C,M,G,GL]:
 for x in data[t]:
  name=names.get(x.get('shortName'),names.get(x.get('nameId'),t));lane=t+' dev'+str(x.get('deviceId'))+'/ctx'+str(x.get('contextId'))+'/stream'+str(x.get('streamId'))
  event(x,name,2,lane)
(a.output/'timeline.json').write_text(json.dumps(dict(traceEvents=events,displayTimeUnit='ms'),separators=(',',':')))
if a.plot:
 os.environ['MPLCONFIGDIR']=str(a.output/'matplotlib-cache')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 end=frames[min(n-1,2)]['end'];fig,ax=plt.subplots(figsize=(15,6));labels=[]
 wanted=['control','native.step','quality.all','quality.topology','state.snapshot','render.update_mesh','render.rgbd','cv.cvtColor','policy.observation']
 groups=[]
 for name in wanted:
  for tid in sorted({x['globalTid'] for x in data['NVTX_EVENTS'] if x.get('text')==name}):
   ranges=clipped([x for x in data['NVTX_EVENTS'] if x.get('text')==name and x['globalTid']==tid],lo,end)
   if ranges:groups.append((name+' [thread '+str(tid%2**24)+']',ranges))
 for t,label in [(K,'GPU: CUDA kernels'),(C,'GPU: CUDA copies'),(G,'CUDA graph envelopes (not additive)'),(GL,'GPU: OpenGL workloads')]:
  ranges=clipped(data[t],lo,end)
  if ranges:groups.append((label,ranges))
 for i,(label,ranges) in enumerate(groups):
  ax.broken_barh([((s-lo)/1e6,(e-s)/1e6) for s,e in ranges],(i-.34,.68),facecolors=plt.get_cmap('tab10')(i%10),edgecolors='none');labels.append(label)
 for i,f in enumerate(frames[:3]):ax.axvline((f['start']-lo)/1e6,color='grey',lw=.5)
 ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,(end-lo)/1e6);ax.set_xlabel('Milliseconds from captured frame start');ax.set_title('Measured CPU / GPU timeline (overlapping lanes must not be added)');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(a.output/'timeline.svg');fig.savefig(a.output/'timeline.png',dpi=150);plt.close(fig)
print(json.dumps(dict(frames=n,mean=result['mean']),indent=2))
