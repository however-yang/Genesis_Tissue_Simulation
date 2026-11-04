"""Summaries use interval unions, not sums of overlapping CUDA/API wall times."""
import argparse,json,sqlite3,csv,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('database',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
c=sqlite3.connect(a.database);c.row_factory=sqlite3.Row
names={r['id']:r['value'] for r in c.execute('select * from StringIds')}
tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
frames=[dict(r) for r in c.execute("select start,end from NVTX_EVENTS where text='frame' order by start")]

def union(rows):
 total=0;last=-1
 for start,end in sorted(rows):
  if end>last:total+=end-max(last,start);last=end
 return total

def intervals(table,lo=None,hi=None):
 if table not in tables:return []
 rows=c.execute('select start,end from '+table).fetchall()
 return [(max(r['start'],lo) if lo is not None else r['start'],min(r['end'],hi) if hi is not None else r['end']) for r in rows if (lo is None or r['end']>lo) and (hi is None or r['start']<hi)]

def summary(table,name_col):
 if table not in tables:return []
 values=c.execute(f'select {name_col},count(*) as calls,sum(end-start) as ns from {table} group by {name_col} order by ns desc').fetchall()
 return [dict(name=names.get(v[name_col],str(v[name_col])),calls=v['calls'],calls_per_frame=v['calls']/len(frames),sum_ms_per_frame=v['ns']/len(frames)/1e6) for v in values]
rows=[];cuda_tables=['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_GRAPH_TRACE','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']
for i,f in enumerate(frames):
 lo,hi=f['start'],f['end'];groups={t:intervals(t,lo,hi) for t in cuda_tables};cuda=[v for group in groups.values() for v in group];gl=intervals('OPENGL_WORKLOAD',lo,hi)
 rows.append(dict(frame=i,wall_ms=(hi-lo)/1e6,cuda_union_ms=union(cuda)/1e6,kernel_union_ms=union(groups[cuda_tables[0]])/1e6,graph_envelope_ms=union(groups[cuda_tables[1]])/1e6,memcpy_union_ms=union(groups[cuda_tables[2]])/1e6,memset_union_ms=union(groups[cuda_tables[3]])/1e6,opengl_workload_union_ms=union(gl)/1e6,combined_gpu_union_ms=union(cuda+gl)/1e6))
result=dict(captured_frames=len(frames),mean={k:sum(r[k] for r in rows)/len(rows) for k in rows[0] if k!='frame'},frames=rows,
 kernels=summary('CUPTI_ACTIVITY_KIND_KERNEL','demangledName'),cuda_api=summary('CUPTI_ACTIVITY_KIND_RUNTIME','nameId'),gl_api=summary('OPENGL_API','nameId'),gl_workload=summary('OPENGL_WORKLOAD','nameId'),
 caveat='CUDA API durations include waiting and overlap GPU intervals; never add them to GPU time. Graph envelopes include internal launch gaps. OpenGL workload intervals are Nsight GPU records, not CPU GL call durations. Instrumented times are not acceptance FPS.')
result['copies']=[dict(r) for r in c.execute('select e.label,count(*) as calls,sum(bytes) as bytes,sum(end-start)/1e6 as gpu_ms from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on m.copyKind=e.id group by m.copyKind')]
(a.output/'summary.json').write_text(json.dumps(result,indent=2))
with (a.output/'frame-times.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
# Chrome/Perfetto timeline combines NVTX CPU ranges, CUDA kernels/graphs/copies, GL workloads and CUDA calls on independent lanes.
origin=frames[0]['start'];events=[]
for r in c.execute('select start,end,text from NVTX_EVENTS where end is not null and text is not null'):
 events.append(dict(name=r['text'],ph='X',pid=1,tid='CPU NVTX ranges (nested)',ts=(r['start']-origin)/1000,dur=(r['end']-r['start'])/1000))
for table,lane,col in [('CUPTI_ACTIVITY_KIND_KERNEL','CUDA kernels','shortName'),('CUPTI_ACTIVITY_KIND_GRAPH_TRACE','CUDA graph envelopes',None),('CUPTI_ACTIVITY_KIND_MEMCPY','CUDA copies',None),('CUPTI_ACTIVITY_KIND_RUNTIME','CPU CUDA API','nameId'),('OPENGL_WORKLOAD','OpenGL GPU workloads','nameId')]:
 if table not in tables:continue
 for r in c.execute('select * from '+table):
  if r['end']<=origin:continue
  events.append(dict(name=names.get(r[col],str(r[col])) if col else table,ph='X',pid=2 if table!='CUPTI_ACTIVITY_KIND_RUNTIME' else 1,tid=lane,ts=(r['start']-origin)/1000,dur=(r['end']-r['start'])/1000))
(a.output/'timeline.json').write_text(json.dumps(dict(traceEvents=events,displayTimeUnit='ms'),separators=(',',':')))
print(json.dumps(result['mean'],indent=2))
