"""Read existing measurements; interval unions avoid counting concurrent work twice."""
import json,sqlite3,statistics,csv,os
from pathlib import Path
os.sched_setaffinity(0,[30])
p=Path(__file__).resolve().parents[1];out=p/'results/pool-analysis';out.mkdir(exist_ok=True)
def merged(rows):
 result=[]
 for a,b in sorted(rows):
  if b<=a:continue
  if result and a<=result[-1][1]:result[-1][1]=max(b,result[-1][1])
  else:result.append([a,b])
 return result
def duration(rows):return sum(b-a for a,b in merged(rows))
def intersect(a,b):
 a=merged(a);b=merged(b);i=j=0;total=0
 while i<len(a) and j<len(b):
  total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return total
summaries=[]
for f in sorted((p/'results').glob('pool-*/result.json')):
 d=json.loads(f.read_text())
 if not d.get('passed'):continue
 lo,hi=d['measure_start_ns'],d['measure_end_ns'];wall=hi-lo
 wanted={(x['worker_id'],x['request']):x for x in d['rows']};workers={};send=[];compute=[];cpu=[];return_gap=[];transport=[]
 events=[]
 for q in sorted(f.parent.glob('worker-*/worker-spans.jsonl')):
  w=int(q.parent.name.split('-')[-1]);ranges=[]
  for line in q.read_text().splitlines():
   x=json.loads(line);key=(w,x['request'])
   if key not in wanted:continue
   row=wanted[key];ranges.append((x['start_ns'],x['ready_ns']));send.append((x['end_ns']-x['ready_ns'])/1e6);compute.append((x['ready_ns']-x['start_ns'])/1e6);cpu.append(x['thread_cpu_ns']/1e6);return_gap.append((row['received_ns']-x['ready_ns'])/1e6)
   if 'image_transport' in row:transport.append(row['image_transport'])
   for name,a,b in [('worker compute (includes GPU/API waits)',x['start_ns'],x['ready_ns']),('pipe send (includes blocking)',x['ready_ns'],x['end_ns'])]:events.append(dict(name=name,ph='X',pid=x['pid'],tid=w,ts=(a-lo)/1000,dur=(b-a)/1000))
  workers[w]=ranges
 allranges=[r for rs in workers.values() for r in rs]
 r=dict(name=f.parent.name,**d['performance'],workers=d['workers'],profiled='trace' in f.parent.name,
  compute_span_mean_ms=statistics.mean(compute),pipe_send_mean_ms=statistics.mean(send),worker_thread_cpu_mean_ms=statistics.mean(cpu),ready_to_parent_mean_ms=statistics.mean(return_gap),
  cpu_compute_span_union_pct=100*duration(allranges)/wall,
  cpu_two_worker_compute_overlap_pct=100*intersect(workers.get(0,[]),workers.get(1,[]))/wall,
  mean_payload_bytes=statistics.mean(x['payload_bytes'] for x in d['rows']))
 for key in ['native_step','quality_and_export','surface_export','snapshot','vision']:
  vals=[x['timings_ms'][key] for x in d['rows'] if key in x['timings_ms']]
  r[key+'_mean_ms']=statistics.mean(vals) if vals else None
 if transport:
  r['image_transport_mean']={k:statistics.mean(t[k] for t in transport if k in t) for k in ['worker_copy_ms','parent_copy_ms','bytes'] if any(k in t for t in transport)}
 samples=[]
 q=f.parent/'gpu-monitor.jsonl'
 if q.exists():samples=[json.loads(line) for line in q.read_text().splitlines()]
 selected=[x for x in samples if d['measure_start_unix_s']<=x['time']<=d['measure_end_unix_s']]
 r['steady_resource_samples']=len(selected)
 if selected:r.update(gpu_mean_pct=statistics.mean(x['gpu_percent'] for x in selected),gpu_peak_pct=max(x['gpu_percent'] for x in selected),peak_vram_mib=max(x['used_mib'] for x in selected),host_cpu_mean_pct=statistics.mean(x['host_cpu_percent'] for x in selected))
 summaries.append(r);(out/(f.parent.name+'-cpu-timeline.json')).write_text(json.dumps(dict(traceEvents=events,displayTimeUnit='ms')))
(out/'summary.json').write_text(json.dumps(dict(runs=summaries,caveat='CPU compute spans include CUDA/GL API waits; pipe send includes blocking. Parent receive gap includes pickle/copy/scheduling. Resources sampled during measurement only; util is NVML, not SM occupancy. Short incision window, not complete episodes.'),indent=2))
# Device timelines in one Nsight clock; PID decoding includes machine prefix in high bits.
db=p/'results/pool-final-2-trace.sqlite';c=sqlite3.connect(db);c.row_factory=sqlite3.Row
lo,hi=c.execute("select start,end from NVTX_EVENTS where text='pool-measure'").fetchone();tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")};names=dict(c.execute('select id,value from StringIds'));lanes={};events=[]
for table,label,col in [('NVTX_EVENTS','CPU NVTX','text'),('CUPTI_ACTIVITY_KIND_RUNTIME','CPU CUDA API','nameId'),('CUPTI_ACTIVITY_KIND_KERNEL','GPU CUDA kernels','shortName'),('CUPTI_ACTIVITY_KIND_MEMCPY','GPU CUDA copies',None),('CUPTI_ACTIVITY_KIND_MEMSET','GPU memset',None),('OPENGL_WORKLOAD','GPU OpenGL workloads','nameId')]:
 if table not in tables:continue
 for x in c.execute(f'select * from {table} where start<? and end>?',(hi,lo)):
  a,b=max(x['start'],lo),min(x['end'],hi);ident=x['globalPid'] if 'globalPid' in x.keys() else x['globalTid'];pid=(ident>>24)&0xffffff
  if table=='NVTX_EVENTS' and x['text']=='pool-measure':continue
  name=x[col] if col=='text' else names.get(x[col],str(x[col])) if col else label
  events.append(dict(name=name,ph='X',pid=pid,tid=label,ts=(a-lo)/1000,dur=(b-a)/1000))
  lanes.setdefault((pid,label),[]).append((a,b))
pids=sorted({pid for pid,label in lanes if label=='GPU CUDA kernels'});groups={pid:[r for (q,label),rs in lanes.items() if q==pid and label.startswith('GPU') for r in rs] for pid in pids};both=[r for rs in groups.values() for r in rs]
trace=dict(window_s=(hi-lo)/1e9,pids=pids,combined_gpu_union_pct=100*duration(both)/(hi-lo),two_process_gpu_overlap_pct=100*intersect(groups[pids[0]],groups[pids[1]])/(hi-lo),per_process_gpu_union_pct={str(pid):100*duration(rs)/(hi-lo) for pid,rs in groups.items()},lane_union_ms={str(pid)+' '+label:duration(rs)/1e6 for (pid,label),rs in lanes.items()},caveat='Captured NVTX window excludes Nsight start/stop overhead; each lane is an interval union, not a CPU/GPU additive partition. CPU GPU-wait ranges may overlap actual device activity. GPU overlap does not prove concurrent SM execution.')
(out/'device-summary.json').write_text(json.dumps(trace,indent=2));(out/'device-timeline.json').write_text(json.dumps(dict(traceEvents=events,displayTimeUnit='ms'),separators=(',',':')))
# Compact standalone chart for first second; full timeline above is inspectable in Perfetto.
os.environ['MPLCONFIGDIR']=str(p/'tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(14,6));labels=[]
for pid in pids:
 for label in ['CPU NVTX','CPU CUDA API','GPU CUDA kernels','GPU CUDA copies','GPU OpenGL workloads']:
  rs=lanes.get((pid,label),[]);points=[((a-lo)/1e6,(min(b,lo+1000000000)-a)/1e6) for a,b in rs if a<lo+1000000000]
  index=len(labels);ax.broken_barh(points,(index-.35,.7),facecolors=('tab:blue' if 'CPU' in label else 'tab:orange'));labels.append(f'PID {pid}: {label}')
ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,1000);ax.set_xlabel('Milliseconds from Nsight capture start');ax.set_title('Measured two-worker CPU/GPU timeline (inclusive CPU ranges; do not add lanes)');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(p/'figures/pool-timeline.svg');fig.savefig(p/'figures/pool-timeline.png',dpi=160);plt.close(fig)
print(json.dumps(trace,indent=2))
print([(x['name'],round(x['total_steps_s'],2),round(x['pipe_send_mean_ms'],2)) for x in summaries])
