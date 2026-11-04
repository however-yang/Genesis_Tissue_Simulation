"""Summarize disjoint frame envelopes and sampled GPU telemetry; no simulation."""
import argparse,json,os,statistics
from pathlib import Path
os.sched_setaffinity(0,[30,31])
ROOT=Path(__file__).resolve().parents[1]

def gpu_samples(name,data):
 path=ROOT/'logs'/(name+'-gpu.jsonl')
 samples=[]
 for line in path.read_text().splitlines():
  row=json.loads(line)
  if data.get('measurement_start_unix_s',float('-inf'))<=row['unix']<=data.get('measurement_end_unix_s',float('inf')) and row['returncode']==0:
   fields=[x.strip() for x in row['value'].split(',')]
   if len(fields)==5:
    try:samples.append(tuple(float(x) for x in fields[1:]))
    except ValueError:pass
 if not samples:return dict(samples=0)
 return dict(samples=len(samples),sampling_interval_s=2,memory_peak_mib=max(x[0] for x in samples),gpu_utilization_mean_pct=statistics.mean(x[1] for x in samples),memory_controller_utilization_mean_pct=statistics.mean(x[2] for x in samples),power_mean_w=statistics.mean(x[3] for x in samples),scope='measurement window only; NVML sampled percentages, not SM occupancy or achieved memory bandwidth')

def case(name):
 d=json.loads((ROOT/'results'/name/'result.json').read_text())
 frames=d['frames'];mean=lambda key:statistics.mean(f['timings_ms'].get(key,0) for f in frames)
 parts=dict(native_step=mean('native_step'),quality_including_surface_export=mean('quality_and_export'),snapshot=mean('snapshot'),render_and_rgbd=statistics.mean(f['render_ms'] for f in frames))
 parts['other_control_policy_python']=d['performance']['mean_ms']-sum(parts.values())
 return dict(name=name,completed=d.get('completed'),quality_passed=d.get('quality_passed'),performance=d['performance'],parts_ms=parts,parts_pct={k:v/d['performance']['mean_ms']*100 for k,v in parts.items()},gpu=gpu_samples(name,d))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('names',nargs='+');parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 data=[case(n) for n in a.names]
 a.output.write_text(json.dumps(data,indent=2)+'\n')
 for x in data:print(x['name'],json.dumps({k:x[k] for k in ('performance','gpu')},ensure_ascii=False))
