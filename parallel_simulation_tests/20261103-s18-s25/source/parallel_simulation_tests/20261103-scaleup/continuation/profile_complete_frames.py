"""Attribute native CUDA work to whole-frame APIs on a fresh >100k-node run.

Profiler overhead and shared CPU load make this diagnostic, not FPS acceptance.
The fixture, solver, iteration budgets and physical quality guards are intact.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time


def summarize_trace(path):
    events=json.loads(path.read_text())['traceEvents']
    spans=[e for e in events if e.get('ph')=='X' and e.get('name','').startswith('native_api/')]
    runtime={e.get('args',{}).get('correlation'):e for e in events
             if e.get('cat')=='cuda_runtime' and 'correlation' in e.get('args',{})}
    by_api=defaultdict(lambda: dict(cpu_calls=0,cpu_inclusive_ms=0.,kernel_launches=0,
                                  kernel_duration_sum_ms=0.,transfer_duration_sum_ms=0.))
    for e in spans:
        row=by_api[e['name'].split('/',1)[1]]
        row['cpu_calls']+=1
        row['cpu_inclusive_ms']+=e['dur']/1000
    kernels=defaultdict(list)
    intervals=[]
    unattributed=0
    for e in events:
        category=e.get('cat')
        if category not in ('kernel','gpu_memcpy','gpu_memset') or e.get('ph')!='X':
            continue
        api='unattributed'
        call=runtime.get(e.get('args',{}).get('correlation'))
        if call:
            parents=[s for s in spans if s.get('tid')==call.get('tid') and s['ts']<=call['ts']<s['ts']+s['dur']]
            if parents:
                api=min(parents,key=lambda s:s['dur'])['name'].split('/',1)[1]
        row=by_api[api]
        if category=='kernel':
            row['kernel_launches']+=1
            row['kernel_duration_sum_ms']+=e['dur']/1000
            kernels[e['name']].append(e['dur']/1000)
            intervals.append((e['ts'],e['ts']+e['dur']))
            unattributed+=api=='unattributed'
        else:
            row['transfer_duration_sum_ms']+=e['dur']/1000
    merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(merged[-1][1],hi))
        else:
            merged.append((lo,hi))
    return dict(apis=dict(by_api),unattributed_kernel_launches=unattributed,
        kernel_union_active_ms=sum(hi-lo for lo,hi in merged)/1000,
        kernel_span_ms=(max(hi for lo,hi in intervals)-min(lo for lo,hi in intervals))/1000 if intervals else 0.,
        kernels=sorted([dict(name=name,launches=len(values),total_ms=sum(values))
                        for name,values in kernels.items()],key=lambda x:-x['total_ms']),
        note='CPU API durations include waits. GPU sums can overlap; do not add CPU and GPU columns or nested phases.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--library-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'
    root=args.release.resolve(strict=True)
    sys.path[:0]=[str(root),str(root/'endoscopic_tissue_backups/2026-10-31-interaction-fixes')]
    import torch
    import validate_strict as worker
    from endoscopic_tissue.training.provenance import sha256
    args.output.mkdir(parents=True,exist_ok=False)
    torch.cuda.init()
    assert torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0)
    targets={300,600,900,1230,1244,1245,1300}
    record=dict(diagnostic_only=True,performance_accepted=False,candidate_adopted=False,completed=False,
        worker_sha256=sha256(__file__),library_sha256=sha256(args.library_dir/'libLiverCudaSim.so'),
        device=torch.cuda.get_device_name(0),requested_ticks=sorted(targets),frames=[])
    original=worker.NativeTissue
    class Recorder(original):
        def check(self,name,*values):
            if getattr(self,'recording_frame',False):
                with torch.profiler.record_function('native_api/'+name):
                    return super().check(name,*values)
            return super().check(name,*values)

        def step(self,command):
            tick=self.tick+1
            if tick not in targets:
                return super().step(command)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False) as profiler:
                started=time.perf_counter()
                self.recording_frame=True
                try:
                    with torch.profiler.record_function('native_complete_step'):
                        observed=super().step(command)
                        torch.cuda.synchronize()
                finally:
                    self.recording_frame=False
                elapsed=(time.perf_counter()-started)*1000
            trace=args.output/f'frame-{tick}.json'
            profiler.export_chrome_trace(str(trace))
            summary=summarize_trace(trace)
            summary.update(native_tick=tick,profiled_frame_wall_ms=elapsed,trace_sha256=sha256(trace),
                material_nodes=int(self.grid.inside.sum()),native_material_quality=self.material_quality().tolist())
            record['frames'].append(summary)
            (args.output/'progress.json').write_text(json.dumps(record,indent=2))
            print('PROFILED',tick,'wall_ms',elapsed,'gpu_union_ms',summary['kernel_union_active_ms'],flush=True)
            return observed
    worker.NativeTissue=Recorder
    sys.argv=['validate_strict','--case','kidney-reference','--dll',str(args.library_dir/'libLiverCudaSim.so'),
              '--output',str(args.output/'full-run')]
    try:
        worker.main()
        assert {f['native_tick'] for f in record['frames']}==targets
        assert all(f['material_nodes']>100000 and f['kernels'] for f in record['frames'])
        record['completed']=True
    except BaseException as error:
        record['error']=repr(error)
        raise
    finally:
        with (args.output/'result.json').open('x') as stream:json.dump(record,stream,indent=2)


if __name__=='__main__':
    main()
