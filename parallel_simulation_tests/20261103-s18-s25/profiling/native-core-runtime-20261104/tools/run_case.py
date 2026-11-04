"""Run one resource-guarded case; no environment changes or concurrent GPU jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=json.loads((ROOT/'baseline.json').read_text())
PROJECT=ROOT.parents[3]
PYTHON=PROJECT/'envs/genesis-py311-cu125-v1/bin/python'
GPU='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'


def run(name,library='baseline',core=0,kind='benchmark',max_frames=None,batch_observation=False,validate_observation=False,query_threads=64):
    assert name and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name)
    output=ROOT/'results'/name
    assert not output.exists(), output
    dll=Path(BASE['baseline_library']) if library=='baseline' else ROOT/'native/build/lib/libLiverCudaSim.so' if library=='candidate' else Path(library).resolve()
    assert dll==Path(BASE['baseline_library']) or dll.is_relative_to(ROOT), dll
    common=['--source',BASE['borrowed_python_source'],'--dll',str(dll),'--output',str(output),
            '--query-mode','3','--query-threads',str(query_threads),'--bound-cache','--core-optimizations',str(core),'--gpu-topology','--gpu-body-check']
    if batch_observation:common+=['--batch-observation']
    if validate_observation:
        assert batch_observation
        common+=['--validate-observation']
    if kind in ('rgbd','rgbd-validated','snapshot'):
        script='check_current_frame_final_cache.py' if kind in ('rgbd','rgbd-validated') else 'check_snapshot_roundtrip_cache.py'
        command=[str(PYTHON),str(ROOT/'tools'/script),*common,'--assembly-blocks','512','--manifold-blocks','1024']
        if kind in ('rgbd','rgbd-validated'):command+=['--persistent-soup','--direct-rgb']
        if kind=='rgbd-validated':command+=['--validate-native']
    else:
        command=[str(PYTHON),str(ROOT/'tools/profile_all_queries.py'),*common,'--cpus','7,19,20,25',
                 '--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1']
        if max_frames is not None:command+=['--max-frames',str(max_frames)]
        if kind=='validate':command+=['--validate-native','--validate-body-check','--start','0','--count','1410']
        if kind=='profile':
            command+=['--profile','--start','470','--count','12','--max-frames','482']
            command=['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys','profile','--trace=cuda,nvtx,opengl',
                     '--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi',
                     '--capture-range-end=stop','--force-overwrite=false','--output',str(output)+'-trace',*command]
    metadata={'name':name,'kind':kind,'core':core,'library':str(dll),'library_sha256':hashlib.sha256(dll.read_bytes()).hexdigest(),
              'command':command,'started_unix':time.time(),'gpu':GPU,'batch_observation':batch_observation,'observation_validation':validate_observation,'query_threads':query_threads}
    if batch_observation:metadata['observation_adapter_sha256']=hashlib.sha256((ROOT/'tools/batch_observation.py').read_bytes()).hexdigest()
    (ROOT/'logs'/(name+'-command.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    stop=threading.Event()
    def sample():
        with (ROOT/'logs'/(name+'-gpu.jsonl')).open('w') as stream:
            while not stop.is_set():
                result=subprocess.run(['nvidia-smi','-i',GPU,'--query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory,power.draw','--format=csv,noheader,nounits'],text=True,capture_output=True)
                stream.write(json.dumps({'unix':time.time(),'value':result.stdout.strip(),'returncode':result.returncode})+'\n');stream.flush();stop.wait(2)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(ROOT/'tmp'))
    try:
        with (ROOT/'logs'/(name+'.log')).open('w') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    finally:stop.set();monitor.join(timeout=5)
    metadata.update(returncode=result.returncode,finished_unix=time.time())
    path=output/'result.json'
    if path.exists():
        data=json.loads(path.read_text())
        metadata.update({k:data[k] for k in ('completed','quality_passed','passed','error','performance','interaction_stats','runtime_graph_stats','native_validation','coarse_bound_validation','gpu_body_statistics','batched_observation_stats','grip_feedback_validation','zero_mode_gate_stats') if k in data})
        if data.get('frames') and 'frame_ms' in data['frames'][0]:
            stages={};timing={}
            for frame in data['frames']:
                stages.setdefault(frame['stage'],[]).append(frame)
                for key,value in frame.get('timings_ms',{}).items():
                    if isinstance(value,(int,float)):timing.setdefault(key,[]).append(value)
            metadata['mean_timings_ms']={key:statistics.mean(v) for key,v in timing.items()}
            metadata['stages']={key:{'frames':len(v),'frame_ms':statistics.mean(x['frame_ms'] for x in v),
                'render_ms':statistics.mean(x['render_ms'] for x in v),'timings_ms':{k:statistics.mean(x['timings_ms'].get(k,0) for x in v) for k in timing}} for key,v in stages.items()}
    (ROOT/'results'/(name+'-summary.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('command','stages')},ensure_ascii=False),flush=True)
    return result.returncode


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--library',default='baseline');p.add_argument('--core',type=int,default=0)
    p.add_argument('--kind',choices=['benchmark','validate','profile','rgbd','rgbd-validated','snapshot'],default='benchmark');p.add_argument('--query-threads',type=int,choices=[32,64,128],default=64);p.add_argument('--max-frames',type=int);p.add_argument('--batch-observation',action='store_true');p.add_argument('--validate-observation',action='store_true');a=p.parse_args()
    raise SystemExit(run(a.name,a.library,a.core,a.kind,a.max_frames,a.batch_observation,a.validate_observation,a.query_threads))
