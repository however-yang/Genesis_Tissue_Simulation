"""One-GPU, resource-guarded parallel validation supervisor. Spawn, never fork CUDA."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parent
RELEASE = Path('/srv/disk2/hryang/genesis-exec/releases/0643e10991cf')
PYTHON = '/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python'
LIBRARY = Path('/srv/disk2/hryang/genesis-exec/builds/0643e10991cf-contact-groups31-sm80/lib/libLiverCudaSim.so')
GPU = 'GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def gpu_sample(uuid):
    fields = 'uuid,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu'
    q = subprocess.run(['nvidia-smi', '-i', uuid, '--query-gpu=' + fields,
                        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    row = [x.strip() for x in q.stdout.strip().split(',')]
    s = dict(zip(['uuid','total_mib','used_mib','gpu_percent','memory_percent','power_w','temp_c'],
                 [row[0]] + [float(x) for x in row[1:]]))
    q = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    s['processes'] = []
    for line in q.stdout.splitlines():
        values = [x.strip() for x in line.split(',')]
        if len(values) == 3 and values[0] == uuid:
            s['processes'].append({'pid': int(values[1]), 'memory_mib': float(values[2])})
    s.update(time=time.time(), host_available_gib=psutil.virtual_memory().available/2**30,
             host_cpu_percent=psutil.cpu_percent(), loadavg=list(os.getloadavg()))
    return s


def summarize(out, records, samples, released, finished):
    import statistics
    all_results, timing = [], {}
    worker_rates = []
    for record in records:
        wd = out / record['name']
        steps = 0
        for f in sorted(wd.glob('episode-*/result.json')):
            d = json.loads(f.read_text()); d['path'] = str(f.relative_to(ROOT)); all_results.append(d)
            steps += d['transitions'] + len(json.loads((f.parent / 'preparation.json').read_text())) if (f.parent / 'preparation.json').exists() else d['transitions']
            for row in json.loads((f.parent / 'steps.json').read_text()):
                for k,v in row['timings_ms'].items():
                    if isinstance(v,(int,float)): timing.setdefault(k,[]).append(v)
        wr = wd / 'worker-result.json'
        if wr.exists():
            d = json.loads(wr.read_text())
            worker_rates.append({'name':record['name'],'steps':steps,'steps_s':steps/max(d['finished']-d['started'],1e-9),
                                 'completed':d['completed']})
    duration = max(finished-released,1e-9)
    full_passed = sum(r['passed'] and r['full_script'] for r in all_results)
    expected = sum(r['episodes'] for r in records)
    complete = (len(all_results)==expected and all(r.get('exit_code')==0 for r in records)
                and len(worker_rates)==len(records) and all(r['completed'] for r in worker_rates))
    steady = [s for s in samples if released <= s['time'] <= finished]
    def stats(a):
        if not a:return None
        a=sorted(a);return {'mean':statistics.mean(a),'median':statistics.median(a),
                           'p95':a[min(len(a)-1,int(.95*len(a)))],'max':max(a)}
    result={'duration_s_including_reset_checks_and_output':duration,'workers':len(records),
        'complete':complete,'expected_episodes':expected,'recorded_episodes':len(all_results),
        'passed_full_episodes':full_passed,'failed_or_partial_episodes':len(all_results)-full_passed,
        'aggregate_steps_s':sum(x['steps'] for x in worker_rates)/duration,
        'valid_episodes_per_hour':full_passed*3600/duration,
        'worker_rates':worker_rates,'timings_ms':{k:stats(v) for k,v in timing.items()},
        'gpu_memory_mib':stats([s['used_mib'] for s in steady]),
        'gpu_utilization_percent':stats([s['gpu_percent'] for s in steady]),
        'gpu_memory_utilization_percent':stats([s['memory_percent'] for s in steady]),
        'host_cpu_percent':stats([s['host_cpu_percent'] for s in steady]),
        'summed_worker_rss_mib':stats([s.get('workers_rss_mib',0) for s in steady]),
        'results':[{'path':r['path'],'passed':r['passed'],'full_script':r['full_script'],
                    'quality_failures':r['quality_failures'],'scope_failures':r['scope_failures'],
                    'snips':r['snips']} for r in all_results]}
    dump(out/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--gpu',default=GPU)
    p.add_argument('--cpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--stop-after',default='hold')
    p.add_argument('--scenes',default='wall-traction')
    p.add_argument('--geometry',type=Path)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--max-gpu-mib',type=float,default=78000)
    p.add_argument('--attempt-capacity',action='store_true',help='User-authorized real initialization attempt despite predicted memory shortage; live resource guards remain active')
    p.add_argument('--per-worker-reserve-mib',type=float,default=2300)
    p.add_argument('--max-host-rss-gib',type=float,default=48)
    p.add_argument('--allow-mps-pid',type=int,action='append',default=[])
    a=p.parse_args()
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.name):raise ValueError('Unsafe run name')
    if not 1<=a.workers<=64 or not 1<=a.episodes<=100:raise ValueError('Invalid bounded worker/episode count')
    cpus=[int(c) for c in a.cpus.split(',')]
    if not set(cpus).issubset(os.sched_getaffinity(0)):raise ValueError('CPU affinity outside allowed set')
    scenes=a.scenes.split(',')
    if not set(scenes).issubset({'wall-traction','wall','flap'}):raise ValueError('Unsupported validated scene')
    lock=(ROOT/('gpu-'+a.gpu+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/'runs'/a.name;out.mkdir(parents=True,exist_ok=False)
    before=gpu_sample(a.gpu)
    dump(out/'preflight.json',before)
    allowed=set(a.allow_mps_pid)
    foreign=[x for x in before['processes'] if x['pid'] not in allowed]
    rejection=None
    if foreign or before['used_mib']>max(512,2048*len(allowed)) or before['gpu_percent']>5:
        rejection='Selected GPU is not idle'
    if before['host_available_gib']<24:rejection='Insufficient available host RAM'
    estimate=before['used_mib']+a.workers*a.per_worker_reserve_mib
    if not a.attempt_capacity and estimate>min(a.max_gpu_mib,before['total_mib']-8192):
        rejection='Predicted memory reserve exceeds GPU budget; no allocations attempted'
    if rejection:
        dump(out/'status.json',{'phase':'rejected_preflight','reason':rejection,'workers':a.workers,
            'predicted_reserve_mib':estimate,'per_worker_reserve_mib':a.per_worker_reserve_mib,
            'gpu_budget_mib':a.max_gpu_mib,'no_workers_launched':True})
        raise RuntimeError(rejection)
    sha=hashlib.sha256(LIBRARY.read_bytes()).hexdigest()
    records=[];processes=[];logs=[];samples=[];release_time=None;finish_time=None;error=None
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(RELEASE),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        XDG_CACHE_HOME=str(ROOT/'cache'),PYTHONUNBUFFERED='1')
    dump(out/'configuration.json',vars(a)|{'geometry':str(a.geometry) if a.geometry else None,
         'release':str(RELEASE),'library':str(LIBRARY),'library_sha256':sha,
         'test_source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')},
         'mps_pipe':env.get('CUDA_MPS_PIPE_DIRECTORY'),'mps_active_thread_percentage':env.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'),
         'quality_policy':'unchanged full_surface_checks and original validation thresholds',
         'scope':'scripted grasper+scissors end effectors, not articulated dual robot arms'})
    def monitor():
        s=gpu_sample(a.gpu);rss=0
        for proc in processes:
            try:rss+=psutil.Process(proc.pid).memory_info().rss
            except psutil.NoSuchProcess:pass
        s['workers_rss_mib']=rss/2**20;samples.append(s)
        with (out/'monitor.jsonl').open('a') as stream:stream.write(json.dumps(s)+'\n')
        owned={p.pid for p in processes}|allowed
        if env.get('CUDA_MPS_PIPE_DIRECTORY'):
            for item in s['processes']:
                try:
                    candidate=psutil.Process(item['pid'])
                    if (candidate.uids().real==os.getuid() and candidate.name()=='nvidia-cuda-mps-server'
                        and candidate.environ().get('CUDA_MPS_PIPE_DIRECTORY')==env['CUDA_MPS_PIPE_DIRECTORY']):
                        owned.add(candidate.pid)
                        allowed.add(candidate.pid)
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        foreign=[x for x in s['processes'] if x['pid'] not in owned]
        if foreign:raise RuntimeError('Another job appeared on selected GPU; stopping only our workers')
        if s['used_mib']>a.max_gpu_mib or rss>a.max_host_rss_gib*2**30 or s['host_available_gib']<16:
            raise RuntimeError('Resource ceiling reached')
        if s['temp_c']>83:raise RuntimeError('Temperature ceiling reached')
        return s
    try:
        for i in range(a.workers):
            name=f'worker-{i:03d}';wd=out/name;wd.mkdir()
            spec={'release':str(RELEASE),'library':str(LIBRARY),'library_sha256':sha,
         'test_source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')},'seed':i,
                  'scene':scenes[i%len(scenes)],'stop_after':a.stop_after,'episodes':a.episodes,
                  'cpus':[cpus[i%len(cpus)]], 'geometry':str(a.geometry.resolve()) if a.geometry else None,
                  'profile_ticks':[300,*range(495,511),520,600,900,1200] if a.profile else []}
            dump(wd/'spec.json',spec)
            log=(wd/'stdout.log').open('xb');logs.append(log)
            proc=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'worker.py'),'--spec',str(wd/'spec.json')],
                 cwd=RELEASE,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            processes.append(proc);records.append({'name':name,'pid':proc.pid,'episodes':a.episodes})
        deadline=time.monotonic()+180
        while not all((out/r['name']/'ready.json').exists() for r in records):
            if any(p.poll() is not None for p in processes):raise RuntimeError('Worker failed before barrier')
            if time.monotonic()>deadline:raise RuntimeError('Ready barrier timeout')
            monitor();time.sleep(1)
        release_time=time.time();(out/'GO').write_text(str(release_time))
        dump(out/'status.json',{'phase':'running','records':records,'released':release_time})
        last_print=0
        while any(p.poll() is None for p in processes):
            s=monitor()
            if time.time()-last_print>30:
                print(a.name,round(time.time()-release_time,1),'s',s['used_mib'],'MiB',s['gpu_percent'],'% GPU',flush=True);last_print=time.time()
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('A worker crashed; retaining all results')
            if time.time()-release_time>a.timeout:raise RuntimeError('Batch deadline exceeded')
            time.sleep(2)
        finish_time=time.time()
    except BaseException as exc:
        error=repr(exc);(out/'STOP').write_text(error)
        deadline=time.monotonic()+20
        while any(p.poll() is None for p in processes) and time.monotonic()<deadline:time.sleep(.25)
        for proc in processes:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
        for proc in processes:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    finally:
        for r,proc in zip(records,processes):r['exit_code']=proc.poll()
        for log in logs:log.close()
        finish_time=finish_time or time.time()
        dump(out/'status.json',{'phase':'failed' if error else 'complete','error':error,
             'records':records,'released':release_time,'finished':finish_time})
        if release_time:
            result=summarize(out,records,samples,release_time,finish_time)
            print(json.dumps({k:result[k] for k in ['complete','workers','aggregate_steps_s','passed_full_episodes','valid_episodes_per_hour']}),flush=True)
    if error:raise RuntimeError(error)


if __name__=='__main__':main()
