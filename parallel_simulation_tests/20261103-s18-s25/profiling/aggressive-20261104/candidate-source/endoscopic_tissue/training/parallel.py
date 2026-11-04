"""Experimental asynchronous native environment pool, one CUDA session per process.

Keep the physics and quality checks in the existing NativeEnvironment. Full states
stay in the worker/EpisodeWriter; policy observations cross a bounded local pipe.
This module does not start MPS or install dependencies.
"""
import atexit
import fcntl
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys
import time
import traceback
import psutil

from .resources import gpu_sample


def _serve(connection, config, release, directory, gpu_uuid, cpus):
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid, PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', XDG_CACHE_HOME=str(directory/'cache'))
    if cpus:os.sched_setaffinity(0,cpus)
    sys.path.insert(0,str(release))
    log=(directory/'worker.log').open('a',buffering=1)
    os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    environment=writer=session=None
    try:
        from endoscopic_tissue.training.env import EnvironmentConfig, SceneConfig, NativeEnvironment
        from endoscopic_tissue.training.actions import ActionBounds
        from endoscopic_tissue.training.physics import PhysicsOverrides
        from endoscopic_tissue.training.geometry import WallGeometry
        from endoscopic_tissue.training.policies import observation
        from endoscopic_tissue.training.data import EpisodeWriter
        from threadpoolctl import threadpool_limits
        from .pool_session import WorkerSession, environment_config
        session_keys = {'record','vision','observation_contract','visual_tracking','privileged_fields',
                        'max_snapshot_bytes','max_snapshot_capture_bytes','max_response_bytes'}
        options={k:v for k,v in config.items() if k not in session_keys | {'scene','bounds'}}
        if options.get('diagnostic_continue'):raise ValueError('Policy pool forbids diagnostic_continue')
        if options.get('full_surface_checks',True) is not True:raise ValueError('Full checks required')
        ec=environment_config(dict(options,scene=config['scene'],bounds=config['bounds']))
        environment=NativeEnvironment(ec)
        session=WorkerSession(environment,directory,**{k:v for k,v in config.items() if k in session_keys})
        with threadpool_limits(limits=1):
            connection.send({'ready':True,'pid':os.getpid(),'gpu_uuid':gpu_uuid})
            while True:
                try:message=connection.recv()
                except EOFError:break
                request=message['request'];command=message['command'];started=time.perf_counter()
                try:
                    if command=='close':
                        connection.send({'request':request,'ok':True,'closed':True});break
                    response=session.execute(command,message)
                    response.update(request=request,ok=True,worker_elapsed_ms=(time.perf_counter()-started)*1000)
                    connection.send(response)
                except Exception as exc:
                    requires_reset = command in ('step','reset') or environment.native is None
                    if requires_reset:
                        environment.done=True
                        if session.writer is not None:
                            try:session.writer.close('worker_exception:'+repr(exc))
                            finally:session.writer=None
                    connection.send({'request':request,'ok':False,'error':repr(exc),'traceback':traceback.format_exc(),'requires_reset':requires_reset})
    except Exception:
        try:connection.send({'ready':False,'error':traceback.format_exc()})
        except (BrokenPipeError,EOFError):pass
        raise
    finally:
        if session is not None:session.close()
        if environment is not None:environment.close()
        connection.close();log.close()


class ParallelNativePool:
    """One outstanding request per environment; caller consumes whichever finishes first.

    Requires idle, explicitly selected GPU UUIDs. Each GPU is locked against our
    benchmark. Use the guarded benchmark for capacity/performance acceptance;
    this pool is a small integration pilot, not a certified RL training domain.
    """
    def __init__(self, configs, *, release, output, gpu_uuids, cpus, startup_timeout=120):
        if not configs or not gpu_uuids or not cpus or len(configs)>16:raise ValueError('Pilot pool supports 1..16 workers with explicit GPUs and CPUs')
        self.processes=[];self.connections=[];self.pending={};self.locks=[];self.next_request=0;self.closed=False
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.release=Path(release).resolve()
        if self.release != Path(__file__).resolve().parents[2]:raise ValueError("Import the pool from its requested source checkout")
        sys.path.insert(0,str(self.release))
        self.gpus=set(gpu_uuids);self.last_guard=0
        self.mps_pipe=os.environ.get('CUDA_MPS_PIPE_DIRECTORY')
        if self.mps_pipe and not Path(self.mps_pipe).resolve().is_relative_to(Path(__file__).parent/'mps'):
            raise ValueError('Pilot pool accepts only this test directory private MPS namespace')
        context=mp.get_context('spawn')
        try:
            for gpu in set(gpu_uuids):
                lock=(Path(__file__).parent/('gpu-'+gpu+'.lock')).open('a')
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.locks.append(lock)
                sample=gpu_sample(gpu)
                assigned=sum(gpu_uuids[i%len(gpu_uuids)]==gpu for i in range(len(configs)))
                if sample['used_mib']+2300*assigned>min(60000,sample['total_mib']-8192) or sample['host_available_gib']<24:
                    raise RuntimeError('Pool startup resource reserve exceeded')
                if any(not self._own_mps_server(p['pid']) for p in sample['processes']) or sample['used_mib']>512 or sample['gpu_percent']>5:
                    raise RuntimeError('GPU is not idle: '+gpu)
                (self.output/(gpu+'-preflight.json')).write_text(json.dumps(sample,indent=2))
            for index,config in enumerate(configs):
                directory=self.output/f'worker-{index:03d}';directory.mkdir()
                (directory/'configuration.json').write_text(json.dumps(config,indent=2))
                parent,child=context.Pipe(duplex=True)
                proc=context.Process(target=_serve,args=(child,config,self.release,directory,
                    gpu_uuids[index%len(gpu_uuids)],[cpus[index%len(cpus)]]))
                proc.start();child.close();self.processes.append(proc);self.connections.append(parent)
            deadline=time.monotonic()+startup_timeout
            for connection in self.connections:
                if not connection.poll(max(0,deadline-time.monotonic())):raise TimeoutError('Worker startup timeout')
                result=connection.recv()
                if not result.get('ready'):raise RuntimeError(str(result))
            atexit.register(self.close)
        except BaseException:
            self.close();raise

    def _own_mps_server(self,pid):
        if not self.mps_pipe:return False
        try:
            process=psutil.Process(pid)
            return (process.uids().real==os.getuid() and process.name()=='nvidia-cuda-mps-server'
                    and process.environ().get('CUDA_MPS_PIPE_DIRECTORY')==self.mps_pipe)
        except (psutil.NoSuchProcess,psutil.AccessDenied):return False

    def _guard(self):
        if time.monotonic()-self.last_guard<2:return
        self.last_guard=time.monotonic()
        owned={p.pid for p in self.processes}
        for gpu in self.gpus:
            sample=gpu_sample(gpu)
            if (any(p['pid'] not in owned and not self._own_mps_server(p['pid']) for p in sample['processes']) or sample['used_mib']>60000
                    or sample['host_available_gib']<16 or sample['temp_c']>83):
                self.close();raise RuntimeError('Resource guard stopped only this pool')

    def send(self, worker, command, **payload):
        if self.closed:raise RuntimeError('Pool closed')
        self._guard()
        if command in ('snapshot','restore') and any(p[2] in ('snapshot','restore') for p in self.pending.values()):
            raise RuntimeError('Consume the pending snapshot/restore response before another host-memory operation')
        if not 0<=worker<len(self.connections):raise IndexError(worker)
        if worker in self.pending:raise RuntimeError('Consume the outstanding response before sending another action')
        self.next_request+=1;request=self.next_request
        self.connections[worker].send(dict(payload,request=request,command=command))
        self.pending[worker]=(request,time.perf_counter(),command);return request

    def recv_ready(self, timeout=120):
        if not self.pending:return []
        self._guard()
        deadline=time.monotonic()+timeout
        ready=[]
        while not ready:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('No worker response before timeout')
            ready=wait([self.connections[i] for i in self.pending],min(2,remaining))
            self._guard()
        results=[]
        for connection in ready:
            worker=self.connections.index(connection);request,started,command=self.pending.pop(worker)
            message=connection.recv()
            if message.get('request')!=request:raise RuntimeError('Request/response mismatch')
            message.update(worker_id=worker,roundtrip_ms=(time.perf_counter()-started)*1000)
            results.append(message)
        return results

    def close(self):
        if self.closed:return
        self.closed=True
        # Closing the pipe gives idle workers EOF; busy workers finish their current command.
        for connection in self.connections:connection.close()
        deadline=time.monotonic()+15
        for proc in self.processes:proc.join(max(0,deadline-time.monotonic()))
        for proc in self.processes:
            if proc.is_alive():proc.terminate();proc.join(5)
            if proc.is_alive():proc.kill();proc.join()
        for lock in self.locks:lock.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
