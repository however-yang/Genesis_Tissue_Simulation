"""Original/new training entry, identical full scripts, isolated sequential runs."""
import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation'))
sys.path.insert(0, str(ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes'))
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from replay_archive import ReplayArchiveWriter
from validate_strict import pack_frame
from endoscopic_tissue import native_backend
from endoscopic_tissue.scenario import SCISSORS_SCENES, stage_start, interpolate
from endoscopic_tissue.training import env as current
from endoscopic_tissue.training.actions import ActionBounds, command_to_action
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.grasp import GraspMonitor
from endoscopic_tissue.training.provenance import sha256

OUT = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/default-comparison'
DLL = ROOT/'endoscopic_tissue/native/bin-contact-groups31/LiverCudaSim.dll'
SHA = '4515f14aa15309162c2c54b6370792e128e63a7e8d3054c506da19e3aeb37d2f'


def foreign_python():
    own = {psutil.Process().pid, *(p.pid for p in psutil.Process().children(recursive=True))}
    return [{'pid': p.pid, 'name': p.info['name']} for p in psutil.process_iter(['name'])
            if 'python' in (p.info['name'] or '').lower() and p.pid not in own]


def wait_idle():
    while foreign := foreign_python():
        print('Waiting for other Python/GPU work:', foreign, flush=True)
        time.sleep(15)


def load_entry(variant):
    commit = 'b3d301cf4c85774d214669ecebbab98b56325419' if variant=='original' else '1d3e518'
    path = HERE/('baseline-env.py' if variant=='original' else 's17-checkpoint-env.py')
    if not path.exists():
        path.write_bytes(subprocess.check_output(['git', 'show',
            commit+':endoscopic_tissue/training/env.py'], cwd=ROOT))
    name = 'endoscopic_tissue.training._s17_'+variant+'_env'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stats(values):
    return {'mean': float(np.mean(values)), 'p95': float(np.percentile(values,95)),
            'max': float(np.max(values))} if values else None


@threadpool_limits.wrap(limits=1, user_api='blas')
def worker(name, variant, scene_name, capture):
    directory = OUT/name
    directory.mkdir(parents=True, exist_ok=False)
    module = load_entry(variant)
    scene = SCISSORS_SCENES[scene_name]
    trajectory, stages, timings = [], [], []
    active_stage = {'name': scene.stages[0].name}
    stream = ReplayArchiveWriter(directory/'replay.rdx') if capture else None
    class Recorder(native_backend.NativeTissue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.monitor = GraspMonitor(self.settings, self.grid.inside)
            if stream:
                pickle.dump((self.settings, scene_name, name), stream, protocol=5)
                pack_frame(stream, 0, 'initial', self.surface(), self.observe())
        def step(self, command):
            obs = super().step(command)
            if stream:
                grasp = self.monitor.assess_native(obs)
                lost = (self.grid.neighbors[:,1::2] >= 0) & (obs.neighbors[:,1::2] < 0)
                trajectory.append({'step': self.tick, 'stage': active_stage['name'],
                    'pad_slip_mm': grasp['witness_tangential_slip_m']*1000,
                    'grip_points': obs.grip_points, 'bite_nodes': len(grasp['witness_ids']),
                    'held_bite_nodes': round(len(grasp['witness_ids'])*grasp['witness_retained_fraction']),
                    'cut_edges': int(np.count_nonzero(lost)), 'jaw_load_n': list(obs.jaw_load_n),
                    'grasp': grasp})
                pack_frame(stream,self.tick,active_stage['name'],self.surface(),obs)
            return obs
    if capture:
        module.NativeTissue = Recorder
    scene_config = module.SceneConfig(scene_name, 'settled')
    center = command_to_action(scene.stages[0].command)
    lower, upper = center-.2, center+.2
    lower[[3,10]], upper[[3,10]] = 0., 1.
    bounds = ActionBounds(tuple(lower), tuple(upper), (10.,)*11)
    cfg = module.EnvironmentConfig(str(DLL),SHA,bounds,scene_config,cutting_enabled=True,
                                    max_steps=sum(s.steps for s in scene.stages),diagnostic_continue=True)
    started = time.perf_counter()
    rows, errors = [], []
    with module.NativeEnvironment(cfg) as env:
        state, context = env.reset(0)
        atomic_json(directory/'context.json', context)
        stages.append({'name': scene.stages[0].name, 'end': {'step': env.native.tick}})
        requested = scene.stages[0].command
        for stage in scene.stages[1:]:
            active_stage['name'] = stage.name
            start = stage_start(stage.name,requested,env.observed)
            for i in range(stage.steps):
                requested = interpolate(start,stage.command,(i+1)/(stage.steps*stage.ramp))
                r = env.replay_command(requested)
                row = {'native_tick':env.native.tick,'stage':stage.name,'timings_ms':r.timings_ms,
                       'metadata': r.after.metadata if r.after is not None else None, 'reason':r.reason}
                rows.append(row)
                if r.after is None:
                    errors.append(row)
                    break
            stages.append({'name': stage.name, 'end': {'step':env.native.tick}})
            print(name,stage.name,env.native.tick,flush=True)
            if errors:
                break
        atomic_json(directory/'preparation.json', env.preparation)
        result = {'name':name,'variant':variant,'scene':scene_name,'capture':capture,
                  'baseline_commit':'b3d301cf4c85774d214669ecebbab98b56325419',
                  'entry_source_sha256':sha256(Path(module.__file__)),
                  'candidate_entry_commit':'1d3e518',
                  'default_dll_sha256':sha256(DLL),'context_hash':context['context_sha256'],
                  'expected_total_steps':sum(s.steps for s in scene.stages),'total_steps':env.native.tick,
                  'episode_transitions':len(rows),'complete':env.native.tick==sum(s.steps for s in scene.stages),
                  'quality_failure_steps':sum(r['metadata'] is None or not r['metadata']['quality']['checked_metrics_passed'] for r in rows),
                  'scope_failure_steps':sum(r['metadata'] is None or not r['metadata']['quality']['within_declared_local_scope'] for r in rows),
                  'wall_s':time.perf_counter()-started,
                  'timings_ms':{k:stats([r['timings_ms'][k] for r in rows if k in r['timings_ms']])
                                  for k in ('native_step','quality_and_export','snapshot','total')}}
    if stream:
        stream.close()
        with (directory/'trajectory.jsonl').open('w',encoding='utf8') as log:
            for row in trajectory:
                log.write(json.dumps(row)+'\n')
        atomic_json(directory/'report.json',{'metadata':{'case':name},'stages':stages})
    atomic_json(directory/'steps.json', rows)
    atomic_json(directory/'result.json',result)
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker')
    p.add_argument('--variant',choices=['original','new'])
    p.add_argument('--scene',default='wall-traction')
    p.add_argument('--capture',action='store_true')
    args=p.parse_args()
    if args.worker:
        return worker(args.worker,args.variant,args.scene,args.capture)
    OUT.mkdir(parents=True,exist_ok=False)
    jobs=[('timing-a1','original','wall-traction',False),('timing-b1','new','wall-traction',False),
          ('timing-b2','new','wall-traction',False),('timing-a2','original','wall-traction',False),
          ('video-wall-a','original','wall-traction',True),('video-wall-b','new','wall-traction',True),
          ('video-wall-postcut-a','original','wall',True),('video-wall-postcut-b','new','wall',True)]
    records=[]
    for name,variant,scene,capture in jobs:
        wait_idle()
        command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--worker',name,'--variant',variant,'--scene',scene]
        if capture:command.append('--capture')
        print('Starting',name,flush=True)
        with (OUT/(name+'.log')).open('w',encoding='utf8') as log:
            job=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            interference=[]
            while job.poll() is None:
                foreign=foreign_python()
                if foreign:interference.append({'unix_s':time.time(),'processes':foreign})
                time.sleep(5)
        records.append({'name':name,'returncode':job.returncode,'foreign_process_samples':interference})
        atomic_json(OUT/'jobs.json', records)
        print('Finished',name,'exit',job.returncode,'interference',len(interference),flush=True)


if __name__=='__main__':
    main()
