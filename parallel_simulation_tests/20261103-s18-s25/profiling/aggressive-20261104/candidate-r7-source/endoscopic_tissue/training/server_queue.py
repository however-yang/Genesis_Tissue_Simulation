"""S23 file-backed server queue, using the existing SSH account for submission.

Submit/status are CPU-only. run-next requires an idle explicitly selected GPU.
Code and binary hashes are checked again before execution. No service is installed.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

from .data import atomic_json
from .experiment import validate_spec
from .provenance import ROOT, sha256, pinned_dll


def source_inventory():
    package=ROOT/'endoscopic_tissue'
    paths=list(package.glob('*.py'))+list((package/'training').glob('*.py'))
    paths += [p for p in (package/'assets').rglob('*') if p.is_file()]
    paths += list((ROOT/'genesis').rglob('*.py'))
    paths += list((ROOT/'genesis/ext/pyrender/shaders').glob('*'))
    return {str(p):sha256(p) for p in sorted(set(paths)) if p.is_file()}



def submit(queue, spec):
    validate_spec(spec)
    dll = pinned_dll(spec['dll'], spec['dll_sha256'])
    directory = Path(queue).resolve()/('job-'+uuid4().hex)
    directory.mkdir(parents=True,exist_ok=False)
    atomic_json(directory/'spec.json',spec)
    request = dict(schema='server_job_v1',created_unix=time.time(),source_root=str(ROOT),
                   interpreter=sys.executable,source_sha256=source_inventory(),
                   spec_sha256=sha256(directory/'spec.json'),dll=str(dll),dll_sha256=sha256(dll),
                   candidate_physical_acceptance='not_inferred_from_submission')
    atomic_json(directory/'request.json',request)
    atomic_json(directory/'status.json',dict(phase='queued',job_id=directory.name))
    return directory


def verify_request(directory):
    directory=Path(directory).resolve()
    request=json.loads((directory/'request.json').read_text())
    if request['source_root'] != str(ROOT) or request['interpreter'] != sys.executable:
        raise ValueError('Run this request with the same source checkout and interpreter')
    if source_inventory() != request['source_sha256'] or sha256(directory/'spec.json') != request['spec_sha256']:
        raise ValueError('Queued source/spec changed; submit a new job instead of overwriting evidence')
    pinned_dll(request['dll'],request['dll_sha256'])
    spec=json.loads((directory/'spec.json').read_text());validate_spec(spec)
    return request,spec


def run_next(queue, gpu):
    import fcntl
    from .acceptance_queue import Runner, GPUReservation
    queue=Path(queue).resolve();queue.mkdir(parents=True,exist_ok=True)
    with (queue/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        candidates=[]
        for path in queue.glob('job-*'):
            state=json.loads((path/'status.json').read_text())
            if state['phase']=='queued': candidates.append(path)
        if not candidates:return None
        directory=min(candidates,key=lambda p:json.loads((p/'request.json').read_text())['created_unix'])
        request,spec=verify_request(directory)
        # Busy admission leaves the job queued; never touch somebody else's job.
        guard=GPUReservation(gpu,directory)
        try:
            atomic_json(directory/'status.json',dict(phase='running',gpu=gpu,started_unix=time.time()))
            plan={'inputs_sha256':{**request['source_sha256'],request['dll']:request['dll_sha256'],
                                  str(directory/'spec.json'):request['spec_sha256']},
                  'env':{'CUDA_VISIBLE_DEVICES':gpu,'PYTHONPATH':str(ROOT),'PYTHONNOUSERSITE':'1',
                         'PYTHONDONTWRITEBYTECODE':'1','XDG_CACHE_HOME':str(directory/'cache')},
                  'jobs':[{'id':'experiment','args':[sys.executable,'-B','-m','endoscopic_tissue.training.experiment',
                           '--spec',str(directory/'spec.json'),'--output','{attempt}/output'],
                           'cwd':str(ROOT),'max_wall':len(spec['episodes'])*spec.get('episode_timeout_s',900)+120,
                           'max_idle':spec.get('episode_timeout_s',900)+30,
                           'progress_globs':['{attempt}/output/**/heartbeat.json','{attempt}/output/**/worker.log'],
                           'results':[{'path':'{attempt}/output/results.json',
                             'require':{'schema':'tissue_experiment_result_v1','summary.attempted_episodes':len(spec['episodes'])}}]}]}
            atomic_json(directory/'plan.json',plan)
            code=Runner(directory/'execution',plan,monitor=guard).run()
            result_path=directory/'execution/experiment/attempt-000/output/results.json'
            result=json.loads(result_path.read_text())
            atomic_json(directory/'status.json',dict(phase='execution_complete',exit_code=code,
                results=str(result_path),results_sha256=sha256(result_path),summary=result['summary'],
                candidate_acceptance='not_certified',finished_unix=time.time()))
        except BaseException as error:
            atomic_json(directory/'status.json',dict(phase='interrupted',error=repr(error),finished_unix=time.time()))
            raise
        finally:guard.close()
        return directory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('submit','status','run-next'))
    p.add_argument('--queue',type=Path,required=True);p.add_argument('--spec',type=Path);p.add_argument('--gpu')
    a=p.parse_args()
    if a.command=='submit':
        if a.spec is None:p.error('submit needs --spec')
        print(submit(a.queue,json.loads(a.spec.read_text(encoding='utf8'))))
    elif a.command=='run-next':
        if not a.gpu:p.error('run-next needs an explicit --gpu UUID')
        print(run_next(a.queue,a.gpu))
    else:
        print(json.dumps({d.name:json.loads((d/'status.json').read_text()) for d in a.queue.glob('job-*')},indent=2))


if __name__=='__main__':main()
