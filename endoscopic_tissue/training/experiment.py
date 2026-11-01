"""Versioned policy collection / frozen evaluation with isolated episode workers.

python -m endoscopic_tissue.training.experiment --spec SPEC --output NEW_DIR
Workers are sequential; a deadline can terminate only the child this supervisor owns.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from .actions import ActionBounds
from .collect import local_bounds
from .data import EpisodeReader, EpisodeWriter, atomic_json, save_arrays
from .env import EnvironmentConfig, NativeEnvironment, SceneConfig
from .policies import Policy, observation
from .provenance import json_hash, sha256
from .tasks import recompute_episode
from .physics import PhysicsOverrides


def group_key(case):
    return json_hash({'scene': case.get('scene', 'wall-traction'),
                      'preparation': case.get('preparation', 'cut_ready'),
                      'contact_model': case.get('contact_model'), 'seed': int(case['seed'])})


def scene_configuration(case):
    physics = case.get('physics')
    return SceneConfig(case.get('scene','wall-traction'),case.get('preparation','cut_ready'),
                       case.get('contact_model'),case.get('dt_s'),case.get('substeps'),case.get('spacing_scale',1.),
                       PhysicsOverrides(**physics) if physics is not None else None)


def validate_spec(spec):
    if spec.get('schema') != 'tissue_experiment_v1' or spec.get('purpose') not in ('collection', 'evaluation'):
        raise ValueError('Explicit experiment schema/purpose required')
    cases = spec['episodes']
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Nonempty, uniquely named cases required')
    for case in cases:
        if not isinstance(case['id'], str) or not case['id'] or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for x in case['id']):
            raise ValueError('Case ID must be a safe filename')
        if case.get('steps', 120) < 1 or case.get('reset_attempts', 1) not in (1,2,3):
            raise ValueError('Invalid horizon or reset attempts')
        scene_configuration(case)
    if spec['purpose'] == 'evaluation':
        digest = spec.get('frozen_sha256')
        if digest != json_hash({k:v for k,v in spec.items() if k != 'frozen_sha256'}):
            raise ValueError('Evaluation specification must be frozen before running')
        train_groups = set(spec.get('excluded_training_groups', []))
        if train_groups.intersection(group_key(c) for c in cases):
            raise ValueError('Evaluation reset lineage overlaps collection data')
    for key, default in (('episode_timeout_s', 900), ('min_free_disk_bytes', 1024**3)):
        value = spec.get(key, default)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Positive finite resource limits required')


def summarize(records):
    completed = [r for r in records if r.get('status') == 'recorded']
    task_records = [r for r in completed if r.get('task_evaluated')]
    return {'attempted_episodes': len(records), 'recorded_episodes': len(completed),
            'task_evaluated_episodes': len(task_records),
            'task_successes': sum(r.get('task_success', False) for r in task_records),
            'success_fraction_of_all_attempts': sum(r.get('task_success', False) for r in task_records)/len(records) if records else 0.,
            'failures_and_timeouts': [r['id'] for r in records if r.get('status') != 'recorded'],
            'recorded_transitions': sum(r.get('transitions', 0) for r in completed),
            'training_domain_approved': False}


@threadpool_limits.wrap(limits=1, user_api='blas')
def run_episode(spec, case, directory):
    scene = scene_configuration(case)
    bounds = ActionBounds(**case['bounds']) if 'bounds' in case else local_bounds(scene)
    config = EnvironmentConfig(spec['dll'], spec['dll_sha256'], bounds, scene,
                               cutting_enabled=bool(case.get('cutting_enabled', False)),
                               max_steps=int(case.get('steps', 120)))
    record = {'id': case['id'], 'seed': int(case['seed']), 'group': group_key(case),
              'status': 'initializing', 'training_domain_approved': False, 'config': asdict(config)}
    record['split'] = 'evaluation' if spec['purpose'] == 'evaluation' else ('validation' if int(record['group'][:8],16)%5==0 else 'train')
    start = time.perf_counter()
    with NativeEnvironment(config) as env, ExitStack() as resources:
        ready = False
        for attempt in range(case.get('reset_attempts', 1)):
            atomic_json(directory/'heartbeat.json', {'phase': 'reset', 'attempt': attempt, 'time_unix': time.time()})
            try:
                state, context = env.reset(case['seed'], goal=case.get('goal'))
                ready = True
                break
            except (RuntimeError, ValueError, OSError) as error:
                failure = env.last_reset_diagnostic or {'error': repr(error)}
                if env.last_reset_failure_arrays is not None:
                    path = directory/f'reset-failure-{attempt}.npz.zst'
                    failure['arrays_sha256'] = save_arrays(path, env.last_reset_failure_arrays)
                    failure['arrays_file'] = path.name
                atomic_json(directory/f'reset-failure-{attempt}.json', failure)
                record.update(status='reset_failure', error=repr(error), attempts=attempt+1)
        if ready:
            policy = Policy(case['policy'], env.previous_action, case['seed'])
            vision = None
            static = env.static_arrays()
            if case.get('vision') is not None:
                from .vision import SynchronizedVision
                vision = SynchronizedVision(env, case['vision'])
                resources.callback(vision.close)
                context['vision'] = vision.manifest
                static.update(vision.static_arrays)
                vision.capture(env, state)
            context.update(policy=policy.manifest, experiment_spec_sha256=json_hash(spec),
                           experiment_purpose=spec['purpose'], episode_group=record['group'])
            context['context_sha256'] = json_hash({k:v for k,v in context.items() if k != 'context_sha256'})
            timings, rewards = [], []
            last = None
            collect_start = time.perf_counter()
            with EpisodeWriter(directory/'episode', context, state, static, env.preparation) as writer:
                while not env.done:
                    if shutil.disk_usage(directory).free < spec.get('min_free_disk_bytes', 1024**3):
                        raise OSError('disk_budget_exceeded; partial episode retained')
                    obs = observation(env.state, env.previous_action, privileged=bool(case['policy'].get('privileged', False)))
                    last = env.step(policy.act(obs))
                    if vision is not None and last.after is not None:
                        try:
                            last.timings_ms['vision'] = vision.capture(env, last.after)
                        except Exception as error:
                            last = replace(last, after=None, terminated=True, truncated=False,
                                           reason='vision_observation_failure:'+repr(error), task=None,
                                           bootstrap_allowed=False,
                                           failure_nodes_native_cm=env.observed.nodes.copy())
                    writer.append(last)
                    timings.append(last.timings_ms)
                    if last.task is not None and last.task['reward_valid']:
                        rewards.append(last.task['reward'])
                    atomic_json(directory/'heartbeat.json', {'phase': 'step', 'step': env.step_id, 'time_unix': time.time()})
                    if last.terminated or last.truncated:
                        break
            record['collect_wall_s'] = time.perf_counter()-collect_start
            reader = EpisodeReader(directory/'episode')
            if case.get('goal') is not None:
                recompute_episode(reader)
            record.update(status='recorded', transitions=reader.manifest['normal_transitions'],
                          states=reader.manifest['states'], reason=reader.manifest['reason'],
                          reset_wall_ms=context['reset_wall_time_ms'], policy=policy.manifest,
                          task_evaluated=case.get('goal') is not None,
                          task_success=bool(last and last.task and last.task['success']),
                          task_failure=last.task['failure'] if last and last.task else None,
                          valid_reward_sum=sum(rewards) if rewards else None,
                          valid_reward_transitions=len(rewards),
                          offline_reward_match=case.get('goal') is not None,
                          transitions_per_wall_second=reader.manifest['normal_transitions']/record['collect_wall_s'],
                          timing_ms={k:{'mean':float(np.mean([t[k] for t in timings if k in t])),
                                        'p95':float(np.percentile([t[k] for t in timings if k in t],95)),
                                        'max':float(max(t[k] for t in timings if k in t))}
                                     for k in sorted({k for t in timings for k in t})})
    record['wall_s'] = time.perf_counter()-start
    return record


def supervise(spec, output):
    validate_spec(spec)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'spec.json', spec)
    records = []
    for case in spec['episodes']:
        directory = output/case['id']
        directory.mkdir()
        atomic_json(directory/'input.json', {'spec': spec, 'case': case})
        if shutil.disk_usage(output).free < spec.get('min_free_disk_bytes', 1024**3):
            record = {'id': case['id'], 'status': 'disk_budget_exceeded', 'group': group_key(case)}
        else:
            command = [sys.executable, '-X', 'utf8', '-m', 'endoscopic_tissue.training.experiment',
                       '--worker', str(directory/'input.json')]
            print('running', case['id'], flush=True)
            started = time.perf_counter()
            peak_rss = 0
            with (directory/'worker.log').open('w', encoding='utf8') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                atomic_json(directory/'process.json', {'pid': process.pid, 'started_unix': time.time(), 'command': command})
                try:
                    deadline = started+spec.get('episode_timeout_s',900)
                    while process.poll() is None:
                        try:
                            peak_rss = max(peak_rss,psutil.Process(process.pid).memory_info().rss)
                        except (psutil.NoSuchProcess,psutil.AccessDenied):
                            pass
                        remaining = deadline-time.perf_counter()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command,spec.get('episode_timeout_s',900))
                        try:
                            process.wait(timeout=min(1.,remaining))
                        except subprocess.TimeoutExpired:
                            pass
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                    atomic_json(directory/'timeout.json', {'pid': process.pid, 'owned_child_terminated': True,
                                                           'wall_s': time.perf_counter()-started})
            result_file = directory/'result.json'
            if (directory/'timeout.json').exists():
                record = {'id': case['id'], 'status': 'timeout', 'group': group_key(case)}
            elif result_file.exists():
                record = json.loads(result_file.read_text(encoding='utf8'))
            else:
                record = {'id': case['id'], 'status': 'worker_failure', 'exit_code': process.returncode,
                          'group': group_key(case)}
            record['worker_peak_rss_bytes'] = peak_rss
            record['supervised_wall_s'] = time.perf_counter()-started
        records.append(record)
        atomic_json(output/'results.json', {'schema': 'tissue_experiment_result_v1', 'spec_sha256': json_hash(spec),
                                          'episodes': records, 'summary': summarize(records)})
        print(case['id'], record['status'], record.get('reason', ''), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.worker:
        bundle = json.loads(args.worker.read_text(encoding='utf8'))
        try:
            result = run_episode(bundle['spec'], bundle['case'], args.worker.parent)
        except Exception as error:
            result = {'id': bundle['case']['id'], 'status': 'collection_failure', 'error': repr(error),
                      'group': group_key(bundle['case'])}
            import traceback
            traceback.print_exc()
        atomic_json(args.worker.parent/'result.json', result)
        return
    if args.spec is None or args.output is None:
        parser.error('--spec and --output are required')
    spec = json.loads(args.spec.read_text(encoding='utf8'))
    if args.freeze:
        if args.output.exists():
            parser.error('Frozen spec output must not exist')
        spec['purpose'] = 'evaluation'
        spec['frozen_sha256'] = json_hash({k:v for k,v in spec.items() if k != 'frozen_sha256'})
        validate_spec(spec)
        atomic_json(args.output, spec)
    else:
        supervise(spec, args.output)


if __name__ == '__main__':
    main()
