"""Fail-closed T1-T8 evidence aggregation, bound to this DLL and Python codec.

This report only qualifies native snapshot use. It never promotes a default
DLL, adopts a failure-law threshold or substitutes for full interaction review.
"""
import argparse
import hashlib
import json
from pathlib import Path

from validation_support import ROOT
from validate_snapshot_distribution import summarize
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.training.snapshot import EnvSnapshot, source_binding


def aggregate(plan, jobs, dll_sha, python_binding):
    evidence, failures, data = [], [], {}
    by_name = {j['name']: j for j in jobs}

    def read(label, spec, metadata=True):
        path = Path(spec['path'])
        try:
            raw = path.read_bytes()
            item = json.loads(raw)
        except (OSError, ValueError) as error:
            failures.append(label+': missing/unreadable result: '+str(error))
            return None
        evidence.append({'test': label, 'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        job = by_name.get(spec['job'])
        if job is None or job.get('returncode') != 0:
            failures.append(label+': serial worker did not finish successfully')
        elif job.get('foreign_process_samples'):
            failures.append(label+': concurrent Python task contaminated this run')
        if item.get('passed') is not True:
            failures.append(label+': result did not pass')
        binding = item.get('metadata', item.get('snapshot_metadata', {})) if metadata else item
        if binding.get('dll_sha256') != dll_sha:
            failures.append(label+': DLL SHA mismatch')
        if metadata and binding.get('python_source_sha256') != python_binding:
            failures.append(label+': Python source binding mismatch')
        data[label] = item
        return item

    raw = read('raw', plan['raw'], metadata=False)
    if raw and not all(raw.get(k) is True for k in
            ('T1_saved_bytes_equal', 'T2_observation_exact', 'T5_three_branches_exact',
             'corruption_rejected_without_mutation')):
        failures.append('raw: missing exact state/corruption gates')
    if raw and not 0 <= raw.get('T3_max_node_difference_mm', float('inf')) < .01:
        failures.append('raw: single-step error exceeds the 0.01 mm gate')
    for scene in ('wall-traction', 'flap', 'vision'):
        item = read('smoke-'+scene, plan['smokes'][scene])
        if item is None:
            continue
        if not all(item.get(k) is True for k in ('T1_native_exact', 'T2_observation_exact',
                'T5_distinct_branches_no_pollution', 'snapshot_mechanism_passed')):
            failures.append(scene+': incomplete T1/T2/T5 mechanism evidence')
        if not 0 <= item.get('T3_max_node_difference_mm', float('inf')) < .01:
            failures.append(scene+': T3 single-step error exceeds 0.01 mm')
        t6 = item.get('T6', {})
        if t6.get('restored_topology_exact') is not True:
            failures.append(scene+': topology rollback was not exact')
        for arm in ('first_cut', 'repeated_cut'):
            continuation = t6.get(arm, {})
            q = continuation.get('quality', {})
            if (continuation.get('outcome', {}).get('new_material_edges', 0) <= 0 or
                    any(q.get(k) != 0 for k in ('quality_failure_steps', 'scope_failure_steps', 'healed_edges'))):
                failures.append(scene+': T6 '+arm+' did not cut real material cleanly')
        if item.get('performance_target', {}).get('restore_below_preparation_11500ms') is not True:
            failures.append(scene+': restore does not meet the mandatory <11.5 s limit')
        if scene == 'vision':
            if (item.get('vision') is not True or 'vision_rerender_ms' not in item or
                    item.get('metadata', {}).get('vision_required') is not True):
                failures.append('vision: no fresh synchronized render evidence')
        elif item.get('new_native_session_exact') is not True:
            failures.append(scene+': new native session restore was not exact')

    resource = read('T8', plan['resources'])
    if resource and (resource.get('count') != 1000 or resource.get('native_inventory_exact') is not True or
            resource.get('performance_target', {}).get('restore_below_preparation_11500ms') is not True):
        failures.append('T8: incomplete 1000-cycle resource/performance check')

    distributions = {}
    for fork in ('before_contact', 'first_cut', 'squeezing', 'regrasp', 'cache'):
        specs = plan['distributions'].get(fork, [])
        rows = [read(fork+f'-{i}', spec) for i, spec in enumerate(specs)]
        if len(rows) != 20 or any(r is None for r in rows):
            failures.append(fork+': requires 20 recorded independent pairs')
            continue
        if ({r.get('repeat') for r in rows} != set(range(20)) or
                any(r.get('fork') != fork or r.get('T1_native_exact') is not True or
                    (fork != 'cache' and r.get('T2_observation_exact') is not True) for r in rows)):
            failures.append(fork+': repeat IDs or exact fork checks are incomplete')
        try:
            distributions[fork] = summarize([s['path'] for s in specs])
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            failures.append(fork+': '+str(error))
            continue
        if not distributions[fork]['passed']:
            failures.append(fork+': distribution or full quality acceptance failed')
        if fork == 'cache':
            cache_shas = {r.get('cache_sha256') for r in rows}
            if len(cache_shas) != 1 or None in cache_shas:
                failures.append('T7: all restored arms must use the same immutable cached state')
            try:
                cache = EnvSnapshot.load(plan['cache'], dll_sha256=dll_sha,
                    physics_config_sha256=rows[0]['snapshot_metadata']['physics_config_sha256'])
                if cache.sha256 not in cache_shas or cache.metadata['python_source_sha256'] != python_binding:
                    failures.append('T7: cached artifact does not match the tested snapshot/code')
            except (OSError, ValueError, KeyError) as error:
                failures.append('T7: invalid common cache: '+str(error))
    return {'schema': 's15_native_acceptance_v1', 'dll_sha256': dll_sha,
        'python_source_sha256': python_binding, 'passed': not failures,
        'failures': failures, 'distributions': distributions, 'evidence': evidence,
        'resource_targets': None if resource is None else resource.get('performance_target'),
        'default_replacement_authorized': False,
        'scope': 'Snapshot T1-T8 only; native default-off timing and complete interaction videos have separate gates.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, default=Path(__file__).with_name('snapshot-validation-plan.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf8'))
    if plan.get('schema') != 's15_validation_plan_v1':
        raise ValueError('Unknown acceptance plan')
    jobfile = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/gpu-queue/results.json'
    jobs = json.loads(jobfile.read_text(encoding='utf8')) if jobfile.exists() else []
    dll = ROOT/'endoscopic_tissue/native/bin-s15-training-snapshot1/LiverCudaSim.dll'
    result = aggregate(plan, jobs, sha256(dll), source_binding())
    atomic_json(args.output, result)
    print(json.dumps({'passed': result['passed'], 'failure_count': len(result['failures'])}))
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
