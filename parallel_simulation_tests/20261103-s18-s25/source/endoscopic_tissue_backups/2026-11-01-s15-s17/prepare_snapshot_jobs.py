"""Schedule S15 acceptance serially, with smoke gates before expensive repeats."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101'


def main():
    queue_path = HERE/'gpu-queue.json'
    queue = json.loads(queue_path.read_text(encoding='utf8'))
    jobs = []
    smoke = OUT/'snapshot/env-smoke-r1/result.json'
    cache = smoke.parent/'cut-ready.lcssnap'
    manifest = {'schema': 's15_validation_plan_v1',
        'raw': {'path': str(OUT/'snapshot/settle-r1/result.json'), 'job': 'snapshot-settle-r1'},
        'smokes': {'wall-traction': {'path': str(smoke), 'job': 'snapshot-env-smoke-r1'}},
        'resources': {}, 'distributions': {}, 'cache': str(cache)}

    def add(name, script, args, requirement, result):
        jobs.append({'name': name, 'args': [str(HERE/'acceptance_job.py'), '--require',
            str(requirement), '--', str(HERE/script), *args]})
        return {'path': str(result), 'job': name}

    for label, args in [('flap', ['--scene', 'flap']),
                        ('vision', ['--vision'])]:
        name = 'snapshot-'+label+'-smoke-r1'
        manifest['smokes'][label] = add(name, 'validate_environment_snapshot.py',
            ['--name', name, *args], smoke, OUT/'snapshot'/name/'result.json')
    name = 'snapshot-resources-1000-r1'
    manifest['resources'] = add(name, 'validate_environment_snapshot.py',
        ['--name', name, '--resources', '1000'], smoke, OUT/'snapshot'/name/'result.json')

    # All four short pilots precede their 19 remaining independent pairs.
    for repeat in range(20):
        for fork in ('before_contact', 'first_cut', 'squeezing', 'regrasp', 'cache'):
            name = f'snapshot-{fork}-r{repeat:02}'
            requirement = smoke if repeat == 0 else OUT/'snapshot-distribution'/f'snapshot-{fork}-r00'/'result.json'
            args = ['--name', name, '--fork', fork, '--repeat', str(repeat)]
            if fork == 'cache':
                args += ['--cache', str(cache)]
            elif repeat == 0:
                args += ['--capture']
            evidence = add(name, 'validate_snapshot_distribution.py', args, requirement,
                           OUT/'snapshot-distribution'/name/'result.json')
            manifest['distributions'].setdefault(fork, []).append(evidence)

    existing = {j['name']: j for j in queue['jobs']}
    new_jobs = []
    for job in jobs:
        if job['name'] in existing and existing[job['name']] != job:
            raise ValueError('Refusing to silently rewrite queued evidence: '+job['name'])
        if job['name'] not in existing:
            new_jobs.append(job)
    pilots = [j for j in new_jobs if j['name'].endswith('-r00')]
    following_smoke = next(i for i, j in enumerate(queue['jobs']) if j['name'] == 'snapshot-env-smoke-r1')+1
    queue['jobs'][following_smoke:following_smoke] = pilots
    queue['jobs'].extend(j for j in new_jobs if j not in pilots)
    temporary = queue_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(queue, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    temporary.replace(queue_path)
    (HERE/'snapshot-validation-plan.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'snapshot_jobs': len(jobs), 'queue_jobs': len(queue['jobs']),
        'does_not_start_gpu_jobs': True, 'prerequisite': str(smoke)}))


if __name__ == '__main__':
    main()
