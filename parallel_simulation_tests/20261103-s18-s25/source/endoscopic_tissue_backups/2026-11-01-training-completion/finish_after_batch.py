"""Finish frozen validation serially, without changing code during native collection."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
OUT = ROOT/'endoscopic_tissue/outputs/training-completion-20261101'


def main():
    completed = P/'final-screen/jobs.json'
    deadline = time.monotonic()+7200
    print('WAIT for final-screen jobs.json; frozen training source stays unchanged', flush=True)
    while not completed.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Frozen batch did not finish; no adapter changes made')
        time.sleep(2)
    batch_jobs = json.loads(completed.read_text())
    if len(batch_jobs) != 1 or batch_jobs[0]['exit_code'] != 0:
        raise RuntimeError('Batch process failed; no adapter changes made')
    result = json.loads((OUT/'final-screen/hundred-screen/results.json').read_text())
    if len(result['episodes']) != 100:
        raise RuntimeError('Expected all 100 attempts before advancing')
    source = ROOT/'endoscopic_tissue/training'
    hashes = json.loads((P/'final-screen/source-hashes.json').read_text())
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
    if current != hashes:
        raise RuntimeError('Source changed during frozen validation; preserve and investigate')
    if (source/'splits.py').exists() or (source/'test_splits.py').exists():
        raise RuntimeError('Refuse to overwrite an existing split module')
    shutil.copyfile(P/'splits_candidate.py', source/'splits.py')
    test = (P/'test_splits_candidate.py').read_text()
    test = test.replace('import importlib.util\n', '')
    old = "_spec = importlib.util.spec_from_file_location('splits_candidate', Path(__file__).with_name('splits_candidate.py'))\nsplits = importlib.util.module_from_spec(_spec)\n_spec.loader.exec_module(splits)"
    if old not in test:
        raise RuntimeError('Unexpected split test source')
    test = test.replace(old, 'from . import splits')
    (source/'test_splits.py').write_text(test, encoding='utf8', newline='\n')
    evidence = P/'final-positive'
    snapshot = evidence/'adapter-source'
    snapshot.mkdir(parents=True, exist_ok=False)
    final_hashes = {}
    for path in source.glob('*.py'):
        final_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, snapshot/path.name)
    (evidence/'source-hashes.json').write_text(json.dumps(final_hashes, indent=2)+'\n')
    jobs = []

    def run(name, arguments, timeout=1200):
        command = [sys.executable, '-X', 'utf8', *map(str, arguments)]
        print('START', name, flush=True)
        started = time.monotonic()
        with (P/(name+'.log')).open('w', encoding='utf8') as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        jobs.append({'name': name, 'command': command, 'exit_code': proc.returncode,
                     'wall_s': time.monotonic()-started})
        (P/'post-batch-jobs.json').write_text(json.dumps(jobs, indent=2)+'\n')
        if proc.returncode:
            raise RuntimeError('Post-batch check failed: '+name)
        print('END', name, flush=True)

    run('final-positive-hold', ['-m', 'endoscopic_tissue.training.experiment', '--spec',
        P/'positive-hold-control.json', '--output', OUT/'final-positive/positive-hold-control'])
    run('cpu-final', [P/'run_cpu_checks.py', 'endoscopic_tissue.test_core',
                     'endoscopic_tissue.training.test_contract', 'endoscopic_tissue.training.test_tasks',
                     'endoscopic_tissue.training.test_splits'])
    run('batch-metadata-audit', [P/'audit_training_batch.py', OUT/'final-screen/hundred-screen',
                               '--output', P/'batch-metadata-audit.json'])
    paths = sorted((OUT/'final-v1').glob('*/results.json'))
    paths += [OUT/'final-screen/hundred-screen/results.json', OUT/'final-positive/positive-hold-control/results.json']
    run('physical-split-index', ['-m', 'endoscopic_tissue.training.splits', *paths,
                               '--evaluation-scope', 'repeatability', '--output', P/'physical-split-index.json'])
    run('evidence-summary', [P/'summarize_evidence.py'])
    print('DONE: inspect positive task status, all batch attempts, CPU log and evidence before committing', flush=True)


if __name__ == '__main__':
    main()
