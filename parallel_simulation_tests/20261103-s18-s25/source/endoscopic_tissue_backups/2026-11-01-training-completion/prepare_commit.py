"""Collect small evidence and an explicit Git pathspec; never stage unrelated work."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
OUT = ROOT/'endoscopic_tissue/outputs/training-completion-20261101'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_evidence(source, target):
    if target.exists():
        if source.read_bytes() != target.read_bytes():
            raise RuntimeError('Refuse to replace evidence with different bytes: '+str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main():
    jobs = json.loads((P/'post-batch-jobs.json').read_text())
    expected_jobs = {'final-positive-hold', 'cpu-final', 'batch-metadata-audit', 'physical-split-index', 'evidence-summary'}
    if len(jobs) != len(expected_jobs) or {j['name'] for j in jobs} != expected_jobs or any(j['exit_code'] for j in jobs):
        raise RuntimeError('Finish and inspect post-batch validation first')
    correction = json.loads((P/'strict-id-offline-equivalence.json').read_text())
    if not correction['all_online_records_equal'] or correction['cpu_exit_code'] != 0:
        raise RuntimeError('Finish strict material-ID regression checks first')
    batch = json.loads((OUT/'final-screen/hundred-screen/results.json').read_text())
    if len(batch['episodes']) != 100:
        raise RuntimeError('Incomplete batch')
    evidence = P/'run-evidence'
    for source in OUT.glob('final-*/**/results.json'):
        relative = source.relative_to(OUT)
        copy_evidence(source, evidence/relative)
        copy_evidence(source.parent/'spec.json', evidence/relative.parent/'spec.json')
    for name in ('policy-pilot', 'frozen-evaluation', 'friction-pilot', 'vision-pilot'):
        source = OUT/name/'results.json'
        if source.exists():
            copy_evidence(source, evidence/'development'/name/'results.json')
            copy_evidence(source.parent/'spec.json', evidence/'development'/name/'spec.json')
    for name in ('timing-0', 'timing-1', 'timing-2', 'timing-3', 'timing-r20-free-0', 'timing-r20-free-1', 'r20-kidney'):
        for filename in ('report.json', 'manifest.json', 'support-cache-audit.json', 'release-overlap-audit.json'):
            source = OUT/name/filename
            if source.exists():
                copy_evidence(source, evidence/name/filename)
    artifact = P/'package/artifacts/release-r20-support-cache'
    for name in ('build-inputs.json', 'build-result.json', 'build.cmd', 'build.log'):
        copy_evidence(artifact/name, evidence/'r20-build'/name)
    selected = [ROOT/('endoscopic_tissue/training/'+name) for name in
                ('data.py', 'env.py', 'tasks.py', 'policies.py', 'experiment.py', 'vision.py', 'grasp.py',
                 'cutting.py', 'test_tasks.py', 'splits.py', 'test_splits.py', '.gitattributes')]
    selected += [ROOT/('endoscopic_tissue/docs/'+name) for name in
                 ('training-completion-design-2026-11-01.md', 'training-completion-results-2026-11-01.md',
                  'training-policy-guide-2026-11-01.md')]
    excluded_parts = {'artifacts', '__pycache__', 'test-temp'}
    excluded_names = {'selected-files.json', 'selected-paths.nul', 'commit-message.txt', 'batch-metadata-audit-partial.json'}
    for path in P.rglob('*'):
        relative = path.relative_to(P)
        if not path.is_file() or excluded_parts.intersection(relative.parts) or path.name in excluded_names:
            continue
        if any(part.startswith('split-test-') for part in relative.parts):
            continue
        if path.suffix in ('.py', '.json', '.md', '.jpg', '.png', '.cu', '.cuh', '.log', '.cmd') or path.name in ('.gitattributes', 'CMakeLists.txt'):
            selected.append(path)
    selected = sorted(set(selected))
    rows = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in selected]
    (P/'selected-files.json').write_text(json.dumps(rows, indent=2)+'\n')
    (P/'selected-paths.nul').write_bytes(b'\0'.join(r['path'].encode() for r in rows)+b'\0')
    print(json.dumps({'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
                      'pathspec': str(P/'selected-paths.nul'), 'git_mutated': False}, indent=2))


if __name__ == '__main__':
    main()
