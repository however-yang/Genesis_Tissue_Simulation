"""Serial section diagnostics. Graph-disabled results are NOT runtime benchmarks."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
DATA = ROOT / 'endoscopic_tissue/outputs/joint-gated-20261031'
PROFILER = ROOT / 'endoscopic_tissue_backups/2026-10-28-speed/profile_sections.py'

assert (DATA / 'gated-review/motion-comparison.json').is_file(), 'Wait for the full normal GPU sequence.'
env = os.environ.copy()
env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
result = {}
for name, binary in [('baseline', 'bin-scissors'), ('gated', 'bin-joint-coarse31-gated')]:
    output = DATA / ('profile-' + name)
    assert not output.exists(), 'Use fresh output directories; do not overwrite evidence.'
    summary = WORK / ('sections-' + name + '.json')
    args = [sys.executable, '-X', 'utf8', str(PROFILER),
            str(ROOT / 'endoscopic_tissue/native' / binary / 'LiverCudaSim.dll'), str(summary),
            '--native-only', '--scene', 'kidney', '--tool', 'scissors', '--output', str(output)]
    print('[SECTIONS] ' + name, flush=True)
    with (WORK / ('sections-' + name + '.log')).open('w', encoding='utf8') as log:
        proc = subprocess.run(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    assert proc.returncode == 0, (name, proc.returncode)
    rows = json.loads(summary.read_text(encoding='utf8'))
    report = json.loads((output / 'report.json').read_text(encoding='utf8'))
    assert report['steps'] == sum(r['steps'] for r in rows) == 1110 and report['metrics']['completed']
    keys = set().union(*(r['sections_ms_per_step'] for r in rows))
    averages = {key: sum(r['steps'] * r['sections_ms_per_step'].get(key, 0) for r in rows) / 1110
                for key in sorted(keys)}
    result[name] = dict(steps=1110, sections_ms_per_step=averages,
                        quality_passed=report['metrics']['acceptance_passed'], dll_sha256=report['dll_sha256'])
    (WORK / 'sections-summary.json').write_text(json.dumps(dict(
        note='CUDA Graphs disabled. Section event timings include launch gaps; compare localization only. Different trajectories and one run per DLL do not isolate kernel causes.',
        runs=result), indent=2), encoding='utf8')
print('[SECTIONS] complete', flush=True)
