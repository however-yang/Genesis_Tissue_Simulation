"""One isolated algorithm, serial GPU work, normal timing separate from probes/video."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
DATA = ROOT/'endoscopic_tissue/outputs/material-contact-next-20261031/additive'
PERF = ROOT/'endoscopic_tissue_backups/2026-10-30-perf-opt'
WRIST = ROOT/'endoscopic_tissue_backups/2026-10-31-wrist-opt'
results = []


def call(label, args, allowed=(0,)):
    print('[ADDITIVE] ' + label, flush=True)
    with (WORK/('additive-'+label+'.log')).open('w', encoding='utf8') as log:
        proc = subprocess.run([sys.executable, '-X', 'utf8', *map(str, args)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    results.append(dict(label=label, args=list(map(str, args)), exit_code=proc.returncode))
    (WORK/'additive-trial-results.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    assert proc.returncode in allowed, (label, proc.returncode)


def complete(path):
    report = json.loads((path/'report.json').read_text(encoding='utf8'))
    assert report['steps'] == 1110 and report['metrics']['completed'], path


for name in ('diagnostic-a3', 'baseline-a1', 'baseline-a2', 'additive-b1', 'additive-b2', 'baseline-capture', 'additive-capture'):
    assert not (DATA/name).exists(), 'Use fresh output directories; preserve this evidence.'
call('diagnostic', [WORK/'check_additive.py', DATA/'diagnostic-a3', '--dll', ROOT/'endoscopic_tissue/native/bin-additive31-accurate-probe/LiverCudaSim.dll'], (0, 1))
complete(DATA/'diagnostic-a3')
checks = json.loads((DATA/'diagnostic-a3/additive-checks.json').read_text(encoding='utf8'))
assert len(checks) == 10 and checks[-1]['step'] == 1110
call('timing', [PERF/'run_batch.py', WORK/'additive-timing-jobs.json'])
for name in ('baseline-a1', 'additive-b1', 'additive-b2', 'baseline-a2'):
    complete(DATA/name)
    assert len(json.loads((DATA/name/'timing-steps.json').read_text(encoding='utf8'))) == 1110
call('captures', [PERF/'run_batch.py', WORK/'additive-capture-jobs.json'])
for name in ('baseline', 'additive'):
    complete(DATA/(name+'-capture'))
    call('render-'+name, [PERF/'replay_single_blas.py', DATA/(name+'-capture/replay.bin'), DATA/(name+'-video')])
call('compare', [PERF/'make_review.py', DATA/'baseline-video/interactions.mp4', DATA/'additive-video/interactions.mp4', DATA/'additive-review', 'ADDITIVE_MG'])
call('motion', [WRIST/'review_motion.py', DATA/'baseline-capture/motion.jsonl', DATA/'additive-capture/motion.jsonl', DATA/'additive-review'])
print('[ADDITIVE] complete; speed and interaction still require review', flush=True)
