"""One isolated algorithm, serial GPU work, normal timing separate from probes/video."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
DATA = ROOT/'endoscopic_tissue/outputs/material-contact-next-20261031/groups'
PERF = ROOT/'endoscopic_tissue_backups/2026-10-30-perf-opt'
WRIST = ROOT/'endoscopic_tissue_backups/2026-10-31-wrist-opt'
results = []


def call(label, args, allowed=(0,)):
    print('[GROUPS] ' + label, flush=True)
    with (WORK/('groups-'+label+'.log')).open('w', encoding='utf8') as log:
        proc = subprocess.run([sys.executable, '-X', 'utf8', *map(str, args)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    results.append(dict(label=label, args=list(map(str, args)), exit_code=proc.returncode))
    (WORK/'groups-trial-results.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    assert proc.returncode in allowed, (label, proc.returncode)


def complete(path):
    report = json.loads((path/'report.json').read_text(encoding='utf8'))
    assert report['steps'] == 1110 and report['metrics']['completed'], path


for name in ('diagnostic', 'baseline-a1', 'baseline-a2', 'groups-b1', 'groups-b2', 'baseline-capture', 'groups-capture'):
    assert not (DATA/name).exists(), 'Use fresh output directories; preserve this evidence.'
call('diagnostic', [WORK/'check_contact_groups.py', DATA/'diagnostic', '--dll', ROOT/'endoscopic_tissue/native/bin-contact-groups31-probe/LiverCudaSim.dll', '--tool', 'scissors'], (0, 1))
complete(DATA/'diagnostic')
checks = json.loads((DATA/'diagnostic/contact-equivalence.json').read_text(encoding='utf8'))
assert len(checks) == 270 and checks[-1]['step'] == 1110
assert all(not r['status'] and not any(r[k] for k in ('different_counts', 'different_records', 'different_degrees', 'different_members', 'unsafe_group_rejections')) for r in checks)
assert sum(r['group_rejected_queries'] for r in checks) > 0
call('timing', [PERF/'run_batch.py', WORK/'groups-timing-jobs.json'])
for name in ('baseline-a1', 'groups-b1', 'groups-b2', 'baseline-a2'):
    complete(DATA/name)
    assert len(json.loads((DATA/name/'timing-steps.json').read_text(encoding='utf8'))) == 1110
call('captures', [PERF/'run_batch.py', WORK/'groups-capture-jobs.json'])
for name in ('baseline', 'groups'):
    complete(DATA/(name+'-capture'))
    call('render-'+name, [PERF/'replay_single_blas.py', DATA/(name+'-capture/replay.bin'), DATA/(name+'-video')])
call('compare', [PERF/'make_review.py', DATA/'baseline-video/interactions.mp4', DATA/'groups-video/interactions.mp4', DATA/'groups-review', 'CONTACT_GROUPS'])
call('motion', [WRIST/'review_motion.py', DATA/'baseline-capture/motion.jsonl', DATA/'groups-capture/motion.jsonl', DATA/'groups-review'])
print('[GROUPS] complete; speed and interaction still require review', flush=True)

call('section-diagnostics', [WORK/'run_sections.py'])
