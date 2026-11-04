"""Replace only our idle diagnostic parent; preserve every completed result."""
import json
from pathlib import Path
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'endoscopic_tissue/outputs/acceptance-fixes-20261102'
OLD = HERE / 'boundary-diagnose-3.json'
NEW = HERE / 'feasible-diagnose-2.json'
old = json.loads(OLD.read_text(encoding='utf8'))
done = json.loads(OLD.with_suffix('.results.json').read_text(encoding='utf8'))
completed = [j['name'] for j in done['jobs']]
assert completed == [j['name'] for j in old['jobs'][:len(completed)]]
remaining = old['jobs'][len(completed):]
assert remaining and remaining[0]['name'] == 'flap-step-candidate-r0'
next_status = OUT / (remaining[0]['name'] + '-supervisor.json')
assert json.loads(next_status.read_text(encoding='utf8'))['phase'] == 'waiting'
assert not NEW.exists()
matches = []
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    cmd = proc.info['cmdline'] or []
    if (proc.info['name'] or '').lower() not in ('python.exe', 'pythonw.exe'):
        continue
    if any(Path(a).name == 'run_batch.py' for a in cmd) and any(Path(a).name == OLD.name for a in cmd):
        matches.append(proc)
assert len(matches) == 1, [p.pid for p in matches]
proc = matches[0]
assert Path(proc.cwd()).resolve() == ROOT.resolve()
record = {'pid': proc.pid, 'creation': proc.create_time(), 'command': proc.cmdline(),
          'unix': time.time(), 'completed_jobs_preserved': completed,
          'reason': 'Prioritize isolated same-state repair probes while our parent is idle',
          'user_processes_touched': False}
proc.suspend()
closed = False
try:
    assert not proc.children(recursive=True), 'Worker active; retain original queue'
    assert json.loads(next_status.read_text(encoding='utf8'))['phase'] == 'waiting'
    proc.terminate()
    proc.wait(10)
    closed = True
finally:
    if not closed and proc.is_running():
        proc.resume()
(OUT/'boundary-idle-parent-reordered.json').write_text(json.dumps(record, indent=2), encoding='utf8')
feasible = json.loads((HERE/'feasible-diagnose-1.json').read_text(encoding='utf8'))
jobs = feasible['jobs'][:2] + remaining[:3] + feasible['jobs'][2:] + remaining[3:]
assert len({j['name'] for j in jobs}) == len(jobs)
NEW.write_text(json.dumps({'note': 'Completed old results retained; GPU work remains serial. All candidates unaccepted.',
                           'jobs': jobs}, indent=2), encoding='utf8')
print(json.dumps(record, indent=2))
print('Replacement queue:', NEW, 'jobs:', len(jobs))
