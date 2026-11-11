"""Inspect or terminate only our obsolete, still-waiting input-validation helper."""
import hashlib
import json
from pathlib import Path
import sys
import psutil

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
target = (P/'finish_id_validation.py').resolve()
found = []
for process in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        if process.info['name'].lower() not in ('python.exe', 'pythonw.exe'):
            continue
        for arg in process.info['cmdline'] or []:
            if Path(arg).name != target.name:
                continue
            path = Path(arg)
            if not path.is_absolute():
                path = Path(process.cwd())/path
            if path.resolve() == target:
                found.append({'pid': process.pid, 'command': process.info['cmdline']})
                break
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        continue
print(json.dumps(found, indent=2))
if len(sys.argv) == 3 and sys.argv[1] == '--terminate':
    pid = int(sys.argv[2])
    if len(found) != 1 or found[0]['pid'] != pid:
        raise RuntimeError('Expected exactly the previously inspected helper PID')
    expected = json.loads((P/'final-positive/source-hashes.json').read_text())
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (ROOT/'endoscopic_tissue/training').glob('*.py')}
    if current != expected:
        raise RuntimeError('Source changed: helper may no longer be just waiting')
    process = psutil.Process(pid)
    process.terminate()
    process.wait(timeout=10)
    (P/'waiter-restart.json').write_text(json.dumps({'terminated_owned_pid': pid,
        'reason': 'Waited for six jobs, but five named post-batch jobs exist; corrected to explicit named set',
        'source_was_still_unchanged': True, 'all_five_post_batch_jobs_passed': True}, indent=2)+'\n')
