"""Serial GPU validation and offline renders; stops on any unhandled failure."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).parent
OUTPUT = ROOT/'endoscopic_tissue/outputs/training-foundation-20261101'
PY = sys.executable
jobs = [
    ('repeat-legacy', [PY, '-X', 'utf8', str(WORK/'paired_replay.py'), '--mode', 'legacy', '--output', str(OUTPUT/'legacy-wall-03')]),
    ('native-reset', [PY, '-X', 'utf8', str(WORK/'verify_native_reset.py')]),
    ('compare-adapter', [PY, '-X', 'utf8', str(WORK/'compare_replays.py'), str(OUTPUT/'legacy-wall-02'), str(OUTPUT/'adapter-wall-01'), '--output', str(WORK/'paired-adapter.json')]),
    ('compare-repeat', [PY, '-X', 'utf8', str(WORK/'compare_replays.py'), str(OUTPUT/'legacy-wall-02'), str(OUTPUT/'legacy-wall-03'), '--output', str(WORK/'paired-repeat.json')]),
]
for case in ('legacy-wall-02', 'adapter-wall-01', 'legacy-wall-03'):
    jobs.append(('render-'+case, [PY, '-X', 'utf8', str(ROOT/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation/render_review.py'), str(OUTPUT/case)]))
results = []
for name, command in jobs:
    start = time.perf_counter()
    print('START', name, flush=True)
    with (WORK/(name+'.log')).open('w', encoding='utf8') as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    results.append(dict(name=name, exit_code=process.returncode, wall_time_s=time.perf_counter()-start, command=command))
    (WORK/'validation-jobs.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    print('END', name, 'exit', process.returncode, flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)
