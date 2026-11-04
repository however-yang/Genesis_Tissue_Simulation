import json,sys,time,subprocess,os
from pathlib import Path
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r/'source'))
from endoscopic_tissue.training.acceptance_queue import Runner
code=Runner(r/'results/remaining-probes-02',json.loads((r/'remaining-probes-plan-02.json').read_text())).run()
print('Diagnostic batch complete',code,flush=True)
from endoscopic_tissue.training.server_queue import submit,run_next
job=submit(r/'queue',json.loads((r/'examples/queue-smoke.json').read_text()))
print('Submitted',job,flush=True)
job=run_next(r/'queue','GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa')
print('Queue result',job,flush=True)
print((job/'status.json').read_text(),flush=True)
