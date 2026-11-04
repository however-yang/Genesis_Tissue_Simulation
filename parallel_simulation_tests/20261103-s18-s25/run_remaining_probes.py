import json,sys,time,subprocess,os
from pathlib import Path
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r/'source'))
from endoscopic_tissue.training.acceptance_queue import Runner
started=time.monotonic()
while not (r/'results/interactive-baseline-cv4-01/result.json').exists():
 if time.monotonic()-started>900:raise TimeoutError('Baseline did not finish; no further GPU work launched')
 time.sleep(2)
code=Runner(r/'results/remaining-probes-01',json.loads((r/'remaining-probes-plan.json').read_text())).run()
print('Diagnostic batch complete',code,flush=True)
from endoscopic_tissue.training.server_queue import submit,run_next
job=submit(r/'queue',json.loads((r/'examples/queue-smoke.json').read_text()))
print('Submitted',job,flush=True)
job=run_next(r/'queue','GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa')
print('Queue result',job,flush=True)
print((job/'status.json').read_text(),flush=True)
