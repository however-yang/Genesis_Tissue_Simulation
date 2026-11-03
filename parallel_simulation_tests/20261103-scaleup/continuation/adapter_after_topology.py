"""Wait for this run's private MPS cleanup, then run independent adapter checks."""
from pathlib import Path
import json,subprocess,sys,time
h=Path(__file__).resolve().parent
sys.path.insert(0,str(h.parent));from benchmark import gpu_sample,GPU,PYTHON
start=time.monotonic()
while not (h.parent/'mps/t32/lifecycle.json').exists():
 if time.monotonic()-start>2100:raise TimeoutError('Topology batch did not clean up')
 time.sleep(2)
summary=json.loads((h.parent/'runs/t32/summary.json').read_text())
if not summary['complete']:raise RuntimeError('Inspect interrupted topology batch before adapter tests')
for i in range(15):
 sample=gpu_sample(GPU)
 if not sample['processes'] and sample['used_mib']<512 and sample['gpu_percent']<=5:break
 time.sleep(2)
else:raise RuntimeError('GPU not idle after own topology batch')
print('START independent snapshot and policy adapter checks',flush=True)
cmd=[PYTHON,'-B','-u',str(h/'acceptance_queue.py'),'--plan',str(h/'training-adapter-plan.json'),'--directory',str(h/'training-adapter-run'),'--gpu',GPU]
code=subprocess.call(cmd,cwd=h)
(h/'adapter-followup-status.json').write_text(json.dumps(dict(exit_code=code,finished=time.time(),command=cmd),indent=2))
raise SystemExit(code)
