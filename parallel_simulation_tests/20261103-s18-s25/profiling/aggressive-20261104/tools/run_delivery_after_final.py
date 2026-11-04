"""Finish snapshot integration checks and offline artifacts after all GPU benchmarks."""
import json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parents[1];py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python'
while not (p/'results/final-selection.json').exists():
 log=(p/'logs/final-v2-queue.txt').read_text() if (p/'logs/final-v2-queue.txt').exists() else ''
 if any(x in log for x in ['Final validation failed','Final full scene','AssertionError','Predecessor stopped']):raise SystemExit('Final prerequisite failed')
 time.sleep(5)
d=json.loads((p/'results/final-selection.json').read_text());cmd=[py,str(p/'tools/check_snapshot_roundtrip.py'),'--source',d['source'],'--dll',d['dll'],'--output',str(p/'results/snapshot-roundtrip-final')]
if '--gpu-topology' in d['flags']:cmd.append('--gpu-topology')
print('START snapshot-roundtrip-final',flush=True)
with (p/'delivery-commands.jsonl').open('a') as f:f.write(json.dumps(dict(name='snapshot-roundtrip-final',argv=cmd))+'\n')
with (p/'logs/snapshot-roundtrip-final.txt').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
print('END snapshot-roundtrip-final',r.returncode,flush=True)
if r.returncode:raise SystemExit('Final snapshot integration regression failed')
reg=json.loads((p/'results/snapshot-roundtrip-final/result.json').read_text());assert reg['passed'];d['real_snapshot_roundtrip_passed']=True;(p/'results/final-selection.json').write_text(json.dumps(d,indent=2))
print('START offline delivery',flush=True)
subprocess.run([py,str(p/'tools/deliver_v2.py')],check=True)
(p/'results/completion-status.json').write_text(json.dumps(dict(completed=True,finished_unix_s=time.time(),source=d['source'],production_replaced=False,stable_20hz_passed=d['stable_20hz_passed'],formal_physics_domain_approved=False),indent=2));print('DONE all artifacts',flush=True)
