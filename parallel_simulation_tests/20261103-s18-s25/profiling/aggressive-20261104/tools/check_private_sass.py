"""Offline binding for a host-only workspace-accounting fix."""
from pathlib import Path
import subprocess,re,hashlib,json,collections
p=Path(__file__).resolve().parents[1];cu='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/cuobjdump'
def functions(lines):
 result=collections.defaultdict(set);name=None;instructions=[]
 def save():
  if name and instructions:result[name].add(hashlib.sha256('\n'.join(instructions).encode()).hexdigest())
 for line in lines:
  if 'Function :' in line:save();name=line.split('Function :',1)[1].strip();instructions=[]
  elif re.search(r'/\*[0-9a-f]+\*/',line):
   text=re.sub(r'/\*.*?\*/','',line).strip()
   if text:instructions.append(' '.join(text.split()))
 save();return result
result={}
for old,new,existing in [('native-gpu-diagnostics','native-gpu-diagnostics-v2','gpu-diagnostics-sass.txt'),('native-precomputed','native-precomputed-v2','precomputed-sass.txt')]:
 with (p/'results'/existing).open() as f:before=functions(f)
 process=subprocess.Popen([cu,'--dump-sass',str(p/new/'build/lib/libLiverCudaSim.so')],stdout=subprocess.PIPE,text=True);after=functions(process.stdout);assert process.wait()==0
 changed=[k for k in before.keys()&after.keys() if before[k]!=after[k]]
 fields=['vendor/cuda/training/state_archive.h','vendor/cuda/training/state_archive.cpp']
 result[new]=dict(before_functions=len(before),after_functions=len(after),changed=changed,missing=sorted(before.keys()-after.keys()),added=sorted(after.keys()-before.keys()),snapshot_code_unchanged={f:hashlib.sha256((p/old/'source'/f).read_bytes()).hexdigest()==hashlib.sha256((p/new/'source'/f).read_bytes()).hexdigest() for f in fields})
 result[new]['all_device_functions_identical']=before==after
(p/'results/private-scratch-v2-binding.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert all(x['all_device_functions_identical'] and all(x['snapshot_code_unchanged'].values()) for x in result.values())
