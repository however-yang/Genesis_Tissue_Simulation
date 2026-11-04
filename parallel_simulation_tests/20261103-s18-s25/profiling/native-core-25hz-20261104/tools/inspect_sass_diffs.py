from pathlib import Path
import subprocess,re,difflib,json,os
os.sched_setaffinity(0,[30,31]);p=Path(__file__).resolve().parents[1]
cu='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/cuobjdump'
keys=['_Z18k_CoarseAccumulatePK12UnifiedStateP11CoarseBlock','_Z15k_UnifiedVertexILb0EEvPK12UnifiedStatePKiS4_','__cuda_sm20_div_rn_f64_full']
def get(path):
 out={};key=None
 proc=subprocess.Popen([cu,'--dump-sass',str(path)],text=True,stdout=subprocess.PIPE)
 for line in proc.stdout:
  if 'Function :' in line:
   key=line.split('Function :',1)[1].strip();key=key if key in keys else None
   if key:out[key]=[]
  elif key and re.search(r'/\*[0-9a-f]+\*/',line):
   line=re.sub(r'/\*.*?\*/','',line).strip()
   if line:out[key].append(' '.join(line.split()))
 assert proc.wait()==0;return out
a=get(p.parent/'native-core-20261104/native-all-queries-v2/build/lib/libLiverCudaSim.so');b=get(p/'native-query-v3/build/lib/libLiverCudaSim.so')
out={}
for k in keys:out[k]=list(difflib.unified_diff(a[k],b[k]))
(p/'results/query-v3-sass-diff-examples.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v[:50] for k,v in out.items()},indent=2))
