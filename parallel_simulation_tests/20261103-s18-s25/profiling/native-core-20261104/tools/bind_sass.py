"""Compare all original device functions; preserve instruction text, ignore addresses."""
import argparse,subprocess,re,json,hashlib,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cu='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/cuobjdump'
def read(path):
 r=subprocess.Popen([cu,'--dump-sass',str(path)],stdout=subprocess.PIPE,text=True);result=collections.defaultdict(set);name=None;lines=[]
 def save():
  if name and lines:result[name].add(hashlib.sha256('\n'.join(lines).encode()).hexdigest())
 for line in r.stdout:
  if 'Function :' in line:save();name=line.split('Function :',1)[1].strip();lines=[]
  elif re.search(r'/\*[0-9a-f]+\*/',line):
   t=re.sub(r'/\*.*?\*/','',line).strip()
   if t:lines.append(' '.join(t.split()))
 save();assert r.wait()==0;return result
before=read(a.before);after=read(a.after);d=dict(before=str(a.before),after=str(a.after),before_functions=len(before),after_functions=len(after),changed=[k for k in before.keys()&after.keys() if not before[k].issubset(after[k])],missing=sorted(before.keys()-after.keys()),added=sorted(after.keys()-before.keys()));d['all_original_functions_unchanged']=not d['changed'] and not d['missing'];a.output.write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
if not d['all_original_functions_unchanged']:raise SystemExit(2)
