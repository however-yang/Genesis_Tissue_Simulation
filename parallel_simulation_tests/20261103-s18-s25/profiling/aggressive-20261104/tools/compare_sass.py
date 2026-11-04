"""Compare normalized per-function SASS, never ELF addresses or host relocations."""
import argparse,collections,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def functions(path):
 result=collections.defaultdict(set);name=None;lines=[]
 def save():
  if name and lines:result[name].add(hashlib.sha256('\n'.join(lines).encode()).hexdigest())
 with path.open() as f:
  for line in f:
   if 'Function :' in line:save();name=line.split('Function :',1)[1].strip();lines=[]
   elif re.search(r'/\*[0-9a-f]+\*/',line):
    instruction=re.sub(r'/\*.*?\*/','',line).strip()
    if instruction:lines.append(' '.join(instruction.split()))
 save();return result
old=functions(a.before);new=functions(a.after)
d=dict(before=str(a.before),after=str(a.after),original_functions=len(old),after_functions=len(new),missing=sorted(old.keys()-new.keys()),changed=[k for k in old.keys()&new.keys() if not old[k].issubset(new[k])],added=sorted(new.keys()-old.keys()))
d['all_original_functions_identical']=not d['missing'] and not d['changed'];a.output.write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
