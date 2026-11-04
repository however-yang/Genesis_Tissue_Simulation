import argparse,json,subprocess,sys,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dll',required=True);p.add_argument('--sequence',required=True);p.add_argument('--prefix',required=True);p.add_argument('--no-render',action='store_true');a=p.parse_args();here=Path(__file__).resolve().parents[1];os.chdir(here)
seq=[int(x) for x in a.sequence.split(',')];records=[]
for i,mode in enumerate(seq):
 name=a.prefix+'-'+str(i)+'-mode'+str(mode);out=here/'results'/name
 cmd=[sys.executable,str(here/'tools/profile_all_queries.py'),'--source',str(here.parent/'aggressive-20261104/final-source'),'--dll',str(Path(a.dll).resolve()),'--output',str(out),'--query-mode','3','--bound-cache','--core-optimizations',str(mode),'--gpu-topology','--cpus','7,19,20,25']
 if a.no_render:cmd+=['--no-render']
 else:cmd+=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1']
 start=time.time();print(json.dumps(dict(start=name,unix_s=start)),flush=True)
 with (here/'logs'/(name+'.txt')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 data=json.loads((out/'result.json').read_text());record=dict(name=name,mode=mode,returncode=r.returncode,completed=data['completed'],quality_passed=data['quality_passed'],performance=data.get('performance'),elapsed_s=time.time()-start);records.append(record);print(json.dumps(record),flush=True)
 (here/'results'/(a.prefix+'-matrix.json')).write_text(json.dumps(records,indent=2))
 if r.returncode or not data['completed'] or not data['quality_passed']:raise SystemExit(2)
