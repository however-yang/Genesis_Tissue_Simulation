import argparse,json,subprocess,sys,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--prefix',required=True);a=p.parse_args();here=Path(__file__).resolve().parents[1];os.chdir(here);config=json.loads(a.config.read_text());records=[]
for i,case in enumerate(config):
 name=a.prefix+'-'+str(i)+'-'+case['name'];out=here/'results'/name
 cmd=[sys.executable,str(here/'tools/profile_all_queries.py'),'--source',str(here/case.get('source','python-gpu-check')),'--dll',str(here/case['dll']),'--output',str(out),'--query-mode','3','--bound-cache','--core-optimizations',str(case['core']),'--gpu-topology','--cpus','7,19,20,25','--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1']
 if case.get('gpu_body'):cmd+=['--gpu-body-check']
 started=time.time();print(json.dumps(dict(start=name,unix_s=started,case=case)),flush=True)
 with (here/'logs'/(name+'.txt')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 data=json.loads((out/'result.json').read_text());record=dict(name=name,case=case,returncode=r.returncode,completed=data['completed'],quality_passed=data['quality_passed'],performance=data.get('performance'),gpu_body_statistics=data.get('gpu_body_statistics'),error=data.get('error'),elapsed_s=time.time()-started);records.append(record);print(json.dumps(record),flush=True)
 (here/'results'/(a.prefix+'-matrix.json')).write_text(json.dumps(records,indent=2))
 if 'Foreign GPU' in (data.get('error') or '') or 'Resource ceiling' in (data.get('error') or ''):raise SystemExit(2)
if any(x['returncode'] or not x['completed'] or not x['quality_passed'] for x in records):raise SystemExit(2)
