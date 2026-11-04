import pathlib,subprocess,os,json,time
r=pathlib.Path('/srv/disk2/hryang/genesis-exec');b=r/'runs/a100-functional-fixes-20261102-r1';release=r/'releases/0643e10991cf';envdir=r/'envs/genesis-py311-cu125-v1';env=os.environ.copy();env['PATH']=str(envdir/'bin')+':'+env['PATH'];env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1'
commands=[[str(envdir/'bin/python'),'-B','endoscopic_tissue_backups/2026-11-02-a100-handoff/prepare_source.py','--variant','boundary-snapshot2'],[str(envdir/'bin/cmake'),'-S',str(release/'endoscopic_tissue/outputs/a100-migration/boundary-snapshot2/source'),'-B',str(r/'builds/0643e10991cf-boundary-snapshot2-sm80'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER='+str(envdir/'bin/nvcc'),'-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12','-DCMAKE_CXX_COMPILER=/usr/bin/g++-12','-DCUDAToolkit_ROOT='+str(envdir)],[str(envdir/'bin/cmake'),'--build',str(r/'builds/0643e10991cf-boundary-snapshot2-sm80'),'--parallel','2']]
results=[]
for command in commands:
 start=time.time();q=subprocess.run(command,cwd=release,env=env);results.append(dict(command=command,exit_code=q.returncode,seconds=time.time()-start));(b/'boundary-build.json').write_text(json.dumps(results,indent=2))
 if q.returncode:raise SystemExit(q.returncode)
