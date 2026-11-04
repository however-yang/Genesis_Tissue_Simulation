import pathlib,subprocess,os,json,time
r=pathlib.Path('/srv/disk2/hryang/genesis-exec');b=r/'runs/a100-functional-fixes-20261102-r1';release=r/'releases/0643e10991cf';envdir=r/'envs/genesis-py311-cu125-v1';env=os.environ.copy();env['PATH']=str(envdir/'bin')+':'+env['PATH'];env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1'
commands=[['/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/cmake', '-S', '/srv/disk2/hryang/genesis-exec/runs/a100-functional-fixes-20261102-r1/release-training-r3/source', '-B', '/srv/disk2/hryang/genesis-exec/builds/57f184c-release-training-r3-sm80', '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_COMPILER=/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/nvcc', '-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12', '-DCMAKE_CXX_COMPILER=/usr/bin/g++-12', '-DCUDAToolkit_ROOT=/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1'], ['/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/cmake', '--build', '/srv/disk2/hryang/genesis-exec/builds/57f184c-release-training-r3-sm80', '--parallel', '2']]
results=[]
for command in commands:
 start=time.time();q=subprocess.run(command,cwd=release,env=env);results.append(dict(command=command,exit_code=q.returncode,seconds=time.time()-start));(b/'combined-build.json').write_text(json.dumps(results,indent=2))
 if q.returncode:raise SystemExit(q.returncode)
