import os,subprocess,pathlib,json,time,hashlib
r=pathlib.Path('/srv/disk2/hryang/genesis-exec'); p=r/'runs/a100-boundary-fans-20261102-r1'; envroot=r/'envs/genesis-py311-cu125-v1'; b=r/'builds/c805c77-boundary-fans-r5-sm80'
env=dict(os.environ,PATH=str(envroot/'bin')+':'+os.environ['PATH'],CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa')
cm=str(envroot/'bin/cmake'); commands=[[cm,'-S',str(p/'source'),'-B',str(b),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER='+str(envroot/'bin/nvcc'),'-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12','-DCMAKE_CXX_COMPILER=/usr/bin/g++-12','-DCUDAToolkit_ROOT='+str(envroot)],[cm,'--build',str(b),'--parallel','2']]
results=[]
for cmd in commands:
 start=time.time(); rc=subprocess.run(cmd,env=env,timeout=1200).returncode; results.append({'command':cmd,'exit_code':rc,'seconds':time.time()-start}); (p/'build-status.json').write_text(json.dumps(results,indent=2))
 if rc:raise SystemExit(rc)
lib=b/'lib/libLiverCudaSim.so'; (p/'library.json').write_text(json.dumps({'library':str(lib),'sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'default_replaced':False},indent=2))
