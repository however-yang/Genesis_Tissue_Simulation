import pathlib,json,subprocess,os,time
p=pathlib.Path(__file__).resolve().parent;args=json.loads((p/'command.json').read_text());env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',PYTHONDONTWRITEBYTECODE='1');start=time.time()
try:
 result=subprocess.run(args,env=env,timeout=3600);status={'exit_code':result.returncode,'wall_s':time.time()-start,'offline_replay_only':True,'gpu_simulation_started':False}
except BaseException as e:status={'error':repr(e),'wall_s':time.time()-start}
(p/'status.json').write_text(json.dumps(status,indent=2))
