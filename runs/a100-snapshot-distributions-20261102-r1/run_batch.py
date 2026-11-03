import os,sys,time,json,subprocess,signal,hashlib,traceback
from pathlib import Path
b=Path(__file__).resolve().parent;r=b.parents[1];release=r/'releases/0643e10991cf';uuid='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa';jobs=json.loads((b/'jobs.json').read_text())
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(release),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYOPENGL_PLATFORM='egl',EGL_DEVICE_ID='1')
s={'phase':'starting','gpu_uuid':uuid,'jobs':[],'started':time.time(),'default_replaced':False}
def save():(b/'status.json').write_text(json.dumps(s,indent=2))
def sample():
 commands=[['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']]
 records=[]
 for cmd in commands:
  q=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
  if q.returncode:raise RuntimeError('GPU monitoring failed: '+q.stderr)
  records.append(q.stdout.strip())
 record={'time':time.time(),'gpus':records[0],'processes':records[1],'loadavg':os.getloadavg()}
 with (b/'gpu-monitor.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 return record
def free(record):
 row=next(x for x in record['gpus'].splitlines() if uuid in x);parts=[x.strip() for x in row.split(',')]
 return float(parts[-2])<256 and float(parts[-1])<5 and not any(uuid in x for x in record['processes'].splitlines())
try:
 save()
 for job in jobs:
  name=job['name'];s.update(phase='waiting_gpu',current=name);save();stable=0
  while stable<3:
   stable=stable+1 if free(sample()) else 0
   time.sleep(5)
  log=b/(name+'.log');start=time.time();row={'name':name,'args':job['args'],'start':start,'gpu_uuid':uuid};s['jobs'].append(row)
  with log.open('xb') as out:
   proc=subprocess.Popen(job['args'],cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   row['pid']=proc.pid;s.update(phase='running');save();last_progress=start;size=0;max_idle=job.get('max_idle',300 if 'snapshot-' not in name or 'three' in name else 900)
   while proc.poll() is None:
    stat=log.stat()
    if stat.st_size!=size:size=stat.st_size;last_progress=time.time()
    rec=sample();row['last_sample']=rec['time'];row['last_progress']=last_progress
    if time.time()-start>job.get('max_wall',1800) or time.time()-last_progress>max_idle:
     row['timeout']=True;os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
     break
    save();time.sleep(10)
   row.update(exit_code=proc.returncode,finished=time.time(),wall_s=time.time()-start)
  result=release/'endoscopic_tissue/outputs/acceptance-fixes-20261102'/name/'result.json'
  if result.exists():row['result_file']=str(result)
  save()
  if row.get('timeout') or proc.returncode not in (0,2):raise RuntimeError('Worker stopped: '+name+'; preserve evidence and diagnose before continuing')
 s.update(phase='complete',finished=time.time());save()
except Exception as e:
 s.update(phase='failed',error=str(e),finished=time.time());save();traceback.print_exc();sys.exit(1)
