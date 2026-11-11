from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import psutil

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
protected = ['endoscopic_tissue/native/bin-contact-groups31/LiverCudaSim.dll',
             'endoscopic_tissue/run.py', 'endoscopic_tissue/scenario.py',
             'endoscopic_tissue/native_backend.py', 'endoscopic_tissue/rendering.py',
             'endoscopic_tissue/native/vendor/cuda/xpbd/xpbd_solver.cu',
             'endoscopic_tissue/native/vendor/cuda/cut.cu']
result = {'disk_free_gb':shutil.disk_usage(ROOT).free/1024**3,
          'protected':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected},
          'processes':[p.info for p in psutil.process_iter(['pid','name','cmdline'])
                       if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe','nvcc.exe','ffmpeg.exe')],
          'gpu':subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
                                '--format=csv'],capture_output=True,text=True).stdout}
(P/'machine.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
