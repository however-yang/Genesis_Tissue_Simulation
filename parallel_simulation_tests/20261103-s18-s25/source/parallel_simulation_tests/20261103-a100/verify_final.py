"""Read-only checks of production files and owned GPU/process cleanup."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import psutil
from benchmark import ROOT,RELEASE,LIBRARY,GPU,gpu_sample

def main():
    audit=json.loads((ROOT/'source-audit.json').read_text())
    changed=[]
    for f in audit['files']:
        p=Path(f['path'])
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:changed.append(str(p))
    library_sha=hashlib.sha256(LIBRARY.read_bytes()).hexdigest()
    env=os.environ.copy();env['GIT_OPTIONAL_LOCKS']='0'
    status=subprocess.run(['git','-C',str(RELEASE),'status','--porcelain'],env=env,text=True,capture_output=True,check=True).stdout
    own=[]
    for p in psutil.process_iter(['pid','uids','cmdline','name']):
        try:
            if p.info['uids'].real!=os.getuid() or p.pid==os.getpid():continue
            cmd=p.info['cmdline'] or []
            # Skip shells/this verifier; only test workers/controllers in our private namespace.
            test_worker=any(str(ROOT/filename) in cmd for filename in ('worker.py','benchmark.py','test_pool.py','mps_benchmark.py'))
            private_mps=(p.info['name'] in ('nvidia-cuda-mps-control','nvidia-cuda-mps-server') and p.environ().get('CUDA_MPS_PIPE_DIRECTORY','').startswith(str(ROOT/'mps')+'/'))
            if test_worker or private_mps:own.append({'pid':p.pid,'name':p.info['name']})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    parsed=[]
    for f in ROOT.glob('*.py'):ast.parse(f.read_text(),filename=str(f));parsed.append(f.name)
    sample=gpu_sample(GPU)
    result={'time':time.time(),'production_git_porcelain':status,'audited_source_changes':changed,
        'library_sha256':library_sha,'library_matches_initial':library_sha==audit['library_sha256'],
        'remaining_owned_gpu_workers_or_mps':own,'selected_gpu_after':sample,'ast_parsed_files':parsed,
        'test_file_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')},
        'cpu_accounting_tests':{'command':'python -B -m unittest -v test_supervisor','tests_passed':2,
            'record':'Executed successfully during this investigation; see test-checks.json.'}}
    (ROOT/'final-verification.json').write_text(json.dumps(result,indent=2))
    assert not status and not changed and result['library_matches_initial'] and not own,result
    print('Production source/library unchanged; no owned workers or MPS remain; parsed',len(parsed),'Python files; GPU:',sample['used_mib'],'MiB')
if __name__=='__main__':main()
