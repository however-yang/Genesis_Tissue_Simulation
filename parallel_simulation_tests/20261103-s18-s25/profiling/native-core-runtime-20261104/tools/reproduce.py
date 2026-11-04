"""Reproduce the selected independent candidate; frozen V8 stays unchanged."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def check():
 manifest=json.loads((ROOT/'delivery/manifest.json').read_text())
 problems=[]
 for key in ('native_source_sha256','tool_sha256'):
  for name,expected in manifest[key].items():
   f=ROOT/name
   if not f.is_file() or digest(f)!=expected:problems.append(name)
 lib=ROOT/manifest['selected_library']
 if not lib.is_file() or digest(lib)!=manifest['library_sha256']:problems.append(str(lib))
 baseline=json.loads((ROOT/'baseline.json').read_text())
 f=Path(baseline['baseline_library'])
 if digest(f)!=baseline['baseline_library_sha256']:problems.append('Frozen V8 library changed')
 for root,key in [(Path(baseline['borrowed_python_source']),'baseline_python_files'),(Path(baseline['baseline_experiment'])/'native-readonly-v8/source','baseline_native_files')]:
  for name,expected in baseline[key].items():
   target=root/name
   if not target.is_file() or digest(target)!=expected:problems.append(str(target))
 if digest(ROOT/'REPORT.md')!=manifest['report_sha256']:problems.append('Report differs from delivery manifest')
 assert not problems, 'Hash mismatch: '+repr(problems)
 return manifest,lib

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run',choices=['benchmark','validate','rgbd-validated','snapshot','profile'])
 p.add_argument('--name',help='New case directory; existing results are never overwritten')
 a=p.parse_args();manifest,library=check()
 print(json.dumps({'hashes_verified':True,'library':str(library),'core_mode':manifest['core_mode'],'stable_25hz_approved':manifest['stable_25hz_approved'],'formal_physics_domain_approved':False,'training_domain_approved':False}),flush=True)
 if a.run:
  assert a.name,'--name is required for --run'
  from run_case import run
  raise SystemExit(run(a.name,library=str(library),core=manifest['core_mode'],kind=a.run,query_threads=manifest['query_threads']))
