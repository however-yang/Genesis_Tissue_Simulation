"""Sequential full-frame ABBA and regression; never run two GPU cases together."""
import hashlib,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_case import run
build=json.loads((ROOT/'results/build-retained-runtime.json').read_text());assert build['returncode']==0
library=ROOT/'native/binaries/runtime-core.so'
assert not library.exists();shutil.copy2(build['dll'],library)
assert hashlib.sha256(library.read_bytes()).hexdigest()==build['sha256']
queue=[('final-a1-v8','baseline',0,'benchmark'),('final-b1-runtime',str(library),43264,'benchmark'),('final-b2-runtime',str(library),43264,'benchmark'),('final-a2-v8','baseline',0,'benchmark'),('final-proof',str(library),43264,'validate'),('final-rgbd-proof',str(library),43264,'rgbd-validated'),('final-snapshot',str(library),43264,'snapshot'),('final-profile',str(library),43264,'profile')]
state={'started_unix':time.time(),'completed':[]}
for name,lib,mode,kind in queue:
 state['running']=name;(ROOT/'results/final-queue.json').write_text(json.dumps(state,indent=2)+'\n')
 code=run(name,library=lib,core=mode,kind=kind)
 if code:
  state['failed']=name;(ROOT/'results/final-queue.json').write_text(json.dumps(state,indent=2)+'\n');raise SystemExit(code)
 data=json.loads((ROOT/'results'/name/'result.json').read_text())
 if kind=='benchmark':assert data['completed'] and data['quality_passed'] and len(data['frames'])==1410
 if kind in ('validate','rgbd-validated'):
  zero=data['zero_mode_gate_stats'];assert zero['reference_proof_executed'] and zero['reference_coordinates_unchanged']
  assert data['grip_feedback_validation']['counters'][0]>0 and data['grip_feedback_validation']['passed']
 if kind=='snapshot':assert data['native_archive_restored_bit_exact'] and data['runtime_core_mode_mismatch_rejected'] and data['live_core_mode_change_rejected']
 state['completed'].append(name)
state.pop('running');state['finished_unix']=time.time();(ROOT/'results/final-queue.json').write_text(json.dumps(state,indent=2)+'\n')
print('FINAL QUEUE COMPLETED',flush=True)
