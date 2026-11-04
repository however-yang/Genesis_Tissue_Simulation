"""Local rollback point only. GPU acceptance is pending; never pushes."""
from pathlib import Path
import hashlib
import json
import py_compile
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
assert git('branch','--show-current')=='dev'
assert not git('diff','--cached','--name-only'),'Do not combine an existing index with this checkpoint'
source_files=list((ROOT/'endoscopic_tissue/training').glob('*.py'))+list(HERE.glob('*.py'))
for file in source_files:py_compile.compile(str(file),doraise=True)
subprocess.run([sys.executable,'-X','utf8',str(HERE/'run_cpu_tests.py')],cwd=ROOT,check=True)
base=json.loads((HERE/'baseline.json').read_text(encoding='utf8'))
for name,want in base['protected'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,name

files=[ROOT/'endoscopic_tissue/training'/n for n in (
    'env.py','experiment.py','native.py','physics.py','quality.py','damage.py','test_damage.py',
    'snapshot.py','test_snapshot.py','lineage.py','tasks.py','data.py','splits.py','vision.py',
    'geometry.py','test_geometry.py','collect.py')]
files += [HERE/n for n in (
    'build_native.py','prepare_native_base.py','prepare_training_native.py','prepare_snapshot_native.py',
    'inventory_state.py','state-inventory.json','state-reviewed-host.json','compare_default.py',
    'validate_damage.py','validate_native_snapshot.py','compare_kernel_code.py','run_cpu_tests.py',
    'gpu_queue.py','gpu-queue.json','replace_waiter.py','task_plan.md','findings.md','progress.md',
    'snapshot-design-notes.md','checkpoint_experimental.py')]
for directory in ('snapshot-native','training-native','native-reference','geometries'):
    files += [f for f in (HERE/directory).glob('*') if f.is_file()]
files += [ROOT/'endoscopic_tissue/docs/training-s15-s17-results-2026-11-01.md']
files += [ROOT/'endoscopic_tissue/outputs/s15-s17-20261101'/n for n in (
    'cpu-tests.json','cpu-tests.txt','protected-current.json','kernel-code.json','default-comparison/supervisor-replaced.json')]
files += [HERE/'artifacts'/v/'build-result.json' for v in ('baseline','training-state1','training-snapshot1')]
names=sorted({f.relative_to(ROOT).as_posix() for f in files})
assert all((ROOT/n).is_file() and (ROOT/n).stat().st_size<2*1024**2 for n in names)
record={'files':names,'bytes':sum((ROOT/n).stat().st_size for n in names),'gpu_validated':False,'pushed':False}
(HERE/'experimental-selected.json').write_text(json.dumps(record,indent=2),encoding='utf8')
paths=HERE/'experimental-selected.nul'
paths.write_bytes(b'\0'.join(n.encode() for n in names)+b'\0')
subprocess.run(['git','add','-f','--pathspec-from-file='+str(paths),'--pathspec-file-nul'],cwd=ROOT,check=True)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
message=HERE/'commit-experimental.txt'
message.write_text('training: checkpoint isolated S15-S17 implementation\n\n'
    'Add optional damage telemetry, explicit native state serialization, immutable Python snapshots and branch lineage, '
    'and a parameterized wall family. Default entrypoints and nine protected binaries/source hashes stay unchanged.\n\n'
    'Validation: 77 CPU checks pass; isolated CUDA builds compile; the rebuilt baseline has 262 identical kernel '
    'instruction listings and training builds preserve 261 unchanged kernels. Native physical, video, performance, '
    'and snapshot T1-T8 acceptance are still pending GPU availability. This is an experimental rollback point, not adoption.\n',encoding='utf8')
subprocess.run(['git','commit','-F',str(message)],cwd=ROOT,check=True)
print(json.dumps({'commit':git('rev-parse','HEAD'),'files':len(names),'bytes':record['bytes'],
                  'gpu_validated':False,'pushed':False},indent=2))
