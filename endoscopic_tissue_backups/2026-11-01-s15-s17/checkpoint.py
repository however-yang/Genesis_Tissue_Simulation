"""Explicit small checkpoint before native work; never pushes or stages history."""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from endoscopic_tissue.training.provenance import sha256
baseline=json.loads((HERE/'baseline.json').read_text(encoding='utf8'))
for path,digest in baseline['protected'].items():
    if sha256(ROOT/path)!=digest:raise RuntimeError('Protected file changed: '+path)
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
paths=[ROOT/'endoscopic_tissue/training'/p for p in
       ('env.py','experiment.py','provenance.py','physics.py','native.py','audit_physics.py','test_physics.py')]
paths += [ROOT/'endoscopic_tissue/docs'/p for p in
          ('training-s15-s17-design-2026-11-01.md','training-s15-s17-results-2026-11-01.md')]
paths += [p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json')]
paths += [HERE/'.gitignore']
out=ROOT/'endoscopic_tissue/outputs/s15-s17-20261101'
paths += [out/'cpu-tests.json',out/'cpu-tests.txt',out/'parameters/summary.json']
paths += list((out/'parameters').glob('*/result.json'))
paths += list((out/'parameters').glob('*/context.json'))
paths += list((out/'audits').glob('*.json'))
names=sorted({p.relative_to(ROOT).as_posix() for p in paths})
selection=HERE/'selected-paths.nul'
selection.write_bytes(b'\0'.join(n.encode('utf8') for n in names)+b'\0')
print(json.dumps({'files':len(names),'bytes':sum((ROOT/n).stat().st_size for n in names)},indent=2))
subprocess.run(['git','add','-f','--pathspec-from-file='+str(selection),'--pathspec-file-nul'],cwd=ROOT,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode('utf8').split('\0')
if set(filter(None,staged))!=set(names):raise RuntimeError('Unexpected staged files; review before committing')
message=HERE/'commit-message.txt'
message.write_text('training: add opt-in physics overrides and truthful provenance\n\n'
    'Checkpoint before S16/S15 native extensions. Keep all nine default files and pinned DLLs unchanged.\n'
    'Validate strict parameters, actual friction coefficients, full Settings comparisons and read-only historical audits.\n'
    '56 CPU tests and 16 native parameter preparations pass. Full default video/performance comparison remains pending; '
    'no physics-domain approval and no remote push. The legacy core-break kernel is not executed in the unified path.\n',encoding='utf8')
subprocess.run(['git','commit','--file',str(message)],cwd=ROOT,check=True)
