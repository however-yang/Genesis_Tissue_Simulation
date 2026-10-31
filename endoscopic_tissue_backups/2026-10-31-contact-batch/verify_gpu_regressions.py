"""Repeat the saved red/green contact regression and guarded analytic geometry test."""
from pathlib import Path
import json
import subprocess
import sys
root=Path(__file__).resolve().parents[2]
work=Path(__file__).resolve().parent
wrist=root/'endoscopic_tissue_backups/2026-10-31-wrist-opt'
snapshot=root/'endoscopic_tissue/outputs/contact-batch-20261031/debug-dense-1/failure-contact.bin'
results=[]
for name,folder,expected in [('failing','bin-contact-batch31-debug',1),('indexed','bin-contact-indexed31-probe',0),('spread','bin-contact-spread31-probe',0)]:
    path=work/('verify-replay-'+name+'.json')
    result=subprocess.run([sys.executable,'-X','utf8',str(work/'replay_contact_snapshot.py'),str(root/'endoscopic_tissue/native'/folder/'LiverCudaSim.dll'),str(snapshot),str(path)],cwd=root,capture_output=True,text=True,encoding='utf8')
    (work/('verify-replay-'+name+'.log')).write_text(result.stdout+result.stderr,encoding='utf8')
    obj=json.loads(path.read_text())
    assert result.returncode==expected and obj['status']==0 and all(r['different_counts']==0 and r['different_records']==expected for r in obj['paths']), (name,result.returncode,obj)
    results.append(dict(name=name,expected_exit=expected,actual_exit=result.returncode,details=obj))
    print(name+': expected regression result confirmed',flush=True)
result=subprocess.run([sys.executable,'-X','utf8',str(wrist/'check_wrist_geometry.py'),str(root/'endoscopic_tissue/native/bin-wrist-geometry31/LiverCudaSim.dll'),str(work/'verify-geometry.json')],cwd=root,capture_output=True,text=True,encoding='utf8')
(work/'verify-geometry.log').write_text(result.stdout+result.stderr,encoding='utf8')
assert result.returncode==0,result.stdout+result.stderr
geometry=json.loads((work/'verify-geometry.json').read_text())
assert geometry['cases']==1698 and geometry['required_fallback_cases']==42
(work/'gpu-regression-verification.json').write_text(json.dumps(dict(snapshot_replays=results,geometry=geometry),indent=2),encoding='utf8')
print('geometry: 1698 cases confirmed',flush=True)
