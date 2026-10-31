"""Bounded reproduction of the observed post-cut slowdown; never normal ABBA data."""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
DATA=ROOT/'endoscopic_tissue/outputs/material-contact-next-20261031/groups'
DLL=ROOT/'endoscopic_tissue/native/bin-contact-groups31-graphs/LiverCudaSim.dll'
results=[]
for repeat in range(1,4):
    output=DATA/('graph-diagnostic-'+str(repeat));assert not output.exists()
    print('[GRAPH-DIAG] '+str(repeat),flush=True)
    args=[sys.executable,'-X','utf8',str(WORK/'measure_solver_graphs.py'),str(output),'--native-only','--scene','kidney','--tool','scissors','--dll',str(DLL)]
    with (WORK/('graph-diagnostic-'+str(repeat)+'.log')).open('w',encoding='utf8') as log:
        proc=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert proc.returncode in (0,1)
    report=json.loads((output/'report.json').read_text(encoding='utf8'))
    rows=json.loads((output/'solver-graphs.json').read_text(encoding='utf8'))
    assert report['steps']==len(rows)==1110 and report['metrics']['completed']
    assert all(r['loaded']+r['free']+r['recovery']==3 for r in rows)
    stats=dict(repeat=repeat,path=str(output),dll_sha256=report['dll_sha256'],steps=len(rows),exit_code=proc.returncode,
        graphs={k:sum(r[k] for r in rows) for k in ('loaded','free','recovery')},
        slow_unprofiled_steps=[r for r in rows if not r['profiled'] and r['native_ms']>100],
        maximum_unprofiled_ms=max(r['native_ms'] for r in rows if not r['profiled']),
        profile_triggered=(output/'slow-trigger.json').exists())
    results.append(stats)
    (WORK/'graph-diagnostics-summary.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    if stats['profile_triggered']:break
print('[GRAPH-DIAG] done; no reproduction does not explain the original slow run',flush=True)
