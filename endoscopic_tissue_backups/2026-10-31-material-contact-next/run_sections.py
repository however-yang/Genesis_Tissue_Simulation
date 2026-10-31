"""Diagnostic section splits; separate from normal CUDA Graph ABBA timing."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
DATA=ROOT/'endoscopic_tissue/outputs/material-contact-next-20261031'
profiler=ROOT/'endoscopic_tissue_backups/2026-10-28-speed/profile_sections.py'
results={}
for name,folder in [('baseline','bin-scissors'),('additive','bin-additive31-accurate'),('groups','bin-contact-groups31')]:
    output=DATA/('profile-'+name)
    assert not output.exists(), 'Preserve previous profile evidence'
    dll=ROOT/'endoscopic_tissue/native'/folder/'LiverCudaSim.dll'
    rows_path=WORK/('sections-'+name+'.json')
    args=[sys.executable,'-X','utf8',str(profiler),str(dll),str(rows_path),'--output',str(output),'--native-only','--scene','kidney','--tool','scissors']
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT)
    print('[SECTIONS] '+name,flush=True)
    with (WORK/('sections-'+name+'.log')).open('w',encoding='utf8') as log:
        proc=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert proc.returncode==0,(name,proc.returncode)
    rows=json.loads(rows_path.read_text(encoding='utf8'))
    report=json.loads((output/'report.json').read_text(encoding='utf8'))
    assert sum(r['steps'] for r in rows)==report['steps']==1110 and report['metrics']['completed']
    names=set().union(*(r['sections_ms_per_step'] for r in rows))
    results[name]=dict(steps=1110,dll_sha256=hashlib.sha256(dll.read_bytes()).hexdigest(),
        sections_ms_per_step={k:sum(r['steps']*r['sections_ms_per_step'].get(k,0) for r in rows)/1110 for k in sorted(names)})
    (WORK/'sections-summary.json').write_text(json.dumps(dict(note='Graphs disabled; includes launch gaps; nested assembly sections overlap. Do not sum all sections or use as runtime FPS.',runs=results),indent=2),encoding='utf8')

# The normal ABBA outlier is preserved; this bounded follow-up is diagnostic only.
subprocess.run([sys.executable,'-X','utf8',str(WORK/'run_slow_graph_diagnostics.py')],cwd=ROOT,check=True)
