"""Serial normal-graph ABBA, independent real-state captures, full comparison."""
from pathlib import Path
import json
import subprocess
import sys
root=Path(__file__).resolve().parents[2];work=Path(__file__).resolve().parent
data=root/'endoscopic_tissue/outputs/joint-coarse-20261031'
perf=root/'endoscopic_tissue_backups/2026-10-30-perf-opt'
wrist=root/'endoscopic_tissue_backups/2026-10-31-wrist-opt'
results=[]
def call(label,args):
    print('[JOINT-TRIAL] '+label,flush=True)
    with (work/(label+'.log')).open('w',encoding='utf8') as log:r=subprocess.run([sys.executable,'-X','utf8',*map(str,args)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    results.append(dict(label=label,args=list(map(str,args)),exit_code=r.returncode));(work/'trial-results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    assert r.returncode==0,(label,r.returncode)
guard=json.loads((data/'strict-green/joint-diagnostics-summary.json').read_text())
assert guard['steps']==1110 and guard['accepted']>0 and all(guard[k]==0 for k in ('negative_margin','volume_violation','closing_violation','energy_violation')),guard
call('timing',[perf/'run_batch.py',work/'timing-jobs.json'])
for name in ('baseline-a1','joint-b1','joint-b2','baseline-a2'):
    report=json.loads((data/name/'report.json').read_text());rows=json.loads((data/name/'timing-steps.json').read_text())
    assert report['steps']==len(rows)==1110 and report['metrics']['completed']
call('captures',[perf/'run_batch.py',work/'capture-jobs.json'])
for name in ('baseline','joint'):
    report=json.loads((data/(name+'-capture')/'report.json').read_text());assert report['steps']==1110 and report['metrics']['completed']
    call('render-'+name,[perf/'replay_single_blas.py',data/(name+'-capture')/'replay.bin',data/(name+'-video')])
call('compare',[perf/'make_review.py',data/'baseline-video/interactions.mp4',data/'joint-video/interactions.mp4',data/'joint-review','JOINT_12_DOF'])
call('motion',[wrist/'review_motion.py',data/'baseline-capture/motion.jsonl',data/'joint-capture/motion.jsonl',data/'joint-review'])
print('[JOINT-TRIAL] complete; inspect performance and quality before adoption',flush=True)
