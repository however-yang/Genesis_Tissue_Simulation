"""Run one scheduling change independently, after the first candidate completes."""
from pathlib import Path
import json
import subprocess
import sys
root=Path(__file__).resolve().parents[2];work=Path(__file__).resolve().parent
data=root/'endoscopic_tissue/outputs/joint-gated-20261031';first=root/'endoscopic_tissue/outputs/joint-coarse-20261031'
perf=root/'endoscopic_tissue_backups/2026-10-30-perf-opt';wrist=root/'endoscopic_tissue_backups/2026-10-31-wrist-opt'
probe=root/'endoscopic_tissue/native/bin-joint-coarse31-gated-probe/LiverCudaSim.dll';dll=root/'endoscopic_tissue/native/bin-joint-coarse31-gated/LiverCudaSim.dll'
results=[]
def call(label,args,allowed=(0,)):
 print('[JOINT-GATED] '+label,flush=True)
 with (work/('gated-'+label+'.log')).open('w',encoding='utf8') as log:r=subprocess.run([sys.executable,'-X','utf8',*map(str,args)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
 results.append(dict(label=label,args=list(map(str,args)),exit_code=r.returncode));(work/'gated-trial-results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
 assert r.returncode in allowed,(label,r.returncode)
assert (first/'joint-review/comparison.mp4').is_file(),'Wait for first complete GPU sequence'
call('algebra',[work/'check_joint_algebra.py',probe,work/'gated-algebra-results.json'])
call('rounding',[work/'check_joint_rounding.py',probe,work/'gated-rounding-results.json'])
call('diagnostic',[work/'check_joint_dynamics_strict.py',data/'strict-green','--tool','scissors','--dll',probe],(0,1))
stats=json.loads((data/'strict-green/joint-diagnostics-summary.json').read_text())
assert stats['steps']==1110 and stats['accepted']>0 and all(stats[k]==0 for k in ('negative_margin','volume_violation','closing_violation','energy_violation')),stats
call('timing',[perf/'run_batch.py',work/'gated-timing-jobs.json'])
for name in ('baseline-a1','gated-b1','gated-b2','baseline-a2'):
 report=json.loads((data/name/'report.json').read_text());rows=json.loads((data/name/'timing-steps.json').read_text())
 assert report['steps']==len(rows)==1110 and report['metrics']['completed']
call('capture',[wrist/'capture_motion.py',data/'gated-capture','--scene','kidney','--tool','scissors','--dll',dll],(0,1))
report=json.loads((data/'gated-capture/report.json').read_text());assert report['steps']==1110 and report['metrics']['completed']
call('render',[perf/'replay_single_blas.py',data/'gated-capture/replay.bin',data/'gated-video'])
call('compare',[perf/'make_review.py',first/'baseline-video/interactions.mp4',data/'gated-video/interactions.mp4',data/'gated-review','JOINT_GATED'])
call('motion',[wrist/'review_motion.py',first/'baseline-capture/motion.jsonl',data/'gated-capture/motion.jsonl',data/'gated-review'])
print('[JOINT-GATED] complete; quality and speed require review',flush=True)
