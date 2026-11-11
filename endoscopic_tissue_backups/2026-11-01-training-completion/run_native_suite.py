"""Sequential, bounded jobs; never overlaps physics, rendering, or native timing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
OUT = ROOT/'endoscopic_tissue/outputs/training-completion-20261101'
OLD = ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes'
R20 = P/'package/artifacts/release-r20-support-cache/bin/LiverCudaSim.dll'
BASE = ROOT/'endoscopic_tissue/native/bin-contact-groups31/LiverCudaSim.dll'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite',choices=['timing','training','videos','wall-friction'],required=True)
    args = parser.parse_args()
    commands=[]
    def add(name,script,*flags):
        commands.append((name,[sys.executable,'-X','utf8',str(script),*map(str,flags)]))
    if args.suite == 'timing':
        for i,dll in enumerate([BASE,R20,R20,BASE]):
            add(f'timing-{i}',OLD/'time_full_interaction.py','--case','kidney-reference','--dll',dll,
                '--output',OUT/f'timing-{i}','--end-step',1215)
        for i in range(2):
            add(f'timing-r20-free-{i}',OLD/'time_full_interaction.py','--case','kidney-reference','--dll',R20,
                '--output',OUT/f'timing-r20-free-{i}')
    elif args.suite == 'training':
        for name,spec in [('policy-pilot','policy-pilot.json'),('frozen-evaluation','frozen-evaluation.json'),
                          ('friction-pilot','friction-pilot.json'),('vision-pilot','vision-pilot.json')]:
            commands.append((name,[sys.executable,'-X','utf8','-m','endoscopic_tissue.training.experiment',
                                   '--spec',str(P/spec),'--output',str(OUT/name)]))
    elif args.suite == 'wall-friction':
        for name,case in [('r20-wall','wall-tip-completion-wide'),('r20-friction','grasp-central')]:
            add(name,P/'audit_r20.py','--case',case,'--dll',R20,'--output',OUT/name,'--capture','--archive-capture')
    elif args.suite == 'videos':
        for name in ['r20-kidney','r20-wall','r20-friction']:
            add(name,ROOT/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation/render_review.py',OUT/name)
    results=[]
    for name,command in commands:
        print('START',name,flush=True)
        started=time.perf_counter()
        with (P/(name+'.log')).open('w',encoding='utf8') as log:
            try:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=7200,
                                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                code=result.returncode
            except subprocess.TimeoutExpired:
                code='timeout'
        results.append(dict(name=name,command=command,exit_code=code,wall_s=time.perf_counter()-started))
        (P/(args.suite+'-jobs.json')).write_text(json.dumps(results,indent=2))
        print('END',name,code,flush=True)
        if args.suite != 'training' and code != 0:
            raise RuntimeError('Job failed; evidence retained: '+name)


if __name__ == '__main__':main()
