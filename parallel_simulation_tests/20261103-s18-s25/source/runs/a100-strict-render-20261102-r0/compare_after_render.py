import pathlib,json,time,subprocess,os
b=pathlib.Path(__file__).resolve().parent;r=b.parents[1];release=r/'releases/0643e10991cf'
while True:
 s=json.loads((b/'status.json').read_text())
 if s['phase']=='failed':raise SystemExit('Rendering failed; comparison not started')
 if s['phase']=='complete':break
 time.sleep(5)
env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(release),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
args=[str(r/'envs/genesis-py311-cu125-v1/bin/python'),'-B','-u','endoscopic_tissue_backups/2026-11-02-acceptance-fixes/compare_videos.py','--name','a100-kidney-partial-comparison','--baseline','a100-kidney-base-video-r0','--candidate','a100-kidney-snapshot-video-r0']
q=subprocess.run(args,cwd=release,env=env)
(b/'comparison-exit.json').write_text(json.dumps({'exit_code':q.returncode,'full_scenario_completed':False,'frames':1247,'note':'Both scenarios failed at step 1247. Video contains states 0 through 1246 only; no truncation.'}))
raise SystemExit(q.returncode)
