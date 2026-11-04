"""Audit archived trials without changing production code or declaring quality passed."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess
import cv2

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'endoscopic_tissue/outputs'

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def summary(values):
    values = sorted(values)
    def percentile(q):
        p = (len(values)-1)*q
        a = int(p)
        return values[a] + (values[min(a+1,len(values)-1)]-values[a])*(p-a)
    assert values and all(math.isfinite(x) and x >= 0 for x in values)
    return dict(mean=statistics.mean(values),p95=percentile(.95),maximum=max(values))

def video(path, width):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    assert frames == 1111 and abs(fps-30)<1e-8 and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))==width, path
    # Decode beginning, middle and end; full generation already checks every input frame.
    for n in (0,420,750,1110):
        cap.set(cv2.CAP_PROP_POS_FRAMES,n)
        ok, frame=cap.read()
        assert ok and frame is not None, (path,n)
    cap.release()
    return dict(path=str(path.relative_to(ROOT)),frames=frames,fps=fps,bytes=path.stat().st_size)

baseline = read(OUT/'wrist-opt-20261031/baseline-capture/report.json')
settings=baseline['settings']
assert settings['dt']==1/30 and settings['substeps']==3 and settings['young_pa']==15000 and settings['poisson']==.25 and settings['strain_damping_s']==.04
series_specs=[
 ('wrist_inline','wrist-opt-20261031',['baseline-a1','wrist-b1','wrist-b2','baseline-a2']),
 ('wrist_dedicated','wrist-opt-20261031',['dedicated-base-a1','dedicated-b1','dedicated-b2','dedicated-base-a2']),
 ('shared_front','contact-front-20261031',['baseline-a1','shared-b1','shared-b2','baseline-a2']),
 ('indexed_batch','contact-indexed-20261031',['baseline-a1','indexed-b1','indexed-b2','baseline-a2']),
 ('spread_batch','contact-spread-20261031',['baseline-a1','spread-b1','spread-b2','baseline-a2'])]
result=dict(base_commit='39e5ebc655fde6f77259783fecaa43bda78de2d0',production_adopted=False,settings=settings,
 timing_scope='Normal CUDA graphs, no live display; includes control/readback/recorder submission. Excludes stage boundaries and final draining; recording/review runs are separate.',series=[],captures=[],frozen_inputs=[])
for label,folder,names in series_specs:
    runs=[]
    for name in names:
        path=OUT/folder/name
        rows,report,meta=read(path/'timing-steps.json'),read(path/'report.json'),read(path/'timing-summary.json')['metadata']
        assert len(rows)==report['steps']==1110 and report['metrics']['completed'], path
        assert [r['step'] for r in rows]==list(range(1,1111))
        assert report['settings']==settings and report['grid_spacing_cm']==.1, path
        assert (meta['active_nodes'],meta['material_nodes'],meta['movable_active_nodes'])==(121260,102710,118660)
        dll=Path(meta['dll'])
        assert sha(dll)==meta['dll_sha256']==report['dll_sha256'], dll
        stages={s:{k:summary([r['calls_ms']['LCS_Step'] if k=='core_ms' else r[k] for r in rows if r['stage']==s]) for k in ('core_ms','pipeline_ms')} for s in dict.fromkeys(r['stage'] for r in rows)}
        runs.append(dict(name=name,path=str(path.relative_to(ROOT)),dll_sha256=meta['dll_sha256'],core_ms=summary([r['calls_ms']['LCS_Step'] for r in rows]),pipeline_ms=summary([r['pipeline_ms'] for r in rows]),stages=stages,quality_passed=report['metrics']['acceptance_passed'],stray_skin_cuts=report['metrics']['stray_skin_cuts']))
    item=dict(name=label,runs=runs)
    if label=='indexed_batch':
        item['decision']='No acceleration claim: baseline-a2 develops an unexplained slow interval; preserve all four runs, do not average it into a speedup.'
    else:
        item['means']={k:dict(baseline=(runs[0][k]['mean']+runs[3][k]['mean'])/2,candidate=(runs[1][k]['mean']+runs[2][k]['mean'])/2) for k in ('core_ms','pipeline_ms')}
        for value in item['means'].values(): value['change_percent']=100*(value['candidate']/value['baseline']-1)
        item['decision']='Not adopted: no demonstrated end-to-end acceleration or adoption-level interaction equivalence.'
    result['series'].append(item)

capture_specs=[('baseline','wrist-opt-20261031','baseline'),('wrist_inline','wrist-opt-20261031','initial'),('wrist_dedicated','wrist-opt-20261031','dedicated'),('shared_front','contact-front-20261031','shared'),('indexed_batch','contact-indexed-20261031','indexed'),('spread_batch','contact-spread-20261031','spread')]
for label,folder,prefix in capture_specs:
    path=OUT/folder/(prefix+'-capture')
    report=read(path/'report.json')
    rows=[json.loads(line) for line in (path/'motion.jsonl').read_text().splitlines()]
    assert report['steps']==1110 and report['metrics']['completed'] and report['settings']==settings
    assert [r['step'] for r in rows]==list(range(1111))
    assert sha(Path(report['dll']))==report['dll_sha256']
    item=dict(name=label,report=str((path/'report.json').relative_to(ROOT)),dll_sha256=report['dll_sha256'],quality_passed=report['metrics']['acceptance_passed'],metrics=report['metrics'],video=video(OUT/folder/(prefix+'-video')/'interactions.mp4',1280))
    audits=read(path/'surface-audit.json')
    item['topology_audit']={key:max(a[key] for a in audits) for key in ('recorded_nonmanifold_edges','positional_nonmanifold_edges','degenerate_triangles','duplicate_triangles')}
    if prefix!='baseline':
        review=OUT/folder/(prefix+'-review')
        manifest=read(review/'manifest.json')
        assert manifest['frames']==1111 and manifest['fps']==30
        item['comparison']=video(review/'comparison.mp4',2560)
        motion=read(review/'motion-comparison.json')
        assert motion['frames']==1111
        item['motion']=motion['stages']
    result['captures'].append(item)

for folder,name,count,scope in [('wrist-opt-20261031','dedicated-equivalence',74,'non-wrist only'),('contact-front-20261031','shared-equivalence',74,'all tool paths'),('contact-indexed-20261031','indexed-equivalence',270,'all tool paths; dense steps600..810'),('contact-spread-20261031','spread-equivalence',270,'all tool paths; dense steps600..810')]:
    path=OUT/folder/name/'contact-equivalence.json'
    rows=read(path)
    assert len(rows)==count and rows[-1]['step']==1110, path
    assert all(r['status']==0 and all(r[k]==0 for k in ('different_counts','different_records','different_degrees','different_members')) for r in rows),path
    result['frozen_inputs'].append(dict(path=str(path.relative_to(ROOT)),scope=scope,states=len(rows),queries=sum(r['queries'] for r in rows),triangle_queries=sum(r['triangle_queries'] for r in rows),differences=0))

for name,expected in [('failing',1),('indexed',0),('spread',0)]:
    obj=read(WORK/('replay-'+name+'.json'))
    assert obj['status']==0 and all(r['different_counts']==0 and r['different_records']==expected for r in obj['paths'])
    result.setdefault('snapshot_replays',{})[name]=obj
rows=read(OUT/'contact-spread-20261031/graph-baseline-1/solver-graphs.json')
assert len(rows)==1110 and sum(r['loaded']+r['free']+r['recovery'] for r in rows)==3330
result['anomaly_recheck']=dict(steps=len(rows),loaded=sum(r['loaded'] for r in rows),free=sum(r['free'] for r in rows),recovery=sum(r['recovery'] for r in rows),profiled=sum(r['profiled'] for r in rows),maximum_native_ms=max(r['native_ms'] for r in rows),conclusion='Prior slow interval was not reproduced; cause remains unconfirmed.')

hashes={'bin-scissors':'0d09f56a1f4acb813424cd1dee3233d89ce7ef5c0d45e64f079c7d7387912225','bin':'2632a926b385d9c9a81d5f7d7ebb75fe0501917a9cf83d0c19d3a29c62af961b'}
for folder,expected in hashes.items(): assert sha(ROOT/'endoscopic_tissue/native'/folder/'LiverCudaSim.dll')==expected
subprocess.run(['git','diff','--exit-code','--','endoscopic_tissue/native/vendor/cuda'],cwd=ROOT,check=True,capture_output=True)
patches=[ROOT/'endoscopic_tissue_backups'/d/p for d,p in [('2026-10-31-wrist-opt','initial-wrist.patch'),('2026-10-31-wrist-opt','dedicated-wrist.patch'),('2026-10-31-contact-front','block-shared.patch'),('2026-10-31-contact-batch','batch-candidates.patch'),('2026-10-31-contact-batch','indexed-candidates.patch'),('2026-10-31-contact-batch','spread-candidates.patch'),('2026-10-31-contact-batch','graph-counter.patch')]]
for patch in patches: subprocess.run(['git','apply','--check',str(patch)],cwd=ROOT,check=True,capture_output=True)
result['protected_dll_sha256']=hashes
result['patches']={str(p.relative_to(ROOT)):sha(p) for p in patches}
(WORK/'evidence-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(timing_runs=20,captures=6,comparisons=5,frozen_inputs=result['frozen_inputs'],patches_checked=len(patches),baseline_preserved=True,production_adopted=False),ensure_ascii=False,indent=2))
