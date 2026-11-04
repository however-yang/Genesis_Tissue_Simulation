"""Verify this isolated experiment's local evidence; quality failures stay failures."""
from pathlib import Path
import ast
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import cv2
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
DATA=ROOT/'endoscopic_tissue/outputs/material-contact-next-20261031'
BASE='fbcaf71dc0eec30e00b55d46bf5ae0eddac271b3'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return str(p.relative_to(ROOT)).replace(chr(92),'/')
def summary(values):
    values=sorted(values);assert values and all(math.isfinite(x) and x>=0 for x in values)
    p=(len(values)-1)*.95;i=int(p)
    return dict(mean=statistics.mean(values),p95=values[i]+(values[min(i+1,len(values)-1)]-values[i])*(p-i),maximum=max(values),steps_over_33hz_budget=sum(x>1000/33 for x in values))
def video(path,width):
    cap=cv2.VideoCapture(str(path));assert cap.isOpened(),path
    fps=cap.get(cv2.CAP_PROP_FPS);assert abs(fps-30)<1e-8 and cap.get(cv2.CAP_PROP_FRAME_COUNT)==1111,path
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH)==width and cap.get(cv2.CAP_PROP_FRAME_HEIGHT)==(1008 if width==2560 else 960),path
    decoded=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        assert frame is not None and frame.shape[1]==width;decoded+=1
    cap.release();assert decoded==1111,(path,decoded)
    return dict(path=rel(path),frames=decoded,fps=fps,bytes=path.stat().st_size,sha256=sha(path),note='Simulation-time playback, not execution FPS.')
def kernel(source,name):
    start=source.index('__global__ void '+name+'(');body=source.index('{',start)
    masked=re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',lambda m:' '*len(m.group()),source)
    depth=0
    for i in range(body,len(source)):
        depth+=(masked[i]=='{')-(masked[i]=='}')
        if not depth:return source[start:i+1]
    raise AssertionError(name)
build=read(WORK/'build-manifest.json');assert build['base_commit']==BASE
for name,digest in build['dll'].items():assert sha(ROOT/'endoscopic_tissue/native'/name/'LiverCudaSim.dll')==digest,name
for field in ('sources','initial_sources'):
    for name,digest in build[field].items():assert sha(WORK/name)==digest,name
for p in WORK.glob('*.py'):ast.parse(p.read_text(encoding='utf8'),filename=str(p))
assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='dev'
subprocess.run(['git','diff','--exit-code',BASE,'--','endoscopic_tissue/native'],cwd=ROOT,check=True,capture_output=True)
patches=['initial_additive.patch','initial_additive-probe.patch','additive.patch','additive-probe.patch','contact-groups.patch','contact-groups-probe.patch','contact-groups-graphs.patch']
for name in patches:subprocess.run(['git','apply','--check',str(WORK/name)],cwd=ROOT,check=True,capture_output=True)
reference=(WORK/'reference_find_surface.cuh').read_text(encoding='utf8')
original=(ROOT/'endoscopic_tissue/native/vendor/cuda/contact/surface_contact.cuh').read_text(encoding='utf8')
for name in ('triangleWitness','traceTriangleBox','k_FindSurface'):reference=reference.replace(name+'Reference',name)
assert kernel(reference,'k_FindSurface')==kernel(original,'k_FindSurface'),'Reference query drifted'
if '--sources-only' in sys.argv:
    print(json.dumps(dict(default_sources_preserved=True,builds_verified=len(build['dll']),patches_checked=len(patches),reference_query_matches=True)))
    raise SystemExit(0)
diagonal=read(WORK/'diagonal-regression.json');assert diagonal['cases']==1038 and diagonal['new_max_relative_error']<2e-6 and diagonal['old_max_relative_error']>1e-4
red=read(WORK/'additive-error-localization.json');assert red['relative_error']>1e-4
failure=read(WORK/'failure-snapshot-manifest.json')
assert sha(Path(failure['path']))==failure['sha256'] and Path(failure['path']).stat().st_size==failure['bytes']
checks=read(DATA/'additive/diagnostic-a3/additive-checks.json')
assert len(checks)==10 and checks[-1]['step']==1110
for item in checks:
    assert max(item['relative_l2_error'])<1e-4 and item['normalized_symmetry_error']<1e-5
    assert min(item['quadratic_forms'])>0 and item['fixed_output_nonzero']==0 and item['cg_nodes']>100000
contact=read(DATA/'groups/diagnostic/contact-equivalence.json')
assert len(contact)==270 and contact[-1]['step']==1110
assert set(range(600,811)).issubset({r['step'] for r in contact})
for row in contact:
    assert row['status']==0 and not any(row[k] for k in ('different_counts','different_records','different_degrees','different_members','unsafe_group_rejections'))
assert sum(r['group_rejected_queries'] for r in contact)>0
settings=read(DATA/'additive/baseline-a1/report.json')['settings']
assert (settings['dt'],settings['substeps'],settings['young_pa'],settings['poisson'],settings['strain_damping_s'])==(1/30,3,15000,.25,.04)
result=dict(base_commit=BASE,production_adopted=False,settings=settings,diagonal_regression=diagonal,
    additive_frozen_checks=dict(states=10,max_relative_error=max(max(x['relative_l2_error']) for x in checks),max_symmetry_error=max(x['normalized_symmetry_error'] for x in checks),path=rel(DATA/'additive/diagnostic-a3/additive-checks.json')),
    contact_frozen_checks=dict(states=270,triangle_queries=sum(r['triangle_queries'] for r in contact),valid_records=sum(r['valid_records'] for r in contact),group_rejected_queries=sum(r['group_rejected_queries'] for r in contact),preload_bypasses=sum(r['preload_bypasses'] for r in contact),unsafe_rejections=0,record_differences=0),
    timing_scope='Normal CUDA Graphs, no display; pipeline includes control/readback/recorder submission and possible backpressure. Stage boundaries and final drain excluded. Probes, captures, rendering and section timing are separate.',series=[],captures=[])
for label in ('additive','groups'):
    directory=DATA/label;runs=[]
    for name in ('baseline-a1',label+'-b1',label+'-b2','baseline-a2'):
        path=directory/name;rows=read(path/'timing-steps.json');report=read(path/'report.json');meta=read(path/'timing-summary.json')['metadata']
        assert len(rows)==report['steps']==1110 and report['metrics']['completed'] and report['settings']==settings
        assert [r['step'] for r in rows]==list(range(1,1111)) and report['grid_spacing_cm']==.1
        assert (meta['total_grid_nodes'],meta['active_nodes'],meta['material_nodes'],meta['movable_active_nodes'])==(354200,121260,102710,118660)
        assert sha(Path(meta['dll']))==meta['dll_sha256']==report['dll_sha256']
        stages={s:{k:summary([r['calls_ms']['LCS_Step'] if k=='core_ms' else r[k] for r in rows if r['stage']==s]) for k in ('core_ms','pipeline_ms')} for s in dict.fromkeys(r['stage'] for r in rows)}
        runs.append(dict(name=name,path=rel(path),dll_sha256=meta['dll_sha256'],rows_sha256=sha(path/'timing-steps.json'),report_sha256=sha(path/'report.json'),core_ms=summary([r['calls_ms']['LCS_Step'] for r in rows]),pipeline_ms=summary([r['pipeline_ms'] for r in rows]),stages=stages,quality_passed=report['metrics']['acceptance_passed'],metrics=report['metrics']))
    means={k:dict(baseline=(runs[0][k]['mean']+runs[3][k]['mean'])/2,candidate=(runs[1][k]['mean']+runs[2][k]['mean'])/2) for k in ('core_ms','pipeline_ms')}
    for m in means.values():m['change_percent']=100*(m['candidate']/m['baseline']-1);m['candidate_hz']=1000/m['candidate']
    result['series'].append(dict(name=label,runs=runs,means=means))
    for prefix in ('baseline',label):
        path=directory/(prefix+'-capture');report=read(path/'report.json');motion=[json.loads(line) for line in (path/'motion.jsonl').read_text(encoding='utf8').splitlines()]
        assert [r['step'] for r in motion]==list(range(1111)) and report['steps']==1110 and report['metrics']['completed'] and report['settings']==settings
        assert sha(Path(report['dll']))==report['dll_sha256']
        loads={}
        for stage in dict.fromkeys(r['stage'] for r in motion):
            group=[r for r in motion if r['stage']==stage]
            loads[stage]={k:[max(r[k][side] for r in group) for side in (0,1)] for k in ('jaw_load_n','scissors_load_n')}
        audits=read(path/'surface-audit.json')
        item=dict(name=label+'/'+prefix,path=rel(path),dll_sha256=report['dll_sha256'],quality_passed=report['metrics']['acceptance_passed'],metrics=report['metrics'],motion_sha256=sha(path/'motion.jsonl'),report_sha256=sha(path/'report.json'),peak_loads_by_stage=loads,topology_audit={key:max(a[key] for a in audits) for key in ('recorded_nonmanifold_edges','positional_nonmanifold_edges','degenerate_triangles','duplicate_triangles')},video=video(directory/(prefix+'-video/interactions.mp4'),1280))
        if prefix==label:
            review=directory/(label+'-review');manifest=read(review/'manifest.json');comparison=read(review/'motion-comparison.json')
            assert manifest['frames']==comparison['frames']==1111 and manifest['fps']==30
            item['comparison']=video(review/'comparison.mp4',2560);item['motion_comparison']=dict(path=rel(review/'motion-comparison.json'),sha256=sha(review/'motion-comparison.json'))
        result['captures'].append(item)
profiles=read(WORK/'sections-summary.json')
assert set(profiles['runs'])=={'baseline','additive','groups'}
for name,item in profiles['runs'].items():
    rows=read(WORK/('sections-'+name+'.json'));report=read(DATA/('profile-'+name)/'report.json')
    assert sum(r['steps'] for r in rows)==report['steps']==item['steps']==1110 and report['metrics']['completed'] and report['settings']==settings
    assert sha(Path(report['dll']))==report['dll_sha256']==item['dll_sha256']
    for key,value in item['sections_ms_per_step'].items():assert abs(sum(r['steps']*r['sections_ms_per_step'].get(key,0) for r in rows)/1110-value)<1e-10
    item['rows_sha256']=sha(WORK/('sections-'+name+'.json'))
graph_runs=read(WORK/'graph-diagnostics-summary.json')
assert 1<=len(graph_runs)<=3
for trial in graph_runs:
    directory=Path(trial['path']);rows=read(directory/'solver-graphs.json');report=read(directory/'report.json')
    assert len(rows)==trial['steps']==report['steps']==1110 and report['metrics']['completed']
    assert sha(Path(report['dll']))==report['dll_sha256']==trial['dll_sha256']
    assert all(r['loaded']+r['free']+r['recovery']==3 for r in rows)
    assert trial['graphs']=={k:sum(r[k] for r in rows) for k in ('loaded','free','recovery')}
    if trial['profile_triggered']:assert (directory/'slow-sections.txt').is_file() and (directory/'slow-gpu-state.txt').is_file()
    trial['rows_sha256']=sha(directory/'solver-graphs.json')
result['slow_graph_diagnostics']=graph_runs
result['section_diagnostics']=profiles
result['build_manifest_sha256']=sha(WORK/'build-manifest.json');result['patches']={p:sha(WORK/p) for p in patches}
result['protected_dll_sha256']={name:build['dll'][name] for name in ('bin-scissors','bin')}
result['initial_additive_failure']=dict(snapshot=failure,localization=red)
result['visual_reviews']={}
for label in ('additive','groups'):
    notes=read(WORK/('visual-review-'+label+'.json'));assert notes['candidate']==label and len(notes['inspected'])==7
    for filename in notes['inspected']:assert (DATA/label/(label+'-review')/filename).is_file()
    result['visual_reviews'][label]=notes
playback=read(WORK/'playback-videos.json');assert len(playback)==2
for item in playback:
    assert sha(Path(item['source']))==item['source_sha256'] and sha(Path(item['path']))==item['sha256']
    assert item['frames']==1111 and item['fps']==30 and Path(item['path']).stat().st_size==item['bytes']
    video(Path(item['path']),2560)
result['playback_videos']=playback
(WORK/'evidence-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(timing_runs=8,captures=4,comparisons=2,decoded_original_video_frames=6666,decoded_playback_video_frames=2222,material_states=10,contact_states=270,section_runs=3,patches_checked=len(patches),default_preserved=True,production_adopted=False,timings=[dict(name=s['name'],means=s['means']) for s in result['series']]),indent=2))
