"""Verify campaign provenance and artifact integrity, NOT simulation acceptance.

Failed interaction tests remain failed. A successful exit only verifies that the
reported failures/successes, complete recorded trajectories, and videos are intact.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import sys
import cv2
from validate import fixture

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASELINE='f61132ba50d3600cc248a12af0a9a6d3922739aa'
VIDEOS={'kidney-reference','grasp-gentle','grasp-central','kidney-one-snip',
    'grasp-friction-reference','wall-reference','flap-reference','wall-grasp-confirmed','wall-repeat-recorded'}
SCRIPTS=['validate.py','render_review.py','time_continuous.py','inspect_cuts.py',
    'analyze_release.py','assess_grasps.py','summarize.py','audit_failures.py',
    'inspect_wall_bridges.py','audit_campaign.py','verify_artifacts.py']


def read(path):
    def reject(value):raise ValueError(f'Nonfinite JSON token {value} in {path}')
    return json.loads(path.read_text(encoding='utf8'),parse_constant=reject)


def canonical(value):return json.loads(json.dumps(value,allow_nan=False))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'endoscopic_tissue/outputs/grasp-cut-validation-20261031')
    a=p.parse_args()
    checks=[];valid=[];excluded=[];timings=[];videos=[]
    for path in sorted(a.root.glob('*/report.json')):
        report=read(path)
        if 'metadata' not in report:
            rows=read(path.parent/'steps.json')
            assert len(rows)==report['steps']
            assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
            assert sha(Path(report['dll']))==report['dll_sha256']
            timings.append(path.parent.name)
            continue
        m=report['metadata'];manifest=read(path.parent/'manifest.json')
        assert m==manifest,(path,'Manifest/report differ')
        if m.get('fixture_valid') is False:
            assert m.get('exclusion_reason') and m.get('superseded_by')
            excluded.append(path.parent.name);continue
        scene_name,scene,settings,stages,kind=fixture(m['case'])
        assert m['scene']==scene_name and m['kind']==kind
        assert m['settings']==canonical(asdict(settings)),(path,'Settings differ from fixture')
        assert m['stages']==canonical([asdict(s) for s in stages]),(path,'Commands differ from fixture')
        assert sha(Path(m['dll']))==m['dll_sha256']
        expected=sum(s.steps for s in stages)
        assert expected==report['expected_steps']
        rows=[]
        for line in (path.parent/'trajectory.jsonl').read_text(encoding='utf8').splitlines():
            rows.append(json.loads(line,parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v))))
        assert len(rows)==report['steps']
        assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
        assert report['completed']==(len(rows)==expected)
        if not report['completed']:
            assert any(e['stage']=='fatal' for e in report['diagnostic_errors'])
        for s in report['stages']:
            assert rows[s['end']['step']-1]==s['end']
            assert (path.parent/(s['stage']+'.npz')).exists()
        if m['settings']['contact_model']==1:
            assert all(r['grip_points']==0 for r in rows),(path,'Unexpected auxiliary anchors')
        if m['capture']:
            verification=read(path.parent/'review/video-verification.json')
            movie=path.parent/'review/interactions-h264.mp4'
            assert verification['frames']==verification['decoded']==verification['expected_frames']==len(rows)+1
            assert verification['completed_scenario']==report['completed']
            assert sha(movie)==verification['movie_sha256']
            cap=cv2.VideoCapture(str(movie))
            assert abs(cap.get(cv2.CAP_PROP_FPS)-1/settings.dt)<.01
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(rows)+1
            for frame_number in (0,len(rows)):
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
                ok,frame=cap.read()
                assert ok and frame.shape[:2]==(verification['height'],verification['width'])
            cap.release()
            videos.append(dict(run=path.parent.name,**verification))
        valid.append(dict(run=path.parent.name,case=m['case'],steps=len(rows),
            completed_scenario=report['completed'],contact_model=m['settings']['contact_model'],
            original_acceptance=report['original_metrics']['acceptance_passed'],
            original_acceptance_applicable=kind=='cut' and any(s['stage']=='hold' for s in report['stages']),
            report_sha256=sha(path),trajectory_sha256=sha(path.parent/'trajectory.jsonl')))
    assert len(valid)==30 and len({r['case'] for r in valid})==23
    assert excluded==['grasp-wall'] and len(timings)==4
    assert {v['run'] for v in videos}==VIDEOS
    evidence=HERE/'evidence'
    if (evidence/'SHA256SUMS.json').exists():
        sums=read(evidence/'SHA256SUMS.json')
        for name,wanted in sums['files'].items():
            data=(evidence/name).read_bytes()
            if name.endswith(('.json','.md')):data=data.replace(b'\r\n',b'\n')
            assert hashlib.sha256(data).hexdigest()==wanted,(name,'Packaged evidence hash mismatch')
        for item in read(evidence/'functional-results.json'):
            assert item['report']==read(a.root/item['run']/'report.json')
    protected=read(HERE/'protected-files-before.json')
    after={file:sha(ROOT/file) for file in protected}
    assert after==protected,'A protected solver/default file changed'
    changed=subprocess.check_output(['git','diff','--name-only',BASELINE,'--','endoscopic_tissue'],cwd=ROOT,text=True).splitlines()
    assert all(x=='endoscopic_tissue/docs/grasp-cut-validation-2026-10-31.md' for x in changed),changed
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='dev'
    for name in SCRIPTS:py_compile.compile(str(HERE/name),doraise=True)
    result=dict(artifact_integrity_passed=True,simulation_acceptance_passed=False,
        note='Integrity verification is not an interaction quality pass. Raw failed checks are retained; controls intentionally violate full-excision criteria.',
        baseline_commit=BASELINE,valid_functional_runs=len(valid),unique_cases=len({r['case'] for r in valid}),
        excluded_runs=excluded,completed_scenarios=sum(r['completed_scenario'] for r in valid),
        aborted_scenarios=[r['run'] for r in valid if not r['completed_scenario']],
        continuous_timing_runs=timings,videos=videos,runs=valid,protected_hashes_unchanged=after,
        compiled_scripts=SCRIPTS)
    (a.root/'artifact-verification.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:result[k] for k in ('artifact_integrity_passed','simulation_acceptance_passed','valid_functional_runs','unique_cases','excluded_runs','completed_scenarios','aborted_scenarios','continuous_timing_runs')},indent=2))
    print('Verified',len(videos),'videos and',len(protected),'protected hashes; all',len(SCRIPTS),'scripts compile.')
if __name__=='__main__':main()
