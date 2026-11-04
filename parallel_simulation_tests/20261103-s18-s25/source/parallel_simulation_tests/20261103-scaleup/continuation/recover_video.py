"""Finish the interrupted recording in a new directory; preserve the original partial video."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/srv/disk2/hryang/genesis-exec')
RELEASE=ROOT/'releases/0643e10991cf'
SOURCE=RELEASE/'endoscopic_tissue/outputs/acceptance-fixes-20261102/a100-s16d-base-pilot-video-r2/held-plusz-4.0-r00'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);case=a.output/'case';case.mkdir()
    evidence={}
    for name in ['report.json','trajectory.jsonl','replay-archive.json','replay.rdx','result.json','context.json','capture-progress.json']:
        source=SOURCE/name;dest=case/name;shutil.copy2(source,dest);evidence[name]=sha(source)
        assert sha(dest)==evidence[name]
    sys.path[:0]=[str(RELEASE),str(RELEASE/'endoscopic_tissue_backups/2026-11-01-s15-s17')]
    from render_validation import preflight
    before=preflight(case);(case/'capture-preflight.json').write_text(json.dumps(before,indent=2))
    renderer=RELEASE/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation/render_review.py'
    with (a.output/'renderer-child.log').open('xb') as log:
        subprocess.run([sys.executable,'-B','-u',str(renderer),str(case)],cwd=RELEASE,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    video=json.loads((case/'review/video-verification.json').read_text())
    assert video['decoded']==video['frames']==before['frames']
    assert all(sha(SOURCE/name)==digest for name,digest in evidence.items())
    result={'completed':True,'frames':video['frames'],'source':str(SOURCE),'source_sha256':evidence,
            'video':str(case/'review/interactions-h264.mp4'),'video_sha256':sha(case/'review/interactions-h264.mp4'),
            'old_partial_video_preserved':True,'physical_acceptance_changed':False,
            'scope':'Restored the last interrupted rendering job; original R3 uncaptured acquisition failure remains rejected.'}
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
