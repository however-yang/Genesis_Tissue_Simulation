"""Create browser-compatible copies; retain original complete comparison videos."""
from pathlib import Path
import hashlib
import json
import subprocess
import cv2
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
DATA=ROOT/'endoscopic_tissue/outputs/material-contact-next-20261031'
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();results=[]
for name in ('additive','groups'):
    folder=DATA/name/(name+'-review');source=folder/'comparison.mp4';target=folder/'comparison-h264.mp4'
    assert source.is_file() and not target.exists(),target
    args=[ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-n','-i',str(source),'-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-threads','4','-movflags','+faststart',str(target)]
    subprocess.run(args,cwd=ROOT,check=True)
    cap=cv2.VideoCapture(str(target));assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==1111 and cap.get(cv2.CAP_PROP_FPS)==30
    assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==(2560,1008)
    count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(1008,2560,3);count+=1
    cap.release();assert count==1111
    results.append(dict(candidate=name,source=str(source),path=str(target),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),frames=count,fps=30,width=2560,height=1008,codec='H.264 yuv420p',crf=17,bytes=target.stat().st_size,note='Playback encoding only; original videos and all simulation states retained.'))
    (WORK/'playback-videos.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    print(name+': encoded and decoded all 1111 frames',flush=True)
