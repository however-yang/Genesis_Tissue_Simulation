"""Render all recorded simulation states; no native solver runs in this process."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from threadpoolctl import threadpool_limits
from endoscopic_tissue.rendering import EndoscopicView
from endoscopic_tissue.scenario import SCISSORS_SCENES


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir', type=Path)
    args = parser.parse_args()
    source = args.case_dir/'replay.bin'
    output = args.case_dir/'review'
    output.mkdir(exist_ok=True)
    if (output/'interactions-h264.mp4').exists():
        raise FileExistsError('Existing video is preserved; inspect it before retrying this output directory')
    report = json.loads((args.case_dir/'report.json').read_text(encoding='utf8'))
    trajectories = [json.loads(line) for line in (args.case_dir/'trajectory.jsonl').read_text(encoding='utf8').splitlines()]
    stage_ends = {s['end']['step'] for s in report['stages']}
    stage_ends.add(len(trajectories))
    width, height = 1280, 960
    thumbs, digest = [], hashlib.sha256()
    count = 0
    movie = output/'interactions-h264.mp4'
    with source.open('rb') as stream:
        settings, scene_name, provenance = pickle.load(stream)
        scene = SCISSORS_SCENES[scene_name]
        view = EndoscopicView(settings, width, height, supersample=1, kidney=scene.shape, camera=scene.shot)
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(1/settings.dt), '-i', '-',
            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-threads', '4',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(movie)]
        # Genesis replaces Windows standard handles. Give ffmpeg real explicit
        # handles instead of inheriting the renderer's console handles.
        error_stream = (output/'encoding.log').open('wb')
        try:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=error_stream)
        except Exception:
            error_stream.close()
            view.close()
            raise
        try:
            while prefix := stream.read(4):
                step, stage, surface, obs = pickle.loads(zlib.decompress(stream.read(struct.unpack('<I',prefix)[0])))
                view.update(surface,obs)
                frame = view.render()[0]
                row = trajectories[step-1] if step else None
                cv2.rectangle(frame,(0,0),(width,78),(12,12,12),-1)
                cv2.putText(frame,f'{report["metadata"]["case"]} | {stage} | step {step} | sim t={step*settings.dt:.2f}s',
                    (12,25),cv2.FONT_HERSHEY_SIMPLEX,.64,(240,240,240),1,cv2.LINE_AA)
                if row:
                    slip = row['pad_slip_mm']
                    info = f'anchors={row["grip_points"]} held={row["held_bite_nodes"]}/{row["bite_nodes"]}  cuts={row["cut_edges"]}  '
                    info += f'jaw N={row["jaw_load_n"][0]:.2f}/{row["jaw_load_n"][1]:.2f}'
                    if slip is not None: info += f' slip={slip:.2f}mm'
                    cv2.putText(frame,info,(12,49),cv2.FONT_HERSHEY_SIMPLEX,.56,(230,230,230),1,cv2.LINE_AA)
                cv2.putText(frame,'Exact solved states; 30 Hz simulation-time replay is NOT measured display FPS',
                    (12,71),cv2.FONT_HERSHEY_SIMPLEX,.49,(170,210,230),1,cv2.LINE_AA)
                digest.update(frame.tobytes())
                encoder.stdin.write(frame.tobytes())
                if step in stage_ends:
                    cv2.imwrite(str(output/(stage+'.png')),frame)
                    print('rendered',step,stage,flush=True)
                if step % 15 == 0 or step in stage_ends:
                    thumb = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((426,320))
                    ImageDraw.Draw(thumb).rectangle((0,0,426,20), fill='black')
                    ImageDraw.Draw(thumb).text((5,4),f'{step}: {stage}',fill='white')
                    thumbs.append(thumb)
                count += 1
        finally:
            encoder.stdin.close()
            code = encoder.wait()
            error_stream.close()
            view.close()
        if code: raise RuntimeError(f'ffmpeg exit {code}')
    for page in range(0,len(thumbs),12):
        sheet=Image.new('RGB',(426*3,320*4),'#101010')
        for i,thumb in enumerate(thumbs[page:page+12]): sheet.paste(thumb,((i%3)*426,(i//3)*320))
        sheet.save(output/f'storyboard-{page//12+1:02d}.jpg',quality=92)
    cap = cv2.VideoCapture(str(movie))
    decoded=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if frame.shape[:2] != (height,width): raise RuntimeError('Wrong frame size')
        decoded += 1
    cap.release()
    assert decoded == count == report['steps']+1, (decoded,count,report['steps'])
    (output/'video-verification.json').write_text(json.dumps(dict(frames=count,decoded=decoded,
        expected_frames=report['steps']+1,completed_scenario=report['completed'],
        fps=1/settings.dt,width=width,height=height,frame_stream_sha256=digest.hexdigest(),
        movie_sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),source=str(source),
        provenance=provenance,note='All recorded frames encoded and decoded. Review sheets sample every 0.5 s plus stage ends.'),indent=2),encoding='utf8')
    print('verified',count,'frames',movie,flush=True)


if __name__=='__main__': main()
