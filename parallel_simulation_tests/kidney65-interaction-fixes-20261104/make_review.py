"""Decode complete videos and produce a time-aligned common-sequence review."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--baseline',default='frontal65-r5');p.add_argument('--name',required=True);p.add_argument('--full',action='store_true');p.add_argument('--stage-frames',help='Comma-separated 1-based physics frames for the review sheet');args=p.parse_args()
    for value in (args.candidate,args.baseline,args.name):
        if not value.replace('-','').replace('_','').isalnum():raise ValueError('Invalid label')
    out=ROOT/'review'/args.name;out.mkdir(exist_ok=False)
    names=[args.baseline,args.candidate]
    paths=[ROOT/'review'/n/'endoscopic_demo.mp4' for n in names]
    caps=[cv2.VideoCapture(str(path)) for path in paths]
    counts=[int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    assert all(count>0 if args.full else count>=1110 for count in counts)
    comparison_frames=max(counts) if args.full else 1110
    output=out/('baseline-candidate-full.mp4' if args.full else 'baseline-candidate-common1110.mp4')
    writer=imageio_ffmpeg.write_frames(str(output),(1280,510),fps=30,codec='libx264',quality=6,macro_block_size=2,output_params=['-threads','2'])
    writer.send(None)
    stages=[329,449,569,659,779,899,1049,1109,1264,1304,1384,1429,1504,1534]
    if args.stage_frames:
        stages=[int(value)-1 for value in args.stage_frames.split(',')]
        assert stages and min(stages)>=0 and len(stages)==len(set(stages))
    sheets=[Image.new('RGB',(1280,260*((len([s for s in stages if s<counts[i]])+3)//4)),(12,12,12)) for i in range(2)]
    selected=[0,0];decoded=[0,0];last_frames=[None,None]
    for k in range(max(counts)):
        frames=[]
        for i,c in enumerate(caps):
            ok,frame=c.read()
            if k<counts[i]:assert ok and frame is not None, f'Truncated video {names[i]} frame {k}'
            if not ok:frames.append(None);continue
            decoded[i]+=1
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);small=Image.fromarray(rgb).resize((640,480))
            last_frames[i]=small
            frames.append(small)
            if k in stages:
                tile=Image.new('RGB',(320,260),(12,12,12));tile.paste(small.resize((320,240)),(0,20));ImageDraw.Draw(tile).text((8,3),f'{names[i]}  frame={k+1}',fill='white')
                j=selected[i];sheets[i].paste(tile,((j%4)*320,(j//4)*260));selected[i]+=1
        if k<comparison_frames:
            both=Image.new('RGB',(1280,510),(12,12,12));draw=ImageDraw.Draw(both)
            for i in range(2):
                ended=frames[i] is None
                both.paste(last_frames[i] if ended else frames[i],(640*i,30))
                label=f'{names[i]} | physics frame {min(k+1,counts[i])}/{counts[i]}'
                if ended:label+=' | CLIP ENDED - last frame held'
                draw.text((640*i+8,8),label,fill='white')
            writer.send(np.asarray(both))
    writer.close()
    for i,c in enumerate(caps):c.release();sheets[i].save(out/(names[i]+'-stages.jpg'),quality=92)
    result=dict(passed=decoded==counts,decoded_frames=dict(zip(names,decoded)),common_frames=min(counts) if args.full else 1110,
        output_frames=comparison_frames,full_comparison=args.full,held_end_frames={name:comparison_frames-count for name,count in zip(names,counts)} if args.full else {},
        fps=30,note='Video playback at simulation dt; not measured rendering/physics throughput. Complete frames decoded without retiming. Unequal clips hold only the ended side with an explicit label. Selected stage images are for human visual review.',
        sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in zip(names,paths)})
    (out/'decode-report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
