import argparse,json
from pathlib import Path
import cv2
import numpy as np
p=argparse.ArgumentParser();p.add_argument('case',type=Path);a=p.parse_args()
rows=[json.loads(s) for s in (a.case/'trajectory.jsonl').read_text().splitlines()]
selected=[r['step'] for r in rows if r['stage'] in ('jaw_open','open_retract') and r.get('anchors',r.get('grip_points'))==0]
cap=cv2.VideoCapture(str(a.case/'review/interactions-h264.mp4'))
first=selected[0];cap.set(cv2.CAP_PROP_POS_FRAMES,first)
page=None
previous=first-1
for i,step in enumerate(selected):
    if step!=previous+1:cap.set(cv2.CAP_PROP_POS_FRAMES,step)
    ok,frame=cap.read();previous=step
    if not ok:raise RuntimeError(('Missing frame',step))
    # Context retained in complete video; crop shows tissue-body interface in detail.
    tile=cv2.resize(frame[360:880,460:1100],(384,312))
    cv2.rectangle(tile,(0,0),(384,25),(0,0,0),-1)
    cv2.putText(tile,str(step),(8,18),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    if i%12==0:page=np.zeros((312*3,384*4,3),np.uint8)
    row,col=divmod(i%12,4);page[row*312:(row+1)*312,col*384:(col+1)*384]=tile
    if i%12==11 or i==len(selected)-1:cv2.imwrite(str(a.case/'review'/f'release-every-frame-{i//12+1:02d}.jpg'),page)
cap.release();print('Every free-release frame',len(selected),'saved to sheets')
