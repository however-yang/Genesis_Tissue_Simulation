"""Review exact encoded frames from baseline, R19 and R20; never rerun physics."""
from pathlib import Path
import json
import cv2
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
P=Path(__file__).resolve().parent
OUT=ROOT/'endoscopic_tissue/outputs'
review=P/'interaction-review'
review.mkdir(exist_ok=True)
movies=[('baseline',OUT/'interaction-fixes-20261031/release-baseline/review/interactions-h264.mp4'),
        ('R19',OUT/'interaction-fixes-20261031/r19-strict-kidney-reference/review/interactions-h264.mp4'),
        ('R20',OUT/'training-completion-20261101/r20-kidney/review/interactions-h264.mp4')]
caps=[cv2.VideoCapture(str(p)) for _,p in movies]
counts=[int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]


def frame(cap,step,width=426,height=320):
    cap.set(cv2.CAP_PROP_POS_FRAMES,step)
    ok,bgr=cap.read()
    return Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((width,height)) if ok else Image.new('RGB',(width,height),'#111111')


steps=[165,210,330,450,600,840,960,1080,1215,1244,1246,1260,1290,1320]
for start in range(0,len(steps),4):
    sheet=Image.new('RGB',(1278,4*320),'#111111')
    for row,step in enumerate(steps[start:start+4]):
        for col,((name,_),cap) in enumerate(zip(movies,caps)):
            image=frame(cap,step)
            draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,426,20),fill='black')
            draw.text((6,4),f'{name}: step {step}'+(' | baseline ended' if step>=counts[col] else ''),fill='white')
            sheet.paste(image,(col*426,row*320))
    sheet.save(review/f'comparison-{start//4+1:02}.jpg',quality=94)
for start in range(1244,1321,20):
    sheet=Image.new('RGB',(1280,1200),'#111111')
    for index,step in enumerate(range(start,min(start+20,1321))):
        image=frame(caps[2],step,320,240)
        draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,320,18),fill='black')
        draw.text((4,3),f'R20 free release: {step}',fill='white')
        sheet.paste(image,((index%4)*320,(index//4)*240))
    sheet.save(review/f'all-release-{start:04}.jpg',quality=94)
for c in caps:c.release()
(review/'index.json').write_text(json.dumps({'movies':[(n,str(p)) for n,p in movies],'frame_counts':counts,
    'comparison_steps':steps,'all_r20_release_steps':list(range(1244,1321)),
    'note':'Complete movies exist; comparison samples plus every free-release frame exported for visual inspection.'},indent=2))
print('Prepared comparisons and all 77 free-release frames')
