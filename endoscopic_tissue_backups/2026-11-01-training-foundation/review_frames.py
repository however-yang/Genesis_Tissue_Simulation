"""Extract matched original video frames for visual review; no image synthesis."""
import hashlib
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).parent
OUT = ROOT/'endoscopic_tissue/outputs/training-foundation-20261101'
cases = ['legacy-wall-02','adapter-wall-01','legacy-wall-03']
steps = [0,120,200,270,345,390,450,600,650,690,810,860,900,930,960,1020,1080,1155]
frames = {}
verification = []
for case in cases:
    video = OUT/case/'review/interactions-h264.mp4'
    cap = cv2.VideoCapture(str(video))
    count = 0
    frames[case] = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count in steps:
            frames[case][count] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    cap.release()
    assert count == 1156 and len(frames[case]) == len(steps)
    verification.append(dict(case=case, frames=count, selected_steps=steps,
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest()))
destination = WORK/'review'
destination.mkdir(exist_ok=True)
for page in range(3):
    canvas = Image.new('RGB', (1280, 6*320), '#101010')
    draw = ImageDraw.Draw(canvas)
    for row, step in enumerate(steps[page*6:(page+1)*6]):
        for col, case in enumerate(cases):
            x, y = col*426, row*320
            canvas.paste(frames[case][step].resize((426,320)), (x,y))
            draw.rectangle((x,y,x+426,y+20), fill='black')
            draw.text((x+5,y+3), f'{case} | frame={step} | t={step/30:.2f}s', fill='white')
    canvas.save(destination/f'paired-{page+1:02}.jpg', quality=93)
detail = Image.new('RGB', (3*560,2*390), '#101010')
draw = ImageDraw.Draw(detail)
for row, step in enumerate((450,1155)):
    for col, case in enumerate(cases):
        crop = frames[case][step].crop((480,370,1040,760))
        detail.paste(crop,(col*560,row*390))
        draw.rectangle((col*560,row*390,(col+1)*560,row*390+24),fill='black')
        draw.text((col*560+5,row*390+6),f'{case} | frame {step}',fill='white')
detail.save(destination/'cut-detail.jpg',quality=95)
(WORK/'video-review-index.json').write_text(json.dumps(dict(videos=verification,
    visual_review_scope='18 selected frames per full video, plus matched full-resolution crops; not every-frame visual certification'), indent=2),encoding='utf8')
print('decoded and extracted all three full videos')
