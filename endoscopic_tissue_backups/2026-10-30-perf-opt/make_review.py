"""Build synchronized comparison video and contact sheets from complete recordings."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np

left_path, right_path, output = [Path(p) for p in sys.argv[1:4]]
label = sys.argv[4]
output.mkdir(parents=True, exist_ok=False)
left, right = cv2.VideoCapture(str(left_path)), cv2.VideoCapture(str(right_path))
assert left.isOpened() and right.isOpened()
fps = left.get(cv2.CAP_PROP_FPS)
assert fps == right.get(cv2.CAP_PROP_FPS)
assert left.get(cv2.CAP_PROP_FRAME_COUNT) == right.get(cv2.CAP_PROP_FRAME_COUNT)
writer = cv2.VideoWriter(str(output / "comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2560, 1008))
assert writer.isOpened()
groups = [(150,165,180,195,210,240,270,300,330,360,420,450),
          (450,465,480,495,510,525,540,555,570,585,600,630),
          (690,720,750,780,810,840,900,960,990,1020,1080,1110)]
selected = set(sum((list(g) for g in groups), []))
frames = ({}, {})
step = 0
try:
    while True:
        ok_a, a = left.read()
        ok_b, b = right.read()
        assert ok_a == ok_b
        if not ok_a:
            break
        assert a.shape == b.shape == (960,1280,3)
        canvas = np.zeros((1008,2560,3), np.uint8)
        canvas[48:,:1280], canvas[48:,1280:] = a, b
        for x, title in ((10,"BASELINE"), (1290,label)):
            cv2.putText(canvas,f"{title}  |  step {step}  |  simulation {step/fps:.2f} s",
                        (x,32),cv2.FONT_HERSHEY_SIMPLEX,.75,(235,235,235),2,cv2.LINE_AA)
        writer.write(canvas)
        if step in selected:
            for store, frame in zip(frames, (a,b)):
                tile = cv2.resize(frame, (320,240), interpolation=cv2.INTER_AREA)
                cv2.putText(tile,f"step {step}",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
                store[step] = tile
        step += 1
finally:
    writer.release()
    left.release()
    right.release()
for side, store in zip(("baseline", "candidate"),frames):
    for index, group in enumerate(groups):
        sheet = np.concatenate([np.concatenate([store[n] for n in group[i:i+4]],axis=1)
                                for i in range(0,len(group),4)],axis=0)
        cv2.imwrite(str(output / f"{side}-{index+1}.png"),sheet)
(output / "manifest.json").write_text(json.dumps(dict(left=str(left_path),right=str(right_path),
    label=label,frames=step,fps=fps,note="Synchronized simulation-time playback, not achieved runtime FPS"),indent=2))
print(f"Verified {step} synchronized frames at playback {fps} fps")
