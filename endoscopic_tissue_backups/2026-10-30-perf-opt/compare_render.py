import json
from pathlib import Path
import sys
import cv2
import numpy as np

baseline, candidate = [Path(p) for p in sys.argv[1:3]]
a = json.loads((baseline / "render-timing.json").read_text())
b = json.loads((candidate / "render-timing.json").read_text())
assert len(a["rows"]) == len(b["rows"])
differences = []
for x, y in zip(a["rows"], b["rows"]):
    assert (x["step"], x["stage"]) == (y["step"], y["stage"])
    if x["frame_sha256"] != y["frame_sha256"]:
        differences.append(x["step"])
result = dict(frames=len(a["rows"]), different_frame_hashes=len(differences), different_steps=differences)
for name, data in (("baseline", a), ("candidate", b)):
    rows = [r for r in data["rows"] if r["step"] > 2]
    update = np.array([r["update_ms"] for r in rows])
    render = np.array([r["render_ms"] for r in rows])
    result[name] = dict(update_mean_ms=float(update.mean()), render_mean_ms=float(render.mean()),
                        total_mean_ms=float((update+render).mean()), total_p95_ms=float(np.percentile(update+render,95)))
stills = []
for path in sorted(baseline.glob("frame-*.png")):
    left, right = cv2.imread(str(path)), cv2.imread(str(candidate / path.name))
    assert right is not None and left.shape == right.shape
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    stills.append(dict(frame=path.name, max_channel_difference=int(delta.max()),
                       mean_channel_difference=float(delta.mean()), changed_pixels=int(np.count_nonzero(delta.any(axis=2)))))
result["sampled_stills"] = stills
(candidate / "comparison.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ("sampled_stills", "different_steps")}, indent=2))
if differences:
    print("First mismatched frames:", differences[:20])
    print("Maximum sampled pixel error:", max(s["max_channel_difference"] for s in stills))
