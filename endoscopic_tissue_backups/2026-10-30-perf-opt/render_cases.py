"""Exercise renderer updates that a single surgical trajectory does not cover."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import struct
import sys
import zlib
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from endoscopic_tissue.rendering import EndoscopicView
from endoscopic_tissue.scenario import SCISSORS_SCENES

source, output = Path(sys.argv[1]), Path(sys.argv[2])
if len(sys.argv)>3:
    spec = importlib.util.spec_from_file_location("endoscopic_tissue._case_renderer",sys.argv[3])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.__file__ = str(ROOT / "endoscopic_tissue/rendering.py")
    EndoscopicView = module.EndoscopicView
saved = {}
with source.open("rb") as stream:
    settings, scene_name, provenance = pickle.load(stream)
    while prefix := stream.read(4):
        data = stream.read(struct.unpack("<I",prefix)[0])
        frame = pickle.loads(zlib.decompress(data))
        if frame[0] in (0,840):
            saved[frame[0]] = frame
        if frame[0] == 840:
            break
_,_,intact,initial = saved[0]
_,_,cut,obs = saved[840]
knife = len(sys.argv)>4 and sys.argv[4] == "knife"
if knife:
    settings=replace(settings,cutting_tool="knife")
    initial,obs=replace(initial,scissors=None),replace(obs,scissors=None)
cases = [
    ("initial",intact,initial,1),
    ("moved",replace(intact,vertices=intact.vertices+np.array((.02,0,0),np.float32)),initial,1),
    ("cut",cut,obs,1),
    ("hidden_scissors",cut,replace(obs,scissors=None),1),
    ("shown_scissors",cut,obs,1),
    ("reordered_triangles",replace(cut,triangles=cut.triangles[::-1].copy()),obs,1),
    ("refined",cut,obs,2),
    ("intact_again",intact,initial,1),
    ("cut_again",cut,obs,1),
]
output.mkdir(parents=True,exist_ok=False)
scene=SCISSORS_SCENES[scene_name]
view=EndoscopicView(settings,1280,960,supersample=1,kidney=None if knife else scene.shape,camera=scene.shot)
hashes={}
try:
    for name,surface,observation,divisions in cases:
        view.update(surface,observation,divisions=divisions)
        frame=view.render()[0]
        hashes[name]=hashlib.sha256(frame.tobytes()).hexdigest()
        cv2.imwrite(str(output / f"{name}.png"),frame)
        print(name,hashes[name],flush=True)
finally:
    view.close()
    (output / "hashes.json").write_text(json.dumps(hashes,indent=2))
