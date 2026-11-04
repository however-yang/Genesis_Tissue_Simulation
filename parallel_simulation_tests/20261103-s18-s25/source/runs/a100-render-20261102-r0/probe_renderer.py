from pathlib import Path
import os,sys,pickle,struct,zlib,json
import cv2
r=Path('/srv/disk2/hryang/genesis-exec/releases/0643e10991cf');sys.path[:0]=[str(r),str(r/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation')]
from replay_archive import open_replay
from endoscopic_tissue.rendering import EndoscopicView
from endoscopic_tissue.scenario import SCISSORS_SCENES
source=r/'endoscopic_tissue/outputs/acceptance-fixes-20261102/a100-base-three-video-r0/replay.bin'
with open_replay(source) as stream:
 settings,scene_name,provenance=pickle.load(stream);prefix=stream.read(4);step,stage,surface,obs=pickle.loads(zlib.decompress(stream.read(struct.unpack('<I',prefix)[0])))
scene=SCISSORS_SCENES[scene_name];view=EndoscopicView(settings,640,480,supersample=1,kidney=scene.shape,camera=scene.shot)
try:
 view.update(surface,obs);frame,depth=view.render();assert frame.shape==(480,640,3)
 from OpenGL import GL
 vendor=GL.glGetString(GL.GL_VENDOR);renderer=GL.glGetString(GL.GL_RENDERER)
 record={'frame_shape':frame.shape,'egl_device_id':os.environ.get('EGL_DEVICE_ID'),'vendor':None if vendor is None else vendor.decode(),'renderer':None if renderer is None else renderer.decode(),'source':str(source),'passed':True}
 here=Path(__file__).resolve().parent;cv2.imwrite(str(here/'egl-probe.png'),frame);(here/'egl-probe-result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
finally:view.close()
