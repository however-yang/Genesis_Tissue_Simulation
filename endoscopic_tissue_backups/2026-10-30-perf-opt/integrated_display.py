"""Measure the existing two-thread live flow with offscreen presentation.

Only window blitting and keyboard input are replaced. Simulation, rendering,
latest-state queue and wall-clock video recording use the production live flow.
"""
import importlib.util
import faulthandler
import json
from pathlib import Path
import runpy
import sys
import time
faulthandler.enable(all_threads=True)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from endoscopic_tissue import run
from threadpoolctl import threadpool_info
output=Path(sys.argv[1]).resolve()
if len(sys.argv)>2:
    spec=importlib.util.spec_from_file_location("endoscopic_tissue._integrated_renderer",sys.argv[2])
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    module.__file__=str(ROOT / "endoscopic_tissue/rendering.py")
    run.EndoscopicView=module.EndoscopicView
after_stage=run.LiveLink.after_stage
rows=[]
inside_pools=[]
opened=run.Flow.open


def record_pools(self,tissue):
    inside_pools.extend(threadpool_info())
    return opened(self,tissue)


def recorded(self,flow,stage,surface):
    rows.append(dict(stage=stage,steps=self.stage_steps,frames=self.shown[stage],
                     compute_s=self.stage_compute_s,wall_s=self.stage_wall_s))
    return after_stage(self,flow,stage,surface)


def wait_key(milliseconds):
    if milliseconds>0:
        time.sleep(milliseconds/1000)
    return -1


run.LiveLink.after_stage=recorded
run.Flow.open=record_pools
run.cv2.imshow=lambda *args: None
run.cv2.waitKey=wait_key
pools_before=threadpool_info()
sys.argv=["dev_run.py","--render-every","1","--scene","kidney","--tool","scissors","--output",str(output)]
try:
    runpy.run_path(str(ROOT / "endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py"),run_name="__main__")
finally:
    if output.is_dir():
        (output / "threaded-timing.json").write_text(json.dumps(dict(
            note="Production live threads with offscreen blitting; video uses elapsed wall-clock time",stages=rows,
            thread_pools_before=pools_before,thread_pools_inside=inside_pools,
            thread_pools_after=threadpool_info()),indent=2))
