"""CPU command-envelope screen, not a solved-trajectory collision acceptance."""
from dataclasses import replace
import json,time
from pathlib import Path
import numpy as np
from endoscopic_tissue.scenario import SCISSORS_SCENES,scissors_settings,interpolate
from endoscopic_tissue.native_backend import scissors_frame
from endoscopic_tissue.instrument_geometry import load_grasper,load_scissors
from endoscopic_tissue.training.instrument_collision import instrument_distance
settings=scissors_settings();grasp=load_grasper(settings.jaw_length,settings.jaw_width,settings.instrument_model);scissors=load_scissors()
results={}
for name,scene in SCISSORS_SCENES.items():
 previous=scene.stages[0].command;rows=[];timing=[]
 for stage in scene.stages:
  start=previous
  for i in range(stage.steps):
   cmd=interpolate(start,stage.command,(i+1)/(stage.steps*stage.ramp))
   axis,normal=scissors_frame(cmd.scissors_pitch_deg,cmd.scissors_roll_deg,cmd.scissors_yaw_deg)
   began=time.perf_counter()
   distance=instrument_distance(grasp,scissors,grasper_hinge=cmd.hinge,grasper_axis=settings.jaw_axis,
    grasper_normal=settings.jaw_normal,grasper_gap=settings.jaw_open_gap*(1-cmd.closure),
    scissors_hinge=cmd.blade_center,scissors_axis=axis,scissors_normal=normal,
    scissors_gap=scissors.max_gap*(1-cmd.scissors_closure))
   timing.append((time.perf_counter()-began)*1000)
   rows.append(dict(stage=stage.name,step=i,distance_m=distance))
   previous=cmd
 rejected=[r for r in rows if r['distance_m']<-.0002]
 results[name]=dict(frames=len(rows),minimum_m=min(r['distance_m'] for r in rows),flagged_frames=len(rejected),
  first_flags=rejected[:6],cpu_ms_mean=float(np.mean(timing)),cpu_ms_p95=float(np.percentile(timing,95)))
result=dict(scope='command poses only; not solved poses; conservative mesh envelopes',scenes=results)
Path(__file__).with_name('results').joinpath('instrument-command-screen.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
