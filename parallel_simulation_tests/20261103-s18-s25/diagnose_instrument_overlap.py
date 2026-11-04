from pathlib import Path
import json
import numpy as np
import trimesh
from endoscopic_tissue.scenario import SCISSORS_SCENES,scissors_settings,interpolate
from endoscopic_tissue.native_backend import scissors_frame
from endoscopic_tissue.instrument_geometry import load_grasper,load_scissors,scissors_body_depths
from endoscopic_tissue.training.instrument_collision import instrument_distance,jaw_boxes,box_distance,capsule_box_distance
s=scissors_settings();g=load_grasper(s.jaw_length,s.jaw_width,s.instrument_model);c=load_scissors()
scene=SCISSORS_SCENES['wall'];previous=scene.stages[0].command;worst=(0,None,None,None)
for stage in scene.stages:
 start=previous
 for i in range(stage.steps):
  cmd=interpolate(start,stage.command,(i+1)/(stage.steps*stage.ramp));previous=cmd
  if stage.name!='approach':continue
  axis,normal=scissors_frame(cmd.scissors_pitch_deg,cmd.scissors_roll_deg,cmd.scissors_yaw_deg)
  d=instrument_distance(g,c,grasper_hinge=cmd.hinge,grasper_axis=s.jaw_axis,grasper_normal=s.jaw_normal,
   grasper_gap=s.jaw_open_gap*(1-cmd.closure),scissors_hinge=cmd.blade_center,scissors_axis=axis,
   scissors_normal=normal,scissors_gap=c.max_gap*(1-cmd.scissors_closure))
  if d<worst[0]:worst=(d,cmd,stage.name,i)
d,cmd,stage,i=worst;axis,normal=scissors_frame(cmd.scissors_pitch_deg,cmd.scissors_roll_deg,cmd.scissors_yaw_deg)
gp=g.posed_jaws(np.array(cmd.hinge),np.array(s.jaw_axis),np.array(s.jaw_normal),s.jaw_open_gap*(1-cmd.closure))
cp=c.jaws.posed_jaws(np.array(cmd.blade_center),axis,normal,c.max_gap*(1-cmd.scissors_closure))
meshes_g=[trimesh.Trimesh(v,f,process=False) for v,f in zip(gp,g.faces)]
meshes_c=[trimesh.Trimesh(v,f,process=False) for v,f in zip(cp,c.jaws.faces)]
report=dict(scope='offline diagnostic at worst command pose, not solved trajectory',stage=stage,step=i,
 envelope_distance_m=d,grasper_watertight=[m.is_watertight for m in meshes_g],scissors_watertight=[m.is_watertight for m in meshes_c],
 original_body_check=[],vertex_mesh_tests=[])
for v,f in zip(gp,g.faces):
 body,wrist=scissors_body_depths(v,f,c,np.array(cmd.blade_center),axis,normal,c.max_gap*(1-cmd.scissors_closure))
 report['original_body_check'].append(dict(blade_body_depth_m=float(body*.01),wrist_depth_m=float(wrist*.01)))
for gi,gm in enumerate(meshes_g):
 for ci,cm in enumerate(meshes_c):
  near=trimesh.proximity.signed_distance(gm,cm.vertices)
  report['vertex_mesh_tests'].append(dict(grasper=gi,scissor=ci,max_inside_depth_m=float(near.max()*.01),
    count_inside_gt_point2mm=int(np.count_nonzero(near>.02)),sign_valid_if_closed=bool(gm.is_watertight)))
Path(__file__).with_name('results').joinpath('instrument-wall-overlap-forensics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
