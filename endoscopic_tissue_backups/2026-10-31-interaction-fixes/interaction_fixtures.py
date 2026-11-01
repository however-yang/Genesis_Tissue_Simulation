"""Explicit diagnostic trajectories; default application fixtures are unchanged."""
from dataclasses import replace
import numpy as np
import validate_strict as v
from endoscopic_tissue.native_backend import scissors_frame
base=v.fixture
EXTRA=['wall-tip-short-wide','wall-tip-completion-wide','wall-tip-short','wall-tip-completion','wall-overlap','wall-yaw-plus','wall-yaw-minus','wall-cut-disabled','grasp-central-long','grasp-central-deeper','grasp-central-zero-friction','grasp-central-regrasp']
def fixture(name):
 if name.startswith('wall-') and name in EXTRA:
  scene_name,scene,settings,stages,kind=base('wall-reference');stages=list(stages)
  if name=='wall-cut-disabled':stages=[replace(s,command=replace(s.command,scissors_cutting=False)) for s in stages]
  if name in ('wall-yaw-plus','wall-yaw-minus'):
   delta=3 if name.endswith('plus') else -3
   stages=[replace(s,command=replace(s.command,scissors_yaw_deg=s.command.scissors_yaw_deg+delta)) for s in stages]
  if name in ('wall-tip-short','wall-tip-completion','wall-tip-short-wide','wall-tip-completion-wide','wall-overlap'):
   last=next(s.command for s in stages if s.name=='incise_3');axis,_=scissors_frame(last.scissors_pitch_deg,last.scissors_roll_deg,last.scissors_yaw_deg)
   shift=(-.30 if name.endswith('-wide') else -.08 if name in ('wall-tip-short','wall-tip-completion') else .10)*axis
   stages=[replace(s,command=replace(s.command,blade_center=tuple(np.array(s.command.blade_center)+shift))) if s.name in ('reposition_3','incise_3') else s for s in stages]
   if name in ('wall-tip-completion','wall-tip-completion-wide'):
    last=next(s.command for s in stages if s.name=='incise_3');advanced=replace(last,blade_center=tuple(np.array(last.blade_center)+(.40 if name.endswith('-wide') else .20)*axis))
    at=next(i for i,s in enumerate(stages) if s.name=='withdraw')
    stages[at:at]=[v.Stage('open_completion',45,replace(last,scissors_closure=0,scissors_cutting=False)),v.Stage('reposition_completion',90,replace(advanced,scissors_closure=0,scissors_cutting=False)),v.Stage('incise_completion',150,advanced,ramp=.5)]
  return scene_name,scene,settings,stages,kind
 if name in EXTRA and name.startswith('grasp-central'):
  scene_name,scene,settings,stages,kind=base('grasp-central');stages=list(stages)
  bite=next(s.command for s in stages if s.name=='close')
  if name=='grasp-central-long':
   start=next(i for i,s in enumerate(stages) if s.name=='lift')
   stages=[replace(s,command=replace(s.command,hinge=tuple(np.array(s.command.hinge)+(0,.4,0)))) if i>=start else s for i,s in enumerate(stages)]
   stages[start]=replace(stages[start],steps=180)
  if name=='grasp-central-deeper':stages=[replace(s,command=replace(s.command,hinge=tuple(np.array(s.command.hinge)+(0,-.1,0)))) for s in stages]
  if name=='grasp-central-regrasp':
   raised=next(s.command for s in stages if s.name=='lift')
   stages += [v.Stage('return_open',120,replace(bite,closure=0)),v.Stage('reclose',90,bite),v.Stage('relift',120,raised),v.Stage('rehold',90,raised),v.Stage('reopen',60,replace(raised,closure=0))]
  return scene_name,scene,settings,stages,kind
 return base(name)
def install():
 v.CASES += [n for n in EXTRA if n not in v.CASES];v.fixture=fixture
