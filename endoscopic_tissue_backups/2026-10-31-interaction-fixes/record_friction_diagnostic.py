"""Full pure-Coulomb grasp recording plus read-only contact force snapshots."""
import sys,ctypes as ct,json
from pathlib import Path
import validate_strict as v
import interaction_fixtures
interaction_fixtures.install()
case=sys.argv[sys.argv.index('--case')+1];out=Path(sys.argv[sys.argv.index('--output')+1]);count=0
original=v.NativeTissue.step
_,_,_,stages,_=v.fixture(case);boundaries={};end=0
for s in stages:end+=s.steps;boundaries[end]=s.name
snapshots=set(boundaries)
for i in range(150,end+1,10):snapshots.add(i)
def step(self,command):
 global count
 if count==0 and case=='grasp-central-zero-friction':self.check('LCS_SetContactModel',1,0.,0.,.1,.01)
 obs=original(self,command);count+=1
 if count in snapshots:
  fn=getattr(self.dll,'LCS_SaveFrictionCase',None)
  if fn:
   fn.argtypes=[ct.c_char_p];fn.restype=ct.c_int;code=fn(str(out/('friction-'+str(count)+'.bin')).encode());assert code==0,code
 return obs
v.NativeTissue.step=step
try:v.main()
finally:
 if case=='grasp-central-zero-friction':(out/'NEGATIVE-CONTROL.txt').write_text('This explicit negative control sets muStatic=muDynamic=0, contact_model=1, after initialization. No other fixture or default is changed.')
