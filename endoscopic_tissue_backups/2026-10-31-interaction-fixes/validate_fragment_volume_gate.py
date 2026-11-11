"""Full interaction with a read-only comparison of gated and reference query sets."""
import sys,json,ctypes as ct
from pathlib import Path
import validate_strict as v
import interaction_fixtures
interaction_fixtures.install()
out=Path(sys.argv[sys.argv.index('--output')+1]);case=sys.argv[sys.argv.index('--case')+1];original=v.NativeTissue.step;rows=[];count=0
_,_,_,stages,_=v.fixture(case);wanted={1244,1250,1280,1320};end=0
for stage in stages:end+=stage.steps;wanted.add(end)
def step(self,command):
 global count
 obs=original(self,command);count+=1
 if count in wanted:
  fn=self.dll.LCS_ValidateFragmentVolumeGate;fn.argtypes=[ct.POINTER(ct.c_int),ct.c_int];fn.restype=ct.c_int;values=(ct.c_int*8)();code=fn(values,8);assert code==0,code
  a=list(values);row=dict(step=count,possible=a[0],reference_queries=a[1],gated_queries=a[2],differences=a[3],expected_possible=a[4],capacity=a[5],surface_triangles=a[6],nodes=a[7]);rows.append(row)
  assert a[0]==a[4] and a[1]==a[2] and a[3]==0,('Fragment query gate mismatch',row)
 return obs
v.NativeTissue.step=step
try:
 v.main()
 if case=='kidney-reference':
  assert any(x['possible']==0 for x in rows),'No empty topology control was tested'
  assert any(x['reference_queries']>0 for x in rows),'No released fragment queries were tested'
finally:
 out.mkdir(exist_ok=True,parents=True);(out/'fragment-query-gate-audit.json').write_text(json.dumps(dict(case=case,checks=rows,read_only=True,sorted_ids_compared=True,pass_all=bool(rows) and all(x['possible']==x['expected_possible'] and x['differences']==0 and x['reference_queries']==x['gated_queries'] for x in rows)),indent=2))
