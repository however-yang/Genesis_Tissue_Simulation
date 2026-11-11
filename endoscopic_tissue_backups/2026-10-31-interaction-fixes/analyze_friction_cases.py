import json,sys
from pathlib import Path
import numpy as np
def analyze(folder):
 report=json.loads((folder/'report.json').read_text());stages={int(s['end']['step']):s['stage'] for s in report['stages']};out=[]
 for path in sorted(folder.glob('friction-*.bin'),key=lambda p:int(p.stem.split('-')[-1])):
  with path.open('rb') as f:
   h=np.fromfile(f,'i4',4);assert h[0]==0x46524931 and h[2]==36;a=np.fromfile(f,'f4').reshape(h[1],h[2]);assert not f.read(1)
  step=int(path.stem.split('-')[-1]);pads=a[(a[:,0]>=1)&(a[:,0]<=2)&(a[:,4]>1e-5)];N=pads[:,4]*.01;T=pads[:,6]*.01;cap=pads[:,7]*.01;demand=pads[:,2]*pads[:,5]*.01
  out.append(dict(step=step,stage=stages.get(step),native_anchor_assist=int(h[3]),contacts=len(a),loaded_pad_contacts=len(pads),left_normal_N=float(N[pads[:,0]==1].sum()),right_normal_N=float(N[pads[:,0]==2].sum()),tangent_N=float(T.sum()),static_capacity_N=float(cap.sum()),spring_demand_N=float(demand.sum()),saturated_count=int(np.sum(demand>cap+1e-5)),persistent_stick_count=int(np.sum(pads[:,8]>1e-6)),max_stick_mm=float(np.max(pads[:,8])*10) if len(pads) else 0,cone_violations=int(np.sum(T>cap+1e-5)),assist_points_max=int(np.max(a[:,33])) if len(a) else 0))
 result=dict(case=report['metadata']['case'],completed=report['completed'],max_invalid_tets=report['max_invalid_tets'],stages=[dict(stage=s['stage'],end=s['end']) for s in report['stages']],snapshots=out,maximum_cone_violations=max((r['cone_violations'] for r in out),default=0),note='Read-only diagnostic recomputes forces on the final material state, separate from frame timing. Native force units converted to N by .01, matching NativeTissue.observe. Coulomb cone check covers positive-pressure jaw contacts only; zero pressure imposes no friction.')
 (folder/'friction-audit.json').write_text(json.dumps(result,indent=2));print(folder.name,json.dumps({k:v for k,v in result.items() if k not in ('stages','snapshots')}));print(json.dumps([r for r in out if r['stage']],indent=2))
if __name__=='__main__':
 for p in map(Path,sys.argv[1:]):analyze(p)
