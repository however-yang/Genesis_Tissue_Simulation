import json,sys
from pathlib import Path
import numpy as np
for path in map(Path,sys.argv[1:]):
 a=np.fromfile(path,'f4').astype(float);xx,yy,zz,xy,xz,yz=a[27:33];H=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]]);g=a[24:27];step=a[21:24];exact=-np.linalg.solve(H,g);e=np.linalg.eigvalsh(H)
 out=dict(file=str(path),node=int(a[1]),position_mm=(a[12:15]*10).tolist(),live_tets=int(a[5]),contact_degrees=a[6:9].astype(int).tolist(),matrix=H.tolist(),gradient=g.tolist(),eigenvalues=e.tolist(),condition_number=float(np.linalg.cond(H)),actual_step_mm=(step*a[4]*10).tolist(),fp64_step_mm=(exact*a[4]*10).tolist(),step_relative_error=float(np.linalg.norm(step-exact)/max(np.linalg.norm(exact),1e-30)),linear_relative_residual=float(np.linalg.norm(H@step+g)/max(np.linalg.norm(g),1e-30)),actual_descent_dot=float(g@step),fp64_descent_dot=float(g@exact),note='Both solves use the SAME captured float32 matrix and gradient. This isolates inversion error from changed material/contact inputs.')
 path.with_suffix('.linear.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
