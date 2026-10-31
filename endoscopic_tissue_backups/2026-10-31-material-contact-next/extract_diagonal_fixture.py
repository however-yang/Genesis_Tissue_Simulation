"""Freeze the twelve largest-error diagonal inputs from the original step800 snapshot."""
from pathlib import Path
import json
import numpy as np
WORK=Path(__file__).resolve().parent
source=json.loads((WORK/'additive-error-localization.json').read_text(encoding='utf8'))
with Path(source['snapshot']).open('rb') as f:
    def get(dtype,count):return np.fromfile(f,dtype=dtype,count=count)
    _,n,top,coarse=get('<i4',4);get('<u2',n);h=get('<f4',6*n).reshape(n,6)
    for level in range(top):
        count,size=get('<i4',2);get('<i4',size);get('<f4',9*count)
    get('<f4',coarse*coarse);r=get('<f4',3*n).reshape(n,3)
nodes=[item['node'] for item in source['worst']]
result=dict(source=source['snapshot'],nodes=nodes,rows=np.concatenate([h[nodes],r[nodes]],axis=1).tolist())
(WORK/'diagonal-frozen-inputs.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('Saved',len(nodes),'frozen blocks')
