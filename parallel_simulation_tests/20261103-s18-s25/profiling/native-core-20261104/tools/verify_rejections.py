from pathlib import Path
import os,sys,json
p=Path(__file__).resolve().parents[1];os.sched_setaffinity(0,[30,31]);os.environ.update(PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(p/'tmp/numba'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');sys.dont_write_bytecode=True;sys.path.insert(0,str(p.parent/'aggressive-20261104/final-source'))
from endoscopic_tissue.training.snapshot import decode_python
from endoscopic_tissue.training.topology_fast import topology_statistics_fast
rows=[]
for name in [x.parent.name for x in sorted((p/'results').glob('*/result.json')) if (x.parent/'failure-render-state.bin').exists()]:
 state=decode_python((p/'results'/name/'failure-render-state.bin').read_bytes());q=topology_statistics_fast(state['surface']);d=json.loads((p/'results'/name/'result.json').read_text());f=d['frames'][-1];g=f['quality']['metrics']['surface_topology'];assert q==g and q['positional_nonmanifold_edges']>0
 rows.append(dict(name=name,rejected=True,completed=False,frame_id=f['frame_id'],stage=f['stage'],cpu_reference=q,cpu_gpu_equal=True,causal_relation_to_scheduling_change_established=False))
(p/'results/rejected-physical-runs.json').write_text(json.dumps(rows,indent=2));print('CPU-confirmed rejected runs',len(rows))
