"""Independent mesh containment audit for every saved free-release frame."""
import argparse,json,sys
from pathlib import Path
import numpy as np,trimesh
from threadpoolctl import threadpool_limits
@threadpool_limits.wrap(limits=1,user_api='blas')
def main():
    p=argparse.ArgumentParser();p.add_argument('case_dir',type=Path);a=p.parse_args()
    report_path=a.case_dir/'report.json'
    if not report_path.exists():raise RuntimeError('Simulation has not finished; refusing a partial audit')
    path=a.case_dir/'trajectory.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()];lookup={r['step']:r for r in rows};audit=[]
    for f in sorted(a.case_dir.glob('state-*.npz')):
        step=int(f.stem.split('-')[1]);r=lookup[step]
        if r['stage'] not in ('jaw_open','open_retract') or r.get('anchors',r.get('grip_points'))>0:continue
        d=np.load(f);m=trimesh.Trimesh(d['vertices_cm'],d['faces'],process=False);m.merge_vertices(digits_vertex=5)
        parts=sorted(m.split(only_watertight=False),key=lambda z:len(z.faces),reverse=True)
        out=dict(step=step,components=len(parts))
        if len(parts)>=2:
            body,piece=parts[:2];inside=body.contains(piece.vertices)
            out.update(body_closed=bool(body.is_watertight),piece_closed=bool(piece.is_watertight),tested=len(inside),inside=int(inside.sum()),fraction=float(inside.mean()),piece_bounds_mm=(piece.bounds*10).tolist(),body_bounds_mm=(body.bounds*10).tolist())
            out['max_penetration_mm']=float(trimesh.proximity.closest_point(body,piece.vertices[inside])[1].max()*10) if inside.any() else 0.
        audit.append(out);print(json.dumps(out),flush=True)
    expected=[r['step'] for r in rows if r['stage'] in ('jaw_open','open_retract') and r.get('anchors',r.get('grip_points'))==0]
    if not expected or [r['step'] for r in audit]!=expected:raise RuntimeError('Missing free-release state frames')
    (a.case_dir/'release-overlap-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
if __name__=='__main__':main()
