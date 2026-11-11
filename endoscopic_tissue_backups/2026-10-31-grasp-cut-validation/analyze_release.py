"""Check separate reconstructed surfaces for overlap during the recorded jaw release."""
import argparse
import json
from pathlib import Path
import pickle
import struct
import sys
import zlib
from replay_archive import open_replay

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=1,user_api='blas')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir',type=Path)
    args=parser.parse_args()
    rows=[]
    frames=[]
    with open_replay(args.case_dir/'replay.bin') as stream:
        settings,_,_=pickle.load(stream)
        while prefix:=stream.read(4):
            step,stage,surface,obs=pickle.loads(zlib.decompress(stream.read(struct.unpack('<I',prefix)[0])))
            if stage!='jaw_open' or obs.grip_points>0:continue
            mesh=trimesh.Trimesh(surface.vertices,surface.triangles,process=False)
            mesh.merge_vertices(digits_vertex=5)
            pieces=sorted(mesh.split(only_watertight=False),key=lambda m:len(m.faces),reverse=True)
            row=dict(step=step,anchors=obs.grip_points,components=len(pieces))
            if len(pieces)>=2:
                body,piece=pieces[:2]
                row.update(body_watertight=bool(body.is_watertight),piece_watertight=bool(piece.is_watertight))
                sample=piece.vertices[::4]
                try:
                    inside=body.contains(sample)
                    row.update(tested_piece_surface_points=len(sample),points_inside_body=int(inside.sum()),
                               fraction_inside_body=float(inside.mean()))
                    if inside.any():
                        _,distance,_=trimesh.proximity.closest_point(body,sample[inside])
                        row['max_sampled_penetration_mm']=float(distance.max()*10)
                except ImportError as error:row['contains_unavailable']=str(error)
                frames.append((step,body.copy(),piece.copy()))
            rows.append(row)
    (args.case_dir/'release-overlap-audit.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    for row in rows:print(json.dumps(row),flush=True)
    if not frames:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    selected=frames[-3:]
    fig,axes=plt.subplots(2,len(selected),figsize=(4*len(selected),8),squeeze=False)
    for col,(step,body,piece) in enumerate(selected):
        for row,axes_indices in enumerate(((0,1),(2,1))):
            ax=axes[row,col]
            for m,color in ((body,'#477dac'),(piece,'#ef9234')):
                triangles=m.vertices[m.faces][:,:,axes_indices]*10
                ax.add_collection(PolyCollection(triangles,facecolors=color,edgecolors='none',alpha=.6))
            ax.set_xlim(-35,35);ax.set_ylim(-35,75);ax.set_aspect('equal')
            ax.set_title(f'Step {step} (dt={settings.dt*1000:.1f} ms)')
            ax.set_xlabel(('x' if row==0 else 'z')+' [mm]');ax.set_ylabel('y [mm]')
            ax.grid(alpha=.2)
    fig.suptitle('Recorded release: body blue, excised piece orange; orthographic projection')
    fig.tight_layout()
    fig.savefig(args.case_dir/'release-projections.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':main()
