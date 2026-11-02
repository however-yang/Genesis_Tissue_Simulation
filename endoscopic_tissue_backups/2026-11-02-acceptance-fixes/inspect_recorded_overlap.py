"""Locate a checked overlap in the actual recorded solved state, without physics."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import struct
import sys
import zlib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation')]
from replay_archive import open_replay
from endoscopic_tissue.instrument_geometry import load_scissors, scissors_body_depths


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('folder', type=Path)
    p.add_argument('--tick', type=int, required=True)
    args = p.parse_args()
    folder = args.folder.resolve()
    if not folder.is_relative_to(ROOT / 'endoscopic_tissue/outputs'):
        p.error('Require a local generated replay')
    geometry = load_scissors()
    with open_replay(folder / 'replay.bin') as stream:
        settings, _, provenance = pickle.load(stream)
        while prefix := stream.read(4):
            encoded = stream.read(struct.unpack('<I', prefix)[0])
            tick, label, surface, observed = pickle.loads(zlib.decompress(encoded))
            if tick != args.tick:
                continue
            s = observed.scissors
            triangles = surface.vertices[surface.triangles]
            samples = np.concatenate((surface.vertices, triangles.mean(axis=1),
                (triangles[:, 0] + triangles[:, 1])*.5,
                (triangles[:, 1] + triangles[:, 2])*.5,
                (triangles[:, 2] + triangles[:, 0])*.5))
            origins = ([('vertex', i) for i in range(len(surface.vertices))]
                       + [(name, i) for name in ('center', 'edge01', 'edge12', 'edge20')
                          for i in range(len(triangles))])
            relative = samples - s.hinge
            width = np.cross(s.axis, s.normal)
            sn = np.clip(s.gap/(2*geometry.jaws.length), 0, .95)
            cs = np.sqrt(1-sn*sn)
            found = []
            for side, box in zip((1, -1), geometry.backs):
                orientation = np.stack((box[6:9], box[9:12], np.cross(box[6:9], box[9:12])))
                frame = np.stack((s.axis*cs+side*s.normal*sn, s.normal*cs-side*s.axis*sn, width))
                local = (relative @ frame.T-box[:3]) @ orientation.T
                depth = (box[3:6]-np.abs(local)).min(axis=1)
                for i in np.argsort(depth)[-6:][::-1]:
                    origin, index = origins[i]
                    face_ids = np.flatnonzero((surface.triangles == index).any(axis=1)) if origin == 'vertex' else np.array([index])
                    found.append(dict(side=side, sample_kind=origin, index=int(index),
                        depth_mm=float(max(0, depth[i])*10), point_cm=samples[i].tolist(),
                        local_cm=local[i].tolist(), half_cm=box[3:6].tolist(),
                        face_ids=face_ids.tolist(), face_vertices_cm=triangles[face_ids].tolist(),
                        wall_flags=surface.anchors[surface.triangles[face_ids, 0], 3].tolist()))
            depths = scissors_body_depths(surface.vertices, surface.triangles, geometry, s.hinge, s.axis, s.normal, s.gap)
            result = dict(tick=tick, stage=label, provenance=provenance,
                body_depth_mm=float(depths[0]*10), wrist_depth_mm=float(depths[1]*10),
                checked_sampling_difference_mm=abs(max(r['depth_mm'] for r in found)-float(depths[0]*10)),
                grasper_hinge_cm=observed.hinge.tolist(), scissors_hinge_cm=s.hinge.tolist(),
                scissors_gap_mm=float(s.gap*10), scissors_phase=int(s.phase),
                scissor_loads=asdict(s.loads), samples=found)
            output = folder / f'overlap-{tick:04}.json'
            output.write_text(json.dumps(result, indent=2), encoding='utf8')
            np.savez_compressed(folder / f'overlap-{tick:04}.npz', vertices=surface.vertices,
                triangles=surface.triangles, anchors=surface.anchors, nodes=observed.nodes,
                neighbors=observed.neighbors, grip_flags=observed.grip_flags)
            print(json.dumps({k:v for k,v in result.items() if k != 'samples'}), flush=True)
            return
    raise ValueError('Requested recorded tick does not exist')


if __name__ == '__main__':
    main()
