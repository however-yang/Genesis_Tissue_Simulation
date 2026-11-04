"""Small two-body, pure-friction recontact probe; not a surgical quality pass.

The two solids start separated, then meet under gravity before a grasper
approaches and closes. This exercises a retained tissue/tissue manifold while
instrument reconstruction repair is needed, unlike the reference release.
"""
import argparse
import ctypes as ct
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from endoscopic_tissue.grid import BoxUnion, build_grid
from endoscopic_tissue.native_backend import NativeTissue, Settings, Command


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', type=int, choices=(0, 1), required=True)
    parser.add_argument('--force-repair', action='store_true',
                        help='Exercise the R23 diagnostic repair export even below the ordinary residual threshold.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shape = BoxUnion(np.array([[-1., -.4, -1.], [-.4, 1., -.4]], dtype=np.float32),
                     np.array([[1., .4, 1.], [.4, 1.8, .4]], dtype=np.float32),
                     np.array([1., .4, 1.], dtype=np.float32))
    settings = Settings(young_pa=15000, contact_model=1, cutting_tool='scissors')
    grid = build_grid(.1, density=settings.density_kg_m3, shape=shape)
    park = Command(hinge=(0., 4., -.6), blade_center=(5., 5., 5.), blade_enabled=False)
    stages = [('settle', 60, park),
              ('approach', 90, replace(park, hinge=(0., 1.8, -.6))),
              ('close', 90, replace(park, hinge=(0., 1.8, -.6), closure=1.)),
              ('release', 60, replace(park, hinge=(0., 1.8, -.6), closure=0.)),
              ('retract', 60, park)]
    control = ct.CDLL(str(args.dll.resolve()))
    control.LCS_SetRepairEmptySpecialization.argtypes = [ct.c_int]
    control.LCS_SetRepairEmptySpecialization.restype = None
    control.LCS_SetRepairEmptySpecialization(args.mode)
    explicit_repair = None
    if args.force_repair:
        explicit_repair = control.LCS_ProbeReconstructedRepair
        explicit_repair.argtypes, explicit_repair.restype = [], ct.c_int
        control.LCS_SetRepairProofAudit.argtypes = [ct.c_int]
        control.LCS_SetRepairProofAudit.restype = None
        control.LCS_SetRepairProofAudit(1)
    fn = control.LCS_ReadRepairDispatchStats
    fn.argtypes, fn.restype = [ct.POINTER(ct.c_int), ct.c_int, ct.c_int], ct.c_int
    stats = (ct.c_int*4)()
    rows = []
    previous = park
    with NativeTissue(grid, settings, args.dll, park) as tissue:
        assert fn(stats, 4, 1) == 4
        for name, count, target in stages:
            for k in range(count):
                t = (k+1)/count
                command = replace(target,
                    hinge=tuple((1-t)*np.array(previous.hinge)+t*np.array(target.hinge)),
                    closure=(1-t)*previous.closure+t*target.closure)
                obs = tissue.step(command)
                retained = None
                if explicit_repair is not None and (len(rows)+1) % 30 == 0:
                    snapshot = args.output/f'paired-{len(rows)+1:04d}.fmc'
                    save = control.LCS_SavePairedConstraints
                    save.argtypes, save.restype = [ct.c_char_p], ct.c_int
                    assert save(str(snapshot.resolve()).encode('utf8')) == 0
                    with snapshot.open('rb') as source:
                        header = np.fromfile(source, dtype=np.int32, count=8)
                    assert header[0] == 0x464d4331 and header[5] == header[6]
                    retained = int(header[3])
                if explicit_repair is not None:
                    status = explicit_repair()
                    if status:
                        raise RuntimeError('Explicit diagnostic repair failed: '+str(status))
                    obs = tissue.observe()
                assert fn(stats, 4, 1) == 4
                quality = tissue.material_quality()
                row = dict(step=len(rows)+1, stage=name, empty=int(stats[0]), nonempty=int(stats[1]),
                    rebuilt=int(stats[2]), proof_audits=int(stats[3]), min_volume=float(quality[0]), invalid_tets=int(quality[1]),
                    retained_constraints_before_diagnostic_repair=retained,
                    jaw_load=list(obs.jaw_load_n), grip_points=obs.grip_points)
                rows.append(row)
                if k == count-1 or stats[1] or quality[1]:
                    print(json.dumps(row), flush=True)
                if k == count-1:
                    surface = tissue.surface()
                    np.savez_compressed(args.output/(name+'.npz'), nodes_cm=obs.nodes,
                        neighbors=obs.neighbors, vertices_cm=surface.vertices, faces=surface.triangles,
                        inside=grid.inside, rest_cm=grid.positions, pinned=grid.pinned)
                if quality[1] or obs.grip_points:
                    raise RuntimeError('Invalid tet or unexpected assisted grasp in pure-friction probe')
            previous = target
    result = dict(mode=args.mode, forced_diagnostic_repair=args.force_repair,
        steps=len(rows), empty=sum(r['empty'] for r in rows),
        nonempty=sum(r['nonempty'] for r in rows), rebuilt=sum(r['rebuilt'] for r in rows),
        proof_audits=sum(r['proof_audits'] for r in rows),
        retained_constraint_samples=[r['retained_constraints_before_diagnostic_repair'] for r in rows
                                     if r['retained_constraints_before_diagnostic_repair'] is not None],
        max_invalid_tets=max(r['invalid_tets'] for r in rows), rows=rows,
        material_nodes=int(np.sum(grid.inside)), dll_sha256=hashlib.sha256(args.dll.read_bytes()).hexdigest(),
        acceptance_scope='Branch coverage and finite noninverted states only; not surgical realism or timing.')
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)


if __name__ == '__main__':
    main()
