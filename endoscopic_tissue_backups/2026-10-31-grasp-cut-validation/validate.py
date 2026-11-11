"""Independent interaction probes. Never changes the default solver or acceptance rules.

Diagnostic stage failures are recorded and execution continues so release and late cuts
can be inspected. Nonfinite states and displacement beyond 60 mm stop execution.
Timing excludes Python diagnostics, snapshots and replay compression; it is not display FPS.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import struct
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from endoscopic_tissue.native_backend import NativeTissue
from endoscopic_tissue.scenario import (SCISSORS_SCENES, Sequence, Stage,
    drive_command, interpolate, scissors_settings, severed_edges, stage_start)
from endoscopic_tissue.instrument_geometry import scissors_body_depths
from endoscopic_tissue.surface_diagnostics import topology_statistics


CASES = ["grasp-assisted","grasp-friction-reference","grasp-gentle","grasp-offset-left","grasp-offset-right","grasp-deep","grasp-shallow","grasp-rotated","grasp-air","grasp-deeper","grasp-central","grasp-wall","grasp-flap","kidney-reference","kidney-one-snip","kidney-cut-disabled","kidney-yaw-plus","kidney-yaw-minus","kidney-repeat","kidney-friction-cut","wall-reference","wall-traction-reference","flap-reference"]

def fixture(name):
    if name not in CASES:
        raise ValueError(f"Unknown fixture: {name}")
    if name in ('grasp-wall', 'grasp-flap'):
        scene_name = 'wall-traction' if name == 'grasp-wall' else 'flap'
        scene = SCISSORS_SCENES[scene_name]
        settings = replace(scissors_settings(), young_pa=scene.young_pa, contact_model=1)
        stages = list(scene.stages[:next(i for i,s in enumerate(scene.stages) if s.name == 'lift')+1])
        bite = next(s.command for s in stages if s.name == 'close')
        raised = replace(bite, hinge=tuple(np.array(bite.hinge) + (0, .2, 0)))
        stages[-1] = replace(stages[-1], command=raised, steps=120)
        stages += [Stage('sustain', 90, raised), Stage('jaw_open', 45, replace(raised, closure=0)),
            Stage('open_retract', 60, replace(raised, closure=0, hinge=tuple(np.array(raised.hinge)+(0,1,0))))]
        return scene_name, scene, settings, stages, 'grasp'
    scene_name = next((n for n in ('wall-traction', 'wall', 'flap') if name.startswith(n)), 'kidney')
    scene = SCISSORS_SCENES[scene_name]
    settings = replace(scissors_settings(), young_pa=scene.young_pa, contact_model=scene.contact_model)
    stages = list(scene.stages)
    kind = 'grasp' if name.startswith('grasp-') else 'cut'
    if kind == 'grasp':
        model = 2 if name == 'grasp-assisted' else 1
        settings = replace(settings, contact_model=model)
        stages = stages[:5]
        # The reference trajectory lifts 16 mm. Gentle probes lift 4 mm, slowly.
        lift_cm = 1.6 if name in ('grasp-assisted', 'grasp-friction-reference') else .4
        shift = np.zeros(3)
        if name == 'grasp-offset-left': shift[0] = -.1
        if name == 'grasp-offset-right': shift[0] = .1
        if name == 'grasp-shallow': shift[1] = .15
        if name == 'grasp-deep': shift[1] = -.1
        if name == 'grasp-deeper': shift[1] = -.3
        if name == 'grasp-central': shift[:2] = (-.3, -.1)
        if name == 'grasp-air': shift[0] = 4.0
        if name == 'grasp-rotated':
            a = np.radians(10.)
            rotation = np.array(((np.cos(a), -np.sin(a), 0), (np.sin(a), np.cos(a), 0), (0, 0, 1)))
            old_axis = np.array(settings.jaw_axis)
            settings = replace(settings, jaw_axis=tuple(rotation @ old_axis))
            # Keep the pad centre approximately at the same bite location.
            shift += 1.5 * (old_axis - rotation @ old_axis)
        stages = [replace(s, command=replace(s.command, hinge=tuple(np.array(s.command.hinge) + shift))) for s in stages]
        bite = stages[3].command
        raised = replace(bite, hinge=tuple(np.array(bite.hinge) + (0, lift_cm, 0)))
        stages[4] = replace(stages[4], command=raised)
        stages += [Stage('sustain', 90, raised), Stage('jaw_open', 45, replace(raised, closure=0)),
                   Stage('open_retract', 60, replace(raised, closure=0, hinge=tuple(np.array(raised.hinge) + (0, 1.0, 0)))),
                   Stage('relax', 45, replace(raised, closure=0, hinge=tuple(np.array(raised.hinge) + (0, 1.0, 0))))]
        if name == 'grasp-gentle':
            stages += [Stage('return_open', 90, replace(bite, closure=0)), Stage('reclose', 60, bite),
                       Stage('relift', 90, raised), Stage('rehold', 60, raised),
                       Stage('reopen', 45, replace(raised, closure=0))]
    elif name == 'kidney-friction-cut':
        settings = replace(settings, contact_model=1)
    elif name == 'kidney-one-snip':
        stages = [s for s in stages if s.name not in ('open_1', 'reposition_2', 'incise_2')]
    elif name == 'kidney-cut-disabled':
        stages = [replace(s, command=replace(s.command, scissors_cutting=False)) for s in stages]
    elif name in ('kidney-yaw-plus', 'kidney-yaw-minus'):
        delta = 5. if name.endswith('plus') else -5.
        a = np.radians(delta)
        rotation = np.array(((np.cos(a), 0, np.sin(a)), (0, 1, 0), (-np.sin(a), 0, np.cos(a))))
        pivot = np.array((.06, scene.plane - .0282, .01))
        stages = [replace(s, command=replace(s.command,
                  blade_center=tuple(pivot + rotation @ (np.array(s.command.blade_center) - pivot)),
                  scissors_yaw_deg=s.command.scissors_yaw_deg + delta)) for s in stages]
    elif name == 'kidney-repeat':
        index = next(i for i, s in enumerate(stages) if s.name == 'withdraw')
        first = next(s.command for s in stages if s.name == 'incise_1')
        second = next(s.command for s in stages if s.name == 'incise_2')
        stages[index:index] = [Stage('open_repeat', 45, replace(second, scissors_closure=0, scissors_cutting=False)),
            Stage('reposition_repeat', 60, replace(first, scissors_closure=0, scissors_cutting=False)),
            Stage('incise_repeat_1', 150, first, ramp=.5),
            Stage('open_repeat_2', 30, replace(first, scissors_closure=0, scissors_cutting=False)),
            Stage('reposition_repeat_2', 60, replace(second, scissors_closure=0, scissors_cutting=False)),
            Stage('incise_repeat_2', 150, second, ramp=.5)]
    if kind == 'cut' and scene_name == 'kidney':
        held = stages[-1].command
        side = replace(held, hinge=tuple(np.array(held.hinge) + (.4, 0, 0)))
        stages += [Stage('side_pull', 60, side), Stage('side_hold', 45, side),
                   Stage('jaw_open', 45, replace(side, closure=0)),
                   Stage('open_retract', 60, replace(side, closure=0, hinge=tuple(np.array(side.hinge) + (0, 1, 0))))]
    return scene_name, scene, settings, stages, kind


def pack_frame(stream, step, stage, surface, obs):
    if stream is None: return
    light = replace(obs, nodes=np.empty((0, 3), np.float32), neighbors=np.empty((0, 6), np.int32),
                    grip_flags=np.empty(0, np.int32))
    data = zlib.compress(pickle.dumps((step, stage, surface, light), protocol=5), 1)
    stream.write(struct.pack('<I', len(data)))
    stream.write(data)


def summary(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                maximum=float(values.max())) if len(values) else {}


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene_name, scene, settings, stages, kind = fixture(args.case)
    if kind == 'grasp' and (any(s.command.scissors_cutting for s in stages)
                           or not {'close','lift'} <= {s.name for s in stages}):
        raise ValueError('A grasp-only fixture must close and lift without enabling scissors cutting')
    grid = scene.build_grid(settings.density_kg_m3)
    metadata = dict(case=args.case, scene=scene_name, kind=kind, settings=asdict(settings),
        dll=str(args.dll.resolve()), dll_sha256=hashlib.sha256(args.dll.read_bytes()).hexdigest(),
        material_nodes=int(np.count_nonzero(grid.inside)), active_nodes=int(np.count_nonzero(grid.active)),
        total_grid_nodes=len(grid.positions), cell_size_cm=grid.cell_size.tolist(),
        stages=[asdict(s) for s in stages], capture=args.capture,
        note='Diagnostics continue past recorded quality failures. Replay follows simulation time, not achieved FPS.')
    (args.output/'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    rows, audits, errors = [], [], []
    healed, outside_cut, no_fracture_cut = 0, 0, 0
    bite_ids = np.array([], dtype=int)
    bite_local = bite_positions = None
    stage_rows = []
    started = time.perf_counter()
    stream = (args.output/'replay.bin').open('wb') if args.capture else None
    if stream: pickle.dump((settings, scene_name, args.case + ': exact solved surface, diagnostic run'), stream, protocol=5)
    seq = None
    try:
        with NativeTissue(grid, settings, args.dll, stages[0].command) as tissue, (args.output/'trajectory.jsonl').open('w', encoding='utf8') as log:
            obs = tissue.observe()
            initial_neighbors = obs.neighbors
            previous_neighbors = obs.neighbors
            previous_fractures = 0
            seq = Sequence(obs, settings, scene, grid.positions, grid.inside)
            # Geometric bite tracking works for both native contact models, independently of anchors.
            probe = Sequence(obs, replace(settings, contact_model=1), scene, grid.positions, grid.inside)
            material = grid.inside != 0
            tumor = material & (scene.shape.tumor_weight(grid.positions, 1e-3) > .5) if scene.shape else material
            body = material & ~tumor if scene.shape else material
            surface = tissue.surface()
            pack_frame(stream, 0, 'initial', surface, obs)
            original_check = tissue.check
            calls = Counter()
            def timed_check(name, *values):
                t = time.perf_counter()
                try: return original_check(name, *values)
                finally: calls[name] += (time.perf_counter()-t)*1000
            tissue.check = timed_check
            previous = stages[0].command
            for stage in stages:
                print(f'[{len(rows):04d}] {args.case}: {stage.name}', flush=True)
                previous = stage_start(stage.name, previous, obs)
                first_row = len(rows)
                for k in range(stage.steps):
                    command = drive_command(interpolate(previous, stage.command, (k+1)/(stage.steps*stage.ramp)), obs, settings)
                    calls.clear()
                    t = time.perf_counter()
                    obs = tissue.step(command)
                    surface = tissue.surface()
                    pipeline_ms = (time.perf_counter()-t)*1000
                    physics_ms = calls['LCS_Step']
                    changed = obs.neighbors is not previous_neighbors
                    cuts = int(severed_edges(initial_neighbors, obs.neighbors))
                    if changed:
                        healed += int(np.count_nonzero((initial_neighbors[:,1::2]>=0) & (previous_neighbors[:,1::2]<0) & (obs.neighbors[:,1::2]>=0)))
                        new = int(severed_edges(previous_neighbors, obs.neighbors))
                        if not command.scissors_cutting: outside_cut += new
                        if obs.scissors.loads.fractures == 0: no_fracture_cut += new
                    previous_neighbors = obs.neighbors
                    if stage.name in ('close','reclose') and k == stage.steps-1:
                        enclosed, local = probe.between_pads(obs.nodes, obs)
                        bite_ids = np.flatnonzero(enclosed & material)
                        bite_local, bite_positions = local[bite_ids].copy(), obs.nodes[bite_ids].copy()
                    held, slip, lift = 0, None, None
                    if len(bite_ids):
                        enclosed, local = probe.between_pads(obs.nodes[bite_ids], obs)
                        held = int(np.count_nonzero(enclosed))
                        slip = float(np.median(np.linalg.norm((local-bite_local)[:,::2], axis=1))*10)
                        lift = float(np.median(obs.nodes[bite_ids,1]-bite_positions[:,1])*10)
                    depth, wrist = scissors_body_depths(surface.vertices, surface.triangles, tissue.scissors_geometry,
                        obs.scissors.hinge, obs.scissors.axis, obs.scissors.normal, obs.scissors.gap)
                    quality = tissue.material_quality()
                    displacement = float(np.linalg.norm(obs.nodes[grid.active.astype(bool)]-grid.positions[grid.active.astype(bool)],axis=1).max()*10)
                    if displacement > 60:
                        np.savez_compressed(args.output/'failure-state.npz', nodes_cm=obs.nodes,
                            rest_cm=grid.positions, inside=grid.inside, neighbors=obs.neighbors,
                            vertices_cm=surface.vertices, faces=surface.triangles, normals=surface.normals,
                            anchors=surface.anchors, grip_flags=obs.grip_flags, hinge_cm=obs.hinge,
                            jaw_gap_cm=obs.jaw_gap, blade=obs.blade, grip_points=obs.grip_points,
                            reaction_n=obs.reaction_n, grip_error_m=obs.grip_error_m,
                            scissors_edges_cm=obs.scissors.edges, scissors_gap_cm=obs.scissors.gap)
                        (args.output/'failure-state.json').write_text(json.dumps(dict(step=len(rows)+1,
                            stage=stage.name, max_displacement_mm=displacement,
                            tumor_com_mm=(obs.nodes[tumor].mean(axis=0)*10).tolist())),encoding='utf8')
                        raise RuntimeError('Material left the 60 mm operating envelope')
                    row = dict(step=len(rows)+1, stage=stage.name, physics_ms=physics_ms, pipeline_ms=pipeline_ms,
                        grip_points=obs.grip_points, jaw_gap_mm=obs.jaw_gap*10, jaw_load_n=list(obs.jaw_load_n),
                        bite_nodes=len(bite_ids), held_bite_nodes=held, pad_slip_mm=slip, bite_lift_mm=lift,
                        hinge_cm=obs.hinge.tolist(), cut_edges=cuts, fractures=obs.scissors.loads.fractures,
                        scissors_gap_mm=obs.scissors.gap*10, scissors_load_n=[obs.scissors.loads.left_n,obs.scissors.loads.right_n],
                        scissors_force_n=obs.scissors.pressure_n, blade_overlap_mm=float(depth)*10,
                        wrist_overlap_mm=float(wrist)*10, min_volume_ratio=float(quality[0]), invalid_tets=int(quality[1]),
                        max_displacement_mm=displacement, tumor_com_mm=(obs.nodes[tumor].mean(axis=0)*10).tolist(),
                        body_com_mm=(obs.nodes[body].mean(axis=0)*10).tolist() if body.any() else None)
                    rows.append(row)
                    log.write(json.dumps(row)+'\n')
                    seq.record_step(stage.name, obs, max(float(depth),float(wrist))*.01, float(depth)*.01)
                    pack_frame(stream, len(rows), stage.name, surface, obs)
                # These original checks retain their original criteria, including known skin-cut failures.
                if kind == 'cut' or stage.name in ('close','lift'):
                    if stage.name == 'lift' and not np.any(seq.tracked):
                        errors.append(dict(stage=stage.name, error='No captured material; lift is undefined'))
                    else:
                        try: seq.finish_stage(stage.name, obs, surface)
                        except RuntimeError as error: errors.append(dict(stage=stage.name,error=str(error)))
                audit = topology_statistics(surface)
                p = seq.pieces(obs.neighbors, material)
                lattice = seq.pieces(obs.neighbors, np.ones(len(material), dtype=bool))
                components, sizes = np.unique(p[material], return_counts=True)
                audit.update(stage=stage.name, step=len(rows),
                    material_components=len(components), material_component_sizes=sorted(sizes.tolist(),reverse=True),
                    lattice_components=len(np.unique(lattice[material])),
                    stray_skin_cuts=seq.stray_skin_cuts(obs.neighbors),
                    max_triangle_edge_mm=float(np.linalg.norm(surface.vertices[surface.triangles]-
                        surface.vertices[np.roll(surface.triangles,1,axis=1)],axis=2).max()*10))
                audits.append(audit)
                np.savez_compressed(args.output/(stage.name+'.npz'), nodes_cm=obs.nodes, neighbors=obs.neighbors,
                    vertices_cm=surface.vertices, faces=surface.triangles, normals=surface.normals, anchors=surface.anchors,
                    grip_flags=obs.grip_flags, hinge_cm=obs.hinge, jaw_gap_cm=obs.jaw_gap, blade=obs.blade,
                    grip_points=obs.grip_points, reaction_n=obs.reaction_n, grip_error_m=obs.grip_error_m,
                    material_labels=p, lattice_labels=lattice, inside=grid.inside, rest_cm=grid.positions,
                    scissors_edges_cm=obs.scissors.edges, scissors_gap_cm=obs.scissors.gap)
                stage_rows.append(dict(stage=stage.name, end=rows[-1], audit=audit,
                    physics_ms=summary([r['physics_ms'] for r in rows[first_row:]]),
                    pipeline_ms=summary([r['pipeline_ms'] for r in rows[first_row:]])))
                print(json.dumps(stage_rows[-1]),flush=True)
                previous = stage.command
                if stream: stream.flush()
    except Exception as error:
        errors.append(dict(stage='fatal', error=repr(error)))
        raise
    finally:
        if stream: stream.close()
        report = dict(metadata=metadata, steps=len(rows), expected_steps=sum(s.steps for s in stages),
            completed=len(rows)==sum(s.steps for s in stages), elapsed_s=time.perf_counter()-started,
            diagnostic_errors=errors, original_metrics=asdict(seq.metrics) if seq else None,
            healed_connections=healed, cuts_while_disabled=outside_cut, cuts_before_any_fracture=no_fracture_cut,
            physics_ms=summary([r['physics_ms'] for r in rows if r['step']>30]),
            pipeline_ms=summary([r['pipeline_ms'] for r in rows if r['step']>30]),
            max_blade_overlap_mm=max((r['blade_overlap_mm'] for r in rows),default=None),
            max_wrist_overlap_mm=max((r['wrist_overlap_mm'] for r in rows),default=None),
            max_invalid_tets=max((r['invalid_tets'] for r in rows),default=None),
            stages=stage_rows)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')


if __name__ == '__main__':
    main()
