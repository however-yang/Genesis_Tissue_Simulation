"""Real release/regrasp and native snapshot replay after the original full demo.

Separate opt-in verification; no change to the demonstration or solver defaults.
Every accepted extra step is recorded into the same complete interaction video.
"""
from dataclasses import asdict, replace
import hashlib
import json
import time
import numpy as np


def install(run, output):
    original = run.Flow.run
    def run_with_stress(flow, tissue, trajectory, sink):
        original(flow, tissue, trajectory, sink)
        if not flow.sequence.metrics.acceptance_passed:
            return
        from endoscopic_tissue.scenario import interpolate
        from endoscopic_tissue.training.snapshot import SnapshotBackend
        from endoscopic_tissue.surface_diagnostics import topology_statistics
        from envelope_audit import capture, components
        import trimesh

        result = dict(passed=False, stages=[], replay={}, contact_model=flow.settings.contact_model)
        report = output/'training-stress.json'
        def save_report():
            report.write_text(json.dumps(result, indent=2)+'\n')
        try:
            inside = flow.grid.inside != 0
            groups = components(flow.obs.neighbors, flow.grid.active != 0)
            held_groups = np.unique(groups[inside & (flow.obs.grip_flags != 0)])
            assert len(held_groups)==1, 'No unique held excision'
            piece = inside & (groups == held_groups[0])
            initial_neighbors = flow.obs.neighbors.copy()
            command = flow.stages[-1].command
            archive = SnapshotBackend(tissue)
            checkpoint = archive.save()
            tick = tissue.tick
            before = tissue.observe()
            flags = capture(tissue.dll, len(flow.grid.edges))['flags']
            # Short continuation, exact state restore, then the same commands.
            first = [tissue.step(command).nodes.copy() for _ in range(8)]
            archive.load(checkpoint)
            tissue.tick, tissue.topology_revision, tissue.neighbors = tick, None, None
            restored = tissue.observe()
            assert np.array_equal(restored.nodes, before.nodes), 'Position restore mismatch'
            assert np.array_equal(restored.neighbors, before.neighbors), 'Topology restore mismatch'
            assert np.array_equal(capture(tissue.dll,len(flow.grid.edges))['flags'],flags), 'Provenance restore mismatch'
            second = [tissue.step(command).nodes.copy() for _ in range(8)]
            errors = [float(np.linalg.norm(a-b,axis=1).max())*10 for a,b in zip(first,second)]
            result['replay'] = dict(snapshot_bytes=len(checkpoint), snapshot_sha256=hashlib.sha256(checkpoint).hexdigest(),
                                    exact_restore=True, continuation_max_error_mm=max(errors), steps=8)
            assert max(errors)<.05, 'Restored continuation diverged over 0.05 mm'
            archive.load(checkpoint)
            tissue.tick, tissue.topology_revision, tissue.neighbors = tick, None, None
            flow.obs = tissue.observe()
            del checkpoint,first,second

            previous = command
            extra_rows=[]
            def stage(name, steps, target):
                nonlocal previous
                flow.stage=name
                print(f'[{flow.step:03d}] {name}',flush=True)
                timing=[]
                for i in range(steps):
                    request=interpolate(previous,target,min(1.,(i+1)/(steps*.8)))
                    start=time.perf_counter();flow.obs=tissue.step(request);elapsed=(time.perf_counter()-start)*1000
                    timing.append(elapsed);flow.step+=1
                    obs=flow.obs
                    assert np.isfinite(obs.nodes).all(), 'Nonfinite material state'
                    assert np.array_equal(obs.neighbors,initial_neighbors), 'Unexpected cut/healing after excision'
                    surface=tissue.surface()
                    if sink is not None and not sink.after_step(flow,tissue,surface):
                        raise RuntimeError('Stress playback interrupted')
                    extra_rows.append(dict(step=flow.step,stage=name,step_readback_ms=elapsed,
                        grip_points=int(obs.grip_points),piece_center_cm=obs.nodes[piece].mean(axis=0).tolist(),
                        blade_force_n=(obs.blade[3:6]*.01).tolist(),command=asdict(request)))
                previous=target
                flow.surface=surface=tissue.surface()
                if sink is not None:sink.after_stage(flow,name,surface)
                np.savez_compressed(output/(name+'.npz'),nodes_cm=flow.obs.nodes,neighbors=flow.obs.neighbors,
                    vertices_cm=surface.vertices,faces=surface.triangles,grip_flags=flow.obs.grip_flags,piece=piece)
                audit=topology_statistics(surface)
                assert not (audit['degenerate_triangles'] or audit['positional_boundary_edges'] or audit['positional_nonmanifold_edges'] or audit['duplicate_triangles']) and audit['winding_consistent'], 'Invalid stress surface'
                # Independent final-stage interior test against the main organ.
                mesh=trimesh.Trimesh(surface.vertices,surface.triangles,process=False)
                mesh.merge_vertices(digits_vertex=6)
                meshes=mesh.split(only_watertight=False)
                organ=max(meshes,key=lambda m:len(m.faces))
                nodes=flow.obs.nodes[piece]
                # All excised MATERIAL nodes, not a convenient hand-picked sample.
                depth=trimesh.proximity.signed_distance(organ,nodes)
                maximum=float(max(0.,depth.max()))*10
                item=dict(stage=name,steps=steps,step_readback_mean_ms=float(np.mean(timing)),
                          step_readback_p95_ms=float(np.percentile(timing,95)),
                          grip_points=int(flow.obs.grip_points),piece_mean_y_mm=float(nodes[:,1].mean()*10),
                          main_organ_watertight=bool(organ.is_watertight),piece_nodes_inside_organ=int((depth>.02).sum()),
                          max_piece_node_depth_mm=maximum,surface_audit=audit)
                result['stages'].append(item);save_report()
                assert organ.is_watertight and maximum<=.2, 'Released piece penetrated main organ by over 0.2 mm'
                return item

            opened=replace(command,closure=0,scissors_cutting=False)
            stage('grasp_release',45,opened)
            assert flow.obs.grip_points==0,'Opening failed to release anchors'
            stage('drop_settle',90,opened)
            away=replace(opened,hinge=tuple(np.asarray(opened.hinge)+(-.7,.7,0)))
            stage('grasp_withdraw',60,away)
            nodes=flow.obs.nodes[piece]
            center=nodes.mean(axis=0);center[1]=np.quantile(nodes[:,1],.85)
            hinge=center-np.asarray(flow.settings.jaw_axis)*flow.settings.jaw_length*.82
            ready=replace(away,hinge=tuple(hinge+(0,.8,0)))
            stage('regrasp_above',60,ready)
            bite=replace(ready,hinge=tuple(hinge))
            stage('regrasp_approach',60,bite)
            held=replace(bite,closure=1)
            stage('regrasp_close',45,held)
            assert flow.obs.grip_points>0,'Reachable regrasp did not capture material'
            y0=float(flow.obs.nodes[piece,1].mean())
            lifted=replace(held,hinge=tuple(hinge+(0,.8,0)))
            stage('regrasp_lift',75,lifted)
            stage('regrasp_hold',30,lifted)
            result['regrasp_lift_mm']=(float(flow.obs.nodes[piece,1].mean())-y0)*10
            assert flow.obs.grip_points>0 and result['regrasp_lift_mm']>=5,'Regrasp did not lift by 5 mm'
            result['passed']=True
        except Exception as error:
            result['error']=str(error)
            flow.sequence.metrics.acceptance_passed=False
            raise
        finally:
            if 'extra_rows' in locals():
                (output/'stress-trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in extra_rows))
            save_report()
    run.Flow.run=run_with_stress
