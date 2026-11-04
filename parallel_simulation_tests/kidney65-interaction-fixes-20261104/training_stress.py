"""Real release/regrasp and native snapshot replay after the original full demo.

Separate opt-in verification; no change to the demonstration or solver defaults.
Every accepted extra step is recorded into the same complete interaction video.
"""
from dataclasses import asdict, replace
import hashlib
import json
import time
import numpy as np


def install(run, output, profile=False, route='legacy', diagnostic=False, replay_repeats=2):
    original = run.Flow.run
    original_open = run.Flow.open
    def open_with_checkpoint(flow,tissue):
        original_open(flow,tissue)
        from endoscopic_tissue.training.snapshot import SnapshotBackend
        flow.stress_initial_state = SnapshotBackend(tissue).save()
        flow.stress_initial_nodes = flow.obs.nodes.copy()
    run.Flow.open = open_with_checkpoint
    def run_with_stress(flow, tissue, trajectory, sink):
        base_error=None
        try:
            original(flow, tissue, trajectory, sink)
        except RuntimeError as error:
            if not diagnostic or not flow.sequence.metrics.completed:
                raise
            base_error=str(error)
        base_passed=bool(flow.sequence.metrics.acceptance_passed)
        if not base_passed and not diagnostic:
            return
        from endoscopic_tissue.scenario import interpolate
        from endoscopic_tissue.training.snapshot import SnapshotBackend
        from endoscopic_tissue.surface_diagnostics import topology_statistics
        from envelope_audit import capture, components
        import trimesh

        result = dict(passed=False, stages=[], replay={}, contact_model=flow.settings.contact_model,
                      route=route, original_sequence_passed=base_passed, original_error=base_error,
                      diagnostic_continuation=diagnostic)
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
            archive.load(flow.stress_initial_state)
            tissue.tick,tissue.topology_revision,tissue.neighbors=0,None,None
            initial=tissue.observe()
            assert np.array_equal(initial.nodes,flow.stress_initial_nodes), 'Pre-cut restore mismatch'
            assert not capture(tissue.dll,len(flow.grid.edges))['flags'].any(), 'Release provenance leaked across reset'
            archive.load(checkpoint)
            tissue.tick,tissue.topology_revision,tissue.neighbors=tick,None,None
            del flow.stress_initial_state,flow.stress_initial_nodes
            # Short continuation, exact state restore, then the same commands.
            tissue.timing_phase='replay'
            def continuation(repeat):
                trace=hasattr(tissue.dll,'LCS_KidneyTraceBegin')
                if trace:
                    import ctypes
                    begin=tissue.dll.LCS_KidneyTraceBegin;begin.argtypes=[ctypes.c_int];begin.restype=ctypes.c_int
                    end=tissue.dll.LCS_KidneyTraceEnd;end.argtypes=[ctypes.c_void_p,ctypes.c_int];end.restype=ctypes.c_int
                    assert begin(int(repeat==0))==0
                try:
                    nodes=[tissue.step(command).nodes.copy()]
                finally:
                    if trace:
                        buffer=ctypes.create_string_buffer(max(1,end(None,0)))
                        assert end(buffer,len(buffer))==0
                        (output/f'replay-stage-{repeat:02d}.tsv').write_text(buffer.value.decode())
                return nodes+[tissue.step(command).nodes.copy() for _ in range(7)]
            first = continuation(0)
            trials=[]
            archive_restore_exact=None
            for repeat in range(1,replay_repeats):
                archive.load(checkpoint)
                if repeat==1:
                    restored_archive=archive.save()
                    archive_restore_exact=restored_archive==checkpoint
                    result['archive_restore_exact']=archive_restore_exact
                    if not archive_restore_exact:
                        result['archive_restore_sizes']=[len(checkpoint),len(restored_archive)]
                        block=1024*1024
                        for offset in range(0,min(len(checkpoint),len(restored_archive)),block):
                            a=checkpoint[offset:offset+block];b=restored_archive[offset:offset+block]
                            if a!=b:
                                unequal=np.flatnonzero(np.frombuffer(a,np.uint8)!=np.frombuffer(b,np.uint8))
                                result['archive_first_differing_byte']=offset+int(unequal[0]);break
                    del restored_archive
                tissue.tick, tissue.topology_revision, tissue.neighbors = tick, None, None
                restored = tissue.observe()
                assert np.array_equal(restored.nodes, before.nodes), 'Position restore mismatch'
                assert np.array_equal(restored.neighbors, before.neighbors), 'Topology restore mismatch'
                assert np.array_equal(capture(tissue.dll,len(flow.grid.edges))['flags'],flags), 'Provenance restore mismatch'
                second = continuation(repeat)
                errors = [float(np.linalg.norm(a-b,axis=1).max())*10 for a,b in zip(first,second)]
                replay_detail=[]
                for a,b in zip(first,second):
                    distance=np.linalg.norm(a-b,axis=1)*10
                    worst=int(distance.argmax())
                    replay_detail.append(dict(worst_node=worst,material=bool(inside[worst]),
                        active=bool(flow.grid.active[worst]),piece=bool(piece[worst]),
                        material_max_mm=float(distance[inside].max()),
                        material_rms_mm=float(np.sqrt(np.mean(distance[inside]**2))),
                        piece_max_mm=float(distance[piece].max()),
                        main_max_mm=float(distance[inside & ~piece].max())))
                trials.append(dict(repeat=repeat,errors_mm=errors,per_step_detail=replay_detail))
            errors=np.max([trial['errors_mm'] for trial in trials],axis=0).tolist()
            result['replay'] = dict(snapshot_bytes=len(checkpoint), snapshot_sha256=hashlib.sha256(checkpoint).hexdigest(),
                                    exact_restore=True, complete_archive_exact=archive_restore_exact,precut_restore_clears_release_flags=True,
                                    continuation_max_error_mm=max(errors), continuation_errors_mm=errors,
                                    trials=trials,repeats=replay_repeats,
                                    passed=max(errors)<.05, steps=8)
            archive.load(checkpoint)
            tissue.tick, tissue.topology_revision, tissue.neighbors = tick, None, None
            flow.obs = tissue.observe()
            tissue.timing_phase='trajectory'
            del checkpoint,first,second

            previous = command
            extra_rows=[]
            def stage(name, steps, target, track=False):
                nonlocal previous
                flow.stage=name
                print(f'[{flow.step:03d}] {name}',flush=True)
                timing=[]
                max_body_depth=0.;min_volume=1.;invalid_tets=0
                if profile:
                    import ctypes
                    enable=tissue.dll.LCS_ProfileSections;enable.argtypes=[ctypes.c_int];enable.restype=None
                    enable(1)
                for i in range(steps):
                    if track:
                        center=flow.obs.nodes[piece].mean(axis=0)
                        next_hinge=center-np.asarray(flow.settings.jaw_axis)*flow.settings.jaw_length*.82
                        # Follow lateral drift only. Chasing a compressed upper
                        # quantile vertically pushes the piece further down.
                        next_hinge[1]=target.hinge[1]
                        target=replace(target,hinge=tuple(next_hinge))
                    request=interpolate(previous,target,min(1.,(i+1)/(steps*.8)))
                    start=time.perf_counter();flow.obs=tissue.step(request);elapsed=(time.perf_counter()-start)*1000
                    timing.append(elapsed);flow.step+=1
                    obs=flow.obs
                    assert np.isfinite(obs.nodes).all(), 'Nonfinite material state'
                    assert np.array_equal(obs.neighbors,initial_neighbors), 'Unexpected cut/healing after excision'
                    surface=tissue.surface()
                    material=tissue.material_quality()
                    min_volume=min(min_volume,float(material[0]));invalid_tets=max(invalid_tets,int(material[1]))
                    assert np.isfinite(material).all() and invalid_tets==0, 'Invalid material tetrahedra'
                    from endoscopic_tissue.instrument_geometry import scissors_body_depths
                    pose=obs.scissors
                    body_depth=max(scissors_body_depths(surface.vertices,surface.triangles,tissue.scissors_geometry,
                                      pose.hinge,pose.axis,pose.normal,pose.gap))*10
                    max_body_depth=max(max_body_depth,float(body_depth))
                    assert max_body_depth<=.2, 'Instrument penetrated tissue by over 0.2 mm during stress'
                    displacement=float(np.linalg.norm(obs.nodes-flow.grid.positions,axis=1).max())*10
                    assert displacement<=60, 'Material left the 60 mm operating envelope'
                    if sink is not None and not sink.after_step(flow,tissue,surface):
                        raise RuntimeError('Stress playback interrupted')
                    extra_rows.append(dict(step=flow.step,stage=name,step_readback_ms=elapsed,
                        grip_points=int(obs.grip_points),piece_center_cm=obs.nodes[piece].mean(axis=0).tolist(),
                        actual_hinge_cm=obs.hinge.tolist(),actual_gap_cm=float(obs.jaw_gap),
                        material_quality=material.tolist(),body_overlap_mm=float(body_depth),
                        blade_force_n=(obs.blade[3:6]*.01).tolist(),command=asdict(request)))
                section_ms={}
                if profile:
                    read=tissue.dll.LCS_ProfileSectionsRead;read.argtypes=[ctypes.c_void_p,ctypes.c_int];read.restype=ctypes.c_int
                    buffer=ctypes.create_string_buffer(max(1,read(None,0)))
                    assert read(buffer,len(buffer))==0
                    section_ms={k:float(v)/steps for k,v in (line.rsplit('=',1) for line in buffer.value.decode().splitlines())}
                    enable(0)
                previous=target
                flow.surface=surface=tissue.surface()
                if sink is not None:sink.after_stage(flow,name,surface)
                np.savez_compressed(output/(name+'.npz'),nodes_cm=flow.obs.nodes,neighbors=flow.obs.neighbors,
                    vertices_cm=surface.vertices,faces=surface.triangles,grip_flags=flow.obs.grip_flags,piece=piece)
                if name in ('drop_settle','regrasp_close') and hasattr(tissue.dll,'LCS_SaveFragmentCase'):
                    import ctypes
                    export=tissue.dll.LCS_SaveFragmentCase;export.argtypes=[ctypes.c_char_p];export.restype=ctypes.c_int
                    assert export(str(output/(name+'.fragment')).encode())==0,'Fragment case export failed'
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
                item.update(max_body_overlap_mm=max_body_depth,min_volume_ratio=min_volume,invalid_tets=invalid_tets)
                if profile:item['profile_gpu_sections_ms_per_step']=section_ms
                result['stages'].append(item);save_report()
                assert organ.is_watertight and maximum<=.2, 'Released piece penetrated main organ by over 0.2 mm'
                return item

            if route=='tracked':
                # Move the other instrument clear before the real release.
                # This is a controller trajectory, not a change in physics.
                from endoscopic_tissue.native_backend import scissors_frame
                axis=scissors_frame(command.scissors_pitch_deg,command.scissors_roll_deg,command.scissors_yaw_deg)[0]
                command=replace(command,blade_center=tuple(np.asarray(command.blade_center)-axis*3+(0,1,0)),scissors_cutting=False)
                stage('park_scissors',45,command)
                # Place the held piece just above the organ before letting go.
                # Native tools have a velocity limit; a short command jump is
                # not proof the actual jaws have reached a regrasp position.
                surface=tissue.surface()
                mesh=trimesh.Trimesh(surface.vertices,surface.triangles,process=False)
                mesh.merge_vertices(digits_vertex=6)
                organ=max(mesh.split(only_watertight=False),key=lambda m:len(m.faces))
                nodes=flow.obs.nodes[piece]
                origin=nodes.mean(axis=0);origin[1]=nodes[:,1].max()+1
                hits,_,_=organ.ray.intersects_location([origin],[[0,-1,0]])
                assert len(hits)>0,'No organ below held piece for placement'
                down=max(0.,float(nodes[:,1].min()-hits[:,1].max()-.1))
                command=replace(command,hinge=tuple(np.asarray(command.hinge)-(0,down,0)))
                stage('place_for_release',max(45,int(np.ceil(down/.02))),command)
            opened=replace(command,closure=0,scissors_cutting=False)
            stage('grasp_release',30 if route=='tracked' else 45,opened)
            assert flow.obs.grip_points==0,'Opening failed to release anchors'
            stage('drop_settle',10 if route=='tracked' else 90,opened)
            away=replace(opened,hinge=tuple(np.asarray(opened.hinge)+((-.3,.3,0) if route=='tracked' else (-.7,.7,0))))
            stage('grasp_withdraw',15 if route=='tracked' else 60,away)
            nodes=flow.obs.nodes[piece]
            center=nodes.mean(axis=0);center[1]=np.quantile(nodes[:,1],.85)
            hinge=center-np.asarray(flow.settings.jaw_axis)*flow.settings.jaw_length*.82
            ready=replace(away,hinge=tuple(hinge+(0,.8,0)))
            stage('regrasp_above',20 if route=='tracked' else 60,ready)
            bite=replace(ready,hinge=tuple(hinge))
            stage('regrasp_approach',45 if route=='tracked' else 60,bite,track=route=='tracked')
            held=replace(previous,closure=1)
            stage('regrasp_close',45,held,track=route=='tracked')
            assert flow.obs.grip_points>0,'Reachable regrasp did not capture material'
            flags=flow.obs.grip_flags!=0
            result['regrasp_material_nodes']=int((flags&piece).sum())
            result['regrasp_other_material_nodes']=int((flags&inside&~piece).sum())
            assert result['regrasp_material_nodes']>0 and result['regrasp_other_material_nodes']==0,'Regrasp captured the wrong material component'
            y0=float(flow.obs.nodes[piece,1].mean())
            # A compliant driven tool may lag its command. Measure and request
            # the lift from the actual closed pose, never the unachieved target.
            lifted=replace(previous,hinge=tuple(np.asarray(flow.obs.hinge)+(0,.8,0)))
            stage('regrasp_lift',75,lifted)
            stage('regrasp_hold',30,lifted)
            result['regrasp_lift_mm']=(float(flow.obs.nodes[piece,1].mean())-y0)*10
            assert flow.obs.grip_points>0 and result['regrasp_lift_mm']>=5,'Regrasp did not lift by 5 mm'
            result['physical_checks_passed']=True
            assert base_passed, 'Original sequence failed; diagnostic stress cannot certify it'
            assert archive_restore_exact,'Full native archive failed exact roundtrip'
            assert result['replay']['passed'], 'Restored continuation diverged over 0.05 mm'
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
