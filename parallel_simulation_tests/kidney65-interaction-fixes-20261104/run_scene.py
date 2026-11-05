"""Live Windows demo and isolated Linux comparison using the same scene code."""
import argparse
import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
GPU = "GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa"
SOURCE = PROJECT / "parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check"
CANDIDATE = PROJECT / "parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("baseline", "mesh65", "frontal65"), default="frontal65")
    parser.add_argument("--lateral-mm", type=float, choices=(1.225, 1.25, 1.275, 1.3), default=1.25)
    parser.add_argument("--name", help="Unique name under this isolated directory's outputs")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="CPU-only scene and asset checks")
    parser.add_argument("--guard", action="store_true", help="Reserve only server A100 GPU1; refuse busy GPU")
    parser.add_argument("--library", default="baseline", help="baseline or an archived candidate build label")
    parser.add_argument("--certify-envelope", action="store_true", help="Require independent air-release evidence; keep raw distance diagnostic")
    parser.add_argument("--stress-tests", action="store_true", help="Append real release/regrasp and native snapshot replay checks")
    parser.add_argument("--profile-stress", action="store_true", help="Diagnostic direct-kernel section profile; not normal FPS")
    parser.add_argument("--stress-route", choices=("legacy","tracked"), default="legacy")
    parser.add_argument("--diagnostic-stress", action="store_true", help="Continue a completed but failed base sequence for diagnosis; never certify it")
    parser.add_argument("--replay-repeats",type=int,default=2,help="Independent continuations from the exact same snapshot (2-20)")
    parser.add_argument('--assembly-blocks',type=int,choices=(64,128,256,512),default=512)
    parser.add_argument('--manifold-blocks',type=int,choices=(64,128,256,512,1024),default=1024)
    parser.add_argument('--nsys-stage',choices=('drop_settle','regrasp_close'),help='Server-only, bounded actual CUDA-graph tracing')
    parser.add_argument('--nsys-frames',type=int,default=3)
    parser.add_argument('--stencil-probe',action='store_true',help='Read-only original/cached assembly comparisons after stages; not a production FPS run')
    parser.add_argument('--mg-probe',action='store_true',help='Read-only full/packed Galerkin comparisons; not a production FPS run')
    parser.add_argument('--interaction-counters',action='store_true',help='Read/reset existing counters at stage boundaries; diagnostic run, no physics changes')
    parser.add_argument('--free-mg-probe',action='store_true',help='Private original/free-only complete preconditioner comparisons; diagnostic run')
    args = parser.parse_args()
    driver_inputs={name:(ROOT/name).read_bytes() for name in ('run_scene.py','scene_config.py','training_stress.py','envelope_audit.py')}
    if (args.stencil_probe or args.mg_probe or args.free_mg_probe) and (os.name=='nt' or args.native_only):parser.error('Operator comparisons require server stage recording')
    if sum((args.stencil_probe,args.mg_probe,args.free_mg_probe))>1:parser.error('Test each operator experiment independently')
    if args.nsys_stage and (os.name=='nt' or not args.stress_tests or not 1<=args.nsys_frames<=10):
        parser.error('Nsight capture requires server stress tests and 1-10 frames')
    if not 2<=args.replay_repeats<=20:
        parser.error('--replay-repeats must be 2-20')
    if args.name and (not args.name.replace("-", "").replace("_", "").isalnum() or len(args.name) > 80):
        parser.error("Use letters, digits, hyphens or underscores in --name")
    if args.guard and os.name == "nt":
        parser.error("--guard is for server GPU1; the Windows live window uses the local GPU")
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be nonnegative")
    os.environ.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if os.name != "nt":
        os.environ.update(CUDA_VISIBLE_DEVICES=GPU, PYOPENGL_PLATFORM="egl")
    sys.path.insert(0, str(SOURCE))
    from endoscopic_tissue import run
    from scene_config import configure
    metadata, view_options = configure(run, args.preset, args.lateral_mm)
    metadata['driver_sha256']=sha(Path(__file__))
    metadata['driver_inputs_sha256']={name:hashlib.sha256(data).hexdigest() for name,data in driver_inputs.items()}
    grid = run.SCISSORS_SCENES["kidney"].build_grid(run.scissors_settings().density_kg_m3)
    metadata.update(material_nodes=int(grid.inside.sum()), active_nodes=int(grid.active.sum()),
                    lattice_nodes=len(grid.positions), grid_edges=len(grid.edges), cell_size_mm=(grid.cell_size * 10).tolist(),
                    mass_kg=float(grid.mass.sum()), python_source=str(SOURCE),
                    source_script_sha256=sha(ROOT / "scene_config.py"),
                    server_gpu=GPU if os.name != "nt" else None)
    del grid
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if args.inspect:
        return 0
    name = args.name or args.preset + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = ROOT / "outputs" / name
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Resource logs are separate from the leaf run.main creates exclusively.
    controls_dir = ROOT / "controls" / name
    controls_dir.mkdir(parents=True, exist_ok=False)
    guard = None
    try:
        if args.guard:
            from endoscopic_tissue.training.acceptance_queue import GPUReservation
            guard = GPUReservation(GPU, controls_dir)
        if os.name != "nt" and not args.native_only:
            from endoscopic_tissue.training.egl_device import bind
            metadata["egl_binding"] = bind(GPU)
        if args.library != "baseline":
            if not args.library.replace("-", "").replace("_", "").isalnum():
                raise ValueError("Invalid candidate build label")
            build = json.loads((ROOT / "results" / ("build-" + args.library + ".json")).read_text())
            dll = ROOT / "binaries" / (args.library + (".dll" if os.name == "nt" else ".so"))
            expected = build["sha256"]
        elif os.name == "nt":
            windows = ROOT.parent / "windows-live-20261104"
            dll = windows / "build/lib/LiverCudaSim.dll"
            expected = json.loads((windows / "build-manifest.json").read_text(encoding="utf-8"))["dll_sha256"]
        else:
            dll = CANDIDATE / "native/binaries/runtime-core.so"
            expected = json.loads((CANDIDATE / "delivery/manifest.json").read_text())["library_sha256"]
        if sha(dll) != expected:
            raise ValueError("Native DLL differs from the recorded candidate build")
        native = ctypes.CDLL(str(dll))
        if args.certify_envelope:
            from envelope_audit import install
            install(run, native, output)
            metadata['envelope_certification_version'] = 1
        if hasattr(native, "LCS_GetEnvelopeReleaseEdges") and not args.certify_envelope:
            original_exit = run.NativeTissue.__exit__
            def exit_with_provenance(self, *exc):
                import numpy as np
                flags = np.zeros(metadata["grid_edges"], dtype=np.uint8)
                fn = native.LCS_GetEnvelopeReleaseEdges
                fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
                code = fn(flags.ctypes.data, len(flags))
                if output.exists():
                    counts = np.zeros(9, np.int32)
                    witness = native.LCS_ReadCutWitnesses
                    witness.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,
                                        ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
                    witness.restype = ctypes.c_int
                    query = witness(counts.ctypes.data,None,0,None,0,None,0,None,None)
                    capture = dict(flags=flags, returncode=code, query_returncode=query, counts=counts)
                    if query == 0:
                        points = np.zeros((max(1,int(counts[5])),32),np.uint8)
                        edges = np.zeros((int(counts[4]),4),np.int32)
                        rotations = np.zeros((int(counts[3]),9),np.float32)
                        cuts = np.zeros((int(counts[3]),6),np.float32)
                        skin = np.zeros_like(cuts)
                        read = witness(counts.ctypes.data,points.ctypes.data,len(points),edges.ctypes.data,len(edges),
                                       rotations.ctypes.data,len(rotations),cuts.ctypes.data,skin.ctypes.data)
                        capture.update(read_returncode=read,points=points[:counts[5]],edges=edges,cuts=cuts,skin=skin)
                    np.savez_compressed(output / "release-provenance.npz", **capture)
                return original_exit(self, *exc)
            run.NativeTissue.__exit__ = exit_with_provenance
        for name, values in {"LCS_SetCoreOptimizations": (43264,), "LCS_SetBatchQueryMode": (3,),
                             "LCS_SetQueryThreads": (64,), "LCS_SetNativeSchedule": (args.assembly_blocks, args.manifold_blocks),
                             "LCS_SetCoarseBoundCache": (1,)}.items():
            fn = getattr(native, name)
            fn.argtypes, fn.restype = [ctypes.c_int] * len(values), ctypes.c_int
            if fn(*values) != 0:
                raise RuntimeError("Candidate configuration rejected: " + name)
        metadata.update(dll=str(dll), dll_sha256=expected, core_mode=43264,
                        native_schedule=dict(assembly_blocks=args.assembly_blocks,manifold_blocks=args.manifold_blocks),
                        native_physics_source_modified=args.library != "baseline", library_label=args.library,
                        timing_definition="Synchronous headless demo RGB frame, including step/readback, surface and recording submit, renderer and video write; excludes stage saves and final async drain. Not Windows displayed FPS or full training RGB-D.")
        base_view = run.EndoscopicView

        def view(*vargs, **kwargs):
            kwargs.setdefault("supersample", 1)
            kwargs.update(gpu_postprocess=True, gpu_aligned_depth=True, fused_mesh=True,
                          persistent_soup=True, **view_options)
            return base_view(*vargs, **kwargs)

        run.EndoscopicView = view
        measurements = []
        native_calls = []
        frame = {}
        original_step, original_check = run.NativeTissue.step, run.NativeTissue.check

        def check(self, name, *values):
            if name != "LCS_Step":
                return original_check(self, name, *values)
            begin = time.perf_counter()
            result = original_check(self, name, *values)
            frame["native_ms"] = (time.perf_counter() - begin) * 1000
            native_calls.append(dict(tick=int(getattr(self,'tick',-1)),phase=getattr(self,'timing_phase','trajectory'),native_ms=frame["native_ms"]))
            return result

        def step(self, command):
            if guard:
                guard({os.getpid()})
            frame["start"] = time.perf_counter()
            result = original_step(self, command)
            frame['tissue']=self
            frame["step_readback_ms"] = (time.perf_counter() - frame["start"]) * 1000
            return result

        run.NativeTissue.check, run.NativeTissue.step = check, step
        original_record = run.Recording.after_step

        def record(self, flow, tissue, surface):
            render_start = time.perf_counter()
            result = original_record(self, flow, tissue, surface)
            end = time.perf_counter()
            measurements.append(dict(step=flow.step, stage=flow.stage, native_ms=frame["native_ms"],
                                     step_readback_ms=frame["step_readback_ms"],
                                     render_write_ms=(end - render_start) * 1000,
                                     frame_ms=(end - frame["start"]) * 1000))
            return result

        run.Recording.after_step = record
        if args.interaction_counters:
            read_stats=native.LCS_ReadInteractionStats
            read_stats.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];read_stats.restype=ctypes.c_int
            stats_before_stage=run.Recording.after_stage;stage_stats=[]
            initial_stats=(ctypes.c_int*16)();assert read_stats(initial_stats,16,1)==16
            def counter_stage(self,flow,stage,surface):
                stats_before_stage(self,flow,stage,surface)
                values=(ctypes.c_int*16)();assert read_stats(values,16,1)==16
                stage_stats.append(dict(stage=stage,raw=list(values),substeps=values[0],
                    fragment_query_sum=values[1],substeps_with_free_nodes=values[2],
                    recovery_substeps=values[3],loaded_only_substeps=values[4],
                    graph_rebuilds=values[5],peak_fragment_queries=values[15]))
                (output/'interaction-counters.json').write_text(json.dumps(stage_stats,indent=2)+'\n')
            run.Recording.after_stage=counter_stage
            metadata['interaction_counters_in_run']=True
        if hasattr(native,'LCS_PressureZeroCounters'):
            read_zero=native.LCS_PressureZeroCounters;read_zero.argtypes=[ctypes.c_void_p,ctypes.c_int];read_zero.restype=ctypes.c_int
            zero_before_stage=run.Recording.after_stage;zero_records=[]
            def zero_stage(self,flow,stage,surface):
                zero_before_stage(self,flow,stage,surface)
                values=(ctypes.c_ulonglong*8)();assert read_zero(values,1)==0
                row=dict(stage=stage,counters=list(values));zero_records.append(row)
                (output/'pressure-zero-proof.json').write_text(json.dumps(zero_records,indent=2)+'\n')
                assert values[2]==values[5]==0, 'An omitted tet had a nonzero reference energy change'
                if values[1]+values[4]:assert values[6]==values[7]==1, 'Perturbed private reference negative control failed'
            run.Recording.after_stage=zero_stage
            metadata['pressure_zero_checks_in_run']=True
        if args.stencil_probe or args.mg_probe:
            import numpy as np
            fn=getattr(native,'LCS_MgGalerkinProbe' if args.mg_probe else 'LCS_MaterialStencilProbe');fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];fn.restype=ctypes.c_int
            original_stage=run.Recording.after_stage
            probes=[]
            def stage_probe(self,flow,stage,surface):
                tissue=frame['tissue']
                original_stage(self,flow,stage,surface)
                stats=(ctypes.c_ulonglong*(49 if args.mg_probe else 9))();errors=(ctypes.c_double*(32 if args.mg_probe else 8))()
                before=tissue.observe()
                for damage in ([0,1] if not probes else [0]):
                    assert fn(damage,stats,errors)==0
                    differences=(sum(stats[1+6*l+k] for l in range(stats[0]) for k in (2,4)) if args.mg_probe else sum(stats[k] for k in (1,3,5,7)))
                    nonfinite=sum(stats[1+6*l+5] for l in range(stats[0])) if args.mg_probe else stats[8]
                    probes.append(dict(stage=stage,negative_control=bool(damage),
                        negative_control_kind='private output perturbation' if args.mg_probe or build.get('shared_stencil') else 'private cached F perturbation',
                        stats=list(stats),max_error_reference_scale=list(errors),bitwise_equal=differences==0))
                    (output/('galerkin-probes.json' if args.mg_probe else 'stencil-probes.json')).write_text(json.dumps(probes,indent=2)+'\n')
                    assert nonfinite==0, 'Nonfinite assembly probe output'
                    if damage:assert differences>0, 'Damaged-cache negative control was not detected'
                    else:assert differences==0, 'Cached assembly differs from original'
                after=tissue.observe()
                assert np.array_equal(before.nodes,after.nodes) and np.array_equal(before.neighbors,after.neighbors), 'Read-only probe changed live state'
            run.Recording.after_stage=stage_probe
            metadata['mg_probes_in_run' if args.mg_probe else 'stencil_probes_in_run']=True
        if args.free_mg_probe:
            import numpy as np
            fn=native.LCS_FreeMultigridProbe;fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];fn.restype=ctypes.c_int
            previous_free_stage=run.Recording.after_stage;free_probes=[]
            def free_stage_probe(self,flow,stage,surface):
                previous_free_stage(self,flow,stage,surface)
                tissue=frame['tissue'];before=tissue.observe()
                for damage in ([2,0,1] if not free_probes else [2,0]):
                    stats=(ctypes.c_ulonglong*16)();errors=(ctypes.c_double*8)()
                    assert fn(damage,stats,errors)==0
                    free_probes.append(dict(stage=stage,negative_control=damage==1,original_repeat_control=damage==2,stats=list(stats),
                        free_cg_nodes=stats[0],loaded_cg_nodes=stats[1],free_output_differences=stats[2],
                        free_dot_differences=stats[3],nonfinite=stats[4],unused_loaded_output_differences=stats[5],
                        free_components=stats[6],dot_components_within_roundoff_bound=stats[8],dot_components_outside_roundoff_bound=stats[9],
                        errors_and_reference_scales=list(errors),
                        dot_criterion='Both complete V-cycles; bit-identical free z; double atomic dot differences bounded by 2*gamma(6*n)*sum(abs(r_k*z_k)); original-original control retained'))
                    (output/'free-mg-probes.json').write_text(json.dumps(free_probes,indent=2)+'\n')
                    assert stats[4]==0, 'Nonfinite free preconditioner output'
                    if damage==1:assert stats[7]>0 and stats[2]>0, 'Private output negative control was not detected'
                    else:assert stats[2]==0 and stats[9]==0, 'Free-component V-cycle differs or dot reduction exceeds arithmetic bound'
                after=tissue.observe()
                assert np.array_equal(before.nodes,after.nodes) and np.array_equal(before.neighbors,after.neighbors), 'Preconditioner probe changed live physical state'
            run.Recording.after_stage=free_stage_probe
            metadata['free_mg_probes_in_run']=True
        if args.stress_tests:
            from training_stress import install
            install(run, output, profile=args.profile_stress,route=args.stress_route,diagnostic=args.diagnostic_stress,replay_repeats=args.replay_repeats,nsys_stage=args.nsys_stage,nsys_frames=args.nsys_frames)
        options = ["--scene", "kidney", "--dll", str(dll), "--output", str(output),
                   "--width", "1280", "--height", "960", "--render-every", "1"]
        if args.headless:
            options.append("--headless")
        if args.native_only:
            options.append("--native-only")
        if args.steps is not None:
            options.extend(("--steps", str(args.steps)))
        if args.viewer:
            options.append("--viewer")
        # Keep the camera centred throughout the run; no final automatic zoom.
        sys.argv = [str(Path(__file__)), *options]
        try:
            run.main()
        finally:
            if output.exists():
                frozen=output/'driver-inputs';frozen.mkdir(exist_ok=False)
                for filename,data in driver_inputs.items():(frozen/filename).write_bytes(data)
                if native_calls:
                    import numpy as np
                    values=np.array([row['native_ms'] for row in native_calls])
                    (output/'native-timing.json').write_text(json.dumps(dict(
                        note='All native calls, including extra snapshot continuations when enabled; use frame-labelled timing.json for the video trajectory.',
                        calls=native_calls,mean_ms=float(values.mean()),p95_ms=float(np.percentile(values,95))),indent=2)+'\n')
                (output / "scene-config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if measurements:
                    import numpy as np
                    summary = {}
                    for key in ("native_ms", "step_readback_ms", "render_write_ms", "frame_ms"):
                        values = np.array([row[key] for row in measurements])
                        summary[key] = dict(mean=float(values.mean()), p95=float(np.percentile(values, 95)), max=float(values.max()))
                    summary["headless_demo_hz"] = 1000 / summary["frame_ms"]["mean"]
                    summary["frames"] = len(measurements)
                    (output / "timing.json").write_text(json.dumps(dict(summary=summary, frames=measurements), indent=2) + "\n")
    finally:
        if guard:
            guard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
