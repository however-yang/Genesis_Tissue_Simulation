"""Continuous native timing, without recording/compression/surface auditing.

Reports physics and native step+surface latency, not the application's display FPS.
Uses the same fixture, grid, time step, solver and contact parameters as validate.py.
"""
import argparse
import ctypes as ct
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation'),str(ROOT)]

from validate import fixture, summary, NativeTissue, drive_command, interpolate, stage_start, Sequence, severed_edges
import numpy as np
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=1,user_api='blas')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True)
    parser.add_argument('--dll',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--end-step',type=int,default=0,help='Optional fixed prefix; never a quality pass')
    parser.add_argument('--diagnostic-spikes',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    scene_name,scene,settings,stages,kind=fixture(args.case)
    grid=scene.build_grid(settings.density_kg_m3)
    rows,telemetry=[],[]
    first_spike=None
    with NativeTissue(grid,settings,args.dll,stages[0].command) as tissue:
        obs=tissue.observe()
        sequence=Sequence(obs,settings,scene,grid.positions,grid.inside)
        previous=stages[0].command
        check=tissue.check
        core=0.
        stat_fn=getattr(tissue.dll,"LCS_ReadInteractionStats",None)
        stat_buf=(ct.c_int*16)()
        if stat_fn:
            stat_fn.argtypes=[ct.POINTER(ct.c_int),ct.c_int,ct.c_int];stat_fn.restype=ct.c_int;stat_fn(stat_buf,16,1)
        def timed(name,*values):
            nonlocal core
            begin=time.perf_counter()
            try:return check(name,*values)
            finally:
                if name=='LCS_Step':core=(time.perf_counter()-begin)*1000
        tissue.check=timed
        for stage in stages:
            previous=stage_start(stage.name,previous,obs)
            for k in range(stage.steps):
                if args.end_step and len(rows)>=args.end_step: break
                command=drive_command(interpolate(previous,stage.command,(k+1)/(stage.steps*stage.ramp)),obs,settings)
                profile=args.diagnostic_spikes and first_spike is not None and len(rows)==first_spike
                if profile:
                    tissue.dll.LCS_ProfileSections.argtypes=[ct.c_int];tissue.dll.LCS_ProfileSections.restype=None;tissue.dll.LCS_ProfileSections(1)
                begin=time.perf_counter()
                obs=tissue.step(command)
                surface=tissue.surface()
                elapsed=(time.perf_counter()-begin)*1000
                rows.append(dict(step=len(rows)+1,stage=stage.name,physics_ms=core,pipeline_ms=elapsed,grip_points=obs.grip_points,jaw_load_n=list(obs.jaw_load_n),fractures=obs.scissors.loads.fractures,surface_triangles=len(surface.triangles),section_profile=profile))
                if stat_fn:
                    assert stat_fn(stat_buf,16,1)==16
                    rows[-1]["host_stats"]=list(stat_buf)
                if profile:
                    buf=ct.create_string_buffer(1048576);fn=tissue.dll.LCS_ProfileSectionsRead;fn.argtypes=[ct.c_char_p,ct.c_int];fn.restype=ct.c_int
                    fn(buf,len(buf));(args.output/f'profile-{len(rows):04d}.txt').write_bytes(buf.value);tissue.dll.LCS_ProfileSections(0)
                if args.diagnostic_spikes and len(rows)>30 and core>100 and first_spike is None:first_spike=len(rows)
                if args.diagnostic_spikes and first_spike and len(rows) in (first_spike,first_spike+5,first_spike+30):
                    for name,suffix in [('LCS_SaveFragmentCase','.bin'),('LCS_SavePairedConstraints','.fmc')]:
                        fn=getattr(tissue.dll,name,None)
                        if fn is None:continue
                        fn.argtypes=[ct.c_char_p];fn.restype=ct.c_int
                        code=fn(str(args.output/f'native-{len(rows):04d}{suffix}').encode('utf8'))
                        if code:raise RuntimeError((name,code))
                    np.savez_compressed(args.output/f'state-{len(rows):04d}.npz',nodes_cm=obs.nodes,neighbors=obs.neighbors,grip_flags=obs.grip_flags,rest_cm=grid.positions,inside=grid.inside,active=grid.active,pinned=grid.pinned,vertices_cm=surface.vertices,faces=surface.triangles)
                    print('SPIKE',len(rows),core,flush=True)
            previous=stage.command
            if args.end_step and len(rows)>=args.end_step: break
            print(stage.name,round(core,2),flush=True)
            sample=subprocess.run(['nvidia-smi','--query-gpu=timestamp,pstate,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True)
            telemetry.append(dict(stage=stage.name,reading=sample.stdout.strip(),error=sample.stderr.strip()))
    # One final readback already present in obs; save after all measured steps.
    np.savez_compressed(args.output/'final-state.npz',nodes_cm=obs.nodes,neighbors=obs.neighbors,grip_flags=obs.grip_flags,rest_cm=grid.positions,inside=grid.inside,active=grid.active,pinned=grid.pinned,vertices_cm=surface.vertices,faces=surface.triangles)
    material=grid.inside.astype(bool)
    components=sequence.pieces(obs.neighbors,material)
    lattice=sequence.pieces(obs.neighbors,np.ones(len(material),dtype=bool))
    report=dict(case=args.case,scene=scene_name,settings=asdict(settings),steps=len(rows),expected_steps=sum(s.steps for s in stages),truncated=bool(args.end_step),
        final_state=dict(material_components=len(np.unique(components[material])),
            lattice_components=len(np.unique(lattice[material])),grip_points=obs.grip_points,
            cut_edges=int(severed_edges(grid.neighbors,obs.neighbors))),
        dll=str(args.dll.resolve()),dll_sha256=hashlib.sha256(args.dll.read_bytes()).hexdigest(),
        physics_ms=summary([r['physics_ms'] for r in rows if r['step']>30 and not r.get('section_profile')]),
        pipeline_ms=summary([r['pipeline_ms'] for r in rows if r['step']>30 and not r.get('section_profile')]),
        stages={s.name:{key:summary([r[key] for r in rows if r['stage']==s.name and not r.get('section_profile')]) for key in ('physics_ms','pipeline_ms')} for s in stages},
        telemetry=telemetry,spike_policy='When enabled: first core >100ms after 30 warmup steps; profile the following frame and retain both raw samples.',note='Native timing only; no renderer or interaction diagnostics; not display FPS.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (args.output/'steps.json').write_text(json.dumps(rows),encoding='utf8')


if __name__=='__main__':main()
