"""Continuous native timing, without recording/compression/surface auditing.

Reports physics and native step+surface latency, not the application's display FPS.
Uses the same fixture, grid, time step, solver and contact parameters as validate.py.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

from validate import fixture, summary, NativeTissue, drive_command, interpolate, stage_start, Sequence, severed_edges
import numpy as np
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=1,user_api='blas')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True)
    parser.add_argument('--dll',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    scene_name,scene,settings,stages,kind=fixture(args.case)
    if kind=='cut': stages=stages[:next(i for i,s in enumerate(stages) if s.name=='hold')+1]
    grid=scene.build_grid(settings.density_kg_m3)
    rows,telemetry=[],[]
    with NativeTissue(grid,settings,args.dll,stages[0].command) as tissue:
        obs=tissue.observe()
        sequence=Sequence(obs,settings,scene,grid.positions,grid.inside)
        previous=stages[0].command
        check=tissue.check
        core=0.
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
                command=drive_command(interpolate(previous,stage.command,(k+1)/(stage.steps*stage.ramp)),obs,settings)
                begin=time.perf_counter()
                obs=tissue.step(command)
                surface=tissue.surface()
                elapsed=(time.perf_counter()-begin)*1000
                rows.append(dict(step=len(rows)+1,stage=stage.name,physics_ms=core,pipeline_ms=elapsed))
            previous=stage.command
            print(stage.name,round(core,2),flush=True)
            sample=subprocess.run(['nvidia-smi','--query-gpu=timestamp,pstate,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True)
            telemetry.append(dict(stage=stage.name,reading=sample.stdout.strip(),error=sample.stderr.strip()))
    material=grid.inside.astype(bool)
    components=sequence.pieces(obs.neighbors,material)
    lattice=sequence.pieces(obs.neighbors,np.ones(len(material),dtype=bool))
    report=dict(case=args.case,scene=scene_name,settings=asdict(settings),steps=len(rows),
        final_state=dict(material_components=len(np.unique(components[material])),
            lattice_components=len(np.unique(lattice[material])),grip_points=obs.grip_points,
            cut_edges=int(severed_edges(grid.neighbors,obs.neighbors))),
        dll=str(args.dll.resolve()),dll_sha256=hashlib.sha256(args.dll.read_bytes()).hexdigest(),
        physics_ms=summary([r['physics_ms'] for r in rows if r['step']>30]),
        pipeline_ms=summary([r['pipeline_ms'] for r in rows if r['step']>30]),
        stages={s.name:{key:summary([r[key] for r in rows if r['stage']==s.name]) for key in ('physics_ms','pipeline_ms')} for s in stages},
        telemetry=telemetry,note='Native timing only; no renderer or interaction diagnostics; not display FPS.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (args.output/'steps.json').write_text(json.dumps(rows),encoding='utf8')


if __name__=='__main__':main()
