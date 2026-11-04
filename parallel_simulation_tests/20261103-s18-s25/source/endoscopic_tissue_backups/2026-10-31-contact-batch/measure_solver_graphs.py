"""Observe loaded/free/recovery graph choices without extra GPU queries."""
import ctypes as ct
import json
from pathlib import Path
import runpy
import sys
import time
import subprocess
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from endoscopic_tissue import native_backend,run
output=Path(sys.argv[1]).resolve()
arguments=sys.argv[2:]
opened,checked=run.Flow.open,native_backend.NativeTissue.check
rows=[]
ready=False
slow_streak=0
profile_remaining=0
profile_triggered=False
def open_graphs(self,tissue):
    global ready
    tissue.dll.LCS_ReadSolverGraphCounts.argtypes=(ct.POINTER(ct.c_int),)
    tissue.dll.LCS_ReadSolverGraphCounts.restype=ct.c_int
    tissue.dll.LCS_ProfileSections.argtypes=(ct.c_int,)
    tissue.dll.LCS_ProfileSections.restype=None
    tissue.dll.LCS_ProfileSectionsRead.argtypes=(ct.c_char_p,ct.c_int)
    tissue.dll.LCS_ProfileSectionsRead.restype=ct.c_int
    result=opened(self,tissue)
    initial=(ct.c_int*3)()
    assert tissue.dll.LCS_ReadSolverGraphCounts(initial)==0
    ready=True
    return result
def check_graphs(self,name,*args):
    global slow_streak,profile_remaining,profile_triggered
    started=time.perf_counter()
    result=checked(self,name,*args)
    elapsed_ms=(time.perf_counter()-started)*1000
    if ready and name=="LCS_Step":
        values=(ct.c_int*3)()
        assert self.dll.LCS_ReadSolverGraphCounts(values)==0
        rows.append(dict(step=self.tick+1,loaded=values[0],free=values[1],recovery=values[2],native_ms=elapsed_ms,profiled=profile_remaining>0))
        if profile_remaining:
            profile_remaining-=1
            if profile_remaining==0:
                buffer=ct.create_string_buffer(262144)
                assert self.dll.LCS_ProfileSectionsRead(buffer,len(buffer))==0
                (output/"slow-sections.txt").write_text(buffer.value.decode(),encoding="utf8")
                self.dll.LCS_ProfileSections(0)
        elif not profile_triggered and self.tick+1>=680:
            slow_streak=slow_streak+1 if elapsed_ms>100 else 0
            if slow_streak>=3:
                (output/"slow-trigger.json").write_text(json.dumps(rows[-3:],indent=2),encoding="utf8")
                with (output/"slow-gpu-state.txt").open("w",encoding="utf8") as log:
                    subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,temperature.gpu,clocks.sm,power.draw,memory.used","--format=csv,noheader"],stdout=log,stderr=subprocess.STDOUT,timeout=10,check=True)
                self.dll.LCS_ProfileSections(1)
                profile_remaining=4
                profile_triggered=True
    return result
run.Flow.open=open_graphs
native_backend.NativeTissue.check=check_graphs
sys.argv=["measure_physics.py",str(output),*arguments]
try:
    runpy.run_path(str(root/"endoscopic_tissue_backups/2026-10-30-perf-opt/measure_physics.py"),run_name="__main__")
finally:
    if output.is_dir(): (output/"solver-graphs.json").write_text(json.dumps(rows,indent=2),encoding="utf8")
