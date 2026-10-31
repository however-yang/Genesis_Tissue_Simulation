"""Compare reference/candidate contacts on identical live states, without altering them."""
import ctypes as ct
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from endoscopic_tissue import native_backend,run

output=Path(sys.argv[1]).resolve()
arguments=sys.argv[2:]
opened=run.Flow.open
check=native_backend.NativeTissue.check
rows=[]
graph_rows=[]
ready=False


def probed_open(self,tissue):
    global ready
    tissue.dll.LCS_ProbeContactEquivalence.argtypes=(ct.POINTER(ct.c_int),)
    tissue.dll.LCS_ProbeContactEquivalence.restype=ct.c_int
    tissue.dll.LCS_SetContactProbeDump.argtypes=(ct.c_char_p,)
    tissue.dll.LCS_SetContactProbeDump.restype=ct.c_int
    assert tissue.dll.LCS_SetContactProbeDump(str(output / "failure-contact.bin").encode()) == 0
    tissue.dll.LCS_ReadSolverGraphCounts.argtypes=(ct.POINTER(ct.c_int),)
    tissue.dll.LCS_ReadSolverGraphCounts.restype=ct.c_int
    ready=True
    return opened(self,tissue)


def probed_check(self,name,*args):
    result=check(self,name,*args)
    if name=="LCS_Step" and ready:
        graph_counts=(ct.c_int*3)()
        assert self.dll.LCS_ReadSolverGraphCounts(graph_counts)==0
        graph_rows.append(dict(step=self.tick+1,loaded=graph_counts[0],free=graph_counts[1],recovery=graph_counts[2]))
    if name=="LCS_Step" and ready and ((self.tick+1)%15==0 or 600 <= self.tick+1 <= 810):
        values=(ct.c_int*8)()
        code=self.dll.LCS_ProbeContactEquivalence(values)
        row=dict(step=self.tick+1,status=code,different_counts=values[0],different_records=values[1],
                 different_degrees=values[2],different_members=values[3],valid_records=values[4],
                 queries=values[5],triangle_queries=values[6],first_different_word=values[7])
        rows.append(row)
        print("[CONTACT-EQUIVALENCE] "+json.dumps(row),flush=True)
        if code or any(values[:4]):
            raise RuntimeError("Frozen-input contact comparison failed")
    return result


run.Flow.open=probed_open
native_backend.NativeTissue.check=probed_check
sys.argv=["dev_run.py","--output",str(output),"--native-only","--scene","kidney",*arguments]
try:
    runpy.run_path(str(ROOT/"endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py"),run_name="__main__")
finally:
    if output.is_dir():
        (output/"solver-graphs.json").write_text(json.dumps(graph_rows,indent=2),encoding="utf8")
        (output/"contact-equivalence.json").write_text(json.dumps(rows,indent=2),encoding="utf8")
