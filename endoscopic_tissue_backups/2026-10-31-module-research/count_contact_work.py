"""Count contact work during a complete interaction; these are NOT timings."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue import native_backend, run

output = Path(sys.argv[1]).resolve()
arguments = sys.argv[2:]
opened, stepped, stage_start = run.Flow.open, native_backend.NativeTissue.step, run.stage_start
rows = []
stage_name = None
metadata = {}


def counters(tissue):
    values = (ct.c_ulonglong * 96)()
    status = tissue.dll.LCS_ReadContactWork(values, 1)
    if status:
        raise RuntimeError(f"Contact work query failed: {status}")
    return list(values)


def probed_open(self, tissue):
    result = opened(self, tissue)
    tissue.dll.LCS_ReadContactWork.argtypes = (ct.POINTER(ct.c_ulonglong), ct.c_int)
    tissue.dll.LCS_ReadContactWork.restype = ct.c_int
    metadata.update(dll=str(tissue.dll_path.resolve()),
                    dll_sha256=hashlib.sha256(tissue.dll_path.read_bytes()).hexdigest())
    counters(tissue)
    return result


def probed_stage(name, *args):
    global stage_name
    stage_name = name
    return stage_start(name, *args)


def probed_step(self, command):
    result = stepped(self, command)
    values = counters(self)
    rows.append(dict(step=self.tick, stage=stage_name, counters=values))
    if self.tick % 60 == 0:
        print(f"[WORK] step={self.tick} stage={stage_name} queries={sum(values[0:64:8])} "
              f"broad_pass={sum(values[1:64:8])} CCD_max={values[69]}", flush=True)
    return result


run.Flow.open = probed_open
run.stage_start = probed_stage
native_backend.NativeTissue.step = probed_step
sys.argv = ["dev_run.py", "--output", str(output), "--native-only", "--scene", "kidney", *arguments]
try:
    runpy.run_path(str(ROOT / "endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py"), run_name="__main__")
finally:
    if output.is_dir():
        (output / "contact-work.json").write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2), encoding="utf8")
