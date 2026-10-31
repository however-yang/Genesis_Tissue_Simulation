"""Exercise exact-sum patterns on the candidate's production reduction helper."""
import ctypes as ct
import json
from pathlib import Path
import sys

dll = ct.CDLL(str(Path(sys.argv[1]).resolve()))
dll.LCS_ProbeBlockAccumulate.argtypes = (ct.POINTER(ct.c_int),)
dll.LCS_ProbeBlockAccumulate.restype = ct.c_int
rows = []
for repeat in range(16):
    values = (ct.c_int * 3)()
    status = dll.LCS_ProbeBlockAccumulate(values)
    row = dict(repeat=repeat, status=status, differences=values[0], patterns=values[1], terms=values[2])
    rows.append(row)
    print(json.dumps(row), flush=True)
    if status or values[0]:
        raise RuntimeError("Exact accumulation validation failed")
Path(sys.argv[2]).write_text(json.dumps(rows, indent=2))
