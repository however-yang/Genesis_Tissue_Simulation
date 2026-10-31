"""Replay an exact contact state through frozen reference and three current paths."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

library_path, snapshot_path, output_path = map(lambda s: Path(s).resolve(), sys.argv[1:4])
library=ct.CDLL(str(library_path))
library.LCS_ReplayContactSnapshot.argtypes=(ct.c_char_p,ct.POINTER(ct.c_int))
library.LCS_ReplayContactSnapshot.restype=ct.c_int
values=(ct.c_int*27)()
code=library.LCS_ReplayContactSnapshot(str(snapshot_path).encode(),values)
rows=[]
for i,name in enumerate(("ordinary_wrapper","candidate_narrow","batched_front_and_narrow")):
    v=list(values)[9*i:9*(i+1)]
    rows.append(dict(path=name,different_counts=v[0],different_records=v[1],valid_records=v[4],first_word=v[7],first_slot=v[8]))
result=dict(status=code,dll_sha256=hashlib.sha256(library_path.read_bytes()).hexdigest(),snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),paths=rows)
output_path.write_text(json.dumps(result,indent=2),encoding="utf8")
print(json.dumps(result),flush=True)
assert code==0 and all(r["different_counts"]==0 and r["different_records"]==0 for r in rows), "Frozen contact regression failed"
