"""Complete R22 interaction and host-only retained-manifold dispatch audit."""
import ctypes as ct
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes'), str(ROOT)]
import validate_strict as validation
import interaction_fixtures
interaction_fixtures.install()
output = Path(sys.argv[sys.argv.index('--output')+1])
step_original = validation.NativeTissue.step
rows = []


def step(self, command):
    obs = step_original(self, command)
    fn = self.dll.LCS_ReadRepairDispatchStats
    fn.argtypes, fn.restype = [ct.POINTER(ct.c_int), ct.c_int, ct.c_int], ct.c_int
    result = (ct.c_int*4)()
    assert fn(result, 4, 1) == 4
    rows.append(dict(step=len(rows)+1, empty_calls=result[0], nonempty_calls=result[1],
                     rebuilt_graphs=result[2]))
    return obs


validation.NativeTissue.step = step
try:
    validation.main()
finally:
    output.mkdir(exist_ok=True, parents=True)
    (output/'repair-dispatch-audit.json').write_text(json.dumps(dict(
        host_only=True, source_gate='retained constraint count == 0; no skipped nonempty constraints',
        empty_calls=sum(r['empty_calls'] for r in rows), nonempty_calls=sum(r['nonempty_calls'] for r in rows),
        rows=rows), indent=2)+'\n')
