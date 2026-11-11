"""Full R23 capture, including an independent readback audit of every empty proof."""
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
dll_path = Path(sys.argv[sys.argv.index('--dll')+1])
control = ct.CDLL(str(dll_path.resolve()))
control.LCS_SetRepairProofAudit.argtypes, control.LCS_SetRepairProofAudit.restype = [ct.c_int], None
control.LCS_SetRepairProofAudit(1)
step_original = validation.NativeTissue.step
rows = []


def step(self, command):
    obs = step_original(self, command)
    fn = self.dll.LCS_ReadRepairDispatchStats
    fn.argtypes, fn.restype = [ct.POINTER(ct.c_int), ct.c_int, ct.c_int], ct.c_int
    result = (ct.c_int*4)()
    assert fn(result, 4, 1) == 4
    rows.append(dict(step=len(rows)+1, proven_empty_calls=result[0], possible_contact_calls=result[1],
                     rebuilt_graphs=result[2], audited_calls=result[3]))
    return obs


validation.NativeTissue.step = step
try:
    validation.main()
finally:
    output.mkdir(exist_ok=True, parents=True)
    (output/'repair-dispatch-audit.json').write_text(json.dumps(dict(
        gpu_readback_audit=True, timing_valid=False,
        source_gate='successful immutable mode-2 prelude query; never the mutable mode-1 query count',
        proven_empty_calls=sum(r['proven_empty_calls'] for r in rows),
        possible_contact_calls=sum(r['possible_contact_calls'] for r in rows),
        audited_calls=sum(r['audited_calls'] for r in rows), rows=rows), indent=2)+'\n')
