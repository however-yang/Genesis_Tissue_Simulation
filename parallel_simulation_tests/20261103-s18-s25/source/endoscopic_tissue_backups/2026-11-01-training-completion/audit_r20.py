"""Full replay with every-step old-rescan equality; no default changes."""
from pathlib import Path
import ctypes as ct
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes'), str(ROOT)]
import validate_strict as v
import interaction_fixtures
interaction_fixtures.install()
output = Path(sys.argv[sys.argv.index('--output')+1])
original = v.NativeTissue.step
rows = []


def step(self, command):
    obs = original(self, command)
    fn = self.dll.LCS_ValidateFragmentSupportCache
    fn.argtypes, fn.restype = [ct.POINTER(ct.c_int), ct.c_int], ct.c_int
    result = (ct.c_int*6)()
    code = fn(result, 6)
    if code != 0:
        raise RuntimeError(f'Cache audit export failed {code}')
    a = list(result)
    rows.append(dict(step=len(rows)+1, different_support=a[0], reference_volume=a[1], cached_volume=a[2],
                     nodes=a[3], revision=a[4], recomputed_last_call=a[5]))
    if a[0] or a[1] != a[2]:
        raise AssertionError(('Support cache mismatch', rows[-1]))
    return obs


v.NativeTissue.step = step
try:
    v.main()
finally:
    output.mkdir(exist_ok=True, parents=True)
    (output/'support-cache-audit.json').write_text(json.dumps(dict(checks=rows,
        all_equal=bool(rows) and all(x['different_support']==0 and x['reference_volume']==x['cached_volume'] for x in rows),
        read_only=True, audited_every_native_step=True), indent=2))
