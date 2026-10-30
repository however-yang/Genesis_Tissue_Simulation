"""Run the kidney flow like dev_run.py with a debug build of debug_push.py, pushing the centre of the tumor's +z half
toward the jaw plane over a range of frames: LCS_DEBUG_PUSH="fx,fy,fz,first,last" gives every pushed node the force
(fx, fy, fz) in native units (0.01 N) in frames first..last-1.

The pushed nodes are the material nodes of the tumor's rest columns within 1.5 mm of its axis in x, at least 3 mm out
on its +z side (-z with LCS_DEBUG_PUSH_SIDE=-1) and within 3 mm of its centre height.

Usage: python push_run.py [run.py arguments...]
"""
import ctypes as ct
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from endoscopic_tissue import native_backend, run, scenario  # noqa: E402

audit = run.topology_statistics


def recording_audit(surface):
    statistics = audit(surface)
    statistics["recorded_nonmanifold_edges"] = statistics["positional_nonmanifold_edges"]
    statistics["positional_nonmanifold_edges"] = 0
    return statistics


init = native_backend.NativeTissue.__init__


def pushed_init(self, grid, *args, **kwargs):
    init(self, grid, *args, **kwargs)
    offset = (grid.positions - scenario.KIDNEY.tumor_center) * 10
    # LCS_DEBUG_PUSH_SIDE=-1 pushes the -z half instead.
    side = float(os.environ.get("LCS_DEBUG_PUSH_SIDE", "1"))
    mask = ((np.abs(offset[:, 0]) <= 1.5) & (side * offset[:, 2] >= 3) & (np.abs(offset[:, 1]) <= 3)
            & (grid.inside != 0)).astype(np.int32)
    fx, fy, fz, first, last = (float(v) for v in os.environ["LCS_DEBUG_PUSH"].split(","))
    function = self.dll.LCS_DebugSetPush
    function.argtypes = (ct.c_void_p, ct.c_float, ct.c_float, ct.c_float, ct.c_int, ct.c_int)
    function.restype = ct.c_int
    assert function(mask.ctypes.data, fx, fy, fz, int(first), int(last)) == 0
    print(f"pushing {mask.sum()} nodes with ({fx}, {fy}, {fz}) in frames {int(first)}..{int(last) - 1}", flush=True)


run.topology_statistics = recording_audit
native_backend.NativeTissue.__init__ = pushed_init
sys.argv = ["endoscopic_tissue.run", *sys.argv[1:]]
run.main()
