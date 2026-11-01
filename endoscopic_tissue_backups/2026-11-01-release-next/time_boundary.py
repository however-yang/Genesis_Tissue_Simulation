"""R24 timing: original budgets, optional boundary grouping, no diagnostic copies."""
import ctypes as ct
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import time_repair as instrument

mode = None
if '--boundary-graph' in sys.argv:
    i = sys.argv.index('--boundary-graph')
    mode = int(sys.argv[i+1])
    assert mode in (0, 1)
    del sys.argv[i:i+2]
original = instrument.timing.NativeTissue.__init__


def initialize(self, *args, **kwargs):
    original(self, *args, **kwargs)
    if mode is not None:
        fn = self.dll.LCS_SetMaterialBoundaryGraph
        fn.argtypes, fn.restype = [ct.c_int], None
        fn(mode)


instrument.timing.NativeTissue.__init__ = initialize
if __name__ == '__main__':
    started = datetime.now(timezone.utc).isoformat()
    instrument.timing.main()
    output = Path(sys.argv[sys.argv.index('--output')+1])
    report = json.loads((output/'report.json').read_text())
    report.update(material_boundary_graph=mode, started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        repair_ms=instrument.timing.summary(instrument.repair_ms[30:]))
    rows = json.loads((output/'steps.json').read_text())
    assert len(rows) == len(instrument.repair_ms)
    for row, value in zip(rows, instrument.repair_ms):
        row['repair_ms'] = value
    (output/'steps.json').write_text(json.dumps(rows)+'\n')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
