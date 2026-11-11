"""Measure existing host ABI calls without enabling the CUDA graph-expanding profiler."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes'), str(ROOT)]
import time_full_interaction as timing
import interaction_fixtures
timing.fixture = interaction_fixtures.fixture
original = timing.NativeTissue.__init__
rows = []


def initialize(self, *args, **kwargs):
    original(self, *args, **kwargs)
    check = self.check
    step = 0

    def measured(name, *values):
        nonlocal step
        if name == 'LCS_Step':
            step += 1
        start = time.perf_counter()
        try:
            return check(name, *values)
        finally:
            rows.append((step, name, 1000*(time.perf_counter()-start)))
    self.check = measured


timing.NativeTissue.__init__ = initialize
if __name__ == '__main__':
    timing.main()
    output = Path(sys.argv[sys.argv.index('--output')+1])
    groups = defaultdict(list)
    for step, name, elapsed in rows:
        if step > 30:
            groups[name].append(elapsed)
    result = {name: dict(calls=len(values), mean=float(np.mean(values)), p95=float(np.percentile(values, 95)),
                        maximum=float(np.max(values)), total_ms=sum(values)) for name, values in groups.items()}
    (output/'host-call-profile.json').write_text(json.dumps(dict(graphs_expanded=False, calls=result), indent=2)+'\n')
    (output/'host-call-samples.json').write_text(json.dumps(rows)+'\n')
    print(json.dumps(result, indent=2))
