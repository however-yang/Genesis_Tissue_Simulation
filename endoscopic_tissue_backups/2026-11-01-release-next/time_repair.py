"""R22 timing with optional host dispatch control; CUDA graphs remain enabled."""
import ctypes as ct
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes'), str(ROOT)]
import time_full_interaction as timing
import interaction_fixtures
timing.fixture = interaction_fixtures.fixture

mode = None
if '--repair-empty' in sys.argv:
    i = sys.argv.index('--repair-empty')
    mode = int(sys.argv[i+1])
    assert mode in (0, 1)
    del sys.argv[i:i+2]
init_original = timing.NativeTissue.__init__
check_original = timing.NativeTissue.check
step_original = timing.NativeTissue.step
repair_ms = []
frame_repair_ms = 0.0


def initialize(self, grid, settings, dll_path, initial_command):
    if mode is not None:
        dll = ct.CDLL(str(Path(dll_path).resolve()))
        fn = dll.LCS_SetRepairEmptySpecialization
        fn.argtypes, fn.restype = [ct.c_int], None
        fn(mode)
        self._dispatch_control_library = dll
    init_original(self, grid, settings, dll_path, initial_command)


def check(self, name, *values):
    global frame_repair_ms
    if name != 'LCS_RefineContactSurface':
        return check_original(self, name, *values)
    started = time.perf_counter()
    try:
        return check_original(self, name, *values)
    finally:
        frame_repair_ms += (time.perf_counter()-started)*1000.0


def step(self, command):
    global frame_repair_ms
    frame_repair_ms = 0.0
    result = step_original(self, command)
    repair_ms.append(frame_repair_ms)
    return result


timing.NativeTissue.__init__ = initialize
timing.NativeTissue.check = check
timing.NativeTissue.step = step
if __name__ == '__main__':
    started = datetime.now(timezone.utc).isoformat()
    timing.main()
    output = Path(sys.argv[sys.argv.index('--output')+1])
    path = output/'report.json'
    result = json.loads(path.read_text())
    result.update(repair_empty_specialization=mode, started_utc=started,
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  repair_ms=timing.summary(repair_ms[30:]))
    steps = json.loads((output/'steps.json').read_text())
    assert len(steps) == len(repair_ms)
    for row, value in zip(steps, repair_ms):
        row['repair_ms'] = value
    (output/'steps.json').write_text(json.dumps(steps)+'\n')
    path.write_text(json.dumps(result, indent=2)+'\n')
