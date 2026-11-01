"""Unrecorded timing with R21 host dispatch switched before native initialization."""
import ctypes as ct
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes'), str(ROOT)]
import time_full_interaction as timing
import interaction_fixtures
timing.fixture = interaction_fixtures.fixture

mode = None
if '--submission-graph' in sys.argv:
    i = sys.argv.index('--submission-graph')
    mode = int(sys.argv[i+1])
    assert mode in (0, 1)
    del sys.argv[i:i+2]
original = timing.NativeTissue.__init__


def initialize(self, grid, settings, dll_path, initial_command):
    if mode is not None:
        dll = ct.CDLL(str(Path(dll_path).resolve()))
        fn = dll.LCS_SetFragmentSubmissionGraph
        fn.argtypes, fn.restype = [ct.c_int], None
        fn(mode)
        self._dispatch_control_library = dll
    original(self, grid, settings, dll_path, initial_command)


timing.NativeTissue.__init__ = initialize
if __name__ == '__main__':
    timing.main()
    output = Path(sys.argv[sys.argv.index('--output')+1])
    path = output/'report.json'
    result = json.loads(path.read_text())
    result['r21_submission_graph'] = mode
    path.write_text(json.dumps(result, indent=2)+'\n')
