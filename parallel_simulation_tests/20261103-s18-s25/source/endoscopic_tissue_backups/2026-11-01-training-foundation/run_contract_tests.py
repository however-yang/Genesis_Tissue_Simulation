import sys
import json
import tempfile
import time
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
directory = Path(__file__).parent / 'test-temp'
directory.mkdir(exist_ok=True)
tempfile.tempdir = str(directory)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'endoscopic_tissue.training.test_contract',
    'endoscopic_tissue.test_core',
])
started = time.perf_counter()
result = unittest.TextTestRunner(verbosity=2).run(suite)
(Path(__file__).parent / 'cpu-test-result.json').write_text(json.dumps({
    'tests': result.testsRun, 'passed': result.wasSuccessful(),
    'errors': [(str(test), trace) for test, trace in result.errors],
    'failures': [(str(test), trace) for test, trace in result.failures],
    'wall_time_s': time.perf_counter()-started,
}, indent=2), encoding='utf8')
raise SystemExit(not result.wasSuccessful())
