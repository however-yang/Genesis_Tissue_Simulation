"""Keep unittest temporary files and evidence inside this workspace."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101'
OUT.mkdir(parents=True, exist_ok=True)
TEMP = OUT/'cpu-temp'
TEMP.mkdir(exist_ok=True)
tempfile.tempdir = str(TEMP)
names = sys.argv[1:] or ['endoscopic_tissue.training.test_physics',
                        'endoscopic_tissue.training.test_contract',
                        'endoscopic_tissue.training.test_tasks',
                        'endoscopic_tissue.training.test_splits']
suite = unittest.defaultTestLoader.loadTestsFromNames(names)
with (OUT/'cpu-tests.txt').open('w', encoding='utf8') as log:
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
summary = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
           'passed': result.wasSuccessful(), 'modules': names}
(OUT/'cpu-tests.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps(summary))
if not result.wasSuccessful():
    for test, trace in result.failures + result.errors:
        print(str(test), trace)
sys.exit(0 if result.wasSuccessful() else 1)
