"""Run CPU regression checks with workspace-local temporary files and evidence."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'endoscopic_tissue_backups/2026-11-01-s15-s17'))
parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
parser.add_argument('names', nargs='*')
args = parser.parse_args()
out = ROOT / 'endoscopic_tissue/outputs/acceptance-fixes-20261102/cpu' / args.label
out.mkdir(parents=True, exist_ok=False)
temp = out / 'temp'
temp.mkdir()
tempfile.tempdir = str(temp)
names = args.names or ['endoscopic_tissue.training.test_' + x for x in
                      ('physics', 'damage', 'snapshot', 'geometry', 'contract', 'tasks', 'splits')]
with (out / 'tests.log').open('w', encoding='utf8') as log:
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromNames(names))
summary = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
               passed=result.wasSuccessful(), modules=names)
(out / 'result.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps(summary))
for test, trace in result.failures + result.errors:
    print(test, trace)
sys.exit(0 if result.wasSuccessful() else 1)
