"""Run CPU regressions with temporary files confined to the workspace."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TEMP = Path(__file__).resolve().parent/'test-temp'
TEMP.mkdir(exist_ok=True)
tempfile.tempdir = str(TEMP.resolve())
suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:] or ['endoscopic_tissue.training.test_contract'])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
