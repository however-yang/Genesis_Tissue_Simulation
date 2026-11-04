"""Keep new failure-law evidence isolated; preserve the validated-cache gate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'endoscopic_tissue_backups/2026-11-01-s15-s17'), str(ROOT)]
import validate_failure_dose as worker

worker.OUT = ROOT / 'endoscopic_tissue/outputs/acceptance-fixes-20261102/dose'

if __name__ == '__main__':
    worker.main()
