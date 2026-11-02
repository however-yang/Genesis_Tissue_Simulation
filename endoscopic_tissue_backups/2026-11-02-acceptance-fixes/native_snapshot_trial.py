"""Native byte, continuation, fork and corrupt-state checks for this candidate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'endoscopic_tissue_backups/2026-11-01-s15-s17'), str(ROOT)]
import validate_native_snapshot as worker

if __name__ == '__main__':
    sys.argv.extend(['--output-root', str(ROOT / 'endoscopic_tissue/outputs/acceptance-fixes-20261102')])
    worker.main()
