"""Run the full, uninstrumented acceptance worker in this batch's output root."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'endoscopic_tissue_backups/2026-11-01-s15-s17'), str(ROOT)]
import validate_damage

validate_damage.OUT = ROOT / 'endoscopic_tissue/outputs/acceptance-fixes-20261102'
validate_damage.main()
