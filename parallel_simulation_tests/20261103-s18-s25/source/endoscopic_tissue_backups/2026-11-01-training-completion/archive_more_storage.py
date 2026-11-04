"""Reserve space for the declared 100-episode screen without losing old evidence."""
from pathlib import Path
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation')]
from replay_archive import archive_replay
allowed=(ROOT/'endoscopic_tissue/outputs/interaction-fixes-20261031').resolve()
source=(allowed/'r11-strict-kidney-reference-full/replay.bin').resolve(strict=True)
if not source.is_relative_to(allowed) or source.name!='replay.bin':raise ValueError('Invalid generated target')
if shutil.disk_usage(allowed).free<1.6*1024**3:raise OSError('Insufficient archive workspace')
result=archive_replay(source,allowed)
Path(__file__).with_name('storage-archive-second.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
