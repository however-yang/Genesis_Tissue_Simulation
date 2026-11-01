"""Losslessly archive one generated historical replay after full byte verification."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation')]
from replay_archive import archive_replay

allowed = (ROOT/'endoscopic_tissue/outputs/interaction-fixes-20261031').resolve()
source = (allowed/'r10-strict-kidney-cut-disabled/replay.bin').resolve(strict=True)
if not source.is_relative_to(allowed) or source.name != 'replay.bin':
    raise ValueError('Generated target is outside the explicit experiment output root')
if shutil.disk_usage(allowed).free < 1.6*1024**3:
    raise OSError('Insufficient temporary space for a verified lossless archive')
temporary=source.with_suffix('.rdx.partial')
if temporary.exists() and temporary.stat().st_size == 0:
    # Empty file left by this script's failed import, not simulation evidence.
    if temporary.resolve().is_relative_to(allowed): temporary.unlink()
result = archive_replay(source,allowed)
Path(__file__).with_name('storage-archive.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
