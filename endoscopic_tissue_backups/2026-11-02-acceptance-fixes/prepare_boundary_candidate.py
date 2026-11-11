"""Port only R25 surface grouping to an isolated S15 snapshot source tree.

Do not import R19/R23 solver/fragment changes. Every selected patch must match
its old context exactly; an incompatible base aborts without a usable build.
"""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'endoscopic_tissue_backups/2026-11-01-s15-s17/artifacts/training-snapshot1/source'
PREVIOUS = ROOT / 'endoscopic_tissue_backups/2026-11-01-release-next/package/artifacts/release-r23-repair-prelude-proof/source'
REVIEWED = ROOT / 'endoscopic_tissue_backups/2026-11-01-release-next/package/artifacts/release-r25-boundary-contact-identity/source'
OUT = HERE / 'artifacts/boundary-snapshot2'
SOURCE = OUT / 'source'
paths = ['vendor/cuda/' + f for f in ('contact/surface_contact.cuh', 'cut.cu', 'cut.h',
         'dc_recon.cu', 'xpbd/xpbd_solver.cu', 'xpbd/xpbd_solver.h')]
prepared, evidence = {}, []
for path in paths:
    old = (PREVIOUS / path).read_text(encoding='utf8').splitlines(keepends=True)
    new = (REVIEWED / path).read_text(encoding='utf8').splitlines(keepends=True)
    original = text = (BASE / path).read_text(encoding='utf8')
    groups = list(difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_grouped_opcodes(3))
    selected = []
    for group in reversed(groups):
        before = ''.join(old[group[0][1]:group[-1][2]])
        after = ''.join(new[group[0][3]:group[-1][4]])
        if path.endswith('xpbd_solver.cu') and 'boundaryMasks' not in after and 'g_surfaceBoundaryMasks' not in after:
            continue
        if text.count(before) != 1:
            raise RuntimeError(f'Patch context is not unique/compatible: {path}\n{before[:1500]}')
        text = text.replace(before, after, 1)
        selected.append((group[0][1] + 1, group[-1][2]))
    prepared[path] = text
    evidence.append(dict(path=path, reviewed_hunks=selected,
        base_sha256=hashlib.sha256(original.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(text.encode()).hexdigest()))

# All cudaMalloc allocations are already tracked by the snapshot backend.
# Add the two owning host pointers, their enable flag and the borrowed pointer
# to its field registry; this keeps relocation and T1 exact comparisons complete.
for filename, anchor, fields in [
        ('cut_state.cuh', '    f.pod("cut.d_voxelCutMask",d_voxelCutMask);',
         [('cut.d_surfaceBoundaryMask', 'd_surfaceBoundaryMask'),
          ('cut.d_airBoundaryMask', 'd_airBoundaryMask'),
          ('cut.g_materialBoundaryGraph', 'g_materialBoundaryGraph')]),
        ('xpbd_state.cuh', '    f.pod("xpbd.g_surfaceMasks",g_surfaceMasks);',
         [('xpbd.g_surfaceBoundaryMasks', 'g_surfaceBoundaryMasks')])]:
    path = 'vendor/cuda/training/' + filename
    text = (BASE / path).read_text(encoding='utf8')
    assert text.count(anchor) == 1
    prepared[path] = text.replace(anchor, anchor + '\n' + '\n'.join(
        f'    f.pod("{name}",{field});' for name, field in fields))

if SOURCE.exists():
    raise FileExistsError('Existing candidate source is preserved: ' + str(SOURCE))
shutil.copytree(BASE, SOURCE)
for path, text in prepared.items():
    (SOURCE / path).write_text(text, encoding='utf8')
(OUT / 'port-review.json').write_text(json.dumps(dict(
    baseline=str(BASE), reviewed_source=str(REVIEWED), selected=evidence,
    new_state_fields=['cut.d_surfaceBoundaryMask', 'cut.d_airBoundaryMask',
                      'cut.g_materialBoundaryGraph', 'xpbd.g_surfaceBoundaryMasks'],
    allocation_policy='both unsigned arrays tracked and saved; all host pointers relocated',
    default_replaced=False, accepted=False), indent=2), encoding='utf8')
print(json.dumps(evidence, indent=2))
