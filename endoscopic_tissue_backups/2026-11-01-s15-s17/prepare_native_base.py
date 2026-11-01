"""Export the original contact-groups31 recipe, without editing default sources."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
BASE = 'fbcaf71dc0eec30e00b55d46bf5ae0eddac271b3'
PREFIX = 'endoscopic_tissue/native/'
OLD = ROOT/'endoscopic_tissue_backups/2026-10-31-material-contact-next'
TARGET = HERE/'native-baseline'
TARGET.mkdir(exist_ok=False)
paths = subprocess.check_output(['git', 'ls-tree', '-rz', '--name-only', BASE, '--', PREFIX+'vendor'], cwd=ROOT).split(b'\0')
manifest = {'base_commit': BASE, 'original_recipe': str(OLD/'build-manifest.json'),
            'files': {}, 'experimental': True, 'default_replaced': False}
for raw in paths:
    if not raw:
        continue
    name = raw.decode('utf8')
    rel = name[len(PREFIX):]
    assert name.startswith(PREFIX) and '..' not in Path(rel).parts
    dest = TARGET/rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(subprocess.check_output(['git', 'show', BASE+':'+name], cwd=ROOT))
for old, relative in {
    'group_xpbd_solver.cu': 'vendor/cuda/xpbd/xpbd_solver.cu',
    'group_surface_contact.cuh': 'vendor/cuda/contact/surface_contact.cuh',
    'group_pressure_blade.cuh': 'vendor/cuda/cutting/pressure_blade.cuh',
    'contact_groups.cuh': 'vendor/cuda/contact/contact_groups.cuh',
}.items():
    data = (OLD/old).read_bytes()
    if old == 'group_xpbd_solver.cu':
        before = b'../../../../../endoscopic_tissue_backups/2026-10-31-material-contact-next/contact_groups.cuh'
        assert data.count(before) == 1
        data = data.replace(before, b'../contact/contact_groups.cuh')
    (TARGET/relative).write_bytes(data)
(TARGET/'CMakeLists.txt').write_bytes((ROOT/'endoscopic_tissue/native/CMakeLists.txt').read_bytes())
for file in sorted(TARGET.rglob('*')):
    if file.is_file():
        manifest['files'][file.relative_to(TARGET).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
(HERE/'native-baseline-source.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
print('Exported', len(manifest['files']), 'source files to', TARGET)
