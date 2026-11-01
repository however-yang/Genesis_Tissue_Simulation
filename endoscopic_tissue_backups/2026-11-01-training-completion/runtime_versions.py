"""Record selected installed versions without importing GPU libraries or exposing env vars."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

packages = ('numpy', 'scipy', 'trimesh', 'psutil', 'threadpoolctl', 'zstandard', 'opencv-python',
            'Pillow', 'imageio', 'taichi', 'torch', 'genesis-world', 'PyOpenGL', 'pyglet')
versions = {}
for name in packages:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
result = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
          'executable': sys.executable, 'platform': platform.platform(), 'packages': versions,
          'package_installations_performed_by_this_check': False}
path = Path(__file__).with_name('runtime-versions.json')
path.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
