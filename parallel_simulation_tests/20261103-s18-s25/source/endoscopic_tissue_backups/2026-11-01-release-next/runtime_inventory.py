"""Read-only process names and Python script names; no process control."""
from pathlib import Path
import json
import psutil
from datetime import datetime, timezone
result = []
for process in psutil.process_iter(['pid', 'name', 'cmdline']):
    info = process.info
    if 'python' not in (info['name'] or '').lower():
        continue
    result.append(dict(pid=info['pid'], name=info['name'],
        created_utc=datetime.fromtimestamp(process.create_time(), timezone.utc).isoformat(),
        scripts=[Path(x).name for x in (info['cmdline'] or []) if x.lower().endswith('.py')]))
print(json.dumps(result, indent=2))
