"""Replace only our idle comparison supervisor, with ownership checks."""
from datetime import datetime,timezone
import json
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[2]
p=psutil.Process(20940)
created=datetime.fromtimestamp(p.create_time(),timezone.utc).isoformat()
command=p.cmdline()
assert created.startswith('2026-11-01T20:05:51.'),(p.pid,created)
assert Path(p.cwd()).resolve()==ROOT.resolve()
assert any(Path(x).name=='compare_default.py' for x in command),command
assert not p.children(recursive=True),'An owned comparison worker started; leave it running'
out=ROOT/'endoscopic_tissue/outputs/s15-s17-20261101/default-comparison'
assert not list(out.iterdir()),'Comparison already has evidence; do not interrupt it'
record={'pid':p.pid,'created_utc':created,'command':command,'reason':'replace idle owned supervisor with persistent serial GPU queue',
        'user_processes_touched':False}
p.terminate()
p.wait(timeout=10)
(out/'supervisor-replaced.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record,indent=2))
