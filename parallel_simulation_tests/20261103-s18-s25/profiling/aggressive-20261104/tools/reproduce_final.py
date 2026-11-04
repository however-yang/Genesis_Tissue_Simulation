"""Reproduce the validated configuration in a new in-project result directory."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parents[1];a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();out=(p/args.output).resolve() if not args.output.is_absolute() else args.output.resolve();assert out.is_relative_to(p),'Output must stay inside this independent experiment directory';assert not out.exists(),'Use a fresh output directory'
d=json.loads((p/'results/final-selection.json').read_text());manifest=json.loads((p/'delivery/manifest.json').read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();assert sha(d['dll'])==d['dll_sha256'],'Native binary changed'
for name,value in manifest['source_sha256'].items():assert sha(Path(d['source'])/name)==value,'Source changed: '+name
cmd=['/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python',str(p/'tools/profile_r7.py'),'--source',d['source'],'--dll',d['dll'],'--output',str(out),'--cpus','7,19,20,25']+d['flags'];raise SystemExit(subprocess.run(cmd).returncode)
