"""Reproduce one isolated configuration with source/library hash checks and GPU guard."""
import argparse,json,hashlib,os
from pathlib import Path
p=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['recommended','conservative','original'],default='recommended');parser.add_argument('--output',type=Path,required=True);parser.add_argument('--physics',action='store_true');parser.add_argument('--validate-native',action='store_true');a=parser.parse_args();out=a.output if a.output.is_absolute() else p/a.output;out=out.resolve();assert out.is_relative_to((p/'results').resolve()) and not out.exists(),'Use a new directory under experiment/results'
if a.variant=='original' and a.validate_native:raise SystemExit('Paired native validation is provided by the optimized configurations')
m=json.loads((p/'delivery/manifest.json').read_text());d=m['variants'][a.variant];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();assert sha(Path(d['dll']))==d['dll_sha256']
for rel,h in m['python_source_sha256'].items():assert sha(Path(m['python_source'])/rel)==h,rel
for rel,h in d['native_source_sha256'].items():assert sha(Path(d['native_source'])/rel)==h,rel
py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';args=[py,str(p/'tools'/d['harness']),'--source',m['python_source'],'--dll',d['dll'],'--output',str(out),'--cpus','7,19,20,25','--gpu-topology']
for k,v in d['execution'].items():
 if isinstance(v,bool):
  if v:args+=['--'+k.replace('_','-')]
 else:args+=['--'+k.replace('_','-'),str(v)]
if a.physics or a.validate_native:args+=['--no-render']
else:args+=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1']
if a.validate_native:args+=['--validate-native','--start','0','--count','1410']
os.execv(py,args)
