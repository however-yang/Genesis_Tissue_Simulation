"""Run pinned, isolated read-only diagnostic comparisons. No dependency changes."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--kind',choices=['benchmark','validate','profile','rgbd','snapshot'],default='benchmark');a.add_argument('--variant',choices=['reference','readonly'],default='readonly');a.add_argument('--output',type=Path);a.add_argument('--dry-run',action='store_true');args=a.parse_args()
output=(args.output or p/'results'/('reproduce-'+args.kind+'-'+args.variant+'-'+time.strftime('%Y%m%d-%H%M%S'))).resolve()
if not output.is_relative_to((p/'results').resolve()) or output.exists():raise SystemExit('Choose a new output directory below this experiment/results')
manifest=p/'delivery/manifest.json'
if not manifest.exists():raise SystemExit('Final audit manifest not written yet')
m=json.loads(manifest.read_text());dll=p/m['native_dll'];source=p/m['python_source']
if hashlib.sha256(dll.read_bytes()).hexdigest()!=m['native_dll_sha256']:raise SystemExit('Native library SHA256 differs from audited version')
for root,files in [(p/m['native_source'],m['native_files']),(source,m['python_files'])]:
 for name,digest in files.items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise SystemExit('Audited source changed: '+str(root/name))
py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python';common=['--source',str(source),'--dll',str(dll),'--output',str(output),'--query-mode','3','--bound-cache','--core-optimizations','0','--gpu-topology'];enabled=args.variant=='readonly'
if enabled:common+=['--gpu-body-check']
if args.kind in ['benchmark','validate','profile']:
 cmd=[py,str(p/'tools/profile_all_queries.py'),*common,'--cpus','7,19,20,25']
 if args.kind=='validate':cmd+=['--validate-native','--start','0','--count','1410']+(['--validate-body-check'] if enabled else [])
 cmd+=['--vision','--gpu-postprocess','--gpu-aligned-depth','--fused-mesh','--persistent-soup','--direct-rgb','--opencv-threads','1']
 if args.kind=='profile':
  cmd+=['--profile','--start','470','--count','12','--max-frames','482']
  cmd=['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys','profile','--trace=cuda,nvtx,opengl','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(output)+'-trace',*cmd]
else:
 name='check_current_frame_final_cache.py' if args.kind=='rgbd' else 'check_snapshot_roundtrip_cache.py';cmd=[py,str(p/'tools'/name),*common,'--query-threads','64','--assembly-blocks','512','--manifold-blocks','1024']
 if args.kind=='rgbd':cmd+=['--persistent-soup','--direct-rgb']
print(json.dumps(dict(variant=args.variant,kind=args.kind,command=cmd,not_formal_physics_approval=True),ensure_ascii=False,indent=2),flush=True)
if not args.dry_run:
 os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1');raise SystemExit(subprocess.run(cmd,cwd=p).returncode)
