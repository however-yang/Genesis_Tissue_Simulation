"""Attribute unchanged R3 restore, preserving binding and exact native state."""
import argparse,cProfile,io,json,pstats,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--library-dir',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path[:0]=[str(a.release),str(a.release/'endoscopic_tissue_backups/2026-11-01-s15-s17')]
 from validation_support import environment
 from endoscopic_tissue.training.snapshot import EnvSnapshot,SnapshotBackend,binding
 from endoscopic_tissue.training.provenance import json_hash,sha256
 from threadpoolctl import threadpool_limits
 a.output.mkdir(parents=True,exist_ok=False);r=dict(completed=False,diagnostic_only=True,samples=[],worker_sha256=sha256(__file__))
 try:
  with threadpool_limits(limits=1),environment(dll=str(a.library_dir)) as env:
   env.reset(0);snap=EnvSnapshot.load(a.snapshot,dll_sha256=env.context['dll_sha256'],physics_config_sha256=json_hash(binding(env)))
   r.update(library_sha256=env.context['dll_sha256'],snapshot_sha256=snap.sha256,snapshot_bytes=snap.size_bytes,native_raw_bytes=snap.metadata['native_bytes'])
   for i in range(6):
    profiler=cProfile.Profile();start=time.perf_counter()
    if i==0:profiler.enable()
    _,context=env.restore(snap,branch_label='restore-profile-'+str(i))
    if i==0:profiler.disable()
    elapsed=(time.perf_counter()-start)*1000
    if i==0:
     profiler.dump_stats(str(a.output/'restore.pstats'));stream=io.StringIO();pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(45);(a.output/'restore-profile.txt').write_text(stream.getvalue())
    exact=SnapshotBackend(env.native).save()==snap.native_bytes();assert exact
    r['samples'].append(dict(repeat=i,cprofile_enabled=i==0,wall_ms=elapsed,reported_restore_ms=context['restore_wall_time_ms'],native_exact=exact))
    (a.output/'progress.json').write_text(json.dumps(r,indent=2));print('RESTORE',i,elapsed,exact,flush=True)
   r['completed']=True
 finally:(a.output/'result.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':main()
