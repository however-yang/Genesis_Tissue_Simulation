"""Same-library ABBA cold-memory versus explicit prepared hot restores.

No disk cold-start or portable snapshot latency claim is made for the hot path.
"""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
import psutil

def main():
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--library-dir',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path[:0]=[str(a.release),str(a.release/'endoscopic_tissue_backups/2026-11-01-s15-s17')]
 from validation_support import environment
 from endoscopic_tissue.training.snapshot import EnvSnapshot,SnapshotBackend,binding
 from endoscopic_tissue.training.provenance import json_hash,sha256
 from prepared_snapshot import PreparedSnapshot
 from threadpoolctl import threadpool_limits
 import numpy as np
 a.output.mkdir(parents=True,exist_ok=False);proc=psutil.Process();r=dict(completed=False,passed=False,samples=[],source_sha256=sha256(__file__),adapter_sha256=sha256(Path(__file__).with_name('prepared_snapshot.py')),scope='Repeated same-library in-memory restores; explicit warm preparation and memory cost. Not disk cold restore, full snapshot distribution or physical acceptance.')
 try:
  with threadpool_limits(limits=1),environment(dll=str(a.library_dir)) as env:
   env.reset(0);s=EnvSnapshot.load(a.snapshot,dll_sha256=env.context['dll_sha256'],physics_config_sha256=json_hash(binding(env)))
   r.update(library_sha256=env.context['dll_sha256'],snapshot_sha256=s.sha256,native_bytes=s.metadata['native_bytes'],rss_before_prepare=proc.memory_info().rss)
   if psutil.virtual_memory().available<8*1024**3:raise MemoryError('Need at least 8 GiB host free for diagnostic')
   hot=PreparedSnapshot(s,max_host_bytes=2*1024**3);assert hot.sha256==s.sha256
   r.update(prepare_ms=hot.preparation_ms,retained_host_bytes=hot.reserved_host_bytes,rss_after_prepare=proc.memory_info().rss)
   for index,mode in enumerate(['cold','hot','hot','cold']*3):
    chosen=hot if mode=='hot' else s;start=time.perf_counter();_,c=env.restore(chosen,branch_label='prepared-'+mode+'-'+str(index));elapsed=(time.perf_counter()-start)*1000
    # Native serialization verifies every registered mutable allocation/field.
    exact=SnapshotBackend(env.native).save()==hot.native_bytes();assert exact
    r['samples'].append(dict(index=index,mode=mode,wall_ms=elapsed,reported_ms=c['restore_wall_time_ms'],native_exact=exact,rss_bytes=proc.memory_info().rss))
    (a.output/'progress.json').write_text(json.dumps(r,indent=2));print('ABBA',index,mode,elapsed,exact,flush=True)
   for mode in ['cold','hot']:
    v=[x['wall_ms'] for x in r['samples'] if x['mode']==mode];r[mode]=dict(count=len(v),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),max_ms=max(v))
   chosen=None;del hot;gc.collect();r['rss_after_cache_release']=proc.memory_info().rss
   r.update(completed=True,passed=True,hot_p95_below_1s=r['hot']['p95_ms']<=1000,cold_restore_target_unchanged=True)
 finally:(a.output/'result.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':main()
