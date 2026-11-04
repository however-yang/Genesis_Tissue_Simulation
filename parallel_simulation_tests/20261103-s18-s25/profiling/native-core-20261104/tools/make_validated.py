"""Instrument old/candidate manifold dispatch without changing either kernel."""
from pathlib import Path
import shutil,re
p=Path(__file__).resolve().parents[1];s=p/'native-validated/source';s.parent.mkdir(exist_ok=False);shutil.copytree(p/'native-scheduled/source',s)
h=s/'vendor/cuda/contact/surface_contact.cuh';t=h.read_text().replace('16*sizeof(int)','32*sizeof(int)');h.write_text(t)
f=s/'vendor/cuda/xpbd/xpbd_solver.cu';t=f.read_text();matches=list(re.finditer(r'k_UnifiedManifold<<<(.*?)>>>\((.*?)\);',t,re.S));print('solver manifold sites',len(matches));assert len(matches)==5
for m in reversed(matches):
 cfg=m.group(1).split(',');stream=cfg[3].strip() if len(cfg)>3 else '0';args=m.group(2).strip();t=t[:m.start()]+f'launchUnifiedManifold({stream},'+args+',3,3*g_surfaceCount);'+t[m.end():]
# Add the validator after the header that defines the original manifold kernel.
headers=re.findall(r'#include "([^"]+)"',t)
key=next(x for x in headers if 'unified_vertex' in x)
needle=f'#include "{key}"';assert t.count(needle)==1;t=t.replace(needle,needle+'\n#include "../contact/manifold_validation.cuh"')
# Buffer lifetime follows the model, but derived buffers never enter its snapshot.
t=t.replace('    freeQueryGroups();','    freeManifoldValidation();\n    freeQueryGroups();',1)
t=t.replace('e=allocateQueryValidation(g_N);if(e!=cudaSuccess)return -(int)e;','e=allocateQueryValidation(g_N);if(e!=cudaSuccess)return -(int)e;\n        e=allocateManifoldValidation(g_N);if(e!=cudaSuccess)return -(int)e;')
t=t.replace('16*sizeof(int)','32*sizeof(int)');f.write_text(t)
f=s/'vendor/cuda/cutting/pressure_blade.cuh';b=f.read_bytes();t=b.decode().replace('\r\n','\n');m=re.search(r'k_UnifiedManifold<<<(.*?)>>>\((.*?)\);',t,re.S);assert m;args=m.group(2).strip();assert args.endswith(',BLADE_KIND_SLOTS');t=t[:m.start()]+'launchUnifiedManifold(stream,'+args+',BLADE_KIND_SLOTS*g_surfaceCount);'+t[m.end():];f.write_bytes(t.replace('\n','\r\n').encode() if b'\r\n' in b else t.encode())
shutil.copy2(p/'tools/manifold_validation.cuh',s/'vendor/cuda/contact/manifold_validation.cuh')
# Reuse the benchmark but require the new manifold comparison as well.
t=(p/'tools/profile_scheduled.py').read_text();t=t.replace('values=(ctypes.c_int*16)()','values=(ctypes.c_int*32)()');t=t.replace('fallbacks=values[10],first_mismatch_entry=', 'fallbacks=values[10],manifold_batches=values[16],manifold_mismatches=values[17],manifold_records=values[18],captured_manifold_batches=values[19],manifold_degree_diffs=values[20],manifold_record_diffs=values[21],quadrature_diffs=values[22],manifold_mapping_diffs=values[23],manifold_skips=values[24],first_mismatch_entry=')
t=t.replace('sum(values[k] for k in range(5,11))==0','sum(values[k] for k in range(5,11))==0 and values[16]>0 and values[18]>0 and values[19]>0 and values[17]==0 and sum(values[k] for k in range(20,25))==0');(p/'tools/profile_validated.py').write_text(t);compile(t,str(p/'tools/profile_validated.py'),'exec')
t=(p/'tools/build_scheduled.py').read_text().replace("base=p/'native-scheduled'","base=p/'native-validated'").replace('build-scheduled.txt','build-validated.txt').replace('build-scheduled.json','build-validated.json');(p/'tools/build_validated.py').write_text(t)
print('Generated canonical full-manifold witness/quadrature comparison.')
