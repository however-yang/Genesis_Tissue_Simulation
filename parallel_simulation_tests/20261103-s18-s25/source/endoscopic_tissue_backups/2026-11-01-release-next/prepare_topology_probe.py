"""Read-only surface provenance export; not a physical or rendering correction."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

P = Path(__file__).resolve().parent/'package'
ROOT = P.parents[2]
PARENT = 'release-r21-fragment-submission'
VARIANT = 'release-r21-topology-probe'
manifest = json.loads((P/(PARENT+'.json')).read_text())
destination = P/'overrides'/VARIANT
if destination.exists():
    raise RuntimeError('Probe sources already exist')
for item in manifest['files']:
    source = P/'overrides'/PARENT/item['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
    target = destination/item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
relative = 'vendor/cuda/dc_recon.cu'
data = subprocess.check_output(['git', 'show', manifest['base_commit']+':'+manifest['base_source_prefix']+relative], cwd=ROOT)
addition = r'''

// Read-only diagnostic. Query counts first with null buffers. Triangle IDs are
// the actual native feature IDs before expansion; no position welding is used.
extern "C" __declspec(dllexport) int LCS_ReadSurfaceTopology(
        int* counts,int* ids,int idCapacity,unsigned* masks,int voxelCapacity){
    if(!g_ready || !counts)return -1;
    unsigned count=0;cudaError_t e=cudaMemcpy(&count,d_triCounter,sizeof(unsigned),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    const int indices=(int)count>3*g_desc.triCapacity?3*g_desc.triCapacity:(int)count;
    counts[0]=g_dims.x;counts[1]=g_dims.y;counts[2]=g_dims.z;
    counts[3]=g_desc.voxelCount;counts[4]=indices;
    if(!ids && !masks)return 0;
    if(!ids || !masks || idCapacity<indices || voxelCapacity<g_desc.voxelCount)return -2;
    e=cudaMemcpy(ids,d_tri,indices*sizeof(int),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    if(d_cut_voxelCutMask)e=cudaMemcpy(masks,d_cut_voxelCutMask,g_desc.voxelCount*sizeof(unsigned),cudaMemcpyDeviceToHost);
    else for(int i=0;i<g_desc.voxelCount;i++)masks[i]=0;
    return e==cudaSuccess?0:-(int)e;
}
'''
target = destination/relative
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(data+addition.encode())
manifest['files'].append(dict(path=relative, sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size))
manifest.update(variant=VARIANT, parent_variant=PARENT, status='read-only surface provenance diagnostic; not a deployable repair')
(P/(VARIANT+'.json')).write_text(json.dumps(manifest, indent=2)+'\n')
builder = (P/'build_experiment.py').read_text()
builder = builder.replace("choices=['release-r21-fragment-submission']", "choices=['release-r21-fragment-submission', 'release-r21-topology-probe']")
builder = builder.replace("('release-r21-fragment-submission',)", "('release-r21-fragment-submission', 'release-r21-topology-probe')")
(P/'build_experiment.py').write_text(builder, encoding='utf8', newline='\n')
print(VARIANT)
