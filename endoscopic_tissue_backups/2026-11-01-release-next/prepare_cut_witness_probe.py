"""Freeze actual edge fractions and cut witnesses; never modify a live state."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

P = Path(__file__).resolve().parent/'package'
ROOT = P.parents[2]
PARENT = 'release-r21-topology-probe'
VARIANT = 'release-r24-cut-witness-probe'
manifest = json.loads((P/(PARENT+'.json')).read_text())
destination = P/'overrides'/VARIANT
if destination.exists():
    raise FileExistsError(destination)
for item in manifest['files']:
    source = P/'overrides'/PARENT/item['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
    target = destination/item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
relative = 'vendor/cuda/cut.cu'
data = subprocess.check_output(['git', 'show', manifest['base_commit']+':'+manifest['base_source_prefix']+relative], cwd=ROOT)
addition = r'''

// Opt-in, read-only forensic capture. No new device kernel or state mutation.
// Query the nine sizes first with all data buffers null, then supply capacities.
extern "C" __declspec(dllexport) int LCS_ReadCutWitnesses(
        int* counts,void* points,int pointCapacity,void* edges,int edgeCapacity,
        void* rotations,int cornerCapacity,float* cuts,float* skin){
    if(!counts || !d_cutPointCounter || !c_particleRot)return -1;
    unsigned n=0;cudaError_t e=cudaMemcpy(&n,d_cutPointCounter,sizeof(unsigned),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    if(n>(unsigned)g_cdesc.cutPointCapacity)return -2;
    counts[0]=g_cdims.x;counts[1]=g_cdims.y;counts[2]=g_cdims.z;
    counts[3]=g_cdesc.cornerCount;counts[4]=g_cdesc.gridEdgeCount;counts[5]=(int)n;
    counts[6]=sizeof(CutPointGpu);counts[7]=sizeof(GridEdgeGpu2);counts[8]=sizeof(RotGpu);
    if(!points && !edges && !rotations && !cuts && !skin)return 0;
    if(!points || !edges || !rotations || !cuts || !skin || pointCapacity<(int)n ||
       edgeCapacity<g_cdesc.gridEdgeCount || cornerCapacity<g_cdesc.cornerCount)return -3;
    e=cudaMemcpy(points,d_cutPoint,(size_t)n*sizeof(CutPointGpu),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(edges,d_gridEdges,(size_t)g_cdesc.gridEdgeCount*sizeof(GridEdgeGpu2),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(rotations,c_particleRot,(size_t)g_cdesc.cornerCount*sizeof(RotGpu),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(cuts,d_edgeCutT,(size_t)6*g_cdesc.cornerCount*sizeof(float),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(skin,recon_surface_edge_t(),(size_t)6*g_cdesc.cornerCount*sizeof(float),cudaMemcpyDeviceToHost);
    return e==cudaSuccess?0:-(int)e;
}
'''
target = destination/relative
target.write_bytes(data+addition.encode())
manifest['files'].append(dict(path=relative, sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size))
manifest.update(variant=VARIANT, parent_variant=PARENT, status='read-only actual cut witness diagnostic; no physical change or default adoption')
(P/(VARIANT+'.json')).write_text(json.dumps(manifest, indent=2)+'\n')
builder = (P/'build_experiment.py').read_text()
last = 'release-r23-repair-prelude-proof'
assert builder.count("'"+last+"']") == 1
assert builder.count("'"+last+"')") == 1
builder = builder.replace("'"+last+"']", "'"+last+"', '"+VARIANT+"']")
builder = builder.replace("'"+last+"')", "'"+last+"', '"+VARIANT+"')")
(P/'build_experiment.py').write_text(builder, encoding='utf8', newline='\n')
print(VARIANT)
