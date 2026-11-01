"""Also freeze unused feature positions, needed to reproduce zero-area culling."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent/'package'
VARIANT = 'release-r24-cut-witness-probe'
path = P/(VARIANT+'.json')
manifest = json.loads(path.read_text())
relative = 'vendor/cuda/dc_recon.cu'
item = next(item for item in manifest['files'] if item['path'] == relative)
source = P/'overrides'/VARIANT/relative
data = source.read_bytes()
assert hashlib.sha256(data).hexdigest() == item['sha256']
assert b'LCS_ReadFrozenFeatures' not in data
source.write_bytes(data+br'''

// Includes unused points: a fully collapsed quad is absent from expanded soup.
extern "C" __declspec(dllexport) int LCS_ReadFrozenFeatures(
        float* external,int externalCapacity,float* cut,int cutCapacity){
    if(!g_ready || !external || !cut || !d_cut_cutFP ||
       externalCapacity<g_desc.voxelCount || cutCapacity<8*g_desc.voxelCount)return -1;
    cudaError_t e=cudaMemcpy(external,d_voxelExternalFP,(size_t)g_desc.voxelCount*sizeof(float4),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(cut,d_cut_cutFP,(size_t)8*g_desc.voxelCount*sizeof(float4),cudaMemcpyDeviceToHost);
    return e==cudaSuccess?0:-(int)e;
}
''')
item.update(sha256=hashlib.sha256(source.read_bytes()).hexdigest(), bytes=source.stat().st_size)
path.write_text(json.dumps(manifest, indent=2)+'\n')
print('Read-only feature export added; no GPU code changed.')
