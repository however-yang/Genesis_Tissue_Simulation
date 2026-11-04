"""R24: isolated boundary grouping; physical cutting/link masks stay unchanged."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

P = Path(__file__).resolve().parent/'package'
ROOT = P.parents[2]
OLD = ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes/package'
PARENT = 'release-r19-volume-gate'
VARIANT = 'release-r24-material-boundary'
manifest = json.loads((OLD/(PARENT+'.json')).read_text())
destination = P/'overrides'/VARIANT
if destination.exists():
    raise FileExistsError(destination)
for item in manifest['files']:
    source = OLD/'overrides'/item.get('source_variant', PARENT)/item['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
    target = destination/item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    item.pop('source_variant', None)


def base(relative):
    target = destination/relative
    assert not target.exists()
    content = subprocess.check_output(['git', 'show', manifest['base_commit']+':'+manifest['base_source_prefix']+relative], cwd=ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return content.decode().replace('\r\n', '\n')


def once(text, old, new):
    assert text.count(old) == 1, old[:160]
    return text.replace(old, new)


text = base('vendor/cuda/cut.cu')
text = once(text, 'static unsigned int*  d_voxelCutMask   = nullptr;', '''// Keep the actual severed-edge mask independent from surface-only grouping.
// Air corners may still OWN real skin/cut witnesses; none are discarded.
static unsigned* d_surfaceBoundaryMask=nullptr;
static unsigned* d_airBoundaryMask=nullptr;
static bool g_materialBoundaryGraph=true;
unsigned* cut_surface_mask(){return d_surfaceBoundaryMask;}
extern "C" __declspec(dllexport) void LCS_SetMaterialBoundaryGraph(int enabled){g_materialBoundaryGraph=enabled!=0;}
static unsigned int*  d_voxelCutMask   = nullptr;''')
text = once(text, '''                                     unsigned int* voxelFPCount, unsigned int* dbg)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= voxelCount) return;
    unsigned int mask = voxelCutMask[v] & 0xFFFu;''', '''                                     unsigned int* voxelFPCount, unsigned int* dbg,
                                     const unsigned* airBoundary,unsigned* surfaceBoundary,bool enabled)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= voxelCount) return;
    unsigned int actual = voxelCutMask[v] & 0xFFFu;
    unsigned int mask = actual ? actual | (enabled?airBoundary[v]:0u) : 0u;
    surfaceBoundary[v]=mask;''')
start = text.index('__global__ void k_BuildCutTriangles(')
end = text.index('// ── DC_NormalizeCutFP', start)
part = text[start:end]
part = once(part, 'const unsigned int* voxelCutMask, const unsigned int* voxelOccupied,',
            'const unsigned int* voxelCutMask, const unsigned* surfaceBoundary, const unsigned int* voxelOccupied,')
part = once(part, 'unsigned int mask = voxelCutMask[v] & 0xFFFu;', 'unsigned int mask = surfaceBoundary[v] & 0xFFFu;')
# EdgeIsCut continues to use the physical mask: grouping cannot expose a new wall.
assert 'EdgeIsCut(axis, baseC, dims, voxelCutMask)' in part
text = text[:start]+part+text[end:]
anchor = '    CK(cudaMalloc(&d_voxelCutMask,    (size_t)vc     * sizeof(unsigned int)));'
text = once(text, anchor, anchor+'''
    CK(cudaMalloc(&d_surfaceBoundaryMask,(size_t)vc*sizeof(unsigned)));
    CK(cudaMalloc(&d_airBoundaryMask,(size_t)vc*sizeof(unsigned)));
    CK(cudaMemset(d_surfaceBoundaryMask,0,(size_t)vc*sizeof(unsigned)));
    // Geometry connectivity uses material intervals, not paths through air.
    // This is computed once from immutable inside flags. It never severs a link.
    std::vector<unsigned> airBoundary(vc,0u);
    const int offsets[8][3]={{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    const int ends[12][2]={{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
    for(int v=0;v<vc;v++){
        if(!voxelOccupied[v])continue;
        int x=v%desc->dimsX,y=(v/desc->dimsX)%desc->dimsY,z=v/(desc->dimsX*desc->dimsY);
        bool inside[8];
        for(int k=0;k<8;k++){
            int node=x+offsets[k][0]+(desc->dimsX+1)*(y+offsets[k][1]+(desc->dimsY+1)*(z+offsets[k][2]));
            inside[k]=cornerInside[node]!=0;
        }
        for(int e=0;e<12;e++)if(!inside[ends[e][0]] || !inside[ends[e][1]])airBoundary[v]|=1u<<e;
    }
    CK(cudaMemcpy(d_airBoundaryMask,airBoundary.data(),(size_t)vc*sizeof(unsigned),cudaMemcpyHostToDevice));''')
text = once(text, '''    k_LookupConnectivity<<<Gc(vc), 64>>>(vc, d_voxelCutMask, d_voxelOccupied, d_conn4096, d_voxelFPCount, d_dbg);''',
'''    k_LookupConnectivity<<<Gc(vc), 64>>>(vc, d_voxelCutMask, d_voxelOccupied, d_conn4096, d_voxelFPCount, d_dbg,
        d_airBoundaryMask,d_surfaceBoundaryMask,g_materialBoundaryGraph);''')
text = once(text, '                                       d_gridEdges, d_voxelOccupied, d_voxelCutMask, d_conn4096,',
            '                                       d_gridEdges, d_voxelOccupied, d_surfaceBoundaryMask, d_conn4096,')
text = once(text, '    k_ComputeComponentFP<<<Gc(vc), 64>>>(vc, g_cdims, d_voxelCutMask, d_voxelOccupied, d_conn4096,',
            '    k_ComputeComponentFP<<<Gc(vc), 64>>>(vc, g_cdims, d_surfaceBoundaryMask, d_voxelOccupied, d_conn4096,')
text = once(text, '                                                         d_voxelCutMask, d_voxelOccupied, d_conn4096, d_cutFP,',
            '                                                         d_voxelCutMask, d_surfaceBoundaryMask, d_voxelOccupied, d_conn4096, d_cutFP,')
text = once(text, '    cudaFree(d_voxelCutMask);    d_voxelCutMask = nullptr;', '''    cudaFree(d_voxelCutMask);    d_voxelCutMask = nullptr;
    cudaFree(d_surfaceBoundaryMask);d_surfaceBoundaryMask=nullptr;
    cudaFree(d_airBoundaryMask);d_airBoundaryMask=nullptr;g_materialBoundaryGraph=true;''')
# Reuse only the read-only exporter from the forensic probe.
probe = (P/'overrides/release-r24-cut-witness-probe/vendor/cuda/cut.cu').read_text()
text += probe[probe.index('// Opt-in, read-only forensic capture.'):]
text += '''
extern "C" __declspec(dllexport) int LCS_ReadMaterialBoundaryMask(unsigned* output,int capacity){
    if(!output || !d_surfaceBoundaryMask || capacity<g_cdesc.voxelCount)return -1;
    return -(int)cudaMemcpy(output,d_surfaceBoundaryMask,(size_t)g_cdesc.voxelCount*sizeof(unsigned),cudaMemcpyDeviceToHost);
}
'''
(destination/'vendor/cuda/cut.cu').write_text(text, encoding='utf8', newline='\n')

text = base('vendor/cuda/dc_recon.cu')
text = once(text, '                             cutActive, d_cut_voxelCutMask, d_cut_voxelOccupied, d_cut_conn4096,',
            '                             cutActive, cutActive?cut_surface_mask():nullptr, d_cut_voxelOccupied, d_cut_conn4096,')
text = once(text, 'g_cutActive?d_cut_voxelCutMask:nullptr,g_cutActive?(const int*)d_cut_conn4096:nullptr);',
            'g_cutActive?cut_surface_mask():nullptr,g_cutActive?(const int*)d_cut_conn4096:nullptr);')
# The same grouping MUST reach contact embedding; otherwise forces attach to
# unrelated corners. Intact advection still uses the unchanged physical mask.
probe = (P/'overrides/release-r24-cut-witness-probe/vendor/cuda/dc_recon.cu').read_text()
text += probe[probe.index('// Read-only diagnostic.'):]
(destination/'vendor/cuda/dc_recon.cu').write_text(text, encoding='utf8', newline='\n')
text = base('vendor/cuda/cut.h')
text += '\n// Surface grouping only; never pass this to physical cutting or severing.\nunsigned* cut_surface_mask();\n'
(destination/'vendor/cuda/cut.h').write_text(text, encoding='utf8', newline='\n')
for relative in ('vendor/cuda/cut.cu', 'vendor/cuda/dc_recon.cu', 'vendor/cuda/cut.h'):
    data = (destination/relative).read_bytes()
    manifest['files'].append(dict(path=relative, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
manifest.update(variant=VARIANT, parent_variant=PARENT,
    status='isolated surface boundary hypothesis; physical cut mask unchanged; pending full geometry, interaction and timing validation; never default')
(P/(VARIANT+'.json')).write_text(json.dumps(manifest, indent=2)+'\n')
builder = (P/'build_experiment.py').read_text()
last = 'release-r24-cut-witness-probe'
assert builder.count("'"+last+"']") == 1
assert builder.count("'"+last+"')") == 1
builder = builder.replace("'"+last+"']", "'"+last+"', '"+VARIANT+"']")
builder = builder.replace("'"+last+"')", "'"+last+"', '"+VARIANT+"')")
(P/'build_experiment.py').write_text(builder, encoding='utf8', newline='\n')
print(json.dumps(dict(variant=VARIANT, parent=PARENT, physical_cut_mask_unchanged=True,
                     contact_embedding_uses_same_surface_groups=True, default_changed=False)))
