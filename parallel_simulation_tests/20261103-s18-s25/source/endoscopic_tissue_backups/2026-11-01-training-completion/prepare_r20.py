"""Prepare a separate R19-derived cache experiment; verify all inherited bytes."""
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
OLD = ROOT / 'endoscopic_tissue_backups/2026-10-31-interaction-fixes/package'
NEW = P / 'package'
VARIANT = 'release-r20-support-cache'


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Patch does not match exactly once: ' + old[:100])
    return text.replace(old, new)


def main():
    NEW.mkdir(exist_ok=True)
    manifest = json.loads((OLD/'release-r19-volume-gate.json').read_text())
    for item in manifest['files']:
        original = OLD/'overrides'/item.get('source_variant', 'release-r19-volume-gate')/item['path']
        if hashlib.sha256(original.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Inherited override mismatch: '+str(original))
        target = NEW/'overrides'/VARIANT/item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        item.pop('source_variant', None)
    file = NEW/'overrides'/VARIANT/'vendor/cuda/contact/fragment_contact.cuh'
    text = file.read_text()
    text = once(text, '__global__ void k_FragmentSupports(int n,const int* component,const float* inverseMass,const int* active,int* supported){', '''// R20: only these topology-derived flags are cached. Grasp/contact queries stay live.
__global__ void k_ClearFragmentSupport(int n,int* support,int* volume,const unsigned* stale){
    if(!*stale)return;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)support[i]=0;if(i==0)*volume=0;
}
__global__ void k_FragmentSupports(int n,const int* component,const float* inverseMass,const int* active,int* supported,const unsigned* stale=nullptr){
    if(stale && !*stale)return;''')
    text = once(text, 'int* unpinnedVolume=nullptr){\n    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !live[i])return;', 'int* unpinnedVolume=nullptr,const unsigned* stale=nullptr){\n    if(stale && !*stale)return;\n    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !live[i])return;')
    file.write_text(text, newline='\n')
    file = NEW/'overrides'/VARIANT/'vendor/cuda/xpbd/xpbd_solver.cu'
    text = file.read_text()
    text = once(text, 'static unsigned* d_edgeSupportRevision=nullptr;', 'static unsigned* d_edgeSupportRevision=nullptr;\nstatic unsigned* d_fragmentSupportRevision=nullptr;')
    text = once(text, 'static void markLivenessStale(){if(d_livenessRevision)cudaMemset(d_livenessRevision,0xff,4*sizeof(unsigned));}', 'static void markLivenessStale(){if(d_livenessRevision)cudaMemset(d_livenessRevision,0xff,4*sizeof(unsigned));\n    if(d_fragmentSupportRevision)cudaMemset(d_fragmentSupportRevision,0xff,2*sizeof(unsigned));}')
    text = once(text, 'cudaMalloc(&d_topologyRecords, 8 * sizeof(unsigned))', 'cudaMalloc(&d_topologyRecords, 10 * sizeof(unsigned))')
    text = once(text, 'cudaMemset(d_topologyRecords, 0xff, 8 * sizeof(unsigned))', 'cudaMemset(d_topologyRecords, 0xff, 10 * sizeof(unsigned))')
    text = once(text, 'd_multigridRevision = d_topologyRecords + 6;', 'd_multigridRevision = d_topologyRecords + 6;\n    d_fragmentSupportRevision = d_topologyRecords + 8;')
    text = once(text, '''    cudaMemsetAsync(d_fragmentSupported,0,g_N*sizeof(int));
    cudaMemsetAsync(d_fragmentUnpinnedVolume,0,sizeof(int));
    k_FragmentSupports<<<Gr(g_N),64>>>(g_N,d_unifiedComponents,d_invMass,g_liveActive,d_fragmentSupported);
    if(g_T>0)k_FragmentVolumes<<<Gr(g_T),64>>>(g_T,d_tetIds,d_tetActive,d_unifiedComponents,d_fragmentSupported,d_fragmentUnpinnedVolume);''', '''    k_TopologyStale<<<1,1>>>(cut_topology_revision_device(),d_fragmentSupportRevision);
    const unsigned* stale=d_fragmentSupportRevision+1;
    k_ClearFragmentSupport<<<Gr(g_N),64>>>(g_N,d_fragmentSupported,d_fragmentUnpinnedVolume,stale);
    k_FragmentSupports<<<Gr(g_N),64>>>(g_N,d_unifiedComponents,d_invMass,g_liveActive,d_fragmentSupported,stale);
    if(g_T>0)k_FragmentVolumes<<<Gr(g_T),64>>>(g_T,d_tetIds,d_tetActive,d_unifiedComponents,d_fragmentSupported,d_fragmentUnpinnedVolume,stale);''')
    text = once(text, 'd_topologyRecords = d_componentRevision = d_livenessRevision = d_edgeSupportRevision = d_multigridRevision = nullptr;', 'd_topologyRecords = d_componentRevision = d_livenessRevision = d_edgeSupportRevision = d_multigridRevision = d_fragmentSupportRevision = nullptr;')
    text += '''
// Read-only audit of every cached support integer against the old full rescan.
// No ordinary simulation call uses this export; it writes temporary buffers only.
extern "C" __declspec(dllexport) int LCS_ValidateFragmentSupportCache(int* result,int capacity){
    if(!result || capacity<6 || !d_fragmentSupported || !d_fragmentUnpinnedVolume)return -1;
    int *support=nullptr,*volume=nullptr;cudaError_t e=cudaMalloc(&support,g_N*sizeof(int));
    if(e==cudaSuccess)e=cudaMalloc(&volume,sizeof(int));
    auto finish=[&](){cudaFree(support);cudaFree(volume);return e==cudaSuccess?0:-(int)e;};
    if(e!=cudaSuccess)return finish();
    e=cudaMemset(support,0,g_N*sizeof(int));if(e==cudaSuccess)e=cudaMemset(volume,0,sizeof(int));
    if(e!=cudaSuccess)return finish();
    k_FragmentSupports<<<Gr(g_N),64>>>(g_N,d_unifiedComponents,d_invMass,g_liveActive,support);
    if(g_T>0)k_FragmentVolumes<<<Gr(g_T),64>>>(g_T,d_tetIds,d_tetActive,d_unifiedComponents,support,volume);
    e=cudaGetLastError();std::vector<int> a(g_N),b(g_N);int av=0,bv=0;unsigned revision[2]={};
    if(e==cudaSuccess)e=cudaMemcpy(a.data(),support,g_N*sizeof(int),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(b.data(),d_fragmentSupported,g_N*sizeof(int),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(&av,volume,sizeof(int),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(&bv,d_fragmentUnpinnedVolume,sizeof(int),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(revision,d_fragmentSupportRevision,sizeof(revision),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return finish();int different=0;for(int i=0;i<g_N;i++)different+=a[i]!=b[i];
    result[0]=different;result[1]=av;result[2]=bv;result[3]=g_N;result[4]=(int)revision[0];result[5]=(int)revision[1];
    return finish();
}
'''
    file.write_text(text, newline='\n')
    for item in manifest['files']:
        data = (NEW/'overrides'/VARIANT/item['path']).read_bytes()
        item['sha256'], item['bytes'] = hashlib.sha256(data).hexdigest(), len(data)
    manifest.update(variant=VARIANT, parent='release-r19-volume-gate', default_replaced=False)
    (NEW/(VARIANT+'.json')).write_text(json.dumps(manifest, indent=2)+'\n')
    build = (OLD/'build_experiment.py').read_text()
    choices = "['release-r16', 'release-r17', 'vertex-r18', 'baseline-rim-trace2', 'interaction-r13-trace', 'release-r19-volume-gate']"
    build = once(build, choices, repr([VARIANT]))
    build = once(build, "('release-r16', 'release-r17', 'vertex-r18', 'baseline-rim-trace2', 'interaction-r13-trace', 'release-r19-volume-gate')", repr((VARIANT,)))
    (NEW/'build_experiment.py').write_text(build)
    shutil.copyfile(OLD/'CMakeLists.txt', NEW/'CMakeLists.txt')
    (NEW/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
    print('R20 isolated source prepared; all inherited hashes verified')


if __name__ == '__main__':
    main()
