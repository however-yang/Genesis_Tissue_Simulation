from pathlib import Path
import json
p=Path('/srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104')
path=p/'native/source/vendor/cuda/xpbd/xpbd_solver.cu';s=path.read_text()
s=s.replace('if(mode&~(256|2048|8192|16384))return -10001;','if(mode&~(256|2048|8192|16384|32768))return -10001;')
needle='static bool runtimeSolverInitialized();';assert s.count(needle)==1
s=s.replace(needle,needle+'\nextern "C" LCS_COMBINED_EXPORT int LCS_GetRuntimeCoreMode(){return g_runtimeCoreMode;}')
needle='#include "../batch_observation.cuh"';assert s.count(needle)==1
s=s.replace(needle,needle+'\n#include "../zero_mode_gates.cuh"')
needle='    int query[3]={};';assert s.count(needle)==1
s=s.replace(needle,'''    if(g_runtimeCoreMode&32768){e=ensureRuntimeModeGates();if(e!=cudaSuccess)return -(int)e;}
    int query[5]={};''')
needle='''e=cudaMemcpyAsync(d_substepQuery+2,d_fragmentQueryCount,sizeof(int),cudaMemcpyDeviceToDevice,stream);if(e!=cudaSuccess)return e;''';assert s.count(needle)==1
s=s.replace(needle,needle+'\n    if(g_runtimeCoreMode&32768){e=launchRuntimeModeQueries(stream);if(e!=cudaSuccess)return e;}')
needle='''    e=cudaMemcpy(query,d_substepQuery,3*sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;''';assert s.count(needle)==1
s=s.replace(needle,'''    e=cudaMemcpy(query,(g_runtimeCoreMode&32768)?d_runtimeModeQueries:d_substepQuery,
        ((g_runtimeCoreMode&32768)?5:3)*sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    const bool skipRigid=(g_runtimeCoreMode&32768) && query[3]==0;
    const bool skipTranslation=(g_runtimeCoreMode&32768) && query[4]==0;
    if(g_runtimeCoreMode&32768){g_runtimeModeHostStats[0]++;g_runtimeModeHostStats[1]+=skipRigid;g_runtimeModeHostStats[2]+=skipTranslation;
        g_runtimeModeHostStats[3]+=!skipRigid;g_runtimeModeHostStats[4]+=!skipTranslation;}''')
needle='''    if(g_fragmentGraphQueries[fragmentGraphSlot]!=(int)anyFragmentQueries)selectedGraphDirty=true;''';assert s.count(needle)==1
s=s.replace(needle,needle+'''
    const int runtimeModeKey=(skipRigid?1:0)|(skipTranslation?2:0)|(g_queryValidate?4:0);
    if((g_runtimeCoreMode&32768) && g_runtimeModeGraphKey[fragmentGraphSlot]!=runtimeModeKey)selectedGraphDirty=true;''')
start=s.index('            { // Retain rotational inertia in every material/contact sweep.')
end=s.index('            xpbd_section_mark("loop: rigid modes");',start)
block=s[start:end]
assert block.rstrip().endswith('}')
body=block[block.index('\n')+1:block.rfind('            }')]
prefix_end=body.index('                k_RigidCenters')
basis=body[:prefix_end];body=body[prefix_end:]
replacement=basis+'''            if(!skipRigid || g_queryValidate){ // Reference proof retains the original stage.
                if(skipRigid && g_queryValidate)cudaMemcpyAsync(d_runtimeModeBefore,d_pos,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice,g_captureStream);
'''+body+'''
                if(skipRigid && g_queryValidate)k_RuntimeProveNoPositionWrite<<<Gr(g_N),64,0,g_captureStream>>>(g_N,d_runtimeModeBefore,d_pos,d_runtimeModeProof,0);
            }
'''
s=s[:start]+replacement+s[end:]
needle='g_fragmentGraphQueries[fragmentGraphSlot]=(int)anyFragmentQueries;';assert s.count(needle)==1
s=s.replace(needle,needle+'g_runtimeModeGraphKey[fragmentGraphSlot]=runtimeModeKey;')
start=s.index('    if(anyFragmentQueries)k_CoarseWeights<true>',s.index('k_MaterialReady<<<1,1>>>(d_unified,false);'))
end=s.index('    xpbd_section_mark("post: free-component nullspace");',start)
block=s[start:end]
basis_end=block.index('    launchCoarseAccumulateCache')
basis=block[:basis_end];block=block[basis_end:]
s=s[:start]+basis+'''    if(!skipTranslation || g_queryValidate){
        if(skipTranslation && g_queryValidate)cudaMemcpyAsync(d_runtimeModeBefore,d_pos,(size_t)g_N*sizeof(float3),cudaMemcpyDeviceToDevice);
'''+block+'''        if(skipTranslation && g_queryValidate)k_RuntimeProveNoPositionWrite<<<Gr(g_N),64>>>(g_N,d_runtimeModeBefore,d_pos,d_runtimeModeProof,3);
    }
'''+s[end:]
s=s.replace('    freeRuntimeGripTerms();','    freeRuntimeGripTerms();\n    freeRuntimeModeGates();')
path.write_text(s)
print(json.dumps({'mode':32768,'shortcut':'only original stages whose application is disabled by persistent grip/support','positive_fallback':'free/unheld or unanchored component runs original stage','validation':'execute original stages and compare every position bit when shortcut is requested','original_material_contact_cut_iterations_changed':False}))
