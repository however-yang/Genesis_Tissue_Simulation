from pathlib import Path
import shutil
p=Path('/srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104');src=p/'native-query-v3/source';shutil.copytree(p/'native-tool-v2/source',src)
f=src/'vendor/cuda/contact/surface_contact.cuh';s=f.read_text()
a=s.index('__device__ inline float3 triangleWitness(');b=s.index('// Exact finite-segment/triangle distance',a);helper=s[a:b].replace('triangleWitness(', 'triangleWitnessReuse(',1)
needle='float3 p=a*bar.x+b*bar.y+c*bar.z;shapeDistance(shape,p,n,s);';assert helper.count(needle)==1
helper=helper.replace(needle,'float3 p=a*bar.x+b*bar.y+c*bar.z;\n        // End-of-iteration evaluation supplies this exact point/normal on iterations 1..11.\n        if(it==0)shapeDistance(shape,p,n,s);',1)
s=s[:b]+helper+s[b:]
a=s.index('__global__ void k_FindSurfaceLeanRefs(');b=s.index('static_assert(sizeof(SurfaceContact)',a);kernel=s[a:b]
kernel=kernel.replace('__global__ void k_FindSurfaceLeanRefs(', 'template<int MinBlocks> __global__ __launch_bounds__(64,MinBlocks) void k_FindSurfaceFast(',1)
kernel=kernel.replace('bool keep=body && contacts[slot].count>0;', 'bool keep=body && !to.capsule && contacts[slot].count>0;',1)
kernel=kernel.replace('if(!keep && body && cacheSlots', 'if(!to.capsule && !keep && body && cacheSlots')
kernel=kernel.replace('triangleWitness(', 'triangleWitnessReuse(')
comment='// Capsule outputs use either preloadPlane or a fresh capsule trace; retainedPlane\n// and retained.stick are consumed ONLY by the box branch. Bypass their read-only\n// cache search and geometric reconstruction for capsules, without touching preload.\n'
s=s[:b]+comment+kernel+s[b:]
needle='        if(g_coreOptimizations&4){';assert needle in s
args='<<<dim3((triangles+g_queryThreads-1)/g_queryThreads,kindCount),g_queryThreads,0,stream>>>(triangles,packed,bladeBatch,from,to,skin,\n            vertices,masks,prev,pos,contacts,degree,current,previous,body,thickness,minGap,stationary,cacheSlots,cache,bladePhase,stride,groupBase,w.ids,w.counts);'
s=s.replace(needle,'        if((g_coreOptimizations&8) && g_queryThreads<=64){\n            if(g_coreOptimizations&16){k_FindSurfaceFast<8>'+args+'}\n            else{k_FindSurfaceFast<1>'+args+'}\n        }else if(g_coreOptimizations&4){',1);f.write_text(s)
f=src/'vendor/cuda/xpbd/xpbd_solver.cu';s=f.read_text().replace('mode>7 || g_ready','mode>31 || g_ready');f.write_text(s)
s=(p/'tools/build_tool_v2.py').read_text().replace('native-tool-v2','native-query-v3').replace('build-tool-v2','build-query-v3');(p/'tools/build_query_v3.py').write_text(s)
print(src)
