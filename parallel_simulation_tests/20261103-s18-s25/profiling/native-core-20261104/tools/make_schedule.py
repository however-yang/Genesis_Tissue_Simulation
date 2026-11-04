"""Expose launch-grid sizes for complete grid-stride kernels; arithmetic unchanged."""
from pathlib import Path
import shutil
p=Path(__file__).resolve().parents[1];s=p/'native-scheduled/source';s.parent.mkdir(exist_ok=False);shutil.copytree(p/'native-batched/source',s)
f=s/'vendor/cuda/solver/material/unified_vertex.cuh';b=f.read_bytes();assert b.count(b'constexpr int ASSEMBLY_BLOCKS=512;')==1 and b.count(b'constexpr int MANIFOLD_BLOCKS=1024;')==1;b=b.replace(b'constexpr int ASSEMBLY_BLOCKS=512;',b'static int ASSEMBLY_BLOCKS=512;').replace(b'constexpr int MANIFOLD_BLOCKS=1024;',b'static int MANIFOLD_BLOCKS=1024;');f.write_bytes(b)
x=s/'vendor/cuda/xpbd/xpbd_solver.cu';text=x.read_text();text+='''
extern "C" int xpbd_native_schedule(int assembly,int manifold){
    if(g_ready || (assembly!=64 && assembly!=128 && assembly!=256 && assembly!=512) ||
       (manifold!=64 && manifold!=128 && manifold!=256 && manifold!=512 && manifold!=1024))return -10004;
    ASSEMBLY_BLOCKS=assembly;MANIFOLD_BLOCKS=manifold;return 0;
}
extern "C" int xpbd_native_workload(int* values){
    if(!values || !g_ready)return -10005;
    values[0]=g_N;values[1]=g_T;values[2]=g_surfaceCount;values[3]=ASSEMBLY_BLOCKS;values[4]=MANIFOLD_BLOCKS;
    values[5]=values[6]=values[7]=values[8]=0;
    UnifiedState state{};cudaError_t e=cudaSuccess;
    if(d_unified){e=cudaMemcpy(&state,d_unified,sizeof(state),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
        values[6]=state.grips;if(state.contactCount){e=cudaMemcpy(values+5,state.contactCount,sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;}}
    if(d_bladeState){e=cudaMemcpy(&state,d_bladeState,sizeof(state),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
        values[8]=state.grips;if(state.contactCount){e=cudaMemcpy(values+7,state.contactCount,sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;}}
    return 0;
}
''';x.write_text(text)
x=s/'vendor/src/plugin_api.cpp';x.write_bytes(x.read_bytes()+b'''\nextern "C" int xpbd_native_schedule(int,int);\nextern "C" int xpbd_native_workload(int*);\nLCS_API int LCS_SetNativeSchedule(int assembly,int manifold){return xpbd_native_schedule(assembly,manifold);}\nLCS_API int LCS_ReadNativeWorkload(int* values){return xpbd_native_workload(values);}\n''')
# New benchmark freezes execution settings as explicit provenance; physics stays unchanged.
f=p/'tools/profile_native.py';text=f.read_text();text=text.replace("p.add_argument('--query-mode'","p.add_argument('--assembly-blocks',type=int,choices=[64,128,256,512],default=512);p.add_argument('--manifold-blocks',type=int,choices=[64,128,256,512,1024],default=1024);p.add_argument('--workload',action='store_true');p.add_argument('--query-mode'",1)
text=text.replace('query_mode=a.query_mode,frames=[]','query_mode=a.query_mode,assembly_blocks=a.assembly_blocks,manifold_blocks=a.manifold_blocks,workload_profiled=a.workload,frames=[]')
needle=" elif a.query_mode:raise RuntimeError('Requested batch mode absent from native library')\n";extra=""" elif a.query_mode:raise RuntimeError('Requested batch mode absent from native library')
 if hasattr(native_control,'LCS_SetNativeSchedule'):
  native_control.LCS_SetNativeSchedule.argtypes=[ctypes.c_int,ctypes.c_int];native_control.LCS_SetNativeSchedule.restype=ctypes.c_int
  assert native_control.LCS_SetNativeSchedule(a.assembly_blocks,a.manifold_blocks)==0
 elif (a.assembly_blocks,a.manifold_blocks)!=(512,1024):raise RuntimeError('Requested launch grid absent from native library')
""";assert text.count(needle)==1;text=text.replace(needle,extra)
needle='    frame_id+=1\n';extra="""    if a.workload and frame_id%10==0:
     values=(ctypes.c_int*9)();native_control.LCS_ReadNativeWorkload.argtypes=[ctypes.POINTER(ctypes.c_int)];native_control.LCS_ReadNativeWorkload.restype=ctypes.c_int
     assert native_control.LCS_ReadNativeWorkload(values)==0
     result.setdefault('workload_samples',[]).append(dict(frame_id=frame_id,stage=stage.name,nodes=values[0],tets=values[1],surface_triangles=values[2],assembly_blocks=values[3],manifold_blocks=values[4],grasper_manifold=values[5],grasper_anchors=values[6],blade_manifold=values[7],blade_anchors=values[8]))
    frame_id+=1
""";assert text.count(needle)==1;text=text.replace(needle,extra);(p/'tools/profile_scheduled.py').write_text(text);compile(text,str(p/'tools/profile_scheduled.py'),'exec')
f=p/'tools/build.py';text=f.read_text().replace("base=p/'native-batched'","base=p/'native-scheduled'").replace("logs/build.txt","logs/build-scheduled.txt").replace("results/build.json","results/build-scheduled.json");(p/'tools/build_scheduled.py').write_text(text)
print('Prepared grid-stride scheduling experiment; no device arithmetic or iteration changed.')
