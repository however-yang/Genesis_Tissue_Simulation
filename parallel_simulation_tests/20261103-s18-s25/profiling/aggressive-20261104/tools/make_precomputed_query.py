"""Last isolated native experiment: hoist uniform tool-shape construction only.

No compaction, kind specialization, contact filtering or solver changes. Every
query remains and a verbatim baseline kernel is retained for paired validation.
"""
from pathlib import Path
import shutil
p=Path(__file__).resolve().parents[1];target=p/'native-precomputed/source';target.parent.mkdir(exist_ok=False)
shutil.copytree(p/'native-gpu-diagnostics/source',target)
h=target/'vendor/cuda/contact/surface_contact.cuh';text=h.read_text();begin=text.index('__global__ void k_FindSurface(');end=text.index('// Bounded grid-stride launch',begin);original=text[begin:end]
optimized=original.replace('const unsigned char* groupRejected=nullptr){','const unsigned char* groupRejected=nullptr,const ContactShape* sampledShapes=nullptr){')
old='if(body){from=bodyContactShape(body,stationary?body->pose:body->previous,kind,thickness,minGap);to=bodyContactShape(body,body->pose,kind,thickness,minGap);}'
new='if(body){if(sampledShapes){from=sampledShapes[0];to=sampledShapes[1];}else{from=bodyContactShape(body,stationary?body->pose:body->previous,kind,thickness,minGap);to=bodyContactShape(body,body->pose,kind,thickness,minGap);}}'
assert optimized.count(old)==1;optimized=optimized.replace(old,new);assert optimized!=original
helpers=r'''
// Separate scratch for legacy stream zero and the solver's one capture stream.
// Any additional stream falls back to the old uniform calculation, avoiding sharing.
static ContactShape* d_querySampleZero=nullptr;
static ContactShape* d_querySampleCapture=nullptr;
static cudaStream_t g_querySampleCaptureStream=nullptr;
static cudaError_t allocateQuerySample(){
    cudaError_t status=cudaMalloc(&d_querySampleZero,2*sizeof(ContactShape));
    if(status!=cudaSuccess)return status;
    status=cudaMalloc(&d_querySampleCapture,2*sizeof(ContactShape));
    if(status!=cudaSuccess){cudaFree(d_querySampleZero);d_querySampleZero=nullptr;}
    return status;
}
static void freeQuerySample(){
    cudaDeviceSynchronize();
    cudaFree(d_querySampleZero);d_querySampleZero=nullptr;
    cudaFree(d_querySampleCapture);d_querySampleCapture=nullptr;g_querySampleCaptureStream=nullptr;
}
__global__ void k_SampleSurfaceShapes(ContactShape* shapes,const CoupledTool* body,int kind,float thickness,float minGap,bool stationary){
    if(threadIdx.x || blockIdx.x)return;
    shapes[0]=bodyContactShape(body,stationary?body->pose:body->previous,kind,thickness,minGap);
    shapes[1]=bodyContactShape(body,body->pose,kind,thickness,minGap);
}
'''
oldheader=(p/'native-cutting-optimized/source/vendor/cuda/contact/surface_contact.cuh').read_text();a=oldheader.index('// Diagnostic-only paired execution');b=oldheader.index('// Bounded grid-stride launch',a);wrapper=oldheader[a:b]
a=wrapper.index('    auto launch=');b=wrapper.index('    if(!g_findValidation)',a)
launch=r'''    auto specialized=[&](){
        ContactShape* shapes=stream?d_querySampleCapture:d_querySampleZero;
        if(stream){
            if(!g_querySampleCaptureStream)g_querySampleCaptureStream=stream;
            else if(g_querySampleCaptureStream!=stream)shapes=nullptr;
        }
        if(body && shapes)k_SampleSurfaceShapes<<<1,32,0,stream>>>(shapes,body,kind,thickness,minGap,stationary);
        else shapes=nullptr;
        k_FindSurface<<<blocks,64,0,stream>>>(FIND_ARGUMENTS,shapes);
    };
'''
wrapper=wrapper[:a]+launch+wrapper[b:]
h.write_text(text[:begin]+original.replace('void k_FindSurface(','void k_FindSurfaceReference(',1)+helpers+optimized+wrapper+text[end:])
x=target/'vendor/cuda/xpbd/xpbd_solver.cu';x.write_text((p/'native-grasper-specialized/source/vendor/cuda/xpbd/xpbd_solver.cu').read_text());s=x.read_text();assert s.count('    g_N = particleCount; g_E = edgeCount;')==1;s=s.replace('    g_N = particleCount; g_E = edgeCount;','    XK(allocateQuerySample());\n    g_N = particleCount; g_E = edgeCount;');assert s.count('extern "C" void xpbd_shutdown()\n{')==1;s=s.replace('extern "C" void xpbd_shutdown()\n{','extern "C" void xpbd_shutdown()\n{\n    freeQuerySample();');x.write_text(s)
b=target/'vendor/cuda/cutting/pressure_blade.cuh';s=(p/'native-cutting-optimized/source/vendor/cuda/cutting/pressure_blade.cuh').read_text();b.write_text(s)
api=target/'vendor/src/plugin_api.cpp';s=api.read_text();s+='\nextern "C" void xpbd_find_surface_validate(int);\nextern "C" void xpbd_find_surface_validation_read(int*);\nLCS_API void LCS_FindSurfaceValidate(int enabled){xpbd_find_surface_validate(enabled);}\nLCS_API void LCS_FindSurfaceValidationRead(int* values){xpbd_find_surface_validation_read(values);}\n';api.write_text(s)
print(target)
