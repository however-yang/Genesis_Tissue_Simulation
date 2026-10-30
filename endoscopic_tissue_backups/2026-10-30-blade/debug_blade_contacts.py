"""Add LCS_GetBladeContacts, a debug export of the scissors contacts, to the native sources.

For every contact of the blade's manifold it writes 11 floats: its kind (1 and 2 the cutting edges, 3 and 4 the blade
backs, 5 the wrist), its material witness, its normal, the force term the contact adds to the tissue's energy gradient
(the force on the tissue is its negative) and its gap. It also returns the manifold's contact count.

Usage: python debug_blade_contacts.py apply   (from endoscopic_tissue/native/vendor; undo with git checkout)
"""
import sys
from pathlib import Path


def patch(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old[:60])
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


assert sys.argv[1:] == ["apply"]
patch("cuda/cutting/pressure_blade.cuh", "\n// Include the cutter in every coarse direction while leaving its own pose\n", """
__global__ void k_ExportBladeContacts(const UnifiedState* state,float* out,int capacity){
    const UnifiedState& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.contactCount && k<capacity;k+=blockDim.x*gridDim.x){
        const SurfaceContact& c=s.contacts[s.contactIds[k]];float3 f;Symmetric3 H={};unifiedContact(s,c,f,H);
        float3 x=unifiedPoint(c,s.pos),normal;float gap=unifiedGap(s,c,x,normal);
        float* o=out+11*k;o[0]=(float)c.kind;o[1]=x.x;o[2]=x.y;o[3]=x.z;o[4]=normal.x;o[5]=normal.y;o[6]=normal.z;
        o[7]=f.x;o[8]=f.y;o[9]=f.z;o[10]=gap;
    }
}
extern "C" int xpbd_get_blade_contacts(float* values,int capacity,int* count){
    if(!values || !count || capacity<0 || !g_blade.enabled || !d_bladeState || !d_bladeManifoldCount)return XPBD_ERR_BAD_ARG;
    int n=0;cudaError_t e=cudaMemcpy(&n,d_bladeManifoldCount,sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    *count=n;n=n<capacity?n:capacity;if(n<=0)return 0;
    float* d=nullptr;e=cudaMalloc(&d,(size_t)11*n*sizeof(float));if(e!=cudaSuccess)return -(int)e;
    k_ExportBladeContacts<<<(n+63)/64,64>>>(d_bladeState,d,n);
    e=cudaMemcpy(values,d,(size_t)11*n*sizeof(float),cudaMemcpyDeviceToHost);cudaFree(d);return -(int)e;
}

// Include the cutter in every coarse direction while leaving its own pose
""")
patch("cuda/xpbd/xpbd_solver.h", 'extern "C" int xpbd_get_scissors_loads(float* values7);',
      'extern "C" int xpbd_get_scissors_loads(float* values7);\nextern "C" int xpbd_get_blade_contacts(float* values,int capacity,int* count);')
patch("src/plugin_api.cpp", "LCS_API int LCS_GetScissorsLoads(float* values){return xpbd_get_scissors_loads(values);}",
      "LCS_API int LCS_GetScissorsLoads(float* values){return xpbd_get_scissors_loads(values);}\n"
      "LCS_API int LCS_GetBladeContacts(float* values,int capacity,int* count){return xpbd_get_blade_contacts(values,capacity,count);}")
print("applied")
