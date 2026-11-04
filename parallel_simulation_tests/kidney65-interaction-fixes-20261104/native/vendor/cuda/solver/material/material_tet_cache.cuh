// Derived only within one frozen-position stencil assembly. Rebuilt in the
// same stream immediately before every use, including after snapshot restore.
struct MaterialTetCache {float3 F[3],cof[3],G[3];float fft[9],ggt[9];};
#ifdef ENDO_KIDNEY_STENCIL_CACHE
static MaterialTetCache* d_materialTetCache=nullptr;
static int g_materialTetCacheCapacity=0;
static cudaError_t ensureMaterialTetCache(int count){
    if(count<=g_materialTetCacheCapacity)return cudaSuccess;
    if(g_materialTetCacheCapacity)return cudaErrorInvalidValue; // graphs retain this pointer until session close
    cudaError_t e=queryPrivateAlloc(&d_materialTetCache,(size_t)count*sizeof(MaterialTetCache));
    if(e==cudaSuccess)g_materialTetCacheCapacity=count;return e;
}
static void freeMaterialTetCache(){queryPrivateFree(d_materialTetCache);d_materialTetCache=nullptr;g_materialTetCacheCapacity=0;}
#endif
__global__ void k_MaterialTetCache(const UnifiedState* state,MaterialTetCache* cache){
    const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt || !s.live[t])return;
    int id[4];unifiedTetIds(s,t,id);
    float3 e[3]={s.pos[id[1]]-s.pos[id[0]],s.pos[id[2]]-s.pos[id[0]],s.pos[id[3]]-s.pos[id[0]]};
    float3 b[3]={unifiedKuhn(s,t).b0,unifiedKuhn(s,t).b1,unifiedKuhn(s,t).b2};MaterialTetCache out={};
    #pragma unroll
    for(int a=0;a<3;a++)out.F[a]=e[0]*compF3(b[0],a)+e[1]*compF3(b[1],a)+e[2]*compF3(b[2],a);
    out.cof[0]=cross3(out.F[1],out.F[2]);out.cof[1]=cross3(out.F[2],out.F[0]);out.cof[2]=cross3(out.F[0],out.F[1]);
    if(s.damping>0){
        float3 r[3]={s.previousInverse[3*t],s.previousInverse[3*t+1],s.previousInverse[3*t+2]};
        #pragma unroll
        for(int a=0;a<3;a++)out.G[a]=out.F[0]*compF3(r[0],a)+out.F[1]*compF3(r[1],a)+out.F[2]*compF3(r[2],a);
    }
    #pragma unroll
    for(int u=0;u<3;u++)
        #pragma unroll
        for(int v=0;v<3;v++)
            #pragma unroll
            for(int a=0;a<3;a++){
                out.fft[3*u+v]+=compF3(out.F[a],u)*compF3(out.F[a],v);
                if(s.damping>0)out.ggt[3*u+v]+=compF3(out.G[a],u)*compF3(out.G[a],v);
            }
    cache[t]=out;
}
__device__ inline void materialCachedDampingGradient(const UnifiedState& s,int t,float3 q,
    const MaterialTetCache& cache,float3& gradient){
    if(s.damping<=0)return;
    float3 r0=s.previousInverse[3*t],r1=s.previousInverse[3*t+1],r2=s.previousInverse[3*t+2];
    q=r0*q.x+r1*q.y+r2*q.z;
    float3 f0=cache.G[0],f1=cache.G[1],f2=cache.G[2];
    float3 fq=f0*q.x+f1*q.y+f2*q.z;
    float3 dq=mkf3(dot3(f0,fq),dot3(f1,fq),dot3(f2,fq))-q;
    float d=.5f*unifiedKuhn(s,t).muV*s.damping/s.h;
    gradient=gradient+(f0*dq.x+f1*dq.y+f2*dq.z)*d;
}
