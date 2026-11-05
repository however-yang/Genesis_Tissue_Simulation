// Counterfactual assembly on private outputs; live state and production cache
// are untouched. A damaged-cache negative control must be rejected by callers.
__global__ void k_StencilProbeDamage(int count,const int* live,MaterialTetCache* cache){
    if(threadIdx.x || blockIdx.x)return;for(int t=0;t<count;t++)if(live[t]){cache[t].F[0].x+=.1f;return;}
}
__global__ void k_StencilOutputDamage(int count,Symmetric3* h){if(count>0 && !threadIdx.x && !blockIdx.x)h[0].xx+=.1f;}
extern "C" LCS_COMBINED_EXPORT int LCS_MaterialStencilProbe(int damage,unsigned long long* stats,double* errors){
    if(!stats || !errors || !g_ready || !d_unified || (damage!=0 && damage!=1))return -10017;
    struct Work{
        MaterialTetCache* cache=nullptr;__half *matrix[2]={};float *scale[2]={};float3 *r[2]={};Symmetric3 *h[2]={};
        ~Work(){cudaDeviceSynchronize();queryPrivateFree(cache);for(int i=0;i<2;i++){queryPrivateFree(matrix[i]);queryPrivateFree(scale[i]);queryPrivateFree(r[i]);queryPrivateFree(h[i]);}}
    }w;
    cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return -(int)e;
#define STENCIL_PROBE_ALLOC(p,size) do{e=queryPrivateAlloc(&(p),(size));if(e!=cudaSuccess)return -(int)e;e=cudaMemset((p),0,(size));if(e!=cudaSuccess)return -(int)e;}while(0)
    STENCIL_PROBE_ALLOC(w.cache,(size_t)g_T*sizeof(MaterialTetCache));
    for(int i=0;i<2;i++){
        STENCIL_PROBE_ALLOC(w.matrix[i],(size_t)72*g_N*sizeof(__half));STENCIL_PROBE_ALLOC(w.scale[i],(size_t)8*g_N*sizeof(float));
        STENCIL_PROBE_ALLOC(w.r[i],(size_t)g_N*sizeof(float3));STENCIL_PROBE_ALLOC(w.h[i],(size_t)g_N*sizeof(Symmetric3));
    }
#undef STENCIL_PROBE_ALLOC
    k_MaterialStencilAssemble<true,false><<<Gr(g_activeNodes),64>>>(d_unified,w.matrix[0],w.scale[0],w.r[0],w.h[0]);
#ifdef ENDO_KIDNEY_SHARED_STENCIL
    k_MaterialStencilAssemble<true,false,true><<<Gr(g_activeNodes),64>>>(d_unified,w.matrix[1],w.scale[1],w.r[1],w.h[1]);
    if(damage)k_StencilOutputDamage<<<1,1>>>(g_N,w.h[1]);
#else
    k_MaterialTetCache<<<Gr(g_T),64>>>(d_unified,w.cache);
    if(damage)k_StencilProbeDamage<<<1,1>>>(g_T,d_tetActive,w.cache);
    k_MaterialStencilAssemble<true,true><<<Gr(g_activeNodes),64>>>(d_unified,w.matrix[1],w.scale[1],w.r[1],w.h[1],w.cache);
#endif
    for(int k=0;k<9;k++)stats[k]=0;for(int k=0;k<8;k++)errors[k]=0;
    for(int kind=0;kind<4;kind++){
        size_t count=(size_t)g_N*(kind==0?72:kind==1?8:kind==2?3:sizeof(Symmetric3)/sizeof(float));
        size_t element=kind==0?sizeof(__half):sizeof(float);
        const void* a=kind==0?(void*)w.matrix[0]:kind==1?(void*)w.scale[0]:kind==2?(void*)w.r[0]:(void*)w.h[0];
        const void* b=kind==0?(void*)w.matrix[1]:kind==1?(void*)w.scale[1]:kind==2?(void*)w.r[1]:(void*)w.h[1];
        std::vector<unsigned char> first(count*element),second(count*element);
        e=cudaMemcpy(first.data(),a,first.size(),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
        e=cudaMemcpy(second.data(),b,second.size(),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
        stats[2*kind]=count;
        for(size_t j=0;j<count;j++){
            stats[2*kind+1]+=std::memcmp(first.data()+j*element,second.data()+j*element,element)!=0;
            double x,y;
            if(kind==0){__half u,v;std::memcpy(&u,first.data()+j*element,element);std::memcpy(&v,second.data()+j*element,element);x=__half2float(u);y=__half2float(v);}
            else{float u,v;std::memcpy(&u,first.data()+j*element,element);std::memcpy(&v,second.data()+j*element,element);x=u;y=v;}
            if(!std::isfinite(x) || !std::isfinite(y)){stats[8]++;continue;}
            errors[2*kind]=std::max(errors[2*kind],std::abs(x-y));errors[2*kind+1]=std::max(errors[2*kind+1],std::abs(x));
        }
    }
    return 0;
}
