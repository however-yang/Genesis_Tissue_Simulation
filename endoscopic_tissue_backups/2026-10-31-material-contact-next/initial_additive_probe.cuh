// Read-only probe of the real frozen hierarchy. All work arrays are private;
// no production residual, direction, hierarchy work vector or sum is changed.
#include <cstdio>
extern "C" __declspec(dllexport) int LCS_ProbeAdditiveMg(const char* filename,unsigned seed){
    if(!d_unified||!g_multigridLevels||!filename)return -901;
    std::vector<void*> owned;
    auto cleanup=[&](){for(void* p:owned)cudaFree(p);};
    auto alloc=[&](void** p,size_t n){cudaError_t e=cudaMalloc(p,n);if(e==cudaSuccess){owned.push_back(*p);e=cudaMemset(*p,0,n);}return e;};
    #define AP_CHECK(expr) do{cudaError_t e=(expr);if(e!=cudaSuccess){cleanup();return -(int)e;}}while(0)
    const int top=g_multigridLevels,n=g_N,stride=3*MG_COARSEST;
    MultigridLevel L[MG_MAX_LEVELS+1];for(int l=1;l<=top;l++){
        L[l]=g_multigrid[l];
        AP_CHECK(alloc((void**)&L[l].r,(size_t)L[l].capacity*sizeof(float3)));
        AP_CHECK(alloc((void**)&L[l].post,(size_t)L[l].capacity*sizeof(float3)));
    }
    float3 *r=nullptr,*z=nullptr;MaterialCGSum* sums=nullptr;
    AP_CHECK(alloc((void**)&r,(size_t)n*sizeof(float3)));AP_CHECK(alloc((void**)&z,(size_t)n*sizeof(float3)));
    AP_CHECK(alloc((void**)&sums,(size_t)n*sizeof(MaterialCGSum)));
    std::vector<unsigned short> mask(n);std::vector<Symmetric3> diagonal(n);
    AP_CHECK(cudaMemcpy(mask.data(),d_cgStencil,n*sizeof(unsigned short),cudaMemcpyDeviceToHost));
    AP_CHECK(cudaMemcpy(diagonal.data(),d_cgH,n*sizeof(Symmetric3),cudaMemcpyDeviceToHost));
    // Never serialize unspecified storage at inactive/fixed nodes.
    for(int i=0;i<n;i++)if(!(mask[i]>>7&1))diagonal[i]={};
    std::vector<int> counts(top+1);std::vector<std::vector<int>> parents(top+1);
    std::vector<std::vector<float>> inverse(top+1);
    int children=n;
    for(int l=1;l<=top;l++){
        AP_CHECK(cudaMemcpy(&counts[l],L[l].count,sizeof(int),cudaMemcpyDeviceToHost));
        counts[l]=std::min(counts[l],L[l].capacity);parents[l].resize(children);inverse[l].resize((size_t)9*counts[l]);
        AP_CHECK(cudaMemcpy(parents[l].data(),L[l].parent,children*sizeof(int),cudaMemcpyDeviceToHost));
        AP_CHECK(cudaMemcpy(inverse[l].data(),L[l].inverse,inverse[l].size()*sizeof(float),cudaMemcpyDeviceToHost));
        if(l==1)for(int i=0;i<n;i++)if(!(mask[i]>>7&1))parents[l][i]=-1;
        children=counts[l];
    }
    int coarse=3*counts[top];std::vector<float> dense((size_t)stride*stride),compact((size_t)coarse*coarse);
    AP_CHECK(cudaMemcpy(dense.data(),d_multigridInverse,dense.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(int i=0;i<coarse;i++)for(int j=0;j<coarse;j++)compact[i*coarse+j]=dense[i*stride+j];
    std::vector<float3> input[2],output[2];unsigned rng=seed?seed:1;
    auto random=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return float(int(rng&65535)-32768)/32768.f;};
    for(int q=0;q<2;q++){
        input[q].resize(n);output[q].resize(n);
        for(int i=0;i<n;i++)if(mask[i]>>7&1)input[q][i]=mkf3(random(),random(),random());
        AP_CHECK(cudaMemcpy(r,input[q].data(),n*sizeof(float3),cudaMemcpyHostToDevice));
        AP_CHECK(cudaMemset(sums,0,(size_t)n*sizeof(MaterialCGSum)));
        applyAdditiveMultigrid(0,d_unified,g_activeNodes,d_cgStencil,d_cgH,r,z,sums,0,L,top,d_multigridInverse);
        AP_CHECK(cudaGetLastError());AP_CHECK(cudaDeviceSynchronize());
        AP_CHECK(cudaMemcpy(output[q].data(),z,n*sizeof(float3),cudaMemcpyDeviceToHost));
    }
    FILE* file=fopen(filename,"wb");if(!file){cleanup();return -902;}
    bool ok=true;auto write=[&](const void* p,size_t bytes){if(bytes && fwrite(p,1,bytes,file)!=bytes)ok=false;};
    int header[4]={0x414d4731,n,top,coarse};write(header,sizeof(header));
    write(mask.data(),n*sizeof(unsigned short));write(diagonal.data(),n*sizeof(Symmetric3));
    for(int l=1;l<=top;l++){
        int h[2]={counts[l],(int)parents[l].size()};write(h,sizeof(h));
        write(parents[l].data(),parents[l].size()*sizeof(int));write(inverse[l].data(),inverse[l].size()*sizeof(float));
    }
    write(compact.data(),compact.size()*sizeof(float));
    for(int q=0;q<2;q++){write(input[q].data(),n*sizeof(float3));write(output[q].data(),n*sizeof(float3));}
    if(fclose(file))ok=false;cleanup();return ok?0:-903;
    #undef AP_CHECK
}
