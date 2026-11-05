// Complete original/filtered V-cycles on PRIVATE residual/correction/sum
// workspaces. Compare all free-component outputs the next PCG direction reads.
// Live positions, matrices, topology, residual and CG workspaces are read-only.
__global__ void k_FreeMgProbeDamage(float3* out,int i){if(!blockIdx.x && !threadIdx.x && i>=0)out[i].x=__uint_as_float(__float_as_uint(out[i].x)^1u);}
extern "C" LCS_COMBINED_EXPORT int LCS_FreeMultigridProbe(int damage,unsigned long long* stats,double* errors){
    // mode 2 compares the ORIGINAL algorithm with itself: a mandatory control
    // for the floating-point order of atomic dot reductions, not a fast path.
    if(!stats || !errors || !g_ready || !d_unified || g_multigridLevels<1 || damage<0 || damage>2)return -10019;
    struct Work{std::vector<void*> allocations;~Work(){cudaDeviceSynchronize();for(auto p:allocations)queryPrivateFree(p);}}work;
    auto allocate=[&](auto** p,size_t n){auto e=queryPrivateAlloc(p,n*sizeof(**p));if(e==cudaSuccess)work.allocations.push_back(*p);return e;};
    cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return -(int)e;
    std::vector<int> active(g_activeNodes),component(g_N),flags(g_N);std::vector<unsigned short> mask(g_N);
    e=cudaMemcpy(active.data(),d_activeNodes,active.size()*sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(component.data(),d_unifiedComponents,g_N*sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(flags.data(),d_graspComponents,g_N*sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(mask.data(),d_cgStencil,g_N*sizeof(unsigned short),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    int firstFree=-1;for(int i:active)if((mask[i]>>7&1) && !(flags[component[i]]&7)){firstFree=i;break;}
    std::vector<MultigridLevel> levels[2];float3 *fine[2]={},*output[2]={};MaterialCGSum* sums[2]={};
    for(int v=0;v<2;v++){
        levels[v].resize(g_multigridLevels+1);
        e=allocate(&fine[v],g_N);if(e!=cudaSuccess)return -(int)e;
        e=allocate(&output[v],g_N);if(e!=cudaSuccess)return -(int)e;
        e=allocate(&sums[v],g_N);if(e!=cudaSuccess)return -(int)e;
        e=cudaMemset(output[v],0,g_N*sizeof(float3));if(e!=cudaSuccess)return -(int)e;
        e=cudaMemset(sums[v],0,g_N*sizeof(MaterialCGSum));if(e!=cudaSuccess)return -(int)e;
        for(int l=1;l<=g_multigridLevels;l++){
            auto& L=levels[v][l];L=g_multigrid[l];L.r=nullptr;L.z=L.post=nullptr;
            e=allocate(&L.r,L.capacity);if(e!=cudaSuccess)return -(int)e;
            e=allocate(&L.z,L.capacity);if(e!=cudaSuccess)return -(int)e;
            e=allocate(&L.post,L.capacity);if(e!=cudaSuccess)return -(int)e;
        }
    }
    applyMultigridWorkspace<false>(0,d_cgR,output[0],sums[0],2,levels[0].data(),g_multigridLevels,fine[0]);
    if(damage==2)applyMultigridWorkspace<false>(0,d_cgR,output[1],sums[1],2,levels[1].data(),g_multigridLevels,fine[1]);
    else applyMultigridWorkspace<true>(0,d_cgR,output[1],sums[1],2,levels[1].data(),g_multigridLevels,fine[1]);
    if(damage==1)k_FreeMgProbeDamage<<<1,1>>>(output[1],firstFree);
    for(int k=0;k<16;k++)stats[k]=0;for(int k=0;k<8;k++)errors[k]=0;
    stats[7]=damage==1 && firstFree>=0?firstFree+1:0;
    std::vector<float3> out[2];std::vector<MaterialCGSum> sum[2];
    for(int v=0;v<2;v++){
        out[v].resize(g_N);sum[v].resize(g_N);
        e=cudaMemcpy(out[v].data(),output[v],g_N*sizeof(float3),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
        e=cudaMemcpy(sum[v].data(),sums[v],g_N*sizeof(MaterialCGSum),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    }
    std::vector<float3> residual(g_N);e=cudaMemcpy(residual.data(),d_cgR,g_N*sizeof(float3),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    std::vector<unsigned char> seen(g_N);std::vector<int> terms(g_N);std::vector<double> magnitude(g_N);
    for(int i:active)if((mask[i]>>7&1) && !(flags[component[i]]&7)){
        int root=component[i];terms[root]++;
        const float* r=(const float*)&residual[i];const float* z=(const float*)&out[0][i];
        for(int k=0;k<3;k++)magnitude[root]+=std::abs(double(r[k])*z[k]);
    }
    for(int i:active){
        if(!(mask[i]>>7&1))continue;int root=component[i];bool free=!(flags[root]&7);
        stats[free?0:1]++;
        if(std::memcmp(&out[0][i],&out[1][i],sizeof(float3)))stats[free?2:5]++;
        if(!free)continue;
        const float* a=(const float*)&out[0][i];const float* b=(const float*)&out[1][i];
        for(int k=0;k<3;k++){
            if(!std::isfinite(a[k]) || !std::isfinite(b[k])){stats[4]++;continue;}
            errors[0]=std::max(errors[0],std::abs(double(a[k])-b[k]));errors[1]=std::max(errors[1],std::abs(double(a[k])));
        }
        if(!seen[root]){
            seen[root]=1;stats[6]++;double a=sum[0][root].next[2],b=sum[1][root].next[2];
            stats[3]+=std::memcmp(&a,&b,sizeof(double))!=0;
            if(!std::isfinite(a) || !std::isfinite(b)){stats[4]++;continue;}
            errors[2]=std::max(errors[2],std::abs(a-b));errors[3]=std::max(errors[3],std::abs(a));
            // Each original double dot+reduction has at most 6*n rounding
            // operations along any accumulation path. Both algorithms read
            // bit-identical float vectors (checked above). 2*gamma(6*n)*sum
            // |r_k*z_k| bounds their DIFFERENCE, with u=2^-53. This numerical
            // diagnostic is not a replacement for physical trajectory gates.
            double nu=6.*terms[root]*std::ldexp(1.,-53),bound=2*nu/(1-nu)*magnitude[root];
            bool bounded=std::isfinite(bound) && std::abs(a-b)<=bound;stats[bounded?8:9]++;
            errors[4]=std::max(errors[4],bound);errors[5]=std::max(errors[5],bound>0?std::abs(a-b)/bound:0.);
        }
    }
    return 0;
}
