// Compare every Galerkin level with full/packed private shared storage. Keep
// hierarchy metadata read-only and allocate private operator/inverse outputs.
__global__ void k_MgProbeDamage(float* matrix){if(!blockIdx.x && !threadIdx.x)matrix[0]+=.125f;}
extern "C" LCS_COMBINED_EXPORT int LCS_MgGalerkinProbe(int damage,unsigned long long* stats,double* errors){
    if(!stats || !errors || !g_ready || !d_unified || g_multigridLevels<1 || (damage!=0 && damage!=1))return -10018;
    struct Work{std::vector<float*> allocations;~Work(){cudaDeviceSynchronize();for(auto p:allocations)queryPrivateFree(p);}}work;
    std::vector<MultigridLevel> levels[2];for(auto& list:levels)list.resize(g_multigridLevels+1);
    cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return -(int)e;
    for(int variant=0;variant<2;variant++)for(int l=1;l<=g_multigridLevels;l++){
        auto& L=levels[variant][l];L=g_multigrid[l];L.A=L.inverse=nullptr;
        e=queryPrivateAlloc(&L.A,(size_t)27*9*L.capacity*sizeof(float));if(e!=cudaSuccess)return -(int)e;work.allocations.push_back(L.A);
        e=queryPrivateAlloc(&L.inverse,(size_t)9*L.capacity*sizeof(float));if(e!=cudaSuccess)return -(int)e;work.allocations.push_back(L.inverse);
        e=cudaMemset(L.A,0,(size_t)27*9*L.capacity*sizeof(float));if(e!=cudaSuccess)return -(int)e;
        e=cudaMemset(L.inverse,0,(size_t)9*L.capacity*sizeof(float));if(e!=cudaSuccess)return -(int)e;
    }
    for(int l=1;l<=g_multigridLevels;l++){
        auto& a=levels[0][l];auto& b=levels[1][l];int blocks=(a.capacity+MG_GALERKIN_ROWS-1)/MG_GALERKIN_ROWS;
        if(l==1){
            k_MgGalerkinFine<false><<<blocks,8*MG_GALERKIN_ROWS>>>(d_unified,d_cgMatrix,d_cgScale,d_cgStencil,a);
#ifdef ENDO_KIDNEY_GATHER_MG
            k_MgGatherFine<<<min(1024,b.capacity),128>>>(d_unified,d_cgMatrix,d_cgScale,d_cgStencil,b);
#else
            k_MgGalerkinFine<true><<<blocks,8*MG_GALERKIN_ROWS>>>(d_unified,d_cgMatrix,d_cgScale,d_cgStencil,b);
#endif
        }else{
            k_MgGalerkin<false><<<blocks,8*MG_SLOTS*MG_GALERKIN_ROWS>>>(levels[0][l-1],a);
#ifdef ENDO_KIDNEY_GATHER_MG
            k_MgGather<<<min(1024,b.capacity),128>>>(levels[1][l-1],b);
#else
            k_MgGalerkin<true><<<blocks,8*MG_SLOTS*MG_GALERKIN_ROWS>>>(levels[1][l-1],b);
#endif
        }
    }
    if(damage)k_MgProbeDamage<<<1,1>>>(levels[1][1].A);
    for(int j=0;j<49;j++)stats[j]=0;for(int j=0;j<32;j++)errors[j]=0;stats[0]=g_multigridLevels;
    for(int l=1;l<=g_multigridLevels;l++){
        int at=1+6*(l-1),err=4*(l-1),count=0;auto& a=levels[0][l];auto& b=levels[1][l];
        e=cudaMemcpy(&count,a.count,sizeof(count),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;stats[at]=count;
        for(int kind=0;kind<2;kind++){
            size_t n=(size_t)a.capacity*(kind?9:27*9);stats[at+1+2*kind]=n;
            std::vector<float> first(n),second(n);
            e=cudaMemcpy(first.data(),kind?a.inverse:a.A,n*sizeof(float),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
            e=cudaMemcpy(second.data(),kind?b.inverse:b.A,n*sizeof(float),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
            for(size_t j=0;j<n;j++){
                stats[at+2+2*kind]+=std::memcmp(&first[j],&second[j],sizeof(float))!=0;
                if(!std::isfinite(first[j]) || !std::isfinite(second[j])){stats[at+5]++;continue;}
                errors[err+2*kind]=std::max(errors[err+2*kind],std::abs(double(first[j])-second[j]));
                errors[err+2*kind+1]=std::max(errors[err+2*kind+1],std::abs(double(first[j])));
            }
        }
    }
    return 0;
}
