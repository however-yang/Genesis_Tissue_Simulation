// Optional read-only experiment API: both kernels read the same live physical
// inputs and write private copies of coarse blocks. No live state is updated.
// Reports: compared blocks, exact gradients, nonzero reference gradients,
// largest absolute difference, largest relative difference, nonfinite entries,
// changed bytes outside gradient, components with a nonzero search direction.
__global__ void k_PressureProbeRoots(const UnifiedState* state,int* roots){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<s.n)roots[i]=coarseRoot(s,i);
}
extern "C" LCS_COMBINED_EXPORT int LCS_PressureLineProbe(int which,int pass,double* out){
    if(!out || which<0 || which>1 || pass<0 || pass>2 || !g_ready || !d_unified)return -10015;
    CoarseBlock* source=which?d_coarseBlock:d_globalCoarse;if(!source)return -10016;
    struct Work{CoarseBlock *reference=nullptr,*candidate=nullptr;int* roots=nullptr;~Work(){cudaFree(reference);cudaFree(candidate);cudaFree(roots);}}w;
    size_t bytes=(size_t)g_N*sizeof(CoarseBlock);cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return -(int)e;
    e=cudaMalloc(&w.reference,bytes);if(e!=cudaSuccess)return -(int)e;
    e=cudaMalloc(&w.candidate,bytes);if(e!=cudaSuccess)return -(int)e;
    e=cudaMalloc(&w.roots,(size_t)g_N*sizeof(int));if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(w.reference,source,bytes,cudaMemcpyDeviceToDevice);if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(w.candidate,source,bytes,cudaMemcpyDeviceToDevice);if(e!=cudaSuccess)return -(int)e;
    k_PressureLine<<<Gr(g_activeNodes+g_T),64>>>(d_unified,w.reference,pass);
    launchPressureLine(0,Gr(g_activeNodes+g_T),d_unified,w.candidate,pass);
    k_PressureProbeRoots<<<Gr(g_N),64>>>(d_unified,w.roots);
    std::vector<CoarseBlock> a(g_N),b(g_N);
    std::vector<int> roots(g_N);
    e=cudaMemcpy(a.data(),w.reference,bytes,cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(b.data(),w.candidate,bytes,cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    e=cudaMemcpy(roots.data(),w.roots,(size_t)g_N*sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return -(int)e;
    for(int j=0;j<8;j++)out[j]=0;
    for(int i=0;i<g_N;i++){
        if(!roots[i]){out[6]+=std::memcmp(&a[i],&b[i],sizeof(CoarseBlock))!=0;continue;}
        out[0]++;bool exact=true,nonzero=false;
        double x[3]={a[i].gradient.x,a[i].gradient.y,a[i].gradient.z};
        double y[3]={b[i].gradient.x,b[i].gradient.y,b[i].gradient.z};
        for(int j=0;j<3;j++){
            exact&=x[j]==y[j];nonzero|=x[j]!=0;
            if(!std::isfinite(x[j]) || !std::isfinite(y[j])){out[5]++;continue;}
            double error=std::abs(x[j]-y[j]);out[3]=std::max(out[3],error);
            out[4]=std::max(out[4],error/std::max(1e-20,std::max(std::abs(x[j]),std::abs(y[j]))));
        }
        out[1]+=exact;out[2]+=nonzero;
        out[7]+=a[i].scale!=0 && (a[i].step.x!=0 || a[i].step.y!=0 || a[i].step.z!=0);
        a[i].gradient=mkf3(0,0,0);b[i].gradient=mkf3(0,0,0);
        out[6]+=std::memcmp(&a[i],&b[i],sizeof(CoarseBlock))!=0;
    }
    return 0;
}
