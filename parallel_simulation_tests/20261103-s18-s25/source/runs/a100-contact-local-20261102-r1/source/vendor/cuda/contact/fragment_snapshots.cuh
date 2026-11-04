#ifdef _WIN32
#define LCS_COMBINED_EXPORT __declspec(dllexport)
#else
#define LCS_COMBINED_EXPORT __attribute__((visibility("default")))
#endif
struct FragmentProbeHeader{int magic,n,triangles,vertexBytes;};
extern "C" LCS_COMBINED_EXPORT int LCS_SaveFragmentCase(const char* path){
    if(!path || !d_unifiedComponents || !d_fragmentSupported || !d_surfaceVertices)return -1;
    if(cudaDeviceSynchronize()!=cudaSuccess)return -2;
    FILE* file=fopen(path,"wb");if(!file)return -3;
    FragmentProbeHeader header={0x46524731,g_N,g_surfaceCount,(int)sizeof(SurfaceVertex)};
    bool ok=fwrite(&header,sizeof(header),1,file)==1;
    auto save=[&](const void* ptr,size_t bytes){
        std::vector<unsigned char> host(bytes);
        return ptr && cudaMemcpy(host.data(),ptr,bytes,cudaMemcpyDeviceToHost)==cudaSuccess && fwrite(host.data(),1,bytes,file)==bytes;
    };
    ok=ok && save(d_pos,g_N*sizeof(float3)) && save(d_prevPos,g_N*sizeof(float3)) && save(d_vel,g_N*sizeof(float3));
    ok=ok && save(d_unifiedComponents,g_N*sizeof(int)) && save(d_invMass,g_N*sizeof(float));
    ok=ok && save(d_fragmentSupported,g_N*sizeof(int)) && save(d_graspComponents,g_N*sizeof(int)) && save(g_liveActive,g_N*sizeof(int));
    ok=ok && save(d_surfaceVertices,3*g_surfaceCount*sizeof(SurfaceVertex));
    if(fclose(file))ok=false;
    return ok?0:-4;
}

// Read-only snapshot of the active paired manifold AFTER the accepted material solve.
extern "C" LCS_COMBINED_EXPORT int LCS_SavePairedConstraints(const char* path){
    if(!path || !d_fragmentConstraintCount || !d_fragmentConstraints)return -1;
    if(cudaDeviceSynchronize()!=cudaSuccess)return -2;
    int count[2]={};unsigned revision=0;
    if(cudaMemcpy(count,d_fragmentConstraintCount,sizeof(count),cudaMemcpyDeviceToHost)!=cudaSuccess)return -3;
    if(cudaMemcpy(&revision,cut_topology_revision_device(),sizeof(revision),cudaMemcpyDeviceToHost)!=cudaSuccess)return -3;
    if(count[0]<0 || count[0]>g_fragmentConstraintCapacity)return -4;
    FILE* f=fopen(path,"wb");if(!f)return -5;
    int h[8]={0x464d4331,g_N,g_surfaceCount,count[0],sizeof(FragmentConstraint),count[1],(int)revision,0};
    bool ok=fwrite(h,sizeof(h),1,f)==1;
    auto save=[&](const void* p,size_t bytes){std::vector<unsigned char> a(bytes);return p && cudaMemcpy(a.data(),p,bytes,cudaMemcpyDeviceToHost)==cudaSuccess && fwrite(a.data(),1,bytes,f)==bytes;};
    ok=ok && save(d_pos,g_N*sizeof(float3)) && save(d_prevPos,g_N*sizeof(float3)) && save(d_unifiedTarget,g_N*sizeof(float3));
    ok=ok && save(d_unifiedComponents,g_N*sizeof(int)) && save(d_fragmentSupported,g_N*sizeof(int)) && save(d_graspComponents,g_N*sizeof(int));
    ok=ok && save(d_fragmentConstraintDegree,g_N*sizeof(int)) && save(d_fragmentConstraints,count[0]*sizeof(FragmentConstraint));
    if(fclose(f))ok=false;return ok?0:-6;
}
