// Opt-in read-only S25 export. Fresh bindings in disposable scratch storage; never changes physical contact state.
// These are native contact linearization weights, with a required affine offset.
// A row is invalid when stale, unsupported, or inconsistent with the rendered point.
#ifdef _WIN32
#define LCS_LABEL_EXPORT __declspec(dllexport)
#else
#define LCS_LABEL_EXPORT __attribute__((visibility("default")))
#endif
extern "C" LCS_LABEL_EXPORT int LCS_SurfaceBindingsAbiVersion(){return 2;}
extern "C" LCS_LABEL_EXPORT int LCS_GetSurfaceBindings(int capacity,int* written,
    int* ids,float* weights,float* offsets,int* identities,int* valid){
    if(!g_bound || !written || capacity<0)return XPBD_ERR_BAD_ARG;
    const int count=3*g_surfaceCount;*written=count;
    if(capacity<count || !ids || !weights || !offsets || !identities || !valid)return XPBD_ERR_BAD_ARG;
    if(count==0)return 0;
    std::vector<SurfaceVertex> vertices(count);
    std::vector<float3> positions(g_N),expanded(count);
    std::vector<int> tags(count);
    const int voxels=g_dimsX*g_dimsY*g_dimsZ;
    std::vector<unsigned> masks(voxels,0),boundary(voxels,0);
    std::vector<int> corners(8*voxels),conn(9*4096);
    // Reconstruction may reorder or move points after the physical contact cache
    // was used. Reuse its exact binding algorithm into isolated scratch memory.
    SurfaceVertex* scratch=nullptr;
    // Bypass the persistent snapshot allocation tracker for this temporary buffer.
    // Otherwise its monotonically increasing allocation ID changes snapshot bytes.
    cudaError_t e=(::cudaMalloc)((void**)&scratch,count*sizeof(SurfaceVertex));
    if(e!=cudaSuccess)return -(int)e;
    k_BindSurface<<<Gr(count),64>>>(count,g_surfaceTags,g_surfaceExpanded,g_surfaceCorners,
        g_surfaceMasks,g_surfaceConn,d_pos,scratch,g_surfaceRest,g_surfaceMaterialFP,g_surfaceBoundaryMasks);
    e=cudaGetLastError();
    if(e==cudaSuccess)e=cudaMemcpy(vertices.data(),scratch,count*sizeof(SurfaceVertex),cudaMemcpyDeviceToHost);
    cudaError_t released=(::cudaFree)(scratch);
    if(e==cudaSuccess)e=released;
    if(e==cudaSuccess)e=cudaMemcpy(positions.data(),d_pos,g_N*sizeof(float3),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(expanded.data(),g_surfaceExpanded,count*sizeof(float3),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(tags.data(),g_surfaceTags,count*sizeof(int),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(corners.data(),g_surfaceCorners,corners.size()*sizeof(int),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess && g_surfaceMasks)e=cudaMemcpy(masks.data(),g_surfaceMasks,voxels*sizeof(unsigned),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess && g_surfaceBoundaryMasks)e=cudaMemcpy(boundary.data(),g_surfaceBoundaryMasks,voxels*sizeof(unsigned),cudaMemcpyDeviceToHost);
    else boundary=masks;
    if(e==cudaSuccess && g_surfaceConn)e=cudaMemcpy(conn.data(),g_surfaceConn,conn.size()*sizeof(int),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)return -(int)e;
    for(int i=0;i<count;i++){
        const SurfaceVertex& v=vertices[i];unsigned tag=(unsigned)tags[i];
        int index=tag&0x3fffffffu;bool cut=(tag&0x80000000u)!=0;
        bool good=v.count>0 && v.count<=8 && v.voxel==(cut?index/8:index) && v.component==(cut?index%8:-1);
        good=good && v.voxel>=0 && v.voxel<voxels;
        if(good){
            good=v.mask==(masks[v.voxel]&4095u) && (!cut || g_surfaceConn);
            int expected=0;
            for(int k=0;k<8;k++){
                if(cut && g_surfaceConn && conn[9*(boundary[v.voxel]&4095u)+1+k]!=v.component)continue;
                if(expected>=v.count || v.id[expected]!=corners[8*v.voxel+k])good=false;
                ++expected;
            }
            good=good && expected==v.count;
        }
        float total=0;float3 point=v.offset;
        for(int k=0;k<8;k++){
            const int j=8*i+k;ids[j]=-1;weights[j]=0;
            if(k<v.count){
                ids[j]=v.id[k];weights[j]=v.weight[k];
                bool slot=v.id[k]>=0 && v.id[k]<g_N && std::isfinite(v.weight[k]) && v.weight[k]>=0;
                good=good && slot;total+=v.weight[k];
                if(slot){point.x+=positions[v.id[k]].x*v.weight[k];point.y+=positions[v.id[k]].y*v.weight[k];point.z+=positions[v.id[k]].z*v.weight[k];}
            }
        }
        offsets[3*i]=v.offset.x;offsets[3*i+1]=v.offset.y;offsets[3*i+2]=v.offset.z;
        identities[4*i]=v.voxel;identities[4*i+1]=v.component;identities[4*i+2]=(int)v.mask;identities[4*i+3]=v.count;
        float3 error=make_float3(point.x-expanded[i].x,point.y-expanded[i].y,point.z-expanded[i].z);
        good=good && std::isfinite(total) && std::fabs(total-1.f)<1.e-4f
             && std::isfinite(error.x) && std::isfinite(error.y) && std::isfinite(error.z)
             && std::sqrt(error.x*error.x+error.y*error.y+error.z*error.z)<=1.e-4f; // 1 micrometre, in native cm
        valid[i]=good?1:0;
    }
    return 0;
}
