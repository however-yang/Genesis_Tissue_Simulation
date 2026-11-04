#include <type_traits>
// Diagnostic-only frozen fragment inputs; private probe allocations never change the live state.
struct FragmentProbeHeader{int magic,n,triangles,vertexBytes;};
extern "C" __declspec(dllexport) int LCS_SaveFragmentCase(const char* path){
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
struct FragmentProbeStorage{
    std::vector<void*> items;
    ~FragmentProbeStorage(){for(void* p:items)cudaFree(p);}
    template<class T> bool allocate(T*& p,size_t count){
        if(cudaMalloc(&p,count*sizeof(T))!=cudaSuccess)return false;
        items.push_back(p);return true;
    }
};
extern "C" __declspec(dllexport) int LCS_ProbeFragmentFile(const char* input,const char* output,int release){
    FILE* file=fopen(input,"rb");if(!file)return -1;
    FragmentProbeHeader h={};
    if(fread(&h,sizeof(h),1,file)!=1 || h.magic!=0x46524731 || h.n<=0 || h.triangles<=0 || h.vertexBytes!=sizeof(SurfaceVertex)){fclose(file);return -2;}
    FragmentProbeStorage pool;
    float3 *pos=nullptr,*prev=nullptr,*vel=nullptr,*delta=nullptr;
    int *comp=nullptr,*support=nullptr,*grasp=nullptr,*active=nullptr,*queries=nullptr,*count=nullptr,*degree=nullptr;
    unsigned long long *keys=nullptr,*sortedKeys=nullptr;int *order=nullptr,*sortedOrder=nullptr;unsigned char* sortTemp=nullptr;size_t sortBytes=0;
    FragmentTreeNode* tree=nullptr;int base=1;while(32*base<h.triangles)base*=2;
    FragmentBounds* bounds=nullptr;int *roots=nullptr,*rootCount=nullptr;
    float *inv=nullptr,*records=nullptr;SurfaceVertex* vertices=nullptr;FragmentVertex* points=nullptr;
    auto load=[&](auto*& dst,size_t n){
        using T=typename std::remove_pointer<typename std::remove_reference<decltype(dst)>::type>::type;
        std::vector<T> v(n);
        return fread(v.data(),sizeof(T),n,file)==n && pool.allocate(dst,n) && cudaMemcpy(dst,v.data(),n*sizeof(T),cudaMemcpyHostToDevice)==cudaSuccess;
    };
    bool ok=load(pos,h.n)&&load(prev,h.n)&&load(vel,h.n)&&load(comp,h.n)&&load(inv,h.n)&&load(support,h.n)&&load(grasp,h.n)&&load(active,h.n)&&load(vertices,3*h.triangles);
    fclose(file);if(!ok)return -3;
    const int capacity=h.n+3*h.triangles;
    ok=pool.allocate(delta,h.n)&&pool.allocate(degree,h.n)&&pool.allocate(queries,capacity)&&pool.allocate(count,1)&&pool.allocate(records,size_t(20)*capacity)&&pool.allocate(points,3*h.triangles);
    ok=ok && pool.allocate(keys,h.triangles) && pool.allocate(sortedKeys,h.triangles) && pool.allocate(order,h.triangles) && pool.allocate(sortedOrder,h.triangles) && pool.allocate(tree,2*base) && pool.allocate(bounds,h.n) && pool.allocate(roots,h.n) && pool.allocate(rootCount,1);
    if(!ok)return -4;
    if(release)cudaMemset(grasp,0,h.n*sizeof(int));
    cudaMemset(count,0,sizeof(int));cudaMemset(delta,0,h.n*sizeof(float3));cudaMemset(degree,0,h.n*sizeof(int));cudaMemset(records,0,size_t(20)*capacity*sizeof(float));
    k_FragmentQueries<<<Gr(capacity),64>>>(h.triangles,h.n,comp,active,vertices,support,grasp,inv,queries,count);
    int number=0;cudaMemcpy(&number,count,sizeof(int),cudaMemcpyDeviceToHost);if(number<=0)return -5;
    k_FragmentVertices<<<Gr(3*h.triangles),64>>>(3*h.triangles,vertices,comp,pos,prev,pos,points,count);
    k_InitFragmentBounds<<<Gr(h.n),64>>>(h.n,bounds,count);
    k_FragmentSurfaceBounds<<<Gr(3*h.triangles),64>>>(3*h.triangles,points,bounds,count);
    cudaMemset(rootCount,0,sizeof(int));k_ListFragmentRoots<<<Gr(h.n),64>>>(h.n,bounds,roots,rootCount,count);
    k_FragmentTreeKeys<<<(h.triangles+63)/64,64>>>(h.triangles,points,bounds,roots,rootCount,keys,order,count);
    if(cub::DeviceRadixSort::SortPairs(nullptr,sortBytes,keys,sortedKeys,order,sortedOrder,h.triangles)!=cudaSuccess || !pool.allocate(sortTemp,sortBytes))return -8;
    if(cub::DeviceRadixSort::SortPairs(sortTemp,sortBytes,keys,sortedKeys,order,sortedOrder,h.triangles)!=cudaSuccess)return -9;
    k_FragmentTreeLeaves<<<(32*base+63)/64,64>>>(h.triangles,base,points,sortedOrder,tree,count);
    for(int level=base/2;level;level/=2)k_FragmentTreeParents<<<(level+63)/64,64>>>(level,level,tree,count);
    cudaMemcpyToSymbol(g_fragmentProbeRecords,&records,sizeof(records));
    cudaEvent_t first,last;cudaEventCreate(&first);cudaEventCreate(&last);
    cudaEventRecord(first);
    k_FragmentContact<<<2048,64>>>(h.triangles,queries,count,comp,pos,prev,pos,vertices,points,inv,delta,degree,bounds,roots,rootCount,tree,base,sortedOrder);
    cudaEventRecord(last);cudaError_t error=cudaEventSynchronize(last);float ms=0;cudaEventElapsedTime(&ms,first,last);
    cudaEventDestroy(first);cudaEventDestroy(last);
    float* empty=nullptr;cudaMemcpyToSymbol(g_fragmentProbeRecords,&empty,sizeof(empty));
    if(error!=cudaSuccess)return -(100+(int)error);
    std::vector<int> ids(number);std::vector<float> data(size_t(20)*capacity);std::vector<float3> changes(h.n);std::vector<int> degrees(h.n);
    cudaMemcpy(ids.data(),queries,number*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(data.data(),records,data.size()*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(changes.data(),delta,h.n*sizeof(float3),cudaMemcpyDeviceToHost);cudaMemcpy(degrees.data(),degree,h.n*sizeof(int),cudaMemcpyDeviceToHost);
    file=fopen(output,"wb");if(!file)return -6;
    int sizes[5]={0x46525031,number,20,h.n,h.triangles};
    ok=fwrite(sizes,sizeof(sizes),1,file)==1 && fwrite(&ms,sizeof(float),1,file)==1;
    for(int i:ids)ok=ok && fwrite(data.data()+20*i,20*sizeof(float),1,file)==1;
    ok=ok && fwrite(changes.data(),sizeof(float3),h.n,file)==h.n && fwrite(degrees.data(),sizeof(int),h.n,file)==h.n;
    if(fclose(file))ok=false;return ok?0:-7;
}

extern "C" __declspec(dllexport) int LCS_SetFragmentAngleBudget(float value){
    if(!(value>0 && value<=1))return -1;return (int)cudaMemcpyToSymbol(g_fragmentAngleBudget,&value,sizeof(value));
}

// Read-only snapshot of the active paired manifold AFTER the accepted material solve.
extern "C" __declspec(dllexport) int LCS_SavePairedConstraints(const char* path){
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
