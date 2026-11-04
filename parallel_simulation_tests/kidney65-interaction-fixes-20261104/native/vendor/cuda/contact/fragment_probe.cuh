#ifdef _WIN32
#define LCS_COMBINED_EXPORT __declspec(dllexport)
#else
#define LCS_COMBINED_EXPORT __attribute__((visibility("default")))
#endif
#include <type_traits>
// Diagnostic-only frozen fragment inputs; private probe allocations never change the live state.
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
struct FragmentProbeStorage{
    std::vector<void*> items;
    ~FragmentProbeStorage(){for(void* p:items)cudaFree(p);}
    template<class T> bool allocate(T*& p,size_t count){
        if(cudaMalloc(&p,count*sizeof(T))!=cudaSuccess)return false;
        items.push_back(p);return true;
    }
};
#if defined(ENDO_KIDNEY_CLOSED_RAY_PROBE) || defined(ENDO_KIDNEY_CLOSED_RAY)
#include <map>
#include <array>
static int fragmentProbeClosed(int n,int triangles,const SurfaceVertex* vertices,const int* components){
    std::vector<SurfaceVertex> v(3*triangles);std::vector<int> component(n);
    if(cudaMemcpy(v.data(),vertices,v.size()*sizeof(SurfaceVertex),cudaMemcpyDeviceToHost)!=cudaSuccess || cudaMemcpy(component.data(),components,n*sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)return 0;
    std::map<std::array<unsigned,20>,int> unique;std::vector<int> id(v.size());
    for(size_t i=0;i<v.size();i++){
        if(v[i].count<0 || v[i].count>8)return 0;
        for(int k=0;k<v[i].count;k++)if(v[i].id[k]<0 || v[i].id[k]>=n)return 0;
        std::array<unsigned,20> key{};memcpy(key.data(),v[i].id,9*sizeof(unsigned));memcpy(key.data()+9,v[i].weight,11*sizeof(unsigned));
        auto item=unique.emplace(key,(int)unique.size());id[i]=item.first->second;
    }
    struct Edge{int count=0,balance=0,root=-1;};std::map<std::pair<int,int>,Edge> edges;
    for(int t=0;t<triangles;t++){
        int root=-1;bool valid=true;
        for(int j=0;j<3;j++){
            const auto& a=v[3*t+j];int r=a.count?component[a.id[0]]:-1;
            for(int k=1;k<a.count;k++)if(component[a.id[k]]!=r)r=-1;
            if(j==0)root=r;else if(r!=root)valid=false;
        }
        if(root<0 || !valid)continue;
        for(int j=0;j<3;j++){
            int a=id[3*t+j],b=id[3*t+(j+1)%3];if(a==b)return 0;
            auto& edge=edges[{std::min(a,b),std::max(a,b)}];
            if(edge.count && edge.root!=root)return 0;
            edge.root=root;edge.count++;edge.balance+=a<b?1:-1;
        }
    }
    for(const auto& pair:edges)if(pair.second.count!=2 || pair.second.balance!=0)return 0;
    return 1;
}
extern "C" LCS_COMBINED_EXPORT int LCS_ReadClosedRayStats(unsigned long long* out){return out?(int)cudaMemcpyFromSymbol(out,g_fragmentClosedStats,4*sizeof(unsigned long long)):-1;}
#endif
extern "C" LCS_COMBINED_EXPORT int LCS_ProbeFragmentFile(const char* input,const char* output,int release){
    FILE* file=fopen(input,"rb");if(!file)return -1;
    FragmentProbeHeader h={};
    if(fread(&h,sizeof(h),1,file)!=1 || h.magic!=0x46524731 || h.n<=0 || h.triangles<=0 || h.vertexBytes!=sizeof(SurfaceVertex)){fclose(file);return -2;}
    FragmentProbeStorage pool;
    float3 *pos=nullptr,*prev=nullptr,*vel=nullptr,*delta=nullptr;
    int *comp=nullptr,*support=nullptr,*grasp=nullptr,*active=nullptr,*queries=nullptr,*count=nullptr,*degree=nullptr;
    unsigned long long *keys=nullptr,*sortedKeys=nullptr;int *order=nullptr,*sortedOrder=nullptr;unsigned char* sortTemp=nullptr;size_t sortBytes=0;
    FragmentHit* hits=nullptr;FragmentTreeNode* tree=nullptr;int base=1;while(32*base<h.triangles)base*=2;
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
    int sortItems=h.triangles;int* canonical=nullptr;
#ifdef ENDO_KIDNEY_CLOSED_RAY
    sortItems*=3;if(!pool.allocate(canonical,3*h.triangles))return -24;
#endif
    ok=pool.allocate(delta,h.n)&&pool.allocate(degree,h.n)&&pool.allocate(queries,capacity)&&pool.allocate(count,1)&&pool.allocate(records,size_t(20)*capacity)&&pool.allocate(points,3*h.triangles);
    ok=ok && pool.allocate(keys,sortItems) && pool.allocate(sortedKeys,sortItems) && pool.allocate(order,sortItems) && pool.allocate(sortedOrder,sortItems) && pool.allocate(tree,2*base) && pool.allocate(bounds,h.n) && pool.allocate(roots,h.n) && pool.allocate(rootCount,1);
#ifdef ENDO_KIDNEY_SPLIT_FRAGMENT
    ok=ok && pool.allocate(hits,capacity);
#endif
    if(!ok)return -4;
    if(release)cudaMemset(grasp,0,h.n*sizeof(int));
    cudaMemset(count,0,sizeof(int));cudaMemset(delta,0,h.n*sizeof(float3));cudaMemset(degree,0,h.n*sizeof(int));cudaMemset(records,0,size_t(20)*capacity*sizeof(float));
    k_FragmentQueries<<<Gr(capacity),64>>>(h.triangles,h.n,comp,active,vertices,support,grasp,inv,queries,count);
    int number=0;cudaMemcpy(&number,count,sizeof(int),cudaMemcpyDeviceToHost);if(number<=0)return -5;
    k_FragmentVertices<<<Gr(3*h.triangles),64>>>(3*h.triangles,vertices,comp,pos,prev,pos,points,count);
    k_InitFragmentBounds<<<Gr(h.n),64>>>(h.n,bounds,count);
    k_FragmentSurfaceBounds<<<Gr(3*h.triangles),64>>>(3*h.triangles,points,bounds,count);
    cudaMemset(rootCount,0,sizeof(int));k_ListFragmentRoots<<<Gr(h.n),64>>>(h.n,bounds,roots,rootCount,count);
    if(cub::DeviceRadixSort::SortPairs(nullptr,sortBytes,keys,sortedKeys,order,sortedOrder,sortItems)!=cudaSuccess || !pool.allocate(sortTemp,sortBytes))return -8;
#ifdef ENDO_KIDNEY_CLOSED_RAY
    if(prepareFragmentClosedCertificate(0,h.n,h.triangles,vertices,comp,count,canonical,keys,sortedKeys,order,sortedOrder,sortTemp,sortBytes)!=cudaSuccess)return -25;
#endif
    k_FragmentTreeKeys<<<(h.triangles+63)/64,64>>>(h.triangles,points,bounds,roots,rootCount,keys,order,count);
    if(cub::DeviceRadixSort::SortPairs(sortTemp,sortBytes,keys,sortedKeys,order,sortedOrder,h.triangles)!=cudaSuccess)return -9;
    k_FragmentTreeLeaves<<<(32*base+63)/64,64>>>(h.triangles,base,points,sortedOrder,tree,count);
    for(int level=base/2;level;level/=2)k_FragmentTreeParents<<<(level+63)/64,64>>>(level,level,tree,count);
#ifdef ENDO_KIDNEY_CACHE_TREE_BOUNDS
    k_FragmentTreePrepareQueries<<<(2*base+63)/64,64>>>(2*base,tree,count);
#endif
    cudaMemcpyToSymbol(g_fragmentProbeRecords,&records,sizeof(records));
    cudaEvent_t first,last;cudaEventCreate(&first);cudaEventCreate(&last);
#if defined(ENDO_KIDNEY_CLOSED_RAY_PROBE) || defined(ENDO_KIDNEY_CLOSED_RAY)
    int certificate=fragmentProbeClosed(h.n,h.triangles,vertices,comp);
#ifdef ENDO_KIDNEY_CLOSED_RAY
    int deviceCertificate=0;cudaMemcpyFromSymbol(&deviceCertificate,g_fragmentClosedCertificate,sizeof(int));if(certificate!=deviceCertificate)return -26;
#endif
    unsigned long long stats[4]={(unsigned long long)certificate,0,0,0};
    cudaMemcpyToSymbol(g_fragmentClosedCertificate,&certificate,sizeof(certificate));cudaMemcpyToSymbol(g_fragmentClosedStats,stats,sizeof(stats));
#endif
    cudaEventRecord(first);
    launchFragmentContact(0,h.triangles,queries,count,comp,pos,prev,pos,vertices,points,inv,delta,degree,bounds,roots,rootCount,tree,base,sortedOrder,nullptr,{},hits);
    cudaEventRecord(last);cudaError_t error=cudaEventSynchronize(last);float ms=0;cudaEventElapsedTime(&ms,first,last);
    cudaEventDestroy(first);cudaEventDestroy(last);
#if defined(ENDO_KIDNEY_CLOSED_RAY_PROBE) || defined(ENDO_KIDNEY_CLOSED_RAY)
    certificate=0;cudaMemcpyToSymbol(g_fragmentClosedCertificate,&certificate,sizeof(certificate));
#endif
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

extern "C" LCS_COMBINED_EXPORT int LCS_SetFragmentAngleBudget(float value){
    if(!(value>0 && value<=1))return -1;return (int)cudaMemcpyToSymbol(g_fragmentAngleBudget,&value,sizeof(value));
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
