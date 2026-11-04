#ifdef _WIN32
#define LCS_COMBINED_EXPORT __declspec(dllexport)
#else
#define LCS_COMBINED_EXPORT __attribute__((visibility("default")))
#endif
#include <type_traits>
// Private frozen-input probe: no live solver memory or material settings change.
struct PairedProbePool{
    std::vector<void*> allocations;
    ~PairedProbePool(){for(void* p:allocations)cudaFree(p);}
    template<class T> bool alloc(T*& p,size_t n){if(cudaMalloc(&p,n*sizeof(T))!=cudaSuccess)return false;allocations.push_back(p);return true;}
};
extern "C" LCS_COMBINED_EXPORT int LCS_ProbePairedFile(const char* input,const char* output,int deduplicate,int release){
    FILE* f=fopen(input,"rb");if(!f)return -1;
    FragmentProbeHeader h={};if(fread(&h,sizeof(h),1,f)!=1 || h.magic!=0x46524731 || h.n<=0 || h.triangles<=0 || h.vertexBytes!=sizeof(SurfaceVertex)){fclose(f);return -2;}
    PairedProbePool pool;float3 *pos=nullptr,*previous=nullptr,*velocity=nullptr,*rest=nullptr;
    int *component=nullptr,*support=nullptr,*grasp=nullptr,*active=nullptr;float* inv=nullptr;SurfaceVertex* vertices=nullptr;
    auto load=[&](auto*& p,size_t n){using T=typename std::remove_pointer<typename std::remove_reference<decltype(p)>::type>::type;std::vector<T> a(n);return fread(a.data(),sizeof(T),n,f)==n && pool.alloc(p,n) && cudaMemcpy(p,a.data(),n*sizeof(T),cudaMemcpyHostToDevice)==cudaSuccess;};
    bool ok=load(pos,h.n)&&load(previous,h.n)&&load(velocity,h.n)&&load(component,h.n)&&load(inv,h.n)&&load(support,h.n)&&load(grasp,h.n)&&load(active,h.n)&&load(vertices,3*h.triangles);
    fclose(f);if(!ok)return -3;
    int capacity=h.n+3*h.triangles,base=1;while(32*base<h.triangles)base*=2;
    int *queries=nullptr,*queryCount=nullptr,*unique=nullptr,*uniqueCount=nullptr,*heads=nullptr,*ambiguous=nullptr,*next=nullptr,*slots=nullptr,*count=nullptr,*degree=nullptr;
    int *roots=nullptr,*rootCount=nullptr,*order=nullptr,*sortedOrder=nullptr;
    unsigned long long *keys=nullptr,*sortedKeys=nullptr;unsigned char* sortTemp=nullptr;size_t bytes=0;
    FragmentHit* hits=nullptr;FragmentVertex* points=nullptr;FragmentBounds* bounds=nullptr;FragmentTreeNode* tree=nullptr;FragmentConstraint* constraints=nullptr;
    ok=pool.alloc(queries,capacity)&&pool.alloc(queryCount,1)&&pool.alloc(unique,capacity)&&pool.alloc(uniqueCount,1)&&pool.alloc(heads,h.n)&&pool.alloc(ambiguous,h.n)&&pool.alloc(next,capacity)&&pool.alloc(slots,capacity)&&pool.alloc(count,2)&&pool.alloc(degree,h.n)&&pool.alloc(constraints,capacity);
    ok=ok&&pool.alloc(points,3*h.triangles)&&pool.alloc(bounds,h.n)&&pool.alloc(roots,h.n)&&pool.alloc(rootCount,1)&&pool.alloc(tree,2*base)&&pool.alloc(keys,h.triangles)&&pool.alloc(sortedKeys,h.triangles)&&pool.alloc(order,h.triangles)&&pool.alloc(sortedOrder,h.triangles)&&pool.alloc(rest,6*h.n);
#ifdef ENDO_KIDNEY_SPLIT_FRAGMENT
    ok=ok && pool.alloc(hits,capacity);
#endif
    if(!ok)return -4;
    // The frozen kidney fixtures use 1mm cells. Node queries are never deduplicated.
    std::vector<float3> cell(6*h.n);for(int i=0;i<h.n;i++)for(int a=0;a<3;a++){float3 v={};if(a==0)v.x=.1f;if(a==1)v.y=.1f;if(a==2)v.z=.1f;cell[6*i+2*a]=v*(-1.f);cell[6*i+2*a+1]=v;}
    cudaMemcpy(rest,cell.data(),cell.size()*sizeof(float3),cudaMemcpyHostToDevice);
    if(release)cudaMemset(grasp,0,h.n*sizeof(int));
    cudaMemset(queryCount,0,sizeof(int));cudaMemset(count,0,2*sizeof(int));cudaMemset(degree,0,h.n*sizeof(int));cudaMemset(slots,0xff,capacity*sizeof(int));
    k_FragmentQueries<<<Gr(capacity),64>>>(h.triangles,h.n,component,active,vertices,support,grasp,inv,queries,queryCount,2);
    int raw=0,kept=0;cudaError_t error=cudaMemcpy(&raw,queryCount,sizeof(int),cudaMemcpyDeviceToHost);float ms=0;
    if(error!=cudaSuccess)return -(100+(int)error);
    if(raw>0){
        k_FragmentVertices<<<Gr(3*h.triangles),64>>>(3*h.triangles,vertices,component,pos,previous,pos,points,queryCount);
        k_InitFragmentBounds<<<Gr(h.n),64>>>(h.n,bounds,queryCount);
        k_FragmentSurfaceBounds<<<Gr(3*h.triangles),64>>>(3*h.triangles,points,bounds,queryCount);
        cudaMemset(rootCount,0,sizeof(int));k_ListFragmentRoots<<<Gr(h.n),64>>>(h.n,bounds,roots,rootCount,queryCount);
        k_FragmentTreeKeys<<<Gr(h.triangles),64>>>(h.triangles,points,bounds,roots,rootCount,keys,order,queryCount);
        if(cub::DeviceRadixSort::SortPairs(nullptr,bytes,keys,sortedKeys,order,sortedOrder,h.triangles)!=cudaSuccess || !pool.alloc(sortTemp,bytes))return -5;
        if(cub::DeviceRadixSort::SortPairs(sortTemp,bytes,keys,sortedKeys,order,sortedOrder,h.triangles)!=cudaSuccess)return -6;
        k_FragmentTreeLeaves<<<Gr(32*base),64>>>(h.triangles,base,points,sortedOrder,tree,queryCount);
        for(int level=base/2;level;level/=2)k_FragmentTreeParents<<<Gr(level),64>>>(level,level,tree,queryCount);
#ifdef ENDO_KIDNEY_CACHE_TREE_BOUNDS
        k_FragmentTreePrepareQueries<<<Gr(2*base),64>>>(2*base,tree,queryCount);
#endif
        cudaEvent_t first,last;cudaEventCreate(&first);cudaEventCreate(&last);cudaEventRecord(first);
        if(deduplicate){
            cudaMemset(heads,0xff,h.n*sizeof(int));cudaMemset(ambiguous,0,h.n*sizeof(int));cudaMemset(uniqueCount,0,sizeof(int));
            k_FragmentDuplicateLinks<<<128,64>>>(h.n,h.triangles,queries,queryCount,vertices,heads,ambiguous,next);
            k_FragmentUniqueQueries<<<128,64>>>(h.n,h.triangles,queries,queryCount,vertices,heads,ambiguous,next,unique,uniqueCount);
        }
        FragmentManifold manifold{constraints,count,degree,rest,slots,deduplicate?next:nullptr};
        launchFragmentContact(0,h.triangles,deduplicate?unique:queries,deduplicate?uniqueCount:queryCount,component,pos,previous,pos,vertices,points,inv,nullptr,nullptr,bounds,roots,rootCount,tree,base,sortedOrder,nullptr,manifold,hits);
        cudaEventRecord(last);error=cudaEventSynchronize(last);cudaEventElapsedTime(&ms,first,last);cudaEventDestroy(first);cudaEventDestroy(last);
        if(error!=cudaSuccess)return -(100+(int)error);
        cudaMemcpy(&kept,deduplicate?uniqueCount:queryCount,sizeof(int),cudaMemcpyDeviceToHost);
    }
    int contacts=0;cudaMemcpy(&contacts,count,sizeof(int),cudaMemcpyDeviceToHost);if(contacts<0 || contacts>capacity)return -7;
    std::vector<int> slotData(capacity);std::vector<FragmentConstraint> data(contacts);
    cudaMemcpy(slotData.data(),slots,capacity*sizeof(int),cudaMemcpyDeviceToHost);if(contacts)cudaMemcpy(data.data(),constraints,contacts*sizeof(FragmentConstraint),cudaMemcpyDeviceToHost);
    f=fopen(output,"wb");if(!f)return -8;int header[8]={0x46524d31,h.n,h.triangles,capacity,contacts,(int)sizeof(FragmentConstraint),raw,kept};
    ok=fwrite(header,sizeof(header),1,f)==1 && fwrite(&ms,sizeof(float),1,f)==1 && fwrite(slotData.data(),sizeof(int),capacity,f)==capacity && fwrite(data.data(),sizeof(FragmentConstraint),contacts,f)==contacts;
    if(fclose(f))ok=false;return ok?0:-9;
}
