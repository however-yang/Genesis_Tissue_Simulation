// Exact combinatorial certificate of the CURRENT embedded surface. A hash only
// groups candidates; full binding comparison rejects any hash collision.
// A certified edge has exactly two opposite incidents in the same component.
// This derived flag is rebuilt before each initial discovery/contact pass;
// refreshes retain immutable bindings/components, while refitting all positions.
__global__ void k_FragmentCertificateBegin(){g_fragmentClosedCertificate=1;
#ifdef ENDO_KIDNEY_CLOSED_STATS
    atomicAdd(g_fragmentRuntimeStats,1ull);
#endif
}
#ifdef ENDO_KIDNEY_CLOSED_STATS
__global__ void k_FragmentCertificateStats(){if(g_fragmentClosedCertificate)atomicAdd(g_fragmentRuntimeStats+1,1ull);}
#endif
__device__ inline bool fragmentBindingEqual(const SurfaceVertex& a,const SurfaceVertex& b){
    const unsigned* x=(const unsigned*)&a;const unsigned* y=(const unsigned*)&b;
    for(int k=0;k<9;k++)if(x[k]!=y[k])return false;
    for(int k=12;k<23;k++)if(x[k]!=y[k])return false;
    return true;
}
__global__ void k_FragmentBindingKeys(int n,int count,const SurfaceVertex* vertices,
    unsigned long long* keys,int* order,const int* queries){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !*queries)return;
    const auto& v=vertices[i];if(v.count<0 || v.count>8)atomicExch(&g_fragmentClosedCertificate,0);
    for(int k=0;k<v.count && k<8;k++)if(v.id[k]<0 || v.id[k]>=n)atomicExch(&g_fragmentClosedCertificate,0);
    const unsigned* words=(const unsigned*)&v;unsigned long long hash=1469598103934665603ull;
    for(int k=0;k<9;k++)hash=(hash^words[k])*1099511628211ull;
    for(int k=12;k<23;k++)hash=(hash^words[k])*1099511628211ull;
    keys[i]=hash;order[i]=i;
}
__global__ void k_FragmentBindingCanonical(int count,const SurfaceVertex* vertices,const unsigned long long* keys,
    const int* order,int* canonical,const int* queries){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !*queries)return;
    unsigned long long key=keys[i];int low=0,high=i;
    while(low<high){int middle=(low+high)/2;if(keys[middle]<key)low=middle+1;else high=middle;}
    int original=order[i];canonical[original]=low;
    if(!fragmentBindingEqual(vertices[original],vertices[order[low]]))atomicExch(&g_fragmentClosedCertificate,0);
}
__global__ void k_FragmentCertificateEdges(int triangles,const SurfaceVertex* vertices,const int* component,
    const int* canonical,unsigned long long* keys,int* order,const int* queries){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=3*triangles || !*queries)return;
    int face=i/3,j=i%3;order[i]=i;keys[i]=~0ull;
    int root=fragmentRoot(vertices[3*face],component);
    if(root<0 || fragmentRoot(vertices[3*face+1],component)!=root || fragmentRoot(vertices[3*face+2],component)!=root)return;
    int a=canonical[i],b=canonical[3*face+(j+1)%3];
    if(a==b)atomicExch(&g_fragmentClosedCertificate,0);
    keys[i]=((unsigned long long)(unsigned)min(a,b)<<32)|(unsigned)max(a,b);
}
__global__ void k_FragmentCertificateCheck(int count,const SurfaceVertex* vertices,const int* component,
    const int* canonical,const unsigned long long* keys,const int* order,const int* queries){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !*queries)return;
    unsigned long long key=keys[i];if(key==~0ull || (i && keys[i-1]==key))return;
    if(i+1>=count || keys[i+1]!=key || (i+2<count && keys[i+2]==key)){atomicExch(&g_fragmentClosedCertificate,0);return;}
    int a=order[i],b=order[i+1];
    int signA=canonical[a]<canonical[3*(a/3)+(a%3+1)%3]?1:-1;
    int signB=canonical[b]<canonical[3*(b/3)+(b%3+1)%3]?1:-1;
    if(signA==signB || fragmentRoot(vertices[a],component)!=fragmentRoot(vertices[b],component))atomicExch(&g_fragmentClosedCertificate,0);
}
static cudaError_t prepareFragmentClosedCertificate(cudaStream_t stream,int n,int triangles,const SurfaceVertex* vertices,
    const int* component,const int* queryCount,int* canonical,unsigned long long* keys,unsigned long long* sortedKeys,
    int* order,int* sortedOrder,void* temporary,size_t capacity){
    int count=3*triangles;auto blocks=(count+127)/128;
    k_FragmentCertificateBegin<<<1,1,0,stream>>>();
    k_FragmentBindingKeys<<<blocks,128,0,stream>>>(n,count,vertices,keys,order,queryCount);
    size_t bytes=capacity;
    cudaError_t e=cub::DeviceRadixSort::SortPairs(temporary,bytes,keys,sortedKeys,order,sortedOrder,count,0,64,stream);if(e!=cudaSuccess)return e;
    k_FragmentBindingCanonical<<<blocks,128,0,stream>>>(count,vertices,sortedKeys,sortedOrder,canonical,queryCount);
    k_FragmentCertificateEdges<<<blocks,128,0,stream>>>(triangles,vertices,component,canonical,keys,order,queryCount);
    bytes=capacity;e=cub::DeviceRadixSort::SortPairs(temporary,bytes,keys,sortedKeys,order,sortedOrder,count,0,64,stream);if(e!=cudaSuccess)return e;
    k_FragmentCertificateCheck<<<blocks,128,0,stream>>>(count,vertices,component,canonical,sortedKeys,sortedOrder,queryCount);
#ifdef ENDO_KIDNEY_CLOSED_STATS
    k_FragmentCertificateStats<<<1,1,0,stream>>>();
#endif
    return cudaGetLastError();
}
