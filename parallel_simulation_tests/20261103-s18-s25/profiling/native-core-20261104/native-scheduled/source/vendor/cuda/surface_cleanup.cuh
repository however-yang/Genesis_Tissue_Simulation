// Cancel internal pairs using feature-point identity, never positional welding.
// Bit 30 is only a wall-material marker; bit 31 retains cut/intact identity.
static int4* d_faceTable=nullptr;
static int* d_cleanTriangles=nullptr;
static int g_faceCapacity=0,g_cleanCapacity=0;
__device__ inline bool faceKey(const int* triangle,unsigned* key){
    for(int k=0;k<3;k++)key[k]=(unsigned)triangle[k]&0xbfffffffu;
    bool odd=false;
    if(key[0]>key[1]){unsigned x=key[0];key[0]=key[1];key[1]=x;odd=!odd;}
    if(key[1]>key[2]){unsigned x=key[1];key[1]=key[2];key[2]=x;odd=!odd;}
    if(key[0]>key[1]){unsigned x=key[0];key[0]=key[1];key[1]=x;odd=!odd;}
    return odd;
}
__global__ void k_ClearFaceTable(int count,int4* table){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)table[i]=make_int4(-1,0,0x7fffffff,0x7fffffff);
}
__global__ void k_IndexFaces(int count,const int* triangles,int capacity,int4* table){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    unsigned key[3];bool odd=faceKey(triangles+3*i,key);
    unsigned hash=((key[0]*73856093u)^(key[1]*19349663u)^(key[2]*83492791u))&(capacity-1);
    for(int k=0;k<capacity;k++){
        int4* entry=table+((hash+k)&(capacity-1));int old=atomicCAS(&entry->x,-1,i);
        unsigned other[3];if(old>=0)faceKey(triangles+3*old,other);
        if(old<0 || (key[0]==other[0] && key[1]==other[1] && key[2]==other[2])){
            atomicAdd(&entry->y,odd?-1:1);atomicMin(odd?&entry->w:&entry->z,i);return;
        }
    }
}
__global__ void k_CompactFaces(int capacity,const int4* table,const int* triangles,int* clean,unsigned* counter){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=capacity)return;
    int4 entry=table[i];if(entry.x<0 || entry.y==0)return;
    int index=entry.y>0?entry.z:entry.w;unsigned out=atomicAdd(counter,3u);
    for(int k=0;k<3;k++)clean[out+k]=triangles[3*index+k];
}
static cudaError_t cleanInternalFaces(){
    unsigned indices=0;cudaError_t e=cudaMemcpy(&indices,d_triCounter,sizeof(unsigned),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess || indices==0)return e;
    if(indices>(unsigned)(3*g_desc.triCapacity))return cudaErrorInvalidValue;
    int count=indices/3,capacity=1;while(capacity<2*count)capacity*=2;
    if(capacity>g_faceCapacity){
        cudaFree(d_faceTable);d_faceTable=nullptr;g_faceCapacity=0;
        e=cudaMalloc(&d_faceTable,capacity*sizeof(int4));if(e!=cudaSuccess)return e;g_faceCapacity=capacity;
    }
    if((int)indices>g_cleanCapacity){
        cudaFree(d_cleanTriangles);d_cleanTriangles=nullptr;g_cleanCapacity=0;
        e=cudaMalloc(&d_cleanTriangles,indices*sizeof(int));if(e!=cudaSuccess)return e;g_cleanCapacity=indices;
    }
    k_ClearFaceTable<<<G(capacity),64>>>(capacity,d_faceTable);
    k_IndexFaces<<<G(count),64>>>(count,d_tri,capacity,d_faceTable);
    cudaMemsetAsync(d_triCounter,0,sizeof(unsigned));
    k_CompactFaces<<<G(capacity),64>>>(capacity,d_faceTable,d_tri,d_cleanTriangles,d_triCounter);
    return cudaMemcpyAsync(d_tri,d_cleanTriangles,indices*sizeof(int),cudaMemcpyDeviceToDevice);
}
