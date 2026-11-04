// Experimental, conservative surface hierarchy. Triangle order stays unchanged.
// Oriented solid angles may use a far-field flux bound; ambiguous rounding
// always falls back to the original full exact sum, including open cut surfaces.
#ifndef ENDO_FRAGMENT_ANGLE_BUDGET
#define ENDO_FRAGMENT_ANGLE_BUDGET 0.02f
#endif
// This controls which conservative integral bounds are attempted, not the
// accepted winding error. An interval crossing an integer-rounding boundary
// still triggers the unchanged full-triangle sum below.
__device__ __constant__ float g_fragmentAngleBudget=ENDO_FRAGMENT_ANGLE_BUDGET;
struct FragmentTreeNode{
    float3 low,high,currentLow,currentHigh,areaVector;
    float area;int root;
#ifdef ENDO_FRAGMENT_DIPOLE
    // Integral of n_i (x_j-center_j), stored as three diagonal and three
    // symmetric cross entries. The flux Jacobian is symmetric.
    float3 momentDiagonal,momentCross;
#endif
};
#ifdef ENDO_FRAGMENT_DIPOLE
__device__ inline void fragmentMoment(float3 area,float3 offset,float3& diagonal,float3& cross){
    diagonal=mkf3(area.x*offset.x,area.y*offset.y,area.z*offset.z);
    cross=mkf3(area.x*offset.y+area.y*offset.x,area.x*offset.z+area.z*offset.x,area.y*offset.z+area.z*offset.y);
}
#endif
__device__ inline float3 fragmentMin3(float3 a,float3 b){return mkf3(fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z));}
__device__ inline float3 fragmentMax3(float3 a,float3 b){return mkf3(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z));}
__device__ inline int fragmentMergeRoot(int a,int b){return a==-1?b:b==-1?a:a==b?a:-2;}
__global__ void k_FragmentTreeLeaves(int triangles,int base,const FragmentVertex* points,const int* order,FragmentTreeNode* tree,const int* queryCount){
    if(!*queryCount)return;
    int lane=threadIdx.x%32,leaf=(blockIdx.x*blockDim.x+threadIdx.x)/32,index=32*leaf+lane,t=index<triangles?order[index]:-1;
    if(leaf>=base)return;
    FragmentTreeNode a={};a.root=-1;a.low=a.currentLow=mkf3(1e30f,1e30f,1e30f);a.high=a.currentHigh=mkf3(-1e30f,-1e30f,-1e30f);
#ifdef ENDO_FRAGMENT_DIPOLE
    float3 triangleCenter=mkf3(0,0,0);
#endif
    if(t>=0 && t<triangles){
        FragmentVertex p=points[3*t],q=points[3*t+1],r=points[3*t+2];
        if(p.root>=0 && p.root==q.root && p.root==r.root){
            a.root=p.root;a.currentLow=fragmentMin3(p.current,fragmentMin3(q.current,r.current));a.currentHigh=fragmentMax3(p.current,fragmentMax3(q.current,r.current));
            a.low=fragmentMin3(a.currentLow,fragmentMin3(p.previous,fragmentMin3(q.previous,r.previous)));
            a.high=fragmentMax3(a.currentHigh,fragmentMax3(p.previous,fragmentMax3(q.previous,r.previous)));
            a.areaVector=cross3(q.current-p.current,r.current-p.current)*.5f;a.area=len3(a.areaVector);
#ifdef ENDO_FRAGMENT_DIPOLE
            triangleCenter=(p.current+q.current+r.current)*(1.f/3.f);
#endif
        }
    }
#ifdef ENDO_FRAGMENT_DIPOLE
    float3 triangleArea=a.areaVector;
#endif
    for(int off=16;off;off/=2){
        a.low=fragmentMin3(a.low,mkf3(__shfl_down_sync(0xffffffff,a.low.x,off),__shfl_down_sync(0xffffffff,a.low.y,off),__shfl_down_sync(0xffffffff,a.low.z,off)));
        a.high=fragmentMax3(a.high,mkf3(__shfl_down_sync(0xffffffff,a.high.x,off),__shfl_down_sync(0xffffffff,a.high.y,off),__shfl_down_sync(0xffffffff,a.high.z,off)));
        a.currentLow=fragmentMin3(a.currentLow,mkf3(__shfl_down_sync(0xffffffff,a.currentLow.x,off),__shfl_down_sync(0xffffffff,a.currentLow.y,off),__shfl_down_sync(0xffffffff,a.currentLow.z,off)));
        a.currentHigh=fragmentMax3(a.currentHigh,mkf3(__shfl_down_sync(0xffffffff,a.currentHigh.x,off),__shfl_down_sync(0xffffffff,a.currentHigh.y,off),__shfl_down_sync(0xffffffff,a.currentHigh.z,off)));
        a.areaVector.x+=__shfl_down_sync(0xffffffff,a.areaVector.x,off);a.areaVector.y+=__shfl_down_sync(0xffffffff,a.areaVector.y,off);a.areaVector.z+=__shfl_down_sync(0xffffffff,a.areaVector.z,off);
        a.area+=__shfl_down_sync(0xffffffff,a.area,off);int other=__shfl_down_sync(0xffffffff,a.root,off);if(lane+off<32)a.root=fragmentMergeRoot(a.root,other);
    }
#ifdef ENDO_FRAGMENT_DIPOLE
    float3 center=fragmentBroadcast((a.currentLow+a.currentHigh)*.5f,0);
    float3 md,mc;fragmentMoment(triangleArea,triangleCenter-center,md,mc);
    for(int off=16;off;off/=2){
        md.x+=__shfl_down_sync(0xffffffff,md.x,off);md.y+=__shfl_down_sync(0xffffffff,md.y,off);md.z+=__shfl_down_sync(0xffffffff,md.z,off);
        mc.x+=__shfl_down_sync(0xffffffff,mc.x,off);mc.y+=__shfl_down_sync(0xffffffff,mc.y,off);mc.z+=__shfl_down_sync(0xffffffff,mc.z,off);
    }
    a.momentDiagonal=md;a.momentCross=mc;
#endif
    if(!lane)tree[base+leaf]=a;
}
__global__ void k_FragmentTreeParents(int first,int count,FragmentTreeNode* tree,const int* queryCount){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !*queryCount)return;i+=first;
    FragmentTreeNode a=tree[2*i],b=tree[2*i+1],c;
    c.low=fragmentMin3(a.low,b.low);c.high=fragmentMax3(a.high,b.high);c.currentLow=fragmentMin3(a.currentLow,b.currentLow);c.currentHigh=fragmentMax3(a.currentHigh,b.currentHigh);
    c.areaVector=a.areaVector+b.areaVector;c.area=a.area+b.area;c.root=fragmentMergeRoot(a.root,b.root);
#ifdef ENDO_FRAGMENT_DIPOLE
    float3 center=(c.currentLow+c.currentHigh)*.5f,ad,ac,bd,bc;
    fragmentMoment(a.areaVector,(a.currentLow+a.currentHigh)*.5f-center,ad,ac);
    fragmentMoment(b.areaVector,(b.currentLow+b.currentHigh)*.5f-center,bd,bc);
    c.momentDiagonal=a.momentDiagonal+b.momentDiagonal+ad+bd;
    c.momentCross=a.momentCross+b.momentCross+ac+bc;
#endif
    tree[i]=c;
}
__device__ inline bool fragmentTreeSegment(const FragmentTreeNode& n,FragmentVertex p,float skin){
    return !(fminf(p.current.x,p.previous.x)>n.high.x+skin || fmaxf(p.current.x,p.previous.x)<n.low.x-skin ||
        fminf(p.current.y,p.previous.y)>n.high.y+skin || fmaxf(p.current.y,p.previous.y)<n.low.y-skin ||
        fminf(p.current.z,p.previous.z)>n.high.z+skin || fmaxf(p.current.z,p.previous.z)<n.low.z-skin);
}
#ifdef ENDO_KIDNEY_CACHE_TREE_BOUNDS
// All parent bounds have finished before this pass. The query phase needs only
// center/radius, so reuse the two current-AABB vectors without growing nodes.
// Every later refit overwrites ALL leaves and parents before preparing again.
__global__ void k_FragmentTreePrepareQueries(int count,FragmentTreeNode* tree,const int* queries){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<1 || i>=count || !*queries)return;
    auto& n=tree[i];float3 center=(n.currentLow+n.currentHigh)*.5f;
    float radius=len3(n.currentHigh-n.currentLow)*.50001f+1e-6f;
    n.currentLow=center;n.currentHigh=mkf3(radius,0,0);
}
#endif
__device__ inline bool fragmentTreeRay(const FragmentTreeNode& n,float3 p,float3 d,float lower){
    float lo=lower,hi=1e30f;
    for(int j=0;j<3;j++){
        float v=compF3(d,j),x=compF3(p,j),a=compF3(n.low,j)-1e-6f,b=compF3(n.high,j)+1e-6f;
        if(fabsf(v)<1e-20f){if(x<a || x>b)return false;}
        else{float t0=(a-x)/v,t1=(b-x)/v;lo=fmaxf(lo,fminf(t0,t1));hi=fminf(hi,fmaxf(t0,t1));if(lo>hi)return false;}
    }
    return hi>=lo;
}
__device__ inline bool fragmentBoundedAngle(const FragmentTreeNode& n,float3 p,float& angle,float& error){
    if(n.root<0)return false;
#ifdef ENDO_KIDNEY_CACHE_TREE_BOUNDS
    float3 center=n.currentLow,r=center-p;
    float distance=len3(r),radius=n.currentHigh.x;
#else
    float3 center=(n.currentLow+n.currentHigh)*.5f,r=center-p;
    float distance=len3(r),radius=len3(n.currentHigh-n.currentLow)*.50001f+1e-6f;
#endif
    if(!(distance>radius))return false;
    // Flux field f(x)=(x-p)/|x-p|^3 has Jacobian norm <=2/d_min^3.
    // Every triangle point is within radius of center, hence the area integral
    // differs from areaVector.dot(f(center)) by at most 2*radius*area/d_min^3.
    float nearest=distance-radius;
    angle=dot3(n.areaVector,r)/(distance*distance*distance);
    error=2.0001f*radius*n.area/(nearest*nearest*nearest)+1e-5f*(fabsf(angle)+n.area/(distance*distance))+1e-6f;
#ifdef ENDO_FRAGMENT_DIPOLE
    const float3 d=n.momentDiagonal,m=n.momentCross;
    float d2=distance*distance,d3=d2*distance;
    float quadratic=d.x*r.x*r.x+d.y*r.y*r.y+d.z*r.z*r.z+m.x*r.x*r.y+m.y*r.x*r.z+m.z*r.y*r.z;
    float corrected=angle+(d.x+d.y+d.z)/d3-3.f*quadratic/(d3*d2);
    // For f(r)=r/|r|^3, ||D^2 f[u,u]|| <= 6/|r|^4 for unit u.
    // Taylor's integral remainder is therefore <= 3 radius^2 area/d_min^4.
    // Keep a coordinate-aware roundoff allowance for the float tree moments.
    float coordinate=fmaxf(fabsf(center.x),fmaxf(fabsf(center.y),fabsf(center.z)));
    float bound=3.0002f*radius*radius*n.area/(nearest*nearest*nearest*nearest)+
        2e-5f*(fabsf(corrected)+n.area/d2*(1.f+(radius+coordinate)/distance))+1e-6f;
    if(bound<error){angle=corrected;error=bound;}
#endif
    return error<g_fragmentAngleBudget;
}

__device__ inline unsigned fragmentSpreadBits(unsigned x){
    x&=1023u;x=(x|(x<<16))&0x030000FFu;x=(x|(x<<8))&0x0300F00Fu;x=(x|(x<<4))&0x030C30C3u;x=(x|(x<<2))&0x09249249u;return x;
}
__global__ void k_FragmentTreeKeys(int triangles,const FragmentVertex* points,const FragmentBounds* bounds,const int* roots,const int* rootCount,
    unsigned long long* keys,int* order,const int* queryCount){
    int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=triangles || !*queryCount)return;order[t]=t;
    FragmentVertex a=points[3*t],b=points[3*t+1],c=points[3*t+2];
    if(a.root<0 || a.root!=b.root || a.root!=c.root){keys[t]=~0ull;return;}
    // Root-local normalization avoids loss of spatial resolution for a distant component.
    FragmentBounds box=bounds[a.root];float3 center=(a.current+b.current+c.current)*(1.f/3.f),span=box.high-box.low;
    unsigned xyz[3];for(int j=0;j<3;j++)xyz[j]=(unsigned)fminf(1023.f,fmaxf(0.f,1023.f*(compF3(center,j)-compF3(box.low,j))/fmaxf(compF3(span,j),1e-6f)));
    unsigned code=fragmentSpreadBits(xyz[0])|(fragmentSpreadBits(xyz[1])<<1)|(fragmentSpreadBits(xyz[2])<<2);
    keys[t]=((unsigned long long)(unsigned)a.root<<32)|code;
}
__device__ inline bool fragmentTieBefore(int a,int b){return a>=0 && (b<0 || (a%32<b%32) || (a%32==b%32 && a<b));}
__device__ inline int fragmentBestHitLane(float value,int triangle,bool minimum){
    float best=value;int chosen=triangle;
    for(int off=16;off;off/=2){
        float other=__shfl_down_sync(0xffffffff,best,off);int hit=__shfl_down_sync(0xffffffff,chosen,off);
        if(threadIdx.x%32+off<32 && ((minimum?other<best:other>best) || (other==best && fragmentTieBefore(hit,chosen)))){best=other;chosen=hit;}
    }
    chosen=__shfl_sync(0xffffffff,chosen,0);best=__shfl_sync(0xffffffff,best,0);
    if(chosen<0)return 0;return __ffs(__ballot_sync(0xffffffff,triangle==chosen && value==best))-1;
}
