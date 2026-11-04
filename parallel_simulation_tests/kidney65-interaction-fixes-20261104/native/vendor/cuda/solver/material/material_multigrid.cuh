// Aggregation multigrid preconditioner of the material PCG (coupled_material.cuh) on the regular lattice.
//
// A level-1 aggregate is the set of nodes of one 2x2x2 block of lattice sites that belong to one component; each
// coarser level aggregates the aggregates of 2x2x2 blocks of its own sites the same way. Aggregates never mix
// components, so the preconditioner keeps the per-component PCG independent, and a cut that parts two pieces parts
// their aggregates. The coarse operators are Galerkin products of the PCG's own stencil with piecewise constant
// prolongation, assembled at every solve, so a severed link inside an aggregate needs no rebuild: the hierarchy
// follows the components alone (k_TopologyStale). One application is a V(1,1) cycle with damped block-Jacobi
// smoothing and a dense Cholesky solve of the coarsest level.

// Aggregates one site holds, one per component: a site of a cut seam holds the pieces on both sides.
constexpr int MG_SLOTS=2;
// Block-Jacobi smoothing weight.
constexpr float MG_OMEGA=.7f;
// Aggregates the dense coarsest solve takes. A coarsest level beyond it smooths once instead.
constexpr int MG_COARSEST=128;
constexpr int MG_MAX_LEVELS=8;
#ifdef ENDO_KIDNEY_PRECISE_MG
using MgReduction=double;
struct MgResidual {
    double x,y,z;
    __host__ __device__ operator float3() const{return mkf3((float)x,(float)y,(float)z);}
};
#else
using MgReduction=float;
using MgResidual=float3;
#endif

struct MultigridLevel {
    int sx,sy,sz,capacity;
    // Per slot of a site (MG_SLOTS*sites): the component that claimed it (-1: none), and its aggregate (-1: none).
    int *slots,*index,*count;
    // The aggregate of each child: of each fine node at level 1, of each aggregate of the level below otherwise.
    int *parent;
    // Per aggregate: its site, its component, and the aggregate at each of the 27 site offsets in its component (-1 at
    // an offset without coupling, mgCoupled).
    int *site,*component,*neighbor;
    // Per aggregate: its Galerkin row blocks (row-major 3x3) by site offset, written at the coupled offsets alone, and
    // the inverse of its diagonal block.
    float *A,*inverse;
    // Per aggregate: the restricted residual, the correction after pre-smoothing and prolongation, and after the post
    // smoothing, which the level above prolongs.
    MgResidual *r;float3 *z,*post;
};
__device__ inline int mgSites(const MultigridLevel& L){return L.sx*L.sy*L.sz;}
// The offset slot 0..26 of a site offset in {-1,0,1}^3.
__device__ inline int mgOffset(int dx,int dy,int dz){return (dx+1)+3*((dy+1)+3*(dz+1));}
// The site offset at the next coarser level of a child at corner p (0 or 1) moved by d (-1, 0 or 1) along one axis.
__device__ inline int mgCoarseStep(int p,int d){int v=p+d;return v<0?-1:v>1?1:0;}
// The 15 site offsets a coarse operator couples: those whose nonzero components share one sign. The Kuhn split couples
// a node to its neighbors along such directions alone (materialSlotOffset), and a 2x2x2 aggregation moves each such
// direction to such an offset, so the blocks at the other 12 offsets are zero at every level.
constexpr unsigned MG_COUPLED=0x6c3761bu;
__device__ inline bool mgCoupled(int o){return MG_COUPLED>>o&1;}
// Aggregates per block of the Galerkin kernels, which sum their rows in shared memory.
constexpr int MG_GALERKIN_ROWS=16;

__device__ inline void mgInverse3(const float (&a)[9],float* out){
    float c00=a[4]*a[8]-a[5]*a[7],c01=a[5]*a[6]-a[3]*a[8],c02=a[3]*a[7]-a[4]*a[6];
    float det=a[0]*c00+a[1]*c01+a[2]*c02;
    if(!(fabsf(det)>1e-30f)){for(int k=0;k<9;k++)out[k]=0;return;}
    float d=1.f/det;
    out[0]=c00*d;out[1]=(a[2]*a[7]-a[1]*a[8])*d;out[2]=(a[1]*a[5]-a[2]*a[4])*d;
    out[3]=c01*d;out[4]=(a[0]*a[8]-a[2]*a[6])*d;out[5]=(a[2]*a[3]-a[0]*a[5])*d;
    out[6]=c02*d;out[7]=(a[1]*a[6]-a[0]*a[7])*d;out[8]=(a[0]*a[4]-a[1]*a[3])*d;
}
__device__ inline float3 mgApply3(const float* a,float3 v){
    return mkf3(a[0]*v.x+a[1]*v.y+a[2]*v.z,a[3]*v.x+a[4]*v.y+a[5]*v.z,a[6]*v.x+a[7]*v.y+a[8]*v.z);
}

// ---- Hierarchy, rebuilt when the topology revision moves (`stale`, k_TopologyStale).
__global__ void k_MgClear(MultigridLevel L,const unsigned* stale){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(!*stale)return;
    if(k<MG_SLOTS*mgSites(L))L.slots[k]=-1;
    if(k==0)*L.count=0;
}
// Claims the slot of `component` at `site`, returning its code (MG_SLOTS*site+slot), or -1 once all are taken.
__device__ inline int mgClaim(const MultigridLevel& L,int site,int component){
    for(int k=0;k<MG_SLOTS;k++){
        int previous=atomicCAS(L.slots+MG_SLOTS*site+k,-1,component);
        if(previous==-1 || previous==component)return MG_SLOTS*site+k;
    }
    return -1;
}
// Level 1: every node the PCG can solve (movable and active) joins the aggregate of its block and component.
__global__ void k_MgClaimFine(const UnifiedState* state,MultigridLevel L,const unsigned* stale){
    const auto& s=*state;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.n || !*stale)return;
    if(!(s.invMass[i]>0) || (s.active && !s.active[i])){L.parent[i]=-1;return;}
    int x=i%s.nx,y=(i/s.nx)%s.ny,z=i/(s.nx*s.ny);
    L.parent[i]=mgClaim(L,x/2+L.sx*(y/2+L.sy*(z/2)),s.component[i]);
}
__global__ void k_MgClaim(MultigridLevel child,MultigridLevel L,const unsigned* stale){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=child.capacity || !*stale)return;
    if(c>=*child.count){L.parent[c]=-1;return;}
    int site=child.site[c],x=site%child.sx,y=(site/child.sx)%child.sy,z=site/(child.sx*child.sy);
    L.parent[c]=mgClaim(L,x/2+L.sx*(y/2+L.sy*(z/2)),child.component[c]);
}
// Numbers the claimed slots. An aggregate beyond the capacity gets none: its children go without a coarse correction.
__global__ void k_MgNumber(MultigridLevel L,const unsigned* stale){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=MG_SLOTS*mgSites(L) || !*stale)return;
    int component=L.slots[k],index=-1;
    if(component!=-1){
        index=atomicAdd(L.count,1);
        if(index<L.capacity){L.site[index]=k/MG_SLOTS;L.component[index]=component;}else index=-1;
    }
    L.index[k]=index;
}
__global__ void k_MgParent(MultigridLevel L,int children,const unsigned* stale){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=children || !*stale)return;
    int code=L.parent[c];L.parent[c]=code<0?-1:L.index[code];
}
__global__ void k_MgNeighbors(MultigridLevel L,const unsigned* stale){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=L.capacity || !*stale)return;
    if(c==0 && *L.count>L.capacity)*L.count=L.capacity;
    if(c>=min(*L.count,L.capacity))return;
    int site=L.site[c],x=site%L.sx,y=(site/L.sx)%L.sy,z=site/(L.sx*L.sy),component=L.component[c];
    for(int dz=-1;dz<=1;dz++)for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
        int found=-1,u=x+dx,v=y+dy,w=z+dz;
        if(mgCoupled(mgOffset(dx,dy,dz)) && u>=0 && u<L.sx && v>=0 && v<L.sy && w>=0 && w<L.sz){
            int other=u+L.sx*(v+L.sy*w);
            for(int k=0;k<MG_SLOTS;k++)if(L.slots[MG_SLOTS*other+k]==component)found=L.index[MG_SLOTS*other+k];
        }
        L.neighbor[27*c+mgOffset(dx,dy,dz)]=found;
    }
}

// ---- Galerkin operators, assembled at every solve from the PCG stencil (material_stencil.cuh).
// One block of the PCG stencil row of node i: the coupling to its neighbor in `slot`, row-major.
__device__ inline void mgStencilBlock(const UnifiedState& s,const __half* matrix,const float* blockScale,int i,int slot,float (&a)[9]){
    int j=i+materialSlotOffset(s,slot),block=abs(slot-7),owner=slot<7?j:i;
    float scale=blockScale[block*s.n+owner];const __half* h=matrix+9*block*s.n+owner;
    for(int u=0;u<3;u++)for(int v=0;v<3;v++)
        a[3*u+v]=__half2float(h[(slot<7?3*v+u:3*u+v)*s.n])*scale;
}
__device__ inline void mgFinish(MultigridLevel& L,int c,int o,const float (&sum)[9]){
    for(int k=0;k<9;k++)L.A[(27*c+o)*9+k]=sum[k];
    if(o==13)mgInverse3(sum,L.inverse+9*c);
}
// Clears the rows of the MG_GALERKIN_ROWS aggregates a Galerkin block sums in shared memory, before any thread adds.
__device__ inline void mgClearRows(MgReduction (*rows)[27*9]){
    for(int k=threadIdx.x;k<MG_GALERKIN_ROWS*27*9;k+=blockDim.x)rows[k/(27*9)][k%(27*9)]=0;
    __syncthreads();
}
// Writes the coupled blocks of the rows a block summed, and the inverse of each diagonal block (mgFinish).
__device__ inline void mgFinishRows(MultigridLevel& L,int count,const MgReduction (*rows)[27*9]){
    __syncthreads();
    for(int e=threadIdx.x;e<MG_GALERKIN_ROWS*27;e+=blockDim.x){
        int row=e/27,o=e%27,c=blockIdx.x*MG_GALERKIN_ROWS+row;
        if(c>=count || !mgCoupled(o))continue;
        float sum[9];for(int k=0;k<9;k++)sum[k]=rows[row][9*o+k];
        mgFinish(L,c,o,sum);
    }
}
// The level-1 operator. Each of the 8 threads of an aggregate adds the stencil row of the node at one corner of its
// block into the aggregate's row, at the site offset of each neighbor, so every stencil block is read once.
__device__ inline void mgAddRow(MgReduction* row,const float* values){
#ifdef ENDO_KIDNEY_MG_WARP_REDUCE
    // Group identical destinations before the shared atomics. Every original
    // term is retained, accumulated in double; no matrix entry is approximated.
    auto active=cooperative_groups::coalesced_threads();
    auto group=cooperative_groups::labeled_partition(active,row);
    for(int k=0;k<9;k++){
        double value=cooperative_groups::reduce(group,(double)values[k],cooperative_groups::plus<double>());
        if(group.thread_rank()==0)atomicAdd(row+k,value);
    }
#else
    for(int k=0;k<9;k++)atomicAdd(row+k,values[k]);
#endif
}
__global__ void k_MgGalerkinFine(const UnifiedState* state,const __half* matrix,const float* blockScale,
    const unsigned short* stencil,MultigridLevel L){
    __shared__ MgReduction sh_rows[MG_GALERKIN_ROWS][27*9];
    const auto& s=*state;const int count=min(*L.count,L.capacity);
    if(blockIdx.x*MG_GALERKIN_ROWS>=count)return;
    mgClearRows(sh_rows);
    int row=threadIdx.x/8,p=threadIdx.x%8,c=blockIdx.x*MG_GALERKIN_ROWS+row;
    if(c<count){
        int site=L.site[c],X=site%L.sx,Y=(site/L.sx)%L.sy,Z=site/(L.sx*L.sy);
        int px=p&1,py=(p>>1)&1,pz=p>>2,x=2*X+px,y=2*Y+py,z=2*Z+pz,i=x+s.nx*(y+s.ny*z);
        unsigned mask=x<s.nx && y<s.ny && z<s.n/(s.nx*s.ny) && L.parent[i]==c?stencil[i]:0;
        if(mask>>7&1)for(int slot=0;slot<15;slot++){
            if(!(mask>>slot&1))continue;
            int sign=slot<7?-1:1,bits=abs(slot-7);
            int o=mgOffset(mgCoarseStep(px,sign*(bits&1)),mgCoarseStep(py,sign*((bits>>1)&1)),mgCoarseStep(pz,sign*((bits>>2)&1)));
            float a[9];mgStencilBlock(s,matrix,blockScale,i,slot,a);
            mgAddRow(&sh_rows[row][9*o],a);
        }
    }
    mgFinishRows(L,count,sh_rows);
}
// A coarser operator. Each of the 8*MG_SLOTS threads of an aggregate adds the row of one child, an aggregate of the
// level below at a corner site of its block, into the aggregate's row, at the site offset of each child neighbor.
__global__ void k_MgGalerkin(MultigridLevel child,MultigridLevel L){
    __shared__ MgReduction sh_rows[MG_GALERKIN_ROWS][27*9];
    const int count=min(*L.count,L.capacity);
    if(blockIdx.x*MG_GALERKIN_ROWS>=count)return;
    mgClearRows(sh_rows);
    int row=threadIdx.x/(8*MG_SLOTS),p=threadIdx.x%(8*MG_SLOTS)/MG_SLOTS,C=blockIdx.x*MG_GALERKIN_ROWS+row;
    if(C<count){
        int site=L.site[C],X=site%L.sx,Y=(site/L.sx)%L.sy,Z=site/(L.sx*L.sy);
        int px=p&1,py=(p>>1)&1,pz=p>>2,x=2*X+px,y=2*Y+py,z=2*Z+pz;
        int c=x<child.sx && y<child.sy && z<child.sz?child.index[MG_SLOTS*(x+child.sx*(y+child.sy*z))+threadIdx.x%MG_SLOTS]:-1;
        if(c>=0 && L.parent[c]==C)for(int o=0;o<27;o++){
            if(!mgCoupled(o) || child.neighbor[27*c+o]<0)continue;
            int O=mgOffset(mgCoarseStep(px,o%3-1),mgCoarseStep(py,(o/3)%3-1),mgCoarseStep(pz,o/9-1));
            mgAddRow(&sh_rows[row][9*O],child.A+(27*c+o)*9);
        }
    }
    mgFinishRows(L,count,sh_rows);
}
// The coarsest operator as a dense matrix over its aggregates (rows of 3*MG_COARSEST), cleared beforehand.
__global__ void k_MgDense(MultigridLevel L,float* dense){
    int t=blockIdx.x*blockDim.x+threadIdx.x,c=t/27,o=t%27,n=min(*L.count,L.capacity);
    if(c>=n || n>MG_COARSEST)return;
    int other=L.neighbor[27*c+o];if(other<0)return;
    for(int u=0;u<3;u++)for(int v=0;v<3;v++)dense[(3*c+u)*3*MG_COARSEST+3*other+v]=L.A[(27*c+o)*9+3*u+v];
}
// In-place lower Cholesky factor of the dense coarsest operator, by one CUDA block. A row without coupling (an
// aggregate whose nodes are all held fixed in this solve) gets a unit pivot, which leaves its correction zero.
__global__ void k_MgCholesky(const MultigridLevel L,float* dense){
    int n=3*min(*L.count,L.capacity);if(n>3*MG_COARSEST)return;
    const int stride=3*MG_COARSEST;
    __shared__ float pivot;
    for(int k=0;k<n;k++){
        if(threadIdx.x==0){float d=dense[k*stride+k];pivot=d>1e-20f?sqrtf(d):0;dense[k*stride+k]=pivot>0?pivot:1;}
        __syncthreads();
        for(int i=k+1+threadIdx.x;i<n;i+=blockDim.x)dense[i*stride+k]=pivot>0?dense[i*stride+k]/pivot:0;
        __syncthreads();
        int m=n-k-1;
        for(int e=threadIdx.x;e<m*m;e+=blockDim.x){
            int i=k+1+e/m,j=k+1+e%m;
            if(j<=i)dense[i*stride+j]-=dense[i*stride+k]*dense[j*stride+k];
        }
        __syncthreads();
    }
}
// Column j of the inverse of the lower factor, by forward substitution: one thread per column.
__global__ void k_MgFactorInverse(const MultigridLevel L,const float* factor,float* inverse){
    int n=3*min(*L.count,L.capacity),j=blockIdx.x*blockDim.x+threadIdx.x;const int stride=3*MG_COARSEST;
    if(n>3*MG_COARSEST || j>=n)return;
    for(int i=0;i<n;i++){
        float sum=i==j?1.f:0.f;
        for(int k=j;k<i;k++)sum-=factor[i*stride+k]*inverse[k*stride+j];
        inverse[i*stride+j]=i<j?0.f:sum/factor[i*stride+i];
    }
}
// The inverse of the coarsest operator, L^-T L^-1, from the inverse factor: one thread per entry.
__global__ void k_MgCoarseInverse(const MultigridLevel L,const float* inverseFactor,float* inverse){
    int n=3*min(*L.count,L.capacity),e=blockIdx.x*blockDim.x+threadIdx.x;const int stride=3*MG_COARSEST;
    if(n>3*MG_COARSEST || e>=n*n)return;
    int i=e/n,j=e%n;float sum=0;
    for(int k=max(i,j);k<n;k++)sum+=inverseFactor[k*stride+i]*inverseFactor[k*stride+j];
    inverse[i*stride+j]=sum;
}
// The coarsest correction: the dense inverse applied to the residual, or one smoothing step beyond MG_COARSEST.
__global__ void k_MgCoarsest(MultigridLevel L,const float* inverse){
    int count=min(*L.count,L.capacity),row=blockIdx.x*blockDim.x+threadIdx.x;const int stride=3*MG_COARSEST;
    if(count>MG_COARSEST){
        if(row<count)L.post[row]=mgApply3(L.inverse+9*row,L.r[row])*MG_OMEGA;
        return;
    }
    if(row>=3*count)return;
    float sum=0;
    for(int c=0;c<count;c++){float3 r=L.r[c];const float* m=inverse+row*stride+3*c;sum+=m[0]*r.x+m[1]*r.y+m[2]*r.z;}
    float* out=(float*)(L.post+row/3);out[row%3]=sum;
}
// ---- One V(1,1) application z = M^-1 r.
// (A x)_i of the PCG stencil restricted to the CG nodes of node i's component (the stencil mask).
__device__ inline float3 mgStencilProduct(const UnifiedState& s,const __half* matrix,const float* blockScale,unsigned mask,
    int i,const float3* x){
    float3 out=mkf3(0,0,0);
    for(int slot=0;slot<15;slot++){
        if(!(mask>>slot&1))continue;
        float a[9];mgStencilBlock(s,matrix,blockScale,i,slot,a);
        out=out+mgApply3(a,x[i+materialSlotOffset(s,slot)]);
    }
    return out;
}
__global__ void k_MgFinePre(const UnifiedState* state,const unsigned short* stencil,const Symmetric3* diagonal,
    const float3* r,float3* x){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    x[i]=materialCgNode(stencil,i)?diagonal[i].solve(r[i])*MG_OMEGA:mkf3(0,0,0);
}
__global__ void k_MgFineRestrict(const UnifiedState* state,const __half* matrix,const float* blockScale,
    const unsigned short* stencil,const float3* r,const float3* x,MultigridLevel L){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    unsigned mask=stencil[i];int c=L.parent[i];if(!(mask>>7&1) || c<0)return;
    float3 t=r[i]-mgStencilProduct(s,matrix,blockScale,mask,i,x);
    atomicAdd(&L.r[c].x,t.x);atomicAdd(&L.r[c].y,t.y);atomicAdd(&L.r[c].z,t.z);
}
__device__ inline float3 mgRowProduct(const MultigridLevel& L,int c,const float3* x){
    float3 out=mkf3(0,0,0);
    #pragma unroll
    for(int o=0;o<27;o++){
        if(!mgCoupled(o))continue;
        int other=L.neighbor[27*c+o];if(other>=0)out=out+mgApply3(L.A+(27*c+o)*9,x[other]);
    }
    return out;
}
__global__ void k_MgPre(MultigridLevel L){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=min(*L.count,L.capacity))return;
    L.z[c]=mgApply3(L.inverse+9*c,L.r[c])*MG_OMEGA;
}
__global__ void k_MgRestrict(MultigridLevel L,MultigridLevel next){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=min(*L.count,L.capacity))return;
    int C=next.parent[c];if(C<0)return;
    float3 t=L.r[c]-mgRowProduct(L,c,L.z);
    atomicAdd(&next.r[C].x,t.x);atomicAdd(&next.r[C].y,t.y);atomicAdd(&next.r[C].z,t.z);
}
__global__ void k_MgProlong(MultigridLevel L,MultigridLevel next){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=min(*L.count,L.capacity))return;
    int C=next.parent[c];if(C>=0)L.z[c]=L.z[c]+next.post[C];
}
__global__ void k_MgPost(MultigridLevel L){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=min(*L.count,L.capacity))return;
    L.post[c]=L.z[c]+mgApply3(L.inverse+9*c,L.r[c]-mgRowProduct(L,c,L.z))*MG_OMEGA;
}
__global__ void k_MgFineProlong(const UnifiedState* state,const unsigned short* stencil,MultigridLevel L,float3* x){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    int c=L.parent[i];if(materialCgNode(stencil,i) && c>=0)x[i]=x[i]+L.post[c];
}
// The final post smoothing, which writes the preconditioned residual z and adds r'z into the component's slot.
__global__ void k_MgFinePost(const UnifiedState* state,const __half* matrix,const float* blockScale,
    const unsigned short* stencil,const Symmetric3* diagonal,const float3* r,const float3* x,float3* z,
    MaterialCGSum* sums,int slot){
    const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i<0)return;
    unsigned mask=stencil[i];if(!(mask>>7&1)){z[i]=mkf3(0,0,0);return;}
    float3 out=x[i]+diagonal[i].solve(r[i]-mgStencilProduct(s,matrix,blockScale,mask,i,x))*MG_OMEGA;z[i]=out;
    materialDotAdd(&sums[s.component[i]].next[slot],unifiedDot(r[i],out));
}
