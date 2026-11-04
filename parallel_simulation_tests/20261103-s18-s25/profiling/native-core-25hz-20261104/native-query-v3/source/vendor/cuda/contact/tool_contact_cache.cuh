// Per-call contact laws evaluated over all SMs. The final 32-lane solve retains
// the original k=lane, lane+32,... summation and four-DOF Cholesky/guide solve.
// Scratch is derived, overwritten before every use, and excluded from snapshots.
constexpr int TOOL_CONTACT_CAPACITY=65536;
struct ToolContactTerm {float3 force;Symmetric3 H;};
struct ToolContactWorkspace {ToolContactTerm* terms=nullptr;CoupledTool *before=nullptr,*reference=nullptr;};
static ToolContactWorkspace g_toolContactWorkspace[2];
static unsigned long long* d_toolContactStats=nullptr;
static cudaStream_t g_toolContactStream=nullptr;
static cudaError_t ensureToolContactCache(){
    if(!(g_coreOptimizations&2))return cudaSuccess;
    for(auto& w:g_toolContactWorkspace)if(!w.terms){cudaError_t e=queryPrivateAlloc(&w.terms,TOOL_CONTACT_CAPACITY*sizeof(ToolContactTerm));if(e!=cudaSuccess)return e;}
    return cudaSuccess;
}
static cudaError_t ensureToolContactValidation(){
    if(!(g_coreOptimizations&2))return cudaSuccess;
    if(!d_toolContactStats){cudaError_t e=queryPrivateAlloc(&d_toolContactStats,5*sizeof(unsigned long long));if(e!=cudaSuccess)return e;}
    for(auto& w:g_toolContactWorkspace){
        if(!w.before){cudaError_t e=queryPrivateAlloc(&w.before,sizeof(CoupledTool));if(e!=cudaSuccess)return e;}
        if(!w.reference){cudaError_t e=queryPrivateAlloc(&w.reference,sizeof(CoupledTool));if(e!=cudaSuccess)return e;}
    }
    return cudaMemset(d_toolContactStats,0,5*sizeof(unsigned long long));
}
static void freeToolContactCache(){
    cudaDeviceSynchronize();for(auto& w:g_toolContactWorkspace){queryPrivateFree(w.terms);queryPrivateFree(w.before);queryPrivateFree(w.reference);w={};}
    queryPrivateFree(d_toolContactStats);d_toolContactStats=nullptr;g_toolContactStream=nullptr;
}
__global__ void k_EvaluateToolContacts(const UnifiedState* state,ToolContactTerm* terms){
    const UnifiedState& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<min(*s.contactCount,TOOL_CONTACT_CAPACITY);k+=gridDim.x*blockDim.x){
        const SurfaceContact& c=s.contacts[s.contactIds[k]];float3 force;Symmetric3 H={};unifiedContact(s,c,force,H);
        terms[k].force=force;terms[k].H=H;
    }
}
__global__ void k_CompareToolContactTerms(const UnifiedState* state,const ToolContactTerm* terms,unsigned long long* stats){
    const UnifiedState& s=*state;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.contactCount;k+=gridDim.x*blockDim.x){
        if(k>=TOOL_CONTACT_CAPACITY){atomicAdd(stats+4,1ull);continue;}
        const SurfaceContact& c=s.contacts[s.contactIds[k]];float3 force;Symmetric3 H={};unifiedContact(s,c,force,H);
        ToolContactTerm ref={force,H};const unsigned *a=reinterpret_cast<const unsigned*>(&ref),*b=reinterpret_cast<const unsigned*>(terms+k);bool diff=false;
        for(int j=0;j<sizeof(ToolContactTerm)/sizeof(unsigned);j++)diff|=a[j]!=b[j];
        atomicAdd(stats+1,1ull);if(diff)atomicAdd(stats+2,1ull);
    }
}
__global__ void k_CopyCoupledTool(UnifiedState* state,CoupledTool* copy,bool restore){
    if(threadIdx.x||blockIdx.x)return;if(restore)*state->tool=*copy;else *copy=*state->tool;
}
__global__ void k_CompareCoupledTool(const UnifiedState* state,const CoupledTool* reference,unsigned long long* stats){
    if(threadIdx.x||blockIdx.x)return;static_assert(sizeof(CoupledTool)%sizeof(unsigned)==0,"word comparison");
    const unsigned* a=reinterpret_cast<const unsigned*>(state->tool),*b=reinterpret_cast<const unsigned*>(reference);bool diff=false;
    for(int j=0;j<sizeof(CoupledTool)/sizeof(unsigned);j++)diff|=a[j]!=b[j];
    atomicAdd(stats,1ull);if(diff)atomicAdd(stats+3,1ull);
}
__global__ void k_UnifiedToolCached(UnifiedState* state,const ToolContactTerm* terms){
    UnifiedState& s=*state;CoupledTool& b=*s.tool;
    float A[4][4]={},r[4]={};float inertia=1.f/(b.inverseMass*s.h*s.h),gapInertia=b.gapInverseMass>0?1.f/(b.gapInverseMass*s.h*s.h):1e20f;
    if(threadIdx.x==0){
        for(int j=0;j<3;j++){A[j][j]=inertia;r[j]=(compF3(b.pose.jawHinge,j)-compF3(s.toolTarget,j))*inertia;}
        A[3][3]=gapInertia;r[3]=(b.pose.jawGap-s.gapTarget)*gapInertia;
    }
    for(int k=threadIdx.x;k<*s.contactCount;k+=blockDim.x){
        SurfaceContact& c=s.contacts[s.contactIds[k]];float3 force;Symmetric3 H={};
        if(k<TOOL_CONTACT_CAPACITY){force=terms[k].force;H=terms[k].H;}else unifiedContact(s,c,force,H);
        float3 hg=mkf3(H.xx*c.gapGradient.x+H.xy*c.gapGradient.y+H.xz*c.gapGradient.z,H.xy*c.gapGradient.x+H.yy*c.gapGradient.y+H.yz*c.gapGradient.z,H.xz*c.gapGradient.x+H.yz*c.gapGradient.y+H.zz*c.gapGradient.z);
        float mat[3][3]={{H.xx,H.xy,H.xz},{H.xy,H.yy,H.yz},{H.xz,H.yz,H.zz}};
        for(int i=0;i<3;i++){r[i]-=compF3(force,i);for(int j=0;j<3;j++)A[i][j]+=mat[i][j];A[i][3]+=compF3(hg,i);A[3][i]+=compF3(hg,i);}
        r[3]-=dot3(force,c.gapGradient);A[3][3]+=dot3(c.gapGradient,hg);
    }
    for(int k=threadIdx.x;k<s.grips;k+=blockDim.x)if(s.points[k].valid){
        const GripPoint& a=s.points[k];float stiffness=s.gripK/a.complianceScale;
        float3 target=gripTarget(a,b.pose,s.minGap),error=unifiedGripPoint(a,s.pos)-target;
        XpbdToolState plus=b.pose;float eps=1e-4f;plus.jawGap+=eps;
        float3 g=(gripTarget(a,plus,s.minGap)-target)*(1.f/eps);
        for(int i=0;i<3;i++){r[i]-=stiffness*compF3(error,i);A[i][i]+=stiffness;A[i][3]+=stiffness*compF3(g,i);A[3][i]+=stiffness*compF3(g,i);}
        r[3]-=stiffness*dot3(error,g);A[3][3]+=stiffness*dot3(g,g);
    }
    // One warp assembles the same four-DOF objective. A single lane walking
    // every polygon witness and grip stencil serialized every local sweep.
    for(int d=16;d;d>>=1){
        for(int i=0;i<4;i++){
            r[i]+=__shfl_down_sync(0xffffffffu,r[i],d);
            for(int j=0;j<4;j++)A[i][j]+=__shfl_down_sync(0xffffffffu,A[i][j],d);
        }
    }
    if(threadIdx.x!=0)return;
    if(b.guided){
        // Keyboard instrument is supported by the operator. Solve only motion
        // along the requested travel, plus aperture; lateral reaction is borne
        // by the guide, not converted to an unsolicited sideways displacement.
        float3 axis=b.guideAxis;float aa=0,ab=0,rr=0;
        for(int i=0;i<3;i++){rr+=compF3(axis,i)*r[i];ab+=compF3(axis,i)*A[i][3];for(int j=0;j<3;j++)aa+=compF3(axis,i)*A[i][j]*compF3(axis,j);}
        float dd=0,dg=0,den=aa*A[3][3]-ab*ab;
        if(b.guideLength>0 && den>1e-20f){dd=(-rr*A[3][3]+ab*r[3])/den;dg=(-aa*r[3]+ab*rr)/den;}
        else if(A[3][3]>0)dg=-r[3]/A[3][3];
        float current=dot3(b.pose.jawHinge-b.guideOrigin,axis);
        float accepted=fminf(b.guideLength,fmaxf(0.f,current+dd));
        b.pose=shiftTool(b.pose,b.guideOrigin+axis*accepted-b.pose.jawHinge);
        b.pose.jawGap=fminf(b.gapHi,fmaxf(b.gapLo,b.pose.jawGap+dg));return;
    }
    // SPD Cholesky solve; four physical DOFs, with no gap/translation latch.
    float L[4][4]={},y[4]={},dx[4]={};
    for(int i=0;i<4;i++)for(int j=0;j<=i;j++){float v=A[i][j];for(int k=0;k<j;k++)v-=L[i][k]*L[j][k];L[i][j]=i==j?sqrtf(fmaxf(v,1e-12f)):v/L[j][j];}
    for(int i=0;i<4;i++){float v=-r[i];for(int j=0;j<i;j++)v-=L[i][j]*y[j];y[i]=v/L[i][i];}
    for(int i=3;i>=0;i--){float v=y[i];for(int j=i+1;j<4;j++)v-=L[j][i]*dx[j];dx[i]=v/L[i][i];}
    b.pose=shiftTool(b.pose,mkf3(dx[0],dx[1],dx[2]));b.pose.jawGap=fminf(b.maxGap,fmaxf(0.f,b.pose.jawGap+dx[3]));
}
static void launchUnifiedToolCached(UnifiedState* state,cudaStream_t stream){
    int k=stream?1:0;
    if(stream){if(!g_toolContactStream)g_toolContactStream=stream;else if(g_toolContactStream!=stream)k=-1;}
    if(!(g_coreOptimizations&2) || k<0 || !g_toolContactWorkspace[k].terms){k_UnifiedTool<<<1,32,0,stream>>>(state);return;}
    auto& w=g_toolContactWorkspace[k];
    if(g_queryValidate){
        k_CopyCoupledTool<<<1,1,0,stream>>>(state,w.before,false);
        k_UnifiedTool<<<1,32,0,stream>>>(state);
        k_CopyCoupledTool<<<1,1,0,stream>>>(state,w.reference,false);
        k_CopyCoupledTool<<<1,1,0,stream>>>(state,w.before,true);
    }
    k_EvaluateToolContacts<<<128,64,0,stream>>>(state,w.terms);
    if(g_queryValidate)k_CompareToolContactTerms<<<128,64,0,stream>>>(state,w.terms,d_toolContactStats);
    k_UnifiedToolCached<<<1,32,0,stream>>>(state,w.terms);
    if(g_queryValidate)k_CompareCoupledTool<<<1,1,0,stream>>>(state,w.reference,d_toolContactStats);
}
