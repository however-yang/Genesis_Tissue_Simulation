// Diagnostic-only: read accepted rounded states without changing the solve.
struct JointStats {unsigned long long roots,accepted,zero,fallback,disabled,negativeMargin,volumeViolation,closingViolation,energyViolation;float minimumNegativeMargin;};
__device__ JointStats g_jointStats;
static float3* d_jointBefore=nullptr;
__global__ void k_JointSnapshot(const UnifiedState* state,float3* before){const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(i>=0)before[i]=s.pos[i];}
__global__ void k_JointRecordRoots(const UnifiedState* state,const JointBlock* blocks){
 int i=unifiedActiveNode(*state,blockIdx.x*blockDim.x+threadIdx.x);if(!coarseRoot(*state,i))return;const auto& b=blocks[i];
 atomicAdd(&g_jointStats.roots,1ull);if(b.disabled)atomicAdd(&g_jointStats.disabled,1ull);
 if(b.accepted){if(b.zero)atomicAdd(&g_jointStats.zero,1ull);else{atomicAdd(&g_jointStats.accepted,1ull);bool valid=false;for(int k=1;k<5;k++)valid|=jointTrialAccepted(b,k);if(!valid)atomicAdd(&g_jointStats.energyViolation,1ull);}}
 else atomicAdd(&g_jointStats.fallback,1ull);
}
__global__ void k_JointCheckVolume(const UnifiedState* state,const float3* before,const JointBlock* blocks){
 const auto& s=*state;int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=s.nt||!s.live[t])return;int id[4];unifiedTetIds(s,t,id);
 const auto& b=blocks[s.component[id[0]]];if(!b.accepted||b.scale<=0)return;float3 a=before[id[0]],p=s.pos[id[0]];float det=unifiedKuhn(s,t).metric[6];
 float oldJ=dot3(before[id[1]]-a,cross3(before[id[2]]-a,before[id[3]]-a))*det;
 float newJ=dot3(s.pos[id[1]]-p,cross3(s.pos[id[2]]-p,s.pos[id[3]]-p))*det,margin=newJ-fminf(oldJ,s.minimum);
 if(!isfinite(margin)||margin< -2e-5f)atomicAdd(&g_jointStats.volumeViolation,1ull);
 if(margin<0){atomicAdd(&g_jointStats.negativeMargin,1ull);int* dst=(int*)&g_jointStats.minimumNegativeMargin;int old=*dst,assumed;
 do{assumed=old;old=atomicCAS(dst,assumed,__float_as_int(fminf(__int_as_float(assumed),margin)));}while(old!=assumed);}
}
__global__ void k_JointCheckClosing(const UnifiedState* state,const float3* before,const JointBlock* blocks,bool blade){
 const auto& s=*state;for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<*s.contactCount;k+=gridDim.x*blockDim.x){const auto& c=s.contacts[s.contactIds[k]];const auto& b=blocks[s.component[c.id[0]]];if(!b.accepted||b.scale<=0)continue;
 float3 d=mkf3(0,0,0);for(int j=0;j<c.count;j++)d=d+(s.pos[c.id[j]]-before[c.id[j]])*c.weight[j];
 ContactShape shape=bodyContactShape(s.tool,s.tool->pose,c.kind,s.thickness,s.minGap);float feature=blade?s.tool->pose.shaftR:shape.capsule?shape.radius:fminf(shape.half.x,fminf(shape.half.y,shape.half.z));
 if(dot3(d,c.plane.normal)<-.5f*feature-1e-6f)atomicAdd(&g_jointStats.closingViolation,1ull);
 }
}
