// Finite tool SDFs, conservative advancement and unilateral Coulomb contact.
// Included after xpbd_solver.cu's float3 helpers. This is NOT the full IPC optimizer.
struct ContactShape {
    float3 center, u, n, w, half, a, b;
    float radius;
    int capsule;
};
struct ContactRecord { float3 normal, target, motion; int valid; };
// Bounds in each jaw's CLOSED hinge frame (axis, opening normal, width).
// Uploaded from the same visible mesh as the articulation, in simulation units.
struct JawMeshBounds { float3 center,half; int enabled; };
static JawMeshBounds h_jawMeshBounds[2]={};
__constant__ JawMeshBounds c_jawMeshBounds[2];
__host__ __device__ inline float3 contactUnit(float3 x,float3 fallback) {
    float l=len3(x);return l>1e-8f?x*(1.f/l):fallback;
}
__host__ __device__ inline ContactShape contactShape(XpbdToolState t,int kind,float thickness,float minGap) {
    ContactShape s={};s.capsule=kind==0;
    if(kind==0){s.a=t.shaftA;s.b=t.shaftB;s.radius=t.shaftR;return s;}
    float3 u=contactUnit(t.jawAxis,mkf3(0,0,1));
    float3 n=t.jawNormal-u*dot3(u,t.jawNormal);
    n=contactUnit(n,fabsf(u.y)<.9f?contactUnit(cross3(u,mkf3(0,1,0)),mkf3(1,0,0)):mkf3(1,0,0));
    float side=kind==1?1.f:-1.f;
    float sn=fminf(.95f,fmaxf(0.f,t.jawGap)/(2.f*fmaxf(t.jawLen,1e-6f)));
    float cs=sqrtf(1-sn*sn);
    s.u=u*cs+n*(side*sn);s.n=n*cs-u*(side*sn);s.w=cross3(s.u,s.n);
    s.half=mkf3(t.jawLen*.5f,thickness*.5f,t.jawHalfWidth);
    s.center=t.jawHinge+s.u*s.half.x+s.n*(side*(s.half.y+.5f*minGap));
#ifdef __CUDA_ARCH__
    JawMeshBounds bounds=c_jawMeshBounds[kind-1];
#else
    JawMeshBounds bounds=h_jawMeshBounds[kind-1];
#endif
    if(bounds.enabled){
        s.half=bounds.half;
        s.center=t.jawHinge+s.u*bounds.center.x+s.n*bounds.center.y+s.w*bounds.center.z;
    }
    return s;
}
__device__ inline float shapeDistance(ContactShape s,float3 x,float3& normal,float3& surface) {
    if(s.capsule){
        float3 ab=s.b-s.a;float tt=fminf(1.f,fmaxf(0.f,dot3(x-s.a,ab)/fmaxf(dot3(ab,ab),1e-20f)));
        float3 d=x-(s.a+ab*tt);float l=len3(d);
        float3 fallback=contactUnit(cross3(ab,fabsf(ab.x)<fabsf(ab.y)?mkf3(1,0,0):mkf3(0,1,0)),mkf3(0,0,1));
        // On the axis, projection roundoff is parallel to the shaft. Normalizing
        // that residual would push along the shaft instead of out of its volume.
        normal=l>1e-6f*fmaxf(1.f,len3(ab))?d*(1.f/l):fallback;
        surface=s.a+ab*tt+normal*s.radius;return l-s.radius;
    }
    float3 r=x-s.center;float v[3]={dot3(r,s.u),dot3(r,s.n),dot3(r,s.w)};
    float h[3]={s.half.x,s.half.y,s.half.z};float q[3],cl[3];
    for(int k=0;k<3;k++){q[k]=fabsf(v[k])-h[k];cl[k]=fminf(h[k],fmaxf(-h[k],v[k]));}
    float3 closest=s.center+s.u*cl[0]+s.n*cl[1]+s.w*cl[2];
    // Form the outside vector in the orthonormal tool frame. Subtracting two
    // world positions here can turn a tiny normal gap into a spurious tangent.
    float3 d=s.u*(copysignf(fmaxf(q[0],0.f),v[0]))+
             s.n*(copysignf(fmaxf(q[1],0.f),v[1]))+
             s.w*(copysignf(fmaxf(q[2],0.f),v[2]));float len=len3(d);
    if(q[0]>0 || q[1]>0 || q[2]>0){normal=len>0?d*(1.f/len):s.n;surface=closest;return len;}
    int k=q[1]>q[0]?1:0;if(q[2]>q[k])k=2;
    float3 axes[3]={s.u,s.n,s.w};normal=axes[k]*(v[k]>=0?1.f:-1.f);
    surface=x-normal*q[k];return q[k];
}
__device__ inline ContactShape shapeLerp(ContactShape a,ContactShape b,float t) {
    ContactShape s=b;s.center=a.center*(1-t)+b.center*t;s.a=a.a*(1-t)+b.a*t;s.b=a.b*(1-t)+b.b*t;
    s.u=contactUnit(a.u*(1-t)+b.u*t,b.u);s.n=a.n*(1-t)+b.n*t;s.n=contactUnit(s.n-s.u*dot3(s.u,s.n),b.n);s.w=cross3(s.u,s.n);
    return s;
}
__device__ inline float3 rotateShaftVector(ContactShape from,ContactShape to,float3 r) {
    float3 a=from.b-from.a,b=to.b-to.a;
    if(len3(a)<1e-8f || len3(b)<1e-8f)return r;
    a=contactUnit(a,mkf3(0,0,1));b=contactUnit(b,a);
    float c=fminf(1.f,fmaxf(-1.f,dot3(a,b)));float3 v=cross3(a,b);
    if(c<-.9999f){float3 u=contactUnit(cross3(a,fabsf(a.x)<.9f?mkf3(1,0,0):mkf3(0,1,0)),mkf3(0,0,1));return u*(2*dot3(u,r))-r;}
    return r+cross3(v,r)+cross3(v,cross3(v,r))*(1.f/(1.f+c));
}
__device__ inline float3 witnessAt(ContactShape from,ContactShape to,float3 p) {
    if(from.capsule){
        float3 ab=from.b-from.a;float t=fminf(1.f,fmaxf(0.f,dot3(p-from.a,ab)/fmaxf(dot3(ab,ab),1e-20f)));
        return to.a+(to.b-to.a)*t+rotateShaftVector(from,to,p-from.a-ab*t);
    }
    float3 r=p-from.center;return to.center+to.u*dot3(r,from.u)+to.n*dot3(r,from.n)+to.w*dot3(r,from.w);
}
// Shared swept point query for lattice nodes and embedded triangle witnesses.
__device__ inline bool pointBoxTime(ContactShape box,float radius,float3 a,float3 b,float& time){
    float3 axes[3]={box.u,box.n,box.w};float p[3],v[3],half[3],cuts[8]={0,1};int count=2;
    for(int k=0;k<3;k++){
        p[k]=dot3(a,axes[k]);v[k]=dot3(b-a,axes[k]);half[k]=compF3(box.half,k);
        if(fabsf(v[k])>1e-12f)for(int side=-1;side<=1;side+=2){float t=(side*half[k]-p[k])/v[k];if(t>0 && t<1)cuts[count++]=t;}
    }
    for(int i=1;i<count;i++){float x=cuts[i];int j=i;while(j>0 && cuts[j-1]>x){cuts[j]=cuts[j-1];--j;}cuts[j]=x;}
    for(int i=0;i<count-1;i++){
        double A=0,B=0,C=-(double)radius*radius;float lo=cuts[i],hi=cuts[i+1],mid=.5f*(lo+hi);
        for(int k=0;k<3;k++){
            float q=p[k]+v[k]*mid;if(fabsf(q)<=half[k])continue;
            double d=p[k]-copysignf(half[k],q);A+=(double)v[k]*v[k];B+=2*d*v[k];C+=d*d;
        }
        double atLo=(A*lo+B)*lo+C;
        if(atLo<=1e-14){time=lo;return true;}
        if(A<=1e-24)continue;
        double disc=B*B-4*A*C;if(disc<0)continue;
        double root=(-B-sqrt(disc))/(2*A);
        if(root>=lo-1e-8 && root<=hi+1e-8){time=fminf(hi,fmaxf(lo,(float)root));return true;}
    }
    return false;
}
__device__ inline ContactRecord traceToolContact(ContactShape from,ContactShape to,float radius,float3 x0,float3 x1) {
    ContactRecord out={};float3 normal,surface;
    // Exact piecewise-quadratic distance to an OBB for translational sweeps.
    // No iteration-budget pseudo-contact, including grazing rounded corners.
    if(!to.capsule && len3(to.u-from.u)+len3(to.n-from.n)<1e-7f){
        float time=0;if(!pointBoxTime(to,radius,x0-from.center,x1-to.center,time))return out;
        ContactShape hit=shapeLerp(from,to,time);shapeDistance(hit,x0*(1-time)+x1*time,normal,surface);
        float3 end=witnessAt(hit,to,surface),start=witnessAt(hit,from,surface);
        out.normal=normal;out.target=end+normal*radius;out.motion=end-start;out.valid=1;return out;
    }
    float speed=len3(x1-x0);
    if(to.capsule)speed+=fmaxf(len3(to.a-from.a),len3(to.b-from.b));
    else speed+=len3(to.center-from.center)+4.f*len3(to.half)*(len3(to.u-from.u)+len3(to.n-from.n));
    // Swept bounding sphere rejects distant nodes without iterating the SDF.
    float3 center0=to.capsule?(from.a+from.b)*.5f:from.center;
    float3 center1=to.capsule?(to.a+to.b)*.5f:to.center;
    float bound=to.capsule?fmaxf(len3(from.b-from.a),len3(to.b-to.a))*.5f+to.radius:len3(to.half);
    if(len3(x0-center0)>bound+radius+len3(x1-x0)+len3(center1-center0)+1e-5f)return out;
    float tol=fmaxf(1e-6f,1e-4f*(bound+radius));double t=0;ContactShape hit=from;
    bool found=false;
    while(t<1.){
        hit=shapeLerp(from,to,t);float d=shapeDistance(hit,x0*(1-t)+x1*t,normal,surface)-radius;
        if(d<=tol){found=true;break;}
        if(speed<=1e-12f)break;
        float step=.8f*d/speed;
        if(t+step>=1.f){t=1;hit=to;float de=shapeDistance(to,x1,normal,surface)-radius;found=de<=tol;break;}
        t+=step;
        // A work budget is not evidence of collision. Advance to a certified
        // hit/end; double time avoids float stagnation in grazing trajectories.
    }
    if(!found)return out;
    float3 end=witnessAt(hit,to,surface),start=witnessAt(hit,from,surface);
    if(to.capsule)normal=rotateShaftVector(hit,to,normal);
    else normal=contactUnit(to.u*dot3(normal,hit.u)+to.n*dot3(normal,hit.n)+to.w*dot3(normal,hit.w),normal);
    out.normal=normal;out.target=end+normal*radius;out.motion=end-start;out.valid=1;return out;
}
__global__ void k_FindToolContacts(int N,ContactShape from,ContactShape to,int kind,float radius,
    const float3* previous,const float3* pos,const float* invMass,ContactRecord* records) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
    ContactRecord out={};
    if(invMass[i]>0 && (to.capsule?to.radius>0:to.half.x>0&&to.half.z>0))
        out=traceToolContact(from,to,radius,previous[i],pos[i]);
    records[3*i+kind]=out;
}
__global__ void k_SolveFrictionContacts(int N,float alpha,float muS,float muD,int closing,
    const float* invMass,const float3* previous,ContactRecord* records,float* lambdaN,float3* lambdaT,
    int* contactFlags,float3* pos) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;float w=invMass[i];if(w<=0)return;
    int gripping=0;
    for(int kind=0;kind<3;kind++){
        int j=3*i+kind;ContactRecord c=records[j];if(!c.valid)continue;
        float C=dot3(pos[i]-c.target,c.normal),old=lambdaN[j];
        float next=fmaxf(0.f,old+(-C-alpha*old)/(w+alpha));
        pos[i]=pos[i]+c.normal*(w*(next-old));lambdaN[j]=next;
        float3 slip=pos[i]-previous[i]-c.motion;slip=slip-c.normal*dot3(slip,c.normal);
        float3 oldT=lambdaT[j];float3 trial=oldT-slip*(1.f/w);float len=len3(trial);
        float3 nextT=trial;
        if(len>muS*next)nextT=len>1e-15f?trial*(muD*next/len):mkf3(0,0,0);
        pos[i]=pos[i]+(nextT-oldT)*w;lambdaT[j]=nextT;
        if(kind>0 && closing && next>0.f)gripping=1;
    }
    contactFlags[i]=gripping; // Diagnostic only: never changes mass, topology, or captures an anchor.
}
