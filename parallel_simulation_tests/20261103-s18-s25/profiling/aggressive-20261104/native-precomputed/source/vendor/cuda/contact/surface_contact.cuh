// Triangle/SDF local optimization (Macklin et al., 2020) with embedded XPBD response.
// Intact production surfaces use the renderer's fixed material weights. Cut components
// retain a local affine approximation, rebuilt with the DC surface each outer frame.
struct SurfaceVertex { int id[8],count,voxel,component; unsigned mask; float weight[8]; float3 offset; };
struct SurfaceContact {
    int id[24],count,kind; float weight[24];
    float3 offset,previous; ContactRecord plane; float lambdaN; float3 lambdaT;
    int owner[3],component[3]; unsigned mask[3];
    float3 toolOrigin,gapGradient; float toolGap; ContactShape referenceShape; int incoming;
    // Jaw contacts of the friction grasp: tangential spring elongation carried
    // from the last substep in the jaw box frame (u, n, w), and the spring force
    // and stiffness summed over the loaded samples that set it (see
    // k_UnifiedStick). Zero elsewhere.
    float3 stick,stickSpring; float stickStiffness;
};
constexpr int CONTACT_CACHE_SIZE=8192;
__device__ inline void contactKey(int* ids){
    if(ids[0]>ids[1]){int x=ids[0];ids[0]=ids[1];ids[1]=x;}
    if(ids[1]>ids[2]){int x=ids[1];ids[1]=ids[2];ids[2]=x;}
    if(ids[0]>ids[1]){int x=ids[0];ids[0]=ids[1];ids[1]=x;}
}
__device__ inline unsigned contactHash(const int* ids,int kind){
    unsigned h=2166136261u;for(int k=0;k<3;k++)h=(h^(unsigned)ids[k])*16777619u;
    return (h^(unsigned)kind)&(CONTACT_CACHE_SIZE-1);
}
__device__ inline bool sameContactKey(const SurfaceContact& c,const int* ids,int kind){
    if(!c.count || c.kind!=kind || c.mask[0] || c.mask[1] || c.mask[2])return false;
    int key[3]={c.owner[0],c.owner[1],c.owner[2]};contactKey(key);
    return key[0]==ids[0] && key[1]==ids[1] && key[2]==ids[2];
}
__global__ void k_CacheContacts(const SurfaceContact* contacts,const int* active,const int* count,int* slots,SurfaceContact* cache){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=*count)return;
    int index=active[i];const SurfaceContact& c=contacts[index];if(!c.count || c.mask[0] || c.mask[1] || c.mask[2])return;
    int key[3]={c.owner[0],c.owner[1],c.owner[2]};contactKey(key);unsigned h=contactHash(key,c.kind);
    for(int probe=0;probe<CONTACT_CACHE_SIZE;probe++){
        unsigned slot=(h+probe)&(CONTACT_CACHE_SIZE-1);int old=atomicCAS(slots+slot,-1,index);
        if(old==-1){cache[slot]=c;return;}
        if(sameContactKey(contacts[old],key,c.kind))return;
    }
}
// Stable material identity for cut faces. Reconstruction can reorder triangles;
// voxel identity alone would merge the two sides of a crack.
__device__ inline bool materialKeyLess(uint3 a,uint3 b){
    return a.x<b.x || (a.x==b.x && (a.y<b.y || (a.y==b.y && a.z<b.z)));
}
__device__ inline void sortMaterialKey(uint3* key){
    if(materialKeyLess(key[1],key[0])){uint3 x=key[0];key[0]=key[1];key[1]=x;}
    if(materialKeyLess(key[2],key[1])){uint3 x=key[1];key[1]=key[2];key[2]=x;}
    if(materialKeyLess(key[1],key[0])){uint3 x=key[0];key[0]=key[1];key[1]=x;}
}
__device__ inline void materialContactKey(const SurfaceContact& c,uint3* key){
    for(int k=0;k<3;k++)key[k]=make_uint3(c.owner[k],c.mask[k],c.component[k]+1);
    sortMaterialKey(key);
}
__device__ inline unsigned materialContactHash(const uint3* key,int kind){
    unsigned hash=2166136261u;
    for(int k=0;k<3;k++){hash=(hash^key[k].x)*16777619u;hash=(hash^key[k].y)*16777619u;hash=(hash^key[k].z)*16777619u;}
    return (hash^(unsigned)kind)&(CONTACT_CACHE_SIZE-1);
}
__device__ inline bool sameMaterialContactKey(const SurfaceContact& c,const uint3* key,int kind){
    if(!c.count || c.kind!=kind)return false;
    uint3 other[3];materialContactKey(c,other);
    for(int k=0;k<3;k++)if(other[k].x!=key[k].x || other[k].y!=key[k].y || other[k].z!=key[k].z)return false;
    return true;
}
__global__ void k_CacheBladeBackContacts(const SurfaceContact* contacts,const int* active,const int* count,int* slots,SurfaceContact* cache){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=*count)return;
    int index=active[i];const SurfaceContact& c=contacts[index];if(!c.count || c.kind<3)return;
    uint3 key[3];materialContactKey(c,key);unsigned hash=materialContactHash(key,c.kind);
    for(int probe=0;probe<CONTACT_CACHE_SIZE;probe++){
        unsigned slot=(hash+probe)&(CONTACT_CACHE_SIZE-1);int old=atomicCAS(slots+slot,-1,index);
        if(old==-1){cache[slot]=c;return;}
        if(sameMaterialContactKey(contacts[old],key,c.kind))return;
    }
}
__device__ inline void add3(float3* p,float3 v){atomicAdd(&p->x,v.x);atomicAdd(&p->y,v.y);atomicAdd(&p->z,v.z);}
__device__ inline float3 surfacePoint(SurfaceVertex v,const float3* p){
    float3 x=v.offset;for(int k=0;k<v.count;k++)x=x+p[v.id[k]]*v.weight[k];return x;
}
// Inverse deformed-hexahedron coordinates. Equal weights erase sub-cell spatial
// differences, making the two jaws act on the same stencil even at opposite faces.
__device__ inline bool surfaceCoordinates(const float3* p,float3 x,float* weights){
    float3 uv=mkf3(.5f,.5f,.5f);
    for(int it=0;it<8;it++){
        float3 q=mkf3(0,0,0),dx=q,dy=q,dz=q;
        for(int k=0;k<8;k++){
            int bx=(k==1||k==2||k==5||k==6),by=(k==2||k==3||k==6||k==7),bz=k>=4;
            float a=bx?uv.x:1-uv.x,b=by?uv.y:1-uv.y,c=bz?uv.z:1-uv.z;
            q=q+p[k]*(a*b*c);dx=dx+p[k]*((bx?1.f:-1.f)*b*c);
            dy=dy+p[k]*(a*(by?1.f:-1.f)*c);dz=dz+p[k]*(a*b*(bz?1.f:-1.f));
        }
        float det=dot3(dx,cross3(dy,dz));
        if(fabsf(det)<1e-8f*fmaxf(1e-12f,len3(dx)*len3(dy)*len3(dz)))return false;
        float3 r=x-q,du=mkf3(dot3(r,cross3(dy,dz)),dot3(dx,cross3(r,dz)),dot3(dx,cross3(dy,r)))*(1.f/det);
        uv=uv+du;uv.x=fminf(1.f,fmaxf(0.f,uv.x));uv.y=fminf(1.f,fmaxf(0.f,uv.y));uv.z=fminf(1.f,fmaxf(0.f,uv.z));
    }
    for(int k=0;k<8;k++){
        int bx=(k==1||k==2||k==5||k==6),by=(k==2||k==3||k==6||k==7),bz=k>=4;
        weights[k]=(bx?uv.x:1-uv.x)*(by?uv.y:1-uv.y)*(bz?uv.z:1-uv.z);
    }
    return true;
}
__global__ void k_BindSurface(int count,const int* tags,const float3* expanded,const int* corners,
    const unsigned* masks,const int* conn,const float3* pos,SurfaceVertex* vertices,
    const float3* rest=nullptr,const float4* materialFP=nullptr,const unsigned* boundaryMasks=nullptr){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    unsigned tag=(unsigned)tags[i];int index=tag&0x3fffffffu;bool cut=(tag&0x80000000u)!=0;
    SurfaceVertex v={};v.voxel=cut?index/8:index;v.component=cut?index%8:-1;v.mask=masks?masks[v.voxel]&4095u:0;
    // A cut-voxel skin fallback has no component identity. Do not interpolate
    // across both fragments in order to manufacture a contact attachment.
    if((!cut && v.mask!=0) || (cut && !conn)){vertices[i]=v;return;}
    // The cache key remains the ACTUAL severed-edge mask. Distinct true cuts
    // can have the same surface grouping after air paths are excluded.
    // Only the interpolation group lookup uses the reconstruction mask.
    const unsigned boundary=boundaryMasks?boundaryMasks[v.voxel]&4095u:v.mask;
    float3 center=mkf3(0,0,0);
    for(int k=0;k<8;k++){
        if(cut && conn && conn[9*boundary+1+k]!=index%8)continue;
        int id=corners[8*v.voxel+k];v.id[v.count++]=id;center=center+pos[id];
    }
    for(int k=0;k<v.count;k++)v.weight[k]=1.f/v.count;
    if(!cut && v.count==8){
        if(rest && materialFP && materialFP[v.voxel].w>.5f){
            // The intact renderer advects a fixed material point. Use exactly
            // that Jacobian even when the deformed cell is folded or ill-conditioned.
            float3 a=rest[v.id[0]],b=rest[v.id[6]];float4 r=materialFP[v.voxel];
            float u=fminf(1.f,fmaxf(0.f,(r.x-a.x)/(b.x-a.x)));
            float v1=fminf(1.f,fmaxf(0.f,(r.y-a.y)/(b.y-a.y)));
            float w=fminf(1.f,fmaxf(0.f,(r.z-a.z)/(b.z-a.z)));
            for(int k=0;k<8;k++){
                int bx=k==1||k==2||k==5||k==6,by=k==2||k==3||k==6||k==7,bz=k>=4;
                v.weight[k]=(bx?u:1-u)*(by?v1:1-v1)*(bz?w:1-w);
            }
        } else {
            float3 p[8];for(int k=0;k<8;k++)p[k]=pos[v.id[k]];
            surfaceCoordinates(p,expanded[i],v.weight);
        }
    }
    center=mkf3(0,0,0);for(int k=0;k<v.count;k++)center=center+pos[v.id[k]]*v.weight[k];
    v.offset=expanded[i]-center;vertices[i]=v;
}
// Frank-Wolfe on the triangle simplex, with a bounded golden-section line search.
// Keep the best iterate; include vertices so narrow features are not lost at a bad seed.
// Clip the actual triangle against the finite OBB. At a CCD hit the SDF
// optimizer need not have converged to the intersection; using its approximate
// barycentric point can select an unrelated edge normal.
__device__ inline bool clippedBoxWitness(ContactShape shape,const float3* q,float margin,float3& bary){
    float3 p[12],b[12],np[12],nb[12];
    for(int k=0;k<3;k++){
        float3 r=q[k]-shape.center;p[k]=mkf3(dot3(r,shape.u),dot3(r,shape.n),dot3(r,shape.w));
        b[k]=mkf3(k==0,k==1,k==2);
    }
    int count=3;
    for(int axis=0;axis<3 && count;axis++)for(int side=-1;side<=1 && count;side+=2){
        int next=0;float half=compF3(shape.half,axis)+margin;
        if(half<0)return false;
        float3 prev=p[count-1],pb=b[count-1];float pd=side*compF3(prev,axis)-half;
        for(int k=0;k<count;k++){
            float d=side*compF3(p[k],axis)-half;
            if((d<=0)!=(pd<=0)){
                if(next>=12)return false;float t=pd/(pd-d);
                np[next]=prev+(p[k]-prev)*t;nb[next++]=pb+(b[k]-pb)*t;
            }
            if(d<=0){if(next>=12)return false;np[next]=p[k];nb[next++]=b[k];}
            prev=p[k];pb=b[k];pd=d;
        }
        count=next;for(int k=0;k<count;k++){p[k]=np[k];b[k]=nb[k];}
    }
    if(!count)return false;
    bary=mkf3(0,0,0);for(int k=0;k<count;k++)bary=bary+b[k];bary=bary*(1.f/count);
    return true;
}
__device__ inline float3 triangleWitness(ContactShape shape,float3 a,float3 b,float3 c){
    float3 q[3]={a,b,c},bar=mkf3(1.f/3,1.f/3,1.f/3),best=bar,n,s;
    float bestD=shapeDistance(shape,(a+b+c)*(1.f/3),n,s);
    for(int k=0;k<3;k++){float d=shapeDistance(shape,q[k],n,s);if(d<bestD){bestD=d;best=mkf3(k==0,k==1,k==2);}}
    for(int it=0;it<12;it++){
        float3 p=a*bar.x+b*bar.y+c*bar.z;shapeDistance(shape,p,n,s);
        int k=dot3(n,b)<dot3(n,a)?1:0;if(dot3(n,c)<dot3(n,q[k]))k=2;
        float3 target=mkf3(k==0,k==1,k==2),dir=q[k]-p;
        // Reuse the surviving golden-section sample. Same eight interval
        // reductions; nine SDF evaluations instead of sixteen per line search.
        float lo=0,hi=1,l=.381966f,r=.618034f;
        float dl=shapeDistance(shape,p+dir*l,n,s),dr=shapeDistance(shape,p+dir*r,n,s);
        for(int j=0;j<8;j++){
            if(dl<dr){
                hi=r;r=l;dr=dl;l=lo+(hi-lo)*.381966f;
                if(j<7)dl=shapeDistance(shape,p+dir*l,n,s);
            }else{
                lo=l;l=r;dl=dr;r=lo+(hi-lo)*.618034f;
                if(j<7)dr=shapeDistance(shape,p+dir*r,n,s);
            }
        }
        float t=.5f*(lo+hi);bar=bar*(1-t)+target*t;
        float d=shapeDistance(shape,a*bar.x+b*bar.y+c*bar.z,n,s);
        if(d<bestD){bestD=d;best=bar;}
    }
    return best;
}
// Exact finite-segment/triangle distance for the thin pressure blade. A bounded
// SDF minimizer can miss a sub-millimetre blade on a large/skinny cut triangle.
// The minimum is a face crossing, an endpoint/face pair, or an edge/edge pair.
__device__ inline float3 bladeTriangleWitness(ContactShape shape,const float3* q){
    float3 e0=q[1]-q[0],e1=q[2]-q[0],axis=shape.b-shape.a,n=cross3(e0,e1);
    float nn=dot3(n,n),aa=dot3(axis,axis),bestD=1e30f;float3 best=mkf3(1,0,0);
    float d00=dot3(e0,e0),d01=dot3(e0,e1),d11=dot3(e1,e1);
    // Project both endpoints and the segment/plane crossing onto the face.
    float denom=dot3(n,axis),time=fabsf(denom)>1e-20f?dot3(n,q[0]-shape.a)/denom:-1;
    for(int j=0;j<3;j++){
        float t=j==0?0:j==1?1:time;if(t<0||t>1||nn<1e-20f)continue;
        float3 x=shape.a+axis*t,r=x-q[0];float d20=dot3(r,e0),d21=dot3(r,e1);
        float v=(d11*d20-d01*d21)/nn,w=(d00*d21-d01*d20)/nn,u=1-v-w;
        if(u<0||v<0||w<0)continue;
        float3 delta=x-(q[0]*u+q[1]*v+q[2]*w);float d=dot3(delta,delta);
        if(d<bestD){bestD=d;best=mkf3(u,v,w);}
    }
    for(int j=0;j<3;j++){
        int k=(j+1)%3;float3 edge=q[k]-q[j],r=shape.a-q[j];
        float ee=dot3(edge,edge),ae=dot3(axis,edge),ar=dot3(axis,r),er=dot3(edge,r),t=0,u=0;
        if(aa<=1e-20f)u=ee>1e-20f?fminf(1.f,fmaxf(0.f,er/ee)):0;
        else if(ee<=1e-20f)t=fminf(1.f,fmaxf(0.f,-ar/aa));
        else{
            float det=aa*ee-ae*ae;t=det>1e-12f*aa*ee?fminf(1.f,fmaxf(0.f,(ae*er-ar*ee)/det)):0;
            u=(ae*t+er)/ee;
            if(u<0){u=0;t=fminf(1.f,fmaxf(0.f,-ar/aa));}
            else if(u>1){u=1;t=fminf(1.f,fmaxf(0.f,(ae-ar)/aa));}
        }
        float3 delta=shape.a+axis*t-q[j]-edge*u;float d=dot3(delta,delta);
        if(d<bestD){bestD=d;best=mkf3(0,0,0);if(j==0)best.x=1-u;else if(j==1)best.y=1-u;else best.z=1-u;if(k==0)best.x=u;else if(k==1)best.y=u;else best.z=u;}
    }
    return best;
}
__device__ inline void shapeBounds(ContactShape s,float radius,float3& lo,float3& hi){
    float3 center=s.capsule?(s.a+s.b)*.5f:s.center;
    float3 e=s.capsule?mkf3(fabsf(s.b.x-s.a.x)*.5f+s.radius,fabsf(s.b.y-s.a.y)*.5f+s.radius,fabsf(s.b.z-s.a.z)*.5f+s.radius):
        mkf3(fabsf(s.u.x)*s.half.x+fabsf(s.n.x)*s.half.y+fabsf(s.w.x)*s.half.z,
             fabsf(s.u.y)*s.half.x+fabsf(s.n.y)*s.half.y+fabsf(s.w.y)*s.half.z,
             fabsf(s.u.z)*s.half.x+fabsf(s.n.z)*s.half.y+fabsf(s.w.z)*s.half.z);
    e=e+mkf3(radius,radius,radius);lo=center-e;hi=center+e;
}
__device__ inline float triangleAxisGap(float3 a,float3 b,float3 c,float3 axis,float3 half){
    float len=len3(axis);if(len<1e-10f)return 0;
    float p=dot3(a,axis),q=dot3(b,axis),r=dot3(c,axis);
    float radius=fabsf(axis.x)*half.x+fabsf(axis.y)*half.y+fabsf(axis.z)*half.z;
    return fmaxf(0.f,fmaxf(fminf(p,fminf(q,r))-radius,-radius-fmaxf(p,fmaxf(q,r))))/len;
}
// Separating-axis gaps are LOWER bounds on triangle/box distance, unlike a
// local SDF minimizer. Conservative advancement therefore cannot step over a
// narrow jaw merely because the final closest point differs from the first hit.
__device__ inline float triangleBoxGap(ContactShape s,float3 a,float3 b,float3 c){
    float3 q[3],world[3]={a-s.center,b-s.center,c-s.center};
    for(int k=0;k<3;k++)q[k]=mkf3(dot3(world[k],s.u),dot3(world[k],s.n),dot3(world[k],s.w));
    float3 axes[3]={mkf3(1,0,0),mkf3(0,1,0),mkf3(0,0,1)};float gap=0;
    for(int k=0;k<3;k++){
        gap=fmaxf(gap,triangleAxisGap(q[0],q[1],q[2],axes[k],s.half));
        float3 edge=q[(k+1)%3]-q[k];
        for(int j=0;j<3;j++)gap=fmaxf(gap,triangleAxisGap(q[0],q[1],q[2],cross3(edge,axes[j]),s.half));
    }
    return fmaxf(gap,triangleAxisGap(q[0],q[1],q[2],cross3(q[1]-q[0],q[2]-q[0]),s.half));
}
__device__ inline bool sweptAxisInterval(float3 a,float3 b,float3 c,float3 velocity,float3 axis,float3 half,float& enter,float& leave){
    if(dot3(axis,axis)<1e-18f)return true;
    float p=dot3(a,axis),q=dot3(b,axis),r=dot3(c,axis),v=dot3(velocity,axis);
    float low=fminf(p,fminf(q,r)),high=fmaxf(p,fmaxf(q,r));
    float radius=fabsf(axis.x)*half.x+fabsf(axis.y)*half.y+fabsf(axis.z)*half.z;
    if(fabsf(v)<1e-12f)return low<=radius && high>=-radius;
    float lo=(-radius-high)/v,hi=(radius-low)/v;if(lo>hi){float tmp=lo;lo=hi;hi=tmp;}
    enter=fmaxf(enter,lo);leave=fminf(leave,hi);return enter<=leave;
}
__device__ inline ContactRecord traceTriangleBox(ContactShape from,ContactShape to,float skin,const float3* p,const float3* q,float3& bary){
    // ACCD's relative-motion bound: common translation cannot close the gap.
    // Retain the existing conservative rotation bound for the finite jaw.
    ContactRecord out={};float3 translation=to.center-from.center;
    float3 motion=q[0]-p[0];
    if(len3(to.u-from.u)+len3(to.n-from.n)<1e-7f){
        float3 axes[3]={to.u,to.n,to.w};
        for(int ax=0;ax<3;ax++){
            float lo=1e30f,hi=-1e30f;
            for(int k=0;k<3;k++){
                float a=dot3(p[k]-from.center,axes[ax]),b=dot3(q[k]-to.center,axes[ax]);
                lo=fminf(lo,fminf(a,b));hi=fmaxf(hi,fmaxf(a,b));
            }
            float half=compF3(to.half,ax)+skin;if(lo>half || hi<-half)return out;
        }
    }
    if(len3(to.u-from.u)+len3(to.n-from.n)<1e-7f && len3(q[1]-p[1]-motion)+len3(q[2]-p[2]-motion)<1e-7f){
        float3 a[3],axes[3]={mkf3(1,0,0),mkf3(0,1,0),mkf3(0,0,1)};
        for(int k=0;k<3;k++){float3 x=p[k]-from.center;a[k]=mkf3(dot3(x,to.u),dot3(x,to.n),dot3(x,to.w));}
        float3 v=motion-translation;v=mkf3(dot3(v,to.u),dot3(v,to.n),dot3(v,to.w));
        float3 half=to.half+mkf3(skin,skin,skin);float enter=0,leave=1;
        for(int k=0;k<3;k++){
            if(!sweptAxisInterval(a[0],a[1],a[2],v,axes[k],half,enter,leave))return out;
            float3 edge=a[(k+1)%3]-a[k];
            for(int j=0;j<3;j++)if(!sweptAxisInterval(a[0],a[1],a[2],v,cross3(edge,axes[j]),half,enter,leave))return out;
        }
        if(!sweptAxisInterval(a[0],a[1],a[2],v,cross3(a[1]-a[0],a[2]-a[0]),half,enter,leave))return out;
        ContactShape hit=shapeLerp(from,to,enter);float3 x[3];for(int k=0;k<3;k++)x[k]=p[k]+motion*enter;
        if(!clippedBoxWitness(hit,x,skin+2e-6f,bary))bary=triangleWitness(hit,x[0],x[1],x[2]);float3 n,s;
        shapeDistance(hit,x[0]*bary.x+x[1]*bary.y+x[2]*bary.z,n,s);
        float3 end=witnessAt(hit,to,s),start=witnessAt(hit,from,s);
        out.normal=n;out.target=end+n*skin;out.motion=end-start;out.valid=1;return out;
    }
    float speed=4.f*len3(to.half)*(len3(to.u-from.u)+len3(to.n-from.n));
    float vertexSpeed=0;for(int k=0;k<3;k++)vertexSpeed=fmaxf(vertexSpeed,len3(q[k]-p[k]-translation));speed+=vertexSpeed;
    double time=0;float tol=fmaxf(1e-6f,1e-4f*len3(to.half));ContactShape hit=from;float3 x[3];bool found=false;
    while(time<=1.){
        hit=shapeLerp(from,to,time);for(int k=0;k<3;k++)x[k]=p[k]*(1-time)+q[k]*time;
        float gap=triangleBoxGap(hit,x[0],x[1],x[2])-skin;
        if(gap<=tol){found=true;break;}
        if(speed<=1e-12f || time>=1.f)return out;
        time=fmin(1.,time+.8*(double)gap/speed);
    }
    if(!found)return out;
    if(!clippedBoxWitness(hit,x,skin+tol,bary))return out;float3 n,s;
    shapeDistance(hit,x[0]*bary.x+x[1]*bary.y+x[2]*bary.z,n,s);
    float3 end=witnessAt(hit,to,s),start=witnessAt(hit,from,s);
    n=contactUnit(to.u*dot3(n,hit.u)+to.n*dot3(n,hit.n)+to.w*dot3(n,hit.w),n);
    out.normal=n;out.target=end+n*skin;out.motion=end-start;out.valid=1;return out;
}
// The same embedded surface is queried by shaft and both jaws. Evaluate it once
// per substep; broad phase then reads compact points rather than 24-node stencils.
__global__ void k_UpdateSurfacePoints(int count,const SurfaceVertex* vertices,const unsigned* masks,
    const float3* prev,const float3* pos,float4* current,float3* previous){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const SurfaceVertex& v=vertices[i];
    if(v.count==0 || (masks && (masks[v.voxel]&4095u)!=v.mask)){current[i]=make_float4(0,0,0,0);return;}
    float3 q=surfacePoint(v,pos);current[i]=make_float4(q.x,q.y,q.z,1);
    previous[i]=surfacePoint(v,prev);
}
__global__ void k_FindSurfaceReference(int triangles,int kind,ContactShape from,ContactShape to,float skin,
    const SurfaceVertex* vertices,const unsigned* masks,const float3* prev,const float3* pos,
    SurfaceContact* contacts,int* degree,int* active,int* activeCount,const float4* current,const float3* previous,
    const CoupledTool* body=nullptr,float thickness=0,float minGap=0,bool stationary=false,
    const int* cacheSlots=nullptr,const SurfaceContact* cache=nullptr,const int* bladePhase=nullptr,int stride=3,bool cacheComponents=false,const unsigned char* groupRejected=nullptr){
    int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=triangles)return;
    int slot=stride*t+kind;
    // A carried cutting preload may survive outside the current tool AABB.
    if(groupRejected && groupRejected[t/64] &&
       !(body && bladePhase && (*bladePhase==0 || *bladePhase==4) && contacts[slot].count>0)){
        contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;return;
    }
    if(body){from=bodyContactShape(body,stationary?body->pose:body->previous,kind,thickness,minGap);to=bodyContactShape(body,body->pose,kind,thickness,minGap);}
    // Reject distant triangles before reading/probing the material-contact
    // cache or copying a 24-node witness. This is the dominant no-contact path.
    if(to.capsule?to.radius<=0:to.half.x<=0||to.half.z<=0){contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;return;}
    float3 q[3],p[3];
    for(int k=0;k<3;k++){
        float4 point=current[3*t+k];if(point.w==0){contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;return;}
        q[k]=mkf3(point.x,point.y,point.z);p[k]=previous[3*t+k];
    }
    // An unfractured surface is a unilateral barrier on its incident side.
    // Re-querying the closest side of a thin cylinder every substep caps the
    // preload at K*radius, then reverses the normal and lets the blade through.
    // Carry this material witness until fracture or separation, including cut
    // components. Rebind its interpolation below to the current reconstruction.
    bool preload=body && to.capsule && bladePhase && (*bladePhase==0 || *bladePhase==4) && contacts[slot].count>0;
    for(int k=0;k<3 && preload;k++)preload=contacts[slot].owner[k]==vertices[3*t+k].voxel &&
        contacts[slot].mask[k]==vertices[3*t+k].mask && contacts[slot].component[k]==vertices[3*t+k].component;
    ContactRecord preloadPlane={};float3 preloadBary=mkf3(0,0,0);
    if(preload){
        const SurfaceContact& old=contacts[slot];preloadBary=bladeTriangleWitness(to,q);
        float3 normal=rotateShaftVector(old.referenceShape,to,old.plane.normal);
        float3 end=witnessAt(old.referenceShape,to,old.plane.target-old.plane.normal*skin);
        float3 x=q[0]*preloadBary.x+q[1]*preloadBary.y+q[2]*preloadBary.z;
        // Project onto the incident plane and require the finite capsule's
        // footprint. This is not an infinite plane or an enlarged collision box.
        float gap=dot3(x-end,normal);float3 nn,ss;
        float lateral=shapeDistance(to,x-normal*(gap+to.radius),nn,ss);
        preload=gap<=0 && lateral<=fmaxf(1e-6f,1e-3f*to.radius);
        if(preload){preloadPlane.normal=normal;preloadPlane.target=end+normal*skin;
            preloadPlane.motion=end-witnessAt(to,from,end);preloadPlane.valid=1;}
    }
    float3 lo,hi,l0,h0;shapeBounds(to,skin,lo,hi);shapeBounds(from,skin,l0,h0);
    // Endpoint OBB bounds alone miss intermediate rotational sweeps.
    if(!to.capsule && (len3(to.u-from.u)+len3(to.n-from.n)>1e-5f)){
        float r=len3(to.half)+skin;float3 extent=mkf3(r,r,r);lo=to.center-extent;hi=to.center+extent;l0=from.center-extent;h0=from.center+extent;
    }
    for(int ax=0;ax<3;ax++){
        float mn=1e30f,mx=-1e30f;for(int k=0;k<3;k++){mn=fminf(mn,fminf(compF3(q[k],ax),compF3(p[k],ax)));mx=fmaxf(mx,fmaxf(compF3(q[k],ax),compF3(p[k],ax)));}
        if(!preload && (mx<fminf(compF3(lo,ax),compF3(l0,ax)) || mn>fmaxf(compF3(hi,ax),compF3(h0,ax)))){contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;return;}
    }
    // Persist a material witness while its triangle owners and cut masks match.
    // A nearest-face query inside a thin jaw can otherwise flip its normal and
    // push tissue through the opposite face on the next substep.
    // Retain the incident side only for the same voxel, cut mask AND component.
    // The actual material witness is rebuilt below from current vertex weights.
    SurfaceContact retained;bool keep=body && contacts[slot].count>0;
    for(int k=0;k<3 && keep;k++)keep=vertices[3*t+k].count>0 &&
        contacts[slot].component[k]==vertices[3*t+k].component &&
        contacts[slot].owner[k]==vertices[3*t+k].voxel && contacts[slot].mask[k]==vertices[3*t+k].mask;
    if(keep)retained=contacts[slot];
    if(!keep && body && cacheSlots && cache && cacheComponents){
        uint3 key[3];bool valid=true;
        for(int k=0;k<3;k++){const SurfaceVertex& v=vertices[3*t+k];valid=valid && v.count>0;key[k]=make_uint3(v.voxel,v.mask,v.component+1);}
        if(valid){
            sortMaterialKey(key);unsigned hash=materialContactHash(key,kind);
            for(int probe=0;probe<CONTACT_CACHE_SIZE;probe++){
                unsigned slot=(hash+probe)&(CONTACT_CACHE_SIZE-1);if(cacheSlots[slot]<0)break;
                if(sameMaterialContactKey(cache[slot],key,kind)){retained=cache[slot];keep=true;break;}
            }
        }
    }
    if(!keep && body && cacheSlots && cache && !cacheComponents){
        int key[3];bool intact=true;
        for(int k=0;k<3;k++){const SurfaceVertex& v=vertices[3*t+k];key[k]=v.voxel;intact=intact && v.count>0 && v.component<0 && v.mask==0;}
        if(intact){
            contactKey(key);unsigned h=contactHash(key,kind);
            for(int probe=0;probe<CONTACT_CACHE_SIZE;probe++){
                unsigned s=(h+probe)&(CONTACT_CACHE_SIZE-1);if(cacheSlots[s]<0)break;
                if(sameContactKey(cache[s],key,kind)){
                    retained=cache[s];keep=true;
                    for(int k=0;k<3;k++){retained.owner[k]=vertices[3*t+k].voxel;retained.mask[k]=vertices[3*t+k].mask;}
                    break;
                }
            }
        }
    }
    // Rejected triangles have no live contact. Do not clear the full 24-node
    // stencil for every triangle, jaw and substep: only the validity fields.
    contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;
    ContactRecord retainedPlane={};
    if(keep){
        float3 x=retained.offset,old=x;
        for(int j=0;j<retained.count;j++){x=x+pos[retained.id[j]]*retained.weight[j];old=old+prev[retained.id[j]]*retained.weight[j];}
        float3 normal=retained.plane.normal;
        float3 start=witnessAt(retained.referenceShape,from,retained.plane.target-normal*skin);
        if(!to.capsule)normal=contactUnit(from.u*dot3(normal,retained.referenceShape.u)+from.n*dot3(normal,retained.referenceShape.n)+from.w*dot3(normal,retained.referenceShape.w),normal);
        float3 fromNormal=normal;
        float3 end=witnessAt(from,to,start),nn,ss;
        float distance=shapeDistance(to,x,nn,ss);
        if(to.capsule)normal=rotateShaftVector(from,to,normal);
        else normal=contactUnit(to.u*dot3(normal,from.u)+to.n*dot3(normal,from.n)+to.w*dot3(normal,from.w),normal);
        // Only retain while within the current finite tool. No infinite-plane
        // adhesion when the witness slides off a jaw edge or withdraws.
        // A material witness may slide past the finite face while another part
        // of its triangle is still inside the jaw. Dropping the incident side
        // based on that one witness lets a nearest-face query flip through a
        // thin blade. Retain a box contact using the whole finite triangle;
        // release still occurs when its finite overlap disappears.
        float finiteDistance=to.capsule?distance:triangleBoxGap(to,q[0],q[1],q[2]);
        // The inertial predictor may leave the finite face, then be rejected by
        // the material solve. Keep its entry side for the complete swept pair;
        // inspecting only the predicted endpoint discards the accepted contact
        // before CCD starts from that same touching configuration. At a corner
        // the fresh nearest-face query can then select opposite sides on
        // triangles sharing one node. The finite manifold still drops samples
        // outside the CURRENT footprint and positive gaps exert no force.
        float signedDistance=dot3(x-end,normal);
        if(!to.capsule)signedDistance=fminf(dot3(q[0]-end,normal),fminf(dot3(q[1]-end,normal),dot3(q[2]-end,normal)));
        bool startContact=false;
        if(!to.capsule){
            float startDistance=fminf(dot3(p[0]-start,fromNormal),fminf(dot3(p[1]-start,fromNormal),dot3(p[2]-start,fromNormal)));
            startContact=triangleBoxGap(from,p[0],p[1],p[2])<=2.f*skin && startDistance<=2.f*skin;
        }
        if(startContact || (finiteDistance<=2.f*skin && signedDistance<=2.f*skin)){
            // Keep the incident SIDE, not a stale barycentric witness. A triangle
            // can rotate about its first contact while another portion penetrates.
            retainedPlane.normal=normal;retainedPlane.valid=1;
            retainedPlane.target=end+normal*skin;retainedPlane.motion=end-start;
        }
    }
    SurfaceContact out={};out.kind=kind;
    float3 bary=mkf3(0,0,0);
    if(preload){bary=preloadBary;out.plane=preloadPlane;}
    else if(retainedPlane.valid && !to.capsule){
        bary=triangleWitness(to,q[0],q[1],q[2]);
        out.plane=retainedPlane;out.stick=retained.stick;
        float3 normal=out.plane.normal;
        // Supporting plane on the retained incident side, on the finite box.
        // Previous positions use the same refreshed material interpolation.
        float support=fabsf(dot3(normal,to.u))*to.half.x+
            fabsf(dot3(normal,to.n))*to.half.y+fabsf(dot3(normal,to.w))*to.half.z;
        float3 x=q[0]*bary.x+q[1]*bary.y+q[2]*bary.z;
        float3 surface=x+normal*(support-dot3(x-to.center,normal));
        out.plane.target=surface+normal*skin;
        out.plane.motion=surface-witnessAt(to,from,surface);
    }else if(to.capsule){
        bary=bladePhase?bladeTriangleWitness(to,q):triangleWitness(to,q[0],q[1],q[2]);
        out.plane=traceToolContact(from,to,skin,p[0]*bary.x+p[1]*bary.y+p[2]*bary.z,q[0]*bary.x+q[1]*bary.y+q[2]*bary.z);
        // The swept endpoint witness is not the first point on a face hit by
        // a grazing capsule. Its radial TOI normal can be almost tangential,
        // and persisting that normal artificially caps the preload. On the
        // interior of a triangle use its incident face normal; keep edge/vertex
        // contacts radial and retain the finite capsule footprint checks.
        if(bladePhase && (*bladePhase==0 || *bladePhase==2 || *bladePhase==4) && out.plane.valid &&
           fminf(bary.x,fminf(bary.y,bary.z))>1e-4f){
            float3 normal=cross3(q[1]-q[0],q[2]-q[0]);
            if(dot3(normal,normal)>1e-16f){
                normal=contactUnit(normal,out.plane.normal);
                float3 oldPoint=p[0]*bary.x+p[1]*bary.y+p[2]*bary.z;
                float3 oldAxis=from.b-from.a;
                float oldT=fminf(1.f,fmaxf(0.f,dot3(oldPoint-from.a,oldAxis)/fmaxf(dot3(oldAxis,oldAxis),1e-20f)));
                if(dot3(normal,oldPoint-(from.a+oldAxis*oldT))<0)normal=normal*(-1.f);
                float3 point=q[0]*bary.x+q[1]*bary.y+q[2]*bary.z,axis=to.b-to.a;
                float tLine=fminf(1.f,fmaxf(0.f,dot3(point-to.a,axis)/fmaxf(dot3(axis,axis),1e-20f)));
                float3 surface=to.a+axis*tLine+normal*to.radius;
                out.plane.normal=normal;out.plane.target=surface+normal*skin;
                out.plane.motion=surface-witnessAt(to,from,surface);
            }
        }
    }else out.plane=traceTriangleBox(from,to,skin,p,q,bary);
    float3 x=q[0]*bary.x+q[1]*bary.y+q[2]*bary.z;
    float3 old=p[0]*bary.x+p[1]*bary.y+p[2]*bary.z;
    if(!out.plane.valid)return;
    if(body && !to.capsule){
        // A finite face manifold must use an actual OBB face, not a corner SDF
        // diagonal extended into an artificial slanted supporting half-space.
        float3 axes[3]={to.u,to.n,to.w};int face=0;
        for(int a=1;a<3;a++)if(fabsf(dot3(out.plane.normal,axes[a]))>fabsf(dot3(out.plane.normal,axes[face])))face=a;
        float sign=dot3(out.plane.normal,axes[face])>=0?1.f:-1.f;
        out.plane.normal=axes[face]*sign;
        float3 surface=x+out.plane.normal*(compF3(to.half,face)-dot3(x-to.center,out.plane.normal));
        out.plane.target=surface+out.plane.normal*skin;
        out.plane.motion=surface-witnessAt(to,from,surface);
    }
    if(body && stationary){
        float3 surface=out.plane.target-out.plane.normal*skin;
        out.plane.motion=surface-witnessAt(to,bodyContactShape(body,body->previous,kind,thickness,minGap),surface);
    }
    SurfaceVertex v[3]={vertices[3*t],vertices[3*t+1],vertices[3*t+2]};
    out.previous=old;
    out.toolOrigin=body?body->pose.jawHinge:mkf3(0,0,0);
    out.referenceShape=to;out.toolGap=body?body->pose.jawGap:0;
    out.gapGradient=body?jawGapGradient(body->pose,kind,thickness,minGap,out.plane.target-out.plane.normal*skin,body):mkf3(0,0,0);
    out.incoming=dot3(x-old-out.plane.motion,out.plane.normal)<0;
    for(int k=0;k<3;k++){
        out.owner[k]=v[k].voxel;out.mask[k]=v[k].mask;out.component[k]=v[k].component;
        out.offset=out.offset+v[k].offset*compF3(bary,k);
        for(int j=0;j<v[k].count;j++){
            float w=compF3(bary,k)*v[k].weight[j];if(w<1e-8f)continue;
            int id=v[k].id[j],s=0;while(s<out.count && out.id[s]!=id)s++;
            if(s==out.count){out.id[s]=id;out.count++;}
            out.weight[s]+=w;
        }
    }
    for(int j=0;j<out.count;j++)atomicAdd(&degree[out.id[j]],1);
    contacts[slot]=out;
    if(out.count)active[atomicAdd(activeCount,1)]=slot;
}

// Separate scratch for legacy stream zero and the solver's one capture stream.
// Any additional stream falls back to the old uniform calculation, avoiding sharing.
static ContactShape* d_querySampleZero=nullptr;
static ContactShape* d_querySampleCapture=nullptr;
static cudaStream_t g_querySampleCaptureStream=nullptr;
static cudaError_t allocateQuerySample(){
    cudaError_t status=cudaMalloc(&d_querySampleZero,2*sizeof(ContactShape));
    if(status!=cudaSuccess)return status;
    status=cudaMalloc(&d_querySampleCapture,2*sizeof(ContactShape));
    if(status!=cudaSuccess){cudaFree(d_querySampleZero);d_querySampleZero=nullptr;}
    return status;
}
static void freeQuerySample(){
    cudaDeviceSynchronize();
    cudaFree(d_querySampleZero);d_querySampleZero=nullptr;
    cudaFree(d_querySampleCapture);d_querySampleCapture=nullptr;g_querySampleCaptureStream=nullptr;
}
__global__ void k_SampleSurfaceShapes(ContactShape* shapes,const CoupledTool* body,int kind,float thickness,float minGap,bool stationary){
    if(threadIdx.x || blockIdx.x)return;
    shapes[0]=bodyContactShape(body,stationary?body->pose:body->previous,kind,thickness,minGap);
    shapes[1]=bodyContactShape(body,body->pose,kind,thickness,minGap);
}
__global__ void k_FindSurface(int triangles,int kind,ContactShape from,ContactShape to,float skin,
    const SurfaceVertex* vertices,const unsigned* masks,const float3* prev,const float3* pos,
    SurfaceContact* contacts,int* degree,int* active,int* activeCount,const float4* current,const float3* previous,
    const CoupledTool* body=nullptr,float thickness=0,float minGap=0,bool stationary=false,
    const int* cacheSlots=nullptr,const SurfaceContact* cache=nullptr,const int* bladePhase=nullptr,int stride=3,bool cacheComponents=false,const unsigned char* groupRejected=nullptr,const ContactShape* sampledShapes=nullptr){
    int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=triangles)return;
    int slot=stride*t+kind;
    // A carried cutting preload may survive outside the current tool AABB.
    if(groupRejected && groupRejected[t/64] &&
       !(body && bladePhase && (*bladePhase==0 || *bladePhase==4) && contacts[slot].count>0)){
        contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;return;
    }
    if(body){if(sampledShapes){from=sampledShapes[0];to=sampledShapes[1];}else{from=bodyContactShape(body,stationary?body->pose:body->previous,kind,thickness,minGap);to=bodyContactShape(body,body->pose,kind,thickness,minGap);}}
    // Reject distant triangles before reading/probing the material-contact
    // cache or copying a 24-node witness. This is the dominant no-contact path.
    if(to.capsule?to.radius<=0:to.half.x<=0||to.half.z<=0){contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;return;}
    float3 q[3],p[3];
    for(int k=0;k<3;k++){
        float4 point=current[3*t+k];if(point.w==0){contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;return;}
        q[k]=mkf3(point.x,point.y,point.z);p[k]=previous[3*t+k];
    }
    // An unfractured surface is a unilateral barrier on its incident side.
    // Re-querying the closest side of a thin cylinder every substep caps the
    // preload at K*radius, then reverses the normal and lets the blade through.
    // Carry this material witness until fracture or separation, including cut
    // components. Rebind its interpolation below to the current reconstruction.
    bool preload=body && to.capsule && bladePhase && (*bladePhase==0 || *bladePhase==4) && contacts[slot].count>0;
    for(int k=0;k<3 && preload;k++)preload=contacts[slot].owner[k]==vertices[3*t+k].voxel &&
        contacts[slot].mask[k]==vertices[3*t+k].mask && contacts[slot].component[k]==vertices[3*t+k].component;
    ContactRecord preloadPlane={};float3 preloadBary=mkf3(0,0,0);
    if(preload){
        const SurfaceContact& old=contacts[slot];preloadBary=bladeTriangleWitness(to,q);
        float3 normal=rotateShaftVector(old.referenceShape,to,old.plane.normal);
        float3 end=witnessAt(old.referenceShape,to,old.plane.target-old.plane.normal*skin);
        float3 x=q[0]*preloadBary.x+q[1]*preloadBary.y+q[2]*preloadBary.z;
        // Project onto the incident plane and require the finite capsule's
        // footprint. This is not an infinite plane or an enlarged collision box.
        float gap=dot3(x-end,normal);float3 nn,ss;
        float lateral=shapeDistance(to,x-normal*(gap+to.radius),nn,ss);
        preload=gap<=0 && lateral<=fmaxf(1e-6f,1e-3f*to.radius);
        if(preload){preloadPlane.normal=normal;preloadPlane.target=end+normal*skin;
            preloadPlane.motion=end-witnessAt(to,from,end);preloadPlane.valid=1;}
    }
    float3 lo,hi,l0,h0;shapeBounds(to,skin,lo,hi);shapeBounds(from,skin,l0,h0);
    // Endpoint OBB bounds alone miss intermediate rotational sweeps.
    if(!to.capsule && (len3(to.u-from.u)+len3(to.n-from.n)>1e-5f)){
        float r=len3(to.half)+skin;float3 extent=mkf3(r,r,r);lo=to.center-extent;hi=to.center+extent;l0=from.center-extent;h0=from.center+extent;
    }
    for(int ax=0;ax<3;ax++){
        float mn=1e30f,mx=-1e30f;for(int k=0;k<3;k++){mn=fminf(mn,fminf(compF3(q[k],ax),compF3(p[k],ax)));mx=fmaxf(mx,fmaxf(compF3(q[k],ax),compF3(p[k],ax)));}
        if(!preload && (mx<fminf(compF3(lo,ax),compF3(l0,ax)) || mn>fmaxf(compF3(hi,ax),compF3(h0,ax)))){contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;return;}
    }
    // Persist a material witness while its triangle owners and cut masks match.
    // A nearest-face query inside a thin jaw can otherwise flip its normal and
    // push tissue through the opposite face on the next substep.
    // Retain the incident side only for the same voxel, cut mask AND component.
    // The actual material witness is rebuilt below from current vertex weights.
    SurfaceContact retained;bool keep=body && contacts[slot].count>0;
    for(int k=0;k<3 && keep;k++)keep=vertices[3*t+k].count>0 &&
        contacts[slot].component[k]==vertices[3*t+k].component &&
        contacts[slot].owner[k]==vertices[3*t+k].voxel && contacts[slot].mask[k]==vertices[3*t+k].mask;
    if(keep)retained=contacts[slot];
    if(!keep && body && cacheSlots && cache && cacheComponents){
        uint3 key[3];bool valid=true;
        for(int k=0;k<3;k++){const SurfaceVertex& v=vertices[3*t+k];valid=valid && v.count>0;key[k]=make_uint3(v.voxel,v.mask,v.component+1);}
        if(valid){
            sortMaterialKey(key);unsigned hash=materialContactHash(key,kind);
            for(int probe=0;probe<CONTACT_CACHE_SIZE;probe++){
                unsigned slot=(hash+probe)&(CONTACT_CACHE_SIZE-1);if(cacheSlots[slot]<0)break;
                if(sameMaterialContactKey(cache[slot],key,kind)){retained=cache[slot];keep=true;break;}
            }
        }
    }
    if(!keep && body && cacheSlots && cache && !cacheComponents){
        int key[3];bool intact=true;
        for(int k=0;k<3;k++){const SurfaceVertex& v=vertices[3*t+k];key[k]=v.voxel;intact=intact && v.count>0 && v.component<0 && v.mask==0;}
        if(intact){
            contactKey(key);unsigned h=contactHash(key,kind);
            for(int probe=0;probe<CONTACT_CACHE_SIZE;probe++){
                unsigned s=(h+probe)&(CONTACT_CACHE_SIZE-1);if(cacheSlots[s]<0)break;
                if(sameContactKey(cache[s],key,kind)){
                    retained=cache[s];keep=true;
                    for(int k=0;k<3;k++){retained.owner[k]=vertices[3*t+k].voxel;retained.mask[k]=vertices[3*t+k].mask;}
                    break;
                }
            }
        }
    }
    // Rejected triangles have no live contact. Do not clear the full 24-node
    // stencil for every triangle, jaw and substep: only the validity fields.
    contacts[slot].count=0;contacts[slot].lambdaN=0;contacts[slot].kind=kind;
    ContactRecord retainedPlane={};
    if(keep){
        float3 x=retained.offset,old=x;
        for(int j=0;j<retained.count;j++){x=x+pos[retained.id[j]]*retained.weight[j];old=old+prev[retained.id[j]]*retained.weight[j];}
        float3 normal=retained.plane.normal;
        float3 start=witnessAt(retained.referenceShape,from,retained.plane.target-normal*skin);
        if(!to.capsule)normal=contactUnit(from.u*dot3(normal,retained.referenceShape.u)+from.n*dot3(normal,retained.referenceShape.n)+from.w*dot3(normal,retained.referenceShape.w),normal);
        float3 fromNormal=normal;
        float3 end=witnessAt(from,to,start),nn,ss;
        float distance=shapeDistance(to,x,nn,ss);
        if(to.capsule)normal=rotateShaftVector(from,to,normal);
        else normal=contactUnit(to.u*dot3(normal,from.u)+to.n*dot3(normal,from.n)+to.w*dot3(normal,from.w),normal);
        // Only retain while within the current finite tool. No infinite-plane
        // adhesion when the witness slides off a jaw edge or withdraws.
        // A material witness may slide past the finite face while another part
        // of its triangle is still inside the jaw. Dropping the incident side
        // based on that one witness lets a nearest-face query flip through a
        // thin blade. Retain a box contact using the whole finite triangle;
        // release still occurs when its finite overlap disappears.
        float finiteDistance=to.capsule?distance:triangleBoxGap(to,q[0],q[1],q[2]);
        // The inertial predictor may leave the finite face, then be rejected by
        // the material solve. Keep its entry side for the complete swept pair;
        // inspecting only the predicted endpoint discards the accepted contact
        // before CCD starts from that same touching configuration. At a corner
        // the fresh nearest-face query can then select opposite sides on
        // triangles sharing one node. The finite manifold still drops samples
        // outside the CURRENT footprint and positive gaps exert no force.
        float signedDistance=dot3(x-end,normal);
        if(!to.capsule)signedDistance=fminf(dot3(q[0]-end,normal),fminf(dot3(q[1]-end,normal),dot3(q[2]-end,normal)));
        bool startContact=false;
        if(!to.capsule){
            float startDistance=fminf(dot3(p[0]-start,fromNormal),fminf(dot3(p[1]-start,fromNormal),dot3(p[2]-start,fromNormal)));
            startContact=triangleBoxGap(from,p[0],p[1],p[2])<=2.f*skin && startDistance<=2.f*skin;
        }
        if(startContact || (finiteDistance<=2.f*skin && signedDistance<=2.f*skin)){
            // Keep the incident SIDE, not a stale barycentric witness. A triangle
            // can rotate about its first contact while another portion penetrates.
            retainedPlane.normal=normal;retainedPlane.valid=1;
            retainedPlane.target=end+normal*skin;retainedPlane.motion=end-start;
        }
    }
    SurfaceContact out={};out.kind=kind;
    float3 bary=mkf3(0,0,0);
    if(preload){bary=preloadBary;out.plane=preloadPlane;}
    else if(retainedPlane.valid && !to.capsule){
        bary=triangleWitness(to,q[0],q[1],q[2]);
        out.plane=retainedPlane;out.stick=retained.stick;
        float3 normal=out.plane.normal;
        // Supporting plane on the retained incident side, on the finite box.
        // Previous positions use the same refreshed material interpolation.
        float support=fabsf(dot3(normal,to.u))*to.half.x+
            fabsf(dot3(normal,to.n))*to.half.y+fabsf(dot3(normal,to.w))*to.half.z;
        float3 x=q[0]*bary.x+q[1]*bary.y+q[2]*bary.z;
        float3 surface=x+normal*(support-dot3(x-to.center,normal));
        out.plane.target=surface+normal*skin;
        out.plane.motion=surface-witnessAt(to,from,surface);
    }else if(to.capsule){
        bary=bladePhase?bladeTriangleWitness(to,q):triangleWitness(to,q[0],q[1],q[2]);
        out.plane=traceToolContact(from,to,skin,p[0]*bary.x+p[1]*bary.y+p[2]*bary.z,q[0]*bary.x+q[1]*bary.y+q[2]*bary.z);
        // The swept endpoint witness is not the first point on a face hit by
        // a grazing capsule. Its radial TOI normal can be almost tangential,
        // and persisting that normal artificially caps the preload. On the
        // interior of a triangle use its incident face normal; keep edge/vertex
        // contacts radial and retain the finite capsule footprint checks.
        if(bladePhase && (*bladePhase==0 || *bladePhase==2 || *bladePhase==4) && out.plane.valid &&
           fminf(bary.x,fminf(bary.y,bary.z))>1e-4f){
            float3 normal=cross3(q[1]-q[0],q[2]-q[0]);
            if(dot3(normal,normal)>1e-16f){
                normal=contactUnit(normal,out.plane.normal);
                float3 oldPoint=p[0]*bary.x+p[1]*bary.y+p[2]*bary.z;
                float3 oldAxis=from.b-from.a;
                float oldT=fminf(1.f,fmaxf(0.f,dot3(oldPoint-from.a,oldAxis)/fmaxf(dot3(oldAxis,oldAxis),1e-20f)));
                if(dot3(normal,oldPoint-(from.a+oldAxis*oldT))<0)normal=normal*(-1.f);
                float3 point=q[0]*bary.x+q[1]*bary.y+q[2]*bary.z,axis=to.b-to.a;
                float tLine=fminf(1.f,fmaxf(0.f,dot3(point-to.a,axis)/fmaxf(dot3(axis,axis),1e-20f)));
                float3 surface=to.a+axis*tLine+normal*to.radius;
                out.plane.normal=normal;out.plane.target=surface+normal*skin;
                out.plane.motion=surface-witnessAt(to,from,surface);
            }
        }
    }else out.plane=traceTriangleBox(from,to,skin,p,q,bary);
    float3 x=q[0]*bary.x+q[1]*bary.y+q[2]*bary.z;
    float3 old=p[0]*bary.x+p[1]*bary.y+p[2]*bary.z;
    if(!out.plane.valid)return;
    if(body && !to.capsule){
        // A finite face manifold must use an actual OBB face, not a corner SDF
        // diagonal extended into an artificial slanted supporting half-space.
        float3 axes[3]={to.u,to.n,to.w};int face=0;
        for(int a=1;a<3;a++)if(fabsf(dot3(out.plane.normal,axes[a]))>fabsf(dot3(out.plane.normal,axes[face])))face=a;
        float sign=dot3(out.plane.normal,axes[face])>=0?1.f:-1.f;
        out.plane.normal=axes[face]*sign;
        float3 surface=x+out.plane.normal*(compF3(to.half,face)-dot3(x-to.center,out.plane.normal));
        out.plane.target=surface+out.plane.normal*skin;
        out.plane.motion=surface-witnessAt(to,from,surface);
    }
    if(body && stationary){
        float3 surface=out.plane.target-out.plane.normal*skin;
        out.plane.motion=surface-witnessAt(to,bodyContactShape(body,body->previous,kind,thickness,minGap),surface);
    }
    SurfaceVertex v[3]={vertices[3*t],vertices[3*t+1],vertices[3*t+2]};
    out.previous=old;
    out.toolOrigin=body?body->pose.jawHinge:mkf3(0,0,0);
    out.referenceShape=to;out.toolGap=body?body->pose.jawGap:0;
    out.gapGradient=body?jawGapGradient(body->pose,kind,thickness,minGap,out.plane.target-out.plane.normal*skin,body):mkf3(0,0,0);
    out.incoming=dot3(x-old-out.plane.motion,out.plane.normal)<0;
    for(int k=0;k<3;k++){
        out.owner[k]=v[k].voxel;out.mask[k]=v[k].mask;out.component[k]=v[k].component;
        out.offset=out.offset+v[k].offset*compF3(bary,k);
        for(int j=0;j<v[k].count;j++){
            float w=compF3(bary,k)*v[k].weight[j];if(w<1e-8f)continue;
            int id=v[k].id[j],s=0;while(s<out.count && out.id[s]!=id)s++;
            if(s==out.count){out.id[s]=id;out.count++;}
            out.weight[s]+=w;
        }
    }
    for(int j=0;j<out.count;j++)atomicAdd(&degree[out.id[j]],1);
    contacts[slot]=out;
    if(out.count)active[atomicAdd(activeCount,1)]=slot;
}
// Diagnostic-only paired execution on identical inputs. Graph capture is never
// altered by host readback; validation only inspects ordinary launch sites.
static bool g_findValidation=false;
static int g_findValidationStats[4]={}; // paired calls, failures, active contacts, capture skips
extern "C" int xpbd_find_surface_validation_nodes();
extern "C" void xpbd_find_surface_validate(int enabled){
    g_findValidation=enabled!=0;
    if(enabled)for(int& x:g_findValidationStats)x=0;
}
extern "C" void xpbd_find_surface_validation_read(int* values){
    for(int k=0;k<4;k++)values[k]=g_findValidationStats[k];
}
static inline void launchFindSurface(int triangles,int kind,cudaStream_t stream,
    ContactShape from,ContactShape to,float skin,const SurfaceVertex* vertices,const unsigned* masks,
    const float3* prev,const float3* pos,SurfaceContact* contacts,int* degree,int* active,int* activeCount,
    const float4* current,const float3* previous,const CoupledTool* body=nullptr,float thickness=0,
    float minGap=0,bool stationary=false,const int* cacheSlots=nullptr,const SurfaceContact* cache=nullptr,
    const int* bladePhase=nullptr,int stride=3,bool cacheComponents=false,const unsigned char* groupRejected=nullptr,bool knownScissors=false){
#define FIND_ARGUMENTS triangles,kind,from,to,skin,vertices,masks,prev,pos,contacts,degree,active,activeCount,current,previous,body,thickness,minGap,stationary,cacheSlots,cache,bladePhase,stride,cacheComponents,groupRejected
    const int blocks=(triangles+63)/64;
    auto specialized=[&](){
        ContactShape* shapes=stream?d_querySampleCapture:d_querySampleZero;
        if(stream){
            if(!g_querySampleCaptureStream)g_querySampleCaptureStream=stream;
            else if(g_querySampleCaptureStream!=stream)shapes=nullptr;
        }
        if(body && shapes)k_SampleSurfaceShapes<<<1,32,0,stream>>>(shapes,body,kind,thickness,minGap,stationary);
        else shapes=nullptr;
        k_FindSurface<<<blocks,64,0,stream>>>(FIND_ARGUMENTS,shapes);
    };
    if(!g_findValidation){specialized();return;}
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    cudaError_t status=cudaStreamIsCapturing(stream,&capture);
    if(status!=cudaSuccess || capture!=cudaStreamCaptureStatusNone){g_findValidationStats[3]++;specialized();return;}
    const int entries=triangles*stride,nodes=xpbd_find_surface_validation_nodes();
    std::vector<SurfaceContact> before(entries),reference(entries),actual(entries);
    std::vector<int> degreeBefore(nodes),degreeRef(nodes),degreeActual(nodes),activeBefore(entries),activeRef(entries),activeActual(entries);
    int countBefore=0,countRef=0,countActual=0;
    auto read=[&](std::vector<SurfaceContact>& c,std::vector<int>& d,std::vector<int>& a,int& count){
        if(cudaStreamSynchronize(stream)!=cudaSuccess)return false;
        return cudaMemcpy(c.data(),contacts,entries*sizeof(SurfaceContact),cudaMemcpyDeviceToHost)==cudaSuccess &&
            cudaMemcpy(d.data(),degree,nodes*sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess &&
            cudaMemcpy(a.data(),active,entries*sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess &&
            cudaMemcpy(&count,activeCount,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess;
    };
    if(!read(before,degreeBefore,activeBefore,countBefore)){g_findValidationStats[1]++;specialized();return;}
    k_FindSurfaceReference<<<blocks,64,0,stream>>>(FIND_ARGUMENTS);
    bool ok=read(reference,degreeRef,activeRef,countRef);
    ok=ok && cudaMemcpy(contacts,before.data(),entries*sizeof(SurfaceContact),cudaMemcpyHostToDevice)==cudaSuccess;
    ok=ok && cudaMemcpy(degree,degreeBefore.data(),nodes*sizeof(int),cudaMemcpyHostToDevice)==cudaSuccess;
    ok=ok && cudaMemcpy(active,activeBefore.data(),entries*sizeof(int),cudaMemcpyHostToDevice)==cudaSuccess;
    ok=ok && cudaMemcpy(activeCount,&countBefore,sizeof(int),cudaMemcpyHostToDevice)==cudaSuccess;
    specialized();ok=read(actual,degreeActual,activeActual,countActual) && ok;
    int positive=0;
    for(int i=0;i<entries;i++){
        const SurfaceContact& a=reference[i];const SurfaceContact& b=actual[i];
        if(a.count!=b.count || a.kind!=b.kind || a.lambdaN!=b.lambdaN)ok=false;
        if(a.count>0){positive++;if(std::memcmp(&a,&b,sizeof(SurfaceContact)))ok=false;}
    }
    if(degreeRef!=degreeActual || countRef!=countActual || countRef<0 || countRef>entries || countActual<0 || countActual>entries)ok=false;
    else{
        activeRef.resize(countRef);activeActual.resize(countActual);
        std::sort(activeRef.begin(),activeRef.end());std::sort(activeActual.begin(),activeActual.end());
        if(activeRef!=activeActual)ok=false;
    }
    g_findValidationStats[0]++;g_findValidationStats[2]+=positive;
    if(!ok)g_findValidationStats[1]++;
#undef FIND_ARGUMENTS
}
// Bounded grid-stride launch consumes every active contact without a CPU count readback.
__global__ void k_SolveSurface(int count,float alpha,float muS,float muD,const float* invMass,
    const float3* pos,const int* degree,SurfaceContact* contacts,float3* delta,int* graspFlags,int closing,const int* active,const int* activeCount,CoupledTool* body=nullptr){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<*activeCount;i+=blockDim.x*gridDim.x){
        SurfaceContact& c=contacts[active[i]];if(!c.count)continue;
        float3 x=c.offset;float denom=0,tangentDenom=0;
        // Mass-split Jacobi denominator bounds simultaneous updates on shared material nodes.
        // Multipliers and positions receive the SAME increment, preserving the friction cone.
        for(int j=0;j<c.count;j++){int id=c.id[j];float w=c.weight[j],scale=w*w*fmaxf(1.f,(float)degree[id]);x=x+pos[id]*w;
            denom+=dot3(c.plane.normal,contactMobility(body,id,invMass[id],c.plane.normal))*scale;tangentDenom+=invMass[id]*scale;}
        float3 shift=body?body->pose.jawHinge-c.toolOrigin+c.gapGradient*(body->pose.jawGap-c.toolGap):mkf3(0,0,0);
        if(body){float nGap=dot3(c.gapGradient,c.plane.normal),count=fmaxf(1.f,(float)*activeCount);
            denom+=(body->inverseMass+body->gapInverseMass*nGap*nGap)*count;
            tangentDenom+=(body->inverseMass+body->gapInverseMass*dot3(c.gapGradient,c.gapGradient))*count;}
        if(denom<=0)continue;
        float C=dot3(x-c.plane.target-shift,c.plane.normal),old=c.lambdaN;
        c.lambdaN=fmaxf(0.f,old+(-C-alpha*old)/(denom+alpha));
        float3 slip=x-c.previous-c.plane.motion-shift;slip=slip-c.plane.normal*dot3(slip,c.plane.normal);
        float3 oldT=c.lambdaT,trial=oldT-slip*(1.f/fmaxf(tangentDenom,1e-20f));float l=len3(trial);
        c.lambdaT=l>muS*c.lambdaN?trial*(muD*c.lambdaN/fmaxf(l,1e-20f)):trial;
        float3 impulse=c.plane.normal*(c.lambdaN-old)+(c.lambdaT-oldT);
        if(body){add3(&body->delta,impulse*(-body->inverseMass));atomicAdd(&body->gapDelta,-body->gapInverseMass*dot3(c.gapGradient,impulse));}
        for(int j=0;j<c.count;j++){
            int id=c.id[j];add3(&delta[id],contactMobility(body,id,invMass[id],impulse)*c.weight[j]);
            if(closing && c.kind>0 && c.lambdaN>0)atomicExch(&graspFlags[id],1);
        }
    }
}
__global__ void k_ApplySurface(int N,float3* pos,const float3* delta){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)pos[i]=pos[i]+delta[i];
}
// The instrument couples every active contact. Solve that small island on the
// GPU in Gauss-Seidel order instead of splitting its mass by contact count.
// Material remains parallel. This removes an artificial iteration-dependent
// tool inertia and immediately exposes each contact's displacement to its peers.
__device__ inline float warpSum(float x){for(int d=16;d>0;d/=2)x+=__shfl_down_sync(0xffffffff,x,d);return x;}
__global__ void k_SolveContactIsland(int iterations,float alpha,float muS,float muD,
    const float* invMass,float3* pos,SurfaceContact* contacts,const int* active,const int* activeCount,CoupledTool* body){
    int lane=threadIdx.x;__shared__ float3 impulse;
    for(int it=0;it<iterations;it++)for(int j=0;j<*activeCount;j++){
        SurfaceContact& c=contacts[active[(it&1)?*activeCount-1-j:j]];if(!c.count)continue;
        int id=lane<c.count?c.id[lane]:0;float w=lane<c.count?c.weight[lane]:0;
        float3 xp=pos[id]*w;
        float mass=w*w*dot3(c.plane.normal,contactMobility(body,id,invMass[id],c.plane.normal));
        xp.x=warpSum(xp.x);xp.y=warpSum(xp.y);xp.z=warpSum(xp.z);mass=warpSum(mass);
        if(lane==0){
            float3 x=c.offset+xp,shift=body->pose.jawHinge-c.toolOrigin+c.gapGradient*(body->pose.jawGap-c.toolGap);
            float ng=dot3(c.gapGradient,c.plane.normal),den=mass+body->inverseMass+body->gapInverseMass*ng*ng;
            float old=c.lambdaN,C=dot3(x-c.plane.target-shift,c.plane.normal);
            c.lambdaN=fmaxf(0.f,old+(-C-alpha*old)/fmaxf(den+alpha,1e-20f));
            float3 slip=x-c.previous-c.plane.motion-shift;slip=slip-c.plane.normal*dot3(slip,c.plane.normal);
            float tangentMass=mass+body->inverseMass+body->gapInverseMass*fmaxf(0.f,dot3(c.gapGradient,c.gapGradient)-ng*ng);
            float3 oldT=c.lambdaT,trial=oldT-slip*(1.f/fmaxf(tangentMass,1e-20f));float l=len3(trial);
            c.lambdaT=l>muS*c.lambdaN?trial*(muD*c.lambdaN/fmaxf(l,1e-20f)):trial;
            impulse=c.plane.normal*(c.lambdaN-old)+(c.lambdaT-oldT);
            body->pose=shiftTool(body->pose,impulse*(-body->inverseMass));
            body->pose.jawGap=fminf(body->maxGap,fmaxf(0.f,body->pose.jawGap-body->gapInverseMass*dot3(c.gapGradient,impulse)));
        }
        __syncwarp();
        if(lane<c.count)pos[id]=pos[id]+contactMobility(body,id,invMass[id],impulse)*w;
        __syncwarp();
    }
}
// Re-linearize the active material witness after other constraints have moved
// its triangle. Retaining one barycentric point through every nonlinear solve
// allows the triangle to pivot into a jaw around that satisfied point.
__global__ void k_RefreshActiveWitness(const int* active,const int* count,SurfaceContact* contacts,
    const SurfaceVertex* vertices,const float3* pos,const float3* previous,CoupledTool* body,float thickness,float minGap,float skin){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=*count)return;
    int slot=active[i];SurfaceContact& c=contacts[slot];if(!c.count || c.kind==0)return;
    ContactShape shape=bodyContactShape(body,body->pose,c.kind,thickness,minGap);
    float3 q[3];const SurfaceVertex* v=vertices+3*(slot/3);
    for(int j=0;j<3;j++)q[j]=surfacePoint(v[j],pos);
    float3 normal=contactUnit(shape.u*dot3(c.plane.normal,c.referenceShape.u)+
        shape.n*dot3(c.plane.normal,c.referenceShape.n)+shape.w*dot3(c.plane.normal,c.referenceShape.w),c.plane.normal);
    // Restrict to the finite face footprint, then minimize its incident plane.
    ContactShape footprint=shape;float3 axes[3]={shape.u,shape.n,shape.w};int ax=0;
    if(fabsf(dot3(normal,axes[1]))>fabsf(dot3(normal,axes[ax])))ax=1;
    if(fabsf(dot3(normal,axes[2]))>fabsf(dot3(normal,axes[ax])))ax=2;
    float extent=0;for(int j=0;j<3;j++)extent=fmaxf(extent,fabsf(dot3(q[j]-shape.center,axes[ax])));
    if(ax==0)footprint.half.x=extent+1;else if(ax==1)footprint.half.y=extent+1;else footprint.half.z=extent+1;
    float3 bar;if(!clippedBoxWitness(footprint,q,skin,bar))return;
    // Start with the actual deepest SDF witness; the footprint witness handles
    // an already crossed incident face without selecting the opposite normal.
    float3 deep=triangleWitness(shape,q[0],q[1],q[2]);
    float3 dp=q[0]*deep.x+q[1]*deep.y+q[2]*deep.z,nn,ss;
    if(shapeDistance(shape,dp,nn,ss)<=skin)bar=deep;
    c.count=0;c.offset=c.previous=mkf3(0,0,0);
    for(int j=0;j<3;j++){
        float factor=compF3(bar,j);c.offset=c.offset+v[j].offset*factor;
        c.previous=c.previous+surfacePoint(v[j],previous)*factor;
        for(int k=0;k<v[j].count;k++){
            float w=factor*v[j].weight[k];if(w<1e-8f)continue;
            int id=v[j].id[k],index=0;while(index<c.count && c.id[index]!=id)index++;
            if(index==c.count){c.id[index]=id;c.weight[index]=0;c.count++;}c.weight[index]+=w;
        }
    }
    float support=fabsf(dot3(normal,shape.u))*shape.half.x+fabsf(dot3(normal,shape.n))*shape.half.y+fabsf(dot3(normal,shape.w))*shape.half.z;
    float3 x=q[0]*bar.x+q[1]*bar.y+q[2]*bar.z;
    float3 surface=x+normal*(support-dot3(x-shape.center,normal));
    c.plane.normal=normal;c.plane.target=surface+normal*skin;
    c.plane.motion=surface-witnessAt(shape,bodyContactShape(body,body->previous,c.kind,thickness,minGap),surface);
    c.referenceShape=shape;c.toolOrigin=body->pose.jawHinge;c.toolGap=body->pose.jawGap;
    c.gapGradient=jawGapGradient(body->pose,c.kind,thickness,minGap,surface,body);
}
// Inelastic normal response. Position projection alone feeds its correction into
// next-step velocity and repeatedly bounces resting tissue off a stationary jaw.
// Tangential motion remains governed by the Coulomb constraint above.
__global__ void k_SurfaceVelocity(int count,float invH,const float* invMass,const int* degree,
    const SurfaceContact* contacts,const float3* velocity,float3* delta,const int* active,const int* activeCount,CoupledTool* body=nullptr){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<*activeCount;i+=blockDim.x*gridDim.x){
        const SurfaceContact& c=contacts[active[i]];if(!c.count || c.lambdaN<=0)continue;
        float3 v=mkf3(0,0,0);float denom=0;
        for(int j=0;j<c.count;j++){int id=c.id[j];float w=c.weight[j];v=v+velocity[id]*w;denom+=invMass[id]*w*w*fmaxf(1.f,(float)degree[id]);}
        float3 toolVelocity=c.plane.motion*invH;
        float gapNormal=dot3(c.gapGradient,c.plane.normal);
        if(body){
            toolVelocity=body->velocity+c.gapGradient*body->gapVelocity;
            denom+=(body->inverseMass+body->gapInverseMass*gapNormal*gapNormal)*fmaxf(1.f,(float)*activeCount);
        }
        if(denom<=0)continue;
        // Use physical tool velocity, not a zero-time reconstruction correction
        // divided by h. Apply the same inelastic impulse to both bodies.
        float vn=dot3(v-toolVelocity,c.plane.normal);
        // Restitution zero for an incident contact. Preserve free separating
        // motion, but don't turn position correction into an outward rebound.
        if(!c.incoming)vn=fminf(0.f,vn);
        if(body){add3(&body->delta,c.plane.normal*(vn/denom*body->inverseMass));atomicAdd(&body->gapDelta,vn/denom*body->gapInverseMass*gapNormal);}
        for(int j=0;j<c.count;j++){int id=c.id[j];add3(delta+id,c.plane.normal*(-vn/denom*invMass[id]*c.weight[j]));}
    }
}
__global__ void k_SurfaceDepth(int triangles,ContactShape shape,const float3* positions,float* result,
    const CoupledTool* body=nullptr,int kind=0,float thickness=0,float minGap=0){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=triangles)return;
    if(body){
        if(!body->initialized)return;
        shape=bodyContactShape(body,body->pose,kind,thickness,minGap);
    }
    if(shape.capsule?shape.radius<=0:shape.half.x<=0 || shape.half.z<=0)return;
    float3 a=positions[3*i],b=positions[3*i+1],c=positions[3*i+2],lo,hi;
    shapeBounds(shape,0,lo,hi);
    for(int ax=0;ax<3;ax++)if(fmaxf(compF3(a,ax),fmaxf(compF3(b,ax),compF3(c,ax)))<compF3(lo,ax) || fminf(compF3(a,ax),fminf(compF3(b,ax),compF3(c,ax)))>compF3(hi,ax))return;
    float3 bar=triangleWitness(shape,a,b,c),n,s;
    float depth=fmaxf(0.f,-shapeDistance(shape,a*bar.x+b*bar.y+c*bar.z,n,s));
    atomicMax((unsigned*)result,__float_as_uint(depth));
}
__global__ void k_TouchedByJaws(int count,const SurfaceContact* contacts,const SurfaceVertex* vertices,const unsigned* masks,float* touched){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const SurfaceContact& c=contacts[i];if(c.kind==0 || c.lambdaN<=0)return;
    for(int k=0;k<3;k++){const SurfaceVertex& v=vertices[3*(i/3)+k];if(masks && (masks[v.voxel]&4095u)!=v.mask)return;}
    for(int j=0;j<c.count;j++)if(c.weight[j]>1e-6f)atomicExch(&touched[c.id[j]],1.f);
}
__global__ void k_KeepTouched(int n,int* candidates,const float* touched){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && touched[i]==0)candidates[i]=0;
}
__global__ void k_AnchorReaction(int n,float scale,const int* core,const float3* lambda,float4* result){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || !core[i])return;
    float3 f=lambda[i]*scale;atomicAdd(&result->x,f.x);atomicAdd(&result->y,f.y);atomicAdd(&result->z,f.z);
}
__global__ void k_SurfaceReaction(int count,float scale,const SurfaceContact* contacts,float4* result,float4* jawLoad,ContactShape upper,ContactShape lower){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const SurfaceContact& c=contacts[i];if(!c.count || c.lambdaN<=0)return;
    float3 f=(c.plane.normal*c.lambdaN+c.lambdaT)*(-scale);
    atomicAdd(&result->x,f.x);atomicAdd(&result->y,f.y);atomicAdd(&result->z,f.z);atomicAdd(&result->w,1.f);
    // Only the inward jaw faces resist closure. Tip/shaft pressing must not
    // prematurely stop an otherwise open gripper before tissue enters its gap.
    if(c.kind>0){
        float inward=c.kind==1?-dot3(c.plane.normal,upper.n):dot3(c.plane.normal,lower.n);
        float load=fmaxf(0.f,inward)*c.lambdaN*scale;
        if(c.kind==1){atomicAdd(&jawLoad->x,load);if(load>0)atomicAdd(&jawLoad->z,1.f);}
        else {atomicAdd(&jawLoad->y,load);if(load>0)atomicAdd(&jawLoad->w,1.f);}
    }
}
__global__ void k_NodeReaction(int N,float scale,const ContactRecord* c,const float* ln,const float3* lt,float4* result){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=3*N || !c[i].valid || ln[i]<=0)return;
    float3 f=(c[i].normal*ln[i]+lt[i])*(-scale);
    atomicAdd(&result->x,f.x);atomicAdd(&result->y,f.y);atomicAdd(&result->z,f.z);atomicAdd(&result->w,1.f);
}

// Candidate-only root-cause diagnostic: deepest render triangle versus its
// actual material constraint after the accepted step. No state mutation.
__global__ void k_ContactDiagnostic(int triangles,int kind,ContactShape shape,const float3* expanded,
    const float3* pos,const SurfaceContact* contacts,const CoupledTool* body,float maximum,float* out,
    const int* tags,const int* active,const int* activeCount){
    int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=triangles)return;
    float3 a=expanded[3*t],b=expanded[3*t+1],c=expanded[3*t+2];
    float3 bar=triangleWitness(shape,a,b,c),n,s;float3 witness=a*bar.x+b*bar.y+c*bar.z;
    float depth=-shapeDistance(shape,witness,n,s);if(depth<maximum-1e-6f)return;
    if(atomicCAS((int*)out,0,1)!=0)return;
    int key[3];for(int j=0;j<3;j++)key[j]=tags[3*t+j]&0x3fffffff;contactKey(key);
    int found=-1;for(int j=0;j<*activeCount;j++)if(sameContactKey(contacts[active[j]],key,kind)){found=active[j];break;}
    SurfaceContact empty={};const SurfaceContact& contact=found>=0?contacts[found]:empty;float3 x=contact.offset;
    for(int k=0;k<contact.count;k++)x=x+pos[contact.id[k]]*contact.weight[k];
    float3 shift=body?body->pose.jawHinge-contact.toolOrigin+contact.gapGradient*(body->pose.jawGap-contact.toolGap):mkf3(0,0,0);
    out[1]=kind;out[2]=t;out[3]=contact.count;out[4]=contact.lambdaN;
    out[5]=dot3(x-contact.plane.target-shift,contact.plane.normal);
    out[6]=len3(witness-x);out[7]=dot3(n,contact.plane.normal);out[8]=depth;
}

__global__ void k_ContactResidual(const float3* pos,const SurfaceContact* contacts,const int* active,const int* count,const CoupledTool* body,float* maximum){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=*count)return;
    const SurfaceContact& c=contacts[active[i]];float3 x=c.offset;
    for(int k=0;k<c.count;k++)x=x+pos[c.id[k]]*c.weight[k];
    float3 shift=body->pose.jawHinge-c.toolOrigin+c.gapGradient*(body->pose.jawGap-c.toolGap);
    float violation=fmaxf(0.f,-dot3(x-c.plane.target-shift,c.plane.normal));
    atomicMax((unsigned*)maximum,__float_as_uint(violation));
}
