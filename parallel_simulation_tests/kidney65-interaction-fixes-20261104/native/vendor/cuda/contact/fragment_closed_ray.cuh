// Closed oriented 2-cycles admit signed ray degree. Exact binding/edge
// certificates are built on GPU; the probe cross-checks them independently on CPU.
// Ambiguous projected edges/origin crossings retain the original solid angle.
__device__ int g_fragmentClosedCertificate=0;
#ifdef ENDO_KIDNEY_CLOSED_STATS
__device__ unsigned long long g_fragmentRuntimeStats[6]={};
#endif
__device__ unsigned long long g_fragmentClosedStats[4]={};
__device__ inline bool fragmentClosedRayBox(const FragmentTreeNode& box,float3 origin,float3 direction){
    double near=0,far=1e30;
    for(int k=0;k<3;k++){
        double x=compF3(origin,k),d=compF3(direction,k);
        double a=((double)compF3(box.low,k)-1e-6-x)/d;
        double b=((double)compF3(box.high,k)+1e-6-x)/d;
        near=fmax(near,fmin(a,b));far=fmin(far,fmax(a,b));
        if(near>far)return false;
    }
    return true;
}
__device__ bool fragmentClosedRay(int triangles,const FragmentVertex* points,FragmentVertex p,
    const FragmentTreeNode* tree,int base,const int* order,int& winding){
    const int lane=threadIdx.x%32;
    const float3 direction=mkf3(.317596f,1.f,.173205f);
    __shared__ int traversal[2][32];int* stack=traversal[threadIdx.x/32];
    int top=1,sum=0;bool ambiguous=false;
    if(!lane)stack[0]=1;
    while(top){
        __syncwarp();int node=stack[--top];__syncwarp();
        const auto& box=tree[node];
        if(box.root==-1 || box.root==p.root || !fragmentClosedRayBox(box,p.current,direction))continue;
        if(node<base){if(!lane){stack[top]=2*node+1;stack[top+1]=2*node;}top+=2;continue;}
        int index=32*(node-base)+lane;if(index>=triangles)continue;int t=order[index];
        const auto& a=points[3*t];const auto& b=points[3*t+1];const auto& c=points[3*t+2];
        if(a.root<0 || a.root==p.root || a.root!=b.root || a.root!=c.root)continue;
        double ay=(double)a.current.y-p.current.y,by=(double)b.current.y-p.current.y,cy=(double)c.current.y-p.current.y;
        double ax=((double)a.current.x-p.current.x)-(double)direction.x*ay;
        double az=((double)a.current.z-p.current.z)-(double)direction.z*ay;
        double bx=((double)b.current.x-p.current.x)-(double)direction.x*by;
        double bz=((double)b.current.z-p.current.z)-(double)direction.z*by;
        double cx=((double)c.current.x-p.current.x)-(double)direction.x*cy;
        double cz=((double)c.current.z-p.current.z)-(double)direction.z*cy;
        double u=cx*bz-cz*bx,v=ax*cz-az*cx,w=bx*az-bz*ax;
        double scale=fmax(fmax(fabs(ax)+fabs(az),fabs(bx)+fabs(bz)),fabs(cx)+fabs(cz));
        scale=fmax(scale,fmax(fabs(ay),fmax(fabs(by),fabs(cy))));
        double error=1e-12*scale*scale+1e-28;
        bool negative=u<-error || v<-error || w<-error;
        bool positive=u>error || v>error || w>error;
        if(negative && positive)continue;
        if(fabs(u)<=error || fabs(v)<=error || fabs(w)<=error){ambiguous=true;continue;}
        double determinant=u+v+w,time=(u*ay+v*by+w*cy)/determinant;
        double tolerance=1e-6*(1+fabs(ay)+fabs(by)+fabs(cy));
        if(time<-tolerance)continue;
        if(time<=tolerance){ambiguous=true;continue;}
        sum+=determinant>0?1:-1;
    }
    for(int offset=16;offset;offset/=2)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    winding=__shfl_sync(0xffffffff,sum,0);
    return !__any_sync(0xffffffff,ambiguous);
}
