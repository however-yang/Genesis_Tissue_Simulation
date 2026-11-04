// R9 reuses the validated R8 three-dimensional quadratic algebra only.
// Its application here is per vertex, not the rejected R8 regional mode.
#pragma once

namespace repair_cone {
struct Vector {double x,y,z;};
struct Matrix {double xx,yy,zz,xy,xz,yz;};
struct Plane {Vector normal;double lower;};
__device__ inline Vector add(Vector a,Vector b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
__device__ inline Vector mul(Vector a,double b){return {a.x*b,a.y*b,a.z*b};}
__device__ inline double dot(Vector a,Vector b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ inline Vector product(Matrix a,Vector b){return {
    a.xx*b.x+a.xy*b.y+a.xz*b.z,a.xy*b.x+a.yy*b.y+a.yz*b.z,a.xz*b.x+a.yz*b.y+a.zz*b.z};}
__device__ inline bool inverse(Matrix a,Matrix& b,double tolerance){
    double cxx=a.yy*a.zz-a.yz*a.yz,cyy=a.xx*a.zz-a.xz*a.xz,czz=a.xx*a.yy-a.xy*a.xy;
    double cxy=a.xz*a.yz-a.xy*a.zz,cxz=a.xy*a.yz-a.xz*a.yy,cyz=a.xy*a.xz-a.xx*a.yz;
    double determinant=a.xx*cxx+a.xy*cxy+a.xz*cxz;
    if(!(a.xx>0 && czz>0 && determinant>tolerance*a.xx*a.yy*a.zz) || !isfinite(determinant))return false;
    double scale=1./determinant;b={cxx*scale,cyy*scale,czz*scale,cxy*scale,cxz*scale,cyz*scale};return true;
}
__device__ inline double energy(Matrix h,Vector g,Vector x){return dot(g,x)+.5*dot(x,product(h,x));}
__device__ inline void consider(Matrix h,Vector g,const Plane* planes,int count,Vector candidate,Vector& best,double& cost){
    if(!isfinite(candidate.x+candidate.y+candidate.z))return;
    for(int p=0;p<count;p++)if(dot(planes[p].normal,candidate)<planes[p].lower-1e-9)return;
    double trial=energy(h,g,candidate);if(isfinite(trial) && trial<cost){best=candidate;cost=trial;}
}
// At most eight planes in three dimensions. Enumerate active sets of size
// zero to three; compare feasible candidates in the same positive quadratic.
// Final simulator bounds still run in their original precision afterwards.
__device__ inline bool solve(Matrix h,Vector g,const Plane* planes,int count,Vector& best){
    Matrix inv;if(!inverse(h,inv,1e-14))return false;
    Vector free=mul(product(inv,g),-1),directions[8];
    for(int i=0;i<count;i++)directions[i]=product(inv,planes[i].normal);
    best={0,0,0};double cost=0;consider(h,g,planes,count,free,best,cost);
    for(int i=0;i<count;i++){
        double a=dot(planes[i].normal,directions[i]);if(!(a>0))continue;
        double ri=planes[i].lower-dot(planes[i].normal,free);
        consider(h,g,planes,count,add(free,mul(directions[i],ri/a)),best,cost);
        for(int j=i+1;j<count;j++){
            double b=dot(planes[i].normal,directions[j]),c=dot(planes[j].normal,directions[j]);
            double rj=planes[j].lower-dot(planes[j].normal,free),det=a*c-b*b;
            if(det>1e-12*a*c){
                Vector candidate=add(add(free,mul(directions[i],(ri*c-rj*b)/det)),mul(directions[j],(rj*a-ri*b)/det));
                consider(h,g,planes,count,candidate,best,cost);
            }
            for(int k=j+1;k<count;k++){
                Matrix gram={a,c,dot(planes[k].normal,directions[k]),b,
                    dot(planes[i].normal,directions[k]),dot(planes[j].normal,directions[k])};
                Matrix gramInv;if(!inverse(gram,gramInv,1e-12))continue;
                Vector rhs={ri,rj,planes[k].lower-dot(planes[k].normal,free)},lambda=product(gramInv,rhs);
                Vector candidate=add(add(add(free,mul(directions[i],lambda.x)),mul(directions[j],lambda.y)),mul(directions[k],lambda.z));
                consider(h,g,planes,count,candidate,best,cost);
            }
        }
    }
    return true;
}
}

