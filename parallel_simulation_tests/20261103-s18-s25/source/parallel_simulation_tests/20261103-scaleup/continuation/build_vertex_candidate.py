"""R9 isolated feasible vertex-direction experiment; unchanged force law and bounds."""
from pathlib import Path
import difflib,hashlib,json,os,shutil,subprocess,time
here=Path(__file__).resolve().parent
project=here.parents[2]
base=project/'runs/a100-functional-fixes-20261102-r1/release-training-r3/source'
root=here/'r9-vertex-cone'
source=root/'source'
root.mkdir(exist_ok=False)
shutil.copytree(base,source)
cone=(project/'runs/a100-cone-repair-20261102-r1/source/vendor/cuda/solver/material/repair_cone.cuh').read_text().split('#ifndef REPAIR_CONE_ALGEBRA_ONLY')[0]
cone=cone.replace('// Isolated R8 experiment: a feasible direction in the existing three-DOF\n// repair basis, not a new force, projection of positions or relaxed bound.', '// R9 reuses the validated R8 three-dimensional quadratic algebra only.\n// Its application here is per vertex, not the rejected R8 regional mode.')
(source/'vendor/cuda/solver/material/vertex_cone_algebra.cuh').write_text(cone)
p=source/'vendor/cuda/solver/material/unified_vertex.cuh'
before=p.read_text()
s=before.replace('template<bool AllowFragments=true> __device__ inline void unifiedVertexSolve(', 'template<bool AllowFragments=true,bool FeasibleVertex=false> __device__ inline void unifiedVertexSolve(')
s=s.replace('template<bool AllowFragments=true> __global__ void k_UnifiedVertex(', 'template<bool AllowFragments=true,bool FeasibleVertex=false> __global__ void k_UnifiedVertex(')
s=s.replace('unifiedVertexSolve<AllowFragments>(s,sites[k],lane);','unifiedVertexSolve<AllowFragments,FeasibleVertex>(s,sites[k],lane);')
marker='    // Same-color vertices share neither a lattice tetrahedron nor an axial'
assert s.count(marker)==1
patch=r'''    // R9 experiment: a scalar J limiter can discard tangential descent when
    // one incident tetrahedron is at its floor. Solve the SAME local quadratic
    // under affine per-vertex J constraints. No material/contact parameter,
    // attachment, time integration or position projection is changed.
    if constexpr(FeasibleVertex){
        if(scale<.25f && (s.degree[i]>0 || (s.secondary && s.secondary->degree[i]>0))){
            float3 original=step*scale;
            repair_cone::Plane planes[8];int selected[8],count=0;
            repair_cone::Matrix matrix={H.xx,H.yy,H.zz,H.xy,H.xz,H.yz};
            repair_cone::Vector g={gradient.x,gradient.y,gradient.z};
            for(int pass=0;pass<8;pass++){
                float worst=1.f,lower=0;float3 normal=mkf3(0,0,0);int chosen=2147483647;
                for(int k=s.offsets[i]+lane;k<s.offsets[i+1];k+=32){
                    int t=s.incident[k];if(!s.live[t])continue;int id[4];unifiedTetIds(s,t,id);
                    float3 a=s.pos[id[0]],u=s.pos[id[1]]-a,v=s.pos[id[2]]-a,w=s.pos[id[3]]-a;
                    float det=unifiedKuhn(s,t).metric[6],J=dot3(u,cross3(v,w))*det;
                    float3 g1=cross3(v,w),g2=cross3(w,u),g3=cross3(u,v);
                    float3 n=(i==id[0]?(g1+g2+g3)*(-1.f):i==id[1]?g1:i==id[2]?g2:g3)*det;
                    float margin=fmaxf(0.f,.9f*(J-s.minimum)),d=dot3(n,step);
                    float alpha=d<0?fminf(1.f,margin/(-d)):1.f;
                    if(alpha<worst || (alpha==worst && t<chosen)){
                        worst=alpha;chosen=t;normal=n;lower=-margin;
                    }
                }
                float bound=worst;
                for(int o=16;o;o/=2)bound=fminf(bound,__shfl_xor_sync(0xffffffff,bound,o));
                int owner=(worst==bound)?lane:32;
                for(int o=16;o;o/=2)owner=min(owner,__shfl_xor_sync(0xffffffff,owner,o));
                chosen=__shfl_sync(0xffffffff,chosen,owner);
                normal=mkf3(__shfl_sync(0xffffffff,normal.x,owner),__shfl_sync(0xffffffff,normal.y,owner),__shfl_sync(0xffffffff,normal.z,owner));
                lower=__shfl_sync(0xffffffff,lower,owner);
                int done=bound>=.99999f;
                if(lane==0 && !done){
                    for(int j=0;j<count;j++)if(selected[j]==chosen)done=1;
                    double length=sqrt(double(normal.x)*normal.x+double(normal.y)*normal.y+double(normal.z)*normal.z);
                    if(!(length>0))done=1;
                    if(!done){
                        selected[count]=chosen;planes[count++]={{normal.x/length,normal.y/length,normal.z/length},lower/length};
                        repair_cone::Vector best;
                        if(repair_cone::solve(matrix,g,planes,count,best))step=mkf3(best.x,best.y,best.z);
                        else done=1;
                    }
                }
                done=__shfl_sync(0xffffffff,done,0);
                step=mkf3(__shfl_sync(0xffffffff,step.x,0),__shfl_sync(0xffffffff,step.y,0),__shfl_sync(0xffffffff,step.z,0));
                if(done)break;
            }
            // Recheck ALL live incident tets, including constraints outside the
            // bounded eight-plane search. Preserve the original safety factor.
            float safe=1.f;
            for(int k=s.offsets[i]+lane;k<s.offsets[i+1];k+=32){
                int t=s.incident[k];if(!s.live[t])continue;int id[4];unifiedTetIds(s,t,id);
                float3 a=s.pos[id[0]],u=s.pos[id[1]]-a,v=s.pos[id[2]]-a,w=s.pos[id[3]]-a;
                float det=unifiedKuhn(s,t).metric[6],J=dot3(u,cross3(v,w))*det;
                float3 g1=cross3(v,w),g2=cross3(w,u),g3=cross3(u,v);
                float3 n=(i==id[0]?(g1+g2+g3)*(-1.f):i==id[1]?g1:i==id[2]?g2:g3)*det;
                float d=dot3(n,step);
                if(d<0)safe=fminf(safe,fmaxf(0.f,.9f*(J-s.minimum)/(-d)));
            }
            for(int o=16;o;o/=2)safe=fminf(safe,__shfl_xor_sync(0xffffffff,safe,o));
            float3 trial=step*safe;
            double oldCost=repair_cone::energy(matrix,g,{original.x,original.y,original.z});
            double newCost=repair_cone::energy(matrix,g,{trial.x,trial.y,trial.z});
            step=isfinite(newCost) && newCost<=oldCost?trial:original;scale=1.f;
        }
    }
'''
s=s.replace(marker,patch+marker)
s='#include "vertex_cone_algebra.cuh"\n'+s
p.write_text(s)
diffs=list(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='R3/unified_vertex.cuh',tofile='R9/unified_vertex.cuh'))
p=source/'vendor/cuda/xpbd/xpbd_solver.cu';before=p.read_text()
start=before.index('extern "C" int xpbd_relax_reconstructed_contact(){')
end=before.index('\nextern "C"',start+10)
section=before[start:end]
needle='k_UnifiedVertex<<<g_sweepBlocks,64,0,g_captureStream>>>'
assert section.count(needle)==1
section=section.replace(needle,'k_UnifiedVertex<true,true><<<g_sweepBlocks,64,0,g_captureStream>>>')
s=before[:start]+section+before[end:];p.write_text(s)
diffs+=difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='R3/xpbd_solver.cu',tofile='R9/xpbd_solver.cu')
(root/'changes.patch').write_text(''.join(diffs))
(root/'source-sha256.json').write_text(json.dumps({str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()},indent=2))
envdir=project/'envs/genesis-py311-cu125-v1';env=os.environ.copy();env['PATH']=str(envdir/'bin')+':'+env['PATH'];env['PYTHONDONTWRITEBYTECODE']='1'
commands=[[str(envdir/'bin/cmake'),'-S',str(source),'-B',str(root/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER='+str(envdir/'bin/nvcc'),'-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12','-DCMAKE_CXX_COMPILER=/usr/bin/g++-12','-DCUDAToolkit_ROOT='+str(envdir)],[str(envdir/'bin/cmake'),'--build',str(root/'build'),'--parallel','2']]
records=[]
with (root/'build.log').open('x') as log:
 for command in commands:
  start=time.time();q=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
  records.append(dict(command=command,exit_code=q.returncode,wall_s=time.time()-start));(root/'build.json').write_text(json.dumps(records,indent=2))
  if q.returncode:raise SystemExit(q.returncode)
print('R9 build completed; candidate is experimental and not adopted',flush=True)
