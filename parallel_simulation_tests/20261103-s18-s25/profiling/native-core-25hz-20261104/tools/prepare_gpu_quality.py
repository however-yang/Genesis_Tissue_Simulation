from pathlib import Path
import shutil
p=Path(__file__).resolve().parents[1];src=p/'python-gpu-check';shutil.copytree(p.parent/'aggressive-20261104/final-source',src,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
module='''"""Read-only current-soup GPU counterpart of scissors_body_depths.

All vertex, centroid and three midpoint samples are retained. Small floating
point differences are bounded in regression; values near the UNCHANGED physical
acceptance threshold use the original CPU method. Unsupported inputs fall back.
"""
import ctypes,weakref
import numpy as np
from ..instrument_geometry import scissors_body_depths,SCISSORS_WRIST_FRONT

def parameters(geometry,hinge,axis,normal,gap):
    width=np.cross(axis,normal);sn=np.clip(gap/(2*geometry.jaws.length),0,.95);cs=np.sqrt(1-sn*sn)
    frames=[np.stack((axis*cs+side*normal*sn,normal*cs-side*axis*sn,width)) for side in (1,-1)]
    orientations=[np.stack((b[6:9],b[9:12],np.cross(b[6:9],b[9:12]))) for b in geometry.backs]
    packed=np.ascontiguousarray(np.concatenate((np.asarray(hinge).ravel(),np.asarray(axis).ravel(),frames[0].ravel(),frames[1].ravel(),geometry.backs[0,:6],geometry.backs[1,:6],orientations[0].ravel(),orientations[1].ravel(),[geometry.jaws.shaft_length,SCISSORS_WRIST_FRONT,geometry.jaws.shaft_radius])),dtype=np.float64)
    flags=int(np.asarray(hinge).dtype==np.float32) | (int(np.asarray(axis).dtype==np.float32)<<1)
    assert packed.shape==(57,)
    return packed,flags

class GPUScissorsBodyCheck:
    def __init__(self,env,validate=False):
        self.environment=weakref.ref(env);self.validate=bool(validate);self.stats=dict(calls=0,gpu_calls=0,cpu_fallbacks=0,threshold_fallbacks=0,paired_calls=0,numeric_mismatches=0,decision_mismatches=0,max_abs_error_native_cm=0.)
        self.function=env.native.dll.LCS_CurrentBodyDepth
        self.function.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];self.function.restype=ctypes.c_int
    def __call__(self,vertices,triangles,geometry,hinge,axis,normal,gap):
        env=self.environment();self.stats['calls']+=1
        args=(vertices,triangles,geometry,hinge,axis,normal,gap)
        if env is None or env.last_surface is None or vertices is not env.last_surface.vertices or triangles is not env.last_surface.triangles:
            self.stats['cpu_fallbacks']+=1;return scissors_body_depths(*args)
        packed,flags=parameters(geometry,hinge,axis,normal,gap);output=np.empty(2,np.float64)
        code=self.function(len(vertices),packed.ctypes.data,flags,output.ctypes.data)
        if code<0:raise RuntimeError('Current GPU surface body diagnostic CUDA/native error '+str(code))
        if code>0:
            self.stats['cpu_fallbacks']+=1;return scissors_body_depths(*args)
        self.stats['gpu_calls']+=1
        if not np.isfinite(output).all():raise RuntimeError('Nonfinite GPU body diagnostic')
        limit=env.config.limits.max_body_overlap_m
        # A fallback band selects a method. It is NOT a changed acceptance limit.
        # Outside this band, verified arithmetic differences are much smaller.
        near_limit=abs(float(output.max())*.01-limit)<=5e-5
        reference=None
        if self.validate or near_limit:reference=scissors_body_depths(*args)
        if self.validate:
            self.stats['paired_calls']+=1;error=float(np.max(np.abs(output-reference)));self.stats['max_abs_error_native_cm']=max(self.stats['max_abs_error_native_cm'],error)
            if error>2e-5:self.stats['numeric_mismatches']+=1;raise RuntimeError('GPU/CPU body depth exceeds 0.2 micrometre tolerance: '+str((output,reference,error)))
            if not near_limit and ((float(output.max())*.01>limit)!=(max(reference)*.01>limit)):
                self.stats['decision_mismatches']+=1;raise RuntimeError('GPU/CPU body diagnostic acceptance mismatch')
        if near_limit:self.stats['threshold_fallbacks']+=1;return reference
        return float(output[0]),float(output[1])
'''
(src/'endoscopic_tissue/training/gpu_body_check.py').write_text(module)
f=src/'endoscopic_tissue/training/quality.py';s=f.read_text();needle='body, wrist = scissors_body_depths(surface.vertices, surface.triangles, geometry,';assert needle in s;s=s.replace(needle,'body, wrist = getattr(self,"body_depth_function",scissors_body_depths)(surface.vertices, surface.triangles, geometry,',1);f.write_text(s)
f=src/'endoscopic_tissue/training/snapshot.py';s=f.read_text();s=s.replace("'gpu_topology.py')]}","'gpu_topology.py','gpu_body_check.py')]}",1);needle="'gpu_topology_checks': getattr(env.config,'gpu_topology_checks',False)}";assert needle in s;s=s.replace(needle,"'gpu_topology_checks': getattr(env.config,'gpu_topology_checks',False), 'gpu_body_checks': hasattr(env.quality,'body_depth_function')}",1);f.write_text(s)
f=p/'tools/profile_all_queries.py';s=f.read_text();s=s.replace("p.add_argument('--core-optimizations',type=int,default=0)","p.add_argument('--core-optimizations',type=int,default=0);p.add_argument('--gpu-body-check',action='store_true');p.add_argument('--validate-body-check',action='store_true')",1)
needle="state,context=env.reset(0);(a.output/'context.json').write_text(json.dumps(context,indent=2))";assert needle in s
s=s.replace(needle,needle+"\n  if a.gpu_body_check:\n   from endoscopic_tissue.training.gpu_body_check import GPUScissorsBodyCheck\n   env.quality.body_depth_function=GPUScissorsBodyCheck(env,validate=a.validate_body_check)\n  result['gpu_body_check']=a.gpu_body_check;result['gpu_body_debug']=a.validate_body_check",1)
# V6's bit 16 is a different, manifest-bound experiment than V5's uniform mode.
s=s.replace('if a.core_optimizations&16:\n      uniform_stats=',"if a.core_optimizations&16 and hasattr(native_control,'LCS_CoarseUniformStats'):\n      uniform_stats=",1)
needle=' if env:env.close()';assert needle in s;s=s.replace(needle," if env and hasattr(env.quality,'body_depth_function'):\n  result['gpu_body_statistics']=dict(env.quality.body_depth_function.stats)\n"+needle,1);f.write_text(s)
for name in ['check_current_frame_final_cache.py','check_snapshot_roundtrip_cache.py']:
 f=p/'tools'/name;s=f.read_text().replace("parser.add_argument('--core-optimizations',type=int,default=0)","parser.add_argument('--core-optimizations',type=int,default=0);parser.add_argument('--gpu-body-check',action='store_true')",1)
 needle='env.reset(7);';assert needle in s
 s=s.replace(needle,"env.reset(7);\n  if a.gpu_body_check:\n   from endoscopic_tissue.training.gpu_body_check import GPUScissorsBodyCheck\n   env.quality.body_depth_function=GPUScissorsBodyCheck(env,validate=True)\n  ",1)
 f.write_text(s)
print(src)
