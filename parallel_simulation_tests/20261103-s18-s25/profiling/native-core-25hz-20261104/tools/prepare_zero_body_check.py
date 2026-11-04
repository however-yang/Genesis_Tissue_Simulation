from pathlib import Path
import shutil,ast
p=Path(__file__).resolve().parents[1];src=p/'python-zero-check';shutil.copytree(p.parent/'aggressive-20261104/final-source',src,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
f=src/'endoscopic_tissue/instrument_geometry.py';s=f.read_text();tree=ast.parse(s);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scissors_body_depths');lines=s.splitlines(keepends=True);original=''.join(lines[node.lineno-1:node.end_lineno])
reference=original.replace('def scissors_body_depths(', 'def _scissors_body_depths_reference(',1)
candidate=original.replace('def scissors_body_depths(', 'def _scissors_body_depths_candidate(',1)
needle='    body = 0.0\n';assert candidate.count(needle)==1
insert='''    from .training import zero_body_check as certificate
    frames = [np.stack((axis * cs + side * normal * sn,
                         normal * cs - side * axis * sn, width)) for side in (1, -1)]
    certificate.calls += 1
    if certificate.definitely_clear(relative, axis, frames[0], frames[1],
                                    geometry.backs[0], geometry.backs[1],
                                    orientations[0], orientations[1],
                                    geometry.jaws.shaft_length, SCISSORS_WRIST_FRONT,
                                    geometry.jaws.shaft_radius):
        certificate.clear_returns += 1
        if certificate.validate:
            reference = _scissors_body_depths_reference(surface_vertices, triangles, geometry, hinge, axis, normal, gap)
            certificate.validated_clear += 1
            if reference != (0.0, 0.0):
                certificate.mismatches += 1
                raise RuntimeError('Zero-body certificate/reference mismatch: '+str(reference))
        return 0.0, 0.0
    certificate.fallbacks += 1
'''
candidate=candidate.replace(needle,insert+needle,1)
wrapper='''\n\ndef scissors_body_depths(surface_vertices, triangles, geometry, hinge, axis, normal, gap):
    from .training import zero_body_check as certificate
    if not certificate.enabled:
        return _scissors_body_depths_reference(surface_vertices, triangles, geometry, hinge, axis, normal, gap)
    return _scissors_body_depths_candidate(surface_vertices, triangles, geometry, hinge, axis, normal, gap)
'''
f.write_text(''.join(lines[:node.lineno-1])+reference+'\n\n'+candidate+wrapper+''.join(lines[node.end_lineno:]))
module='''"""Exact-zero shortcut for diagnostic envelopes; no contact forces are changed.

Finite samples certified farther outside than a conservative float32 arithmetic
error bound return the exact reference zero. Any possible contact, ambiguous
rounding, nonfinite or unsupported input uses the unchanged NumPy calculations.
The existing sample selection/centroids/midpoints remain unchanged.
"""
import numpy as np
from numba import njit
enabled=False
validate=False
calls=clear_returns=fallbacks=validated_clear=mismatches=0

@njit(cache=False,nogil=True)
def definitely_clear(relative,axis,frame0,frame1,box0,box1,orientation0,orientation1,shaft_length,front,radius):
    if len(relative)==0:return False
    axis_norm=0.;frame_norm=0.;orientation_norm=0.;box_norm=0.;relative_norm=0.
    for i in range(3):
        if not np.isfinite(axis[i]):return False
        axis_norm+=abs(float(axis[i]))
        for j in range(3):
            for value in (np.float64(frame0[i,j]),np.float64(frame1[i,j]),np.float64(orientation0[i,j]),np.float64(orientation1[i,j])):
                if not np.isfinite(value):return False
            frame_norm=max(frame_norm,abs(float(frame0[i,j])),abs(float(frame1[i,j])))
            orientation_norm=max(orientation_norm,abs(float(orientation0[i,j])),abs(float(orientation1[i,j])))
        for value in (box0[i],box1[i],box0[i+3],box1[i+3]):
            if not np.isfinite(value):return False
            box_norm=max(box_norm,abs(float(value)))
    if axis_norm>4 or frame_norm>4 or orientation_norm>4:return False
    if not np.isfinite(shaft_length+front+radius) or min(shaft_length,front,radius)<0:return False
    body=-np.inf;wrist=-np.inf
    for i in range(len(relative)):
        x=float(relative[i,0]);y=float(relative[i,1]);z=float(relative[i,2])
        if not np.isfinite(x+y+z):return False
        relative_norm=max(relative_norm,abs(x),abs(y),abs(z))
        if relative_norm>1e5:return False
        for which in range(2):
            frame=frame0 if which==0 else frame1;box=box0 if which==0 else box1;orientation=orientation0 if which==0 else orientation1
            # float64 working values only certify a strict outside margin.
            lx=(x*float(frame[0,0])+y*float(frame[0,1])+z*float(frame[0,2]))-float(box[0])
            ly=(x*float(frame[1,0])+y*float(frame[1,1])+z*float(frame[1,2]))-float(box[1])
            lz=(x*float(frame[2,0])+y*float(frame[2,1])+z*float(frame[2,2]))-float(box[2])
            depth=np.inf
            for k in range(3):
                local=lx*float(orientation[k,0])+ly*float(orientation[k,1])+lz*float(orientation[k,2])
                depth=min(depth,float(box[k+3])-abs(local))
            body=max(body,depth)
        along=x*float(axis[0])+y*float(axis[1])+z*float(axis[2]);along=min(front,max(-shaft_length,along))
        dx=x-along*float(axis[0]);dy=y-along*float(axis[1]);dz=z-along*float(axis[2])
        wrist=max(wrist,radius-np.sqrt(dx*dx+dy*dy+dz*dz))
    # Upper bound covers both three-term dot products, translations, clipping,
    # norm evaluation and any legal float32 dot-product summation/FMA ordering.
    # It only decides fallback, never changes an acceptance/contact threshold.
    scale=1.+(3.*relative_norm*frame_norm+box_norm)*3.*orientation_norm+box_norm
    scale=max(scale,1.+relative_norm*(1.+axis_norm*axis_norm)+abs(shaft_length)+abs(front)+abs(radius))
    error=128.*np.finfo(np.float32).eps*scale
    return body < -error and wrist < -error
'''
(src/'endoscopic_tissue/training/zero_body_check.py').write_text(module.replace('float(', 'np.float64(')) )
f=p/'tools/profile_all_queries.py';s=f.read_text();s=s.replace("p.add_argument('--core-optimizations',type=int,default=0)","p.add_argument('--core-optimizations',type=int,default=0);p.add_argument('--zero-body-check',action='store_true');p.add_argument('--validate-zero-check',action='store_true')",1)
needle='from endoscopic_tissue.scenario import stage_start,interpolate';s=s.replace(needle,needle+"\nif a.zero_body_check or a.validate_zero_check:\n from endoscopic_tissue.training import zero_body_check\n zero_body_check.enabled=a.zero_body_check;zero_body_check.validate=a.validate_zero_check",1)
s=s.replace("result['core_optimizations']=a.core_optimizations","result['core_optimizations']=a.core_optimizations\n result['zero_body_check']=a.zero_body_check;result['zero_check_debug']=a.validate_zero_check",1)
needle=" (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));";assert needle in s
s=s.replace(needle," if a.zero_body_check:\n  from endoscopic_tissue.training import zero_body_check as z\n  result['zero_body_statistics']={k:getattr(z,k) for k in ['calls','clear_returns','fallbacks','validated_clear','mismatches']}\n"+needle,1);f.write_text(s)
print(src)
