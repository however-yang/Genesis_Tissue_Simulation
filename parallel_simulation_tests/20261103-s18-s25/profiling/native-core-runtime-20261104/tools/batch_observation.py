"""Opt-in same-frame packed readback. Original package files stay untouched."""
import ctypes as ct
import dataclasses
import hashlib
from pathlib import Path
import struct

import numpy as np

from endoscopic_tissue import native_backend as native

stats={'calls':0,'paired_calls':0,'fallbacks':0,'mismatches':0}
_original_observe=native.NativeTissue.observe
_installed=False
_validate=False


class RuntimeObservation(ct.Structure):
    _fields_=[('hinge',native.V3),('error',native.F),('live',native.I),('gap',native.F),
              ('blade',native.F*10),('reaction',native.V3),('contacts',native.I),('jaw',native.F*4)]


def identical(a,b,path='observation'):
    if isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),path
    elif dataclasses.is_dataclass(a):
        assert type(a)==type(b),path
        for field in dataclasses.fields(a):identical(getattr(a,field.name),getattr(b,field.name),path+'.'+field.name)
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):identical(x,y,path+'.'+str(i))
    elif isinstance(a,float):
        assert struct.pack('d',a)==struct.pack('d',b),path
    else:assert a==b,path


def observe(self):
    if self.scissors_geometry is None:
        stats['fallbacks']+=1
        return _original_observe(self)
    if not getattr(self,'_runtime_observation_bound',False):
        size=self.dll.LCS_RuntimeObservationSize;size.argtypes=[];size.restype=ct.c_int
        if size()!=ct.sizeof(RuntimeObservation):raise RuntimeError('Packed observation ABI mismatch')
        function=self.dll.LCS_ReadRuntimeObservation
        function.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p,ct.POINTER(ct.c_uint),ct.POINTER(RuntimeObservation),ct.POINTER(native.ScissorsResult),ct.c_void_p]
        function.restype=ct.c_int;self.functions['LCS_ReadRuntimeObservation']=function
        self._runtime_observation_bound=True
    n=len(self.grid.positions)
    nodes=np.empty((n,3),np.float32);flags=np.empty(n,np.int32);revision=ct.c_uint();data=RuntimeObservation();result=native.ScissorsResult();loads=np.empty(7,np.float32)
    self.check('LCS_ReadRuntimeObservation',n,native.pointer(nodes),native.pointer(flags),ct.byref(revision),ct.byref(data),ct.byref(result),native.pointer(loads))
    if revision.value!=self.topology_revision:
        neighbors=np.empty((n,6),np.int32);self.check('LCS_GetNbrIdx',native.pointer(neighbors))
        neighbors.flags.writeable=False;self.neighbors,self.topology_revision=neighbors,revision.value
    hinge=np.array(data.hinge);blade=np.array(data.blade,np.float32);reaction=np.array(data.reaction,np.float32);jaw=np.array(data.jaw,np.float32)
    scissors=native.ScissorsPose(np.array(result.hinge),np.array(result.axis),np.array(result.normal),result.gap,result.radius,
        np.array((result.left_a,result.left_b,result.right_a,result.right_b)).reshape(2,2,3),
        result.pressure*native.METERS_PER_UNIT,result.phase,result.contacts,
        native.ScissorsLoads(*(float(v) for v in loads[:5]*native.METERS_PER_UNIT),int(loads[5]),bool(loads[6])))
    if not np.isfinite((*scissors.hinge,*scissors.axis,*scissors.normal,scissors.gap,*scissors.edges.reshape(12),*loads)).all():
        raise RuntimeError('Non-finite scissors state; simulation stopped')
    if (not np.isfinite(nodes).all() or not np.isfinite(blade).all() or not np.isfinite(reaction).all()
        or not np.isfinite(jaw).all() or not np.isfinite((*hinge,data.gap,data.error)).all()):
        raise RuntimeError('Non-finite native state; simulation stopped')
    output=native.Observation(nodes,self.neighbors,flags,hinge,data.gap,data.live,data.error*native.METERS_PER_UNIT,
        blade,reaction*native.METERS_PER_UNIT,scissors,tuple(float(v) for v in jaw[:2]*native.METERS_PER_UNIT))
    stats['calls']+=1
    if _validate:
        reference=_original_observe(self)
        try:identical(output,reference)
        except AssertionError:
            stats['mismatches']+=1;raise
        stats['paired_calls']+=1
    return output


def install(*,validate=False):
    global _installed,_validate
    if _installed:raise RuntimeError('Readback adapter already installed')
    _validate=validate;_installed=True
    native.NativeTissue.observe=observe
    # Snapshot provenance includes this opt-in adapter, rather than pretending
    # it is one of the frozen package files. Old snapshots stay unchanged.
    from endoscopic_tissue.training import snapshot
    original_binding=snapshot.source_binding
    binding_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def source_binding():
        return {**original_binding(),'runtime/batch_observation.py':binding_hash}
    snapshot.source_binding=source_binding
