"""Explicit, bounded host cache for repeated restores of one immutable snapshot.

The portable file stays compressed and unchanged. Preparing costs time and host
memory; hot restore timing MUST be reported separately from cold file restore.
"""
from dataclasses import dataclass,field
import time
from endoscopic_tissue.training.snapshot import EnvSnapshot

@dataclass(frozen=True,init=False)
class PreparedSnapshot(EnvSnapshot):
    _native: bytes=field(repr=False,compare=False)
    preparation_ms: float
    reserved_host_bytes: int

    def __init__(self,snapshot,*,max_host_bytes):
        if not isinstance(snapshot,EnvSnapshot):raise TypeError('Expected a validated EnvSnapshot')
        if type(max_host_bytes) is not int or max_host_bytes<=0:raise ValueError('Explicit positive host-memory budget required')
        size=snapshot.metadata['native_bytes']
        # Reserve the retained compressed payload plus decompressed native state.
        # The caller must also leave room for the live simulator and restore scratch.
        reserved=size+snapshot.size_bytes
        if reserved>max_host_bytes:raise MemoryError(f'Prepared snapshot needs {reserved} host bytes; budget {max_host_bytes}')
        started=time.perf_counter()
        super().__init__(snapshot.native_compressed,snapshot.python_compressed,snapshot.metadata_json)
        # Use the base decoder, not an overridable caller method. Size and frame
        # checks remain intact; callers cannot inject arbitrary unverified raw data.
        raw=EnvSnapshot.native_bytes(self)
        object.__setattr__(self,'_native',raw)
        object.__setattr__(self,'reserved_host_bytes',reserved)
        object.__setattr__(self,'preparation_ms',(time.perf_counter()-started)*1000)

    def native_bytes(self):
        return self._native
