"""Bind opt-in native runtime modes to snapshots without editing frozen packages."""
import ctypes
import hashlib
from pathlib import Path


def install():
    from endoscopic_tissue.training import snapshot
    original=snapshot.binding
    original_source=snapshot.source_binding
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def binding(env):
        data=original(env)
        if hasattr(env.native.dll,'LCS_GetRuntimeCoreMode'):
            function=env.native.dll.LCS_GetRuntimeCoreMode
            function.argtypes=[];function.restype=ctypes.c_int
            data['native_runtime_core_mode']=function()
        data['batched_runtime_observation']=bool(getattr(env.native,'_runtime_observation_bound',False))
        return data
    def source_binding():
        return {**original_source(),'runtime/runtime_binding.py':digest}
    snapshot.binding=binding;snapshot.source_binding=source_binding
