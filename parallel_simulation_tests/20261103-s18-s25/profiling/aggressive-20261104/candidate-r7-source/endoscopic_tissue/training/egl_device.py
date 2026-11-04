"""Read-only EGL/CUDA UUID mapping, before creating any OpenGL context."""
import json
import os
from pathlib import Path
import subprocess
import sys


def probe():
    import ctypes as ct
    import uuid
    from genesis.ext.pyrender.platforms import egl
    from OpenGL.raw.EGL.NV.device_cuda import EGL_CUDA_DEVICE_NV
    query=egl._get_egl_func('eglQueryDeviceAttribEXT',ct.c_uint,egl._EGLDeviceEXT,ct.c_int,ct.POINTER(ct.c_ssize_t))
    if query is None:raise RuntimeError('EGL CUDA device mapping unavailable')
    driver=ct.CDLL('libcuda.so.1')
    if driver.cuInit(0):raise RuntimeError('CUDA driver metadata initialization failed')
    class UUID(ct.Structure):_fields_=[('bytes',ct.c_ubyte*16)]
    rows=[]
    for index,device in enumerate(egl.query_devices()):
        ordinal=ct.c_ssize_t(-1)
        if not query(device._display,EGL_CUDA_DEVICE_NV,ct.byref(ordinal)):continue
        value=UUID()
        if driver.cuDeviceGetUuid(ct.byref(value),ordinal.value):continue
        rows.append(dict(egl_index=index,gpu_uuid='GPU-'+str(uuid.UUID(bytes=bytes(value.bytes)))))
    return rows


def bind(gpu_uuid):
    if not isinstance(gpu_uuid,str) or not gpu_uuid.startswith('GPU-') or ',' in gpu_uuid:
        raise ValueError('Rendering requires one explicit GPU UUID')
    # The child only queries device metadata. Remove CUDA visibility remapping
    # in that child so EGL CUDA ordinals map to the driver UUIDs unambiguously.
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('CUDA_DEVICE_ORDER',None)
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    output=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--probe'],env=env,
                          check=True,capture_output=True,text=True,timeout=120)
    matches=[x for x in json.loads(output.stdout) if x['gpu_uuid']==gpu_uuid]
    if len(matches)!=1:raise RuntimeError('No unique hardware EGL device for '+gpu_uuid)
    selected=matches[0];os.environ['EGL_DEVICE_ID']=str(selected['egl_index'])
    return dict(selected,method='EGL_CUDA_DEVICE_NV and CUDA driver UUID, queried without visibility remapping')


if __name__=='__main__':
    if sys.argv[1:]!=['--probe']:raise SystemExit('Use --probe')
    print(json.dumps(probe()))
