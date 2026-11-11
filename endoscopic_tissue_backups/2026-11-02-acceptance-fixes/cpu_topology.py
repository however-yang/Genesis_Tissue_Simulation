"""Read Windows CPU-set topology without changing any process or system setting."""
import ctypes
from ctypes import wintypes
import struct


def probe():
    # Layout verified against the installed 10.0.26100.0 SDK, um/winnt.h.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    query = kernel.GetSystemCpuSetInformation
    query.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                      wintypes.HANDLE, wintypes.DWORD]
    query.restype = wintypes.BOOL
    length = wintypes.DWORD()
    query(None, 0, ctypes.byref(length), None, 0)
    if length.value == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(length.value)
    if not query(buffer, length.value, ctypes.byref(length), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    raw, offset, rows = buffer.raw[:length.value], 0, []
    while offset < len(raw):
        size, kind = struct.unpack_from('<II', raw, offset)
        if size < 8 or offset+size > len(raw):
            raise ValueError('Invalid SYSTEM_CPU_SET_INFORMATION record')
        if kind == 0:
            if size < 32:
                raise ValueError('Truncated CpuSet record')
            identity, group, logical, core, cache, numa, efficiency, flags, reserved, tag = \
                struct.unpack_from('<IH6BIQ', raw, offset+8)
            rows.append({'id': identity, 'group': group, 'logical_processor': logical,
                         'core': core, 'last_level_cache': cache, 'numa': numa,
                         'efficiency_class': efficiency, 'flags': flags,
                         'scheduling_class': reserved & 255})
        offset += size
    return {'source': 'GetSystemCpuSetInformation', 'cpu_sets': rows,
            'read_only': True}
