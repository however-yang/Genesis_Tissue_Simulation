"""Per-renderer recompilation of unchanged GL command loops with GIL released.
Only the owning render thread calls GL; the quality worker never touches GL.
No arithmetic, draw calls, buffers, camera state or physical work is changed.
"""
from numba import njit

class RenderScheduling:
    def __init__(self,view):
        self.jit=view.scene.visualizer.context.jit
        self.original={};self.compiled={};offscreen=view.camera._rasterizer._renderer
        offscreen.make_current()
        try:
            if self.jit._forward_pass is None:self.jit.gen_func_ptr()
            for name in ('_forward_pass','_update_buffer_fn'):
                fn=getattr(self.jit,name)
                options=dict(fn.targetoptions);options.pop('nopython',None);options.pop('nogil',None)
                replacement=njit(fn.signatures,nogil=True,cache=False,**options)(fn.py_func)
                self.original[name]=fn;self.compiled[name]=replacement
        finally:offscreen.make_uncurrent()
        self.set_enabled(True)
    def set_enabled(self,enabled):
        self.enabled=bool(enabled)
        for name,fn in (self.compiled if enabled else self.original).items():setattr(self.jit,name,fn)
