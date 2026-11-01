"""Training-only parameter adapter. Production native_backend.py stays intact."""
import ctypes as ct

from ..native_backend import CutDesc, GraspParams, NativeTissue
from .physics import native_defaults


class TrainingNativeTissue(NativeTissue):
    def __init__(self, *args, native_parameters=None, **kwargs):
        self.training_parameters = dict(native_parameters or native_defaults())
        super().__init__(*args, **kwargs)

    def initialize(self):
        p = self.training_parameters
        # With defaults, execute the original method without wrapping any call.
        if p == native_defaults():
            return super().initialize()
        originals = {name: self.functions[name] for name in
                     ("LCS_InitCut", "LCS_SetGraspParams", "LCS_SetContactModel")}

        def init_cut(description, *args):
            ct.cast(description, ct.POINTER(CutDesc)).contents.tear_ratio = p["tear_stretch_ratio"]
            return originals["LCS_InitCut"](description, *args)

        def grasp_params(description):
            gp = ct.cast(description, ct.POINTER(GraspParams)).contents
            gp.exempt_rings = p["tear_exempt_rings"]
            gp.break_force = p["grasp_core_break_force_n"] * 100.
            gp.friction = p["grasp_friction"]
            return originals["LCS_SetGraspParams"](description)

        def contact_model(model, mu_s, mu_d, thickness, radius):
            return originals["LCS_SetContactModel"](
                model, p["grasp_friction"], p["grasp_dynamic_friction"], thickness, radius)

        self.functions.update(LCS_InitCut=init_cut, LCS_SetGraspParams=grasp_params,
                              LCS_SetContactModel=contact_model)
        try:
            return super().initialize()
        finally:
            self.functions.update(originals)
