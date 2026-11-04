"""Training-only parameter adapter. Production native_backend.py stays intact."""
import ctypes as ct

import numpy as np

from ..native_backend import CutDesc, GraspParams, NativeTissue, pointer
from .physics import native_defaults


class TrainingMetrics(ct.Structure):
    _fields_ = [(name, ct.c_float) for name in ('maxStretch','rimStretch','graspRingStretch',
                                               'coreForceMax','coreForceMedian')]
    _fields_ += [(name, ct.c_uint32) for name in ('activeCoreCorners','brokenCoreCorners','brokenGripPoints',
                 'tearEdges','materialTearEdges','rimTearEdges','interiorTearEdges','nearGraspTearEdges')]
    _fields_ += [('minVolumeRatio',ct.c_float),('invalidTets',ct.c_uint32)]


class TrainingNativeTissue(NativeTissue):
    def __init__(self, *args, native_parameters=None, telemetry=False, **kwargs):
        self.training_parameters = dict(native_parameters or native_defaults())
        self.training_flags = (1 if telemetry or self.training_parameters['tear_stretch_ratio'] > 0 else 0)
        if self.training_parameters['grasp_core_break_force_n'] != 150.:
            self.training_flags |= 3
        self.last_training_metrics = None
        super().__init__(*args, **kwargs)

    def _enable_training(self):
        if not getattr(self,'training_flags',0):
            return
        signatures = {'LCS_TrainingAbiVersion': ((),ct.c_int),
                      'LCS_TrainingMetricsSize': ((),ct.c_int),
                      'LCS_EnableTrainingDiagnostics': ((ct.c_int,),ct.c_int),
                      'LCS_GetTrainingMetrics': ((ct.c_void_p,),ct.c_int),
                      'LCS_GetTrainingCoreForces': ((ct.c_int,ct.c_void_p),ct.c_int),
                      'LCS_GetTrainingTearCauses': ((ct.c_int,ct.c_void_p),ct.c_int)}
        try:
            for name,(args,result) in signatures.items():
                fn=self.dll[name]
                fn.argtypes,fn.restype=args,result
                self.functions[name]=fn
        except AttributeError as error:
            raise RuntimeError('Damage telemetry/overload release requires the isolated training ABI DLL; '
                               'the old unified solver does not execute its legacy core-break kernel') from error
        if self.functions['LCS_TrainingAbiVersion']()!=1 or self.functions['LCS_TrainingMetricsSize']()!=ct.sizeof(TrainingMetrics):
            raise RuntimeError('Training native ABI mismatch')
        self.check('LCS_EnableTrainingDiagnostics',self.training_flags)

    def initialize(self):
        p = self.training_parameters
        # With defaults, execute the original method without wrapping any call.
        if p == native_defaults() and not getattr(self,'training_flags',0):
            return super().initialize()
        originals = {name: self.functions[name] for name in
                     ("LCS_InitCut", "LCS_SetGraspParams", "LCS_SetContactModel")}

        def init_cut(description, *args):
            ct.cast(description, ct.POINTER(CutDesc)).contents.tear_ratio = p["tear_stretch_ratio"]
            result=originals["LCS_InitCut"](description, *args)
            if result==0:
                self._enable_training()
            return result

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

    def material_quality(self):
        if not self.training_flags:
            return super().material_quality()
        metrics=TrainingMetrics()
        self.check('LCS_GetTrainingMetrics',ct.byref(metrics))
        self.last_training_metrics={name:getattr(metrics,name) for name,_ in metrics._fields_}
        return np.array([metrics.minVolumeRatio,metrics.invalidTets],np.float32)

    def core_forces_n(self):
        """Diagnostic full-array read; normal step telemetry uses compact GPU reductions."""
        out=np.empty(len(self.grid.positions),np.float32)
        self.check('LCS_GetTrainingCoreForces',len(out),pointer(out))
        return out*.01

    def tear_causes(self):
        out=np.empty(len(self.grid.edges),np.uint8)
        self.check('LCS_GetTrainingTearCauses',len(out),pointer(out))
        return out
