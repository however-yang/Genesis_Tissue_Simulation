"""Read-only pad geometry and material witnesses, independent of grasp assistance."""
import numpy as np
from types import SimpleNamespace

from ..instrument_geometry import load_grasper


def pad_coordinates(positions_m, tools):
    axis = np.asarray(tools['grasper_axis'], float)
    axis /= np.linalg.norm(axis)
    normal = np.asarray(tools['grasper_normal'], float)
    normal -= axis*np.dot(axis, normal)
    normal /= np.linalg.norm(normal)
    frame = np.stack((axis, normal, np.cross(axis, normal)))
    return (positions_m - np.asarray(tools['grasper_hinge_m'])) @ frame.T


def between_pads(positions_m, tools, pad_bounds_m, jaw_length_m):
    """Same two rotating pad surfaces as the existing interaction diagnostic."""
    # Same orthonormal frame, fused transform and pad comparisons.
    axis=np.asarray(tools['grasper_axis'],float);axis=axis/np.linalg.norm(axis)
    normal=np.asarray(tools['grasper_normal'],float);normal=normal-axis*np.dot(axis,normal)
    normal=normal/np.linalg.norm(normal);frame=np.stack((axis,normal,np.cross(axis,normal)))
    sn=min(tools['grasper_gap_m']/(2*jaw_length_m),.95);cs=np.sqrt(1-sn*sn)
    from .distance_kernels import between_pads as fused
    return fused(positions_m,np.asarray(tools['grasper_hinge_m'],float),frame,np.asarray(pad_bounds_m).reshape(2,2,3),sn,cs)


class GraspMonitor:
    """First bilateral material bite establishes persistent witnesses.

    Force/retention thresholds are experiment definitions, not a tissue calibration.
    An assisted hold is labelled as assisted even when its geometry is convincing.
    """
    def __init__(self, settings, material_mask, force_threshold_n=.05):
        self.settings = settings
        self.material_ids = np.flatnonzero(material_mask)
        self.bounds = np.asarray(load_grasper(settings.jaw_length, settings.jaw_width,
                                              settings.instrument_model).bounds)*.01
        self.jaw_length_m = settings.jaw_length*.01
        self.contact_model = settings.contact_model
        self.force_threshold_n = float(force_threshold_n)
        self.ids = np.empty(0, np.int32)
        self.initial_local = self.initial_world = None

    def assess_native(self, observed):
        tools = {'grasper_hinge_m':observed.hinge*.01, 'grasper_gap_m':observed.jaw_gap*.01,
                 'grasper_axis':self.settings.jaw_axis, 'grasper_normal':self.settings.jaw_normal,
                 'jaw_load_n':observed.jaw_load_n, 'grip_points':observed.grip_points}
        return self.assess(SimpleNamespace(arrays={'positions_m':observed.nodes*.01},metadata={'tools':tools}))

    def assess(self, state):
        positions, tools = state.arrays['positions_m'], state.metadata['tools']
        if not np.isfinite(positions).all():
            return {'contact_model':int(self.contact_model), 'assisted':self.contact_model == 2,
                    'invalid_geometry':True, 'bilateral_load':False, 'between_pads_material_nodes':0,
                    'witness_ids':self.ids.tolist(), 'witness_retained_fraction':0.,
                    'witness_tangential_slip_m':None, 'witness_mean_lift_m':None,
                    'geometric_loaded_hold':False, 'friction_hold_evidence':False,
                    'anchor_points':int(tools['grip_points'])}
        enclosed = between_pads(positions[self.material_ids], tools, self.bounds, self.jaw_length_m)
        candidates = self.material_ids[enclosed]
        loads = np.asarray(tools['jaw_load_n'], float)
        bilateral = bool(np.isfinite(loads).all() and np.all(loads > self.force_threshold_n))
        if not len(self.ids) and len(candidates) and bilateral:
            self.ids = candidates.copy()
            self.initial_local = pad_coordinates(positions[self.ids], tools)
            self.initial_world = positions[self.ids].copy()
        retained = slip = rise = 0.
        if len(self.ids):
            retained = float(between_pads(positions[self.ids], tools, self.bounds, self.jaw_length_m).mean())
            local = pad_coordinates(positions[self.ids], tools)
            slip = float(np.linalg.norm((local-self.initial_local)[:, [0, 2]], axis=1).mean())
            rise = float(np.mean(positions[self.ids, 1]-self.initial_world[:, 1]))
        assisted = self.contact_model == 2
        return {'contact_model': int(self.contact_model), 'assisted': assisted,
                'force_threshold_n': self.force_threshold_n,
                'bilateral_load': bilateral, 'between_pads_material_nodes': len(candidates),
                'witness_ids': self.ids.tolist(), 'witness_retained_fraction': retained,
                'witness_tangential_slip_m': slip, 'witness_mean_lift_m': rise,
                'geometric_loaded_hold': bilateral and len(candidates) > 0 and retained > 0,
                'friction_hold_evidence': (not assisted and bilateral and retained >= .5
                                           and tools['grip_points'] == 0),
                'anchor_points': int(tools['grip_points'])}
