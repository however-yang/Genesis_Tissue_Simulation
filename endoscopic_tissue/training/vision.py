"""Optional synchronous vision; no Genesis import on the headless training path."""
import time
import weakref

import cv2
import numpy as np


def align_depth_to_endoscope(depth_pinhole_m, lens_map, vignette, near, far):
    """RGB uses bottom-up GL coordinates; depth is exported upright by Genesis."""
    # The RGB map's second array encodes bilinear fractions. Feeding that pair
    # directly to INTER_NEAREST shifts integer sample locations in OpenCV.
    xy, _ = cv2.convertMaps(*lens_map, cv2.CV_32FC2)
    aligned = cv2.remap(depth_pinhole_m[::-1], xy, None, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = ((aligned > near) & (aligned < far*(1-1e-3)) & np.isfinite(aligned)
             & (vignette[...,0] > 0))
    aligned = np.where(valid, aligned, 0).astype(np.float32)
    return aligned, valid


class SynchronizedVision:
    def __init__(self, env, config):
        from ..rendering import BARREL, EndoscopicView
        from ..scenario import SCISSORS_SCENES
        from ..grid import TO_GENESIS
        from .provenance import ROOT, sha256
        width, height = int(config.get('width',640)), int(config.get('height',480))
        self.every = int(config.get('every_steps',1))
        if width < 64 or height < 64 or self.every < 1:
            raise ValueError('Invalid image dimensions or sampling period')
        scene = getattr(env,'scene',SCISSORS_SCENES[env.scene_config.scene])
        self.view = EndoscopicView(env.settings,width,height,supersample=1,kidney=scene.shape,camera=scene.shot)
        camera = self.view.camera
        # Coordinates of each final image pixel in the upright pinhole image.
        xy, _ = cv2.convertMaps(*self.view.lens_map, cv2.CV_32FC2)
        xy[...,1] = height-1-xy[...,1]
        self.static_arrays = {'rgb_to_pinhole_xy':xy, 'camera_intrinsics':np.asarray(camera.intrinsics),
                              'native_to_genesis_rotation':TO_GENESIS}
        self.manifest = {'schema':'synchronous_endoscopic_rgbd_v1', 'width':width, 'height':height,
                         'every_steps':self.every, 'nominal_rate_hz':1/(self.every*env.settings.dt),
                         'time_source':'exact accepted native state step_id and t_sim_s',
                         'color_order':'RGB', 'depth_units':'m', 'depth_convention':'camera axial Z',
                         'depth_invalid_value':0, 'pixel_centers':'u+0.5,v+0.5 (Genesis convention)',
                         'intrinsics_domain':'upright undistorted pinhole; use stored remap for final RGB',
                         'extrinsics_convention':'Genesis world metres to OpenCV camera (+x right,+y down,+z forward)',
                         'barrel':BARREL, 'supersample':1, 'rgb_interpolation':'linear',
                         'depth_interpolation':'nearest, preserving depth discontinuities',
                         'rgb_postprocessing':'existing barrel warp and vignette; no overlay',
                         'alignment':'depth warped by same fixed-point pixel map as RGB',
                         'segmentation':'not exported; any future simulator masks are privileged truth',
                         'drop_policy':'no silent drops; capture error terminates observation',
                         'source_sha256':{p:sha256(ROOT/p) for p in
                             ('endoscopic_tissue/rendering.py','genesis/vis/camera.py')}}
        self._environment = weakref.ref(env)
        env._snapshot_vision = self

    def capture(self, env, state):
        step = state.metadata['step_id']
        if step % self.every:
            state.metadata['vision'] = {'scheduled':False, 'step_id':step, 't_sim_s':state.metadata['t_sim_s'],
                                         'reason':'declared_lower_camera_rate', 'dropped':False}
            return 0.
        started = time.perf_counter()
        surface = env.last_surface if env.last_surface is not None else env.native.surface()
        self.view.update(surface, env.observed)
        frame_bgr, pinhole = self.view.render(depth=True)
        depth, valid = align_depth_to_endoscope(pinhole, self.view.lens_map, self.view.vignette,
                                                self.view.camera.near, self.view.camera.far)
        arrays = {'rgb':cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB), 'depth_pinhole_m':pinhole,
                  'depth_rgb_aligned_m':depth, 'depth_valid':valid,
                  'camera_world_to_cv':np.asarray(self.view.camera.extrinsics).copy()}
        if frame_bgr.shape[:2] != pinhole.shape or not np.isfinite(pinhole).all():
            raise ValueError('RGB/depth dimensions or depth values are invalid')
        for name,value in arrays.items():
            value.flags.writeable = False
            state.arrays[name] = value
        elapsed = (time.perf_counter()-started)*1000
        state.metadata['vision'] = {'scheduled':True, 'step_id':step, 't_sim_s':state.metadata['t_sim_s'],
                                     'native_tick':state.metadata['native_tick'], 'dropped':False,
                                     'intrinsics_pinhole':self.static_arrays['camera_intrinsics'].tolist(),
                                     'world_to_cv':arrays['camera_world_to_cv'].tolist(),
                                     'render_ms':elapsed}
        return elapsed

    def close(self):
        self.view.close()
        env = self._environment()
        if env is not None and getattr(env, '_snapshot_vision', None) is self:
            env._snapshot_vision = None
