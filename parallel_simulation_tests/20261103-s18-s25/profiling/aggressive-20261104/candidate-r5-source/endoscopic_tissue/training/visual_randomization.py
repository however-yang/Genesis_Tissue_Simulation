"""Seeded opt-in appearance randomization, with no changes to physical state."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class VisualRandomization:
    camera_translation_m: float = 0.
    camera_target_m: float = 0.
    fov_delta_deg: float = 0.
    lamp_gain_delta: float = 0.
    roughness_scale_delta: float = 0.
    rgb_noise_std: float = 0.
    depth_noise_std_m: float = 0.

    def __post_init__(self):
        maxima = dict(camera_translation_m=.01,camera_target_m=.01,fov_delta_deg=15.,
                      lamp_gain_delta=.8,roughness_scale_delta=.8,rgb_noise_std=30.,depth_noise_std_m=.005)
        for key, maximum in maxima.items():
            value = getattr(self,key)
            if type(value) not in (int,float) or not np.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f'{key} must be finite and in [0,{maximum}]')

    def sample(self, seed, camera):
        if type(seed) is not int or seed < 0:raise ValueError('Explicit nonnegative visual seed required')
        rng=np.random.default_rng(seed)
        return dict(schema='visual_randomization_v1',seed=seed,configuration=asdict(self),
            camera_pos=(np.asarray(camera[0])+rng.uniform(-self.camera_translation_m,self.camera_translation_m,3)).tolist(),
            camera_target=(np.asarray(camera[1])+rng.uniform(-self.camera_target_m,self.camera_target_m,3)).tolist(),
            fov_deg=float(43+rng.uniform(-self.fov_delta_deg,self.fov_delta_deg)),
            lamp_gain=float(1+rng.uniform(-self.lamp_gain_delta,self.lamp_gain_delta)),
            roughness_scale=float(1+rng.uniform(-self.roughness_scale_delta,self.roughness_scale_delta)),
            noise_index='native_tick; snapshot restores preserve noise realization')

    def noise(self, rgb, depth, valid, *, seed, native_tick):
        if not self.rgb_noise_std and not self.depth_noise_std_m:return rgb,depth,valid
        rng=np.random.default_rng(np.random.SeedSequence([seed,int(native_tick)]))
        if self.rgb_noise_std:
            rgb=np.clip(np.rint(rgb.astype(float)+rng.normal(0,self.rgb_noise_std,rgb.shape)),0,255).astype(np.uint8)
        if self.depth_noise_std_m:
            depth=np.where(valid,depth+rng.normal(0,self.depth_noise_std_m,depth.shape),0).astype(np.float32)
            valid=valid & np.isfinite(depth) & (depth > 0)
            depth=np.where(valid,depth,0).astype(np.float32)
        return rgb,depth,valid
