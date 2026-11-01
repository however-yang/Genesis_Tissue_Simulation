"""Explicit training experiments; canonical defaults are frozen at import time.

No changes to production Settings or native initialization are made here.
The contact model uses its own static/dynamic coefficients, not the legacy
GraspParams.mu alone. A friction sweep changes both with the original 2:3 ratio.
"""
from dataclasses import asdict, dataclass, fields, replace
import json
import math

from ..scenario import SCISSORS_SCENES, scissors_settings


_CANONICAL_SETTINGS_JSON = json.dumps({
    name: asdict(replace(scissors_settings(), young_pa=scene.young_pa,
                        contact_model=scene.contact_model))
    for name, scene in SCISSORS_SCENES.items()
}, sort_keys=True, allow_nan=False)
_NATIVE_DEFAULTS_JSON = json.dumps({
    "grasp_friction": .6, "grasp_dynamic_friction": .4,
    "tear_stretch_ratio": 0., "tear_exempt_rings": 1,
    "grasp_core_break_force_n": 150.,
}, sort_keys=True)


@dataclass(frozen=True)
class PhysicsOverrides:
    young_pa: float | None = None
    start_force_n: float | None = None
    blade_friction: float | None = None
    grasp_friction: float | None = None
    tear_stretch_ratio: float | None = None
    tear_exempt_rings: int | None = None
    grasp_core_break_force_n: float | None = None

    def __post_init__(self):
        ranges = {"young_pa": (3000., 24000.), "start_force_n": (.2, .6),
                  "blade_friction": (.05, .4), "grasp_friction": (.3, 1.2),
                  "grasp_core_break_force_n": (.1, 150.)}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if f.name == "tear_exempt_rings":
                if type(value) is not int or not 0 <= value <= 2:
                    raise ValueError("tear_exempt_rings must be an integer in [0,2]")
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{f.name} must be a finite number, not a bool/string")
            if f.name == "tear_stretch_ratio":
                if value != 0 and not 1.3 <= value <= 2.:
                    raise ValueError("tear_stretch_ratio must be 0 (disabled) or in [1.3,2.0]")
            else:
                low, high = ranges[f.name]
                if not low <= value <= high or (f.name == "grasp_core_break_force_n" and value == 0):
                    raise ValueError(f"{f.name} outside supported training experiment range {ranges[f.name]}")

    def requested(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


def canonical_settings(scene):
    """Owned copy; monkeypatching settings factories cannot rewrite the baseline."""
    return json.loads(_CANONICAL_SETTINGS_JSON)[scene]


def native_defaults():
    return json.loads(_NATIVE_DEFAULTS_JSON)


def resolve_physics(settings, overrides):
    if overrides is not None and not isinstance(overrides, PhysicsOverrides):
        raise TypeError("physics must be PhysicsOverrides or None")
    requested = overrides.requested() if overrides is not None else {}
    settings = replace(settings, **{k: v for k, v in requested.items()
                                    if k in ("young_pa", "start_force_n", "blade_friction")})
    native = native_defaults()
    native.update({k: v for k, v in requested.items() if k in native})
    if "grasp_friction" in requested:
        if settings.contact_model != 1:
            raise ValueError("grasp_friction override requires explicit pure-friction contact_model=1")
        # Preserve the baseline dynamic/static ratio. The gp.mu field by itself
        # is used only by the legacy slab solver and would be an ineffective sweep.
        native["grasp_dynamic_friction"] = .4 * (native["grasp_friction"] / .6)
    return settings, native


def physical_differences(settings, scene_config, native_parameters=None):
    """Compare every executed setting, including changes outside the public API."""
    scene = scene_config["scene"]
    actual = json.loads(json.dumps(asdict(settings), allow_nan=False))
    baseline = canonical_settings(scene)
    changes = {f"settings.{k}": {"canonical": baseline.get(k), "actual": v}
               for k, v in actual.items() if baseline.get(k) != v}
    native = native_parameters if native_parameters is not None else native_defaults()
    for key, default in native_defaults().items():
        if native[key] != default:
            changes[f"native.{key}"] = {"canonical": default, "actual": native[key]}
    if scene_config.get("spacing_scale", 1.) != 1.:
        changes["scene.spacing_scale"] = {"canonical": 1., "actual": scene_config["spacing_scale"]}
    if scene_config.get('geometry') is not None:
        changes['scene.geometry'] = {'canonical':None, 'actual':scene_config['geometry']}
    return changes
