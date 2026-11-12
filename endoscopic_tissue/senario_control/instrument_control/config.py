"""Load and validate the geometry, joint limits and initial setup of both instruments.

TOML sections become typed motion configuration objects. Scene selection and physical settings belong to the
parent package. Runtime joint angles and targets live in instrument instances.
"""
import tomllib
from dataclasses import dataclass, replace
from math import asin, degrees, sin
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class JointConfig:
    """A revolute control coordinate in its parent's native centimetre frame.

    Axis is normalized when constructing a joint. Angles use degrees and speed uses degrees per second.
    Opening denotes the full included angle between the two jaws.
    """
    axis: tuple[float, float, float]
    origin_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lower_deg: float = -90.0
    upper_deg: float = 90.0
    initial_deg: float = 0.0
    max_speed_deg_s: float = 90.0

    def __post_init__(self):
        for vector in (self.axis, self.origin_cm):
            if len(vector) != 3 or not np.isfinite(vector).all():
                raise ValueError("Joint axis and origin require three finite coordinates")
        values = (self.lower_deg, self.upper_deg, self.initial_deg, self.max_speed_deg_s)
        if not np.isfinite(values).all() or not self.lower_deg <= self.initial_deg <= self.upper_deg:
            raise ValueError("Joint initial angle must lie within finite limits")
        if np.linalg.norm(self.axis) < 1e-12 or self.max_speed_deg_s <= 0:
            raise ValueError("Joint axis must be nonzero and angular speed must be positive")


@dataclass(frozen=True)
class InstrumentConfig:
    """A Roll -> Pitch -> Yaw chain with a coupled opening coordinate at the Yaw pivot.

    At zero wrist angles the native local X axis points toward the tip and Y is the opening direction.
    Base position locates the Roll pivot. Base quaternion uses world-frame (x,y,z,w) order.
    """
    roll: JointConfig
    pitch: JointConfig
    yaw: JointConfig
    opening: JointConfig
    jaw_length_cm: float
    max_gap_cm: float
    shaft_length_cm: float
    base_position_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self):
        if len(self.base_position_cm) != 3 or not np.isfinite(self.base_position_cm).all():
            raise ValueError("Base position needs three finite coordinates")
        quaternion = self.base_quaternion_xyzw
        if len(quaternion) != 4 or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-12:
            raise ValueError("Base quaternion needs four finite coordinates and a nonzero norm")
        if not np.isfinite((self.jaw_length_cm, self.max_gap_cm, self.shaft_length_cm)).all():
            raise ValueError("Instrument dimensions must be finite")
        if min(self.jaw_length_cm, self.max_gap_cm, self.shaft_length_cm) <= 0:
            raise ValueError("Instrument dimensions must be positive")
        if not 0 <= self.opening.lower_deg <= self.opening.upper_deg < 180:
            raise ValueError("Opening angle limits must lie within [0,180) degrees")
        if np.linalg.norm(self.opening.origin_cm) > 1e-9:
            raise ValueError("Opening and Yaw must share the final joint pivot")
        opening_axis = np.array(self.opening.axis) / np.linalg.norm(self.opening.axis)
        if not np.allclose(opening_axis, (0, 0, 1)):
            raise ValueError("The current blade geometry requires local +Z as its opening axis")
        required_gap = 2 * self.jaw_length_cm * sin(np.radians(self.opening.upper_deg) / 2)
        # Both jaw tips must fit inside the configured gap at the largest included angle
        if required_gap > self.max_gap_cm + 1e-6 or self.max_gap_cm > 1.9 * self.jaw_length_cm:
            raise ValueError("Opening angle limits and maximum gap are inconsistent")


def grasper_config():
    """Return the current Cadiere dimensions and configurable 8 mm wrist geometry."""
    # Recover the full two-jaw angle from the current model's maximum tip separation
    max_angle = 2 * degrees(asin(1.0 / (2 * 1.99)))
    return InstrumentConfig(
        roll=JointConfig(axis=(1, 0, 0), lower_deg=-180, upper_deg=180),
        pitch=JointConfig(axis=(0, -1, 0), origin_cm=(.91, 0, 0)),
        yaw=JointConfig(axis=(0, 0, 1), origin_cm=(.91, 0, 0)),
        opening=JointConfig(axis=(0, 0, 1), lower_deg=0, upper_deg=max_angle, max_speed_deg_s=60),
        jaw_length_cm=1.99, max_gap_cm=1.0, shaft_length_cm=4.0,
    )


def scissors_config():
    """Return the current 179 blade dimensions with the same three-place wrist chain."""
    max_angle = degrees(1.39626)
    return InstrumentConfig(
        roll=JointConfig(axis=(1, 0, 0), lower_deg=-180, upper_deg=180),
        pitch=JointConfig(axis=(0, -1, 0), origin_cm=(.91, 0, 0)),
        yaw=JointConfig(axis=(0, 0, 1), origin_cm=(.91, 0, 0)),
        opening=JointConfig(axis=(0, 0, 1), lower_deg=0, upper_deg=max_angle, max_speed_deg_s=60),
        jaw_length_cm=1.305, max_gap_cm=2 * 1.305 * sin(1.39626 / 2), shaft_length_cm=6.0,
    )


@dataclass(frozen=True)
class MotionConfig:
    """Group both instruments' geometry and motion limits independently of a scene or solver."""
    grasper: InstrumentConfig
    scissors: InstrumentConfig


def update_instrument(default: InstrumentConfig, values):
    """Apply a parsed instrument section to a typed configuration, rejecting unknown fields."""
    remaining = dict(values)
    joints = {}
    for name in ("roll", "pitch", "yaw", "opening"):
        section = dict(remaining.pop(name, {}))
        for key in ("axis", "origin_cm"):
            if key in section:
                section[key] = tuple(section[key])
        default_joint = {"roll": default.roll, "pitch": default.pitch, "yaw": default.yaw,
                         "opening": default.opening}[name]
        joints[name] = replace(default_joint, **section)
    for key in ("base_position_cm", "base_quaternion_xyzw"):
        if key in remaining:
            remaining[key] = tuple(remaining[key])
    return replace(default, **remaining, **joints)


def load_config(path: str | Path):
    """Read only grasper and scissors motion sections from a TOML file.

    Missing instrument fields retain the matching defaults. Unsupported keys raise an error.
    Use the parent config module to load scene selection and physical settings.
    """
    with Path(path).open("rb") as source:
        values = tomllib.load(source)
    unknown = set(values) - {"grasper", "scissors"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    return MotionConfig(
        grasper=update_instrument(grasper_config(), values.get("grasper", {})),
        scissors=update_instrument(scissors_config(), values.get("scissors", {})),
    )
