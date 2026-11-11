"""Absolute end-effector targets in native world axes, metres and radians."""
from dataclasses import dataclass

import numpy as np

from ..native_backend import Command

ACTION_FIELDS = (
    "grasper_x_m", "grasper_y_m", "grasper_z_m", "grasper_closure",
    "scissors_x_m", "scissors_y_m", "scissors_z_m",
    "scissors_pitch_rad", "scissors_roll_rad", "scissors_yaw_rad", "scissors_closure",
)


def command_to_action(command):
    return np.array((*np.asarray(command.hinge) * .01, command.closure,
                     *np.asarray(command.blade_center) * .01,
                     *np.radians((command.scissors_pitch_deg, command.scissors_roll_deg,
                                  command.scissors_yaw_deg)), command.scissors_closure), dtype=np.float64)


def action_to_command(action, *, cutting_enabled):
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (11,) or not np.isfinite(a).all():
        raise ValueError("Action must contain exactly 11 finite SI targets")
    if not 0 <= a[3] <= 1 or not 0 <= a[10] <= 1:
        raise ValueError("Closure targets must be in [0,1]")
    return Command(tuple(a[:3] * 100), float(a[3]), tuple(a[4:7] * 100), True,
                   float(a[10]), bool(cutting_enabled), *np.degrees(a[7:10]))


@dataclass(frozen=True)
class ActionBounds:
    """Per-coordinate absolute bounds and target slew rates in SI units / s.

    Bounds are operational limits, not proof that the enclosed domain is valid.
    The grasper orientation and all physical/collision switches remain fixed.
    """
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    max_rate: tuple[float, ...]

    def __post_init__(self):
        lo, hi, rate = (np.asarray(v, dtype=float) for v in (self.lower, self.upper, self.max_rate))
        if any(v.shape != (11,) or not np.isfinite(v).all() for v in (lo, hi, rate)):
            raise ValueError("Action limits need 11 finite entries")
        if np.any(lo > hi) or np.any(rate < 0) or np.any(lo[[3, 10]] < 0) or np.any(hi[[3, 10]] > 1):
            raise ValueError("Invalid bounds, slew rate or closure range")

    def process(self, action, previous, dt):
        raw = np.asarray(action, dtype=float)
        if raw.shape != (11,) or not np.isfinite(raw).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Action must contain 11 finite targets; dt must be positive")
        previous = np.asarray(previous, dtype=float)
        if previous.shape != (11,) or not np.isfinite(previous).all():
            raise ValueError("Invalid previous target")
        lo, hi, rate = (np.asarray(v) for v in (self.lower, self.upper, self.max_rate))
        if np.any(previous < lo - 1e-10) or np.any(previous > hi + 1e-10):
            raise ValueError("Previous target lies outside configured action bounds")
        bounded = np.clip(raw, lo, hi)
        limited = previous + np.clip(bounded - previous, -rate * dt, rate * dt)
        return limited, {"policy": raw.tolist(), "bounded": bounded.tolist(),
                         "slew_limited": limited.tolist(),
                         "clipped_coordinates": np.flatnonzero(bounded != raw).tolist(),
                         "rate_limited_coordinates": np.flatnonzero(limited != bounded).tolist()}

