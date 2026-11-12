"""Calculate cubic motion interpolation for end-effector and articulated joint targets.

Stage uses integer caller-defined steps. Positions use centimetres and orientations use degrees.
The caller advances joint states and consumes computed poses; this module never invokes a physics backend.
"""
from dataclasses import dataclass
from math import ceil

import numpy as np

from .commands import Command
from .joints import validate_command


@dataclass(frozen=True)
class Stage:
    name: str
    steps: int
    command: Command
    # Fraction of the stage over which the command reaches its target; the
    # rest of the stage holds it.
    ramp: float = 1.0

    def __post_init__(self):
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps <= 0:
            raise ValueError("Stage steps must be a positive integer")
        if not np.isfinite(self.ramp) or not 0 < self.ramp <= 1:
            raise ValueError("Stage ramp must be in (0,1]")
        validate_command(self.command)


def interpolate(a: Command, b: Command, fraction: float):
    """Blend continuous targets with cubic smoothstep and take enable switches from the target.

    Fraction is clipped to [0,1]. Positions and wrist angles retain their native centimetre and degree units.
    The curve has zero velocity at both endpoints; cutting switches take effect from the start of the stage.
    """
    if not np.isfinite(fraction):
        raise ValueError("Interpolation fraction must be finite")
    u = np.clip(fraction, 0, 1)
    # Cubic smoothstep has zero derivative at both endpoints
    u = u * u * (3 - 2 * u)
    hinge = tuple((1 - u) * np.array(a.hinge) + u * np.array(b.hinge))
    blade = tuple((1 - u) * np.array(a.blade_center) + u * np.array(b.blade_center))
    return Command(hinge, (1 - u) * a.closure + u * b.closure, blade, b.blade_enabled,
                   (1 - u) * a.scissors_closure + u * b.scissors_closure, b.scissors_cutting,
                   (1 - u) * a.scissors_pitch_deg + u * b.scissors_pitch_deg,
                   (1 - u) * a.scissors_roll_deg + u * b.scissors_roll_deg,
                   (1 - u) * a.scissors_yaw_deg + u * b.scissors_yaw_deg)


def joint_transition(instrument_targets, *, duration_s: float, dt_s: float):
    """Set successive smooth joint targets and yield their interpolation fraction.

    instrument_targets contains (instrument, JointTargets or None) pairs. Each iteration sets targets only;
    the caller must advance each instrument by dt_s before requesting the next sample. Duration is rounded up
    to whole timesteps, preserving the existing trajectory and peak angular-speed check.
    """
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Transition duration must be finite and positive")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Joint timestep must be finite and positive")
    steps = ceil(duration_s / dt_s)
    duration = steps * dt_s
    transitions = []
    for instrument, targets in instrument_targets:
        if targets is None:
            continue
        for joint, target in zip(instrument.joints, targets.angles):
            if target is not None:
                joint.validate_angle(target)
                # The peak derivative of cubic smoothstep is 1.5 / duration
                if 1.5 * abs(target - joint.angle_deg) / duration > joint.config.max_speed_deg_s + 1e-9:
                    raise ValueError(f"Transition duration is too short for {joint.name}'s angular speed limit")
                transitions.append((joint, joint.angle_deg, target))
    for index in range(steps):
        fraction = (index + 1) / steps
        fraction = fraction * fraction * (3 - 2 * fraction)
        for joint, start, target in transitions:
            joint.set_target(start + fraction * (target - start))
        yield fraction
