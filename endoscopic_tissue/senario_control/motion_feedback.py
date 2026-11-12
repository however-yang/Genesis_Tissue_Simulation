"""Apply solved contact feedback and actuator preload limits to scripted motion targets.

These operations depend on native observations and physical settings, so they stay outside instrument_control.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ..grid import METERS_PER_UNIT
from .commands import Settings
from .instrument_control.commands import Command

if TYPE_CHECKING:
    from ..native_backend import Observation


def stage_start(name: str, previous: Command, observed: Observation):
    """Start withdrawal at the observed scissors position; retain the previous command for other stages."""
    # Withdraw from the solved pose. Starting from the old drive target would
    # first push farther into tissue to catch up with an unfinished incision.
    if name.startswith("withdraw"):
        return replace(previous, blade_center=tuple(observed.blade[:3]))
    return previous


def drive_command(requested: Command, observed: Observation, settings: Settings):
    """Limit the scissors target offset from its solved pose using actuator preload."""
    # Bound actuator preload using the accepted pose, as in the Unity operator
    # controller. A blocked knife must not accumulate the rest of a scripted path.
    travel = np.array(requested.blade_center) - observed.blade[:3]
    limit = settings.blade_preload_n / (settings.blade_drive_stiffness * METERS_PER_UNIT)
    distance = np.linalg.norm(travel)
    center = observed.blade[:3] + travel * min(1, limit / max(distance, 1e-12))
    return replace(requested, blade_center=tuple(center))
