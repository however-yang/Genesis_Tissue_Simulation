"""Advance one revolute control coordinate and construct its parent-to-child transform.

Angles use degrees, angular speeds use degrees per second, and joint origins use centimetres.
set_target records an endpoint; advance changes the commanded angle when time is explicitly advanced.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .config import JointConfig


@dataclass(frozen=True)
class JointState:
    """A sampled control coordinate; solved tissue feedback is stored separately."""
    angle_deg: float
    target_deg: float
    velocity_deg_s: float


class RevoluteJoint:
    """Own the target, commanded angle and rate limit of one rotational coordinate."""

    def __init__(self, name: str, config: JointConfig):
        self.name = name
        self.config = config
        self.axis = np.array(config.axis, dtype=float)
        self.axis /= np.linalg.norm(self.axis)
        # Keep the destination separate from the rate-limited control coordinate
        self.angle_deg = config.initial_deg
        self.target_deg = config.initial_deg
        self.velocity_deg_s = 0.0

    def validate_angle(self, angle_deg):
        """Reject a non-finite angle or one outside this joint's limits."""
        if (not np.isfinite(angle_deg)
                or not self.config.lower_deg - 1e-9 <= angle_deg <= self.config.upper_deg + 1e-9):
            raise ValueError(f"{self.name}: angle must be in [{self.config.lower_deg}, {self.config.upper_deg}] degrees")

    def set_target(self, angle_deg):
        """Set the destination angle without advancing the control coordinate."""
        self.validate_angle(angle_deg)
        self.target_deg = np.clip(angle_deg, self.config.lower_deg, self.config.upper_deg)

    def advance(self, dt_s):
        """Advance the commanded angle toward its target subject to the configured speed limit."""
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Joint timestep must be finite and positive")
        # This is a control slew limit; contact-resisted motion is solved by the native backend
        delta = np.clip(self.target_deg - self.angle_deg,
                        -self.config.max_speed_deg_s * dt_s, self.config.max_speed_deg_s * dt_s)
        self.angle_deg += delta
        self.velocity_deg_s = delta / dt_s

    def local_transform(self, *, angle_deg=None):
        """Return the parent-to-child transform for a commanded or supplied joint angle."""
        angle = self.angle_deg if angle_deg is None else angle_deg
        # The origin locates the pivot in the parent frame; rotation acts about that local pivot
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_rotvec(self.axis * np.radians(angle)).as_matrix()
        transform[:3, 3] = self.config.origin_cm
        return transform

    @property
    def state(self):
        return JointState(self.angle_deg, self.target_deg, self.velocity_deg_s)
