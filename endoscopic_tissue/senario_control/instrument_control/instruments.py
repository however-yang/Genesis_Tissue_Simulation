"""Calculate each instrument's base, four control coordinates and link transforms.

The parent chain is base -> Roll -> Pitch -> Yaw. Opening splits the two jaws around the final pivot.
Forward kinematics produces motion targets. Optional numeric feedback is recorded from an external caller;
this module does not load a solver or calculate contact forces.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from .articulation import RevoluteJoint
from .config import InstrumentConfig, grasper_config, scissors_config


@dataclass(frozen=True)
class InstrumentPose:
    """World transforms in centimetres for the shaft, three wrist links and two jaws.

    These are commanded kinematic transforms. Contact-resisted hinge and gap are exposed through InstrumentFeedback.
    """
    shaft: np.ndarray
    roll: np.ndarray
    pitch: np.ndarray
    yaw: np.ndarray
    upper_jaw: np.ndarray
    lower_jaw: np.ndarray
    gap_cm: float
    shaft_start_cm: np.ndarray
    shaft_end_cm: np.ndarray

    def translated(self, delta_cm):
        """Translate every link together while preserving the wrist articulation."""
        displacement = np.eye(4)
        displacement[:3, 3] = delta_cm
        return replace(self, shaft=displacement @ self.shaft, roll=displacement @ self.roll,
                       pitch=displacement @ self.pitch, yaw=displacement @ self.yaw,
                       upper_jaw=displacement @ self.upper_jaw, lower_jaw=displacement @ self.lower_jaw,
                       shaft_start_cm=self.shaft_start_cm + delta_cm, shaft_end_cm=self.shaft_end_cm + delta_cm)


@dataclass(frozen=True)
class InstrumentFeedback:
    """Externally supplied hinge position, tip separation and inferred included angle."""
    hinge_cm: tuple[float, float, float]
    gap_cm: float
    opening_deg: float


class ArticulatedInstrument:
    """Own independent base-pose and Roll, Pitch, Yaw, opening control states."""

    def __init__(self, name: str, config: InstrumentConfig):
        self.name = name
        self.config = config
        self.roll = RevoluteJoint(f"{name}.roll", config.roll)
        self.pitch = RevoluteJoint(f"{name}.pitch", config.pitch)
        self.yaw = RevoluteJoint(f"{name}.yaw", config.yaw)
        self.opening = RevoluteJoint(f"{name}.opening", config.opening)
        self.base = np.eye(4)
        self.set_base_pose(config.base_position_cm, config.base_quaternion_xyzw)
        self.feedback = None

    @property
    def joints(self):
        return self.roll, self.pitch, self.yaw, self.opening

    def set_base_pose(self, position_cm, quaternion_xyzw=None):
        """Move the whole chain in native world axes, optionally changing the mounting orientation."""
        position = np.asarray(position_cm, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Base position must contain three finite centimetre coordinates")
        rotation = self.base[:3, :3]
        if quaternion_xyzw is not None:
            quaternion = np.asarray(quaternion_xyzw, dtype=float)
            if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-12:
                raise ValueError("Base quaternion must be finite and nonzero in (x,y,z,w) order")
            rotation = Rotation.from_quat(quaternion).as_matrix()
        self.base[:3, 3] = position
        self.base[:3, :3] = rotation

    def set_joint_targets(self, *, roll_deg=None, pitch_deg=None, yaw_deg=None, opening_deg=None):
        """Set any subset of joint targets after validating the complete update."""
        targets = (roll_deg, pitch_deg, yaw_deg, opening_deg)
        for joint, target in zip(self.joints, targets):
            if target is not None:
                joint.validate_angle(target)
        for joint, target in zip(self.joints, targets):
            if target is not None:
                joint.set_target(target)

    def advance(self, dt_s):
        """Advance all four rate-limited control coordinates by one timestep."""
        for joint in self.joints:
            joint.advance(dt_s)

    def pose(self):
        """Compute parent-child transforms, including two opposite half-opening rotations."""
        # Column-vector transforms are multiplied from parent to child
        roll = self.base @ self.roll.local_transform()
        pitch = roll @ self.pitch.local_transform()
        yaw = pitch @ self.yaw.local_transform()
        # Opening is the full included angle; each jaw turns by one opposite half-angle
        gap = 2 * self.config.jaw_length_cm * np.sin(np.radians(self.opening.angle_deg) / 2)
        upper = yaw @ self.opening.local_transform(angle_deg=self.opening.angle_deg / 2)
        lower = yaw @ self.opening.local_transform(angle_deg=-self.opening.angle_deg / 2)
        wrist_span = np.linalg.norm(np.array(self.config.pitch.origin_cm) + np.array(self.config.yaw.origin_cm))
        exposed_length = max(0.0, self.config.shaft_length_cm - wrist_span)
        shaft_start = roll[:3, 3] - self.base[:3, 0] * exposed_length
        return InstrumentPose(self.base.copy(), roll, pitch, yaw, upper, lower, gap, shaft_start, roll[:3, 3].copy())

    def align_hinge_pose(self, position_cm, rotation_matrix, gap_cm):
        """Initialize the base around a supplied hinge pose while retaining the configured wrist angles.

        This operation places a new instance at a legacy scene pose. Subsequent motion uses joint targets.
        """
        local = self.roll.local_transform() @ self.pitch.local_transform() @ self.yaw.local_transform()
        desired = np.eye(4)
        desired[:3, :3] = rotation_matrix
        desired[:3, 3] = position_cm
        angle = np.degrees(2 * np.arcsin(np.clip(gap_cm / (2 * self.config.jaw_length_cm), 0, .95)))
        self.opening.validate_angle(angle)
        self.opening.angle_deg = self.opening.target_deg = angle
        self.opening.velocity_deg_s = 0.0
        # Remove the local chain from the supplied world endpoint to recover its mounting base
        self.base = desired @ np.linalg.inv(local)

    def observe(self, hinge_cm, gap_cm):
        """Store contact-resisted hinge and opening feedback separately from control targets."""
        hinge = np.asarray(hinge_cm, dtype=float)
        gap = np.asarray(gap_cm, dtype=float)
        if hinge.shape != (3,) or not np.isfinite(hinge).all():
            raise ValueError("Feedback hinge must contain three finite centimetre coordinates")
        if gap.shape != () or not np.isfinite(gap) or not 0 <= gap <= 2 * self.config.jaw_length_cm:
            raise ValueError("Feedback gap must be finite and between zero and twice the jaw length")
        # Validate the measured geometry before replacing the last accepted feedback.
        # A measurement remains separate from the commanded angle and its configured control range.
        angle = np.degrees(2 * np.arcsin(float(gap) / (2 * self.config.jaw_length_cm)))
        self.feedback = InstrumentFeedback(tuple(hinge.tolist()), float(gap), float(angle))


class Grasper(ArticulatedInstrument):
    """Create an independent Cadiere grasper with its own four joint-control instances."""
    def __init__(self, config: InstrumentConfig | None = None):
        super().__init__("grasper", grasper_config() if config is None else config)


class Scissors(ArticulatedInstrument):
    """Create independent scissors with four joint-control instances and a passive cutting switch."""
    def __init__(self, config: InstrumentConfig | None = None):
        super().__init__("scissors", scissors_config() if config is None else config)
        self.is_cutting = False

    def set_cutting(self, is_cutting: bool):
        """Record an external cutting switch; no cutting is computed by this instrument instance."""
        if not isinstance(is_cutting, bool):
            raise ValueError("Cutting permission must be a boolean")
        self.is_cutting = is_cutting


@dataclass(frozen=True)
class JointCommand:
    """Pair the requested link poses of both instruments with the scissors fracture switch."""
    grasper: InstrumentPose
    scissors: InstrumentPose
    is_cutting: bool


@dataclass(frozen=True)
class JointTargets:
    """Optional destinations for the four independent coordinates, in degrees."""
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    opening_deg: float | None = None

    @property
    def angles(self):
        return self.roll_deg, self.pitch_deg, self.yaw_deg, self.opening_deg
