"""Update existing instruments once for each externally supplied timestep.

The caller owns instrument creation, configuration, time and scheduling. Position and linear velocity use
centimetres, angular coordinates and velocity use degrees, and dt_s uses seconds. Velocity vectors use world axes.
"""
from collections.abc import Mapping
from dataclasses import asdict
from math import isfinite
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation

from ..instrument_control.instruments import Grasper, Scissors


ANGLE_FIELDS = ("roll_deg", "pitch_deg", "yaw_deg", "opening_deg")
BASE_FIELDS = ("base_position_cm", "base_quaternion_xyzw")
VELOCITY_FIELDS = ("linear_velocity_cm_s", "angular_velocity_deg_s")


def finite_number(value, name):
    """Reject non-numeric and non-finite external values before changing instrument state."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def finite_vector(value, size, name):
    if (not isinstance(value, (list, tuple, np.ndarray))
            or isinstance(value, np.ndarray) and value.ndim != 1 or len(value) != size):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return tuple(finite_number(coordinate, name) for coordinate in value)


class MotionBridge:
    """Drive caller-owned instruments without constructing them or maintaining a simulation clock.

    submit stages input only. step consumes pose/angle updates and advances exactly one supplied dt_s.
    Linear and angular velocity commands persist until replaced or set to zero. Absolute pose input cancels
    the corresponding velocity command. snapshot returns detached JSON-compatible motion state.
    """

    def __init__(self, grasper: Grasper, scissors: Scissors):
        if not isinstance(grasper, Grasper) or not isinstance(scissors, Scissors):
            raise ValueError("Pass an existing Grasper and Scissors instance")
        self.grasper = grasper
        self.scissors = scissors
        self._commands = {"grasper": {}, "scissors": {}}
        self._cutting = None

    def prepare(self, payload):
        """Validate and merge staged input without changing either instrument or accepted commands."""
        if not isinstance(payload, Mapping):
            raise ValueError("Input must be a mapping")
        if any(not isinstance(field, str) for field in payload):
            raise ValueError("Input field names must be strings")
        unknown = set(payload) - {"grasper", "scissors", "is_cutting"}
        if unknown:
            raise ValueError(f"Unknown input fields: {sorted(unknown)}")
        cutting = payload.get("is_cutting", self._cutting)
        if "is_cutting" in payload and not isinstance(cutting, bool):
            raise ValueError("is_cutting must be a boolean")
        commands = {name: dict(section) for name, section in self._commands.items()}
        for name, instrument in (("grasper", self.grasper), ("scissors", self.scissors)):
            section = payload.get(name, {})
            if not isinstance(section, Mapping) or any(not isinstance(field, str) for field in section):
                raise ValueError(f"{name} must be a mapping with string field names")
            unknown = set(section) - set(ANGLE_FIELDS + BASE_FIELDS + VELOCITY_FIELDS)
            if unknown:
                raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
            for pose_field, velocity_field in zip(BASE_FIELDS, VELOCITY_FIELDS):
                if pose_field in section and velocity_field in section:
                    raise ValueError(f"Choose either {pose_field} or {velocity_field} for {name}")
                if pose_field in section:
                    commands[name].pop(velocity_field, None)
                if velocity_field in section:
                    commands[name].pop(pose_field, None)
            for field in BASE_FIELDS + VELOCITY_FIELDS:
                if field in section:
                    size = 4 if field == "base_quaternion_xyzw" else 3
                    vector = finite_vector(section[field], size, f"{name}.{field}")
                    if field == "base_quaternion_xyzw":
                        scale = max(abs(coordinate) for coordinate in vector)
                        if scale == 0:
                            raise ValueError(f"{name}.{field} must have a nonzero norm")
                        # Scaling before normalization avoids overflow for finite quaternion components.
                        quaternion = np.array(vector) / scale
                        vector = tuple(quaternion / np.linalg.norm(quaternion))
                    commands[name][field] = vector
            for field, joint in zip(ANGLE_FIELDS, instrument.joints):
                if field in section:
                    angle = finite_number(section[field], f"{name}.{field}")
                    joint.validate_angle(angle)
                    commands[name][field] = angle
        return commands, cutting

    def submit(self, payload):
        """Stage valid input; existing instrument coordinates remain unchanged until step."""
        self._commands, self._cutting = self.prepare(payload)

    def step(self, payload=None, *, dt_s: float):
        """Update both instruments once using the caller's timestep, with no internal loop or clock."""
        dt_s = finite_number(dt_s, "dt_s")
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        commands, cutting = self.prepare({} if payload is None else payload)
        prepared = []
        for name, instrument in (("grasper", self.grasper), ("scissors", self.scissors)):
            section = commands[name]
            targets = {field: section[field] for field in ANGLE_FIELDS if field in section}
            if not any(field in section for field in BASE_FIELDS + VELOCITY_FIELDS):
                prepared.append((instrument, None, None, targets))
                continue
            position = np.array(section.get("base_position_cm", instrument.base[:3, 3]), dtype=float)
            rotation = (Rotation.from_quat(section["base_quaternion_xyzw"]) if "base_quaternion_xyzw" in section
                        else Rotation.from_matrix(instrument.base[:3, :3]))
            with np.errstate(over="ignore", invalid="ignore"):
                position += np.array(section.get("linear_velocity_cm_s", (0, 0, 0))) * dt_s
                turn = np.radians(np.array(section.get("angular_velocity_deg_s", (0, 0, 0)))) * dt_s
            if not np.isfinite(position).all() or not np.isfinite(turn).all() or not isfinite(float(np.linalg.norm(turn))):
                raise ValueError(f"{name} motion over dt_s must remain finite")
            # World-frame angular velocity acts on the left of the current rotation.
            rotation = Rotation.from_rotvec(turn) * rotation
            prepared.append((instrument, position, rotation.as_quat(), targets))
        # Both pose updates are calculated and validated before either caller-owned instance changes.
        for instrument, position, quaternion, targets in prepared:
            if position is not None:
                instrument.set_base_pose(position, quaternion)
            instrument.set_joint_targets(**targets)
            instrument.advance(dt_s)
        if cutting is not None:
            self.scissors.set_cutting(cutting)
        self._commands = {name: {field: value for field, value in section.items() if field in VELOCITY_FIELDS}
                          for name, section in commands.items()}
        self._cutting = None
        return self.snapshot()

    def snapshot(self):
        """Read base/link poses and joint coordinates without advancing any state."""
        result = {"is_cutting": self.scissors.is_cutting}
        for name, instrument in (("grasper", self.grasper), ("scissors", self.scissors)):
            pose = instrument.pose()
            result[name] = {
                "base_transform": instrument.base.tolist(),
                "base_quaternion_xyzw": Rotation.from_matrix(instrument.base[:3, :3]).as_quat().tolist(),
                "joints": {field.removesuffix("_deg"): asdict(joint.state)
                           for field, joint in zip(ANGLE_FIELDS, instrument.joints)},
                "gap_cm": float(pose.gap_cm),
                "transforms": {field: getattr(pose, field).tolist()
                               for field in ("shaft", "roll", "pitch", "yaw", "upper_jaw", "lower_jaw")},
                "shaft_start_cm": pose.shaft_start_cm.tolist(),
                "shaft_end_cm": pose.shaft_end_cm.tolist(),
            }
        return result
