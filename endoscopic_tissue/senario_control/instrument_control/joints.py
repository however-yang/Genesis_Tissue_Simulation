"""Validate and update end-effector commands used by the scripted motion path.

Positions use native centimetres, orientations use degrees, and closures are fractions in [0,1].
RevoluteJoint instances are defined in articulation.py; these helpers return command data.
"""
from dataclasses import replace

import numpy as np

from .commands import Command


# Offset of the blade shear face from the hinge, in centimetres
SCISSORS_FACE_WIDTH = .0282


def validate_command(command: Command):
    """Check finite actuator targets, three-coordinate positions and closure limits."""
    if len(command.hinge) != 3 or len(command.blade_center) != 3:
        raise ValueError("Instrument positions need three coordinates in centimetres")
    if not np.isfinite((*command.hinge, *command.blade_center, command.closure, command.scissors_closure,
                        command.scissors_pitch_deg, command.scissors_roll_deg, command.scissors_yaw_deg)).all():
        raise ValueError("Tool command contains non-finite values")
    if not 0 <= command.closure <= 1 or not 0 <= command.scissors_closure <= 1:
        raise ValueError("Jaw closure must be in [0,1]")
    if not isinstance(command.blade_enabled, (bool, np.bool_)) or not isinstance(command.scissors_cutting, (bool, np.bool_)):
        raise ValueError("Instrument enable and cutting switches must be booleans")


def set_grasper(command: Command, *, position_cm=None, closure=None):
    """Return a command with updated grasper hinge position and jaw closure.

    Position uses native world centimetres, Y up. Closure ranges from zero (open) to one (closed).
    Omitted targets retain their current values.
    """
    target = replace(command, hinge=command.hinge if position_cm is None else tuple(position_cm),
                     closure=command.closure if closure is None else closure)
    validate_command(target)
    return target


def set_scissors(command: Command, *, position_cm=None, pitch_deg=None, roll_deg=None, yaw_deg=None, closure=None):
    """Return a command with updated scissors hinge, wrist angles and blade closure.

    Position uses native world centimetres, angles use the existing pitch/roll/yaw convention in degrees.
    Closure ranges from zero (open) to one (closed). Omitted targets retain their current values.
    """
    target = replace(
        command,
        blade_center=command.blade_center if position_cm is None else tuple(position_cm),
        scissors_pitch_deg=command.scissors_pitch_deg if pitch_deg is None else pitch_deg,
        scissors_roll_deg=command.scissors_roll_deg if roll_deg is None else roll_deg,
        scissors_yaw_deg=command.scissors_yaw_deg if yaw_deg is None else yaw_deg,
        scissors_closure=command.scissors_closure if closure is None else closure,
    )
    validate_command(target)
    return target


def scissors_frame(pitch_deg, roll_deg=0.0, yaw_deg=0.0):
    """Return the scissors axis and opening normal in native coordinates.

    At zero angles the scissors point toward native -x and the first blade opens toward +y. The roll turns the opening
    normal about the axis toward +z, the pitch then tilts the axis down, and the yaw turns it about +y toward +z.
    """
    roll, pitch, yaw = np.radians((roll_deg, pitch_deg, yaw_deg))
    # Preserve the established end-effector convention when converting angles to world axes
    turn_roll = np.array(((1, 0, 0), (0, np.cos(roll), -np.sin(roll)), (0, np.sin(roll), np.cos(roll))))
    turn_pitch = np.array(((np.cos(pitch), -np.sin(pitch), 0), (np.sin(pitch), np.cos(pitch), 0), (0, 0, 1)))
    turn_yaw = np.array(((np.cos(yaw), 0, np.sin(yaw)), (0, 1, 0), (-np.sin(yaw), 0, np.cos(yaw))))
    rotation = turn_yaw @ turn_pitch @ turn_roll
    return rotation @ np.array((-1., 0, 0)), rotation @ np.array((0, 1., 0))
