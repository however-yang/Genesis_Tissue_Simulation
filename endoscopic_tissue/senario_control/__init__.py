"""Scenario-level exports with instrument motion isolated in instrument_control."""
from importlib import import_module

from .commands import Command, Settings
from .config import ScenarioConfig, load_config
from .cutting import scissors_settings, set_cutting
from .instrument_control.articulation import JointState, RevoluteJoint
from .instrument_control.config import InstrumentConfig, JointConfig, MotionConfig
from .instrument_control.instruments import (
    Grasper, InstrumentFeedback, InstrumentPose, JointCommand, JointTargets, Scissors,
)
from .instrument_control.joints import scissors_frame, set_grasper, set_scissors, validate_command
from .instrument_control.smoothing import Stage, interpolate, joint_transition

__all__ = (
    "Command", "Settings", "Stage", "Kidney", "set_grasper", "set_scissors", "scissors_frame", "set_cutting",
    "scissors_settings", "validate_command", "kidney_stages", "interpolate", "stage_start", "drive_command",
    "Grasper", "Scissors", "RevoluteJoint", "JointState", "JointConfig", "InstrumentConfig", "ScenarioConfig",
    "MotionConfig", "InstrumentPose", "InstrumentFeedback", "JointCommand", "JointTargets", "load_config",
    "joint_transition",
)


def __getattr__(name):
    # Python imports the parent package first. Delay tissue-related exports so importing instrument_control
    # does not pull in the tissue grid or native observation types through the parent's convenience exports.
    if name in ("Kidney", "kidney_stages"):
        return getattr(import_module(".kidney", __name__), name)
    if name in ("stage_start", "drive_command"):
        return getattr(import_module(".motion_feedback", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
