"""Coordinate config loading, both instrument instances, motion stepping and native feedback.

The joint path calls advance -> forward kinematics -> preload limiting -> native physics -> feedback.
The existing end-effector path accepts Command and Stage data. Creating a session loads the native library;
use a context manager to release its process-global resources.
"""
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path

import numpy as np

from ..native_backend import Observation, Surface
from ..scenario import SCISSORS_SCENES
from .commands import Command, Settings
from .config import ScenarioConfig, load_config
from .cuda_adapter import CudaJointAdapter
from .cutting import scissors_settings
from .instrument_control.config import MotionConfig, grasper_config, scissors_config
from .instrument_control.instruments import Grasper, JointCommand, JointTargets, Scissors
from .instrument_control.joints import scissors_frame, validate_command
from .instrument_control.smoothing import Stage, interpolate, joint_transition
from .kidney import KIDNEY, kidney_stages
from .motion_feedback import drive_command, stage_start


@dataclass(frozen=True)
class SimulationState:
    """One accepted physics step, with requested and applied actuator commands.

    Time uses seconds. Observation nodes and surface vertices use native world centimetres.
    Surface is populated when requested; observation always includes tissue nodes and instrument feedback.
    """
    step: int
    time_s: float
    stage: str
    requested: Command | JointCommand
    command: Command | JointCommand
    observation: Observation
    surface: Surface | None


class SimulationInterface:
    """Own a single scissors simulation with explicit stepping and resource cleanup.

    Scene selects kidney, wall, wall-traction or flap. An explicit Settings instance replaces scene defaults.
    Supply a compatible native library through dll_path, or use the repository default.
    Use a context manager to close the process-global native session.
    """

    def __init__(self, scene=None, *, settings: Settings | None = None, dll_path: Path | None = None,
                 config: ScenarioConfig | MotionConfig | str | Path | None = None):
        if isinstance(config, (str, Path)):
            config = load_config(config)
        if isinstance(config, MotionConfig):
            config = ScenarioConfig("kidney" if scene is None else scene, config.grasper, config.scissors)
        scene = (config.scene if config is not None else "kidney") if scene is None else scene
        if scene not in SCISSORS_SCENES:
            raise ValueError(f"Unknown scissors scene: {scene}")
        self.scene = SCISSORS_SCENES[scene]
        if scene == "kidney":
            self.scene = replace(self.scene, shape=KIDNEY, stages=kidney_stages())
        if config is None:
            config = ScenarioConfig(scene, grasper_config(), scissors_config())
            has_configured_pose = False
        else:
            has_configured_pose = True
        if settings is not None and config.physics is not None:
            raise ValueError("Supply physical settings through either settings or config.physics")
        self.config = config
        self.settings = settings if settings is not None else config.physics
        if self.settings is None:
            self.settings = replace(scissors_settings(), young_pa=self.scene.young_pa,
                                    contact_model=self.scene.contact_model)
        self.settings = replace(self.settings, jaw_length=config.grasper.jaw_length_cm,
                                jaw_open_gap=config.grasper.max_gap_cm)
        if self.settings.cutting_tool != "scissors":
            raise ValueError("SimulationInterface requires cutting_tool='scissors'")
        if not np.isfinite(self.settings.dt) or self.settings.dt <= 0:
            raise ValueError("Simulation timestep must be finite and positive")
        if (not np.isfinite(self.settings.blade_drive_stiffness) or self.settings.blade_drive_stiffness <= 0
                or not np.isfinite(self.settings.blade_preload_n) or self.settings.blade_preload_n < 0):
            raise ValueError("Blade stiffness must be positive and preload must be non-negative")
        self.grid = self.scene.build_grid(self.settings.density_kg_m3)
        self.grasper = Grasper(config.grasper)
        self.scissors = Scissors(config.scissors)
        self.requested = self.scene.stages[0].command
        self.command = self.requested
        if has_configured_pose:
            self.requested = self.command = JointCommand(self.grasper.pose(), self.scissors.pose(), False)
        else:
            axis, normal = np.array(self.settings.jaw_axis), np.array(self.settings.jaw_normal)
            self.grasper.align_hinge_pose(self.command.hinge, np.column_stack((axis, normal, np.cross(axis, normal))),
                                         self.settings.jaw_open_gap * (1 - self.command.closure))
            axis, normal = scissors_frame(self.command.scissors_pitch_deg, self.command.scissors_roll_deg,
                                          self.command.scissors_yaw_deg)
            self.scissors.align_hinge_pose(self.command.blade_center,
                                          np.column_stack((axis, normal, np.cross(axis, normal))),
                                          config.scissors.max_gap_cm * (1 - self.command.scissors_closure))
        self.step_count = 0
        self.native = CudaJointAdapter(self.grid, self.settings, dll_path, config=config, initial_command=self.command)
        self.observation = None
        is_ready = False
        try:
            self.observation = self.native.observe()
            self.update_joint_feedback()
            is_ready = True
        finally:
            if not is_ready:
                self.native.close()

    def step(self, command: Command | None = None, *, stage="external", include_surface=False):
        """Advance one physics timestep and return instrument and tissue feedback.

        Omit command to advance the articulated instrument instances. An explicit command uses the legacy
        end-effector mode. Set include_surface to read the current mesh for rendering or export.
        """
        if command is None:
            return self.step_joints(stage=stage, include_surface=include_surface)
        if self.native.closed:
            raise RuntimeError("Simulation interface is closed")
        if isinstance(self.command, JointCommand):
            raise ValueError("This session uses articulated control; use step_joints or transition_joints")
        validate_command(command)
        applied = drive_command(command, self.observation, self.settings)
        observation = self.native.step(applied)
        self.requested, self.command, self.observation = command, applied, observation
        self.update_joint_feedback()
        self.step_count += 1
        return SimulationState(self.step_count, self.step_count * self.settings.dt, stage, command, applied, observation,
                               self.native.surface() if include_surface else None)

    def update_joint_feedback(self):
        """Update solved hinge/gap feedback while retaining each joint's prescribed control state."""
        self.grasper.observe(self.observation.hinge, self.observation.jaw_gap)
        if self.observation.scissors is not None:
            self.scissors.observe(self.observation.scissors.hinge, self.observation.scissors.gap)

    def step_joints(self, *, stage="joints", include_surface=False):
        """Advance both instrument instances and solve tissue contact for their articulated targets."""
        if self.native.closed:
            raise RuntimeError("Simulation interface is closed")
        self.grasper.advance(self.settings.dt)
        self.scissors.advance(self.settings.dt)
        # Build a synchronized target from the independently advanced instrument instances
        requested = JointCommand(self.grasper.pose(), self.scissors.pose(), self.scissors.is_cutting)
        applied = self.native.limit_joint_command(requested, self.observation)
        # The returned observation is authoritative for actual hinge positions and gaps
        observation = self.native.step(applied)
        self.requested, self.command, self.observation = requested, applied, observation
        self.update_joint_feedback()
        self.step_count += 1
        return SimulationState(self.step_count, self.step_count * self.settings.dt, stage, requested, applied, observation,
                               self.native.surface() if include_surface else None)

    def transition_joints(self, *, duration_s: float, grasper: JointTargets | None = None,
                          scissors: JointTargets | None = None, include_surface=False):
        """Yield a cubic smoothstep transition of supplied joint coordinates.

        Duration must allow the peak cubic speed to stay within every participating joint's configured limit.
        Omitted coordinates retain their current control targets.
        """
        # The same trajectory can be consumed by a motion-only caller without constructing this session.
        for _ in joint_transition(((self.grasper, grasper), (self.scissors, scissors)),
                                  duration_s=duration_s, dt_s=self.settings.dt):
            yield self.step_joints(stage="joint_transition", include_surface=include_surface)

    def transition(self, target: Command, duration_s: float, *, stage="external", include_surface=False):
        """Yield accepted steps of a smooth transition from the current command to a target.

        Duration is rounded up to whole physics timesteps. A withdrawal stage starts its scissors motion at the
        observed pose, using the same rule as the existing demonstration.
        """
        validate_command(target)
        if isinstance(self.command, JointCommand):
            raise ValueError("Use transition_joints for articulated control")
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Transition duration must be finite and positive")
        steps = ceil(duration_s / self.settings.dt)
        yield from self.run_stages((Stage(stage, steps, target),), include_surface=include_surface)

    def run_stages(self, stages=None, *, include_surface=False):
        """Yield one accepted state per step of the supplied or scene-default sequence.

        Stages are executed from the current applied command; each completed stage sets the next interpolation
        start to its requested endpoint, matching the existing demonstration. Consume the iterator to advance time.
        """
        stages = self.scene.stages if stages is None else stages
        if isinstance(self.command, JointCommand):
            raise ValueError("Use joint targets for articulated control; run_stages consumes legacy end-effector targets")
        previous = self.command
        for stage in stages:
            previous = stage_start(stage.name, previous, self.observation)
            for local_step in range(stage.steps):
                requested = interpolate(previous, stage.command, (local_step + 1) / (stage.steps * stage.ramp))
                yield self.step(requested, stage=stage.name, include_surface=include_surface)
            previous = stage.command

    def surface(self):
        """Read the currently deformed and cut tissue mesh in native centimetres."""
        if self.native.closed:
            raise RuntimeError("Simulation interface is closed")
        return self.native.surface()

    def close(self):
        """Release the native session so another simulation can be opened."""
        self.native.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
