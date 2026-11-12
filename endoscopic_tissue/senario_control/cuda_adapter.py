"""Bridge the new instrument instances to the existing process-global CUDA session.

Only endpoint position, orientation, gap and collision proxies are sent through the current native structures.
Intermediate wrist transforms remain available to callers for rendering and future articulated physics.
"""
import ctypes as ct
from dataclasses import replace

import numpy as np

from ..grid import METERS_PER_UNIT
from ..instrument_geometry import SCISSORS_WRIST_FRONT
from ..native_backend import Grasper, NativeTissue, ScissorsBack, ScissorsControl, V3
from .instrument_control.instruments import JointCommand


class CudaJointAdapter(NativeTissue):
    """Reuse the native session while accepting prescribed wrist-chain poses.

    Native mechanics solve translation, gap, contact and fracture. Wrist rotation is prescribed by forward
    kinematics; intermediate wrist-link inertias and motor torques require a further native extension.
    The scissors shaft/wrist collision proxy remains one capsule along the current tip axis.
    """

    def __init__(self, grid, settings, dll_path, *, config, initial_command):
        self.joint_config = config
        if not np.isclose(config.scissors.jaw_length_cm, 1.305):
            raise ValueError("The loaded 179 blades require jaw_length_cm=1.305")
        source_max_gap = 2 * 1.305 * np.sin(1.39626 / 2)
        if config.scissors.max_gap_cm > source_max_gap + 1e-6:
            raise ValueError("Scissors maximum gap exceeds the loaded blade model's opening limit")
        super().__init__(grid, settings, dll_path, initial_command=initial_command)
        is_ready = False
        try:
            self.grasper_geometry = replace(self.grasper_geometry, shaft_length=config.grasper.shaft_length_cm)
            jaws = replace(self.scissors_geometry.jaws, shaft_length=config.scissors.shaft_length_cm)
            self.scissors_geometry = replace(self.scissors_geometry, jaws=jaws)
            self.check("LCS_SetScissorsWrist", SCISSORS_WRIST_FRONT, jaws.shaft_length, jaws.shaft_radius)
            is_ready = True
        finally:
            if not is_ready:
                self.close()

    def limit_joint_command(self, command: JointCommand, observation):
        """Bound the articulated scissors drive offset while translating all of its links together."""
        travel = command.scissors.yaw[:3, 3] - observation.blade[:3]
        limit = self.settings.blade_preload_n / (self.settings.blade_drive_stiffness * METERS_PER_UNIT)
        distance = np.linalg.norm(travel)
        center = observation.blade[:3] + travel * min(1, limit / max(distance, 1e-12))
        return replace(command, scissors=command.scissors.translated(center - command.scissors.yaw[:3, 3]))

    def command(self, command):
        """Send either a legacy end-effector command or a new articulated command to CUDA."""
        if not isinstance(command, JointCommand):
            return super().command(command)
        settings = self.settings
        for pose in (command.grasper, command.scissors):
            matrices = (pose.shaft, pose.roll, pose.pitch, pose.yaw, pose.upper_jaw, pose.lower_jaw)
            if any(matrix.shape != (4, 4) or not np.isfinite(matrix).all() for matrix in matrices):
                raise ValueError("Instrument transforms must be finite 4x4 matrices")
            if any(not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
                   or np.linalg.det(matrix[:3, :3]) < 0 for matrix in matrices):
                raise ValueError("Instrument rotation matrices must be right-handed and orthonormal")
            if not np.isfinite(pose.gap_cm) or pose.gap_cm < 0:
                raise ValueError("Instrument gap must be finite and non-negative")
        grasper = command.grasper
        geometry = self.grasper_geometry
        axis, normal = grasper.yaw[:3, 0], grasper.yaw[:3, 1]
        hinge = grasper.yaw[:3, 3]
        max_gap = self.joint_config.grasper.max_gap_cm
        if grasper.gap_cm > max_gap + 1e-6:
            raise ValueError("Grasper gap exceeds its configured maximum")
        # The native grasper carries separate shaft endpoints and the distal jaw frame
        tool = Grasper(
            V3(*grasper.shaft_start_cm), V3(*grasper.shaft_end_cm), geometry.shaft_radius,
            V3(*hinge), V3(*axis), V3(*(hinge + settings.jaw_length / 2 * axis)), V3(*normal),
            settings.jaw_width / 2, settings.jaw_length, grasper.gap_cm,
            float(np.clip(1 - grasper.gap_cm / max_gap, 0, 1)),
        )
        self.check("LCS_SetGraspTool", ct.byref(tool))
        scissors = command.scissors
        geometry = self.scissors_geometry
        max_gap = self.joint_config.scissors.max_gap_cm
        if scissors.gap_cm > max_gap + 1e-6:
            raise ValueError("Scissors gap exceeds its configured maximum")
        backs = tuple(ScissorsBack(*(V3(*vector) for vector in box.reshape(4, 3))) for box in geometry.backs)
        # Geometry and gaps use centimetres; external newton thresholds are converted to native force units
        control = ScissorsControl(
            V3(*scissors.yaw[:3, 3]), V3(*scissors.yaw[:3, 0]), V3(*scissors.yaw[:3, 1]),
            *(V3(*vector) for vector in geometry.edges.reshape(4, 3)),
            geometry.jaws.length, scissors.gap_cm, max_gap, 1.2,
            settings.start_force_n / METERS_PER_UNIT, settings.cut_force_n / METERS_PER_UNIT,
            settings.blade_drive_stiffness, settings.blade_mass_kg, settings.blade_friction,
            .005, settings.scissors_jaw_stiffness, 1, int(command.is_cutting), 0,
            *backs, 1, settings.scissors_squeeze_cm,
        )
        self.check("LCS_SetScissors", ct.byref(control))
