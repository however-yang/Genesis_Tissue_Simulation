"""Define end-effector motion targets without any solver or material parameters.

Command carries centimetre positions, degree orientations, closure fractions and enable switches.
Enable switches are passive data for external consumers; this module does not perform cutting.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """A single end-effector target used by the existing scripted motion path.

    Hinge is the grasper pivot; blade_center is the scissors pivot. Closure fractions run from open (0) to closed (1).
    Articulated motion generates JointCommand from the instrument instances instead.
    """
    hinge: tuple[float, float, float] = (-1.4, 3.384, -2.288)
    closure: float = 0.0
    blade_center: tuple[float, float, float] = (0.2, 2.2, -0.6)
    blade_enabled: bool = True
    scissors_closure: float = 0.0
    scissors_cutting: bool = False
    # Scissors orientation in degrees, see scissors_frame.
    scissors_pitch_deg: float = 19.9
    scissors_roll_deg: float = 0.0
    scissors_yaw_deg: float = 0.0
