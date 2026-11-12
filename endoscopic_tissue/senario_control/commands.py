"""Define physical session settings and expose the shared end-effector command type.

Solver parameters stay at scenario level. The instrument_control package uses only motion targets.
"""
from dataclasses import dataclass

from .instrument_control.commands import Command


@dataclass(frozen=True)
class Settings:
    """Physical and numerical parameters passed to the existing native session.

    Time uses seconds, geometry uses centimetres, density uses kg/m^3, mass uses kg and labelled forces use newtons.
    Remaining material and drive fields retain the native backend's established parameter conventions.
    """
    dt: float = 1 / 30
    substeps: int = 3
    young_pa: float = 6000
    poisson: float = 0.25
    strain_damping_s: float = 0.04
    density_kg_m3: float = 1050
    jaw_length: float = 1.99
    jaw_width: float = 0.52
    jaw_open_gap: float = 1.0
    jaw_axis: tuple[float, float, float] = (0.0, -0.8, 0.6)
    jaw_normal: tuple[float, float, float] = (1.0, 0.0, 0.0)
    blade_length: float = 1.6
    blade_radius: float = 0.04
    blade_axis: tuple[float, float, float] = (2 ** -.5, 0.0, 2 ** -.5)
    start_force_n: float = 0.2
    cut_force_n: float = 0.12
    blade_drive_stiffness: float = 1200
    blade_mass_kg: float = 0.02
    blade_friction: float = 0.15
    blade_preload_n: float = 1.0
    contact_refine_passes: int = 8
    has_knife_body: bool = True
    instrument_model: str = "davinci"
    cutting_tool: str = "knife"
    # Spring from the commanded to the solved tip gap, in native force per cm.
    # It sets how hard the closing jaw presses against tissue resistance.
    scissors_jaw_stiffness: float = 150
    # Tip closure in cm over which both blades must load the tissue before it
    # can fracture. A smaller travel cuts tissue that the blades have barely
    # deformed; a larger one squeezes it more, and the blades may close fully
    # before the travel is reached. A fractured snip whose tips close this far
    # without touching tissue has left it, and must squeeze new tissue again.
    scissors_squeeze_cm: float = .2
    # Grasper contact model. 1: the jaws hold tissue by normal pressure and
    # Coulomb friction alone, so tissue pulled harder than friction allows
    # slips out. 2: persistent compliant anchors also tie the bitten surface
    # to the jaws from capture until release.
    contact_model: int = 2


__all__ = ("Command", "Settings")
