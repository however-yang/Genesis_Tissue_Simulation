"""Control fracture permission and provide the demonstrated scissors physical defaults.

Jaw opening and fracture permission are separate inputs. The native solver uses contact, compression and squeeze
to decide when tissue fractures; this module supplies control data and parameters.
"""
from dataclasses import replace

from .commands import Settings
from .instrument_control.commands import Command
from .instrument_control.joints import validate_command


def set_cutting(command: Command, *, is_cutting: bool):
    """Return a command with scissors fracture enabled or disabled.

    Blade closure is controlled separately in joints. Actual fracture depends on contact, load and tissue squeeze
    in the native solver; the switch permits cutting rather than forcing a cut.
    """
    target = replace(command, scissors_cutting=is_cutting)
    validate_command(target)
    return target


def scissors_settings():
    # The grasper enters from the upper left. Its jaws close along z: a bite
    # that squeezes the block from both sides lifts it reliably, while jaws
    # opening in the x-y plane mostly drag the surface by friction.
    # The jaw drive closes the first snip fully. In the later snips of the wall
    # the far blade body, below the shear plane, sweeps along the uncut lower
    # wall, which no edge can cut, and holds the jaws 1.5 to 2.5 mm open
    # whatever the drive: a stiffer drive only presses the blade back into it.
    # The scissors hold their pose against up to 5 N: a far blade driving uncut
    # wall along its length meets 1.5 to 6 N and would push a 1 N drive several
    # millimetres off its bite, short of the planned cut.
    # Three solver substeps per frame. With one, the third snip fractures with
    # 2-3 N on one blade instead of 0.1-0.4 N, some snips after less than
    # 0.1 mm of two-blade squeeze, and the blade backs overlap the tissue by
    # more than 0.2 mm: the squeeze rule and the overlap acceptance fail. With
    # two, the blade backs overlap the wall by 0.21-0.34 mm and the flap's
    # grasp keeps less than half of its anchors.
    return Settings(substeps=3, cutting_tool="scissors", jaw_axis=(.6, -.8, 0), jaw_normal=(0, 0, 1),
                    blade_preload_n=5.0, scissors_jaw_stiffness=2500)
