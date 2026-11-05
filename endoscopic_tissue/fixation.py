"""Choose which tissue nodes the solver holds still.

The grid is built with its bed already pinned: the bottom face of a block, or the fatty bed of a kidney. A demo can
replace that attachment with one surface point or with the surface nodes inside a box before the native session starts.
The session copies the pin flags once, at initialization.
"""
from dataclasses import dataclass

import numpy as np


def _coordinates(values, message):
    point = np.array(values, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(message)
    return tuple(float(component) for component in point)


@dataclass(frozen=True)
class PointFixation:
    """Hold the tissue at the surface node nearest one point, in centimetres."""

    point: tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "point", _coordinates(self.point, "a fixation point has 3 finite coordinates"))


@dataclass(frozen=True)
class RegionFixation:
    """Hold the surface nodes inside an axis-aligned box, in centimetres.

    The box is the patch that stays still. A face patch is a box about one cell thick lying on that face. Surface nodes
    outside the box are released, including the scene bed.
    """

    low: tuple[float, float, float]
    high: tuple[float, float, float]

    def __post_init__(self):
        low = _coordinates(self.low, "a fixation region needs a finite low corner")
        high = _coordinates(self.high, "a fixation region needs a finite high corner")
        if np.any(np.array(high) - np.array(low) <= 1e-6):
            raise ValueError("a fixation region must be thicker than 1e-6 cm on every axis")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)


def bed_pins(positions, bed, active):
    """Return the pin flags of a tissue bed: the active nodes `bed` selects."""
    return (bed(positions) & (active != 0)).astype(np.int32)


def surface_node_mask(grid):
    """Return the active nodes that have a free neighbor, the lattice surface a fixation can hold."""
    return (grid.active != 0) & (grid.neighbors < 0).any(axis=1)


def apply_fixation(grid, fixation):
    """Replace the grid pins with the nodes `fixation` holds, and return how many.

    A point holds one surface node, the nearest to its coordinates, when that node lies within one cell diagonal.
    A region holds every surface node inside its box. The previous attachment, including the scene bed, is cleared.
    """
    if isinstance(fixation, PointFixation):
        selected = np.zeros(len(grid.positions), dtype=bool)
        selected[_nearest_surface_node(grid, fixation.point)] = True
    elif isinstance(fixation, RegionFixation):
        low, high = np.array(fixation.low), np.array(fixation.high)
        inside = np.all(grid.positions >= low - 1e-4, axis=1) & np.all(grid.positions <= high + 1e-4, axis=1)
        selected = surface_node_mask(grid) & inside
        if not np.any(selected):
            raise ValueError("the fixation region contains no tissue surface")
    else:
        raise TypeError("fixation must be a point or a region")
    grid.pinned[:] = selected.astype(np.int32)
    return int(selected.sum())


def _nearest_surface_node(grid, point):
    candidates = np.flatnonzero(surface_node_mask(grid))
    if len(candidates) == 0:
        raise ValueError("the tissue has no surface node to hold")
    distance = np.linalg.norm(grid.positions[candidates] - np.array(point), axis=1)
    nearest = int(np.argmin(distance))
    if distance[nearest] > float(np.linalg.norm(grid.cell_size)):
        raise ValueError("the fixation point is more than one cell away from the tissue surface")
    return int(candidates[nearest])


def add_fixation_arguments(parser):
    """Add the shared fixation choice to a demo command line.

    bed keeps the attachment the scene was built with. point and region replace it, so the tissue is held only where
    the new fixation says.
    """
    parser.add_argument("--fixation", choices=("bed", "point", "region"), default="bed",
                        help="How the tissue is held. bed keeps the scene attachment. point holds the surface node "
                             "nearest --fixation-point and releases that attachment, so the tissue can swing about the "
                             "point. region holds the surface nodes inside the box from --fixation-low to "
                             "--fixation-high and releases the attachment. A face patch is a box about one cell thick "
                             "on that face. Coordinates are centimetres.")
    parser.add_argument("--fixation-point", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Point fixation target, centimetres")
    parser.add_argument("--fixation-low", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Low corner of a region fixation box, centimetres")
    parser.add_argument("--fixation-high", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="High corner of a region fixation box, centimetres")


def fixation_from_args(parser, args):
    """Return the fixation selected on the command line, or None to keep the scene bed."""
    if args.fixation == "bed":
        if args.fixation_point is not None or args.fixation_low is not None or args.fixation_high is not None:
            parser.error("bed fixation takes no coordinates")
        return None
    if args.fixation == "point":
        if args.fixation_point is None or args.fixation_low is not None or args.fixation_high is not None:
            parser.error("point fixation needs --fixation-point and no region box")
        return PointFixation(tuple(args.fixation_point))
    if args.fixation_low is None or args.fixation_high is None or args.fixation_point is not None:
        parser.error("region fixation needs --fixation-low and --fixation-high and no point")
    return RegionFixation(tuple(args.fixation_low), tuple(args.fixation_high))


def fixation_command(args):
    """Return the command-line tokens that repeat the parsed fixation choice."""
    command = ["--fixation", args.fixation]
    if args.fixation_point is not None:
        command.extend(["--fixation-point", *(str(component) for component in args.fixation_point)])
    if args.fixation_low is not None:
        command.extend(["--fixation-low", *(str(component) for component in args.fixation_low)])
    if args.fixation_high is not None:
        command.extend(["--fixation-high", *(str(component) for component in args.fixation_high)])
    return command


def use_fixation(parser, args, grid):
    """Apply a command-line point or region fixation. The scene bed stays when the choice is bed."""
    fixation = fixation_from_args(parser, args)
    if fixation is None:
        return None
    held = apply_fixation(grid, fixation)
    print(f"FIXATION: {args.fixation} holds {held} nodes", flush=True)
    return fixation
