"""Instrument-force readout and the slot a later fitted model occupies.

The solver already reports contact reactions. Those reactions are what `readout` returns. They are the contact law's
own numbers, so the demo labels them as uncalibrated until a fitted model is installed on `ForceMonitor`.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

import cv2

from .grid import METERS_PER_UNIT
from .native_backend import Observation


def force_vector(values):
    """Return a frozen copy of one 3-component force, in newtons.

    Every stored force passes through here, so a caller cannot alias the solver buffer or hand the monitor a vector of
    another shape.
    """
    vector = np.array(values, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("a force vector has 3 finite components")
    vector.flags.writeable = False
    return vector


def finite_scalar(value):
    """Return one finite force component as a Python float, in newtons."""
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("a force component is not finite")
    return number


@dataclass(frozen=True)
class SolverForceReadout:
    """Contact forces copied from one solved step, in newtons, plus the scissors stroke that produced them.

    `jaw_upper_n` and `jaw_lower_n` are the inward loads on the two grasper pads. `reaction_n` is the contact force
    on the tissue. `blade_force_n` and `blade_pressure_n` come from the cutting tool. The scissors channels stay
    None for a knife. Fracture channels keep the loads at the latest cut, and `scissors_fractures` counts cuts.
    """

    jaw_upper_n: float
    jaw_lower_n: float
    reaction_n: np.ndarray
    blade_force_n: np.ndarray
    blade_pressure_n: float
    blade_phase: int
    scissors_left_n: float | None
    scissors_right_n: float | None
    scissors_pressure_n: float | None
    scissors_fracture_left_n: float | None
    scissors_fracture_right_n: float | None
    scissors_fracture_total_n: float | None
    scissors_fractures: int | None
    is_squeezing: bool | None


def solver_force_readout(obs: Observation) -> SolverForceReadout:
    """Copy the contact forces of one solved step.

    Jaw loads, the contact reaction and the scissors loads are already in newtons. The blade record is still the native
    buffer, so its force channels take the scale `observe` applies to those other channels.
    """
    blade = obs.blade
    if blade.shape[0] < 8:
        raise ValueError("the blade record has no force channels")
    scissors = obs.scissors
    loads = None if scissors is None else scissors.loads
    return SolverForceReadout(
        finite_scalar(obs.jaw_load_n[0]), finite_scalar(obs.jaw_load_n[1]), force_vector(obs.reaction_n),
        force_vector(blade[3:6] * METERS_PER_UNIT), finite_scalar(blade[6] * METERS_PER_UNIT), int(blade[7]),
        None if loads is None else finite_scalar(loads.left_n),
        None if loads is None else finite_scalar(loads.right_n),
        None if scissors is None else finite_scalar(scissors.pressure_n),
        None if loads is None else finite_scalar(loads.fracture_left_n),
        None if loads is None else finite_scalar(loads.fracture_right_n),
        None if loads is None else finite_scalar(loads.fracture_total_n),
        None if loads is None else int(loads.fractures),
        None if loads is None else bool(loads.is_squeezing))


@dataclass(frozen=True)
class ForceEstimate:
    """Forces reported to callers, in newtons.

    `is_calibrated` is true only after a fitted model produced the numbers. The solver model leaves it false.
    Scissors channels stay None when the step has no scissors.
    """

    jaw_upper_n: float
    jaw_lower_n: float
    reaction_n: np.ndarray
    blade_force_n: np.ndarray
    blade_pressure_n: float
    scissors_left_n: float | None
    scissors_right_n: float | None
    scissors_pressure_n: float | None
    is_calibrated: bool

    def lines(self):
        """Return the demo overlay, one string per row."""
        status = "model estimate" if self.is_calibrated else "solver readout, not a measured force"
        rows = [
            f"Forces: {status}",
            f"Jaw upper/lower  {self.jaw_upper_n: .3f} / {self.jaw_lower_n: .3f} N",
            "Reaction  " + " ".join(f"{component: .3f}" for component in self.reaction_n) + " N",
            "Blade  " + " ".join(f"{component: .3f}" for component in self.blade_force_n) + " N",
            f"Blade pressure  {self.blade_pressure_n: .3f} N"]
        if self.scissors_left_n is not None and self.scissors_right_n is not None:
            rows.append(f"Scissors L/R  {self.scissors_left_n: .3f} / {self.scissors_right_n: .3f} N")
        if self.scissors_pressure_n is not None:
            rows.append(f"Scissors pressure  {self.scissors_pressure_n: .3f} N")
        return tuple(rows)


@dataclass(frozen=True)
class MeasuredForce:
    """One sensor sample in newtons, taken at the same step as a solver readout.

    A later fit uses whichever channels the sensor provides. Channels the sensor does not measure stay None.
    """

    jaw_upper_n: float | None = None
    jaw_lower_n: float | None = None
    reaction_n: np.ndarray | None = None
    blade_force_n: np.ndarray | None = None
    blade_pressure_n: float | None = None
    scissors_left_n: float | None = None
    scissors_right_n: float | None = None
    scissors_pressure_n: float | None = None


@dataclass(frozen=True)
class ForcePair:
    """One row of a later fit: the solver readout beside the sensor sample from the same step."""

    readout: SolverForceReadout
    measured: MeasuredForce


class ForceModel(Protocol):
    """Estimates instrument forces in newtons from one solver readout.

    A fitted model keeps its coefficients on the object and returns a `ForceEstimate` with `is_calibrated` set. Channels
    the model does not correct are copied from the readout.
    """

    def estimate(self, readout: SolverForceReadout) -> ForceEstimate:
        """Return the force estimate for one solver readout."""


class SolverForceModel:
    """Reports the solver readout unchanged and marks it uncalibrated."""

    def estimate(self, readout: SolverForceReadout) -> ForceEstimate:
        """Copy the solver readout into an uncalibrated estimate."""
        return ForceEstimate(
            readout.jaw_upper_n, readout.jaw_lower_n, readout.reaction_n, readout.blade_force_n,
            readout.blade_pressure_n, readout.scissors_left_n, readout.scissors_right_n, readout.scissors_pressure_n,
            is_calibrated=False)


class ForceMonitor:
    """Report the instrument forces of one solved state.

    `estimate` is the force the demo draws and the force later callers should use. With no model, it copies the solver
    readout and marks the copy uncalibrated: those reactions are the contact law, not a sensor measurement. `readout`
    is that solver record. `pair` puts it beside one sensor sample so a later fit can learn a correction. Pass the
    fitted object, which must provide `estimate(readout)`, as `model` or assign it to `model` afterwards.
    """

    def __init__(self, model: ForceModel | None = None):
        self.model: ForceModel = SolverForceModel() if model is None else model

    def readout(self, obs: Observation) -> SolverForceReadout:
        """Return the solver contact forces of one solved observation."""
        return solver_force_readout(obs)

    def estimate(self, obs: Observation) -> ForceEstimate:
        """Return the installed model's force for one solved observation."""
        estimate = self.model.estimate(self.readout(obs))
        left = None if estimate.scissors_left_n is None else finite_scalar(estimate.scissors_left_n)
        right = None if estimate.scissors_right_n is None else finite_scalar(estimate.scissors_right_n)
        pressure = None if estimate.scissors_pressure_n is None else finite_scalar(estimate.scissors_pressure_n)
        return ForceEstimate(
            finite_scalar(estimate.jaw_upper_n), finite_scalar(estimate.jaw_lower_n),
            force_vector(estimate.reaction_n), force_vector(estimate.blade_force_n),
            finite_scalar(estimate.blade_pressure_n), left, right, pressure, bool(estimate.is_calibrated))

    def pair(self, obs: Observation, measured: MeasuredForce) -> ForcePair:
        """Pack one solver readout with the sensor sample from the same step."""
        return ForcePair(self.readout(obs), measured)


def draw_force_readout(frame, estimate: ForceEstimate):
    """Draw a force estimate on the top-left of a BGR uint8 frame, in place.

    Only the text block is darkened. The first line says whether the numbers are the uncalibrated solver readout or a
    fitted model.
    """
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("the force readout is drawn on a BGR uint8 frame")
    lines = estimate.lines()
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    sizes = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
    text_height = max(size[0][1] for size in sizes)
    gap = 6
    origin_x, origin_y = 12, 12
    width = max(size[0][0] for size in sizes) + 16
    height = len(lines) * (text_height + gap) + 10
    x1 = min(frame.shape[1], origin_x + width)
    y1 = min(frame.shape[0], origin_y + height)
    roi = frame[origin_y:y1, origin_x:x1]
    frame[origin_y:y1, origin_x:x1] = (roi.astype(np.uint16) * 2 // 5).astype(np.uint8)
    status_color = (180, 255, 200) if estimate.is_calibrated else (140, 200, 255)
    for i_line, line in enumerate(lines):
        baseline = origin_y + 8 + text_height + i_line * (text_height + gap)
        color = status_color if i_line == 0 else (235, 235, 235)
        cv2.putText(frame, line, (origin_x + 8, baseline), font, scale, color, thickness, cv2.LINE_AA)
