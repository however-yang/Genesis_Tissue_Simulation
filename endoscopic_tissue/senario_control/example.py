"""Inspect a modular scene sequence or execute it with the native solver."""
import argparse
from itertools import islice
from pathlib import Path

from ..scenario import SCISSORS_SCENES
from .simulator import SimulationInterface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(SCISSORS_SCENES), default="kidney")
    parser.add_argument("--simulate", action="store_true", help="Open the native CUDA solver and execute the sequence")
    parser.add_argument("--steps", type=int, default=None, help="Maximum physics steps, default the full sequence")
    parser.add_argument("--dll", type=Path, default=None, help="Compatible native solver library")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")
    stages = SCISSORS_SCENES[args.scene].stages
    print(f"Scene: {args.scene}; {len(stages)} stages; {sum(stage.steps for stage in stages)} steps")
    if not args.simulate:
        for stage in stages:
            command = stage.command
            grasper_position = ", ".join(f"{coordinate:.4f}" for coordinate in command.hinge)
            scissors_position = ", ".join(f"{coordinate:.4f}" for coordinate in command.blade_center)
            print(f"{stage.name:20s} steps={stage.steps:3d} grasper=({grasper_position}) "
                  f"closure={command.closure:g} scissors=({scissors_position}) "
                  f"closure={command.scissors_closure:g} cutting={command.scissors_cutting}")
        return
    with SimulationInterface(args.scene, dll_path=args.dll) as simulation:
        previous_stage = None
        for state in islice(simulation.run_stages(), args.steps):
            if state.stage != previous_stage:
                print(f"[{state.step:04d}] {state.stage}; t={state.time_s:.3f}s")
                previous_stage = state.stage
        print(f"Completed {simulation.step_count} physics steps")


if __name__ == "__main__":
    main()
