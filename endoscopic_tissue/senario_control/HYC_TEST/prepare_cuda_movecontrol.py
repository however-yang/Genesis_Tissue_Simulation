"""Prepare a verification input from the same scene config used by SimulationInterface."""
import argparse
from pathlib import Path

from ...scenario import SCISSORS_SPACING
from ..config import load_config, resolve_settings
from ..hyc_movecontrol_config import build_scene_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "configs/kidney.toml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    scene = build_scene_input(config, resolve_settings(config), grid_spacing_cm=SCISSORS_SPACING)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(scene))
    print(f"Scene input: {args.config} -> {args.output} ({len(bytes(scene))} bytes)")


if __name__ == "__main__":
    main()
