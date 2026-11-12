"""Read sample external inputs and save motion results for inspection."""
import argparse
import json
from pathlib import Path

from ..instrument_control import Grasper, Scissors, load_config
from ..instrument_driver import MotionBridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("sample_inputs.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "sample_motion.json")
    args = parser.parse_args()
    sequence = json.loads(args.input.read_text(encoding="utf-8"))
    # The test caller supplies configured instances; the driver never creates instruments.
    configured = load_config(Path(__file__).parents[1] / "instrument_control" / "configs" / "instruments.toml")
    bridge = MotionBridge(Grasper(configured.grasper), Scissors(configured.scissors))
    frames = []
    step_count = 0
    for stage in sequence["stages"]:
        bridge.submit(stage["input"])
        for _ in range(stage["steps"]):
            frame = bridge.step(dt_s=sequence["dt_s"])
            step_count += 1
            frame["step"] = step_count
            frame["time_s"] = step_count * sequence["dt_s"]
            frame["stage"] = stage["name"]
            frames.append(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frames, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(f"Computed {step_count} motion steps over {step_count * sequence['dt_s']:.3f} seconds: {args.output}")


if __name__ == "__main__":
    main()
