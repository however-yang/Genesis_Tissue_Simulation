"""Small, explicitly unvalidated headless data pilot; does not change run.py."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ..scenario import SCISSORS_SCENES
from .actions import ActionBounds, command_to_action
from .data import EpisodeReader, EpisodeWriter, atomic_json, save_arrays
from .env import EnvironmentConfig, NativeEnvironment, SceneConfig
from .provenance import json_hash


def local_bounds(scene_config):
    """Small screening box around the scripted prepared target, not a certified domain."""
    end = {"settled": "settle", "held": "lift", "cut_ready": "blade_approach"}[scene_config.preparation]
    command = next(s.command for s in SCISSORS_SCENES[scene_config.scene].stages if s.name == end)
    center = command_to_action(command)
    span = np.array([.0005]*3 + [0] + [.0005]*3 + [np.radians(2)]*3 + [0])
    lo, hi = center-span, center+span
    lo[10], hi[10] = 0, 1
    # Auxiliary grasp is kept at its declared target in this initial local task.
    rates = np.array([.002]*3 + [0] + [.002]*3 + [np.radians(5)]*3 + [1.0])
    return ActionBounds(tuple(lo), tuple(hi), tuple(rates))


def pilot_actions(initial, steps, dt, mode, rng):
    """Bounded smooth exploratory motions, with the entire draw saved by seed/version."""
    phases = rng.uniform(0, 2*np.pi, 6)
    amplitudes = rng.uniform(.00005, .00015, 6)
    for i in range(steps):
        action = initial.copy()
        t = (i+1)*dt
        envelope = min(t/.5, 1.0)
        delta = amplitudes * (np.sin(t*1.5+phases)-np.sin(phases)) * envelope
        if mode in ("grasper", "joint"):
            action[:3] += delta[:3]
        if mode in ("scissors", "joint"):
            action[4:7] += delta[3:]
            action[10] = min(1.0, t/2.5)
        yield action


@threadpool_limits.wrap(limits=1, user_api="blas")
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", required=True)
    parser.add_argument("--dll-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", choices=("wall-traction", "flap", "kidney"), default="wall-traction")
    parser.add_argument("--preparation", choices=("settled", "held", "cut_ready"), default="cut_ready")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-attempts", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--modes", nargs="+", choices=("hold", "grasper", "scissors", "joint"),
                        default=["hold", "grasper", "scissors", "joint"])
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        parser.error("episodes and steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    scene = SceneConfig(args.scene, args.preparation)
    bounds = local_bounds(scene)
    records = []
    started = time.perf_counter()
    for episode in range(args.episodes):
        mode, seed = args.modes[episode % len(args.modes)], args.seed+episode
        config = EnvironmentConfig(args.dll, args.dll_sha256, bounds, scene,
                                   cutting_enabled=mode in ("scissors", "joint"), max_steps=args.steps)
        print(f"episode={episode} seed={seed} mode={mode} preparing", flush=True)
        record = {"index": episode, "seed": seed, "mode": mode, "status": "initializing",
                  "group": f"{args.scene}:{args.preparation}:{seed}", "approved_for_training": False,
                  "config": asdict(config)}
        # The split is by whole reset lineage, never by individual frames.
        group_hash = json_hash(record["group"])
        record["split"] = "validation" if int(group_hash[:8], 16) % 5 == 0 else "train"
        with NativeEnvironment(config) as env:
            ready = False
            for attempt in range(args.reset_attempts):
                try:
                    state, context = env.reset(seed)
                    ready = True
                    break
                except (RuntimeError, ValueError, OSError) as error:
                    failure = env.last_reset_diagnostic or {"error": repr(error)}
                    if env.last_reset_failure_arrays is not None:
                        name = f"reset-failure-{episode:04}-{attempt:02}.npz.zst"
                        failure["arrays_file"] = name
                        failure["arrays_sha256"] = save_arrays(args.output/name, env.last_reset_failure_arrays)
                    atomic_json(args.output/f"reset-failure-{episode:04}-{attempt:02}.json", failure)
                    record.update(status="reset_failure", error=repr(error), attempts=attempt+1)
            if ready:
                directory = args.output/f"episode-{episode:04}"
                wall_start = time.perf_counter()
                try:
                    with EpisodeWriter(directory, context, state, env.static_arrays(), env.preparation) as writer:
                        for action in pilot_actions(env.previous_action, args.steps, env.settings.dt, mode, env.rng):
                            result = env.step(action)
                            writer.append(result)
                            if result.terminated or result.truncated:
                                break
                    reader = EpisodeReader(directory)
                    # Counts only checked exploratory windows, not training admission.
                    windows = sum(1 for _ in reader.windows(min(4, args.steps), allow_unvalidated_scope=True))
                    record.update(status="recorded", episode_id=context["episode_id"],
                                  context_sha256=context["context_sha256"],
                                  states=reader.manifest["states"], transitions=reader.manifest["normal_transitions"],
                                  failed_attempts=reader.manifest["failed_attempts"],
                                  reason=reader.manifest["reason"], exploratory_windows=windows,
                                  reset_wall_time_ms=context["reset_wall_time_ms"],
                                  collect_and_verify_wall_time_s=time.perf_counter()-wall_start)
                except (RuntimeError, OSError, ValueError, TimeoutError) as error:
                    record.update(status="collection_failure", error=repr(error))
        records.append(record)
        atomic_json(args.output/"pilot.json", {"schema": "pilot_v1", "episodes": records,
                    "wall_time_s": time.perf_counter()-started, "training_domain_approved": False,
                    "scope": "interface and sampler screening only; auxiliary grasp, local motions",
                    "sampling_version": "smooth_local_v1; no automatic physical acceptance"})
        print(f"episode={episode} status={record['status']} reason={record.get('reason', record.get('error'))}", flush=True)


if __name__ == "__main__":
    main()
