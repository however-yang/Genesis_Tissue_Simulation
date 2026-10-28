"""Run the native grasp/lift/cut sequence, optionally viewed through Genesis."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .grid import build_grid, to_genesis, METERS_PER_UNIT
from .native_backend import NativeTissue, Settings, Command
from .rendering import EndoscopicView
from .scenario import (SCISSORS_SCENES, SCISSORS_SPACING, STAGES, Sequence, drive_command, interpolate, scissors_settings,
                       severed_edges, stage_start)
from .surface_diagnostics import topology_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Save camera frames without opening a window")
    parser.add_argument("--native-only", action="store_true", help="Validate native mechanics without constructing a Genesis scene")
    parser.add_argument("--viewer", action="store_true", help="Also show the free-orbit Genesis viewer")
    parser.add_argument("--steps", type=int, default=None, help="Stop after this many steps (partial runs are not accepted as complete)")
    parser.add_argument("--spacing", type=float, default=None,
                        help="Lateral material grid spacing in cm; the scissors flow is tuned for 0.1, the knife default is 0.2")
    parser.add_argument("--vertical-spacing", type=float, default=None,
                        help="Cut-depth resolution in cm; default per scissors scene, 0.05 for the knife")
    parser.add_argument("--render-every", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tool", choices=("scissors", "knife"), default="scissors")
    parser.add_argument("--scene", choices=tuple(SCISSORS_SCENES), default="wall",
                        help="Scissors only: cut a long wound along a wall standing on the block, or split a short flap")
    parser.add_argument("--start-force", type=float, default=None,
                        help="Cut initiation force in N (scissors: each blade needs half of it); default from Settings")
    parser.add_argument("--dll", type=Path, default=None, help="Native library; default depends on --tool")
    parser.add_argument("--demo", action="store_true",
                        help="Scissors only: move the camera in on the scissors and end the video on a 3 s still of the cut")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("steps must be nonnegative")
    if args.demo and (args.tool != "scissors" or args.native_only):
        parser.error("--demo records the scissors flow and needs the Genesis view")
    if args.spacing is None:
        args.spacing = SCISSORS_SPACING if args.tool == "scissors" else 0.2
    elif args.tool == "scissors" and not np.isclose(args.spacing, SCISSORS_SPACING):
        parser.error(f"the scissors poses put the shear plane mid-cell for spacing {SCISSORS_SPACING} cm only")
    if args.render_every < 1 or min(args.width, args.height) < 128:
        parser.error("render-every must be positive and image dimensions at least 128")
    output = args.output or Path(__file__).parent / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    settings = scissors_settings() if args.tool == "scissors" else Settings()
    if args.start_force is not None:
        if not args.start_force > settings.cut_force_n:
            parser.error(f"start force must exceed the cutting force {settings.cut_force_n} N")
        settings = replace(settings, start_force_n=args.start_force)
    scene = SCISSORS_SCENES[args.scene] if args.tool == "scissors" else None
    if args.vertical_spacing is None:
        args.vertical_spacing = scene.vertical_spacing if scene else .05
    elif scene and not np.isclose(args.vertical_spacing, scene.vertical_spacing):
        parser.error(f"the {args.scene} scissors poses put the shear plane mid-cell for vertical spacing "
                     f"{scene.vertical_spacing} cm only")
    stages = scene.stages if scene else STAGES
    (output / "stages.json").write_text(json.dumps([asdict(stage) for stage in stages], indent=2), encoding="utf8")
    grid = build_grid(args.spacing, density=settings.density_kg_m3, vertical_spacing=args.vertical_spacing,
                      tab=scene.tab if scene else None)
    view, video, sequence, dll = None, None, None, None
    status, error, stage_name = "partial", None, "initial"
    started = time.monotonic()
    step = 0
    surface_audits = []
    window = "Endoscopic tissue | SPACE pause | Q quit"
    try:
        with NativeTissue(grid, settings, args.dll, stages[0].command) as tissue, (output / "trajectory.jsonl").open("w", encoding="utf8") as trajectory:
            dll = tissue.dll_path
            obs = tissue.observe()
            sequence = Sequence(obs, settings, scene, grid.positions, grid.inside)
            if not args.native_only:
                view = EndoscopicView(settings, args.width, args.height, args.viewer)
                video = cv2.VideoWriter(str(output / "endoscopic_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                        1 / (settings.dt * args.render_every), (args.width, args.height))
                if not video.isOpened():
                    raise RuntimeError("Could not open MP4 writer")
                if args.demo:
                    view.camera.set_pose(pos=scene.shot[0], lookat=scene.shot[1])
                view.update(tissue.surface(), obs)
                image, depth = view.capture("initial", 0, 0, obs.grip_points)
                image.save(output / "initial.png")
                np.save(output / "initial_depth_m.npy", depth)
            previous = stages[0].command
            interrupted = False
            for stage in stages:
                if args.steps is not None and step >= args.steps:
                    break
                stage_name = stage.name
                previous = stage_start(stage.name, previous, obs)
                print(f"[{step:03d}] {stage.name}", flush=True)
                for local_step in range(stage.steps):
                    if args.steps is not None and step >= args.steps:
                        interrupted = True
                        break
                    requested = interpolate(previous, stage.command, (local_step + 1) / (stage.steps * stage.ramp))
                    command = drive_command(requested, obs, settings)
                    obs = tissue.step(command)
                    if obs.scissors is not None:
                        blade_body_depth, wrist_depth = tissue.scissors_body_depths_m()
                        knife_body_depth = max(blade_body_depth, wrist_depth)
                    else:
                        blade_body_depth, knife_body_depth = 0.0, tissue.knife_body_depth_m()
                    step += 1
                    cut_edges = int(severed_edges(sequence.initial_neighbors, obs.neighbors))
                    trajectory.write(json.dumps({"step": step, "time_s": step * settings.dt, "stage": stage.name,
                        "command": asdict(command), "requested_command": asdict(requested),
                        "hinge_cm": obs.hinge.tolist(), "jaw_gap_cm": obs.jaw_gap,
                        "grip_points": obs.grip_points, "grip_error_m": obs.grip_error_m,
                        "blade_center_cm": obs.blade[:3].tolist(), "blade_force_n": (obs.blade[3:6] * METERS_PER_UNIT).tolist(),
                        "knife_body_penetration_mm": knife_body_depth * 1000,
                        "blade_body_overlap_mm": blade_body_depth * 1000,
                        "blade_phase": int(obs.blade[7]), "cut_edges": cut_edges,
                        "scissors_gap_cm": obs.scissors.gap if obs.scissors else None,
                        "scissors_edges_cm": obs.scissors.edges.tolist() if obs.scissors else None,
                        "scissors_pressure_n": obs.scissors.pressure_n if obs.scissors else None,
                        "scissors_phase": obs.scissors.phase if obs.scissors else None,
                        "scissors_contacts": obs.scissors.contacts if obs.scissors else None,
                        "scissors_loads": asdict(obs.scissors.loads) if obs.scissors else None}) + "\n")
                    sequence.record_step(stage.name, obs, knife_body_depth, blade_body_depth)
                    if step % args.render_every == 0 and view is not None:
                        surface = tissue.surface()
                        view.update(surface, obs)
                        image, depth = view.capture(stage.name, step * settings.dt, cut_edges, obs.grip_points)
                        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        video.write(frame)
                        if not args.headless:
                            cv2.imshow(window, frame)
                            key = cv2.waitKey(1) & 0xff
                            if key == 32:
                                key = cv2.waitKey(0) & 0xff
                            if key in (ord("q"), 27):
                                interrupted = True
                                break
                if interrupted:
                    break
                surface = tissue.surface()
                np.savez_compressed(output / f"{stage.name}.npz", nodes_cm=obs.nodes, neighbors=obs.neighbors,
                                    vertices_cm=surface.vertices, faces=surface.triangles, grip_flags=obs.grip_flags,
                                    normals=surface.normals, anchors=surface.anchors, hinge_cm=obs.hinge,
                                    jaw_gap_cm=obs.jaw_gap, blade=obs.blade, grip_points=obs.grip_points,
                                    reaction_n=obs.reaction_n, grip_error_m=obs.grip_error_m,
                                    scissors_gap_cm=obs.scissors.gap if obs.scissors else -1,
                                    scissors_edges_cm=obs.scissors.edges if obs.scissors else np.empty((0, 2, 3)))
                if view is not None:
                    view.update(surface, obs)
                    image, depth = view.capture(stage.name, step * settings.dt,
                                               int(severed_edges(sequence.initial_neighbors, obs.neighbors)), obs.grip_points)
                    image.save(output / f"{stage.name}.png")
                sequence.finish_stage(stage.name, obs, surface)
                audit = topology_statistics(surface)
                audit["stage"] = stage.name
                surface_audits.append(audit)
                if (audit["degenerate_triangles"] or audit["positional_boundary_edges"]
                        or audit["positional_nonmanifold_edges"] or audit["duplicate_triangles"]
                        or not audit["winding_consistent"]):
                    sequence.metrics.acceptance_passed = False
                    raise RuntimeError(f"Invalid surface topology at {stage.name}; inspect surface-audit.json")
                print(f"    grip={obs.grip_points}, cut_edges={severed_edges(sequence.initial_neighbors, obs.neighbors)}, "
                      f"lift={sequence.metrics.lift_m * 1000:.2f} mm", flush=True)
                if obs.scissors is not None:
                    m = sequence.metrics
                    print(f"    scissors gap={obs.scissors.gap * 10:.2f} mm, fractures={m.fractures}, "
                          f"fracture loads L/R={m.fracture_left_n:.3f}/{m.fracture_right_n:.3f} N, "
                          f"two-blade squeeze={m.two_blade_squeeze_m * 1000:.2f} mm, non-cutting overlap="
                          f"{m.max_knife_body_penetration_m * 1000:.2f} mm (blade backs {m.max_blade_body_overlap_m * 1000:.2f} mm)",
                          flush=True)
                previous = stage.command
            status = "passed" if sequence.metrics.acceptance_passed else "partial"
            if args.demo and status == "passed":
                # The final solved state, seen from a camera that moves in over
                # the wound for 2 s and then holds for 3 s; the simulation does
                # not advance.
                wall = surface.anchors[surface.triangles[:, 0], 3] > .5
                wound = surface.vertices[surface.triangles[wall]].reshape(-1, 3)
                target = to_genesis(((wound.min(axis=0) + wound.max(axis=0)) / 2)[None])[0]
                fps = 1 / (settings.dt * args.render_every)
                moves = round(2 * fps)
                for frame_index in range(moves + round(3 * fps)):
                    u = min(1, (frame_index + 1) / moves)
                    u = u * u * (3 - 2 * u)
                    pos = (1 - u) * np.array(scene.shot[0]) + u * (target + scene.closeup_offset)
                    lookat = (1 - u) * np.array(scene.shot[1]) + u * target
                    view.camera.set_pose(pos=pos, lookat=lookat)
                    image, _ = view.capture("closeup", step * settings.dt, 0, obs.grip_points)
                    video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
                image.save(output / "wound-closeup.png")
            if view is not None and not args.headless:
                view.update(tissue.surface(), obs)
                image, _ = view.capture(status + " | press any key to close", step * settings.dt,
                                       int(severed_edges(sequence.initial_neighbors, obs.neighbors)), obs.grip_points)
                cv2.imshow(window, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
    except (RuntimeError, ValueError, OSError) as ex:
        status, error = "failed", str(ex)
        raise
    finally:
        if video is not None:
            video.release()
        if view is not None:
            view.close()
        if not args.headless and not args.native_only:
            cv2.destroyAllWindows()
        report = {"status": status, "error": error, "last_stage": stage_name, "steps": step,
                  "elapsed_s": time.monotonic() - started, "settings": asdict(settings),
                  "grid_spacing_cm": args.spacing, "cell_size_cm": grid.cell_size.tolist(), "nodes": len(grid.positions),
                  "scene": args.scene if scene else None, "tab_cm": asdict(scene.tab) if scene else None,
                  "dll": str(dll) if dll else None,
                  "dll_sha256": hashlib.sha256(dll.read_bytes()).hexdigest() if dll else None,
                  "genesis_view_created": view is not None,
                  "metrics": asdict(sequence.metrics) if sequence else None,
                  "limitations": ["Single native session", "End-effector control; no robot arm dynamics",
                                  "Anchor-assisted grasp", "Finite capsule cutting edge", "Uncalibrated tissue"]}
        (output / "surface-audit.json").write_text(json.dumps(surface_audits, indent=2), encoding="utf8")
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        print(f"{status.upper()}: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
