"""Run the native grasp/lift/cut sequence, optionally viewed through Genesis.

The window plays the flow at wall-clock speed: a worker thread steps the native session, holding every step to its
simulated time, while the main thread draws the latest solved state. --headless records one frame every
--render-every steps instead, in step with the simulation.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np

import cv2

from .grid import METERS_PER_UNIT, build_grid, to_genesis
from .instrument_geometry import scissors_body_depths
from .native_backend import Command, NativeTissue, Observation, Settings, Surface
from .rendering import EndoscopicView
from .scenario import (SCISSORS_SCENES, SCISSORS_SPACING, STAGES, Sequence, drive_command, interpolate, scissors_settings,
                       severed_edges, stage_start)
from .surface_diagnostics import topology_statistics


WINDOW = "Endoscopic tissue | SPACE pause | Q quit"
# Frame rate of the live window's recording: each tick writes the image shown at that moment.
LIVE_FPS = 30


@dataclass(frozen=True)
class Snapshot:
    """A solved state to draw. Its arrays come fresh from the native session and are never written afterwards."""
    stage: str
    step: int
    surface: Surface
    obs: Observation


class StepRecorder:
    """Checks, logs and records every solved step on a worker thread, in step order, while the session steps on.

    The overlap of the scissors bodies with the tissue, the cut count, the trajectory line and the sequence metrics are
    numpy work on fresh copies of the solved state, so they run alongside the next native step instead of delaying it.
    `submit` and `drain` raise the first error a recorded step raised; `drain` also waits for every queued step. Used as
    a context manager, it records the steps still queued on exit and then raises that error too.
    """

    def __init__(self, flow, tissue: NativeTissue, trajectory):
        self.flow, self.trajectory = flow, trajectory
        self.geometry = tissue.scissors_geometry
        self.steps = queue.Queue()
        self.error = None
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, trace):
        self.steps.put(None)
        self.thread.join()
        if kind is None:
            self.raise_error()

    def work(self):
        while (job := self.steps.get()) is not None:
            if self.error is None:
                try:
                    self.record(*job)
                # Forwarded to the stepping thread, which stops the flow on it.
                except Exception as error:
                    self.error = error
            self.steps.task_done()

    def raise_error(self):
        if self.error is not None:
            raise self.error

    def submit(self, stage, step, command, requested, obs: Observation, surface, knife_body_depth):
        self.raise_error()
        self.steps.put((stage, step, command, requested, obs, surface, knife_body_depth))

    def drain(self):
        self.steps.join()
        self.raise_error()

    def record(self, stage, step, command, requested, obs: Observation, surface, knife_body_depth):
        settings, sequence = self.flow.settings, self.flow.sequence
        blade_body_depth = 0.0
        if obs.scissors is not None:
            pose = obs.scissors
            depths = scissors_body_depths(surface.vertices, surface.triangles, self.geometry, pose.hinge, pose.axis,
                                          pose.normal, pose.gap)
            blade_body_depth, wrist_depth = (float(depth) * METERS_PER_UNIT for depth in depths)
            knife_body_depth = max(blade_body_depth, wrist_depth)
        cut_edges = int(severed_edges(sequence.initial_neighbors, obs.neighbors))
        self.trajectory.write(json.dumps({"step": step, "time_s": step * settings.dt, "stage": stage,
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
        sequence.record_step(stage, obs, knife_body_depth, blade_body_depth)


class Flow:
    """Steps one native session through the scripted stages, checking, logging and saving every step and stage.

    After every step `sink.after_step(flow, tissue, surface)` runs, with the surface read for the step's checks or
    None, and stops the flow when it returns False. After every completed stage `sink.after_stage(flow, stage, surface)`
    runs, before the stage checks. The per-step checks run on a StepRecorder; every stage waits for them before its own.
    """

    def __init__(self, settings: Settings, scene, stages, grid, step_limit, output: Path):
        self.settings, self.scene, self.stages, self.grid = settings, scene, stages, grid
        self.step_limit, self.output = step_limit, output
        self.step, self.stage = 0, "initial"
        self.dll = self.sequence = self.obs = self.surface = None
        self.surface_audits = []

    def open(self, tissue: NativeTissue):
        self.dll = tissue.dll_path
        self.obs = tissue.observe()
        self.sequence = Sequence(self.obs, self.settings, self.scene, self.grid.positions, self.grid.inside)

    def run(self, tissue: NativeTissue, trajectory, sink):
        with StepRecorder(self, tissue, trajectory) as recorder:
            settings, sequence = self.settings, self.sequence
            previous = self.stages[0].command
            for stage in self.stages:
                if self.step_limit is not None and self.step >= self.step_limit:
                    return
                self.stage = stage.name
                previous = stage_start(stage.name, previous, self.obs)
                print(f"[{self.step:03d}] {stage.name}", flush=True)
                for local_step in range(stage.steps):
                    if self.step_limit is not None and self.step >= self.step_limit:
                        return
                    requested = interpolate(previous, stage.command, (local_step + 1) / (stage.steps * stage.ramp))
                    command = drive_command(requested, self.obs, settings)
                    obs = self.obs = tissue.step(command)
                    # The native session is single-threaded: read here what the recorder needs from it.
                    surface = tissue.surface() if obs.scissors is not None else None
                    knife_body_depth = tissue.knife_body_depth_m() if obs.scissors is None else 0.0
                    self.step += 1
                    recorder.submit(stage.name, self.step, command, requested, obs, surface, knife_body_depth)
                    if sink is not None and not sink.after_step(self, tissue, surface):
                        return
                recorder.drain()
                obs = self.obs
                surface = self.surface = tissue.surface()
                np.savez_compressed(self.output / f"{stage.name}.npz", nodes_cm=obs.nodes, neighbors=obs.neighbors,
                                    vertices_cm=surface.vertices, faces=surface.triangles, grip_flags=obs.grip_flags,
                                    normals=surface.normals, anchors=surface.anchors, hinge_cm=obs.hinge,
                                    jaw_gap_cm=obs.jaw_gap, blade=obs.blade, grip_points=obs.grip_points,
                                    reaction_n=obs.reaction_n, grip_error_m=obs.grip_error_m,
                                    scissors_gap_cm=obs.scissors.gap if obs.scissors else -1,
                                    scissors_edges_cm=obs.scissors.edges if obs.scissors else np.empty((0, 2, 3)))
                if sink is not None:
                    sink.after_stage(self, stage.name, surface)
                sequence.finish_stage(stage.name, obs, surface)
                audit = topology_statistics(surface)
                audit["stage"] = stage.name
                self.surface_audits.append(audit)
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


class Recording:
    """Renders one frame every `every` steps into the video and every completed stage as a still, in step."""

    def __init__(self, view: EndoscopicView, video, every, output: Path):
        self.view, self.video, self.every, self.output = view, video, every, output

    def after_step(self, flow: Flow, tissue: NativeTissue, surface):
        if flow.step % self.every == 0:
            self.view.update(tissue.surface() if surface is None else surface, flow.obs)
            self.video.write(self.view.render()[0])
        return True

    def after_stage(self, flow: Flow, stage, surface):
        self.view.update(surface, flow.obs)
        cv2.imwrite(str(self.output / f"{stage}.png"), self.view.render()[0])


class LiveLink:
    """Couples the stepping thread to the window, which runs on the main thread.

    The stepping thread holds every step to its simulated duration: it waits when the simulation runs ahead of the
    wall clock and runs flat out when behind, without catching up later, so playback never exceeds real time. It
    posts a state to draw only once the window took the previous one, and every completed stage for its still. The
    window pauses it by clearing `running` and stops it by setting `stopping`.
    """

    def __init__(self, dt, every):
        self.dt, self.every = dt, every
        self.latest = queue.Queue(maxsize=1)
        self.stills = queue.Queue()
        self.posted = threading.Event()
        self.running = threading.Event()
        self.stopping = threading.Event()
        # Frames the window showed, per stage of the state drawn.
        self.shown = Counter()
        # Wall-clock time the latest step was released at, set when stepping starts.
        self.release = None
        self.stage, self.stage_steps, self.stage_compute_s, self.stage_wall_s = None, 0, 0.0, 0.0
        # Totals over the completed stages.
        self.total_steps, self.total_frames, self.total_wall_s = 0, 0, 0.0

    def post_still(self, snapshot: Snapshot):
        self.stills.put(snapshot)
        self.posted.set()

    def after_step(self, flow: Flow, tissue: NativeTissue, surface):
        computed = time.perf_counter()
        if flow.stage != self.stage:
            self.stage, self.stage_steps, self.stage_compute_s, self.stage_wall_s = flow.stage, 0, 0.0, 0.0
        started = self.release
        self.release = max(self.release + self.dt, computed)
        time.sleep(max(0.0, self.release - computed))
        self.stage_steps += 1
        self.stage_compute_s += computed - started
        self.stage_wall_s += self.release - started
        if flow.step % self.every == 0 and self.latest.empty():
            self.latest.put(Snapshot(flow.stage, flow.step, tissue.surface() if surface is None else surface, flow.obs))
            self.posted.set()
        if not self.running.is_set():
            paused = time.perf_counter()
            self.running.wait()
            # The pause holds the clock of the playback too.
            self.release += time.perf_counter() - paused
        return not self.stopping.is_set()

    def after_stage(self, flow: Flow, stage, surface):
        self.post_still(Snapshot(stage, flow.step, surface, flow.obs))
        self.total_steps += self.stage_steps
        self.total_frames += self.shown[stage]
        self.total_wall_s += self.stage_wall_s
        print(f"    realtime x{self.stage_steps * self.dt / self.stage_wall_s:.2f}: "
              f"{1000 * self.stage_compute_s / self.stage_steps:.1f} ms per step, "
              f"display {self.shown[stage] / self.stage_wall_s:.1f} fps", flush=True)


def step_live(flow: Flow, link: LiveLink, grid, settings: Settings, dll, initial_command: Command):
    """Run the flow on a native session owned by this thread, once the window has drawn the initial state."""
    with NativeTissue(grid, settings, dll, initial_command) as tissue, \
            (flow.output / "trajectory.jsonl").open("w", encoding="utf8") as trajectory:
        flow.open(tissue)
        link.post_still(Snapshot("initial", 0, tissue.surface(), flow.obs))
        link.running.wait()
        if not link.stopping.is_set():
            link.release = time.perf_counter()
            flow.run(tissue, trajectory, link)
        return Snapshot(flow.stage, flow.step, tissue.surface(), flow.obs)


def take_still(link: LiveLink, worker):
    """Return the next stage still, or None once the worker has ended without posting one."""
    while not link.stills.empty() or not worker.done():
        if link.posted.wait(timeout=.1):
            link.posted.clear()
        if not link.stills.empty():
            return link.stills.get()
    return None


def play_live(view: EndoscopicView, video, link: LiveLink, worker, shown, output: Path):
    """Show the latest posted state, starting from the image `shown`, until the worker ends; save each stage still.

    The video receives the image on screen at every tick of LIVE_FPS, paused time excepted, so it plays back what the
    window showed. SPACE pauses and resumes (any key resumes), Q or Esc stops the flow.
    """
    period = 1 / LIVE_FPS
    tick = time.perf_counter()
    drawn_step = 0
    paused = None
    while not worker.done() or not link.stills.empty() or not link.latest.empty():
        still = None if link.stills.empty() else link.stills.get()
        snapshot = still if still is not None else None if link.latest.empty() else link.latest.get()
        # A state posted before a still that was drawn first is stale.
        if snapshot is not None and (still is not None or snapshot.step > drawn_step):
            drawn_step = snapshot.step
            view.update(snapshot.surface, snapshot.obs)
            shown, _ = view.render()
            if still is not None:
                cv2.imwrite(str(output / f"{still.stage}.png"), shown)
            link.shown[snapshot.stage] += 1
            cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(1) & 0xff
        now = time.perf_counter()
        if paused is not None and key != 0xff:
            link.running.set()
            tick += now - paused
            paused = None
        elif key == 32:
            link.running.clear()
            paused = now
        if key in (ord("q"), 27):
            link.stopping.set()
            link.running.set()
        while paused is None and tick <= now:
            video.write(shown)
            tick += period
        if snapshot is None:
            if link.posted.wait(timeout=.005):
                link.posted.clear()


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
    parser.add_argument("--render-every", type=int, default=None,
                        help="Steps per drawn state: default 3 with --headless, where every drawn state is a video "
                             "frame, and 1 in the window, which draws the latest state whenever it can")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tool", choices=("scissors", "knife"), default="scissors")
    parser.add_argument("--scene", choices=tuple(SCISSORS_SCENES), default="wall",
                        help="Scissors only: cut a long wound along a wall standing on the block, split a short flap, "
                             "or excise a tumor from a kidney")
    parser.add_argument("--start-force", type=float, default=None,
                        help="Cut initiation force in N (scissors: each blade needs half of it); default from Settings")
    parser.add_argument("--dll", type=Path, default=None, help="Native library; default native/bin-scissors")
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
    live = not args.headless and not args.native_only
    if args.render_every is None:
        args.render_every = 1 if live else 3
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
    if scene:
        settings = replace(settings, contact_model=scene.contact_model, young_pa=scene.young_pa)
    if args.vertical_spacing is None:
        args.vertical_spacing = scene.vertical_spacing if scene else .05
    elif scene and not np.isclose(args.vertical_spacing, scene.vertical_spacing):
        parser.error(f"the {args.scene} scissors poses put the shear plane mid-cell for vertical spacing "
                     f"{scene.vertical_spacing} cm only")
    stages = scene.stages if scene else STAGES
    (output / "stages.json").write_text(json.dumps([asdict(stage) for stage in stages], indent=2), encoding="utf8")
    if scene:
        grid = scene.build_grid(settings.density_kg_m3)
    else:
        grid = build_grid(args.spacing, density=settings.density_kg_m3, vertical_spacing=args.vertical_spacing)
    # A scene with its own tissue shape frames it with its demo shot from the start.
    kidney = scene.shape if scene else None
    camera = scene.shot if kidney is not None else None
    flow = Flow(settings, scene, stages, grid, args.steps, output)
    view, video = None, None
    status, error = "partial", None
    started = time.monotonic()
    try:
        if live:
            # The OpenGL context and the window stay on this thread, the native session on the worker.
            link = LiveLink(settings.dt, args.render_every)
            with ThreadPoolExecutor(max_workers=1) as executor:
                worker = executor.submit(step_live, flow, link, grid, settings, args.dll, stages[0].command)
                try:
                    view = EndoscopicView(settings, args.width, args.height, args.viewer, supersample=1, kidney=kidney,
                                          camera=camera)
                    video = cv2.VideoWriter(str(output / "endoscopic_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                            LIVE_FPS, (args.width, args.height))
                    if not video.isOpened():
                        raise RuntimeError("Could not open MP4 writer")
                    if args.demo:
                        view.camera.set_pose(pos=scene.shot[0], lookat=scene.shot[1])
                    initial = take_still(link, worker)
                    if initial is not None:
                        view.update(initial.surface, initial.obs)
                        frame, depth = view.render(depth=True)
                        cv2.imwrite(str(output / "initial.png"), frame)
                        np.save(output / "initial_depth_m.npy", depth)
                        cv2.imshow(WINDOW, frame)
                        cv2.waitKey(1)
                        link.running.set()
                        play_live(view, video, link, worker, frame, output)
                finally:
                    link.stopping.set()
                    link.running.set()
            final = worker.result()
            if link.total_wall_s:
                print(f"PLAYBACK: realtime x{link.total_steps * settings.dt / link.total_wall_s:.2f}, display "
                      f"{link.total_frames / link.total_wall_s:.1f} fps over the {link.total_steps} steps of the "
                      f"completed stages", flush=True)
        else:
            with NativeTissue(grid, settings, args.dll, stages[0].command) as tissue, \
                    (output / "trajectory.jsonl").open("w", encoding="utf8") as trajectory:
                flow.open(tissue)
                sink = None
                if not args.native_only:
                    view = EndoscopicView(settings, args.width, args.height, args.viewer, kidney=kidney, camera=camera)
                    video = cv2.VideoWriter(str(output / "endoscopic_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                            1 / (settings.dt * args.render_every), (args.width, args.height))
                    if not video.isOpened():
                        raise RuntimeError("Could not open MP4 writer")
                    if args.demo:
                        view.camera.set_pose(pos=scene.shot[0], lookat=scene.shot[1])
                    view.update(tissue.surface(), flow.obs)
                    frame, depth = view.render(depth=True)
                    cv2.imwrite(str(output / "initial.png"), frame)
                    np.save(output / "initial_depth_m.npy", depth)
                    sink = Recording(view, video, args.render_every, output)
                flow.run(tissue, trajectory, sink)
        status = "passed" if flow.sequence.metrics.acceptance_passed else "partial"
        if args.demo and status == "passed":
            # The final solved state, seen from a camera that moves in over
            # the wound for 2 s and then holds for 3 s; the simulation does
            # not advance.
            surface = flow.surface
            wall = surface.anchors[surface.triangles[:, 0], 3] > .5
            wound = surface.vertices[surface.triangles[wall]].reshape(-1, 3)
            target = to_genesis(((wound.min(axis=0) + wound.max(axis=0)) / 2)[None])[0]
            fps = LIVE_FPS if live else 1 / (settings.dt * args.render_every)
            moves = round(2 * fps)
            due = time.perf_counter()
            for frame_index in range(moves + round(3 * fps)):
                u = min(1, (frame_index + 1) / moves)
                u = u * u * (3 - 2 * u)
                pos = (1 - u) * np.array(scene.shot[0]) + u * (target + scene.closeup_offset)
                lookat = (1 - u) * np.array(scene.shot[1]) + u * target
                view.camera.set_pose(pos=pos, lookat=lookat)
                frame, _ = view.render()
                video.write(frame)
                if live:
                    cv2.imshow(WINDOW, frame)
                    cv2.waitKey(1)
                    due += 1 / fps
                    time.sleep(max(0.0, due - time.perf_counter()))
            cv2.imwrite(str(output / "wound-closeup.png"), frame)
        if live and view is not None:
            view.update(final.surface, final.obs)
            cv2.imshow(WINDOW, view.render()[0])
            cv2.waitKey(0)
    except (RuntimeError, ValueError, OSError) as ex:
        status, error = "failed", str(ex)
        raise
    finally:
        if video is not None:
            video.release()
        if view is not None:
            view.close()
        if live:
            cv2.destroyAllWindows()
        dll = flow.dll
        report = {"status": status, "error": error, "last_stage": flow.stage, "steps": flow.step,
                  "elapsed_s": time.monotonic() - started, "settings": asdict(settings),
                  "grid_spacing_cm": args.spacing, "cell_size_cm": grid.cell_size.tolist(), "nodes": len(grid.positions),
                  "scene": args.scene if scene else None, "tab_cm": asdict(scene.tab) if scene else None,
                  "dll": str(dll) if dll else None,
                  "dll_sha256": hashlib.sha256(dll.read_bytes()).hexdigest() if dll else None,
                  "genesis_view_created": view is not None,
                  "metrics": asdict(flow.sequence.metrics) if flow.sequence else None,
                  "limitations": ["Single native session", "End-effector control; no robot arm dynamics",
                                  "Anchor-assisted grasp", "Finite capsule cutting edge", "Uncalibrated tissue"]}
        (output / "surface-audit.json").write_text(json.dumps(flow.surface_audits, indent=2), encoding="utf8")
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        print(f"{status.upper()}: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
