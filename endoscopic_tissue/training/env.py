"""Opt-in, headless reset/step adapter over the unchanged native solver."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from ..native_backend import NativeTissue
from ..scenario import SCISSORS_SCENES, SCISSORS_SPACING, drive_command, interpolate, scissors_settings, stage_start
from ..grid import build_grid
from .actions import ActionBounds, action_to_command, command_to_action
from .provenance import json_hash, manifest, pinned_dll
from .quality import QualityLimits, QualityMonitor
from .grasp import GraspMonitor
from .tasks import TaskGoal, TaskMonitor
from .cutting import cut_origin_bracket
from .physics import PhysicsOverrides, resolve_physics
from .native import TrainingNativeTissue


@dataclass(frozen=True)
class SceneConfig:
    scene: str = "wall-traction"
    preparation: str = "held"
    contact_model: int | None = None
    dt_s: float | None = None
    substeps: int | None = None
    spacing_scale: float = 1.
    physics: PhysicsOverrides | None = None

    def __post_init__(self):
        if self.physics is not None and not isinstance(self.physics, PhysicsOverrides):
            raise TypeError("physics must be PhysicsOverrides or None")
        if self.scene not in SCISSORS_SCENES or self.preparation not in ("settled", "held", "cut_ready"):
            raise ValueError("Unknown scene or preparation")
        if self.scene == "wall" and self.preparation == "cut_ready":
            raise ValueError("The wall scene approaches the scissors before grasping; use wall-traction for cut_ready")
        if self.contact_model not in (None, 1, 2):
            raise ValueError("Contact model must be 1 (pure friction) or 2 (assisted)")
        if self.dt_s is not None and (not np.isfinite(self.dt_s) or self.dt_s <= 0):
            raise ValueError('Invalid sensitivity time step')
        if self.substeps is not None and (not isinstance(self.substeps,int) or self.substeps < 1):
            raise ValueError('Invalid sensitivity substeps')
        if not np.isfinite(self.spacing_scale) or not .5 <= self.spacing_scale <= 2:
            raise ValueError('Sensitivity spacing scale must be within [0.5,2]')


@dataclass(frozen=True)
class EnvironmentConfig:
    dll_path: str
    dll_sha256: str
    bounds: ActionBounds
    scene: SceneConfig = SceneConfig()
    cutting_enabled: bool = False
    max_steps: int = 120
    full_surface_checks: bool = True
    diagnostic_continue: bool = False
    limits: QualityLimits = QualityLimits()

    def __post_init__(self):
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")


@dataclass(frozen=True)
class State:
    metadata: dict
    arrays: dict
    topology: dict


@dataclass(frozen=True)
class StepResult:
    before: State
    after: State | None
    action: dict
    terminated: bool
    truncated: bool
    reason: str | None
    timings_ms: dict
    failure_nodes_native_cm: np.ndarray | None = None
    task: dict | None = None
    bootstrap_allowed: bool = False


class ResetFailure(RuntimeError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__("Preparation failed its checks; no initial training state returned")


class NativeEnvironment:
    """One native session. Standard actions cannot change physical switches.

    Cutting permission is fixed for the episode. In a cutting-enabled local
    task the native law still decides whether closing/squeeze/load can fracture.
    Arbitrary withdrawal/release is not a validated action domain. Diagnostic
    command replay is explicitly separated from the 11D policy interface.
    """
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.native = None
        self.state = None
        self.context = None
        self.last_reset_diagnostic = None
        self.last_reset_failure_arrays = None
        self.done = True
        self.task = None

    def reset(self, seed=0, scene_config=None, goal=None):
        reset_start = time.perf_counter()
        dll = pinned_dll(self.config.dll_path, self.config.dll_sha256)
        scene_config = scene_config or self.config.scene
        if not isinstance(scene_config, SceneConfig):
            raise TypeError("scene_config must be a SceneConfig")
        self.close()
        self.state = self.context = None
        self.observed = None
        self.rng = np.random.default_rng(seed)
        self.episode_id = uuid4().hex
        self.scene_config = scene_config
        scene = SCISSORS_SCENES[scene_config.scene]
        contact_model = scene.contact_model if scene_config.contact_model is None else scene_config.contact_model
        self.settings = replace(scissors_settings(), young_pa=scene.young_pa, contact_model=contact_model)
        base_dt = self.settings.dt
        self.settings = replace(self.settings,
                                dt=scene_config.dt_s if scene_config.dt_s is not None else self.settings.dt,
                                substeps=scene_config.substeps if scene_config.substeps is not None else self.settings.substeps)
        self.settings, self.native_parameters = resolve_physics(self.settings, scene_config.physics)
        if scene_config.spacing_scale == 1:
            self.grid = scene.build_grid(self.settings.density_kg_m3)
        else:
            geometry = {'shape':scene.shape} if scene.shape is not None else {'tab':scene.tab}
            self.grid = build_grid(SCISSORS_SPACING*scene_config.spacing_scale,
                                   vertical_spacing=scene.vertical_spacing*scene_config.spacing_scale,
                                   density=self.settings.density_kg_m3, **geometry)
        self.quality = QualityMonitor(self.grid, self.config.limits)
        self.grasp = GraspMonitor(self.settings, self.grid.inside)
        self.task = None
        self.preparation = []
        self.step_id = 0
        self.last_surface = None
        self._topology_revision = None
        self._topology = None
        self.last_reset_diagnostic = None
        self.last_reset_failure_arrays = None
        stop_stage = {"settled": "settle", "held": "lift", "cut_ready": "blade_approach"}[scene_config.preparation]
        try:
            if scene_config.physics is None:
                self.native = NativeTissue(self.grid, self.settings, dll, scene.stages[0].command)
            else:
                self.native = TrainingNativeTissue(self.grid, self.settings, dll, scene.stages[0].command,
                                                  native_parameters=self.native_parameters)
            self.observed = self.native.observe()
            previous = None
            requested = scene.stages[0].command
            for stage in scene.stages:
                start = stage_start(stage.name, requested, self.observed)
                stage_steps = max(1,round(stage.steps*base_dt/self.settings.dt))
                for index in range(stage_steps):
                    preparation_start = time.perf_counter()
                    requested = interpolate(start, stage.command, (index + 1) / (stage_steps * stage.ramp))
                    executed = drive_command(requested, self.observed, self.settings)
                    previous = self.observed
                    self.observed = self.native.step(executed)
                    assessed, _ = self._assess(previous, executed)
                    self.preparation.append({"native_tick": self.native.tick, "stage": stage.name,
                                             "requested_command_native": asdict(requested),
                                             "executed_command_native": asdict(executed),
                                             "grasper_hinge_m": (self.observed.hinge * .01).tolist(),
                                             "jaw_load_n": list(self.observed.jaw_load_n),
                                             "grip_points": self.observed.grip_points, "quality": assessed,
                                             "wall_time_ms": (time.perf_counter()-preparation_start)*1000})
                    if self._invalid(assessed) and not self.config.diagnostic_continue:
                        raise ResetFailure({"episode_id": self.episode_id, "seed": int(seed),
                                            "failure": assessed, "preparation": self.preparation})
                if stage.name == 'close':
                    # Bind witnesses before lifting, as the original Sequence diagnostic does.
                    self.preparation[-1]['grasp'] = self.grasp.assess_native(self.observed)
                if stage.name == stop_stage:
                    break
            else:
                raise ValueError("Preparation end stage is missing")
            self.previous_action = command_to_action(requested)
            # Check the declared domain before exposing s0; no hidden clamping at reset.
            self.config.bounds.process(self.previous_action, self.previous_action, self.settings.dt)
            self.state = self._snapshot(previous, assessed)
            if scene_config.preparation in ("held", "cut_ready"):
                grasp = self.state.metadata['grasp']
                acquired = (self.observed.grip_points > 0 if contact_model == 2
                            else grasp['friction_hold_evidence'])
                if not acquired:
                    raise ResetFailure({"episode_id": self.episode_id, "seed": int(seed),
                                        "failure": "hold_not_acquired", "contact_model": contact_model,
                                        "grasp": grasp, "preparation": self.preparation})
            if goal is not None:
                goal = goal if isinstance(goal, TaskGoal) else TaskGoal(**goal)
                self.task = TaskMonitor(goal, self.static_arrays(), self.state)
                goal = self.task.goal.to_dict()
            self.context = manifest(self.native, self.settings, asdict(scene_config), self.config.bounds,
                                    self.native_parameters)
            self.context.update(episode_id=self.episode_id, seed=int(seed), goal=goal,
                                sensitivity_overrides_are_not_default_replacements=True,
                                cutting_enabled=self.config.cutting_enabled,
                                diagnostic_continue=self.config.diagnostic_continue,
                                full_surface_checks=self.config.full_surface_checks,
                                quality_limits=asdict(self.config.limits),
                                preparation_steps=self.native.tick, initial_zero_gravity_native_steps=1,
                                action_semantics="absolute_target; one native dt per step",
                                initialization_tick_excluded_from_episode_time=True,
                                node_count=len(self.grid.positions), material_node_count=int(self.grid.inside.sum()),
                                active_node_count=int(self.grid.active.sum()))
            self.context["reset_wall_time_ms"] = (time.perf_counter()-reset_start)*1000
            self.context["context_sha256"] = json_hash(self.context)
            self.done = False
            return self.state, self.context
        except Exception as error:
            self.last_reset_diagnostic = getattr(error, "diagnostic", {"error": str(error), "preparation": self.preparation})
            self.last_reset_diagnostic["reset_wall_time_ms"] = (time.perf_counter()-reset_start)*1000
            if self.native is not None and self.observed is not None:
                self.last_reset_failure_arrays = {"nodes_native_cm": self.observed.nodes.copy(),
                                                  "neighbors": self.observed.neighbors.copy(),
                                                  "rest_positions_native_cm": self.grid.positions.copy(),
                                                  "grip_flags": self.observed.grip_flags.copy()}
                self.last_reset_diagnostic["raw_state_note"] = "last successfully observed preparation state; native failure may be later"
            self.close()
            raise

    @staticmethod
    def _invalid(quality):
        return not quality["checked_metrics_passed"] or not quality["within_declared_local_scope"]

    def _assess(self, previous, command):
        started = time.perf_counter()
        material = self.native.material_quality()
        export_start = time.perf_counter()
        self.last_surface = self.native.surface() if self.config.full_surface_checks else None
        export_ms = (time.perf_counter() - export_start) * 1000
        result = self.quality.assess(self.observed, previous, command, self.settings, material,
                                     self.last_surface, self.native.scissors_geometry)
        return result, {"surface_export": export_ms, "quality_and_export": (time.perf_counter()-started)*1000}

    def _snapshot(self, previous, quality):
        o = self.observed
        arrays = {"positions_m": o.nodes * .01, "grip_flags": o.grip_flags.copy()}
        arrays["estimated_velocity_m_s"] = ((o.nodes - previous.nodes) * (.01 / self.settings.dt)
                                               if previous is not None else np.zeros_like(o.nodes))
        for value in arrays.values():
            value.flags.writeable = False
        if self._topology_revision != self.native.topology_revision:
            fractions = self.native.cut_edge_fractions()
            fractions.flags.writeable = False
            self._topology_revision = self.native.topology_revision
            self._topology = {"revision": int(self._topology_revision), "neighbors": o.neighbors,
                              "cut_fractions": fractions}
        s = o.scissors
        tools = {"grasper_hinge_m": (o.hinge*.01).tolist(), "grasper_gap_m": o.jaw_gap*.01,
                 "grasper_axis": list(self.settings.jaw_axis), "grasper_normal": list(self.settings.jaw_normal),
                 "jaw_load_n": list(o.jaw_load_n), "grip_points": o.grip_points,
                 "grip_error_m": o.grip_error_m,
                 "scissors_hinge_m": (s.hinge*.01).tolist(), "scissors_axis": s.axis.tolist(),
                 "scissors_normal": s.normal.tolist(), "scissors_gap_m": s.gap*.01,
                 "scissors_edges_m": (s.edges*.01).tolist(), "scissors_phase": s.phase,
                 "scissors_loads": asdict(s.loads), "reaction_n": o.reaction_n.tolist()}
        metadata = {"episode_id": self.episode_id, "step_id": self.step_id,
                    "t_sim_s": self.step_id*self.settings.dt, "dt_s": self.settings.dt,
                    "native_tick": self.native.tick, "gravity_fraction": min(1, self.native.tick*self.settings.dt/.5),
                    "topology_revision": int(self._topology_revision), "tools": tools, "quality": quality}
        state = State(metadata, arrays, self._topology)
        metadata['grasp'] = self.grasp.assess(state)
        return state

    def step(self, action):
        if self.done or self.native is None:
            raise RuntimeError("Call reset before step, and after every terminal result")
        limited, record = self.config.bounds.process(action, self.previous_action, self.settings.dt)
        record['previous_target_si'] = self.previous_action.tolist()
        command = action_to_command(limited, cutting_enabled=self.config.cutting_enabled)
        result = self._advance(command, record)
        self.previous_action = limited.copy()
        return result

    def replay_command(self, requested):
        """Exact legacy request for paired diagnostic replay, never a policy bypass."""
        if not self.config.diagnostic_continue:
            raise RuntimeError("Raw command replay requires explicit diagnostic mode")
        if self.done or self.native is None:
            raise RuntimeError("Reset required")
        return self._advance(requested, {"diagnostic_command_replay": True,
                                         "policy": command_to_action(requested).tolist()})

    def _advance(self, requested, action_record):
        started = time.perf_counter()
        before, previous = self.state, self.observed
        executed = drive_command(requested, previous, self.settings)
        action_record = {**action_record, "requested_command_native": asdict(requested),
                         "executed_command_native": asdict(executed),
                         "executed_target_si": command_to_action(executed).tolist(),
                         "flags_owner": "environment_configuration_or_explicit_diagnostic_replay"}
        native_start = time.perf_counter()
        try:
            self.observed = self.native.step(executed)
        except (RuntimeError, OSError) as error:
            from ..native_backend import pointer
            raw = np.empty_like(self.grid.positions)
            try:
                self.native.check("LCS_GetCornerPos", pointer(raw))
            except (RuntimeError, OSError):
                raw = None
            self.done = True
            return StepResult(before, None, action_record, True, False, "native_failure: "+str(error),
                              {"total": (time.perf_counter()-started)*1000}, raw)
        timings = {"native_step": (time.perf_counter()-native_start)*1000}
        self.step_id += 1
        try:
            quality, quality_timings = self._assess(previous, executed)
            timings.update(quality_timings)
            snapshot_start = time.perf_counter()
            next_state = self._snapshot(previous, quality)
            if quality['metrics']['new_cut_edges']:
                next_state.metadata['cut_origin_bracket'] = cut_origin_bracket(before,next_state,self.grid.inside)
        except (RuntimeError, OSError, ValueError) as error:
            # A completed physics step with a failed observation is not an accepted s(t+1).
            self.done = True
            timings["total"] = (time.perf_counter()-started)*1000
            return StepResult(before, None, action_record, True, False,
                              "observation_failure: "+str(error), timings, self.observed.nodes.copy())
        self.state = next_state
        timings["snapshot"] = (time.perf_counter()-snapshot_start)*1000
        invalid = self._invalid(quality)
        terminated = invalid and not self.config.diagnostic_continue
        truncated = self.step_id >= self.config.max_steps and not terminated
        reason = ("numerical_failure" if not quality["checked_metrics_passed"] else "outside_local_scope") if terminated else None
        if truncated:
            reason = "time_limit"
        task = None
        if self.task is not None:
            task = self.task.evaluate(before, self.state, action_record, self.config.bounds)
            if not invalid and (task['success'] or task['failure'] is not None):
                terminated, truncated = True, False
                reason = 'task_success' if task['success'] else 'task_failure:'+task['failure']
        self.done = terminated or truncated
        timings["total"] = (time.perf_counter()-started)*1000
        bootstrap = not invalid and not terminated
        return StepResult(before, self.state, action_record, terminated, truncated, reason, timings,
                          task=task, bootstrap_allowed=bootstrap)

    def static_arrays(self):
        if self.native is None:
            raise RuntimeError("No active environment")
        return {"node_ids": np.arange(len(self.grid.positions), dtype=np.int32),
                "rest_positions_m": self.grid.positions*.01, "material_mask": self.grid.inside,
                "active_mask": self.grid.active, "pinned_mask": self.grid.pinned,
                "initial_neighbors": self.grid.neighbors}

    def close(self):
        if self.native is not None:
            self.native.close()
        self.native = None
        self.done = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
