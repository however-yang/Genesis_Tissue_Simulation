# Isolated stable-acquisition overlay. Frozen parent SHA256: 39c3768f6ebf1cfc4e21b30c36f237467a944b2a6e285bab9e456429356d03a8
"""Opt-in, headless reset/step adapter over the unchanged native solver."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time
import threading
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
from .damage import DamageMonitor
from .geometry import WallGeometry, scene_for_config


@dataclass(frozen=True)
class SceneConfig:
    scene: str = "wall-traction"
    preparation: str = "held"
    contact_model: int | None = None
    dt_s: float | None = None
    substeps: int | None = None
    spacing_scale: float = 1.
    physics: PhysicsOverrides | None = None
    geometry: WallGeometry | None = None
    acquisition_max_steps: int = 0
    acquisition_stable_steps: int = 5

    def __post_init__(self):
        if type(self.acquisition_max_steps) is not int or not 0 <= self.acquisition_max_steps <= 240:
            raise ValueError('acquisition_max_steps must be an integer in [0,240]')
        if type(self.acquisition_stable_steps) is not int or not 1 <= self.acquisition_stable_steps <= 30:
            raise ValueError('acquisition_stable_steps must be an integer in [1,30]')
        if self.acquisition_max_steps and (self.contact_model != 1 or self.preparation not in ('held','cut_ready')
                or self.acquisition_max_steps < self.acquisition_stable_steps):
            raise ValueError('Stable acquisition requires explicit pure-friction held/cut_ready and sufficient waiting steps')
        if self.physics is not None and not isinstance(self.physics, PhysicsOverrides):
            raise TypeError("physics must be PhysicsOverrides or None")
        if self.geometry is not None:
            if not isinstance(self.geometry,WallGeometry):
                raise TypeError('geometry must be WallGeometry or None')
            if self.scene!='wall-traction' or self.spacing_scale < 1:
                raise ValueError('Parametric geometry requires wall-traction and spacing at least 1 mm')
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
    failure_telemetry: bool = False
    outcome_semantics: str = "legacy_v1"
    instrument_collision_checks: bool = False
    fast_topology_checks: bool = False
    gpu_topology_checks: bool = False

    def __post_init__(self):
        if self.outcome_semantics not in ("legacy_v1", "physical_outcome_v1"):
            raise ValueError("Unknown outcome semantics")
        if type(self.instrument_collision_checks) is not bool:
            raise TypeError("instrument_collision_checks must be bool")
        if type(self.fast_topology_checks) is not bool:
            raise TypeError("fast_topology_checks must be bool")
        if type(self.gpu_topology_checks) is not bool or (self.gpu_topology_checks and not self.full_surface_checks):
            raise ValueError('GPU topology checks require full surface checks and explicit bool')
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if type(self.failure_telemetry) is not bool:
            raise ValueError("failure_telemetry must be bool")


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
    valid_outcome: bool | None = None


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
        self._state_lock = threading.RLock()
        self._step_in_progress = False
        self.config = config
        self.native = None
        self.state = None
        self.context = None
        self.last_reset_diagnostic = None
        self.last_reset_failure_arrays = None
        self.done = True
        self.task = None

    def reset(self, seed=0, scene_config=None, goal=None):
        with self._state_lock:
            return self._reset(seed, scene_config, goal)

    def _reset(self, seed=0, scene_config=None, goal=None):
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
        scene = scene_for_config(scene_config)
        self.scene = scene
        contact_model = scene.contact_model if scene_config.contact_model is None else scene_config.contact_model
        self.settings = replace(scissors_settings(), young_pa=scene.young_pa, contact_model=contact_model)
        base_dt = self.settings.dt
        self.settings = replace(self.settings,
                                dt=scene_config.dt_s if scene_config.dt_s is not None else self.settings.dt,
                                substeps=scene_config.substeps if scene_config.substeps is not None else self.settings.substeps)
        self.settings, self.native_parameters = resolve_physics(self.settings, scene_config.physics)
        if scene_config.geometry is not None:
            self.grid = scene_config.geometry.build_grid(self.settings.density_kg_m3,scene_config.spacing_scale)
        elif scene_config.spacing_scale == 1:
            self.grid = scene.build_grid(self.settings.density_kg_m3)
        else:
            geometry = {'shape':scene.shape} if scene.shape is not None else {'tab':scene.tab}
            self.grid = build_grid(SCISSORS_SPACING*scene_config.spacing_scale,
                                   vertical_spacing=scene.vertical_spacing*scene_config.spacing_scale,
                                   density=self.settings.density_kg_m3, **geometry)
        if self.config.gpu_topology_checks:
            from .gpu_topology import topology_statistics
            self.quality=QualityMonitor(self.grid,self.config.limits,topology_function=lambda surface:topology_statistics(self.native,surface))
        elif self.config.fast_topology_checks:
            from .topology_fast import topology_statistics_fast
            self.quality = QualityMonitor(self.grid, self.config.limits, topology_function=topology_statistics_fast)
        else:
            self.quality = QualityMonitor(self.grid, self.config.limits)
        telemetry = (self.config.outcome_semantics == "physical_outcome_v1" or
                     self.config.failure_telemetry or self.native_parameters['tear_stretch_ratio'] > 0
                     or self.native_parameters['grasp_core_break_force_n'] != 150.)
        self.damage = DamageMonitor(self.grid) if telemetry else None
        self.failure_metrics = None
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
            if scene_config.physics is None and not telemetry:
                self.native = NativeTissue(self.grid, self.settings, dll, scene.stages[0].command)
            else:
                self.native = TrainingNativeTissue(self.grid, self.settings, dll, scene.stages[0].command,
                                                  native_parameters=self.native_parameters,telemetry=telemetry)
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
                    if self.failure_metrics is not None:
                        # _assess creates a fresh record each step. Keep the
                        # already computed preparation peaks and break events;
                        # exporting only s0 would hide them from calibration.
                        self.preparation[-1]['failure_physics'] = self.failure_metrics
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
            # Opt-in preparation protocol, not a different contact law. Keep
            # the final requested tool pose and continue ordinary physical
            # steps until the unchanged friction evidence remains true for a
            # consecutive window. Every extra step receives the same quality
            # checks and is included in preparation/video/context provenance.
            if scene_config.acquisition_max_steps:
                consecutive = 0
                for acquisition_step in range(scene_config.acquisition_max_steps):
                    preparation_start = time.perf_counter()
                    previous = self.observed
                    executed = drive_command(requested, self.observed, self.settings)
                    self.observed = self.native.step(executed)
                    assessed, _ = self._assess(previous, executed)
                    grasp = self.grasp.assess_native(self.observed)
                    self.preparation.append({"native_tick": self.native.tick,
                        "stage": "friction_acquisition_settle", "requested_command_native": asdict(requested),
                        "executed_command_native": asdict(executed),
                        "grasper_hinge_m": (self.observed.hinge*.01).tolist(),
                        "jaw_load_n": list(self.observed.jaw_load_n), "grip_points": self.observed.grip_points,
                        "quality": assessed, "grasp": grasp,
                        "wall_time_ms": (time.perf_counter()-preparation_start)*1000})
                    if self.failure_metrics is not None:
                        self.preparation[-1]['failure_physics'] = self.failure_metrics
                    if self._invalid(assessed):
                        raise ResetFailure({"episode_id": self.episode_id, "seed": int(seed),
                            "failure": assessed, "preparation": self.preparation})
                    consecutive = consecutive+1 if grasp['friction_hold_evidence'] else 0
                    self.preparation[-1]['acquisition_consecutive_steps'] = consecutive
                    if consecutive >= scene_config.acquisition_stable_steps:
                        break
                else:
                    raise ResetFailure({"episode_id": self.episode_id, "seed": int(seed),
                        "failure": "hold_not_acquired", "contact_model": contact_model,
                        "grasp": grasp, "stable_window_not_acquired": True, "preparation": self.preparation})
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
                                reset_lineage_id=self.episode_id,
                                sensitivity_overrides_are_not_default_replacements=True,
                                cutting_enabled=self.config.cutting_enabled,
                                diagnostic_continue=self.config.diagnostic_continue,
                                full_surface_checks=self.config.full_surface_checks,
                                failure_telemetry=telemetry,
                                outcome_semantics=self.config.outcome_semantics,
                                instrument_collision_checks=self.config.instrument_collision_checks,
                                fast_topology_checks=self.config.fast_topology_checks,
                                gpu_topology_checks=self.config.gpu_topology_checks,
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
        permitted_tears=0
        if self.damage is not None:
            self.failure_metrics=self.damage.assess(self.observed,self.native.last_training_metrics)
            permitted_tears=self.failure_metrics['new_tear_edges']
        export_start = time.perf_counter()
        self.last_surface = self.native.surface() if self.config.full_surface_checks else None
        export_ms = (time.perf_counter() - export_start) * 1000
        result = self.quality.assess(self.observed, previous, command, self.settings, material,
                                     self.last_surface, self.native.scissors_geometry,
                                     permitted_tear_edges=permitted_tears)
        if self.config.instrument_collision_checks:
            from .instrument_collision import instrument_distance
            pose = self.observed.scissors
            distance = instrument_distance(self.native.grasper_geometry, self.native.scissors_geometry,
                grasper_hinge=self.observed.hinge, grasper_axis=self.settings.jaw_axis,
                grasper_normal=self.settings.jaw_normal, grasper_gap=self.observed.jaw_gap,
                scissors_hinge=pose.hinge, scissors_axis=pose.axis, scissors_normal=pose.normal,
                scissors_gap=pose.gap)
            result['metrics']['instrument_min_distance_m'] = distance
            result['metrics']['instrument_distance_geometry'] = 'mesh_enclosing_jaw_boxes_and_native_wrist_capsule_v1'
            if distance < -self.config.limits.max_body_overlap_m:
                result['numerical_failure_reasons'].append('instrument_overlap_exceeded')
                result['checked_metrics_passed'] = False
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
                    "topology_revision": int(self._topology_revision), "tools": tools, "quality": quality,
                    "previous_target_si": self.previous_action.tolist()}
        if self.config.outcome_semantics == "physical_outcome_v1":metadata["events"] = []
        state = State(metadata, arrays, self._topology)
        metadata['grasp'] = self.grasp.assess(state)
        if self.failure_metrics is not None:
            metadata['failure_physics']=self.failure_metrics
        return state

    def step(self, action):
        with self._state_lock:
            return self._policy_step(action)

    def _policy_step(self, action):
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
        with self._state_lock:
            if not self.config.diagnostic_continue:
                raise RuntimeError("Raw command replay requires explicit diagnostic mode")
            if self.done or self.native is None:
                raise RuntimeError("Reset required")
            return self._advance(requested, {"diagnostic_command_replay": True,
                                             "policy": command_to_action(requested).tolist()})

    def _advance(self, requested, action_record):
        with self._state_lock:
            if self._step_in_progress:
                raise RuntimeError('Native step is already in progress')
            self._step_in_progress = True
            try:
                return self._advance_active(requested, action_record)
            finally:
                self._step_in_progress = False

    def _advance_active(self, requested, action_record):
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
            return StepResult(before, None, action_record, True, False, ("numerical:native_failure:" if self.config.outcome_semantics != "legacy_v1" else "native_failure: ")+str(error),
                              {"total": (time.perf_counter()-started)*1000}, raw, valid_outcome=False)
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
                              ("numerical:observation_failure:" if self.config.outcome_semantics != "legacy_v1" else "observation_failure: ")+str(error),
                              timings, self.observed.nodes.copy(), valid_outcome=False)
        next_state.metadata["previous_target_si"] = list(action_record.get("slew_limited",
            action_record.get("policy", self.previous_action)))
        self.state = next_state
        timings["snapshot"] = (time.perf_counter()-snapshot_start)*1000
        invalid = self._invalid(quality)
        terminated = invalid and not self.config.diagnostic_continue
        truncated = self.step_id >= self.config.max_steps and not terminated
        reason = ("numerical_failure" if not quality["checked_metrics_passed"] else "outside_local_scope") if terminated else None
        outcome = None
        if self.config.outcome_semantics == "physical_outcome_v1":
            from .outcomes import classify
            outcome = classify(quality, before.metadata["quality"], self.failure_metrics,
                               cutting_enabled=self.config.cutting_enabled)
            self.state.metadata["outcome"] = outcome
            invalid = not outcome["valid_outcome"]
            terminated = outcome["terminal_reason"] is not None and not self.config.diagnostic_continue
            reason = outcome["terminal_reason"] if terminated else None
            truncated = self.step_id >= self.config.max_steps and not terminated
        if truncated:
            reason = "time_limit"
        task = None
        if self.task is not None:
            task = self.task.evaluate(before, self.state, action_record, self.config.bounds)
            if not invalid and not (reason or '').startswith('damage:') and (task['success'] or task['failure'] is not None):
                terminated, truncated = True, False
                reason = 'task_success' if task['success'] else 'task_failure:'+task['failure']
        if outcome is not None:
            from .outcomes import step_events
            step_events(before, self.state, action_record, task)
        self.done = terminated or truncated
        timings["total"] = (time.perf_counter()-started)*1000
        bootstrap = not invalid and not terminated
        return StepResult(before, self.state, action_record, terminated, truncated, reason, timings,
                          task=task, bootstrap_allowed=bootstrap,
                          valid_outcome=not invalid if outcome is not None else None)

    def static_arrays(self):
        if self.native is None:
            raise RuntimeError("No active environment")
        return {"node_ids": np.arange(len(self.grid.positions), dtype=np.int32),
                "rest_positions_m": self.grid.positions*.01, "material_mask": self.grid.inside,
                "active_mask": self.grid.active, "pinned_mask": self.grid.pinned,
                "initial_neighbors": self.grid.neighbors}

    def snapshot(self):
        """Save a completed state using an explicitly selected snapshot ABI DLL."""
        from .snapshot import take_snapshot
        with self._state_lock:
            return take_snapshot(self)

    def restore(self, snapshot, *, branch_label=None, **kwargs):
        """Open a new branch. Optional goal replaces task/reward, never physics."""
        from .snapshot import restore_snapshot
        with self._state_lock:
            return restore_snapshot(self, snapshot, branch_label=branch_label, **kwargs)

    def close(self):
        with self._state_lock:
            if self.native is not None:
                self.native.close()
            self.native = None
            self.done = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
