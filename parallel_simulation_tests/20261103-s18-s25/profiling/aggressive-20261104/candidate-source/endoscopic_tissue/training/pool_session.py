"""Worker-local training protocol. No CUDA session or IPC starts on import."""
from pathlib import Path
import json
from uuid import uuid4

import numpy as np
import psutil

from .data import EpisodeWriter
from .env import SceneConfig
from .geometry import WallGeometry
from .physics import PhysicsOverrides
from .policies import observation, privileged_state, contract_manifest
from .provenance import json_hash


def scene_config(value):
    if value is None or isinstance(value, SceneConfig):
        return value
    if not isinstance(value, dict):
        raise TypeError('scene_config must be a mapping')
    value = dict(value)
    for key, cls in (('physics', PhysicsOverrides), ('geometry', WallGeometry)):
        if value.get(key) is not None and not isinstance(value[key], cls):
            value[key] = cls(**value[key])
    return SceneConfig(**value)


def environment_config(value):
    from .env import EnvironmentConfig
    from .actions import ActionBounds
    from .quality import QualityLimits
    value=dict(value)
    value['scene']=scene_config(value['scene'])
    if isinstance(value['bounds'],dict):value['bounds']=ActionBounds(**value['bounds'])
    if isinstance(value.get('limits'),dict):value['limits']=QualityLimits(**value['limits'])
    return EnvironmentConfig(**value)


def payload_bytes(value):
    if isinstance(value, np.ndarray): return value.nbytes
    if isinstance(value, dict): return sum(len(str(k).encode()) + payload_bytes(v) for k,v in value.items())
    if isinstance(value, (list, tuple)): return sum(payload_bytes(v) for v in value)
    return len(json.dumps(value).encode())


class WorkerSession:
    def __init__(self, environment, directory, *, record=True,
                 observation_contract='simulated_tool_measurements_v1', visual_tracking=False,
                 privileged_fields=(), vision=None, max_snapshot_bytes=128*1024**2,
                 max_snapshot_capture_bytes=2*1024**3, max_response_bytes=32*1024**2):
        for limit in (max_snapshot_bytes, max_snapshot_capture_bytes, max_response_bytes):
            if type(limit) is not int or limit < 1: raise ValueError('Explicit positive byte budgets required')
        self.env, self.directory, self.record = environment, Path(directory), record
        self.contract, self.tracking = observation_contract, visual_tracking
        self.declaration = contract_manifest(self.contract, visual_tracking=visual_tracking)
        allowed = {'positions_m','topology','telemetry','solved_tools','visual_labels'}
        if set(privileged_fields)-allowed or len(set(privileged_fields)) != len(privileged_fields):
            raise ValueError('Unknown or duplicate privileged channel')
        self.fields = tuple(privileged_fields)
        self.budget, self.capture_budget, self.response_budget = max_snapshot_bytes, max_snapshot_capture_bytes, max_response_bytes
        self.snapshots, self.reserved = {}, 0
        self.owner = uuid4().hex
        self.writer = None
        self.vision_config, self.vision = vision, None
        self.episode = 0

    def _observe(self, state):
        result = dict(observation=observation(state, self.env.previous_action,
            contract=self.contract, visual_tracking=self.tracking))
        if self.fields:
            result['privileged'] = privileged_state(state, self.fields)
        if payload_bytes(result) > self.response_budget:
            raise MemoryError('Observation payload exceeds explicit IPC budget')
        return result

    def _begin(self, state, context):
        if self.writer is not None: self.writer.close('explicit_branch_or_reset'); self.writer = None
        context.update(observation=self.declaration, privileged_fields=list(self.fields),
                       snapshot_cache_budget_bytes=self.budget)
        context['context_sha256'] = json_hash({k:v for k,v in context.items() if k != 'context_sha256'})
        self.env.context = context
        static=self.env.static_arrays()
        if self.vision is not None:
            context['vision']=self.vision.manifest
            static.update(self.vision.static_arrays)
            if 'vision' not in state.metadata:self.vision.capture(self.env,state)
            context['context_sha256']=json_hash({k:v for k,v in context.items() if k != 'context_sha256'})
        path = self.directory / f'episode-{self.episode:06d}'
        self.episode += 1
        if self.record: self.writer = EpisodeWriter(path, context, state, static)
        return dict(episode_id=state.metadata['episode_id'], context=context,
                    quality=state.metadata['quality'], initial_action=self.env.previous_action.tolist(),
                    **self._observe(state))

    def execute(self, command, message):
        if command == 'reset':
            selected = scene_config(message.get('scene_config'))
            if self.writer is not None: self.writer.close('explicit_reset'); self.writer = None
            if self.vision is not None:self.vision.close();self.vision=None
            self.snapshots.clear(); self.reserved = 0
            state, context = self.env.reset(seed=message.get('seed', self.episode),
                scene_config=selected, goal=message.get('goal'))
            if self.vision_config is not None:
                from .vision import SynchronizedVision
                self.vision=SynchronizedVision(self.env,self.vision_config)
            return self._begin(state, context)
        if command == 'step':
            t = self.env.step(message['action'])
            if self.vision is not None and t.after is not None:
                from dataclasses import replace
                try:t.timings_ms['vision']=self.vision.capture(self.env,t.after)
                except Exception as error:
                    t=replace(t,after=None,terminated=True,truncated=False,task=None,bootstrap_allowed=False,
                              valid_outcome=False,reason='numerical:vision_observation_failure:'+repr(error),
                              failure_nodes_native_cm=self.env.observed.nodes.copy())
                    self.env.done=True
            if self.writer is not None: self.writer.append(t)
            result = dict(episode_id=t.before.metadata['episode_id'],
                quality=t.after.metadata['quality'] if t.after else None,
                terminated=t.terminated, truncated=t.truncated, reason=t.reason,
                bootstrap_allowed=t.bootstrap_allowed, valid_outcome=t.valid_outcome,
                task=t.task, timings_ms=t.timings_ms, action=t.action)
            if t.after is not None: result.update(self._observe(t.after))
            else: result['observation'] = None
            if (t.terminated or t.truncated) and self.writer is not None:
                self.writer.close(t.reason or 'terminal'); self.writer = None
            return result
        if command == 'snapshot':
            if self.env.native is None or self.env.state is None:
                raise RuntimeError('Reset before snapshot')
            hot = message.get('hot', False)
            if type(hot) is not bool: raise TypeError('hot must be bool')
            from .snapshot import SnapshotBackend
            native_size = SnapshotBackend(self.env.native).size()
            if native_size > self.capture_budget or psutil.virtual_memory().available < 16*1024**3 + 2*native_size:
                raise MemoryError('Snapshot capture reserve exceeded before native allocation')
            snap = self.env.snapshot()
            required = snap.size_bytes + (snap.metadata['native_bytes'] if hot else 0)
            if self.reserved + required > self.budget:
                raise MemoryError('Snapshot cache budget exceeded; old snapshots retained')
            if hot:
                from .prepared_snapshot import PreparedSnapshot
                snap = PreparedSnapshot(snap, max_host_bytes=self.budget-self.reserved)
            key = self.owner + ':' + uuid4().hex
            self.snapshots[key] = (snap, required)
            self.reserved += required
            return dict(snapshot_id=key, snapshot_sha256=snap.sha256, hot=hot,
                        cached_bytes=self.reserved, metadata=snap.metadata,
                        capture_native_bytes=native_size, max_capture_bytes=self.capture_budget)
        if command in ('restore', 'drop_snapshot'):
            key = message['snapshot_id']
            if key not in self.snapshots: raise KeyError('Unknown, reset-invalidated, or foreign worker snapshot ID')
            snap, size = self.snapshots[key]
            if command == 'drop_snapshot':
                del self.snapshots[key]; self.reserved -= size
                return dict(dropped=key, cached_bytes=self.reserved)
            options = dict(branch_label=message.get('branch_label'))
            if 'goal' in message: options['goal'] = message['goal']
            state, context = self.env.restore(snap, **options)
            return self._begin(state, context)
        raise ValueError('Unknown pool command')

    def close(self):
        if self.vision is not None:self.vision.close();self.vision=None
        if self.writer is not None:
            self.writer.close('worker_closed'); self.writer = None
        self.snapshots.clear(); self.reserved = 0
