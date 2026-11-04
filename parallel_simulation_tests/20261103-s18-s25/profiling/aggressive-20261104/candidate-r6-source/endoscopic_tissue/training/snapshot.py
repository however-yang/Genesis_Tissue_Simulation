"""Opt-in native snapshots, safe Python-state encoding and branch lineage.

Nothing in this module runs in the normal per-frame path. A snapshot contains
immutable compressed bytes, never live arrays, ctypes pointers or pickles.
"""
import ctypes as ct
import base64
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
import time
from uuid import uuid4

import numpy as np
import zstandard as zstd

from .provenance import json_hash, sha256, pinned_dll
from .tasks import TaskGoal, TaskMonitor, material_reference

MAGIC = b'LCSSNAP1'
MAX_BYTES = 4*1024**3
KEEP_TASK = object()
ENV_FIELDS = ('step_id', 'episode_id', 'previous_action', 'observed', 'state', 'last_surface',
              '_topology', '_topology_revision', 'preparation', 'context', 'done', 'failure_metrics')
GRASP_FIELDS = ('ids', 'initial_local', 'initial_world')
QUALITY_FIELDS = ('previous_neighbors', 'components')
DAMAGE_FIELDS = ('previous_neighbors', 'previous_counts')
TASK_FIELDS = frozenset(('goal', 'static', 'material', 'active', 'targets', 'region', 'hold', 'start_positions',
    'edge_a', 'edge_b', 'edge_slot', 'initial_neighbors', 'initial_alive', 'goal_edge_mask',
    'last_revision', 'topology_metrics', 'previous_progress', 'previous_outside', 'dwell_s', 'last_step', 'episode_id'))
VISION_ARRAYS = ('rgb', 'depth_pinhole_m', 'depth_rgb_aligned_m', 'depth_valid', 'camera_world_to_cv')


def _classes():
    from ..native_backend import Observation, Surface, ScissorsPose, ScissorsLoads
    from .env import State
    return {c.__name__: c for c in (Observation, Surface, ScissorsPose, ScissorsLoads, State, TaskGoal)}


def encode_python(value):
    """Numeric array table preserves shared topology identity without pickle."""
    arrays, ids, blobs = [], {}, []
    classes = _classes()

    def encode(x):
        if isinstance(x, np.ndarray):
            if id(x) not in ids:
                if x.dtype.kind not in 'biuf' or x.ndim > 8:
                    raise TypeError('Unsupported snapshot array dtype/rank')
                a = np.ascontiguousarray(x)
                ids[id(x)] = len(arrays)
                arrays.append({'dtype': a.dtype.str, 'shape': a.shape, 'bytes': a.nbytes,
                               'writeable': x.flags.writeable})
                blobs.append(a.tobytes())
            return {'$a': ids[id(x)]}
        if isinstance(x, np.generic):
            return encode(x.item())
        if is_dataclass(x):
            name = type(x).__name__
            if classes.get(name) is not type(x):
                raise TypeError('Unregistered snapshot dataclass: '+name)
            return {'$c': name, 'v': {f.name: encode(getattr(x, f.name)) for f in fields(x)}}
        if isinstance(x, dict):
            if not all(isinstance(k, str) for k in x):
                raise TypeError('Snapshot dictionaries require string keys')
            return {'$d': [[k, encode(v)] for k, v in x.items()]}
        if isinstance(x, tuple):
            return {'$t': [encode(v) for v in x]}
        if isinstance(x, list):
            return [encode(v) for v in x]
        if x is None or type(x) in (str, bool, int, float):
            return x
        raise TypeError('Unsupported snapshot value: '+type(x).__name__)

    tree = encode(value)
    header = json.dumps({'schema': 1, 'tree': tree, 'arrays': arrays}, separators=(',', ':'), allow_nan=False).encode()
    return struct.pack('<Q', len(header))+header+b''.join(blobs)


def decode_python(data):
    if len(data) < 8 or len(data) > MAX_BYTES:
        raise ValueError('Invalid Python snapshot size')
    size, = struct.unpack_from('<Q', data)
    if size > len(data)-8 or size > 128*1024**2:
        raise ValueError('Invalid Python state header')
    header = json.loads(data[8:8+size])
    if header['schema'] != 1 or len(header['arrays']) > 4096:
        raise ValueError('Unsupported Python state schema')
    arrays, offset = [], 8+size
    for item in header['arrays']:
        dtype = np.dtype(item['dtype'])
        shape = item['shape']
        if dtype.kind not in 'biuf' or len(shape) > 8 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('Invalid numeric array layout')
        count = 1
        for n in shape:
            count *= n
        nbytes = count*dtype.itemsize
        if nbytes != item['bytes'] or nbytes > len(data)-offset or type(item['writeable']) is not bool:
            raise ValueError('Invalid numeric array extent')
        array = np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(shape).copy()
        array.flags.writeable = item['writeable']
        arrays.append(array)
        offset += nbytes
    if offset != len(data):
        raise ValueError('Trailing Python snapshot data')
    classes = _classes()

    def decode(x):
        if isinstance(x, list):
            return [decode(v) for v in x]
        if isinstance(x, dict):
            if set(x) == {'$a'}:
                i = x['$a']
                if type(i) is not int or not 0 <= i < len(arrays):
                    raise ValueError('Invalid array reference')
                return arrays[i]
            if set(x) == {'$d'}:
                pairs = [(k, decode(v)) for k, v in x['$d']]
                if any(type(k) is not str for k, _ in pairs) or len(dict(pairs)) != len(pairs):
                    raise ValueError('Invalid dictionary keys')
                return dict(pairs)
            if set(x) == {'$t'}:
                return tuple(decode(v) for v in x['$t'])
            if set(x) == {'$c', 'v'} and x['$c'] in classes:
                cls = classes[x['$c']]
                if set(x['v']) != {f.name for f in fields(cls)}:
                    raise ValueError('Dataclass field inventory mismatch')
                return cls(**{k: decode(v) for k, v in x['v'].items()})
            raise ValueError('Unknown snapshot object tag')
        return x

    return decode(header['tree'])


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def encode_task_history(task):
    """Make the branch's inherited reward history reproducible offline."""
    if set(task.__dict__) != TASK_FIELDS:
        raise ValueError('Unknown task history fields')
    raw = encode_python(task.__dict__)
    compressed = zstd.ZstdCompressor(level=1).compress(raw)
    return {'schema':'branch_task_history_v1', 'raw_bytes':len(raw),
            'sha256':_digest(compressed), 'zstd_base64':base64.b64encode(compressed).decode('ascii')}


def decode_task_history(value, static, initial, goal):
    if value.get('schema') != 'branch_task_history_v1' or type(value.get('raw_bytes')) is not int:
        raise ValueError('Invalid task snapshot schema')
    n = value['raw_bytes']
    if not 0 < n <= MAX_BYTES:
        raise ValueError('Invalid task snapshot size')
    compressed = base64.b64decode(value['zstd_base64'], validate=True)
    if _digest(compressed) != value['sha256'] or zstd.frame_content_size(compressed) != n:
        raise ValueError('Invalid task snapshot checksum/size')
    data = decode_python(zstd.ZstdDecompressor().decompress(compressed, max_output_size=n))
    if (set(data) != TASK_FIELDS or data['last_step'] != 0
            or data['episode_id'] != initial.metadata['episode_id']
            or json_hash(data['goal'].to_dict()) != json_hash(goal)
            or material_reference(data['static']) != material_reference(static)):
        raise ValueError('Task snapshot does not describe this branch/material/goal')
    monitor = TaskMonitor.__new__(TaskMonitor)
    monitor.__dict__.update(data)
    return monitor


@dataclass(frozen=True)
class EnvSnapshot:
    native_compressed: bytes
    python_compressed: bytes
    metadata_json: bytes

    def __post_init__(self):
        if any(type(v) is not bytes for v in (self.native_compressed, self.python_compressed, self.metadata_json)):
            raise TypeError('Snapshots own immutable bytes only')
        m = self.metadata
        if (m.get('schema') != 'environment_snapshot_v1' or m.get('state_abi') != 1
                or m.get('native_compressed_sha256') != _digest(self.native_compressed)
                or m.get('python_compressed_sha256') != _digest(self.python_compressed)
                or any(type(m.get(k)) is not int or not 0 < m[k] <= MAX_BYTES
                       for k in ('native_bytes', 'python_bytes'))):
            raise ValueError('Snapshot metadata, checksum or size mismatch')

    @property
    def metadata(self):
        return json.loads(self.metadata_json)

    @property
    def sha256(self):
        h = hashlib.sha256(self.metadata_json)
        h.update(self.native_compressed)
        h.update(self.python_compressed)
        return h.hexdigest()

    @property
    def size_bytes(self):
        return len(self.metadata_json)+len(self.native_compressed)+len(self.python_compressed)+32

    def _raw(self, key, encoded):
        size = self.metadata[key+'_bytes']
        # Enforce the advertised expansion bound, including for frames that
        # embed a larger content size. Never trust that size before allocating.
        frame_size = zstd.frame_content_size(encoded)
        if frame_size != size:
            raise ValueError('Compressed snapshot frame size mismatch')
        raw = zstd.ZstdDecompressor().decompress(encoded, max_output_size=size)
        if len(raw) != size:
            raise ValueError('Snapshot expansion size mismatch')
        return raw

    def native_bytes(self):
        return self._raw('native', self.native_compressed)

    def python_state(self):
        return decode_python(self._raw('python', self.python_compressed))

    def save(self, path):
        path = Path(path)
        # A concrete snapshot is never silently overwritten.
        with path.open('xb') as f:
            f.write(struct.pack('<8sQQQ', MAGIC, len(self.metadata_json), len(self.native_compressed), len(self.python_compressed)))
            f.write(self.metadata_json)
            f.write(self.native_compressed)
            f.write(self.python_compressed)

    @classmethod
    def load(cls, path, *, dll_sha256, physics_config_sha256, state_abi=1):
        path = Path(path)
        with path.open('rb') as f:
            header = f.read(32)
            if len(header) != 32:
                raise ValueError('Truncated snapshot header')
            magic, m, n, p = struct.unpack('<8sQQQ', header)
            if magic != MAGIC or m > 1024**2 or n > MAX_BYTES or p > MAX_BYTES or m+n+p+32 != path.stat().st_size:
                raise ValueError('Invalid snapshot envelope')
            metadata, native, python = f.read(m), f.read(n), f.read(p)
        result = cls(native, python, metadata)
        expected = {'dll_sha256': dll_sha256, 'state_abi': state_abi, 'physics_config_sha256': physics_config_sha256}
        for key, value in expected.items():
            if result.metadata[key] != value:
                raise ValueError('Snapshot '+key+' mismatch')
        return result


class SnapshotBackend:
    def __init__(self, native):
        self.native = native
        self.functions = {}
        signatures = (('LCS_StateAbiVersion', [], ct.c_int),
            ('LCS_StateSize', [ct.POINTER(ct.c_size_t)], ct.c_int),
            ('LCS_SaveState', [ct.c_void_p, ct.c_size_t], ct.c_int),
            ('LCS_LoadState', [ct.c_void_p, ct.c_size_t], ct.c_int),
            ('LCS_StateLastError', [], ct.c_char_p))
        try:
            for name, args, result in signatures:
                fn = native.dll[name]
                fn.argtypes, fn.restype = args, result
                self.functions[name] = fn
        except AttributeError as error:
            raise RuntimeError('Snapshot requires an explicitly pinned isolated state ABI DLL') from error
        if self.functions['LCS_StateAbiVersion']() != 1:
            raise RuntimeError('Snapshot native ABI mismatch')

    def check(self, name, *args):
        if self.functions[name](*args):
            raise RuntimeError(name+': '+self.functions['LCS_StateLastError']().decode('utf8'))

    def size(self):
        size = ct.c_size_t()
        self.check('LCS_StateSize', ct.byref(size))
        if not 0 < size.value <= MAX_BYTES:
            raise ValueError('Native snapshot size exceeds supported limit')
        return size.value

    def save(self):
        size = self.size()
        buffer = ct.create_string_buffer(size)
        self.check('LCS_SaveState', buffer, size)
        return buffer.raw

    def load(self, data):
        self.check('LCS_LoadState', ct.c_char_p(data), len(data))


def binding(env):
    """Preparation, task, reward and future actions are not physical state."""
    from dataclasses import asdict
    if env.scene_config.scene not in ('wall', 'wall-traction', 'flap'):
        raise ValueError('Snapshot acceptance is scoped to wall/flap scenes')
    scene = asdict(env.scene_config)
    scene.pop('preparation')
    return {'settings': asdict(env.settings), 'native_physics_parameters': env.native_parameters,
            'scene_config': scene, 'material_reference_sha256': material_reference(env.static_arrays()),
            'training_flags': getattr(env.native, 'training_flags', 0),
            'cutting_enabled': env.config.cutting_enabled,
            'full_surface_checks': env.config.full_surface_checks, 'quality_limits': asdict(env.config.limits),
            'outcome_semantics': env.config.outcome_semantics,
            'instrument_collision_checks': env.config.instrument_collision_checks,
            'fast_topology_checks': env.config.fast_topology_checks}


def source_binding():
    here = Path(__file__).parent
    return {p.name: sha256(p) for p in [here/name for name in
        ('snapshot.py', 'env.py', 'native.py', 'quality.py', 'grasp.py', 'tasks.py', 'damage.py',
         'physics.py', 'actions.py', 'vision.py', 'lineage.py', 'geometry.py', 'outcomes.py', 'policies.py', 'instrument_collision.py', 'visual_randomization.py', 'visual_labels.py','pixel_labels.py', 'topology_fast.py', 'topology_hash.py', 'egl_device.py', 'quality_kernels.py', 'distance_kernels.py', 'mesh_kernels.py', 'gpu_topology.py', 'render_scheduling.py', 'gpu_postprocess.py', 'pixel_compose_fast.py')]}


def take_snapshot(env):
    if env.native is None or env.state is None or env.context is None or getattr(env, '_step_in_progress', False):
        raise RuntimeError('Snapshot requires an active environment at a completed step boundary')
    started = time.perf_counter()
    pinned_dll(env.native.dll_path, env.context['dll_sha256'])
    backend = SnapshotBackend(env.native)
    data = {name: getattr(env, name) for name in ENV_FIELDS}
    from .env import State
    data['state'] = State({k:v for k,v in env.state.metadata.items() if k != 'vision'},
                         {k:v for k,v in env.state.arrays.items() if k not in VISION_ARRAYS and not k.startswith('privileged_')}, env.state.topology)
    data.update(rng=env.rng.bit_generator.state,
                native={'tick': env.native.tick, 'neighbors': env.native.neighbors,
                        'topology_revision': env.native.topology_revision,
                        'last_training_metrics': getattr(env.native, 'last_training_metrics', None)},
                quality={name: getattr(env.quality, name) for name in QUALITY_FIELDS},
                grasp={name: getattr(env.grasp, name) for name in GRASP_FIELDS},
                damage=None if env.damage is None else {name: getattr(env.damage, name) for name in DAMAGE_FIELDS},
                task=None if env.task is None else env.task.__dict__)
    if data['task'] is not None and set(data['task']) != TASK_FIELDS:
        raise RuntimeError('Task state inventory changed; snapshot codec review required')
    native = backend.save()
    python = encode_python(data)
    vision = getattr(env, '_snapshot_vision', None)
    compressor = zstd.ZstdCompressor(level=1)
    nc, pc = compressor.compress(native), compressor.compress(python)
    metadata = {'schema': 'environment_snapshot_v1', 'state_abi': 1, 'dll_sha256': env.context['dll_sha256'],
                'physics_config_sha256': json_hash(binding(env)), 'python_source_sha256': source_binding(),
                'native_bytes': len(native), 'python_bytes': len(python),
                'native_compressed_sha256': _digest(nc), 'python_compressed_sha256': _digest(pc),
                'parent_episode_id': env.episode_id, 'branch_from_step': env.step_id,
                'reset_lineage_id': env.context.get('reset_lineage_id', env.episode_id),
                'native_tick': env.native.tick, 'vision_required': vision is not None,
                'vision_manifest_sha256': None if vision is None else json_hash(vision.manifest),
                'snapshot_wall_ms': (time.perf_counter()-started)*1000}
    return EnvSnapshot(nc, pc, json.dumps(metadata, sort_keys=True, allow_nan=False).encode())


def restore_snapshot(env, snapshot, *, branch_label=None, goal=KEEP_TASK):
    from .env import State
    if not isinstance(snapshot, EnvSnapshot):
        raise TypeError('Expected immutable EnvSnapshot')
    if env.native is None or env.context is None or getattr(env, '_step_in_progress', False):
        raise RuntimeError('Restore requires an initialized session at a completed step boundary')
    if branch_label is not None and (type(branch_label) is not str or len(branch_label) > 256):
        raise ValueError('branch_label must be a string of at most 256 characters')
    started = time.perf_counter()
    m = snapshot.metadata
    pinned_dll(env.native.dll_path, m['dll_sha256'])
    if (env.context['dll_sha256'] != m['dll_sha256'] or json_hash(binding(env)) != m['physics_config_sha256']
            or source_binding() != m['python_source_sha256']):
        raise ValueError('Snapshot DLL, physical configuration or Python codec version differs')
    vision = getattr(env, '_snapshot_vision', None)
    if m['vision_required'] and vision is None:
        raise ValueError('Attach synchronous vision before restoring a snapshot that requires it')
    if m['vision_required'] and json_hash(vision.manifest) != m['vision_manifest_sha256']:
        raise ValueError('Snapshot vision configuration differs from the attached renderer')
    data = snapshot.python_state()
    expected = set(ENV_FIELDS) | {'rng', 'native', 'quality', 'grasp', 'damage', 'task'}
    if set(data) != expected or (data['task'] is not None and set(data['task']) != TASK_FIELDS):
        raise ValueError('Python snapshot field inventory mismatch')
    env.config.bounds.process(data['previous_action'], data['previous_action'], env.settings.dt)
    episode_id = uuid4().hex
    state = data['state']
    metadata = {**state.metadata, 'episode_id': episode_id, 'step_id': 0, 't_sim_s': 0.}
    arrays = {k: v for k, v in state.arrays.items() if k not in VISION_ARRAYS and not k.startswith('privileged_')}
    metadata.pop('vision', None)
    if 'events' in metadata:metadata['events'] = []  # restoration is not a new damage/cutting event
    state = State(metadata, arrays, state.topology)
    if goal is KEEP_TASK:
        task = None
        if data['task'] is not None:
            task = TaskMonitor.__new__(TaskMonitor)
            task.__dict__.update(data['task'])
            task.episode_id, task.last_step = episode_id, 0
    elif goal is None:
        task = None
    else:
        task = TaskMonitor(goal, env.static_arrays(), state)
    rng = np.random.default_rng()
    rng.bit_generator.state = data['rng']
    backend = SnapshotBackend(env.native)
    native = snapshot.native_bytes()
    try:
        backend.load(native)
    except Exception:
        # A CUDA copy failure can leave a partially restored native session.
        # Never expose it as a valid branch or permit further stepping.
        env.close()
        raise
    for name in ENV_FIELDS:
        setattr(env, name, data[name])
    for name, value in data['native'].items():
        if name != 'last_training_metrics' or hasattr(env.native, name):
            setattr(env.native, name, value)
    for target, names, saved in ((env.quality, QUALITY_FIELDS, data['quality']),
                                  (env.grasp, GRASP_FIELDS, data['grasp'])):
        for name in names:
            setattr(target, name, saved[name])
    if env.damage is not None:
        if data['damage'] is None:
            env.close()
            raise ValueError('Damage monitor state is missing')
        for name in DAMAGE_FIELDS:
            setattr(env.damage, name, data['damage'][name])
    env.rng, env.state, env.task = rng, state, task
    env.episode_id, env.step_id, env.done = episode_id, 0, False
    # The restored topology, not the pre-restore branch, is the monitor baseline.
    env.quality.previous_neighbors = env.observed.neighbors
    if env.damage is not None:
        env.damage.previous_neighbors = env.observed.neighbors
    context = dict(data['context'])
    context.pop('context_sha256', None)
    context.pop('task_snapshot_initial', None)
    context.update(episode_id=episode_id, parent_episode_id=m['parent_episode_id'],
        reset_lineage_id=m['reset_lineage_id'], branch_from_step=m['branch_from_step'],
        snapshot_sha256=snapshot.sha256, branch_label=branch_label,
        branch_initial_native_tick=m['native_tick'], snapshot_state_abi=1,
        goal=None if task is None else task.goal.to_dict(),
        action_bounds=asdict(env.config.bounds), diagnostic_continue=env.config.diagnostic_continue,
        max_steps=env.config.max_steps, cutting_enabled=env.config.cutting_enabled,
        preparation_reused_from_snapshot=True)
    if task is not None:
        context['task_snapshot_initial'] = encode_task_history(task)
    vision = getattr(env,'_snapshot_vision',None)
    if vision is not None and hasattr(vision,'discard_prepared'):vision.discard_prepared()
    context['restore_wall_time_ms'] = (time.perf_counter()-started)*1000
    if vision is not None:
        try:
            context['vision'] = json.loads(json.dumps(vision.manifest, allow_nan=False))
            env.last_surface = env.native.surface()
            context['restore_vision_ms'] = vision.capture(env, state)
        except Exception:
            env.close()
            raise
    context['context_sha256'] = json_hash(context)
    env.context = context
    return state, context
