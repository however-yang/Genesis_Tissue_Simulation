"""Material-coordinate task semantics, with identical online/offline evaluation."""
from dataclasses import asdict, dataclass, replace
import hashlib
import copy

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class TaskGoal:
    kind: str
    target_edges: tuple = ()
    region_nodes: tuple = ()
    hold_nodes: tuple = ()
    displacement_m: tuple = (0., 0., 0.)
    position_tolerance_m: float = .0005
    required_progress: float = 1.
    required_hold_s: float = .5
    min_retained_fraction: float = .5
    max_slip_m: float = .001
    release_gap_m: float = .002
    release_clearance_m: float = .001
    max_outside_cut_edges: int = 0
    fail_on_grasp_loss: bool = False
    weights: dict | None = None
    label: str = 'explicit_experiment_goal'
    material_reference_sha256: str | None = None

    def __post_init__(self):
        if self.kind not in ('incision', 'excision', 'hold', 'release'):
            raise ValueError('Unknown task kind')
        def integer_ids(values):
            values = tuple(values)
            if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in values):
                raise ValueError('Material IDs must be integers; fractional and boolean IDs are invalid')
            return tuple(int(i) for i in values)
        edges = tuple(self.target_edges)
        if any(len(edge) != 2 for edge in edges):
            raise ValueError('Target edges must be pairs of material IDs')
        object.__setattr__(self, 'target_edges', tuple(integer_ids(edge) for edge in edges))
        object.__setattr__(self, 'region_nodes', integer_ids(self.region_nodes))
        object.__setattr__(self, 'hold_nodes', integer_ids(self.hold_nodes))
        positive = (self.position_tolerance_m, self.required_progress, self.required_hold_s,
                    self.release_gap_m, self.release_clearance_m)
        if not np.isfinite(positive).all() or min(positive) <= 0 or self.required_progress > 1:
            raise ValueError('Task thresholds must be finite, positive and explicit')
        if (not 0 <= self.min_retained_fraction <= 1 or not np.isfinite(self.max_slip_m)
                or self.max_slip_m < 0 or not isinstance(self.max_outside_cut_edges, int)
                or self.max_outside_cut_edges < 0):
            raise ValueError('Invalid grasp/damage thresholds')
        if len(self.displacement_m) != 3 or not np.isfinite(self.displacement_m).all():
            raise ValueError('Expected a finite 3D displacement')
        if self.kind in ('incision', 'excision') and not self.target_edges:
            raise ValueError('Cut goal needs explicit material edges')
        if self.kind in ('excision', 'release') and not self.region_nodes:
            raise ValueError('Excision/release needs a material region')
        if self.kind == 'hold' and not self.hold_nodes:
            raise ValueError('Hold goal needs explicit material witness nodes')
        keys = {'progress', 'outside_damage', 'grasp', 'slip_m', 'time_s', 'action'}
        if self.weights is not None and (set(self.weights) != keys or
                                         not np.isfinite(list(self.weights.values())).all()):
            raise ValueError('Supply all six finite reward weights; no implicit calibrated weights')

    def to_dict(self):
        return asdict(self)


def graph_labels(neighbors, mask):
    ids = np.flatnonzero(mask)
    index = np.full(len(mask), -1, np.int32)
    index[ids] = np.arange(len(ids))
    owner, slot = np.where(neighbors[ids] >= 0)
    target = neighbors[ids[owner], slot]
    keep = mask[target]
    matrix = coo_matrix((np.ones(np.count_nonzero(keep), np.uint8),
                         (owner[keep], index[target[keep]])), shape=(len(ids), len(ids)))
    _, compact = connected_components(matrix.tocsr(), directed=False)
    labels = np.full(len(mask), -1, np.int32)
    labels[ids] = compact
    return labels


def material_reference(static):
    digest = hashlib.sha256()
    for key in ('rest_positions_m','initial_neighbors','material_mask','active_mask'):
        array = np.ascontiguousarray(static[key])
        digest.update(key.encode())
        digest.update(str((array.dtype.str,array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class TaskMonitor:
    def __init__(self, goal, static, initial):
        self.goal = TaskGoal(**(asdict(goal) if isinstance(goal, TaskGoal) else copy.deepcopy(goal)))
        digest = material_reference(static)
        if self.goal.material_reference_sha256 not in (None,digest):
            raise ValueError('Goal material reference differs from this scene/grid')
        self.goal = replace(self.goal,material_reference_sha256=digest)
        self.static = static
        self.material = np.asarray(static['material_mask'], bool)
        self.active = np.asarray(static['active_mask'], bool)
        n = len(self.material)
        original = np.asarray(static['initial_neighbors'])
        pairs = sorted(set(tuple(sorted(map(int, edge))) for edge in self.goal.target_edges))
        for a, b in pairs:
            if not (0 <= a < b < n and self.material[a] and self.material[b]
                    and b in original[a] and a in original[b]):
                raise ValueError('Target must be an original bidirectional material edge')
        self.targets = np.asarray(pairs, np.int32).reshape(-1, 2)
        for ids in (self.goal.region_nodes, self.goal.hold_nodes):
            if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= n or not self.material[i] for i in ids):
                raise ValueError('Goal IDs must refer to material nodes')
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate goal nodes')
        self.region = np.zeros(n, bool)
        self.region[list(self.goal.region_nodes)] = True
        if self.region.any() and not np.any(self.material & ~self.region):
            raise ValueError('Region must leave body material outside it')
        self.hold = np.asarray(self.goal.hold_nodes, np.int32)
        self.start_positions = initial.arrays['positions_m'].copy()
        owners, slots = np.where(original[:, 1::2] >= 0)
        targets = original[owners, 1+slots*2]
        keep = self.material[owners] & self.material[targets]
        self.edge_a, self.edge_b = owners[keep], targets[keep]
        self.edge_slot = 1+slots[keep]*2
        self.initial_neighbors = initial.topology['neighbors']
        self.initial_alive = self.initial_neighbors[self.edge_a, self.edge_slot] == self.edge_b
        target_keys = {a*n+b for a, b in pairs}
        self.goal_edge_mask = np.array([min(a,b)*n+max(a,b) in target_keys
                                       for a,b in zip(self.edge_a, self.edge_b)], bool)
        if len(self.targets) and np.any(~self._target_alive(self.initial_neighbors)):
            raise ValueError('Target includes already cut edges; goal must be bound before cutting')
        self.last_revision = None
        self.topology_metrics = None
        self.previous_progress = 0.
        self.previous_outside = 0
        self.dwell_s = 0.
        self.last_step = initial.metadata['step_id']
        self.episode_id = initial.metadata['episode_id']

    def _target_alive(self, neighbors):
        if not len(self.targets):
            return np.empty(0, bool)
        a,b = self.targets.T
        return np.any(neighbors[a] == b[:, None], axis=1) | np.any(neighbors[b] == a[:, None], axis=1)

    def _topology(self, state):
        if state.topology['revision'] == self.last_revision:
            return self.topology_metrics
        neighbors = state.topology['neighbors']
        progress = float(np.mean(~self._target_alive(neighbors))) if len(self.targets) else 0.
        removed = self.initial_alive & (neighbors[self.edge_a, self.edge_slot] != self.edge_b)
        outside = int(np.count_nonzero(removed & ~self.goal_edge_mask))
        metrics = {'cut_progress': progress, 'outside_material_cut_edges': outside,
                   'material_connected_to_body': None, 'support_connected_to_body': None}
        if self.region.any():
            body = self.material & ~self.region
            for name, mask in (('material', self.material), ('support', self.active)):
                labels = graph_labels(neighbors, mask)
                connected = np.intersect1d(labels[self.region], labels[body])
                metrics[name+'_connected_to_body'] = bool(np.any(connected >= 0))
        self.last_revision, self.topology_metrics = state.topology['revision'], metrics
        return metrics

    def evaluate(self, before, after, action, bounds):
        if (before.metadata['episode_id'] != self.episode_id or before.metadata['step_id'] != self.last_step
                or after.metadata['episode_id'] != self.episode_id or after.metadata['step_id'] != self.last_step+1):
            raise ValueError('Task history must be adjacent and remain in one episode')
        self.last_step += 1
        g = self.goal
        metrics = dict(self._topology(after))
        quality = after.metadata['quality']
        from .outcomes import state_valid
        physical_valid = state_valid(after.metadata)
        if not physical_valid:
            self.dwell_s = 0.
            return {'schema':'material_task_v1', 'goal':g.label, 'kind':g.kind,
                    'success':False, 'failure':None, 'physical_valid_for_task':False,
                    'metrics':metrics, 'reward_components':None, 'reward_weights':g.weights,
                    'reward':None, 'reward_valid':False, 'bootstrap_allowed':False}
        grasp = after.metadata['grasp']
        hold_ok = (grasp['geometric_loaded_hold'] and
                   grasp['witness_retained_fraction'] >= g.min_retained_fraction and
                   grasp['witness_tangential_slip_m'] <= g.max_slip_m)
        if len(self.hold):
            errors = after.arrays['positions_m'][self.hold]-self.start_positions[self.hold]-g.displacement_m
            metrics['hold_rms_target_error_m'] = float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))
            hold_ok &= metrics['hold_rms_target_error_m'] <= g.position_tolerance_m
        if g.kind == 'incision':
            achieved = metrics['cut_progress'] >= g.required_progress
        elif g.kind == 'excision':
            achieved = (metrics['cut_progress'] >= g.required_progress and
                        not metrics['material_connected_to_body'] and not metrics['support_connected_to_body'])
        elif g.kind == 'hold':
            achieved = hold_ok
        else:
            region = after.arrays['positions_m'][self.region]
            hinge = np.asarray(after.metadata['tools']['grasper_hinge_m'])
            metrics['region_min_distance_to_hinge_m'] = float(np.linalg.norm(region-hinge, axis=1).min())
            # This geometric release criterion is explicit; collision quality is assessed separately.
            achieved = (not metrics['material_connected_to_body'] and not metrics['support_connected_to_body']
                        and after.metadata['tools']['grasper_gap_m'] >= g.release_gap_m
                        and not grasp['bilateral_load'] and grasp['anchor_points'] == 0
                        and grasp['between_pads_material_nodes'] == 0
                        and metrics['region_min_distance_to_hinge_m'] >= g.release_clearance_m)
        damage = after.metadata.get('outcome', {}).get('terminal_reason') or ''
        failure = damage if damage.startswith('damage:') else None
        if failure is not None:
            achieved = False
        elif metrics['outside_material_cut_edges'] > g.max_outside_cut_edges:
            failure = 'outside_target_material_damage'
        elif g.fail_on_grasp_loss and g.kind != 'release' and not hold_ok:
            failure = 'grasp_lost'
        dt = after.metadata['dt_s']
        self.dwell_s = self.dwell_s+dt if achieved and physical_valid and failure is None else 0.
        success = bool(self.dwell_s+1e-12 >= g.required_hold_s)
        progress = metrics['cut_progress'] if g.kind in ('incision', 'excision') else min(self.dwell_s/g.required_hold_s, 1.)
        span = np.asarray(bounds.upper)-np.asarray(bounds.lower)
        executed = np.asarray(action['executed_target_si'])
        previous_target = np.asarray(action['previous_target_si'])
        normalized = np.divide(executed-previous_target, span, out=np.zeros(11), where=span > 0)
        components = {'progress': progress-self.previous_progress,
                      'outside_damage': -float(metrics['outside_material_cut_edges']-self.previous_outside),
                      'grasp': float(hold_ok), 'slip_m': -grasp['witness_tangential_slip_m'],
                      'time_s': -dt, 'action': -float(np.mean(normalized**2))}
        self.previous_progress, self.previous_outside = progress, metrics['outside_material_cut_edges']
        reward = sum(g.weights[k]*v for k,v in components.items()) if g.weights is not None and physical_valid else None
        metrics.update(hold_condition=bool(hold_ok), dwell_s=self.dwell_s)
        return {'schema': 'material_task_v1', 'goal': g.label, 'kind': g.kind,
                'success': success, 'failure': failure if physical_valid else None,
                'physical_valid_for_task': bool(physical_valid), 'metrics': metrics,
                'reward_components': components, 'reward_weights': g.weights, 'reward': reward,
                'reward_valid': reward is not None and physical_valid,
                'bootstrap_allowed': bool(physical_valid and not success and failure is None)}


def recompute_episode(reader):
    """Recompute from recorded states/actions and compare against every online result."""
    from .actions import ActionBounds
    from .data import load_arrays
    from .env import State
    goal = reader.context.get('goal')
    if goal is None:
        raise ValueError('Episode has no task goal')
    static = load_arrays(reader.directory/'static.npz.zst')
    def get_state(i):
        metadata, arrays, topology = reader.state(i)
        return State(metadata, arrays, {**topology, 'revision': metadata['topology_revision']})
    before = get_state(0)
    if reader.context.get('task_snapshot_initial') is not None:
        from .snapshot import decode_task_history
        monitor = decode_task_history(reader.context['task_snapshot_initial'],static,before,goal)
    else:
        monitor = TaskMonitor(goal, static, before)
    bounds = ActionBounds(**reader.context['action_bounds'])
    rows = []
    for record in reader.transitions:
        if record['after'] is None:
            rows.append({'step': record['step_id'], 'raw_failure': True, 'reward': None})
            break
        after = get_state(record['after'])
        task = monitor.evaluate(before, after, record['action'], bounds)
        if task != record.get('task'):
            raise ValueError('Online/offline task mismatch at '+str(record['step_id']))
        rows.append(task)
        before = after
    return rows
