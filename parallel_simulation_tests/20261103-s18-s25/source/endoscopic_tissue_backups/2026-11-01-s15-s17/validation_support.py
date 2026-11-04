"""Shared evidence helpers. These never change a solver or acceptance limit."""
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from endoscopic_tissue.scenario import interpolate, stage_start
from endoscopic_tissue.training.actions import ActionBounds, command_to_action
from endoscopic_tissue.training.env import EnvironmentConfig, NativeEnvironment, SceneConfig
from endoscopic_tissue.training.geometry import scene_for_config
from endoscopic_tissue.training.provenance import sha256


def environment(*, dll='bin-s15-training-snapshot1', scene='wall-traction',
                preparation='settled', physics=None, telemetry=False, geometry=None, contact=None):
    config = SceneConfig(scene, preparation, contact_model=contact, physics=physics, geometry=geometry)
    scenario = scene_for_config(config)
    center = command_to_action(scenario.stages[0].command)
    lo, hi = center-.2, center+.2
    lo[[3, 10]], hi[[3, 10]] = 0., 1.
    path = ROOT/'endoscopic_tissue/native'/dll/'LiverCudaSim.dll'
    return NativeEnvironment(EnvironmentConfig(str(path), sha256(path),
        ActionBounds(tuple(lo), tuple(hi), (10.,)*11), config,
        cutting_enabled=True, max_steps=100000, full_surface_checks=True,
        diagnostic_continue=True, failure_telemetry=telemetry))


def command_sequence(stages, requested):
    """Freeze requested commands for both arms of a comparison.

    Only pre-withdrawal schedules are accepted: withdrawal starts at the solved
    pose and must be generated on its own trajectory, never the paired one.
    """
    result = []
    for stage in stages:
        if stage.name.startswith('withdraw'):
            raise ValueError('A frozen request sequence cannot substitute a solved withdrawal pose')
        start = requested
        for i in range(stage.steps):
            requested = interpolate(start, stage.command, (i+1)/(stage.steps*stage.ramp))
            result.append((stage.name, i+1, requested))
    return result


def row(env, result, label, index):
    if result.after is None:
        raise RuntimeError('No accepted observation: '+str(result.reason))
    meta = result.after.metadata
    return {'stage': label, 'stage_step': index, 'native_tick': env.native.tick,
            'quality': meta['quality'], 'grasp': meta['grasp'],
            'scissors': asdict(env.observed.scissors.loads),
            'failure_physics': meta.get('failure_physics'),
            'timings_ms': result.timings_ms, 'reason': result.reason}


def quality_summary(rows):
    bad = [r for r in rows if not r['quality']['checked_metrics_passed']]
    scope = [r for r in rows if not r['quality']['within_declared_local_scope']]
    return {'steps': len(rows), 'quality_failure_steps': len(bad), 'scope_failure_steps': len(scope),
            'failure_reasons': sorted({s for r in bad for s in r['quality']['numerical_failure_reasons']}),
            'scope_reasons': sorted({s for r in scope for s in r['quality']['scope_failure_reasons']}),
            'healed_edges': sum(r['quality']['metrics']['healed_edges'] for r in rows)}


class WallCutMeasure:
    """Reproducibility measure in material coordinates, not a straight-cut law.

    A column is touched when an actual vertical material edge is severed, or
    the native cut fraction becomes positive. No geometry is modified. Report
    adjacency-only edge counts as well, so a fractional mark cannot fake T6.
    Deviation is from the fixed reference plane in rest coordinates; it is a
    repeatability statistic, not a claim about the deformed cutting edge.
    """
    def __init__(self, env):
        grid, scene = env.grid, env.scene
        initial = grid.neighbors
        material = np.asarray(grid.inside, bool)
        rest = grid.positions*10  # native cm -> mm
        target = initial[:, 1::2]
        self.material_edges = (target >= 0) & material[:, None] & material[np.maximum(target, 0)]
        low, high = np.array(scene.tab.low)*10, np.array(scene.tab.high)*10
        self.plane_slot = 2*scene.plane_axis+1
        endpoint = initial[:, self.plane_slot]
        mid = (rest+rest[np.maximum(endpoint, 0)])*.5
        selected = ((endpoint >= 0) & material & material[np.maximum(endpoint, 0)] &
                    (mid[:, 0] > low[0]+.01) & (mid[:, 0] <= high[0]+.01) &
                    (mid[:, 1] > low[1]+.01) & (mid[:, 1] < high[1]-.01) &
                    (mid[:, 2] >= low[2]-.01) & (mid[:, 2] <= high[2]+.01))
        self.columns = [np.flatnonzero(selected & np.isclose(mid[:, 0], x, atol=1e-3))
                        for x in sorted(np.unique(np.round(mid[selected, 0], 3)), reverse=True)]
        self.mid_normal = mid[:, scene.plane_axis]
        self.plane_mm = scene.plane*10
        xs = np.sort(np.unique(np.round(rest[:, 0], 3)))
        self.spacing_mm = float(np.min(np.diff(xs)))
        self.max_length_mm = float(high[0]-low[0])
        self.initial = initial

    def sample(self, env):
        nbr = env.observed.neighbors
        fractions = env.state.topology['cut_fractions']
        broken = (self.initial[:, 1::2] >= 0) & (nbr[:, 1::2] < 0)
        material_broken = broken & self.material_edges
        cut = (nbr[:, self.plane_slot] < 0) | (fractions[:, self.plane_slot] > 0)
        heights, flags = [], []
        for ids in self.columns:
            ids = ids[cut[ids]]
            flags.append(bool(len(ids)))
            heights.append(float(self.mid_normal[ids].mean()) if len(ids) else None)
        count = 0
        for flag in flags:
            if not flag:
                break
            count += 1
        return {'material_broken_edges': int(material_broken.sum()), 'column_flags': flags,
                'column_heights_rest_mm': heights,
                'contiguous_length_mm': min(count*self.spacing_mm, self.max_length_mm)}

    def outcome(self, before, after, initiation):
        new = [i for i, (a, b) in enumerate(zip(before['column_flags'], after['column_flags'])) if b and not a]
        deviations = [abs(after['column_heights_rest_mm'][i]-self.plane_mm) for i in new]
        return {'delta_length_mm': after['contiguous_length_mm']-before['contiguous_length_mm'],
                'first_new_material_cut_step': initiation,
                'new_material_edges': after['material_broken_edges']-before['material_broken_edges'],
                'max_deviation_rest_mm': max(deviations, default=0.), 'new_columns': len(new),
                'measurement': 'material-coordinate column repeatability; no forced straightening'}


def replay(env, sequence, *, sink=None, before_step=None):
    measure = WallCutMeasure(env)
    beginning = measure.sample(env)
    previous = beginning
    initiation = None
    rows = []
    stages = {}
    for number, (label, index, command) in enumerate(sequence, 1):
        if before_step:
            before_step(label)
        result = env.replay_command(command)
        r = row(env, result, label, index)
        current = measure.sample(env)
        new_edges = current['material_broken_edges']-previous['material_broken_edges']
        r['new_material_cut_edges'] = new_edges
        r['cut_columns'] = current
        rows.append(r)
        if label not in stages:
            stages[label] = {'before': previous, 'first_cut': None}
        if new_edges:
            if initiation is None:
                initiation = number
            if stages[label]['first_cut'] is None:
                stages[label]['first_cut'] = index
        stages[label]['after'] = current
        if sink:
            sink(env, r)
        previous = current
    outcomes = {name: measure.outcome(s['before'], s['after'], s['first_cut'])
                for name, s in stages.items() if name.startswith('incise')}
    return {'outcome': measure.outcome(beginning, previous, initiation), 'snips': outcomes,
            'quality': quality_summary(rows), 'rows': rows}


def exact_values(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact_values(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact_values(x, y)
    else:
        assert a == b, (a, b)
