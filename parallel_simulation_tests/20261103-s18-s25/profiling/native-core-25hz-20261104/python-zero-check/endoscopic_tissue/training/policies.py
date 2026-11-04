"""Policies consume a declared observation and always return the same 11D action."""
import copy
import importlib
import inspect

import numpy as np

from .provenance import json_hash, sha256


def observation(state, previous_action, *, privileged=False,
                contract="simulated_tool_measurements_v1", visual_tracking=False):
    if contract == "tool_kinematics_v1":
        if privileged:
            raise ValueError("Use the separate privileged_state channel with tool_kinematics_v1")
        return tool_kinematics(state, previous_action, visual_tracking=visual_tracking)
    if contract != "simulated_tool_measurements_v1" or visual_tracking:
        raise ValueError("Unknown or incompatible observation contract")
    value = {'step_id': state.metadata['step_id'], 't_sim_s': state.metadata['t_sim_s'],
             'dt_s': state.metadata['dt_s'], 'tools': copy.deepcopy(state.metadata['tools']),
             'previous_target_si': np.asarray(previous_action).copy(),
             'observation_contract': 'simulated_tool_measurements_v1', 'privileged': False}
    # Anchor count/error and solver phase are diagnostic truth, not sensor inputs.
    for key in ('grip_points', 'grip_error_m', 'scissors_phase'):
        value['tools'].pop(key, None)
    value['tools']['scissors_loads'] = {k:v for k,v in value['tools']['scissors_loads'].items()
                                       if k in ('left_n','right_n')}
    if 'vision' in state.metadata:
        value['vision'] = copy.deepcopy(state.metadata['vision'])
        if state.metadata['vision']['scheduled']:
            value['rgb'] = state.arrays['rgb'].copy()
            value['depth_m'] = state.arrays['depth_rgb_aligned_m'].copy()
            value['depth_valid'] = state.arrays['depth_valid'].copy()
    if privileged:
        value.update(observation_contract='partial_simulator_truth_v1', privileged=True,
                     positions_m=state.arrays['positions_m'].copy(),
                     estimated_velocity_m_s=state.arrays['estimated_velocity_m_s'].copy(),
                     topology=state.topology['neighbors'].copy())
    return value


KINEMATICS_TOOL_FIELDS = ('grasper_gap_m', 'grasper_axis', 'grasper_normal',
                          'scissors_gap_m', 'scissors_axis', 'scissors_normal')


def contract_manifest(contract='simulated_tool_measurements_v1', *, visual_tracking=False):
    if contract not in ('simulated_tool_measurements_v1', 'tool_kinematics_v1'):
        raise ValueError('Unknown observation contract')
    if type(visual_tracking) is not bool or (visual_tracking and contract != 'tool_kinematics_v1'):
        raise ValueError('Visual tracking is an explicit kinematics-only assumption')
    return dict(name=contract, visual_tracking=visual_tracking,
                position_source='slew_limited_policy_target' if contract == 'tool_kinematics_v1' else 'solved_simulator_pose',
                force_sensor_required=contract != 'tool_kinematics_v1',
                privileged_fields_separate=contract == 'tool_kinematics_v1')


def tool_kinematics(state, previous_action, *, visual_tracking=False):
    contract_manifest('tool_kinematics_v1', visual_tracking=visual_tracking)
    tools = state.metadata['tools']
    # The feedback-limited native executed_target_si uses a solved blade position.
    # It MUST NOT replace the policy's slew-limited command in this contract.
    target = np.asarray(state.metadata.get('previous_target_si', previous_action), dtype=float)
    if target.shape != (11,) or not np.isfinite(target).all():
        raise ValueError('A finite 11D slew-limited command is required')
    result = dict(step_id=state.metadata['step_id'], t_sim_s=state.metadata['t_sim_s'],
                  dt_s=state.metadata['dt_s'], tools={k:copy.deepcopy(tools[k]) for k in KINEMATICS_TOOL_FIELDS},
                  previous_target_si=target.copy(), observation_contract='tool_kinematics_v1', privileged=False)
    if visual_tracking:
        result['visually_tracked_tools'] = {k:copy.deepcopy(tools[k]) for k in
            ('grasper_hinge_m', 'scissors_hinge_m', 'scissors_edges_m')}
    if 'vision' in state.metadata:
        source = state.metadata['vision']
        result['vision'] = {k:copy.deepcopy(source[k]) for k in
            ('scheduled','step_id','t_sim_s','dropped','reason','intrinsics_pinhole','world_to_cv') if k in source}
        if source['scheduled']:
            for key in ('rgb','depth_valid'):
                result[key] = state.arrays[key].copy()
            result['depth_m'] = state.arrays['depth_rgb_aligned_m'].copy()
    return result


def privileged_state(state, fields):
    allowed = {'positions_m', 'topology', 'telemetry', 'solved_tools', 'visual_labels'}
    if not isinstance(fields, (list, tuple)) or len(set(fields)) != len(fields) or set(fields)-allowed:
        raise ValueError('Unknown or duplicate privileged field')
    result = {'schema':'privileged_state_v1', 'privileged':True, 'step_id':state.metadata['step_id']}
    for key in fields:
        if key == 'positions_m': result[key] = state.arrays[key].copy()
        elif key == 'topology': result[key] = {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in state.topology.items()}
        elif key == 'solved_tools': result[key] = copy.deepcopy(state.metadata['tools'])
        elif key == 'telemetry': result[key] = {k:copy.deepcopy(state.metadata[k]) for k in
            ('failure_physics','grasp','quality','events','outcome') if k in state.metadata}
        else: result[key] = {k:v.copy() for k,v in state.arrays.items() if k.startswith('privileged_')}
    return result


def recorded_observation(reader, index, *, contract=None, visual_tracking=None):
    from .env import State
    metadata, arrays, topology = reader.state(index)
    declared = reader.context.get('observation', {})
    contract = contract or declared.get('name', 'simulated_tool_measurements_v1')
    tracking = declared.get('visual_tracking', False) if visual_tracking is None else visual_tracking
    if 'previous_target_si' not in metadata:
        raise ValueError('Legacy recording lacks the command needed for exact observation reconstruction')
    return observation(State(metadata, arrays, topology), metadata['previous_target_si'],
                       contract=contract, visual_tracking=tracking)


class Policy:
    def __init__(self, config, initial_action, seed):
        self.config = copy.deepcopy(config)
        config = self.config
        self.initial = np.asarray(initial_action, float).copy()
        self.rng = np.random.default_rng(seed)
        self.kind = config['kind']
        if self.kind not in ('hold', 'smooth_random', 'force_feedback', 'script', 'python'):
            raise ValueError('Unknown policy kind')
        self.observation_contract = config.get('observation_contract', 'simulated_tool_measurements_v1')
        self.visual_tracking = config.get('visual_tracking', False)
        declared = contract_manifest(self.observation_contract, visual_tracking=self.visual_tracking)
        if self.observation_contract == 'tool_kinematics_v1' and (self.kind == 'force_feedback' or config.get('privileged')):
            raise ValueError('Deployable observation cannot be used with a force/privileged policy')
        self.manifest = {'observation': declared, 'implementation': 'tissue_policy_v1', 'configuration': config,
                         'seed': int(seed), 'privileged_observation': bool(config.get('privileged', False))}
        self.phases = self.rng.uniform(0, 2*np.pi, 11)
        self.period = float(config.get('frequency_rad_s', 1.5))
        self.translation = float(config.get('translation_amplitude_m', .0001))
        self.angle = float(config.get('angle_amplitude_rad', np.radians(1)))
        if not np.isfinite((self.period, self.translation, self.angle)).all() or min(self.period, self.translation, self.angle) < 0:
            raise ValueError('Invalid exploration parameters')
        self.external = None
        if self.kind == 'script':
            self.actions = np.asarray(config['actions_si'], float)
            if self.actions.ndim != 2 or self.actions.shape[1] != 11 or not np.isfinite(self.actions).all():
                raise ValueError('Script must contain finite 11D actions')
        if self.kind == 'python':
            module = importlib.import_module(config['module'])
            factory = getattr(module, config['factory'])
            source = inspect.getsourcefile(factory)
            if source is None or sha256(source) != config['source_sha256']:
                raise ValueError('Policy source does not match its pinned SHA-256')
            self.manifest['source_sha256'] = sha256(source)
            if 'checkpoint' in config:
                if sha256(config['checkpoint']) != config['checkpoint_sha256']:
                    raise ValueError('Policy checkpoint hash mismatch')
                self.manifest['checkpoint_sha256'] = config['checkpoint_sha256']
            self.external = factory(config.get('parameters', {}), seed=seed)
        self.manifest['policy_sha256'] = json_hash(self.manifest)

    def act(self, obs):
        if self.external is not None:
            return np.asarray(self.external.act(obs), float)
        if self.kind == 'script':
            if obs['step_id'] >= len(self.actions):
                raise ValueError('Script exhausted; no implicit repetition')
            return self.actions[obs['step_id']].copy()
        action = self.initial.copy()
        if self.kind == 'smooth_random':
            t = obs['t_sim_s']+obs['dt_s']
            amplitude = np.array([self.translation]*3+[0]+[self.translation]*3+[self.angle]*3+[0])
            action += amplitude*(np.sin(t*self.period+self.phases)-np.sin(self.phases))*min(t/.5, 1.)
            if self.config.get('close_scissors', False):
                action[10] = min(1., self.initial[10]+t/float(self.config.get('close_seconds', 2.5)))
        elif self.kind == 'force_feedback':
            action = np.asarray(obs['previous_target_si'], float).copy()
            loads = obs['tools']['jaw_load_n']
            minimum = float(self.config.get('min_force_n', .05))
            if min(loads) < minimum:
                action[3] = min(1., action[3]+.2*obs['dt_s'])
            else:
                action[1] += float(self.config.get('lift_m_s', .0001))*obs['dt_s']
            if self.config.get('close_scissors', False) and min(loads) >= minimum:
                action[10] = min(1., action[10]+.25*obs['dt_s'])
        return action
