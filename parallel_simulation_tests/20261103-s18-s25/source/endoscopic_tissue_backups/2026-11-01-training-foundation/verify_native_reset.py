"""Exercise a single adapter across settled -> cut -> settled resets."""
from dataclasses import replace
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.training.actions import ActionBounds, command_to_action
from endoscopic_tissue.training.env import EnvironmentConfig, NativeEnvironment, SceneConfig
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.scenario import SCISSORS_SCENES, severed_edges


@threadpool_limits.wrap(limits=1, user_api='blas')
def main():
    start = time.perf_counter()
    commands = np.stack([command_to_action(s.command) for s in SCISSORS_SCENES['wall-traction'].stages])
    lo, hi = commands.min(0)-.01, commands.max(0)+.01
    lo[[3, 10]], hi[[3, 10]] = 0, 1
    config = EnvironmentConfig(str(ROOT/'endoscopic_tissue/native/bin-contact-groups31/LiverCudaSim.dll'),
        '4515f14aa15309162c2c54b6370792e128e63a7e8d3054c506da19e3aeb37d2f',
        ActionBounds(tuple(lo), tuple(hi), (10.,)*11), SceneConfig('wall-traction','settled'),
        cutting_enabled=True, max_steps=150)
    results = []
    with NativeEnvironment(config) as env:
        first, first_context = env.reset(10)
        first_positions = first.arrays['positions_m'].copy()
        first_neighbors = first.topology['neighbors'].copy()
        print('first settled', flush=True)
        ready, ready_context = env.reset(11, SceneConfig('wall-traction', 'cut_ready'))
        for i in range(150):
            action = env.previous_action.copy()
            action[10] = min(1., (i+1)/75)
            result = env.step(action)
            if result.terminated or result.truncated:
                results.append(dict(steps=i+1, reason=result.reason))
                break
        cut_count = int(severed_edges(env.grid.neighbors, env.observed.neighbors))
        print('cut episode', cut_count, flush=True)
        third, third_context = env.reset(10, SceneConfig('wall-traction','settled'))
        cut_after_reset = int(severed_edges(env.grid.neighbors, env.observed.neighbors))
        error_mm = np.linalg.norm(third.arrays['positions_m']-first_positions, axis=1)*1000
        checks = dict(distinct_episode_ids=len({c['episode_id'] for c in (first_context,ready_context,third_context)})==3,
            cut_exercised=cut_count>0, topology_recreated=np.array_equal(first_neighbors, third.topology['neighbors']),
            no_cuts_after_reset=cut_after_reset==0, no_grip_after_reset=env.observed.grip_points==0,
            clock_restarted=third.metadata['step_id']==0 and third.metadata['native_tick']==30,
            initial_quality_checked=third.metadata['quality']['checked_metrics_passed'])
        env.config = replace(config, bounds=ActionBounds((0.,)*11, (0.,)*11, (0.,)*11))
        try:
            env.reset(12)
        except ValueError:
            checks['failed_reset_clears_public_state'] = env.state is None and env.context is None and env.done
            checks['failed_reset_preserves_observed_scene'] = (env.last_reset_failure_arrays is not None
                and len(env.last_reset_failure_arrays['nodes_native_cm']) == len(env.grid.positions))
        else:
            checks['failed_reset_rejected'] = False
        payload = dict(checks=checks, passed=all(checks.values()), cut_before_reset=cut_count,
            cut_after_reset=cut_after_reset, positions_repeat_max_mm=float(error_mm.max()),
            positions_repeat_rms_mm=float(np.sqrt(np.mean(error_mm**2))),
            outcomes=results, wall_time_s=time.perf_counter()-start,
            partial_state_note='No claim of bitwise CUDA determinism or complete hidden-history observability')
        atomic_json(Path(__file__).parent/'native-reset-result.json', payload)
        print(payload, flush=True)
        if not payload['passed']:
            raise RuntimeError('Native reset checks failed')


if __name__ == '__main__':
    main()
