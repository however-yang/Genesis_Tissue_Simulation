"""Instrument the unchanged S16 worker; diagnostic timings are not benchmarks."""
from dataclasses import asdict
import argparse
import ctypes as ct
import faulthandler
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'endoscopic_tissue_backups/2026-11-01-s15-s17'
sys.path[:0] = [str(OLD), str(ROOT)]
import validate_damage as worker

diagnostics = argparse.ArgumentParser(add_help=False)
diagnostics.add_argument('--trace-section-from', type=int, default=700)
diagnostics.add_argument('--checkpoint-from', type=int, default=400)
diagnostics.add_argument('--stop-numerical', action='store_true')
options, forwarded = diagnostics.parse_known_args()
sys.argv = [sys.argv[0], *forwarded]
name = sys.argv[sys.argv.index('--name') + 1]
OUT = ROOT / 'endoscopic_tissue/outputs/acceptance-fixes-20261102' / name
OUT.mkdir(parents=True, exist_ok=False)
worker.OUT = OUT.parent
# The existing worker creates its output itself after this diagnostic prelude.
trace_dir = OUT.with_name(OUT.name + '-trace')
OUT.rename(trace_dir)
events = (trace_dir / 'events.jsonl').open('w', encoding='utf8', buffering=1)
stacks = (trace_dir / 'python-stacks.log').open('w', encoding='utf8')
faulthandler.enable(stacks)
faulthandler.dump_traceback_later(60, repeat=True, file=stacks)


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def event(kind, **values):
    events.write(json.dumps({'unix': time.time(), 'kind': kind, **values},
                            default=scalar, allow_nan=False) + '\n')


base = worker.entry.TrainingNativeTissue


class TracedNative(base):
    def check(self, api, *args):
        tick = getattr(self, 'tick', -1)
        if api == 'LCS_Step' and tick == options.trace_section_from and 'section' in name:
            profile = self.dll['LCS_ProfileSections']
            profile.argtypes, profile.restype = [ct.c_int], None
            profile(1)
            event('section_trace_enabled', tick=tick)
        event('native_enter', tick=tick, api=api)
        result = super().check(api, *args)
        event('native_exit', tick=tick, api=api)
        return result


worker.entry.TrainingNativeTissue = TracedNative
original_replay = worker.entry.NativeEnvironment.replay_command


def traced_replay(self, command):
    event('step_enter', tick=self.native.tick, command=asdict(command))
    result = original_replay(self, command)
    event('step_exit', tick=self.native.tick, reason=result.reason,
          metadata=result.after.metadata if result.after else None)
    # Keep the last completed state on disk even when the following call stalls.
    # This is diagnostic-only overhead; never compare its FPS with a normal run.
    numerical = result.after and any(x in result.after.metadata['quality']['numerical_failure_reasons']
                                    for x in ('workspace_displacement_exceeded', 'estimated_speed_limit_exceeded',
                                              'invalid_material_tetrahedra', 'non_finite_state'))
    if self.native.tick >= options.checkpoint_from:
        target = trace_dir / ('first-numerical-failure.npz' if numerical and options.stop_numerical else 'last-completed.npz')
        temp = trace_dir / 'last-completed.tmp'
        with temp.open('wb') as f:
            np.savez(f, tick=self.native.tick, nodes=self.observed.nodes,
                     neighbors=self.observed.neighbors, grip_flags=self.observed.grip_flags,
                     rest=self.grid.positions, material=self.grid.inside, active=self.grid.active,
                     surface_vertices=self.last_surface.vertices,
                     surface_triangles=self.last_surface.triangles)
        temp.replace(target)
    if numerical and options.stop_numerical:
        raise RuntimeError('Diagnostic stopped at first numerical quality failure; state preserved')
    return result


worker.entry.NativeEnvironment.replay_command = traced_replay
event('worker_start', argv=sys.argv)
try:
    worker.main()
finally:
    event('worker_exit')
    faulthandler.cancel_dump_traceback_later()
    events.close()
    stacks.close()
