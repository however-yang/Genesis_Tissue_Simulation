"""Fresh-process adapter for the existing full quality validator; no solver edits."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import signal
import sys
import time


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    out = args.spec.parent
    root = Path(spec['release'])
    if spec.get('cpus'):
        os.sched_setaffinity(0, spec['cpus'])
    sys.path[:0] = [str(root / 'endoscopic_tissue_backups/2026-11-01-s15-s17'), str(root)]
    import numpy as np
    import validate_damage as validator
    from endoscopic_tissue.training import env as entry
    from endoscopic_tissue.training.provenance import sha256
    if sha256(spec['library']) != spec['library_sha256']:
        raise RuntimeError('Pinned native library changed')
    validator.OUT = out
    native_class = entry.TrainingNativeTissue
    env_class = entry.NativeEnvironment
    profile_rows = []
    current_episode = [None]

    class MeasuredNative(native_class):
        def check(self, name, *args):
            record=getattr(self,'api_record',None)
            if record is None:return super().check(name,*args)
            begin=time.perf_counter()
            try:return super().check(name,*args)
            finally:record[name]=record.get(name,0.)+(time.perf_counter()-begin)*1000

        def step(self, command):
            tick = getattr(self, 'tick', 0) + 1
            instrument = tick in spec.get('profile_ticks', [])
            self.api_record={} if instrument else None
            if instrument:
                enable = self.dll.LCS_ProfileSections
                enable.argtypes, enable.restype = (ct.c_int,), None
                read = self.dll.LCS_ProfileSectionsRead
                read.argtypes, read.restype = (ct.c_void_p, ct.c_int), ct.c_int
                enable(1)
            before_revision=getattr(self,'topology_revision',None)
            try:
                return super().step(command)
            finally:
                if instrument:
                    buffer = ct.create_string_buffer(65536)
                    code = read(buffer, len(buffer))
                    enable(0)
                    if code:
                        raise RuntimeError(f'Profile read failed: {code}')
                    sections = dict((k, float(v)) for k, v in
                                    (line.split('=') for line in buffer.value.decode().splitlines()))
                    profile_rows.append({'episode': current_episode[0], 'tick': tick, 'sections_ms': sections, 'api_wall_ms': self.api_record, 'topology_before': before_revision, 'topology_after': getattr(self,'topology_revision',None)})

    class RecordedEnvironment(env_class):
        def replay_command(self, requested):
            if (out.parent / 'STOP').exists():
                raise RuntimeError('Supervisor requested cooperative stop')
            result = super().replay_command(requested)
            if result.after is not None:
                quality = result.after.metadata['quality']
                bad = not quality['checked_metrics_passed'] or not quality['within_declared_local_scope']
                episode = current_episode[0]
                if bad and getattr(self, '_captured_failure_episode', None) != episode:
                    self._captured_failure_episode = episode
                    target = out / episode / ('failure-tick-' + str(self.native.tick))
                    target.mkdir(parents=True, exist_ok=False)
                    surface = self.last_surface
                    if surface is None:
                        surface = self.native.surface()
                    np.savez_compressed(target/'state.npz', nodes_cm=self.observed.nodes,
                        neighbors=self.observed.neighbors, rest_cm=self.grid.positions,
                        inside=self.grid.inside, active=self.grid.active,
                        initial_neighbors=self.grid.neighbors, vertices_cm=surface.vertices,
                        normals=surface.normals, faces=surface.triangles, anchors=surface.anchors,
                        grip_flags=self.observed.grip_flags)
                    write_json(target/'metadata.json', result.after.metadata)
                    write_json(target/'diagnostic.json', {'quality_rules_unchanged': True,
                        'capture_after_original_assessment': True, 'native_tick': self.native.tick,
                        'note': 'Failure capture adds I/O after rejection; this run is diagnostic, not a controlled timing comparison.'})
            if self.native.tick % 30 == 0:
                write_json(out / 'progress.json', {'episode': current_episode[0], 'tick': self.native.tick,
                                                  'time': time.time(), 'pid': os.getpid()})
            return result

        def close(self):
            if self.native is not None and self.observed is not None and current_episode[0] is not None:
                target = out / current_episode[0]
                if target.exists():
                    np.savez_compressed(target / 'final-state.npz', nodes_cm=self.observed.nodes,
                        neighbors=self.observed.neighbors, rest_cm=self.grid.positions,
                        inside=self.grid.inside, active=self.grid.active, grip_flags=self.observed.grip_flags)
            return super().close()

    entry.TrainingNativeTissue = MeasuredNative if spec.get('profile_ticks') else native_class
    entry.NativeEnvironment = RecordedEnvironment
    write_json(out / 'ready.json', {'pid': os.getpid(), 'time': time.time(),
        'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'affinity': sorted(os.sched_getaffinity(0))})
    deadline = time.monotonic() + 180
    while not (out.parent / 'GO').exists():
        if time.monotonic() > deadline or (out.parent / 'STOP').exists():
            raise RuntimeError('Start barrier expired or stopped')
        time.sleep(.05)
    started = time.time()
    outcomes = []
    try:
        for ep in range(spec.get('episodes', 1)):
            name = f'episode-{ep:03d}'
            current_episode[0] = name
            sys.argv = ['validate_damage', '--name', name, '--dll', str(Path(spec['library']).parent),
                        '--scene', spec.get('scene', 'wall-traction'), '--stop-after', spec.get('stop_after', 'hold'),
                        '--seed', str(spec.get('seed', 0) + ep)]
            if spec.get('geometry'):
                sys.argv += ['--geometry', spec['geometry']]
            if spec.get('contact') is not None:
                sys.argv += ['--contact', str(spec['contact'])]
            code = 0
            try:
                validator.main()
            except SystemExit as exc:
                code = int(exc.code or 0)
            result = json.loads((out / name / 'result.json').read_text())
            outcomes.append({'episode': name, 'exit_code': code, 'passed': result['passed'],
                             'full_script': result['full_script']})
            if code not in (0, 2):
                raise RuntimeError(f'Validator exit {code}')
    finally:
        write_json(out / 'worker-result.json', {'started': started, 'finished': time.time(),
            'outcomes': outcomes, 'completed': len(outcomes) == spec.get('episodes', 1),
            'profile_instrumented': bool(spec.get('profile_ticks')), 'pid': os.getpid()})
        write_json(out / 'profile.json', profile_rows)


if __name__ == '__main__':
    main()
