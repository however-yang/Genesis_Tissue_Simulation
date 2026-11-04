"""Twenty fresh pairs per fork with fail-fast evidence, never a shortened pass."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--library-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fork', required=True, choices=['regrasp', 'before_contact', 'first_cut', 'squeezing', 'cache'])
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    if (args.fork == 'cache') != (args.cache is not None):
        parser.error('T7 requires one explicitly pinned common cache; T4 cannot use it')
    if args.cache is not None:
        args.cache = args.cache.resolve(strict=True)
    root = args.release.resolve(strict=True)
    sys.path[:0] = [str(root), str(root/'endoscopic_tissue_backups/2026-11-01-s15-s17')]
    from threadpoolctl import threadpool_limits
    import validate_snapshot_distribution as worker
    from endoscopic_tissue.training.data import atomic_json
    from endoscopic_tissue.training.snapshot import source_binding
    from endoscopic_tissue.training.provenance import sha256
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    summary = dict(required_pairs=20, completed_pairs=0, passed=False, full_surface_checks=True,
                   fork=args.fork, default_replaced=False, fail_fast=True,
                   python_source_sha256=source_binding(),
                   dll_sha256=sha256(args.library_dir/'LiverCudaSim.dll'),
                   worker_sha256=sha256(worker.__file__), driver_sha256=sha256(__file__),
                   scope='T4 or T7 only; no T8, video, performance, or physical-domain certification',
                   cache_file_sha256=sha256(args.cache) if args.cache else None)
    original = worker.environment
    current = None
    def environment(**kwargs):
        env = original(dll=str(args.library_dir), **kwargs)
        advance = env.replay_command
        saved = set()
        def step(command):
            result = advance(command)
            if result.after is not None:
                quality = result.after.metadata['quality']
                if ((not quality['checked_metrics_passed'] or not quality['within_declared_local_scope'])
                        and env.episode_id not in saved):
                    saved.add(env.episode_id)
                    folder = current/'first-failures'/f'{len(saved):02}-tick-{env.native.tick}'
                    folder.mkdir(parents=True, exist_ok=False)
                    atomic_json(folder/'metadata.json', result.after.metadata)
                    env.snapshot().save(folder/'state.lcssnap')
                    print('Preserved rejected state', str(folder), flush=True)
            if env.native.tick % 120 == 0:
                print(current.name, 'tick', env.native.tick, flush=True)
            return result
        env.replay_command = step
        return env
    worker.environment = environment
    started = time.time()
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            for repeat in range(20):
                current = args.output/f'{args.fork}-r{repeat:02}'
                current.mkdir(exist_ok=False)
                spec = argparse.Namespace(fork=args.fork, repeat=repeat, capture=False,
                                          cache=args.cache, name=current.name)
                record = worker.paired_run(current, spec)
                results.append(current/'result.json')
                summary['completed_pairs'] = len(results)
                summary['results'] = [str(f) for f in results]
                atomic_json(args.output/'matrix-progress.json', summary)
                print('PAIR', repeat, 'passed', record['passed'], flush=True)
                if not record['passed']:
                    summary['stop_reason'] = 'Physical or mechanism failure; remaining pairs not claimed as tested'
                    return 2
            summary['distribution'] = worker.summarize(results)
            summary['passed'] = summary['distribution']['passed']
            return 0 if summary['passed'] else 2
    except BaseException as error:
        summary['error'] = repr(error)
        raise
    finally:
        worker.environment = original
        summary['wall_s'] = time.time()-started
        atomic_json(args.output/'result.json', summary)


if __name__ == '__main__':
    raise SystemExit(main())
