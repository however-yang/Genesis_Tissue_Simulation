# A100 parallel soft-tissue simulation pilot

All code and artifacts in this directory are isolated from the deployed release.
Read `REPORT.md` for measured conclusions, `DESIGN.md` for architecture and limitations,
`performance.md` for raw run-level comparisons, and `runs/` for original evidence.

The pinned scene is the existing 1,440-step wall-traction grasper/scissors script,
with the existing material parameters, three substeps, and full quality checks.
It controls instrument end effectors, not full articulated dual robot arms.

## Reproduce one complete benchmark

Use a **new** `--name` each time. The UUID below is the card used in this study;
the supervisor rechecks availability and refuses an occupied card.

```bash
cd /srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-a100
/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python -B -u benchmark.py \
  --name my-2-env-check --workers 2 --episodes 1 \
  --gpu GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa \
  --cpus 24,4,25,22,8,35,31,14
```

`--workers` supports up to 64 as a scheduling input, **not a promise of capacity**.
Do not jump to a large count: read the measured memory/CPU limits and use the
validated count from `REPORT.md`. Each worker is pinned to one CPU from the list;
counts above the list length share CPU cores. No environment parameter is relaxed.

- `--episodes 2`: independently close/recreate each worker's environment between episodes.
- `--scenes wall-traction,flap`: alternate the two scene types; validate separately before using the result.
- `--stop-after settle`: short initialization/reset probe; never a full cutting acceptance.
- `--profile --stop-after incise_1`: instrumented native section diagnostic, excluded from normal throughput comparisons.
- Startup admission reserves 2,300 MiB per worker (based on the measured wall scene plus headroom); this is an admission estimate, not a measured maximum.
- GPU ceiling: 60,000 MiB; worker RSS ceiling: 48 GiB; host available memory must stay above 16 GiB.
- If another GPU process appears, the run stops only its own children and preserves evidence.
- No driver, toolkit, dependency, network, compute-mode or MIG changes are performed.

## MPS comparison

`mps_benchmark.py --name my-mps-2 --workers 2` uses a private user controller and
its own pipe/log directory, then requests shutdown in `finally`. It refuses to
start if another MPS controller exists. It does not attach to a system daemon.
Use the project Python executable above. The wrapper preserves the selected
GPU UUID and records controller logs; inspect them to confirm MPS actually ran.

A measured 16-worker / 16-CPU configuration can be reproduced with a short new
name (MPS socket paths have a strict length limit):

```bash
/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python -B -u mps_benchmark.py \
  --name check16 --workers 16 \
  --cpus 0,2,5,9,11,13,15,17,20,21,28,30,32,36,37,6
```

These CPU IDs were selected from this server; inspect current load before reuse.
The default MPS CPU list contains eight cores, so 16 workers otherwise share them.

## Asynchronous policy integration

`parallel_env.ParallelNativePool` accepts one JSON-compatible EnvironmentConfig
specification per worker (see `test_pool.py` for a complete executable example).
It exposes:

```python
pool.send(worker_id, "reset", seed=42)
responses = pool.recv_ready(timeout=120)
pool.send(worker_id, "step", action=eleven_dimensional_target)
responses = pool.recv_ready(timeout=120)
```

Only one request may be outstanding for each worker. Use `episode_id`, `step_id`,
and the returned request ID to match actions and observations. `recv_ready`
returns whichever workers have finished; it does not wait for the entire batch.
A reset affects only the specified worker. Always use a context manager or call
`close()`. Each worker owns its own EpisodeWriter and immutable output directories.

The default response contains compact simulated tool observations and quality
metadata. Full state/topology arrays remain in worker-local recorded episodes.
A learner must still respect invalid/terminated transitions, explicit rewards,
`bootstrap_allowed`, policy versioning and reset-lineage splits. Run `mps_benchmark.py --name poolcheck --workers 16 --pool-test` for the short
heterogeneous-material integration check using the project Python. No RL optimizer,
trained policy, camera rendering throughput, or approved training domain is
claimed by this pilot. Material variants require their own physical validation.

The environment pool currently permits at most sixteen workers. It preflights
all explicitly selected GPUs and monitors resources while requests are submitted
or awaited. The standalone benchmark performs continuous monitoring. Do not leave
a live pool unattended outside its caller's control loop.

## Analysis

```bash
/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python -B analyze.py
/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python -B correctness_report.py
/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python -B profile_report.py
/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python -B integration_summary.py
/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python -B plot_results.py
/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python -B make_report.py
```

`aggregate_steps_s` divides summed steps by the whole batch wall time, including
reset, original checks and output finalization, excluding imports before the start
barrier. `valid_episodes_per_hour` is a projection from this batch and counts only
full passed episodes. Incomplete, interrupted, profiled and smoke runs are not
performance acceptance. GPU memory peaks are sampled every approximately 2 seconds.
