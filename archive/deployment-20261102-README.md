# Genesis isolated execution deployment

Windows remains the code-authoring and Codex host. This server runs ordinary build/test processes only.

- Commit: 0643e10991cf61987250746da4171801c81e742b (dev)
- Git receiver: /srv/disk2/hryang/genesis-exec/repo.git
- Detached execution checkout: /srv/disk2/hryang/genesis-exec/releases/0643e10991cf
- Conda prefix: /srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1
- Native builds: builds/0643e10991cf-contact-groups31-sm80/lib and builds/0643e10991cf-training-snapshot1-sm80/lib
- Extra local sources: resources/local-audit-20261102/untracked-source-reference.tar.gz (reference only; do not overlay)
- Logs, version locks and machine-readable outcomes: logs/

Environment entry after SSH (no shell profile edits required):

```bash
source /home/hryang/miniconda3/etc/profile.d/conda.sh
conda activate /srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/srv/disk2/hryang/genesis-exec/releases/0643e10991cf
cd /srv/disk2/hryang/genesis-exec/releases/0643e10991cf
```

No GPU simulation, rendering validation or training is authorized in this setup stage. Build success and CPU tests do not establish physical fidelity or frame rate. See logs/verification-r1-status.json for actual completion, never infer success from this README.

CUDA Toolkit 12.5.1 is private to the prefix; PyTorch 2.8.0 uses its cu128 runtime (cu126 conflicted with gs-madrona >=12.8 requirements). Global drivers, /usr/local/cuda and other environments are not modified. Native compilation explicitly uses /usr/bin/g++-12 and sm_80. Windows DLLs and snapshots must not be loaded as Linux artifacts.

Future synchronization: inspect local tracked/untracked changes; explicitly commit selected source to dev; push that exact commit through SSH into repo.git; create a NEW detached release directory named by commit. Never git clean/reset hard, mutate a running release, or auto-overlay dirty files. Additional assets require a manifest, explicit allowlist and SHA256 verification. Download experiment results into a new local outputs folder only.

Future execution: record commit, library hash, environment lock and GPU UUID; wait until selected GPU is free without stopping other jobs; create a unique run directory; run one experiment per GPU with stdout/stderr, PID, command and exit status recorded. Use screen or a detached supervisor for approved long jobs; no shell-profile or linger changes. A100 UUIDs: GPU-6ae218e6-3d51-b748-e308-1f0509e87886 and GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa. Recheck identity/availability before use. Do not assume device index stability.

Recovery: retain failed run logs; choose an earlier immutable release/build/environment combination for the NEXT run. Do not overwrite or delete existing data or interrupt other users' processes. Remote source modifications are a conflict to review, never silently merged over the Windows workspace.

Setup verification: both sm_80 native libraries compiled, pip check passed, imports passed without CUDA initialization, all 94 CPU regression tests passed, Git tracked files unchanged. These checks are NOT GPU physics, rendering or performance acceptance. Initial conflict logs are retained; final evidence is logs/verification-r1-status.json.
