# Excised tumor crease (2026-10-29/30)

Tools behind section 11 of `endoscopic_tissue/docs/kidney-33hz-plan.md`. Outputs went to
`endoscopic_tissue/outputs/crease-runs/`.

Offline analysis of stage snapshots (run folders with `*.npz`):

- `fold_profile.py`, `stage_notch.py`, `notch_series.py`: the z thickness of the tumor's rest columns, the measure of
  the crease (centre column against the columns at x = +-3, +-4 mm).
- `tet_strain.py`: inverted and nearly flat tumor tets, and where they sit at rest.
- `relax_piece.py`: static equilibrium of the held piece (or, with `--radius`, of the tissue around the tumor) under the
  solver's tissue energy, with the gripped nodes, the dump's witnesses and the far nodes held.
- `floor_lock.py`: an emulation of the vertex step bound at the volume floor.

Solver internals, from debug builds:

- `debug_crease_dump.py`, `debug_material_dump.py`: per-node dumps at chosen frames (positions, local vertex step and
  its bound, contact and anchor interaction, PCG direction before and after the floor restriction, line search result);
  read them with `analyze_dump.py`, `restriction_cause.py` and `projection_emulation.py`.
- `debug_push.py` with `push_run.py` / `push_runs.sh`: a force on the tumor's +z (or -z) centre over a range of frames,
  which reproduces the lock on demand.
- `debug_floor_count.py`: a stderr line per substep in floor recovery, to check where the fix fires.

Builds and runs: `build_variants.sh` (debug dumps and push, plus the listed scripts) and `build_candidate.sh` (the listed
scripts alone) apply patch scripts to the committed sources and restore them after (`git checkout`), so commit or stash
source edits first. `run_crease.sh`, `flap_runs.sh`, `wall_runs.sh` and `knife_runs.sh` run flows with a build; the
kidney realism probe is `../2026-10-29-stage3/probe_runs.sh` with `PROBE_OUTPUT` set.

The `variant_*.py` scripts are the alternatives tried before the fix and patch the sources of commit 697a4ff: grasped
projection, more line trials, trust region with projection, floor curvature, floor band, active-set re-solve, rigid
clusters and uniform restriction. The fix combines floor curvature, floor band and uniform restriction for 6 substeps,
only for a piece cut free of every pinned node and untouched by the blade, after a free-tissue tet at the floor cut its
first-pass material direction and the line search found no step (section 11.2 of the plan says why each gate is there).
