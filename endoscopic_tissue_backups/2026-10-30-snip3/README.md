# Why the flap's third snip jams (2026-10-30)

Tools behind section 14 of `endoscopic_tissue/docs/kidney-33hz-plan.md`. Run outputs went to
`endoscopic_tissue/outputs/stage56-runs/snip3/`. Nothing here changes the default build, which stays `0d09f56a`.

Running:

- `flap_probe.py <output npz> [run.py arguments...]`: runs a scissors flow and saves the stage-end node positions, every
  severed link with its step, the positions of the nodes around the flap at every step, and the contact-surface depth
  before and after each step's repair. `SCISSORS_DZ` (cm) moves the flap scene's scissors poses along z,
  `REPAIR_PASSES` sets the repair passes per step.
- `flap_runs.sh <prefix> <builds>`: one probed flap flow per build, summarized by `../2026-10-30-stage56/scene_line.py`.
- `build_jaw_iteration.sh <iterations>`: builds the candidate of section 13.4 with its jaw relaxation rounds after the
  given outer iteration instead of the third (bin-jaw2i0, bin-jaw2i1), restoring the sources after each build.
  `jaw_relaxation_first.patch` is the bin-jaw2i0 variant as a patch on the committed sources.

Reading the npz files (each script documents its arguments):

- `snip_trace.py`, `stage_ends.py`: scissors loads, gap and grasper state per step or at stage ends, from the run's
  `trajectory.jsonl`.
- `cut_map.py`, `level_map.py`: which snip severed each tab column and in which link along z (stepped wounds).
- `free_end_trace.py`, `layer_trace.py`: how the free end and the node layers next to the shear plane move during a
  stage.
- `wedge_nodes.py`: the nodes between the two blades at one step, with the cut state of their columns.
- `pre_offset.py`: per build, the x shift of the flap's root columns before the third snip and the run outcomes.
- `volume_floor.py`: the most compressed lattice tetrahedra around the flap at stage ends.
