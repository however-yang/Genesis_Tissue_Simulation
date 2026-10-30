# Stages 5 and 6 of the 33 Hz plan (2026-10-30)

Tools behind section 12 of `endoscopic_tissue/docs/kidney-33hz-plan.md`. Outputs went to
`endoscopic_tissue/outputs/stage56-runs/`. Neither stage was adopted, and the default build stays `0d09f56a`.

Stage 6 (2 substeps instead of 3):

- `substeps.py`: runs a scissors flow with SUBSTEPS substeps (default 2). With `--probe <jsonl>` the run goes through
  `../2026-10-29-stage3/realism_probe.py`, without it like `dev_run.py`.
- `scene_line.py`: one summary line per run folder: status, blade-back overlap, uncut columns, grip points, and each
  snip's fracture loads and two-blade squeeze.

Stage 5 (how far the current solve is from converged):

- `variant_more_iterations.py <count>`: patch script for more outer iterations per substep (first and last as now, the
  ones between alternating the second and third). `build_stage5.sh` builds bin-it6, bin-it8 and the residual-print
  builds bin-dbg4c and bin-dbg8 from the committed sources and restores them after each build (commit source edits
  first).
- The kidney probes ran through `../2026-10-29-stage3/probe_runs.sh` with `PROBE_OUTPUT` set to the output folder, the
  residuals through `../2026-10-29-stage3/residual_stats.py`, the traction notch through
  `../2026-10-29-crease/stage_notch.py`.
- `timing_render.sh <builds>`: times the kidney flow for each build over two alternating rounds
  (`../2026-10-29-speed3/timing_calls.py`), then renders one headless run per build. A headless render run once stopped
  early without an error. Its rerun completed, so check that every stage still exists before composing.
- `scene_runs.sh <scene> <prefix> <builds>`: runs the flap or wall flow once per build and prints `scene_line.py`'s
  line.
- `ab_montage.py`: composes the stage stills of render runs into one comparison image.
