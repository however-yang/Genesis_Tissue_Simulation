# Current isolated work (2026-11-04)

No default replacement and no GitHub push. User requires every physics/performance/training gate before promotion. All GPU work is server A100 physical GPU1 / UUID c1c4c0d5-5b39-9b0e-84a0-a667e65484fa; Windows only edits and CPU video processing. No other GPU/process/system/dependency changes.

Latest local checkpoint: dccdf7c (after 7b67ac7, dca657f). Only this test directory is changed. Uncommitted work after dccdf7c: shared row experiments, packed MG/probes, zero-relative-motion guard/proofs, opt-in stable-friction adapter and provenance/video tools.

## Production-path timings, LCS_Step only

| Run | Original 1110 ms | True release/regrasp ms | Decision |
|---|---:|---:|---|
| control-r23-a1 | 34.828 | 43.544 | correctness reference, not frame-budget pass |
| r26-full-01 | 35.182 | 38.176 | partial speed gain; strict friction reset failed; no promotion |
| r28-full-01 | 35.283 | 38.794 | split pressure scheduling alone rejected |
| r29-full-01 | 34.614 | 38.694 | best original-path candidate so far; still over frame target |
| r30-full-normal-01 | 36.623 | 40.402 | tet cache slower, rejected |
| r32-full-normal-01 | 35.049 | 38.784 | volatile shared rows no gain, rejected |

All these complete native trajectory/replay tests passed their finite-domain physics gates, including 20 x 8-step bitwise continuations. This does not mean all RL domains or clinical realism are certified. Same baseline minimum frame protection has NOT passed. r31 only has a diagnostic full proof (no production FPS claim).

## Current performance experiments

- r33-packed-mg: only pack the 15 actually coupled shared Galerkin slots instead of 27; keep original 3 loaded / 7 free CG iterations and full multigrid. Settled-state same-input all-level matrices/inverses bitwise equal, private-output negative control detected. Full stage proofs/video/replay running as r33-full-proof-01. No normal throughput evidence yet.
- r34-zero-check: built. Exact equality of all four rounded trial displacements identifies a translated tet, whose internal energy change is zero. Keep node inertia/support/contact terms and the same final pressure reduction. Diagnostic calls the complete original energy out of line for EVERY omitted term and tests a private perturbed-displacement negative control. Not run yet.
- r34-zero-production: build started; same guard without reference-recalculation diagnostics. Must run full physics/replay/video and normal timing after the proof. Do not infer performance from the check build.
- No SSOR/Jacobi/lagged-preconditioner solver replacement was implemented. Existing coupled_material.cuh explicitly says the 3 loaded CG iterations were selected to avoid folding a cut piece along the jaw plane; increasing iterations is not automatically an improvement.

## Pure friction, wall-traction only

- r23 ordinary held +z4mm: one pilot passed (465 frames), native 20.243 ms.
- r26 ordinary held: GRASP-01, weak jaw 0.0433 N below unchanged 0.05 N, all material witnesses retained, zero anchors.
- r26 -x0.5mm bite variant: also rejected (weak jaw 0.0260 N).
- r26 fixed stationary 60 steps: one acquisition/dose pilot completed, but force remains intermittent; not a robustness certificate.
- New optional adapter python/endoscopic_tissue/training/env.py copies the pinned frozen adapter (parent SHA 39c3768f6ebf1cfc4e21b30c36f237467a944b2a6e285bab9e456429356d03a8). Default acquisition_max_steps=0. Explicit mode1 held/cut_ready can wait at the SAME final pose until five consecutive unchanged friction-hold checks succeed, bounded at 180 steps in the pilot. Every extra native step gets the same complete quality checks, preparation log and video. No material/friction/force threshold/anchors changed.
- friction-r26-adaptive-held-01 passed one +z4mm pilot, 470 frames, zero anchors throughout, initial last-ten weak-load minimum 0.06008 N, final friction evidence true, native mean 20.217 ms. This is ONE wall pilot, not the full 900-trial S16d matrix and not kidney pure-friction certification.
- Next: adaptive cut_ready, open-jaw/release negative controls, repeats and five-direction doses; same final native candidate must be revalidated. No criteria weakening or friction anchoring.

## Remaining training acceptance

Current full native tests cover archived-state identity/precut rollback and 20 short post-cut futures. Historical held training API smoke for r23 covers six transitions/seven states, file hashes, snapshot/branch/reset and wrong material/DLL rejection. These do not transfer automatically to a new library.
S15 distribution/four fork events, recut/cache/fresh comparisons and 1000 save/load memory/resource checks remain to be completed on the final candidate. Full S16 tear/overload/friction curves, S17 parameter/geometry coverage and arbitrary-policy domains are not certified. Keep those failures/missing scopes explicit. Read requirements-frozen.md in the original root endoscopic_tissue_backups/2026-11-01-s15-s17 for exact definitions.

## Evidence and tools

Server outputs retain every failure. Native builds freeze source-LABEL/build-LABEL and never overwrite binaries. run_scene now retains the actual four helper source inputs in each run; friction_pilot saves its driver and adapter sources. archive_run rejects diagnostic/probe timings as production FPS. archive_friction makes compact summaries without deleting raw trials.
Local full video comparisons decoded r26/r28/r29. A CPU-only sequential job is downloading/decoding r30 normal, r31 proof and r32 normal now. Their videos and the friction cases still require visual inspection and archiving. make_review --full preserves every input frame, labels held ended clips, and allows --stage-frames as 1-based comma-separated frames. 30fps video playback is simulation time, not measured throughput.

## Later updates

- r33-full-normal-01 passed finite-domain physics/replay, native original 34.174746 ms, extra 38.475436 ms. Still not the 33Hz / default-protection gate.
- **r34-zero-proof-01 REJECTED at bite.** Of 900099 proposed full-step omissions, 896442 had a nonzero complete reference result; fractional counts 238644/237661. A mathematical translation-invariance argument is insufficient to establish exact equivalence of the implemented floating-point energy. Magnitudes/root cause were not measured; do not assert a specific FMA or NaN cause. Never run/promote r34-zero-production on this evidence. Both flags remain OFF in other candidates.
- r35-gather-mg built (SHA 9a54ac8c7b6bd3059938d598d2909a67079a150487601dd30211050cd152ed17): one owner per output, eight lanes per spatial corner, all original terms retained. Settled-state five-level A/inverse arrays bitwise equal; private-output negative control detected. Shared storage 0 bytes (coarse REG54, fine REG60, no stack) versus old 31104 B per block. r35-full-proof-01 now runs full geometry/replay/stage comparisons. No normal FPS result yet.
- Full local r30-normal, r31-proof, r32-normal videos now completely decoded and compared; r32 stage images visually inspected. r30/r31 montage review and friction video download/review remain.
- The optional friction adaptive held pilot passed. cut_ready, opening/release controls, all directions/repeats and final-candidate API tests still remain; no full S16d or general RL/WAM acceptance claim.

## Additional verified results and the next experiment

- r35-full-proof-01 finished all 1110 base steps but failed the unchanged excision gate: one tumor node remained. All 14 positive stage Galerkin comparisons were bitwise equal; the negative control was detected. This is not a complete physics pass.
- r35-full-normal-02 passed one complete 1535-frame interaction and 20 x 8-step bitwise continuations. Native original mean 33.487332 ms, true release/regrasp mean 37.601570 ms (13.65% below the r23 43.543612 ms reference). This is not 33Hz, minimum-frame protection, or a distributional quality certificate. Preserve the preceding failure; no default replacement or push.
- r35-mode-diagnostic-01 failed before simulation because the new counter wrapper expected 0 although LCS_ReadInteractionStats returns the number of entries (16). Fixed the wrapper; no native change. r35-mode-diagnostic-02 completed, with read-only counters and explicit diagnostic-continuation semantics. Its timings are not production benchmarks.
- Real counters: after incision some stages trigger free-node/7-iteration scheduling, while drop_settle and regrasp_close in this run are loaded-only/3-iteration stages. Recovery substeps were zero. Do not claim that all release/regrasp cost comes from the 7-iteration path. The code does redundantly apply the full-grid multigrid preconditioner four times after loaded components have stopped updating. Next isolated experiment will filter those unused loaded-component results only, retaining the same free-component iterations and V(1,1) cycle, with same-input private output proofs before full videos.
- friction-r33-adaptive-cutready-01 completed 692 frames, zero anchors, 107 bounded stationary acquisition steps, witness retention 1.0 and max tangential slip 0.06751 mm; native mean 22.278701 ms. Final bilateral force evidence was false. Completion means quality/scope checks passed, not guaranteed robust grasp. +z is a horizontal/depth pull; the small y-lift change is not proof of a failed +z pull. Added explicit 3D material/tool displacement metrics for following trials.
- Full CPU decoding and stage-image review now include r30/r31/r32, r33 normal and r35 normal. r33/r35 comparisons include all 1535 original/candidate frames. Friction r26 ordinary/adaptive comparison decoded all 345/470 frames with the shorter ended clip explicitly labelled. Do not equate video playback30fps with measured throughput or selected-image inspection with watching every instant.
- snapshot_resources.py is a new isolated S15 T8 driver: 20-cycle screen, 300-cycle diagnostic, 1000-cycle acceptance. It retains unchanged memory/performance thresholds, exact native round-trip, positions/topology/jaw-load and allocation checks. The first r33 20-cycle screen has started; no T8 pass yet.
