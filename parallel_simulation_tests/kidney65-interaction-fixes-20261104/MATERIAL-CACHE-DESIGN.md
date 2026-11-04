# Same-assembly tetrahedral cache (r30 experiment)

The stencil assembler reads each tetrahedron from four incident node rows at the same, frozen physical positions. It repeats F, cof(F), G=F*inverse(F_previous), FF' and GG'. The new experiment derives these once per live tetrahedron immediately before the original row assembly, on the same CUDA stream. It preserves each node row's gradient accumulation order, compression barrier, hardening, pressure, damping, half-scaled matrix storage, PCG/multigrid budget and nonlinear acceptance rules.

`material_tet_cache.cuh` stores 45 floats/tet. The only added production launch is the derivation kernel. No cached quantity survives to a subsequent assembly without being overwritten. Dead tets are ignored by both producer and consumer. No reuse occurs between Gauss-Seidel colours, nonlinear iterations, substeps, or topology changes.

Allocation is a private derived workspace, excluded from the physical snapshot just like the existing coarse-bound cache. It is allocated once before graph capture and freed at session close. Capacity growth in a live session is rejected instead of leaving graphs with stale pointers. The graph's pointer remains stable throughout the session. Restore correctness must still be tested by the same 20 x 8-step and training-level checks.

`LCS_MaterialStencilProbe` runs both the original and cached assemblers on identical live inputs and private, zero-initialized outputs. It compares every half-matrix entry, scale, residual and diagonal bit, counts nonfinite outputs, and returns absolute error/reference scales. Its negative control modifies a private cached F and must produce mismatches. It never writes live positions, topology, real stencil outputs, or the production cache.

Initial 30-step settled-kidney probe: 17,014,536 half entries, 1,890,504 scales, 708,939 residual components and 1,417,878 diagonal components matched bitwise; negative control changed 40 matrix entries, 3 scales and 6 diagonal entries. This is only an initial-input check. Full deformed/cut/release/regrasp probes, normal timing without probes, complete videos and training validation are required. Default library remains unchanged.

## Other current experiments

- r28 split material/contact line energy: complete physics/replay screening passed, no performance gain, not adopted (original 35.283 ms; extra 38.794 ms).
- r29 compile-time specialization of full/half search passes: original 34.614 ms; extra 38.694 ms; full interaction and replay screening passed. Still above final frame target. Parent for r30 only inside this test directory.
- r26 strict pure-friction held preparation: rejected (GRASP-01), left jaw 0.0433 N at reset vs unchanged 0.05 N criterion, 5 material nodes between pads, all witnesses retained, zero anchors. Additional 0.5 mm -x bite adjustment was also rejected (left 0.0260 N). Do not present these as certified friction holds. Compare stationary-settling controls before changing physical laws or acceptance criteria.
