# r26: isolated closed-surface contact experiment

Status: physics screening passed; performance gate NOT passed; default library unchanged; no GitHub upload.

Second A100 only, same 65,688 material-node frontal kidney, original 1,110 steps plus 425 real release/placement/regrasp steps. Both runs record all 1,535 video frames. Playback is at simulation dt, not measured real-time throughput.

| Candidate | Original native mean ms | Release/regrasp native mean ms | 20 x 8-step snapshot replay |
|---|---:|---:|---|
| control-r23-a1 | 34.828452 | 43.543612 | All nodes bitwise equal |
| r26-full-01 | 35.182438 | 38.176377 | All nodes bitwise equal |

r26 improves the extra stress sequence by 12.3%, but is still about 26.2 Hz there. The original sequence is 0.35 ms slower. This fails the user's no-frame-regression/default-promotion gate. Material parameters, time step, substeps, iterations, friction, cutting rule, CCD witnesses and constraints are preserved. The baseline is the prior correctness candidate, not a claim that r23 met the final frame budget.

The live certificate was independently cross-checked against a CPU exact-binding certificate. Real frozen cases and open, inconsistent-orientation, duplicate-nonmanifold, ray-edge/vertex/origin cases retain the original physical decisions; ambiguous cases use the original integral. r24 geometry/assembly splitting had no stable gain and remains disabled.

## Actual graph profiling

Nsight Systems 2025.3.2, capture of 3 normal drop-settle steps using cudaProfilerApi, no CPU sampling. SQLite CUDA logical device 0 maps to physical device 1, UUID c1c4c0d5-5b39-9b0e-84a0-a667e65484fa. All 9,448 traced kernels execute on that device. Trace: tmp/r26-nsys-drop.nsys-rep and .sqlite, complete stdout: logs/r26-nsys-drop-01.log.

The kernels total 110.084 ms across the three captured steps, including small QA exports between steps; these traced times include profiler overhead and are not production FPS. Pressure line energy evaluation: 15.364 ms (14.0%); coarse material/contact accumulation: 13.018 ms (11.8%); local vertex solves: 8.004 ms (7.3%); material stencil assembly: 6.048 ms (5.5%); paired fragment queries: 3.313 ms (3.0%). Closure construction ran 9 times, i.e. 3 per step. The section-profile direct path reports extra zero-query work and must not be used to infer actual graph execution counts.

Next isolated experiment: separate material and contact energy evaluation in the pressure line kernel, preserving every term and all search candidates. The current fused kernel has 172 registers and 88 stack bytes/thread. Do not assume splitting helps until resource counts, normal wall time, replay and full interaction video are checked.
