# 进度

已完成代码/历史检查及方案自审。生产源与默认 DLL 未修改。首先实施材料预条件器实验，接触粗筛随后独立实施。

首次探针包装器在 LCS_Init 时过早读取 tick，模拟尚未开始即 AttributeError。已将读取移入 ready/LCS_Step 条件；失败目录保留，重跑使用 diagnostic-a2。候选 CUDA 无需修改。

- Initial additive frozen-state check failed at step 800 (relative L2 1.2003e-4 vs fixed 1e-4 threshold). Localized mainly to a cut-rim fine block with condition number 17634. Kept failed evidence and initial implementation. Candidate-only double cofactor solve passed 1038 actual GPU block tests against independent float64 solves (max relative 5.60e-8). Thresholds unchanged; full trajectory and speed still pending.

- Corrected additive candidate completed all 1110 steps and 10 frozen hierarchy checks (max relative error 9.2631e-8). ABBA normal Graph core: baseline 44.1102 / 44.5928 ms; additive 41.6685 / 41.9922 ms. Pipeline: baseline 57.8802 / 56.1709 ms; additive 50.9732 / 53.0032 ms. All four quality reports still failed; do not equate speed or completed trajectories with acceptance. Full new baseline/candidate captures in progress.
- Contact group candidate prepared independently: six surface-update sites rebuild groups; five jaw paths and blade queries classify conservatively. Original query body retained plus optional early guard; previous blade preload bypasses coarse rejection. Diagnostic compares full valid records and original triangle broad-phase overlap, plus group-rejection coverage.

- All normal additive ABBA timing finished before contact compilation. Contact build may overlap replay capture/render only; no normal timings or physical simulations run concurrently.

- Additive capture/review completed: two fresh 1111-frame videos and full comparison. Inspected all six matched contact sheets and complete motion plot; visible second-snip geometry differences, changed blade loads, hold tumor RMS peak 0.1596 vs 1.0536 mm/s (mean 0.0600 vs 0.2431). Not adopted; no clinical realism claim.
- Groups compiled as independent normal/probe DLLs and all three production CUDA sources restored. Fresh source verification passed: 8 DLL hashes, 6 patches, unchanged default native sources, frozen reference query identical.
- Groups diagnostic completed 1110 steps / 270 checks: 153750112 triangle queries, 50271524 group rejections, zero contact/degree/member/count or original-AABB-coverage differences. Preload bypass counter is zero: this scenario did NOT exercise a retained preload outside a rejected group; keep that coverage limitation. Normal ABBA now running.

- Groups normal ABBA preserved: A1 44.2211/57.1610 ms core/pipeline; B1 102.3365/116.2118; B2 44.7023/61.8042; A2 44.4850/58.7289. B1 slow interval is exactly steps 748–1110 (363 steps >100 ms), typical 210–230 ms after second cut; B2 does not reproduce. No outlier deletion. Added CPU graph-selection counter build and bounded 1–3 diagnostic reproductions after capture/profile, with automatic 4-step section collection if persistent >100ms occurs. All production sources restored after that build too.

- Four 1111-state captures and two complete comparison videos are finished. Both six-sheet sets and full motion plots visually inspected. Both baseline/candidate quality reports remain false; fresh baselines themselves also vary. No claim of clinical equivalence.
- Three separate section profiles completed. Additive PCG iteration section is 5.2760 → 2.9303 ms/step; remaining blocks persist. Bounded graph slowdown diagnosis now running, no simultaneous normal timings.

- Bounded graph diagnostics finished 3×1110 steps without reproducing the persistent slowdown. Loaded/free/recovery counts: 1719/1611/0, 1714/1616/0, 1719/1611/0; maximum unprofiled core 63.063, 60.317, 65.125 ms. Original groups-b1 outlier remains unexplained, not discarded or claimed fixed. All GPU work finished before artifact verification.

- Initial final artifact verification passed: 8 normal timing runs, 4 complete captures, all 6666 frames across 6 original videos, 10 material frozen states, 270 contact states, 3 section profiles, 3 graph diagnostic repeats, 7 applicable patches, unchanged production sources/default DLLs. Added two H.264 playback copies (1111 frames each, all decoded) retaining original videos; final archive verification includes them.

- Final verification passed with the two playback copies: 6666 original video frames + 2222 H.264 frames decoded, all timings/probes/sections/graph diagnostics and protected defaults checked. Chinese report generated from verified evidence. Decision: neither candidate adopted; 33 Hz not reached. Archive includes only this experiment directory's small sources/results and the report, never native DLLs, raw snapshots, replay streams, videos, logs or unrelated workspace files.
