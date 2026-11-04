# Isolated S18–S25 candidate overlay

This is **not an accepted replacement** for the default simulator. The training adapter changes live in this checkout's `endoscopic_tissue/training`; opt-ins preserve the default entrypoint and material/contact configuration.

Native base: `runs/a100-functional-fixes-20261102-r1/release-training-r3/source` in the same checkout. Copy that directory into a **new** build source directory, then overlay `native-overlay/`. Do not overwrite the default native source or binary.

Use the existing CUDA 12.5 toolchain and the base CMake configuration, with `CMAKE_CUDA_ARCHITECTURES=80` and a separate build directory. The local verification used `cmake --build <build> -j2`. The complete resulting source hashes and tested Linux library SHA are in `native-manifest.json`; a rebuilt binary may differ with toolchain paths and must receive its own SHA and tests.

Changes: append a read-only labels export to the XPBD translation unit; new `labels_readback.cuh`. No physical solver kernel or force/material/contact constant changes. ABI `LCS_SurfaceBindingsAbiVersion() == 2`; native snapshot ABI remains 1. Disposable scratch allocations bypass persistent snapshot tracking. Static and post-cut probes check full native archive byte equality before/after label extraction.

Valid pixel support labels use `x = sum(w_i * node_i) + affine_offset`. Offsets are required for the native cut-surface contact binding; these are not exact global FEM shape functions. Unknown bindings remain invalid.

See the report and test scripts supplied beside this file. Production integration is blocked by the rendering-inclusive 20 Hz gate and unresolved S18/S22 acceptance tests.
