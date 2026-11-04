# Builds the stage-4 debug and variant DLLs from the committed sources: bin-dbg4 (4 iterations, residual print),
# bin-dbg3 (3 iterations, residual print) and bin-it3 (3 iterations), restoring the sources after each.
set -e
N=/d/Simulation/genesis-world/endoscopic_tissue/native; B=/d/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-29-stage3
L=/d/Simulation/genesis-world/endoscopic_tissue/outputs/stage3-runs
PY=/d/develop/anaconda3/envs/genesis_world/python.exe
restore() { cd /d/Simulation/genesis-world && git checkout -- endoscopic_tissue/native/vendor/cuda/xpbd/xpbd_solver.cu endoscopic_tissue/native/vendor/cuda/solver/material/unified_vertex.cuh; }
build() { mkdir -p $N/$1; cmd.exe //c "$(cygpath -w $N/build_native.cmd) $(cygpath -w $N/$1)" > $L/build-$1.log 2>&1; echo "$1 built: $(grep -o "Linking CXX shared library.*" $L/build-$1.log)"; }
cd $N/vendor/cuda && $PY $B/debug_vertex_residual.py apply && build bin-dbg4; restore
cd $N/vendor/cuda && $PY $B/debug_vertex_residual.py apply && $PY $B/variant_three_iterations.py apply && build bin-dbg3; restore
cd $N/vendor/cuda && $PY $B/variant_three_iterations.py apply && build bin-it3; restore
cd /d/Simulation/genesis-world && git status --short endoscopic_tissue/native/vendor
