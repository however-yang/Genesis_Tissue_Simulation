# Builds the convergence references of stage 5 from the committed sources, restoring them after each: bin-it6 and
# bin-it8 (6 and 8 outer iterations), bin-dbg4c and bin-dbg8 (4 and 8 outer iterations, with the residual print of
# ../2026-10-29-stage3/debug_vertex_residual.py).
set -e
N=/d/Simulation/genesis-world/endoscopic_tissue/native; B=/d/Simulation/genesis-world/endoscopic_tissue_backups
L=/d/Simulation/genesis-world/endoscopic_tissue/outputs/stage56-runs
PY=/d/develop/anaconda3/envs/genesis_world/python.exe
restore() { cd /d/Simulation/genesis-world && git checkout -- endoscopic_tissue/native/vendor/cuda; }
build() { mkdir -p $N/$1; cmd.exe //c "$(cygpath -w $N/build_native.cmd) $(cygpath -w $N/$1)" > $L/build-$1.log 2>&1; echo "$1 built: $(grep -o "Linking CXX shared library.*" $L/build-$1.log)"; }
cd $N/vendor/cuda && $PY $B/2026-10-30-stage56/variant_more_iterations.py apply 6 && build bin-it6; restore
cd $N/vendor/cuda && $PY $B/2026-10-30-stage56/variant_more_iterations.py apply 8 && build bin-it8; restore
cd $N/vendor/cuda && $PY $B/2026-10-29-stage3/debug_vertex_residual.py apply && build bin-dbg4c; restore
cd $N/vendor/cuda && $PY $B/2026-10-29-stage3/debug_vertex_residual.py apply && $PY $B/2026-10-30-stage56/variant_more_iterations.py apply 8 && build bin-dbg8; restore
cd /d/Simulation/genesis-world && git status --short endoscopic_tissue/native/vendor
