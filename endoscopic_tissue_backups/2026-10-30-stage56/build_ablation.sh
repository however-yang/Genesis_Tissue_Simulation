# Builds one DLL per listed pass sequence of variant_pass_pattern.py into bin-p<sequence>, from the committed sources,
# restoring them after each build. Usage: bash build_ablation.sh <sequence>...
set -e
N=/d/Simulation/genesis-world/endoscopic_tissue/native; B=/d/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-30-stage56
L=/d/Simulation/genesis-world/endoscopic_tissue/outputs/stage56-runs
PY=/d/develop/anaconda3/envs/genesis_world/python.exe
restore() { cd /d/Simulation/genesis-world && git checkout -- endoscopic_tissue/native/vendor/cuda; }
build() { mkdir -p $N/$1; cmd.exe //c "$(cygpath -w $N/build_native.cmd) $(cygpath -w $N/$1)" > $L/build-$1.log 2>&1; echo "$1 built: $(grep -o "Linking CXX shared library.*" $L/build-$1.log)"; }
for sequence in "$@"; do
  cd $N/vendor/cuda && $PY $B/variant_pass_pattern.py apply $sequence && build bin-p$sequence; restore
done
cd /d/Simulation/genesis-world && git status --short endoscopic_tissue/native/vendor
