# Builds release DLLs from the committed sources with the listed variants only (no debug code):
# bash build_candidate.sh <folder>:<variant script>[+<variant script>...] ...
set -e
N=/d/Simulation/genesis-world/endoscopic_tissue/native; B=/d/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-29-crease
L=/d/Simulation/genesis-world/endoscopic_tissue/outputs/crease-runs
PY=/d/develop/anaconda3/envs/genesis_world/python.exe
restore() { cd /d/Simulation/genesis-world && git checkout -- endoscopic_tissue/native/vendor/cuda; }
build() { mkdir -p $N/$1; cmd.exe //c "$(cygpath -w $N/build_native.cmd) $(cygpath -w $N/$1)" > $L/build-$1.log 2>&1; echo "$1 built: $(grep -o "Linking CXX shared library.*" $L/build-$1.log)"; }
for spec in "$@"; do folder=${spec%%:*}; variants=${spec#*:}
  cd $N/vendor/cuda
  for v in ${variants//+/ }; do $PY $B/$v apply; done
  build $folder; restore
done
cd /d/Simulation/genesis-world && git status --short endoscopic_tissue/native/vendor
