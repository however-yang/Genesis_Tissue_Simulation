# Builds the debug export of debug_blade_contacts.py on the candidate of ../2026-10-30-stage56/jaw_relaxation.patch
# (bin-dbgbladej), restoring the sources afterwards.
set -e
R=/d/Simulation/genesis-world; N=$R/endoscopic_tissue/native; B=$R/endoscopic_tissue_backups; L=$R/endoscopic_tissue/outputs/stage56-runs
PY=/d/develop/anaconda3/envs/genesis_world/python.exe
restore() { cd $R && git checkout -- endoscopic_tissue/native/vendor; }
build() { mkdir -p $N/$1; cmd.exe //c "$(cygpath -w $N/build_native.cmd) $(cygpath -w $N/$1)" > $L/build-$1.log 2>&1; echo "$1 built: $(grep -o "Linking CXX shared library.*" $L/build-$1.log)"; }
cd $R && git apply $B/2026-10-30-stage56/jaw_relaxation.patch && cd $N/vendor && $PY $B/2026-10-30-blade/debug_blade_contacts.py apply && build bin-dbgbladej; restore
cd $R && git status --short endoscopic_tissue/native/vendor
