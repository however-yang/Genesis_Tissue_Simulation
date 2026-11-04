# Builds the debug export of debug_blade_contacts.py on the committed sources (bin-dbgblade) and on eight outer
# iterations (bin-dbgblade8, ../2026-10-30-stage56/variant_more_iterations.py), restoring the sources after each.
set -e
R=/d/Simulation/genesis-world; N=$R/endoscopic_tissue/native; B=$R/endoscopic_tissue_backups; L=$R/endoscopic_tissue/outputs/stage56-runs
PY=/d/develop/anaconda3/envs/genesis_world/python.exe
restore() { cd $R && git checkout -- endoscopic_tissue/native/vendor; }
build() { mkdir -p $N/$1; cmd.exe //c "$(cygpath -w $N/build_native.cmd) $(cygpath -w $N/$1)" > $L/build-$1.log 2>&1; echo "$1 built: $(grep -o "Linking CXX shared library.*" $L/build-$1.log)"; }
cd $N/vendor && $PY $B/2026-10-30-blade/debug_blade_contacts.py apply && build bin-dbgblade; restore
cd $N/vendor && $PY $B/2026-10-30-blade/debug_blade_contacts.py apply && cd cuda && $PY $B/2026-10-30-stage56/variant_more_iterations.py apply 8 && build bin-dbgblade8; restore
cd $R && git status --short endoscopic_tissue/native/vendor
