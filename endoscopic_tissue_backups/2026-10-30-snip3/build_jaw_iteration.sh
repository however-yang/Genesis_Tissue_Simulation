# Builds the jaw relaxation candidate with its rounds at the end of a chosen outer iteration instead of the third, one
# DLL per listed iteration into bin-jaw2i<iteration>, from the committed sources, restoring them after each build.
# Usage: bash build_jaw_iteration.sh <iteration>...
set -e
R=/d/Simulation/genesis-world; N=$R/endoscopic_tissue/native; L=$R/endoscopic_tissue/outputs/stage56-runs
P=$R/endoscopic_tissue_backups/2026-10-30-stage56/jaw_relaxation.patch
S=endoscopic_tissue/native/vendor/cuda/xpbd/xpbd_solver.cu
restore() { cd $R && git checkout -- endoscopic_tissue/native/vendor/cuda; }
build() { mkdir -p $N/$1; cmd.exe //c "$(cygpath -w $N/build_native.cmd) $(cygpath -w $N/$1)" > $L/build-$1.log 2>&1; echo "$1 built: $(grep -o "Linking CXX shared library.*" $L/build-$1.log)"; }
for iteration in "$@"; do
  cd $R && git apply $P
  sed -i "s/if(it==2)for(int round=0;round<JawRelaxationRounds;round++)/if(it==$iteration)for(int round=0;round<JawRelaxationRounds;round++)/" $S
  test "$(grep -c "if(it==$iteration)for(int round=0;round<JawRelaxationRounds" $S)" = 1
  build bin-jaw2i$iteration; restore
done
cd $R && git status --short endoscopic_tissue/native/vendor
