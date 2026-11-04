# Runs the probed flap flow (blade_probe.py) once per listed build: run k is <prefix>-<k>-<build>, its npz beside its
# folder in outputs/stage56-runs/blade, and prints scene_line.py's line for each, then BLADE-RUNS-DONE.
# Usage: bash flap_runs.sh <prefix> <dll folder under endoscopic_tissue/native>...   (SCISSORS_DZ and REPAIR_PASSES pass
# through to blade_probe.py)
P=$1; shift; O=D:/Simulation/genesis-world/endoscopic_tissue/outputs/stage56-runs/blade; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
N=D:/Simulation/genesis-world/endoscopic_tissue/native; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $O; cd /d/Simulation/genesis-world
k=0; for build in "$@"; do k=$((k+1)); name="$P-$k-$build"
  PYTHONPATH=. timeout 1200 $PY $HERE/blade_probe.py "$O/$name.npz" --scene flap --native-only --dll $N/$build/LiverCudaSim.dll --output "$O/$name" > "$O/$name.log" 2>&1
  rm -f "${O:?}/${name:?}"/*.npz
  $PY endoscopic_tissue_backups/2026-10-30-stage56/scene_line.py "$O/$name"
done
echo BLADE-RUNS-DONE
