# Times the kidney flow (../2026-10-29-speed3/timing_calls.py) for each listed build, alternating builds over two
# rounds, then renders one headless run per build (stage stills and video). Outputs go to outputs/stage56-runs as
# t5-<round>-<build> and render-<build>.
# Usage: bash timing_render.sh <dll folder under endoscopic_tissue/native>...
O=D:/Simulation/genesis-world/endoscopic_tissue/outputs/stage56-runs; PY=/d/develop/anaconda3/envs/genesis_world/python.exe
N=D:/Simulation/genesis-world/endoscopic_tissue/native
cd /d/Simulation/genesis-world
for k in 1 2; do for b in "$@"; do
  PYTHONPATH=. timeout 2400 $PY endoscopic_tissue_backups/2026-10-29-speed3/timing_calls.py --scene kidney --native-only \
    --dll $N/$b/LiverCudaSim.dll --output "$O/t5-$k-$b" > "$O/t5-$k-$b.log" 2>&1
  rm -f "${O:?}/t5-$k-$b"/*.npz; echo "t5-$k-$b done"
done; done
for b in "$@"; do
  PYTHONPATH=. timeout 2400 $PY endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py --scene kidney --headless \
    --dll $N/$b/LiverCudaSim.dll --output "$O/render-$b" > "$O/render-$b.log" 2>&1
  rm -f "${O:?}/render-$b"/*.npz; echo "render-$b done"
done
