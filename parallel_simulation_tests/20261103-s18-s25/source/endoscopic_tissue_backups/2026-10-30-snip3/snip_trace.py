"""Print the per-step scissors trace of one stage for a run folder, every `stride` steps."""
import json
import sys
from pathlib import Path

folder, stage, stride = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rows = [json.loads(line) for line in open(folder / "trajectory.jsonl")]
rows = [r for r in rows if r["stage"] == stage]
for k, r in enumerate(rows):
    if k % stride and k != len(rows) - 1:
        continue
    L = r["scissors_loads"]
    print(f"{r['step']:5d} closure {r['command']['scissors_closure']:.2f} gap {r['scissors_gap_cm']*10:6.2f} "
          f"L {L['left_n']:6.2f} R {L['right_n']:6.2f} press {r['scissors_pressure_n']:6.2f} "
          f"contacts {r['scissors_contacts']:4d} cut {r['cut_edges']:3d} frac {L['fractures']} sq {L['is_squeezing']} "
          f"overlap {r['blade_body_overlap_mm']:.2f} jaw {r['jaw_gap_cm']*10:.2f} hy {r['hinge_cm'][1]:.3f}")
