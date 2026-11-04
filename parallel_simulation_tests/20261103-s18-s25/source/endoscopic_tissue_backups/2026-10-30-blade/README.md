# Scissors contacts in the flap's snips (2026-10-30)

Tools behind section 15 of `endoscopic_tissue/docs/kidney-33hz-plan.md`. Run outputs went to
`endoscopic_tissue/outputs/stage56-runs/blade/`. Nothing here changes the default build, which stays `0d09f56a`.

Debug builds:

- `debug_blade_contacts.py apply`: adds `LCS_GetBladeContacts` to the native sources, which exports every contact of the
  scissors' manifold (kind, material witness, normal, force term, gap). `build_debug.sh` builds it on the committed
  sources (bin-dbgblade) and on eight outer iterations (bin-dbgblade8), `build_debug_jaw.sh` on the jaw relaxation
  candidate (bin-dbgbladej). Each restores the sources after its build.
- `blade_probe.py <output npz> [run.py arguments...]`: the probe of `../2026-10-30-snip3/flap_probe.py` plus the scissors
  contacts at every step with blade contact. Needs one of the debug builds. `blade_runs.sh <prefix> <builds>` runs the
  flap flow once per build with it.

Reading the npz files:

- `contact_summary.py`, `back_loads.py`: forces on the tissue per contact kind through a snip, and the blade backs' load.
- `back_map.py`, `back_faces.py`: which tissue the contacts press at one step, split by box face.
- `wrong_side.py`: blade backs pushing tissue of their own side across the shear plane.
- `lateral_force.py`, `lateral_drift.py`: the scissors' offset along the blade width axis and the forces behind it.
- `misalignment.py`: the distance from the shear face to the flap's mid-plane at the start of each snip.
