"""Summarize the COARSE lines of a patch_debug_coarse.py run, per flow stage: how often the step bound and the
pressure line shorten each sequential coarse mode of the substep loop.

Usage: python coarse_scale_stats.py <run log>

Within a substep the loop solves modes 1, 2, 3, 4 (outer iteration 0) and mode 1 again (iteration 2); the summary
names them 1-4 and "1b". Each solve prints its bounded scale (phase 0) and its final scale (phase 1) per component.
"""
import collections
import re
import sys

STAGE = re.compile(r"^\[(\d+)\] (\w+)")
LINE = re.compile(r"^COARSE (\d+) ([01]) (\d+) (\S+) (\S+)")
counts = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
stage, previous_mode, seen_four = "initial", {}, {}
for text in open(sys.argv[1], encoding="utf-8", errors="replace"):
    match = STAGE.match(text)
    if match:
        stage = match.group(2)
        continue
    match = LINE.match(text)
    if not match:
        continue
    mode, phase, root, scale = int(match.group(1)), int(match.group(2)), int(match.group(3)), float(match.group(4))
    if phase == 0:
        # A mode-1 solve that follows a mode 1-4 solve of the same root within the substep is the iteration-2 one.
        name = ("1b" if previous_mode.get(root) in ("1", "2", "3", "4") else "1") if mode == 1 else str(mode)
        previous_mode[root] = name
        row = counts[stage][name]
        row["solves"] += 1
        row["bounded"] += scale < 1
        row["bound_zero"] += scale <= 0
        seen_four[root] = (name, scale)
    else:
        name, bounded = seen_four.get(root, ("?", 1.0))
        row = counts[stage][name]
        row["shortened_by_line"] += scale < bounded
        row["final_below_half"] += scale < 0.5
        row["final_zero"] += scale <= 0
for stage, modes in counts.items():
    print(stage)
    for name in ("1", "2", "3", "4", "1b"):
        row = modes.get(name)
        if not row or not row["solves"]:
            continue
        n = row["solves"]
        print(f"  mode {name:2s} solves {n:6d}  bounded {row['bounded'] / n:6.1%}  bound 0 {row['bound_zero'] / n:6.1%}  "
              f"line shortened {row['shortened_by_line'] / n:6.1%}  final<0.5 {row['final_below_half'] / n:6.1%}  "
              f"final 0 {row['final_zero'] / n:6.1%}")
