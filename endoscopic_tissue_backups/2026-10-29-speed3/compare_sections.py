"""Compare two section profiles of profile_sections.py stage by stage, by region and by kernel.

Usage: python compare_sections.py <before json> <after json> [stage ...]

A fine_marks.py build names its marks "fine: <region> <kernel>"; the regions sum those, and the other sections keep
their own names.
"""
import collections
import json
import sys


def load(path):
    return {entry["stage"]: entry for entry in json.load(open(path))}


def region(name):
    return " ".join(name.split()[:2]) if name.startswith("fine: ") else name


before, after = load(sys.argv[1]), load(sys.argv[2])
stages = sys.argv[3:] or [stage for stage in after if stage in before]
for stage in stages:
    sections_before, sections_after = before[stage]["sections_ms_per_step"], after[stage]["sections_ms_per_step"]
    regions_before, regions_after = collections.Counter(), collections.Counter()
    for name, ms in sections_before.items():
        regions_before[region(name)] += ms
    for name, ms in sections_after.items():
        regions_after[region(name)] += ms
    print(f"{stage}: wall {before[stage]['wall_ms_per_step']:.2f} -> {after[stage]['wall_ms_per_step']:.2f} ms, "
          f"sections {sum(sections_before.values()):.2f} -> {sum(sections_after.values()):.2f} ms per step")
    names = sorted(set(regions_before) | set(regions_after), key=lambda r: -max(regions_before[r], regions_after[r]))
    for name in names[:24]:
        print(f"  {name:64s} {regions_before[name]:7.2f} {regions_after[name]:7.2f}")
    print("  top kernels after:")
    for name, ms in sorted(sections_after.items(), key=lambda item: -item[1])[:24]:
        print(f"    {name:70s} {sections_before.get(name, 0):7.3f} {ms:7.3f}")
