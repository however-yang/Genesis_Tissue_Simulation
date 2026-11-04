"""Print the abstract (or the opening text) of each extracted paper in an id range, for triage.

Usage: python paper_abstracts.py <first id> <last id> [characters]
"""
import re
import sys
from pathlib import Path

folder = Path(r"C:\WINDOWS\TEMP\claude\D--Simulation-genesis-world\6de6da1d-3b43-4fa3-91f1-fd2ef998be2d\scratchpad\papers-txt")
first, last = int(sys.argv[1]), int(sys.argv[2])
size = int(sys.argv[3]) if len(sys.argv) > 3 else 1100
index = {line.split("\t")[0]: line.split("\t")[2].strip() for line in (folder / "index.tsv").read_text(encoding="utf-8").splitlines()}
for k in range(first, last + 1):
    key = f"{k:02d}"
    text = (folder / f"{key}.txt").read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\s+", " ", text)
    match = re.search(r"(Abstract|ABSTRACT|摘\s*要)", text)
    start = match.start() if match and match.start() < 6000 else 0
    print(f"### {key} {index[key]}")
    print(text[start:start + size])
    print()
