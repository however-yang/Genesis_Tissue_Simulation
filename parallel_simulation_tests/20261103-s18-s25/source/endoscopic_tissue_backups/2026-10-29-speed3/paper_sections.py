"""Print the abstract and the passages around given keywords of one extracted paper.

Usage: python paper_sections.py <id> <abstract characters> <keyword> [<keyword> ...]
"""
import re
import sys
from pathlib import Path

folder = Path(r"C:\WINDOWS\TEMP\claude\D--Simulation-genesis-world\6de6da1d-3b43-4fa3-91f1-fd2ef998be2d\scratchpad\papers-txt")
key, size, keywords = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
text = re.sub(r"\s+", " ", (folder / f"{key}.txt").read_text(encoding="utf-8", errors="replace"))
match = re.search(r"(Abstract|ABSTRACT|abstract)", text)
start = match.start() if match and match.start() < 6000 else 0
print(text[start:start + size])
for keyword in keywords:
    for hit in list(re.finditer(keyword, text))[:3]:
        s = max(0, hit.start() - 250)
        print(f"\n[{keyword}] ..." + text[s:s + 750] + "...")
