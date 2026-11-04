"""Gather the kidney scene's open problems into one numbered list in its design doc, and refresh the memory hook."""
from pathlib import Path

doc = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\docs\kidney-tumor-plan.md")
s = doc.read_text(encoding="utf8")
start = s.index("**组织层面已经分成两块，但还有外层壳连着。**")
tether_end = s.index("修复需要修改原生剪切规则（建议）：")
fix_end = s.index("另外两项需要用户决定：")
tether = s[start:tether_end].strip().replace("**组织层面已经分成两块，但还有外层壳连着。** ", "")
fix = s[tether_end:fix_end].strip().replace("修复需要修改原生剪切规则（建议）：", "")
topology_start = s.index("1. **表面拓扑检查。** ") + len("1. **表面拓扑检查。** ")
topology_end = s.index("2. **游离表皮切口检查。**")
stray_start = topology_end + len("2. **游离表皮切口检查。** ")
stray_end = s.index("## 已知现象")
spike_start = stray_end + len("## 已知现象")
topology = s[topology_start:topology_end].strip()
stray = s[stray_start:stray_end].strip()
spike = s[spike_start:].strip()
issues = f"""## 待解决问题（2026-10-28 记录）

1. **肿瘤仍被外壳连着（阻塞验收）。** 按组织连接已经分成两块，但还有外层壳连着。{tether}
   - 建议修复，需要修改原生剪切规则：{fix}
   - 状态：等用户同意后试做。
2. **表面拓扑检查。** {topology}
   - 状态：等用户决定是否接受这种闭合裂尖。建议先做第 1 项，再看这里是否还存在。
3. **游离表皮切口检查。** {stray}
   - 状态：等用户决定是否放宽弯曲表面的距离限值。建议同样先做第 1 项。
4. **刀身受力单步尖峰。** {spike}
   - 状态：未查。

在第 1–3 项解决之前，正式运行会在第一刀结束时被拓扑检查中止。要完整观看流程，可以用开发运行脚本 `endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py`：它把拓扑问题记入 `surface-audit.json`，不中止运行。流程最后仍会因第 1 项判为未通过。
"""
s = s[:start] + issues
doc.write_text(s, encoding="utf8")

memory = Path(r"C:\Users\Dell\.claude\projects\D--Simulation-genesis-world\memory\MEMORY.md")
m = memory.read_text(encoding="utf8")
old = ("- [Endoscopic tissue kidney excision](endoscopic-tissue-kidney-excision.md) — rim-up kidney, oblate tumor on a "
       "neck, two-snip excision passes physically; two acceptance checks await the user")
new = ("- [Endoscopic tissue kidney excision](endoscopic-tissue-kidney-excision.md) — rim-up kidney, oblate tumor on a "
       "neck; two snips cut the tissue but shell links still tether the tumor; native fix and two checks await the user")
assert m.count(old) == 1
memory.write_text(m.replace(old, new), encoding="utf8")
print("recorded")
