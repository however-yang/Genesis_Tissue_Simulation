"""Insert or remove per-kernel section marks in the unified substep for profiling builds.

Usage: python fine_marks.py apply|remove <xpbd_solver.cu>

Each mark records the time of the kernel launched just before it under "fine: <region> <kernel>". Marks are written
with a trailing /*FINEMARK*/ so that remove deletes exactly what apply inserted.
"""
import re
import sys

mode, path = sys.argv[1:]
text = open(path, encoding="utf-8", newline="").read()
MARK = re.compile(r'xpbd_section_mark\([^;]*\);/\*FINEMARK\*/')
if mode == "remove":
    text, count = MARK.subn("", text)
    open(path, "w", encoding="utf-8", newline="").write(text)
    print("removed", count)
    sys.exit(0)
assert not MARK.search(text), "marks already applied"
CALL = re.compile(r'(k_\w+)<<<[^;]*?>>>\(([^;]*?)\);')
regions = [
    ("pre", 'xpbd_section_mark("pre: tool, contact discovery, manifold");', 'xpbd_section_mark("pre: predictor, recovery query");'),
    ("coarse", 'if(it==0 || it==2)for(int mode=1;mode<=(it==0?4:1);mode++){', 'xpbd_section_mark("loop: coarse modes");'),
    ("assembly", 'xpbd_section_mark("cg assembly: start");', 'xpbd_section_mark("cg: assembly");'),
    ("cg", 'for(int cg=0;cg<(nonlinear==0', 'xpbd_section_mark("cg: iterations");'),
    ("feasible", 'xpbd_section_mark("cg: trust, projection");', 'xpbd_section_mark("cg: feasible");'),
    ("line", 'xpbd_section_mark("cg: feasible");', 'xpbd_section_mark("cg: line search");'),
    ("discovery", 'xpbd_section_mark("cg: line search");', 'xpbd_section_mark("loop: contact discovery");'),
    ("regional", 'xpbd_section_mark("loop: contact discovery");', 'xpbd_section_mark("loop: regional mode");'),
    ("sweeps", 'xpbd_section_mark("loop: regional mode");', 'xpbd_section_mark("loop: local sweeps");'),
    ("global", 'xpbd_section_mark("loop: local sweeps");', 'xpbd_section_mark("loop: tool, safe and global coarse");'),
    ("rigid", 'xpbd_section_mark("loop: tool, safe and global coarse");', 'xpbd_section_mark("loop: rigid modes");'),
    ("fragment", 'cudaMemsetAsync(d_fragmentLambda,0,6*g_N*sizeof(float));', 'e=finishFeasible();if(e!=cudaSuccess)return e;'),
]
# finishFeasible runs a captured graph, so it is timed as a whole from outside.
FEASIBLE = 'e=finishFeasible();if(e!=cudaSuccess)return e;'
assert text.count(FEASIBLE) == 1
text = text.replace(FEASIBLE, FEASIBLE + 'xpbd_section_mark("fine: fragment finishFeasible");/*FINEMARK*/')
# The substep prologue refreshes the component labels and their support before its fragment pass.
for call, label in (("cudaError_t contactError=refreshUnifiedComponents();", "prologue components"),
                    ("if(contactError==cudaSuccess)contactError=refreshFragmentSupport();", "prologue support")):
    assert text.count(call) == 1
    text = text.replace(call, call + f'xpbd_section_mark("fine: {label}");/*FINEMARK*/')
for region, start_anchor, end_anchor in regions:
    start = text.index(start_anchor)
    end = text.index(end_anchor, start + 1)
    body = text[start:end]

    def mark(match):
        name = match.group(1)
        if match.group(2).replace(" ", "").endswith(",true"):
            name += " bound"
        if name == "k_UnifiedVertex" and region == "sweeps":
            label = 'it==3?"fine: sweeps k_UnifiedVertex full":"fine: sweeps k_UnifiedVertex witness"'
        else:
            label = f'"fine: {region} {name}"'
        return match.group(0) + f"xpbd_section_mark({label});/*FINEMARK*/"

    body, count = CALL.subn(mark, body)
    print(region, count)
    text = text[:start] + body + text[end:]
open(path, "w", encoding="utf-8", newline="").write(text)
