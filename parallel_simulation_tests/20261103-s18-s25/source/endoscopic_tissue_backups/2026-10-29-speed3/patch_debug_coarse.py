"""Debug build only: print, after the step bound and after the pressure-line selection of every coarse mode of the
substep loop, the step scale of each component block that has a Hessian.

Lines read "COARSE <mode> <phase> <root> <scale> <hessian trace>", phase 0 after k_CoarseBound, 1 after the line.
They tell how often the bounds and the line search shorten the sequential modes 1-4, which a joint affine solve would
share. Usage: python patch_debug_coarse.py apply|remove  (from endoscopic_tissue/native/vendor/cuda)
"""
import sys

MARK = "/*DEBUGCOARSE*/"
KERNEL = (MARK + "__global__ void k_DebugCoarseScale(const UnifiedState* state,const CoarseBlock* blocks,int phase){"
          "const auto& s=*state;int i=unifiedActiveNode(s,blockIdx.x*blockDim.x+threadIdx.x);if(!coarseRoot(s,i))return;"
          "const CoarseBlock& b=blocks[i];float h=b.xx+b.yy+b.zz;"
          "if(h>0)printf(\"COARSE %d %d %d %g %g\\n\",s.coarseMode,phase,i,b.scale,h);}" + MARK)
ANCHOR_KERNEL = "static void beginFeasible(){"
BOUND = ("                k_CoarseBound<<<GrAccumulate(g_T),ACCUMULATE_THREADS,0,g_captureStream>>>(d_unified,d_globalCoarse);\n"
         "            if(g_blade.enabled)k_BladeCoarse<<<128,64,0,g_captureStream>>>(d_unified,d_bladeState,d_globalCoarse,true);\n"
         "                if(mode>=1 && mode<=4){")
SELECT = ("                    k_PressureLineSelect<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);\n"
          "                }\n"
          "                k_CoarseApply<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse);\n"
          "            }\n"
          "            xpbd_section_mark(\"loop: coarse modes\");")
path = "xpbd/xpbd_solver.cu"
text = open(path, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in text else "\n"
bound, select = BOUND.replace("\n", nl), SELECT.replace("\n", nl)
probe = "k_DebugCoarseScale<<<Gr(g_activeNodes),64,0,g_captureStream>>>(d_unified,d_globalCoarse,{});" + MARK
if sys.argv[1] == "apply":
    assert MARK not in text and text.count(bound) == 1 and text.count(select) == 1 and text.count(ANCHOR_KERNEL) == 1
    text = text.replace(ANCHOR_KERNEL, KERNEL + nl + ANCHOR_KERNEL)
    text = text.replace(bound, bound.replace("                if(mode>=1 && mode<=4){",
                                             "                " + MARK + probe.format(0) + nl + "                if(mode>=1 && mode<=4){"))
    text = text.replace(select, select.replace("                k_CoarseApply<<<", "                " + MARK + probe.format(1) + nl + "                k_CoarseApply<<<"))
else:
    lines = text.split(nl)
    kept = []
    for line in lines:
        if MARK in line:
            stripped = line.replace(KERNEL, "")
            if stripped.strip() == "" or stripped.strip().startswith(MARK):
                continue
            line = stripped
        kept.append(line)
    text = nl.join(kept)
    assert MARK not in text
open(path, "w", encoding="utf-8", newline="").write(text)
print(sys.argv[1], "done")
