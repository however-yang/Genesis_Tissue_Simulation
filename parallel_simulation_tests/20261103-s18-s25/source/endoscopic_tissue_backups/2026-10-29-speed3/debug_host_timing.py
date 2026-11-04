"""Add host-side timers (with a device synchronization after each timed call) to a debug build.

Usage: python debug_host_timing.py apply|restore  (from endoscopic_tissue/native/vendor)

Prints one line per step (STEPHOST: begin frame, substeps, ribbon ticks, in-substep rebuilds, final sync), one per
contact-surface refinement that runs a pass (REFINE: passes and the time of each part), and one per capture of the
unified or contact-repair graph (GRAPH). The synchronizations attribute GPU time to the call that queued it, at the
cost of the overlap a normal run has.
"""
import shutil
import sys
from pathlib import Path

backup = Path(r"C:\WINDOWS\TEMP\claude\D--Simulation-genesis-world\6de6da1d-3b43-4fa3-91f1-fd2ef998be2d\scratchpad\debug-host-backup")
files = ["src/plugin_api.cpp", "cuda/physics.cu", "cuda/xpbd/xpbd_solver.cu"]
if sys.argv[1] == "restore":
    for f in files:
        shutil.copyfile(backup / Path(f).name, f)
    print("restored")
    sys.exit(0)
backup.mkdir(parents=True, exist_ok=True)
for f in files:
    shutil.copyfile(f, backup / Path(f).name)


def patch(path, old, new):
    text = open(path, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in text else "\n"
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    assert text.count(old) == 1, (path, old[:60], text.count(old))
    open(path, "w", encoding="utf-8", newline="").write(text.replace(old, new))


TIMER = """
#include <chrono>
#include <cstdio>
static double debugNow(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
"""
# physics.cu: per-step host timeline, and a sync helper for the plugin layer.
patch("cuda/physics.cu", '#include "xpbd/xpbd_solver.h"',
      '#include "xpbd/xpbd_solver.h"' + TIMER + 'extern "C" void debug_device_sync(){cudaDeviceSynchronize();}\n')
patch("cuda/physics.cu", """    xpbd_section_mark("step start");
    int rc = xpbd_begin_frame(dt);
    if (rc != 0) return rc;""", """    xpbd_section_mark("step start");
    double dbgT0=debugNow(),dbgSub=0,dbgTick=0,dbgRebuild=0;int dbgRebuilds=0;
    int rc = xpbd_begin_frame(dt);
    cudaDeviceSynchronize();double dbgBegin=debugNow()-dbgT0;
    if (rc != 0) return rc;""")
patch("cuda/physics.cu", """        rc = xpbd_substep();
        if (rc != 0) return rc;
""", """        double dbgS=debugNow();
        rc = xpbd_substep();
        cudaDeviceSynchronize();dbgSub+=debugNow()-dbgS;
        if (rc != 0) return rc;
""")
patch("cuda/physics.cu", """            r = cut_ribbon_tick();
            if (r != 0 && tickErr == 0) tickErr = r;""", """            double dbgK=debugNow();
            r = cut_ribbon_tick();
            cudaDeviceSynchronize();dbgTick+=debugNow()-dbgK;
            if (r != 0 && tickErr == 0) tickErr = r;""")
patch("cuda/physics.cu", """                r=physics_compute_rotations();if(r)return r;
                r=recon_rebuild();if(r)return r;
                r=recon_prepare_contact_surface();if(r)return r;
                r=xpbd_rebind_contact_surface();if(r)return r;""", """                double dbgR=debugNow();
                r=physics_compute_rotations();if(r)return r;
                r=recon_rebuild();if(r)return r;
                r=recon_prepare_contact_surface();if(r)return r;
                r=xpbd_rebind_contact_surface();if(r)return r;
                cudaDeviceSynchronize();dbgRebuild+=debugNow()-dbgR;dbgRebuilds++;""")
patch("cuda/physics.cu", """    if (le == cudaSuccess) le = cudaDeviceSynchronize();   // commit the frame (replaces the errMax sync)""",
      """    double dbgY=debugNow();
    if (le == cudaSuccess) le = cudaDeviceSynchronize();   // commit the frame (replaces the errMax sync)
    printf("STEPHOST begin %.2f substeps %.2f ticks %.2f rebuild %.2f %d sync %.2f total %.2f\\n",dbgBegin,dbgSub,dbgTick,dbgRebuild,dbgRebuilds,debugNow()-dbgY,debugNow()-dbgT0);fflush(stdout);""")
# plugin_api.cpp: the contact-surface refinement, part by part.
patch("src/plugin_api.cpp", '#define LCS_API extern "C" __declspec(dllexport)',
      TIMER + 'extern "C" void debug_device_sync();\n#define LCS_API extern "C" __declspec(dllexport)')
patch("src/plugin_api.cpp", """    int sync=xpbd_sync_solved_tool();if(sync)return sync;
    float previousDepth=1e30f;int stalled=0;
    for(int i=0;i<passes && i<8;i++){
        float depth=0;int e=LCS_SurfaceDepth(&depth);if(e)return e;if(depth<=xpbd_contact_repair_tolerance())return 0;
        // Volume feasibility can make further geometric repair impossible. Avoid
        // repeating identical GPU solves/reconstruction. Residual depth remains
        // visible to diagnostics; early exit does not certify nonpenetration.
        if(i>0){stalled=depth>=previousDepth*.995f?stalled+1:0;if(stalled>=2)return 0;}
        previousDepth=depth;
        e=recon_prepare_contact_surface();if(e)return e;
        e=xpbd_relax_reconstructed_contact();if(e)return e;
        e=physics_compute_rotations();if(e)return e;
        e=recon_rebuild();if(e)return e;
    }
    return 0;""", """    int sync=xpbd_sync_solved_tool();if(sync)return sync;
    float previousDepth=1e30f;int stalled=0;
    double tDepth=0,tPrepare=0,tRelax=0,tRot=0,tRebuild=0,t0=debugNow();int done=0;float first=-1,last=-1;
    auto report=[&](int code){if(done)printf("REFINE passes %d depth %.4f %.4f measure %.2f prepare %.2f relax %.2f rotations %.2f rebuild %.2f total %.2f\\n",done,first,last,tDepth,tPrepare,tRelax,tRot,tRebuild,debugNow()-t0);fflush(stdout);return code;};
    for(int i=0;i<passes && i<8;i++){
        double t=debugNow();
        float depth=0;int e=LCS_SurfaceDepth(&depth);debug_device_sync();tDepth+=debugNow()-t;if(e)return e;if(first<0)first=depth;last=depth;if(depth<=xpbd_contact_repair_tolerance())return report(0);
        // Volume feasibility can make further geometric repair impossible. Avoid
        // repeating identical GPU solves/reconstruction. Residual depth remains
        // visible to diagnostics; early exit does not certify nonpenetration.
        if(i>0){stalled=depth>=previousDepth*.995f?stalled+1:0;if(stalled>=2)return report(0);}
        previousDepth=depth;done++;
        t=debugNow();e=recon_prepare_contact_surface();debug_device_sync();tPrepare+=debugNow()-t;if(e)return e;
        t=debugNow();e=xpbd_relax_reconstructed_contact();debug_device_sync();tRelax+=debugNow()-t;if(e)return e;
        t=debugNow();e=physics_compute_rotations();debug_device_sync();tRot+=debugNow()-t;if(e)return e;
        t=debugNow();e=recon_rebuild();debug_device_sync();tRebuild+=debugNow()-t;if(e)return e;
    }
    return report(0);""")
# xpbd_solver.cu: graph captures of the unified substep and of the contact repair.
patch("cuda/xpbd/xpbd_solver.cu", "static UnifiedState* d_unified=nullptr;", "static UnifiedState* d_unified=nullptr;" + TIMER)
patch("cuda/xpbd/xpbd_solver.cu", """        if(!g_unifiedContactGraph || g_unifiedContactGraphDirty){
        cudaError_t capture=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(capture!=cudaSuccess)return -(int)capture;""",
      """        double dbgCapture=debugNow();bool dbgCaptured=!g_unifiedContactGraph || g_unifiedContactGraphDirty;
        if(!g_unifiedContactGraph || g_unifiedContactGraphDirty){
        cudaError_t capture=cudaStreamBeginCapture(g_captureStream,cudaStreamCaptureModeThreadLocal);if(capture!=cudaSuccess)return -(int)capture;""")
patch("cuda/xpbd/xpbd_solver.cu", """        if(capture==cudaSuccess && !g_unifiedContactGraph)capture=cudaGraphInstantiate(&g_unifiedContactGraph,graph,0);""",
      """        if(capture==cudaSuccess && !g_unifiedContactGraph)capture=cudaGraphInstantiate(&g_unifiedContactGraph,graph,0);
        if(dbgCaptured){printf("GRAPH contact %.2f\\n",debugNow()-dbgCapture);fflush(stdout);}""")
patch("cuda/xpbd/xpbd_solver.cu", """    if(direct || !selectedGraph || selectedGraphDirty){
        if(direct)g_captureStream=0;""", """    double dbgCapture=debugNow();bool dbgCaptured=!direct && (!selectedGraph || selectedGraphDirty);
    if(direct || !selectedGraph || selectedGraphDirty){
        if(direct)g_captureStream=0;""")
patch("cuda/xpbd/xpbd_solver.cu", """    if(!direct){e=cudaGraphLaunch(selectedGraph,0);if(e!=cudaSuccess)return -(int)e;}""",
      """    if(dbgCaptured){printf("GRAPH unified %.2f\\n",debugNow()-dbgCapture);fflush(stdout);}
    if(!direct){e=cudaGraphLaunch(selectedGraph,0);if(e!=cudaSuccess)return -(int)e;}""")
print("applied")
