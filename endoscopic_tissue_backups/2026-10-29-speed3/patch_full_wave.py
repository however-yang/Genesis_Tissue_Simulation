"""Size the list-driven sweep and predictor launches to one full wave of resident blocks.

Usage: python patch_full_wave.py  (from endoscopic_tissue/native/vendor/cuda)
"""


def patch(path, edits):
    text = open(path, encoding="utf-8", newline="").read()
    crlf = "\r\n" in text
    for old, new in edits:
        if crlf:
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        assert text.count(old) == 1, (path, old[:80], text.count(old))
        text = text.replace(old, new)
    open(path, "w", encoding="utf-8", newline="").write(text)
    print("ok", path)


patch("solver/material/unified_vertex.cuh", [
("""// Blocks of 64 threads of one colored launch over listed vertices (k_UnifiedVertex, k_UnifiedPredictor), whose warps
// stride over the vertices of a color.
constexpr int SWEEP_BLOCKS=1024;
""", ""),
])
patch("xpbd/xpbd_solver.cu", [
("""// Grid of the kernels that sum per-component terms through blockAccumulate (unified_vertex.cuh).""",
"""// One full wave of 64-thread blocks of a kernel whose warps stride over the vertices k_SweepSites lists. A larger grid
// leaves a partly filled second wave of warps that each walk several vertices; a smaller one leaves SMs idle.
template<class Kernel> static int fullWave(Kernel kernel){
    int device=0,processors=0,blocks=0;cudaGetDevice(&device);
    cudaDeviceGetAttribute(&processors,cudaDevAttrMultiProcessorCount,device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,64,0);return std::max(1,processors*blocks);
}
static int g_sweepBlocks=0,g_predictorBlocks=0;
// Grid of the kernels that sum per-component terms through blockAccumulate (unified_vertex.cuh)."""),
("""    if(e==cudaSuccess && !d_sweepCount)e=cudaMalloc(&d_sweepCount,8*sizeof(int));""",
"""    if(e==cudaSuccess && !d_sweepCount)e=cudaMalloc(&d_sweepCount,8*sizeof(int));
    if(!g_sweepBlocks){g_sweepBlocks=fullWave(k_UnifiedVertex);g_predictorBlocks=fullWave(k_UnifiedPredictor);}"""),
("""                k_UnifiedVertex<<<SWEEP_BLOCKS,64,0,g_captureStream>>>(d_unified,d_sweepSites+c*sites,d_sweepCount+c);""",
"""                k_UnifiedVertex<<<g_sweepBlocks,64,0,g_captureStream>>>(d_unified,d_sweepSites+c*sites,d_sweepCount+c);"""),
("""k_UnifiedPredictor<<<SWEEP_BLOCKS,64>>>(""", """k_UnifiedPredictor<<<g_predictorBlocks,64>>>("""),
("""k_UnifiedVertex<<<SWEEP_BLOCKS,64,0,g_captureStream>>>(d_unified,d_sweepSites+c*sites,d_sweepCount+c);}""",
"""k_UnifiedVertex<<<g_sweepBlocks,64,0,g_captureStream>>>(d_unified,d_sweepSites+c*sites,d_sweepCount+c);}"""),
])
