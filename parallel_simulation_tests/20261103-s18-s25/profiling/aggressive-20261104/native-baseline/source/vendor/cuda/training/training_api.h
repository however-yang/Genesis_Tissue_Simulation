#pragma once
#include <cstdint>
// Native force units are centinewtons; conversion belongs to Python.
// Scope: diagnostics of the current unified material-point grip, never legacy pins.
struct TrainingMetrics {
    float maxStretch, rimStretch, graspRingStretch;
    float coreForceMax, coreForceMedian;
    std::uint32_t activeCoreCorners, brokenCoreCorners, brokenGripPoints;
    std::uint32_t tearEdges, materialTearEdges, rimTearEdges, interiorTearEdges, nearGraspTearEdges;
    float minVolumeRatio;
    std::uint32_t invalidTets;
};
static_assert(sizeof(TrainingMetrics)==60,"Training metrics ABI");
extern "C" int training_enable(int flags);
extern "C" int training_read(TrainingMetrics* out);
extern "C" int training_core_forces(int count,float* out);
extern "C" int cut_training_enable();
extern "C" int cut_training_reduce(TrainingMetrics* deviceOut);
extern "C" int cut_training_causes(int count,unsigned char* out);
