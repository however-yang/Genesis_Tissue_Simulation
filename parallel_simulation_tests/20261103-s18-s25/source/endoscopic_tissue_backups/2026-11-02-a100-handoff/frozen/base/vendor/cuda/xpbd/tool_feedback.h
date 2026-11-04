#pragma once
#include <cstddef>
#include <vector_types.h>

// Stable C ABI. Native simulation units; Unity performs the existing meter conversion.
struct XpbdToolFeedback {
    float3 hinge;
    float gripError;
    int live;
    float jawGap;
    float load[4];
    int graspLocked;
    float3 reaction;
    int contacts;
    float blade[10];
};
static_assert(sizeof(XpbdToolFeedback)==100, "Tool feedback ABI size");
static_assert(offsetof(XpbdToolFeedback,blade)==60, "Tool feedback ABI offset");
extern "C" int xpbd_get_tool_feedback(XpbdToolFeedback* result);
