#pragma once
// Shared by the constitutive solver and both continuous cutting detectors.
// CutExit still consumes the final accepted sweep; ClearStroke never does.
enum PressureCutPhase { BladePressing=0, BladeCutting=1, BladeWithdrawing=2, BladeCutExit=3, BladeClearStroke=4 };
__device__ inline bool bladeCommits(int phase){return phase==BladeCutting || phase==BladeCutExit;}
__device__ inline bool bladeDetects(int phase){return phase==BladePressing || bladeCommits(phase);}
__device__ inline bool bladeYields(int phase){return bladeCommits(phase) || phase==BladeWithdrawing;}
