#pragma once
// CPU fault-injection stub used only by the archive engine unit executable.
// It is never on the native DLL build's include path.
#include <cstddef>
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1,cudaErrorMemoryAllocation=2;
constexpr int cudaErrorInvalidDevicePointer=17,cudaErrorUnknown=999;
enum cudaMemcpyKind{cudaMemcpyHostToHost,cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice};
cudaError_t cudaMalloc(void**,size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMemcpy(void*,const void*,size_t,cudaMemcpyKind);
cudaError_t cudaDeviceSynchronize();
const char* cudaGetErrorString(cudaError_t);
