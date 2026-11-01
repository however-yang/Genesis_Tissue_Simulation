cmake_minimum_required(VERSION 3.18)
project(EndoscopicTissue LANGUAGES CXX CUDA)
set(ENDO_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin" CACHE PATH "Native library output directory")
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(LiverCudaSim SHARED
    vendor/src/plugin_api.cpp
    vendor/cuda/dc_recon.cu
    vendor/cuda/physics.cu
    vendor/cuda/cut.cu
    vendor/cuda/haptics.cu
    vendor/cuda/xpbd/xpbd_solver.cu)
target_include_directories(LiverCudaSim PRIVATE vendor/cuda)
target_link_libraries(LiverCudaSim PRIVATE CUDA::cudart_static)
target_compile_options(LiverCudaSim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
set_target_properties(LiverCudaSim PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${ENDO_BINARY_DIR}")
