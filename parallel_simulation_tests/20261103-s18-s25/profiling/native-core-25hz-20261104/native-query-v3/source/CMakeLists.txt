cmake_minimum_required(VERSION 3.18)
project(EndoscopicTissueHandoff LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
add_library(LiverCudaSim SHARED
    vendor/src/plugin_api.cpp
    vendor/cuda/training/state_archive.cpp
    vendor/cuda/dc_recon.cu
    vendor/cuda/physics.cu
    vendor/cuda/cut.cu
    vendor/cuda/haptics.cu
    vendor/cuda/xpbd/xpbd_solver.cu)
target_include_directories(LiverCudaSim PRIVATE vendor/cuda)
target_link_libraries(LiverCudaSim PRIVATE CUDA::cudart_static)
if(MSVC)
    target_compile_options(LiverCudaSim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
set_target_properties(LiverCudaSim PROPERTIES
    CUDA_ARCHITECTURES "80"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib")
# Historical Python validators append this basename. The alias points to an
# ELF .so on Linux; it is not a Windows DLL and never occupies a default bin/.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_custom_command(TARGET LiverCudaSim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            "$<TARGET_FILE_NAME:LiverCudaSim>"
            "$<TARGET_FILE_DIR:LiverCudaSim>/LiverCudaSim.dll")
endif()
