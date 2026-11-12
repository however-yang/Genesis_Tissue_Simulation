cmake_minimum_required(VERSION 3.24)
project(HYC_movecontrol LANGUAGES CXX CUDA)

# Standalone verification target. This does not build/replace LiverCudaSim.dll.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
add_executable(hyc_movecontrol_check tests/check_movecontrol.cu)
add_executable(hyc_movecontrol_example example.cu)
find_program(HYC_UV uv REQUIRED)
get_filename_component(HYC_REPO "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_custom_target(hyc_scene_inputs
    COMMAND "${HYC_UV}" run --no-sync python -B -m endoscopic_tissue.senario_control.HYC_TEST.prepare_cuda_movecontrol
        --output "${CMAKE_CURRENT_BINARY_DIR}/scene-input.bin"
    BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/scene-input.bin"
    WORKING_DIRECTORY "${HYC_REPO}"
    VERBATIM)
add_dependencies(hyc_movecontrol_check hyc_scene_inputs)
add_dependencies(hyc_movecontrol_example hyc_scene_inputs)
foreach(target hyc_movecontrol_check hyc_movecontrol_example)
    target_compile_features(${target} PRIVATE cxx_std_17 cuda_std_17)
    if(MSVC)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endforeach()
enable_testing()
add_test(NAME hyc_movecontrol_check COMMAND hyc_movecontrol_check "${CMAKE_CURRENT_BINARY_DIR}/scene-input.bin")
