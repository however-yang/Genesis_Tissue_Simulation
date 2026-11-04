"""Reconstruct a hashed experiment, adapting only C exports and CMake for Linux.

No CUDA computation, default source, or default binary is modified. Run from a
clean Git checkout; generated sources are deliberately outside the native tree.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = ROOT / 'endoscopic_tissue/outputs/a100-migration'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reconstruct(name, manifest):
    variant = manifest['variants'][name]
    files = reconstruct(variant['parent'], manifest) if variant['parent'] else {}
    for relative, stored in variant['overlay'].items():
        file = (HERE / stored).resolve()
        if not file.is_relative_to(HERE / 'frozen'):
            raise ValueError('Source escapes frozen package: ' + stored)
        files[relative] = file.read_bytes()
    actual = {relative: sha(data) for relative, data in files.items()}
    if actual != variant['files']:
        raise ValueError('Frozen source SHA mismatch for ' + name)
    return files


def portable_sources(files, arch):
    files = dict(files)
    # Boundary candidates also export diagnostic C functions directly from .cu.
    export_files = {'vendor/src/plugin_api.cpp': 1}
    if b'LCS_ReadFrozenFeatures' in files['vendor/cuda/dc_recon.cu']:
        export_files.update({'vendor/cuda/dc_recon.cu': 2, 'vendor/cuda/cut.cu': 3})
    old = b'__declspec(dllexport)'
    prefix = (b'#ifdef _WIN32\n#define LCS_HANDOFF_EXPORT __declspec(dllexport)\n'
              b'#else\n#define LCS_HANDOFF_EXPORT __attribute__((visibility("default")))\n#endif\n')
    for relative, data in list(files.items()):
        if data.count(old) != export_files.get(relative, 0):
            raise ValueError('Unexpected C ABI exports: ' + relative)
        if relative in export_files:
            files[relative] = prefix + data.replace(old, b'LCS_HANDOFF_EXPORT')
    snapshot = '    vendor/cuda/training/state_archive.cpp\n' if (
        'vendor/cuda/training/state_archive.cpp' in files) else ''
    files['CMakeLists.txt'] = ('''cmake_minimum_required(VERSION 3.18)
project(EndoscopicTissueHandoff LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
add_library(LiverCudaSim SHARED
    vendor/src/plugin_api.cpp
''' + snapshot + '''    vendor/cuda/dc_recon.cu
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
    CUDA_ARCHITECTURES "''' + str(arch) + '''"
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
''').encode('utf8')
    return files


def main():
    manifest = json.loads((HERE / 'source-manifest.json').read_text(encoding='utf8'))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=manifest['variants'], required=True)
    p.add_argument('--arch', type=int, choices=[80, 86], default=80)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    frozen = reconstruct(args.variant, manifest)
    adapted = portable_sources(frozen, args.arch)
    changes = [key for key in frozen if frozen[key] != adapted[key]]
    expected = {'CMakeLists.txt', 'vendor/src/plugin_api.cpp'}
    if args.variant in ('boundary-snapshot2', 'repair-feasible-snapshot3'):
        expected.update({'vendor/cuda/cut.cu', 'vendor/cuda/dc_recon.cu'})
    if set(changes) != expected:
        raise ValueError('Unexpected portability edits')
    report = dict(variant=args.variant, cuda_arch=args.arch, files=len(frozen),
                  source_manifest_sha256=sha((HERE / 'source-manifest.json').read_bytes()),
                  frozen_sha256={k: sha(v) for k, v in frozen.items()},
                  prepared_sha256={k: sha(v) for k, v in adapted.items()},
                  adapted_files=changes, default_replaced=False,
                  compiled=False, linux_runtime_verified=False, physics_accepted=False)
    if not args.verify_only:
        target = OUTPUT / args.variant / 'source'
        target.mkdir(parents=True, exist_ok=False)
        for relative, data in adapted.items():
            dest = (target / relative).resolve()
            if not dest.is_relative_to(target.resolve()):
                raise ValueError('Source escapes output: ' + relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        (target.parent / 'prepare-result.json').write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf8')
        print('Prepared:', target)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('_sha256')}, indent=2))


if __name__ == '__main__':
    main()
