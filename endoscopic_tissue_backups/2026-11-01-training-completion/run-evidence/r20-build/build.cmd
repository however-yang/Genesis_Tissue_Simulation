@echo off
setlocal
call "D:\develop\Mocrosoft VS\Microsoft Visual Studio\2022 Comumnity\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cmake -S "D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-completion/package/artifacts/release-r20-support-cache/source" -B "D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-completion/package/artifacts/release-r20-support-cache/build" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="D:/develop/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe" -DENDO_BINARY_DIR="D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-completion/package/artifacts/release-r20-support-cache/bin" -DCMAKE_CUDA_FLAGS=""
if errorlevel 1 exit /b 1
cmake --build "D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-completion/package/artifacts/release-r20-support-cache/build" --target LiverCudaSim
exit /b %errorlevel%
