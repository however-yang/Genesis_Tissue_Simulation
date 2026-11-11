@echo off
setlocal
call "D:\develop\Mocrosoft VS\Microsoft Visual Studio\2022 Comumnity\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
"D:/develop/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe" -shared -O3 -arch=sm_86 "D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-31-interaction-fixes/solve_probe_actual.cu" -o "D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-31-interaction-fixes/solve_probe_actual.dll" -Xcompiler /MD
exit /b %errorlevel%
