@echo off
call "D:\develop\Mocrosoft VS\Microsoft Visual Studio\2022 Comumnity\VC\Auxiliary\Build\vcvars64.bat" >nul
"D:\develop\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe" -O3 -arch=sm_86 -o D:\Simulation\genesis-world\endoscopic_tissue\outputs\speed2-runs\atomic_bench.exe %~dp0atomic_bench.cu
