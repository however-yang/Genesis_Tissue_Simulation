@echo off
setlocal
set "ENDO_OUTPUT_DIR=%~dp0bin"
if not "%~1"=="" set "ENDO_OUTPUT_DIR=%~1"
call "D:\develop\Mocrosoft VS\Microsoft Visual Studio\2022 Comumnity\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cmake -S "%~dp0." -B "%~dp0build" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="D:/develop/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe" -DENDO_BINARY_DIR="%ENDO_OUTPUT_DIR%"
if errorlevel 1 exit /b 1
cmake --build "%~dp0build" --target LiverCudaSim
exit /b %errorlevel%
