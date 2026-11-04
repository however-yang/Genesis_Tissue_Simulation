@echo off
setlocal
call "D:\develop\Mocrosoft VS\Microsoft Visual Studio\2022 Comumnity\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.29
if errorlevel 1 exit /b 1
"D:\develop\CMake\bin\cmake.exe" -S "%~dp0native" -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="D:/develop/Mocrosoft VS/Microsoft Visual Studio/2022 Comumnity/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe" -DCMAKE_CUDA_COMPILER="D:/develop/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe" -DCMAKE_OBJECT_PATH_MAX=220
if errorlevel 1 exit /b 1
"D:\develop\CMake\bin\cmake.exe" --build "%~dp0build" --target LiverCudaSim --parallel 2
if errorlevel 1 exit /b 1
"D:\develop\anaconda3\envs\genesis_world\python.exe" -X utf8 -B -s "%~dp0record_build.py"
exit /b %errorlevel%
