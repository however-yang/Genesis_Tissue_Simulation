@echo off
setlocal
set "HYC_DIR=%~dp0"
set "HYC_ARCH=120"
if not "%~1"=="" set "HYC_ARCH=%~1"
set "HYC_CUDA=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if not exist "%HYC_CUDA%\bin\nvcc.exe" set "HYC_CUDA=%CUDA_PATH%"
if not "%~2"=="" set "HYC_CUDA=%~2"
if not exist "%HYC_CUDA%\bin\nvcc.exe" (
    echo CUDA toolkit not found. Pass GPU architecture and toolkit directory.
    exit /b 1
)
set "HYC_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%HYC_VSWHERE%" (
    echo Visual Studio Installer vswhere.exe not found.
    exit /b 1
)
for /f "usebackq tokens=*" %%V in (`"%HYC_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "HYC_VS=%%V"
if not defined HYC_VS (
    echo Visual Studio C++ tools not found.
    exit /b 1
)
call "%HYC_VS%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cmake -S "%HYC_DIR%." -B "%HYC_DIR%build" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="%HYC_CUDA%\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES="%HYC_ARCH%"
if errorlevel 1 exit /b 1
cmake --build "%HYC_DIR%build"
exit /b %errorlevel%
