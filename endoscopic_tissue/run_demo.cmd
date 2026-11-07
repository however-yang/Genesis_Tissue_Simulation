@echo off
setlocal
cd /d "%~dp0.."
if not defined ENDO_PYTHON set "ENDO_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not defined ENDO_DLL set "ENDO_DLL=%~dp0native\bin-sm120\LiverCudaSim.dll"
if not exist "%ENDO_PYTHON%" (
    echo Set ENDO_PYTHON to the Python executable in your Genesis environment.
    pause
    exit /b 1
)
if not exist "%ENDO_DLL%" (
    echo Build native/bin-sm120/LiverCudaSim.dll or set ENDO_DLL to your compiled library.
    pause
    exit /b 1
)
"%ENDO_PYTHON%" -m endoscopic_tissue.run --scene wall-traction --dll "%ENDO_DLL%" %*
set "ENDO_EXIT=%errorlevel%"
if not "%ENDO_EXIT%"=="0" pause
exit /b %ENDO_EXIT%
