@echo off
setlocal
cd /d "%~dp0.."
if not defined ENDO_PYTHON set "ENDO_PYTHON=D:\develop\anaconda3\envs\genesis_world\python.exe"
if not exist "%ENDO_PYTHON%" (
    echo Set ENDO_PYTHON to the Python executable in your Genesis environment.
    exit /b 1
)
"%ENDO_PYTHON%" -m endoscopic_tissue.run %*
set "ENDO_EXIT=%errorlevel%"
if not "%ENDO_EXIT%"=="0" pause
exit /b %ENDO_EXIT%
