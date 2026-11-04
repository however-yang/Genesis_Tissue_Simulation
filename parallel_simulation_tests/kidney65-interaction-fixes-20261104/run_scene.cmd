@echo off
setlocal
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
"D:\develop\anaconda3\envs\genesis_world\python.exe" -X utf8 -B -s "%~dp0run_scene.py" %*
exit /b %errorlevel%
