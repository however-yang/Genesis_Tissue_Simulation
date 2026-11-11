@echo off
setlocal
cd /d D:\Simulation\genesis-world
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-grasp-cut-validation\render_review.py endoscopic_tissue\outputs\release-next-20261101\r24-kidney > endoscopic_tissue_backups\2026-11-01-release-next\r24-render.log 2>&1
if errorlevel 1 exit /b %errorlevel%
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-interaction-fixes\release_frame_sheets.py endoscopic_tissue\outputs\release-next-20261101\r24-kidney > endoscopic_tissue_backups\2026-11-01-release-next\r24-release-sheets.log 2>&1
if errorlevel 1 exit /b %errorlevel%
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-grasp-cut-validation\render_review.py endoscopic_tissue\outputs\release-next-20261101\r25-kidney > endoscopic_tissue_backups\2026-11-01-release-next\r25-render.log 2>&1
if errorlevel 1 exit /b %errorlevel%
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-interaction-fixes\release_frame_sheets.py endoscopic_tissue\outputs\release-next-20261101\r25-kidney > endoscopic_tissue_backups\2026-11-01-release-next\r25-release-sheets.log 2>&1
if errorlevel 1 exit /b %errorlevel%
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-11-01-release-next\run_r22_timing.py --revision r25 --include-r24-screen > endoscopic_tissue_backups\2026-11-01-release-next\r25-timing.log 2>&1
exit /b %errorlevel%
