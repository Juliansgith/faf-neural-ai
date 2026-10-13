@echo off
rem Training dashboard: live, interactive charts of the runs in ..\..\_runs (results, learning, the network,
rem copying the teacher, throughput, every game). Opens http://127.0.0.1:8770/ in your browser; it only reads
rem the run folders, so it is safe next to a training run. Close this window to stop it.
setlocal
set FAF=%~dp0..\..
"%FAF%\ai-venv\Scripts\python.exe" "%~dp0..\tools\dashboard.py" --runs "%FAF%\_runs" --open
pause
