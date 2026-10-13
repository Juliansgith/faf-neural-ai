@echo off
rem The night training run (docs/training-design.md, section 15): games at full speed until the stop time,
rem then every game exits and the report is written to _runs\night-<date>\report.md.
rem
rem   night.cmd [HH:MM stop time, default 07:00] [workers, default 20]
rem
rem Stop early: create a file named STOP in the run folder (printed at the start); the run saves its
rem checkpoint and closes all games within seconds. Closing this window also stops it.
setlocal
set STOP_AT=%~1
if "%STOP_AT%"=="" set STOP_AT=07:00
set WORKERS=%~2
if "%WORKERS%"=="" set WORKERS=20
set FAF=%~dp0..\..
cd /d "%~dp0python"
"%FAF%\ai-venv\Scripts\python.exe" -u -m fa_train.night --run "%FAF%\_runs\night-%%Y%%m%%d" --workers %WORKERS% --stop-at %STOP_AT%
pause
