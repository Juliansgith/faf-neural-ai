@echo off
rem The tools-version night run (docs/tools-design.md): teacher games are played and recorded, the tools
rem network is cloned from all recordings, then it learns from its own games until the stop time.
rem Report: _runs\tools-<date>\report.md
rem
rem   night_tools.cmd [HH:MM stop time, default 12:00] [workers, default 20] [game minutes, default 30]
rem                   [teacher games, default 200] [more options for fa_train.night_tools ...]
rem
rem Stop early: create a file named STOP in the run folder (printed at the start). Closing this window
rem also stops it.
setlocal
set STOP_AT=%~1
if "%STOP_AT%"=="" set STOP_AT=12:00
set WORKERS=%~2
if "%WORKERS%"=="" set WORKERS=20
set MINUTES=%~3
if "%MINUTES%"=="" set MINUTES=30
set TEACHER=%~4
if "%TEACHER%"=="" set TEACHER=200
set MORE=%5 %6 %7 %8 %9
set FAF=%~dp0..\..
cd /d "%~dp0python"
"%FAF%\ai-venv\Scripts\python.exe" -u -m fa_train.night_tools --run "%FAF%\_runs\tools-%%Y%%m%%d" --workers %WORKERS% --stop-at %STOP_AT% --minutes %MINUTES% --collect-games %TEACHER% %MORE%
pause
