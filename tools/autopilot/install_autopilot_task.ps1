# Registers the RL autopilot's watchdog as a scheduled task (2026-10-17). The user runs this once, by hand: it changes
# the PC's configuration. Every 10 minutes, only while the user is logged on (CUDA and the local games need the
# session), it starts fa_train.autopilot_watchdog with pythonw.exe: no window, no focus change. The watchdog relaunches
# the autopilot (hidden) when its heartbeat is older than 10 minutes; the autopilot starts and supervises rl_loop.
#
#   powershell -ExecutionPolicy Bypass -File tools\autopilot\install_autopilot_task.ps1 [-Config C:\faf\_runs\server\rl2\autopilot\autopilot.json]
#
# Remove:  schtasks /Delete /TN faf-autopilot-watchdog /F
# Pause without removing: create <run>\autopilot\WATCHDOG_OFF.
# Paths default to the workspace layout (the folder holding this repo: C:\faf in the docs).
param(
    [string]$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path,
    [string]$Config = ""
)
if (-not $Config) { $Config = Join-Path $Workspace "_runs\server\rl2\autopilot\autopilot.json" }
$pyw = Join-Path $Workspace "ai-venv\Scripts\pythonw.exe"
$repoPython = Join-Path $PSScriptRoot "..\..\ai\python"
if (-not (Test-Path $pyw)) { throw "pythonw.exe not found: $pyw" }
if (-not (Test-Path $Config)) { throw "no autopilot config: $Config (copy tools\autopilot\autopilot.example.json there)" }
$action = New-ScheduledTaskAction -Execute $pyw -Argument "-m fa_train.autopilot_watchdog --config `"$Config`"" `
    -WorkingDirectory (Resolve-Path $repoPython).Path
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 10)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "faf-autopilot-watchdog" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Relaunches the FAF RL autopilot (hidden) when its heartbeat is stale" -Force
