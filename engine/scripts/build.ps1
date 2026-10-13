<#
.SYNOPSIS
Builds faf-re's main.exe and, if the build is clean, copies it next to the FAF game data.

.DESCRIPTION
The project links with /FORCE, so MSBuild "succeeds" even with unresolved symbols; this script checks
the build log for compile and link errors and refuses to deploy a binary that has any. The project's
own post-build staging is disabled (FafRunDir points nowhere) so only a clean build is deployed.

Needs Visual Studio 2022 (Build Tools) with the v143 toolset, and a faf-re checkout whose dependencies
are set up (see faf-re's README; the DirectX SDK is looked up in <faf-re>\.depcache\dx\DXSDK).

.EXAMPLE
./build.ps1
./build.ps1 -Configuration Debug -NoDeploy
#>
[CmdletBinding()]
param(
    [string]$FafRe,
    [ValidateSet('Release', 'Debug')][string]$Configuration = 'Release',
    [string]$DeployTo = 'C:\ProgramData\FAForever\bin',
    [string]$DirectXSdk,
    [string]$EngineVersion,
    [switch]$NoDeploy
)
$ErrorActionPreference = 'Stop'
if (-not $FafRe) { $FafRe = Join-Path $PSScriptRoot '..\..\..\faf-re-dev' }
$FafRe = (Resolve-Path $FafRe).Path
if (-not $DirectXSdk) { $DirectXSdk = Join-Path $FafRe '.depcache\dx\DXSDK' }

# The engine must report the version of the FAF data it runs with (replay headers, GetVersion).
# The FAF client records it as GameVersion in <FAForever>\fa_path.lua.
if (-not $EngineVersion) {
    $faPath = Join-Path (Split-Path $DeployTo -Parent) 'fa_path.lua'
    if (Test-Path $faPath) {
        $match = Select-String -Path $faPath -Pattern 'GameVersion\s*=\s*"(\d+)"' | Select-Object -First 1
        if ($match) { $EngineVersion = $match.Matches[0].Groups[1].Value }
    }
}
$extraArgs = @()
if ($EngineVersion) {
    $extraArgs += "/p:FafEngineVersion=$EngineVersion"
    Write-Host "engine version $EngineVersion"
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$msbuild = & $vswhere -products * -latest -requires Microsoft.Component.MSBuild -find 'MSBuild\**\Bin\MSBuild.exe' | Select-Object -First 1
if (-not $msbuild) { throw "MSBuild not found (install Visual Studio 2022 Build Tools)" }

$logDir = Join-Path $FafRe 'buildstage'
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "build-$Configuration.log"
$env:_CL_ = "/MP$([Environment]::ProcessorCount)"
& $msbuild (Join-Path $FafRe 'src\sdk\main.vcxproj') /t:Build "/p:Configuration=$Configuration" /p:Platform=Win32 `
    "/p:DirectXInclude=$DirectXSdk\Include" "/p:DirectXLibX86=$DirectXSdk\Lib\x86" `
    "/p:FafRunDir=$(Join-Path $FafRe 'buildstage\no-staging')\" @extraArgs /m /nologo /v:m /fl "/flp:logfile=$log;verbosity=normal"

$errors = @(Select-String -Path $log -Pattern '(: error |: fatal error |error LNK)')
if ($errors.Count -gt 0) {
    $errors | Select-Object -First 30 | ForEach-Object { Write-Host $_.Line.Trim() }
    throw "$($errors.Count) build errors (log: $log)"
}
$out = Join-Path $FafRe "output\main\Win32\$Configuration"
Write-Host "built $out\main.exe"
if (-not $NoDeploy) {
    Copy-Item "$out\main.exe", "$out\main.pdb" $DeployTo -Force
    Write-Host "deployed to $DeployTo"
}
