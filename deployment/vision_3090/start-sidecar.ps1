$ErrorActionPreference = "Stop"
# Start the Reachy perception sidecar on a Windows CUDA host.
# Prerequisites: Python 3.12, CUDA torch in .venv-vision, repo sources under $Repo.

$Repo = if ($env:REACHY_PA_REPO) { $env:REACHY_PA_REPO } else { (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$Data = if ($env:REACHY_VISION_DATA) { $env:REACHY_VISION_DATA } else { Join-Path $env:LOCALAPPDATA "reachy-vision-sidecar" }
$EnvFile = Join-Path $Data ".env"
$Python = Join-Path $Repo ".venv-vision\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  throw "Missing $Python - create .venv-vision and install requirements first."
}
if (-not (Test-Path $EnvFile)) {
  throw "Missing $EnvFile - copy deployment\vision_3090\env.example and set VISION_SIDECAR_TOKEN."
}

New-Item -ItemType Directory -Force -Path $Data | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Data "vision\hf") | Out-Null

Get-Content $EnvFile | ForEach-Object {
  $line = $_.Trim()
  if (-not $line -or $line.StartsWith("#")) { return }
  $parts = $line.Split("=", 2)
  if ($parts.Length -eq 2) {
    [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
  }
}

$env:PYTHONPATH = $Repo
$env:PYTHONUNBUFFERED = "1"
Set-Location $Repo

Write-Output "Starting vision sidecar on $($env:VISION_SIDECAR_HOST):$($env:VISION_SIDECAR_PORT)"
& $Python -m vision_sidecar --host $env:VISION_SIDECAR_HOST --port ([int]$env:VISION_SIDECAR_PORT)
