param([switch]$WithBeamNG)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Setup command failed (exit $LASTEXITCODE)." }
}

Invoke-Checked -Executable "py" -Arguments @("-3.12", "-m", "venv", ".venv-win")
$PythonExecutable = Join-Path (Get-Location) ".venv-win\Scripts\python.exe"
$Extras = if ($WithBeamNG) { ".[dev,models,hf,beamng]" } else { ".[dev,models,hf]" }
Invoke-Checked -Executable $PythonExecutable -Arguments @(
    "-m", "pip", "install", "-c", "requirements-dev.lock", "-e", $Extras
)
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
Invoke-Checked -Executable $PythonExecutable -Arguments @("-m", "data_synth", "init-db")
Write-Host "Windows environment ready. Existing .env contents were preserved."
