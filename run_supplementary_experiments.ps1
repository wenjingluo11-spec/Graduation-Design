param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$PythonExe = "python",
    [switch]$Fast,
    [switch]$All,
    [switch]$RunStability,
    [switch]$RunAblation,
    [switch]$RunErrorAnalysis,
    [switch]$RunEfficiency,
    [switch]$RunDataScale
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptPath = Join-Path $ProjectRoot "supplementary_experiments.py"

$argsList = @($ScriptPath, "--device", $Device)
if ($Fast) { $argsList += "--fast" }
if ($All) { $argsList += "--all" }
if ($RunStability) { $argsList += "--run-stability" }
if ($RunAblation) { $argsList += "--run-ablation" }
if ($RunErrorAnalysis) { $argsList += "--run-error-analysis" }
if ($RunEfficiency) { $argsList += "--run-efficiency" }
if ($RunDataScale) { $argsList += "--run-data-scale" }

if (-not ($All -or $RunStability -or $RunAblation -or $RunErrorAnalysis -or $RunEfficiency -or $RunDataScale)) {
    throw "Please select at least one section switch, e.g. -All or -RunStability."
}

Write-Host "Running supplementary experiments..."
Push-Location $ProjectRoot
try {
    & $PythonExe @argsList
    if ($LASTEXITCODE -ne 0) {
        throw ("supplementary_experiments.py failed with exit code {0}" -f $LASTEXITCODE)
    }
}
finally {
    Pop-Location
}

Write-Host "Done."
Write-Host ("Outputs: {0}" -f (Join-Path $ProjectRoot "supplementary_outputs"))
