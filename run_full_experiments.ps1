param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$PythonExe = "python",
    [switch]$SkipSummary,
    [switch]$SkipAudit
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-TaskDirByScript {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$ScriptName
    )

    $candidate = Get-ChildItem -Path $ProjectRoot -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName $ScriptName) } |
        Select-Object -First 1

    if (-not $candidate) {
        throw ("Cannot find task directory containing script: {0}" -f $ScriptName)
    }
    return $candidate.FullName
}

function Invoke-PythonScript {
    param(
        [Parameter(Mandatory = $true)][string]$WorkDir,
        [Parameter(Mandatory = $true)][string]$Script,
        [Parameter(Mandatory = $true)][string]$Device,
        [Parameter(Mandatory = $true)][string]$PythonExe
    )

    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host ("Running: {0} ({1})" -f $Script, $WorkDir)
    Write-Host ("=" * 72)

    Push-Location $WorkDir
    try {
        & $PythonExe $Script "--device" $Device
        if ($LASTEXITCODE -ne 0) {
            throw ("Script failed with exit code {0}: {1}" -f $LASTEXITCODE, $Script)
        }
    }
    finally {
        Pop-Location
    }
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$sentimentDir = Get-TaskDirByScript -ProjectRoot $ProjectRoot -ScriptName "sentiment_analysis.py"
$reutersDir = Get-TaskDirByScript -ProjectRoot $ProjectRoot -ScriptName "reuters_multiclass.py"
$mtDir = Get-TaskDirByScript -ProjectRoot $ProjectRoot -ScriptName "machine_translation.py"

Invoke-PythonScript -WorkDir $sentimentDir -Script "sentiment_analysis.py" -Device $Device -PythonExe $PythonExe
Invoke-PythonScript -WorkDir $reutersDir -Script "reuters_multiclass.py" -Device $Device -PythonExe $PythonExe
Invoke-PythonScript -WorkDir $mtDir -Script "machine_translation.py" -Device $Device -PythonExe $PythonExe

if (-not $SkipSummary) {
    Write-Host ""
    Write-Host "Generating final summary..."
    & $PythonExe (Join-Path $ProjectRoot "summarize_final_results.py")
    if ($LASTEXITCODE -ne 0) {
        throw ("Summary generation failed with exit code {0}" -f $LASTEXITCODE)
    }
}

if (-not $SkipAudit) {
    Write-Host ""
    Write-Host "Running output audit..."
    & $PythonExe (Join-Path $ProjectRoot "audit_outputs.py")
    if ($LASTEXITCODE -ne 0) {
        throw ("Output audit failed with exit code {0}" -f $LASTEXITCODE)
    }
}

Write-Host ""
Write-Host "All experiments completed."
Write-Host ("Summary: {0}" -f (Join-Path $ProjectRoot "final_results_summary.csv"))
Write-Host ("Audit:   {0}" -f (Join-Path $ProjectRoot "outputs_audit_report.txt"))
