$ErrorActionPreference = "Stop"

param(
    [switch]$SkipTrain,
    [switch]$SkipViz
)

function Step($name, $scriptBlock) {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "[Step] $name" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    & $scriptBlock
    Write-Host "[Done] $name" -ForegroundColor Green
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Python venv not found: $Py"
}

Write-Host "[Info] ProjectRoot: $ProjectRoot" -ForegroundColor Yellow
Write-Host "[Info] Python: $Py" -ForegroundColor Yellow
Write-Host "[Info] Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow

Step "Data Cleaning + KG Build" {
    & $Py "preprocess\clean_data.py"
}

if (-not $SkipTrain) {
    Step "Training + Evaluation" {
        & $Py "utils\train_and_eval.py"
    }
} else {
    Write-Host "[Skip] Training + Evaluation" -ForegroundColor DarkYellow
}

Step "Recommendation Simulation (CPU)" {
    $env:INFER_DEVICE = "cpu"
    & $Py "utils\inference_recommend.py"
}

if (-not $SkipViz) {
    Step "Case Study Visualization" {
        & $Py "utils\case_study_viz.py"
    }

    Step "Render All Charts" {
        & $Py "utils\render_all.py"
    }

    Step "Render Logic Diagram" {
        & $Py "utils\render_model_logic_diagram.py"
    }
} else {
    Write-Host "[Skip] Visualization steps" -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "All pipeline steps finished successfully." -ForegroundColor Green
Write-Host "[Info] End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
