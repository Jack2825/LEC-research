param(
    [int]$GrnWorkers = 4,
    [int]$CtxWorkers = 2,
    [int]$AucellWorkers = 2,
    [int]$Seed = 0,
    [switch]$ValidateOnly,
    [switch]$ForceGrn,
    [switch]$ForceCtx,
    [switch]$ForceAucell
)

# Step 10B: run pySCENIC for the combined aging LEC states.
#
# Run after Step 10A:
#   powershell -ExecutionPolicy Bypass -File `
#     .\new_script2_aging\10_run_pyscenic_combined_LEC_states.ps1
#
# The script uses the dedicated Python 3.10 .venv_scenic environment. Valid
# completed stages are reused. If a stage is forced or an output is invalid,
# the old file is renamed with an .invalid_or_replaced timestamp rather than
# deleted.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PyScenic = Join-Path $ProjectRoot ".venv_scenic\Scripts\pyscenic.exe"
$InputRoot = Join-Path $ProjectRoot "new_result2_aging\10_SCENIC\dataset13\input"
$OutputRoot = Join-Path $ProjectRoot "new_result2_aging\10_SCENIC\dataset13\pyscenic_output"
$Resources = Join-Path $ProjectRoot "resources\scenic"

$Expression = Join-Path $InputRoot "10_expression_raw_counts_for_pyscenic.tsv"
$TfList = Join-Path $Resources "allTFs_hg38.txt"
$Db10kb = Join-Path $Resources "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
$Db500bp = Join-Path $Resources "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
$MotifAnnotations = Join-Path $Resources "motifs-v10-nr.hgnc-m0.00001-o0.0.tbl"

$Adjacencies = Join-Path $OutputRoot "adjacencies.tsv"
$Regulons = Join-Path $OutputRoot "regulons.csv"
$AucMatrix = Join-Path $OutputRoot "auc_mtx.csv"
$RunConfig = Join-Path $OutputRoot "10_pyscenic_run_config.json"

function Assert-InputFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing $Label file: $Path"
    }
    $item = Get-Item -LiteralPath $Path -ErrorAction Stop
    if ($item.Length -le 0) {
        throw "$Label file is empty: $Path"
    }
    Write-Host ("OK: {0} ({1:N0} bytes)" -f $Label, $item.Length) -ForegroundColor Green
}

function Test-ValidOutput {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [long]$MinimumBytes = 1000
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    $item = Get-Item -LiteralPath $Path -ErrorAction Stop
    return ($item.Length -ge $MinimumBytes)
}

function Move-ExistingToBackup {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backup = "$Path.invalid_or_replaced_$stamp"
        Move-Item -LiteralPath $Path -Destination $backup -ErrorAction Stop
        Write-Host "Preserved previous output as: $backup" -ForegroundColor Yellow
    }
}

function Invoke-PyScenicStage {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host "Starting pySCENIC stage: $Stage" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
    & $PyScenic @Arguments
    $code = $LASTEXITCODE
    Write-Host "$Stage exit code: $code"
    if ($code -ne 0) {
        throw "pySCENIC $Stage failed with exit code $code."
    }
}

if ($GrnWorkers -lt 1 -or $CtxWorkers -lt 1 -or $AucellWorkers -lt 1) {
    throw "All worker counts must be at least 1."
}

Write-Host "Validating Step 10B inputs..." -ForegroundColor Cyan
Assert-InputFile -Path $PyScenic -Label "pySCENIC executable"
Assert-InputFile -Path $Expression -Label "raw-count expression matrix"
Assert-InputFile -Path $TfList -Label "human TF list"
Assert-InputFile -Path $Db10kb -Label "10 kb cisTarget database"
Assert-InputFile -Path $Db500bp -Label "500 bp cisTarget database"
Assert-InputFile -Path $MotifAnnotations -Label "motif annotation table"

if ($ValidateOnly) {
    Write-Host "Validation completed; no pySCENIC stage was started." -ForegroundColor Green
    return
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$config = [ordered]@{
    script_version = "10B_dataset13_NF_combined_LEC_pySCENIC_2026_08_07"
    project_root = $ProjectRoot
    expression = $Expression
    tf_list = $TfList
    databases = @($Db10kb, $Db500bp)
    motif_annotations = $MotifAnnotations
    output_root = $OutputRoot
    grn_workers = $GrnWorkers
    ctx_workers = $CtxWorkers
    aucell_workers = $AucellWorkers
    seed = $Seed
    validate_only = [bool]$ValidateOnly
    force_grn = [bool]$ForceGrn
    force_ctx = [bool]$ForceCtx
    force_aucell = [bool]$ForceAucell
    started_at = (Get-Date).ToString("o")
}
$config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $RunConfig -Encoding UTF8

Push-Location $ProjectRoot
try {
    $adjValid = Test-ValidOutput -Path $Adjacencies
    if ($ForceGrn -or -not $adjValid) {
        Move-ExistingToBackup -Path $Adjacencies
        $grnArgs = @(
            "grn", $Expression, $TfList,
            "--output", $Adjacencies,
            "--method", "grnboost2",
            "--num_workers", "$GrnWorkers",
            "--seed", "$Seed"
        )
        Invoke-PyScenicStage -Stage "GRNBoost2" -Arguments $grnArgs
        if (-not (Test-ValidOutput -Path $Adjacencies)) {
            throw "GRN completed but adjacencies.tsv is missing or too small: $Adjacencies"
        }
    }
    else {
        Write-Host "Reusing valid adjacencies.tsv." -ForegroundColor Green
    }

    $regValid = Test-ValidOutput -Path $Regulons
    if ($ForceCtx -or -not $regValid) {
        Move-ExistingToBackup -Path $Regulons
        $ctxArgs = @(
            "ctx", $Adjacencies, $Db10kb, $Db500bp,
            "--annotations_fname", $MotifAnnotations,
            "--expression_mtx_fname", $Expression,
            "--output", $Regulons,
            "--mode", "dask_multiprocessing",
            "--num_workers", "$CtxWorkers"
        )
        Invoke-PyScenicStage -Stage "cisTarget motif pruning" -Arguments $ctxArgs
        if (-not (Test-ValidOutput -Path $Regulons)) {
            throw "CTX completed but regulons.csv is missing or too small: $Regulons"
        }
    }
    else {
        Write-Host "Reusing valid regulons.csv." -ForegroundColor Green
    }

    $aucValid = Test-ValidOutput -Path $AucMatrix
    if ($ForceAucell -or -not $aucValid) {
        Move-ExistingToBackup -Path $AucMatrix
        $aucArgs = @(
            "aucell", $Expression, $Regulons,
            "--output", $AucMatrix,
            "--num_workers", "$AucellWorkers",
            "--seed", "$Seed"
        )
        Invoke-PyScenicStage -Stage "AUCell" -Arguments $aucArgs
        if (-not (Test-ValidOutput -Path $AucMatrix)) {
            throw "AUCell completed but auc_mtx.csv is missing or too small: $AucMatrix"
        }
    }
    else {
        Write-Host "Reusing valid auc_mtx.csv." -ForegroundColor Green
    }
}
finally {
    Pop-Location
}

$config.completed_at = (Get-Date).ToString("o")
$config.adjacencies_bytes = (Get-Item -LiteralPath $Adjacencies).Length
$config.regulons_bytes = (Get-Item -LiteralPath $Regulons).Length
$config.aucell_bytes = (Get-Item -LiteralPath $AucMatrix).Length
$config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $RunConfig -Encoding UTF8

Write-Host ""
Write-Host "pySCENIC Step 10B completed successfully." -ForegroundColor Green
Get-Item -LiteralPath $Adjacencies, $Regulons, $AucMatrix |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -AutoSize
Write-Host "Next run: .\.venv\Scripts\python.exe .\new_script2_aging\11_analyze_SCENIC_combined_LEC_states.py"
