param(
    [int]$GrnWorkers = 2,
    [int]$CtxWorkers = 2,
    [int]$AucellWorkers = 2,
    [int]$Seed = 0,
    [switch]$ValidateOnly,
    [switch]$ForceGrn,
    [switch]$ForceCtx,
    [switch]$ForceAucell
)

# Step 10B: run pySCENIC for the combined disease-pipeline LEC states.
#
# Run after Step 10A:
#   powershell -ExecutionPolicy Bypass -File `
#     .\new_script2_disease\10_run_pyscenic_combined_LEC_states.ps1
#
# GRNBoost2 and cisTarget use the donor/dataset-capped expression matrix.
# AUCell uses the all-LEC expression matrix. Completed stages are reused only
# when they are nonempty and newer than every relevant upstream input.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PyScenic = Join-Path $ProjectRoot ".venv_scenic\Scripts\pyscenic.exe"
$InputRoot = Join-Path $ProjectRoot "new_result2_disease\10_SCENIC\input"
$OutputRoot = Join-Path $ProjectRoot "new_result2_disease\10_SCENIC\pyscenic_output"
$Resources = Join-Path $ProjectRoot "resources\scenic"

$NetworkExpression = Join-Path $InputRoot "10_expression_network_balanced_raw_counts.tsv"
$AllExpression = Join-Path $InputRoot "10_expression_all_LECs_raw_counts.tsv"
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

function Test-ValidFreshOutput {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$UpstreamPaths,
        [long]$MinimumBytes = 1000
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    $item = Get-Item -LiteralPath $Path -ErrorAction Stop
    if ($item.Length -lt $MinimumBytes) {
        return $false
    }
    foreach ($upstream in $UpstreamPaths) {
        $source = Get-Item -LiteralPath $upstream -ErrorAction Stop
        if ($item.LastWriteTimeUtc -lt $source.LastWriteTimeUtc) {
            Write-Host "$Path is older than upstream input $upstream." -ForegroundColor Yellow
            return $false
        }
    }
    return $true
}

function Preserve-ReplacedOutput {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backup = "$Path.replaced_$stamp"
        Move-Item -LiteralPath $Path -Destination $backup -ErrorAction Stop
        Write-Host "Preserved previous stage output as: $backup" -ForegroundColor Yellow
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
Assert-InputFile -Path $NetworkExpression -Label "balanced GRN/CTX count matrix"
Assert-InputFile -Path $AllExpression -Label "all-LEC AUCell count matrix"
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
    script_version = "10B_disease_combined_LEC_pySCENIC_balanced_network_2026_08_11"
    project_root = $ProjectRoot
    network_expression = $NetworkExpression
    all_cell_expression = $AllExpression
    tf_list = $TfList
    databases = @($Db10kb, $Db500bp)
    motif_annotations = $MotifAnnotations
    output_root = $OutputRoot
    grn_workers = $GrnWorkers
    ctx_workers = $CtxWorkers
    aucell_workers = $AucellWorkers
    seed = $Seed
    force_grn = [bool]$ForceGrn
    force_ctx = [bool]$ForceCtx
    force_aucell = [bool]$ForceAucell
    started_at = (Get-Date).ToString("o")
}
$config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $RunConfig -Encoding UTF8

Push-Location $ProjectRoot
try {
    $adjUpstream = @($NetworkExpression, $TfList)
    $adjValid = Test-ValidFreshOutput -Path $Adjacencies -UpstreamPaths $adjUpstream
    if ($ForceGrn -or -not $adjValid) {
        Preserve-ReplacedOutput -Path $Adjacencies
        $grnArgs = @(
            "grn", $NetworkExpression, $TfList,
            "--output", $Adjacencies,
            "--method", "grnboost2",
            "--num_workers", "$GrnWorkers",
            "--seed", "$Seed"
        )
        Invoke-PyScenicStage -Stage "GRNBoost2" -Arguments $grnArgs
        if (-not (Test-ValidFreshOutput -Path $Adjacencies -UpstreamPaths $adjUpstream)) {
            throw "GRN completed but adjacencies.tsv is missing, small, or stale."
        }
        $ForceCtx = $true
        $ForceAucell = $true
    }
    else {
        Write-Host "Reusing valid, current adjacencies.tsv." -ForegroundColor Green
    }

    $ctxUpstream = @($Adjacencies, $NetworkExpression, $Db10kb, $Db500bp, $MotifAnnotations)
    $regValid = Test-ValidFreshOutput -Path $Regulons -UpstreamPaths $ctxUpstream
    if ($ForceCtx -or -not $regValid) {
        Preserve-ReplacedOutput -Path $Regulons
        $ctxArgs = @(
            "ctx", $Adjacencies, $Db10kb, $Db500bp,
            "--annotations_fname", $MotifAnnotations,
            "--expression_mtx_fname", $NetworkExpression,
            "--output", $Regulons,
            "--mode", "dask_multiprocessing",
            "--num_workers", "$CtxWorkers"
        )
        Invoke-PyScenicStage -Stage "cisTarget motif pruning" -Arguments $ctxArgs
        if (-not (Test-ValidFreshOutput -Path $Regulons -UpstreamPaths $ctxUpstream)) {
            throw "CTX completed but regulons.csv is missing, small, or stale."
        }
        $ForceAucell = $true
    }
    else {
        Write-Host "Reusing valid, current regulons.csv." -ForegroundColor Green
    }

    $aucUpstream = @($Regulons, $AllExpression)
    $aucValid = Test-ValidFreshOutput -Path $AucMatrix -UpstreamPaths $aucUpstream
    if ($ForceAucell -or -not $aucValid) {
        Preserve-ReplacedOutput -Path $AucMatrix
        $aucArgs = @(
            "aucell", $AllExpression, $Regulons,
            "--output", $AucMatrix,
            "--num_workers", "$AucellWorkers",
            "--seed", "$Seed"
        )
        Invoke-PyScenicStage -Stage "AUCell on all LECs" -Arguments $aucArgs
        if (-not (Test-ValidFreshOutput -Path $AucMatrix -UpstreamPaths $aucUpstream)) {
            throw "AUCell completed but auc_mtx.csv is missing, small, or stale."
        }
    }
    else {
        Write-Host "Reusing valid, current auc_mtx.csv." -ForegroundColor Green
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
Write-Host "Next run: .\.venv\Scripts\python.exe .\new_script2_disease\11_analyze_SCENIC_combined_LEC_states.py"
