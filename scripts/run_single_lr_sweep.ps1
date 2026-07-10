param(
    [int]$Epochs = 200,
    [int]$NumTrain = 10000,
    [int]$NumVal = 2000,
    [int]$BatchSize = 64,
    [int]$Hidden = 64,
    [int]$HiddenComplex = 32,
    [int]$SingleZeroComplex = 24,
    [int]$SingleBranchLayers = 1,
    [string]$Device = "cuda",
    [string]$RootDir = "runs/lr_sweep_single",
    [switch]$RunNoInterRef
)

$ErrorActionPreference = "Stop"

function Invoke-And-Log {
    param(
        [string]$Command,
        [string]$LogPath
    )
    Write-Host "`n>>> $Command"
    $tmpLog = "$LogPath.tmp"
    cmd.exe /d /s /c "$Command > `"$tmpLog`" 2>&1"
    $exitCode = $LASTEXITCODE
    Get-Content $tmpLog | Tee-Object -FilePath $LogPath
    Remove-Item -LiteralPath $tmpLog -Force
    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode. See $LogPath"
    }
}

function Get-BestValFromLog {
    param([string]$LogPath)
    $bestLoss = $null
    $bestBer = $null
    Get-Content $LogPath | ForEach-Object {
        if ($_ -match 'val loss ([0-9.]+) \| val BER ([0-9.]+)') {
            $loss = [double]$Matches[1]
            $ber = [double]$Matches[2]
            if ($null -eq $bestLoss -or $loss -lt $bestLoss) {
                $bestLoss = $loss
                $bestBer = $ber
            }
        }
    }
    return @($bestLoss, $bestBer)
}

New-Item -ItemType Directory -Force -Path $RootDir | Out-Null
$summary = @()

$lrs = @("3e-4", "5e-4", "1e-3", "2e-3")
$weightDecays = @("0", "1e-4")

foreach ($lr in $lrs) {
    foreach ($wd in $weightDecays) {
        $tag = "single_z${SingleZeroComplex}_lr$($lr.Replace('-', 'm'))_wd$($wd.Replace('-', 'm'))"
        $saveDir = Join-Path $RootDir $tag
        $trainLog = Join-Path $saveDir "train.log"
        $evalLog = Join-Path $saveDir "eval_ber.log"
        $evalCsv = Join-Path $saveDir "ber.csv"
        New-Item -ItemType Directory -Force -Path $saveDir | Out-Null

        $trainCmd = "python train.py --model single_branch --hidden $Hidden --hidden_complex $HiddenComplex --zero_complex $SingleZeroComplex --branch_layers $SingleBranchLayers --single_readout_mode low_rank --gate_type swiglu --h_hat_mode dmrs_ls_interp --dmrs_freq_spacing 1 --num_train $NumTrain --num_val $NumVal --epochs $Epochs --batch_size $BatchSize --snr_db_min -5 --snr_db_max 20 --lr $lr --weight_decay $wd --save_dir `"$saveDir`" --device $Device"
        Invoke-And-Log -Command $trainCmd -LogPath $trainLog

        $evalCmd = "python eval_ber.py --model single_branch --hidden $Hidden --hidden_complex $HiddenComplex --zero_complex $SingleZeroComplex --branch_layers $SingleBranchLayers --single_readout_mode low_rank --checkpoint `"$saveDir/best.pt`" --out_csv `"$evalCsv`" --device $Device"
        Invoke-And-Log -Command $evalCmd -LogPath $evalLog

        $best = Get-BestValFromLog -LogPath $trainLog
        $summary += [pscustomobject]@{
            model = "single_branch"
            zero_complex = $SingleZeroComplex
            branch_layers = $SingleBranchLayers
            lr = $lr
            weight_decay = $wd
            best_val_loss = $best[0]
            best_val_ber = $best[1]
            save_dir = $saveDir
            eval_csv = $evalCsv
        }
    }
}

if ($RunNoInterRef) {
    $tag = "nointer_ref_lr1e_m3_wd0"
    $saveDir = Join-Path $RootDir $tag
    $trainLog = Join-Path $saveDir "train.log"
    $evalLog = Join-Path $saveDir "eval_ber.log"
    $evalCsv = Join-Path $saveDir "ber.csv"
    New-Item -ItemType Directory -Force -Path $saveDir | Out-Null

    $trainCmd = "python train.py --model complex_no_interaction --hidden $Hidden --hidden_complex $HiddenComplex --zero_complex 32 --branch_layers 3 --gate_type swiglu --h_hat_mode dmrs_ls_interp --dmrs_freq_spacing 1 --num_train $NumTrain --num_val $NumVal --epochs $Epochs --batch_size $BatchSize --snr_db_min -5 --snr_db_max 20 --lr 1e-3 --weight_decay 0 --save_dir `"$saveDir`" --device $Device"
    Invoke-And-Log -Command $trainCmd -LogPath $trainLog

    $evalCmd = "python eval_ber.py --model complex_no_interaction --hidden $Hidden --hidden_complex $HiddenComplex --zero_complex 32 --branch_layers 3 --checkpoint `"$saveDir/best.pt`" --out_csv `"$evalCsv`" --device $Device"
    Invoke-And-Log -Command $evalCmd -LogPath $evalLog

    $best = Get-BestValFromLog -LogPath $trainLog
    $summary += [pscustomobject]@{
        model = "complex_no_interaction"
        zero_complex = 32
        branch_layers = 3
        lr = "1e-3"
        weight_decay = "0"
        best_val_loss = $best[0]
        best_val_ber = $best[1]
        save_dir = $saveDir
        eval_csv = $evalCsv
    }
}

$summaryPath = Join-Path $RootDir "summary.csv"
$summary | Sort-Object best_val_ber | Export-Csv -NoTypeInformation -Encoding UTF8 $summaryPath
Write-Host "`nSaved summary to $summaryPath"
$summary | Sort-Object best_val_ber | Format-Table -AutoSize
