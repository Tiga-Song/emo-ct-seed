# Run the whole campaign: main -> loso -> ablation, sequentially, in one process.
#
# Each stage is resumable (DONE sentinels + runs.csv ledger), so if this is
# interrupted it can simply be started again and it will pick up where it left
# off.  Stages run one after another rather than concurrently because a single
# GPU would just make them contend.
#
# Usage:  .\run_all.ps1 -Window 5

[CmdletBinding()]
param(
    [string]$Window = '5',
    [string]$Python = 'C:\Users\Administer\miniconda3\envs\emoct-seed\python.exe'
)

$ErrorActionPreference = 'Continue'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$driver = Join-Path $here 'run_experiment.py'
$expRoot = Join-Path $here 'experiments'
New-Item -ItemType Directory -Force -Path $expRoot | Out-Null
$campaignLog = Join-Path $expRoot 'campaign.log'

$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

function Say($message) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $message
    Write-Host $line
    Add-Content -Path $campaignLog -Value $line -Encoding utf8
}

$common = '--epochs 300 --patience 25'
$stages = @(
    @{ name = 'main'
       args = "--protocols cross_session,random_933 --models " +
              "ctnet_add,emo_ct_cat,eegnet,shallowconvnet,deepconvnet,conformer " +
              "--windows $Window --seeds 0,1,2 $common --tag main" },
    @{ name = 'loso'
       args = "--protocols loso --models " +
              "ctnet_add,emo_ct_cat,eegnet,shallowconvnet,deepconvnet,conformer " +
              "--windows $Window --seeds 0 $common --tag loso" },
    @{ name = 'ablation'
       args = "--protocols cross_session --models ctnet_add_cnn_only,emo_ct_cat_cnn_only " +
              "--windows $Window --seeds 0,1,2 $common --tag ablation" }
)

Say "campaign start (window = $Window s)"
$startedAll = Get-Date

foreach ($stage in $stages) {
    Say "--- stage '$($stage.name)' starting ---"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $stageLog = Join-Path $expRoot "$($stage.name)_run.log"

    & $Python $driver ($stage.args -split ' ') *>> $stageLog
    $code = $LASTEXITCODE

    $sw.Stop()
    Say ("--- stage '{0}' finished, exit {1}, {2:N1} min ---" -f
         $stage.name, $code, $sw.Elapsed.TotalMinutes)
}

$total = (Get-Date) - $startedAll
Say ("campaign complete, total {0:N1} h" -f $total.TotalHours)
