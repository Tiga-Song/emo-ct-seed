# Launch one experiment stage as a DETACHED process.
#
# Why detached: a job started inside an assistant session is a child of that
# session and dies when the session ends.  A 6-7 day campaign has to outlive the
# window it was started from, so this uses Start-Process, which survives the
# launching shell.
#
# Resumable: each run writes a DONE sentinel and appends to runs.csv as it
# finishes.  Re-running the same stage with the same --tag skips everything
# already done and continues where it stopped, so an interrupted campaign can
# simply be relaunched.
#
# Usage:
#   .\run_campaign.ps1 -Stage main -Window 10
#   .\run_campaign.ps1 -Stage loso -Window 10
#   .\run_campaign.ps1 -Stage main -Window 10 -DryRun

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('pilot', 'main', 'loso', 'ablation', 'all')]
    [string]$Stage,

    [string]$Window = '10',
    [string]$Python = 'C:\Users\Administer\miniconda3\envs\emoct-seed\python.exe',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$driver = Join-Path $here 'run_experiment.py'
$expRoot = Join-Path $here 'experiments'
New-Item -ItemType Directory -Force -Path $expRoot | Out-Null

$common = "--epochs 300 --patience 25"
$stages = @{
    pilot    = "--protocols cross_session --models emo_ct_cat " +
               "--windows 2.5,5,10,30 --seeds 0 --epochs 100 --patience 15 --tag pilot15"
    main     = "--protocols cross_session,random_933 " +
               "--models ctnet_add,emo_ct_cat,eegnet,shallowconvnet,deepconvnet,conformer " +
               "--windows $Window --seeds 0,1,2 $common --tag main"
    loso     = "--protocols loso " +
               "--models ctnet_add,emo_ct_cat,eegnet,shallowconvnet,deepconvnet,conformer " +
               "--windows $Window --seeds 0 $common --tag loso"
    ablation = "--protocols cross_session " +
               "--models ctnet_add_cnn_only,emo_ct_cat_cnn_only " +
               "--windows $Window --seeds 0,1,2 $common --tag ablation"
}

$stageArgs = $stages[$Stage]
$logOut = Join-Path $expRoot "$Stage`_stdout.log"
$logErr = Join-Path $expRoot "$Stage`_stderr.log"

Write-Host "stage    : $Stage"
Write-Host "driver   : $driver"
Write-Host "arguments: $stageArgs"
Write-Host "stdout   : $logOut"
Write-Host "stderr   : $logErr"

if ($DryRun) {
    Write-Host ''
    Write-Host 'DryRun - nothing launched.'
    exit 0
}

if (-not (Test-Path $driver)) { throw "missing driver: $driver" }
if (-not (Test-Path $Python)) { throw "missing python: $Python" }

# PYTORCH_CUDA_ALLOC_CONF is inherited by the child process
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

if ($Stage -eq 'all') {
    $allScript = Join-Path $here 'run_all.ps1'
    if (-not (Test-Path $allScript)) { throw "missing: $allScript" }
    # Start-Process joins -ArgumentList with spaces WITHOUT quoting, and this
    # project path contains a space ("brain science2"), so the paths must be
    # quoted by hand or powershell.exe receives "-File D:\syl\brain".
    $proc = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass',
                        '-File', ('"' + $allScript + '"'),
                        '-Window', $Window,
                        '-Python', ('"' + $Python + '"')) `
        -WorkingDirectory $here -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $expRoot 'all_stdout.log') `
        -RedirectStandardError (Join-Path $expRoot 'all_stderr.log')
    Write-Host ''
    Write-Host "launched the full campaign (main -> loso -> ablation) detached, PID $($proc.Id)"
    Write-Host "campaign log: $(Join-Path $expRoot 'campaign.log')"
    Write-Host ''
    Write-Host 'Check progress with:'
    Write-Host "  & '$Python' '$(Join-Path $here 'status.py')' --tag main"
    Write-Host "  & '$Python' '$(Join-Path $here 'status.py')' --tag loso"
    Write-Host ''
    Write-Host 'To stop it:'
    Write-Host "  Stop-Process -Id $($proc.Id)"
    exit 0
}

$argString = '"' + $driver + '" ' + $stageArgs
$proc = Start-Process -FilePath $Python -ArgumentList $argString `
    -WorkingDirectory $here -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $logOut -RedirectStandardError $logErr

Write-Host ''
Write-Host "launched detached, PID $($proc.Id)"
Write-Host ''
Write-Host 'This process is independent of the shell that started it.'
Write-Host 'Check progress with:'
Write-Host "  & '$Python' '$(Join-Path $here 'status.py')' --tag $Stage"
Write-Host ''
Write-Host 'To stop it:'
Write-Host "  Stop-Process -Id $($proc.Id)"
