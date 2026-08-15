# Quick smoke test for the new kda-t1 net (8-direction KDA scan + local
# conv, tf/configs/kda-t1.textproto). Runs a short, unsupervised daemon
# session straight to a target step -- enough to confirm the new
# architecture trains without crashing and to eyeball real losses/KDA
# stats -- and exits. Not the long-running production launch: that's
# lc0-directml-tui --config tf/configs/kda-t1.textproto --supervise
# --daemon-flag --target-step --daemon-flag <big number>, once this looks
# healthy.

param(
    [int]$TargetStep = 50,
    [string]$LogFile = "train-kda-t1-smoke.log"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

& .\.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_daemon `
    --config tf/configs/kda-t1.textproto `
    --target-step $TargetStep `
    --report-every 5 `
    2>&1 | Tee-Object -FilePath $LogFile
