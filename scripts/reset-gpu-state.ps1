<#
.SYNOPSIS
    Clear the D3D shader cache before a training run.

.DESCRIPTION
    An empirical finding, not an understood one. After a morning of runs that
    stalled and died, clearing %LOCALAPPDATA%\D3DSCache and resetting the
    display driver produced a run that held 880 ms/step and flat GPU memory
    for 2,000 steps. What is NOT established is which of the two did it, or
    why: the cache is a few megabytes on disk and holds compiled shaders, not
    GPU allocations, so the size alone does not explain a memory symptom.

    The likelier half is the driver reset. Ctrl+Shift+Win+B restarts the
    graphics driver, and that reclaims GPU allocations left behind by
    processes that died badly -- of which this project has produced many. A
    clean exit does return its memory (measured: the adapter drops from
    ~5,100 MB to ~1,200 MB), but a process killed mid-allocation may not.

    So: run this when a run has been crashing, not as a ritual before every
    launch. It costs a few seconds of shader recompilation on the next start.

.PARAMETER WhatIf
    Show what would be removed without removing it.

.EXAMPLE
    .\scripts\reset-gpu-state.ps1
#>
[CmdletBinding(SupportsShouldProcess)]
param()

$ErrorActionPreference = 'Stop'

# Refuse to run while a trainer is up: the cache is in use, and a driver
# reset underneath a live DirectML context is a good way to lose the run
# this is supposed to protect.
$training = Get-CimInstance Win32_Process |
    Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'lczero_training' }
if ($training) {
    Write-Error ("Training is running (PID $($training.ProcessId -join ', ')). " +
                 "Stop it first -- clearing the cache under a live DirectML " +
                 "context risks the run.")
    return
}

$cache = Join-Path $env:LOCALAPPDATA 'D3DSCache'
if (Test-Path $cache) {
    $size = (Get-ChildItem $cache -Recurse -File -ErrorAction SilentlyContinue |
             Measure-Object Length -Sum).Sum
    $mb = [math]::Round(($size / 1MB), 1)
    if ($PSCmdlet.ShouldProcess($cache, "Remove $mb MB of shader cache")) {
        # Contents, not the directory: the driver expects the folder to exist
        # and recreates entries inside it on demand.
        Get-ChildItem $cache -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Cleared $mb MB from $cache"
    }
} else {
    Write-Host "No shader cache at $cache"
}

# The number that actually decides whether a run survives. DirectML allocates
# from the GPU's shared-memory carve-out, capped by WDDM at half of system
# RAM; free system RAM does not extend it. See docs/directml-commands.md.
try {
    $committed = ((Get-Counter "\GPU Adapter Memory(*)\Total Committed" -ErrorAction Stop).CounterSamples |
                  Measure-Object CookedValue -Sum).Sum / 1MB
    $limit = (Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1KB / 2
    "GPU committed {0:N0} MB of ~{1:N0} ({2:N0}%)" -f $committed, $limit, ($committed / $limit * 100)
    if ($committed -gt $limit * 0.4) {
        Write-Warning ("More than 40% of the GPU budget is committed with no " +
                       "trainer running. Something is holding it -- press " +
                       "Ctrl+Shift+Win+B to restart the display driver, then " +
                       "check again.")
    }
} catch {
    Write-Host "GPU memory counters unavailable; skipping the budget check."
}

Write-Host ""
Write-Host "If a run was crashing, also press Ctrl+Shift+Win+B to restart the"
Write-Host "display driver. The screen blanks for a second; nothing is lost."
