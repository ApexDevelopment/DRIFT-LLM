# DRIFT-LLM one-line installer for Windows (PowerShell).
#
#   irm https://raw.githubusercontent.com/ApexDevelopment/petals/main/scripts/install.ps1 | iex
#
# Detects your accelerator, builds the patched hivemind wheel (PyPI ships none for
# Windows), installs a matching PyTorch build into a local .venv, and installs
# DRIFT-LLM (the `petals` package). Override the accelerator with
# $env:DRIFT_DEVICE = 'cpu' | 'cuda' | 'xpu' before running.
#
# Requires uv (https://docs.astral.sh/uv/) and Go (https://go.dev/dl/, for the
# hivemind wheel build). Linux/macOS users: use scripts/install.sh instead.
$ErrorActionPreference = 'Stop'

$RepoUrl   = if ($env:DRIFT_REPO_URL) { $env:DRIFT_REPO_URL } else { 'https://github.com/ApexDevelopment/petals' }
$Device    = if ($env:DRIFT_DEVICE)   { $env:DRIFT_DEVICE }   else { 'auto' }
$TorchSpec = 'torch>=2.6,<2.7'

function Log($msg) { Write-Host "[drift] $msg" -ForegroundColor Cyan }
function Die($msg) { Write-Host "[drift] error: $msg" -ForegroundColor Red; exit 1 }
function Has($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

# 1. Get the code: reuse the current checkout if we're in one, otherwise clone.
if ((Test-Path pyproject.toml) -and (Select-String -Path pyproject.toml -Pattern '^name = "petals"' -Quiet)) {
    Log "using the checkout in $(Get-Location)"
} else {
    if (-not (Has git)) { Die 'git is required to fetch the code' }
    Log "cloning $RepoUrl"
    git clone --depth 1 $RepoUrl petals
    Set-Location petals
}

# 2. Prerequisites.
if (-not (Has uv)) { Die 'uv is required (https://docs.astral.sh/uv/). Install it and re-run.' }
if (-not (Has go)) { Die 'Go is required to build the hivemind wheel (https://go.dev/dl/). Install it and re-run.' }

# 3. Detect the accelerator (best effort; set $env:DRIFT_DEVICE = 'xpu' for Intel Arc).
if ($Device -eq 'auto') {
    $Device = if (Has nvidia-smi) { 'cuda' } else { 'cpu' }
}
Log "target device: $Device"

# 4. Create the environment.
Log 'creating .venv with uv'
uv venv --python 3.12

# 5. Build and install the patched hivemind wheel (must precede `pip install -e .`,
#    which does not pull hivemind on Windows).
Log 'building the patched hivemind wheel (needs Go on PATH)'
uv run python scripts/build_hivemind_windows.py --out-dir dist
$wheel = Get-ChildItem .\dist\hivemind-1.1.12-*-win_amd64.whl | Select-Object -Last 1
if (-not $wheel) { Die 'hivemind wheel build produced no artifact in .\dist' }
uv pip install $wheel.FullName

# 6. Install a PyTorch build for the chosen device.
Log "installing PyTorch ($Device)"
switch ($Device) {
    'cpu'  { uv pip install $TorchSpec }  # default Windows wheels are CPU-only
    'cuda' { uv pip install --index-url https://download.pytorch.org/whl/cu124 $TorchSpec }
    'xpu'  { uv pip install --index-url https://download.pytorch.org/whl/xpu 'torch==2.6.0+xpu' }
    default { Die "unknown DRIFT_DEVICE=$Device (expected cpu|cuda|xpu)" }
}

# 7. Install DRIFT-LLM itself.
Log 'installing DRIFT-LLM'
uv pip install -e .

Log 'done.'
Write-Host "`nStart a swarm on this machine:`n`n    petals up meta-llama/Llama-3.1-8B-Instruct`n" -ForegroundColor Green
Write-Host 'It prints a "petals up ... --join drift://..." command; run that on your other machines to add their compute.'
