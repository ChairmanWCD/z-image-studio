# setup.ps1 - one-time install for the Windows-native Z-Image Studio
# Creates a dedicated .venv and installs torch (CUDA), requirements, and diffusers.
#
# Usage:
#   .\setup.ps1               # default: cu130 torch wheels
#   .\setup.ps1 -CudaIndex cu128   # fall back if cu130 wheels are unavailable
#
# The venv is created from the same Python 3.12 used by ComfyUI (safe: a new
# venv never modifies the base interpreter). Override with -BasePython.

param(
  [ValidateSet("cu130", "cu128")]
  [string]$CudaIndex = "cu130",
  [string]$BasePython = "C:\Users\Ken Bai\White Crane LLC\ComfyUI\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Repo

if (-not (Test-Path $BasePython)) {
  Write-Error "Base Python not found at: $BasePython`
Set -BasePython to your Python 3.12 interpreter (e.g. 'py -3.12')."
}
if (Test-Path ".venv") {
  Write-Host "Existing .venv found - skipping venv creation." -ForegroundColor Yellow
} else {
  Write-Host "Creating venv from $BasePython ..."
  & $BasePython -m venv .venv
  if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Pip = Join-Path $Repo ".venv\Scripts\pip.exe"

& $Py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Write-Host "Installing PyTorch ($CudaIndex)..."
& $Pip install "torch" "torchvision" --index-url "https://download.pytorch.org/whl/$CudaIndex"
if ($LASTEXITCODE -ne 0 -and $CudaIndex -ne "cu128") {
  Write-Host "cu130 install failed; retrying with cu128..." -ForegroundColor Yellow
  & $Pip install "torch" "torchvision" --index-url "https://download.pytorch.org/whl/cu128"
}
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

Write-Host "Installing requirements..."
& $Pip install -r requirements_zimage.txt
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }

Write-Host "Installing diffusers from git (main)..."
& $Pip install "git+https://github.com/huggingface/diffusers"
if ($LASTEXITCODE -ne 0) { throw "diffusers install failed" }

$TorchInfo = & $Py -c "import torch; print(torch.__version__ + ' / cuda=' + str(torch.cuda.is_available()))"
Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "  venv:     $Repo\.venv"
Write-Host "  torch:    $TorchInfo"
Write-Host ""
Write-Host "Now verify your model path, then run:"
Write-Host "  .\start.ps1"