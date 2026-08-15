# start.ps1 - launch the Windows-native Z-Image Studio server
#
# Usage:
#   .\start.ps1                     # default: port 8100, ComfyUI shared models
#   .\start.ps1 -Port 9000
#   .\start.ps1 -ModelsDir "D:\my-models"

param(
  [int]$Port = 8100,
  [string]$ModelsDir = "C:\Users\Ken Bai\ComfyUI-Shared\models"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Repo

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Write-Error "No .venv found. Run .\setup.ps1 first."
}

if (-not (Test-Path (Join-Path $ModelsDir "diffusion_models\z_image_turbo_bf16.safetensors"))) {
  Write-Warning "Model file not found under: $ModelsDir"
}

# Native local-mode config
$env:ZIMAGE_LOAD_MODE = "local"
$env:LOCAL_MODELS_DIR = $ModelsDir
$env:OUTPUT_DIR = Join-Path $Repo "output"
$env:PORT = "$Port"
$env:PUBLIC_PORT = "$Port"
$env:HOST_IP = ""   # empty => result URLs use http://localhost:<port>
$env:PIP_NO_CACHE_DIR = "1"

Write-Host "=== Z-Image Studio (Windows) ==="
Write-Host "  Models:   $ModelsDir"
Write-Host "  Port:     $Port"
Write-Host "  UI:       http://localhost:$Port"
Write-Host ""
Write-Host "First load downloads small tokenizer/config files from HuggingFace,"
Write-Host "then loads ~20GB of local weights (bf16). Allow a few minutes."
Write-Host "Press Ctrl+C to stop."
Write-Host ""

& ".venv\Scripts\python.exe" -m uvicorn server:app --host 0.0.0.0 --port $Port