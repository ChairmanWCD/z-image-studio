# Z-Image Studio

A FAL.ai-style async inference server for **Tongyi-MAI Z-Image-Turbo** — a fast text-to-image diffusion model. Ships with a gold-themed dark-mode web UI, a queue-based REST API (`/submit` → `/status` → `/result`), a synchronous `/generate`, and runs **natively on Windows** (no Docker).

This is the **Windows / consumer-GPU edition**: tuned to run on a GeForce RTX 5070 (12 GB) by sharing ComfyUI's model folder and offloading model weights from VRAM. The original Docker/Linux build (NVIDIA GB10 Grace Blackwell) is archived under [`_archive/`](_archive).

---

## Features

- **Async queue API** — submit a job, poll for status, fetch the result (mirrors FAL.ai's pattern)
- **Sync API** — `POST /generate` returns the result inline
- **Web UI** — prompt controls, size presets, seed control, history strip, live health status, dark/light toggle
- **Local weights mode** — loads `.safetensors` directly from disk; only tiny config/tokenizer files are fetched from HuggingFace
- **12 GB GPU friendly** — `enable_model_cpu_offload` + attention/VAE slicing keeps peak VRAM near the transformer's ~12 GB; any requested size is auto-scaled down to fit `MAX_IMAGE_SIDE` (default 1024)

---

## Windows port — work log

What was done converting the Linux/Docker build into this native-Windows edition (RTX 5070, 12 GB):

- **Docker removed** — `Dockerfile`, `docker-compose.yml`, `start.sh`, `stop.sh` moved to [`_archive/`](_archive). Replaced by `setup.ps1` (venv + deps) and `start.ps1` (uvicorn launcher).
- **VRAM-safe on 12 GB** — `app.py` uses `pipe.enable_model_cpu_offload()` + `pipe.enable_attention_slicing()`, and calls VAE tiling/slicing via `vae.enable_tiling()` / `vae.enable_slicing()` (an older version attempted pipeline-level `enable_vae_tiling`, which `ZImagePipeline` does not expose → startup error, fixed). Requested sizes auto-scale to `MAX_IMAGE_SIDE`.
- **Shared weights with ComfyUI** — `start.ps1` defaults `LOCAL_MODELS_DIR` to `C:\Users\Ken Bai\ComfyUI-Shared\models`; no duplicate ~20 GB download.
- **Windows paths** — `config.py` now auto-detects a `../models` sibling folder instead of the container path `/models`; `OUTPUT_DIR` defaults to `<repo>\output`.
- **Environment** — torch `cu130` wheels (falls back to `cu128`), `diffusers` from git main, `transformers` 5.x. Verified on this machine: `torch 2.13.0+cu130`, `cuda=True`, `NVIDIA GeForce RTX 5070`.
- **Gotchas hit & fixed** — PowerShell execution policy blocks `.ps1` (see Troubleshooting); nested-quote bug in `setup.ps1`'s final torch line (fixed).

---

## Requirements

### Hardware

| Component | Minimum | Tested |
|-----------|---------|--------|
| NVIDIA GPU | 12 GB VRAM (Blackwell / Ampere / Ada) | RTX 5070 (12 GB) |
| Disk (for weights) | ~20 GB | — |
| RAM | ~16 GB | — |

> The three weights total ~19.8 GB in bf16 (12 GB transformer + 7.5 GB Qwen3-4B text encoder + 320 MB VAE). Weights are loaded to VRAM one module at a time; the rest live in system RAM.

### Software

| Component | Version |
|-----------|---------|
| Windows | 10/11 |
| NVIDIA driver | ≥ 580 (CUDA 13.x drivers work with both `cu130` and `cu128` torch wheels) |
| Python | 3.12 (base interpreter reused from ComfyUI's `.venv`) |

### Model files (~19.8 GB total)

Place these in a folder with the structure below (`diffusion_models`, `text_encoders`, `vae`). If you use ComfyUI on this machine, **point the server at ComfyUI's shared models folder** so both tools share the same weights:

```
C:\Users\Ken Bai\ComfyUI-Shared\models
├── diffusion_models\
│   └── z_image_turbo_bf16.safetensors      (12 GB)
├── text_encoders\
│   └── qwen_3_4b.safetensors               (7.5 GB)
└── vae\
    └── ae.safetensors                      (320 MB)
```

| File | Size | Source |
|------|------|--------|
| `z_image_turbo_bf16.safetensors` | 12 GB | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) transformer |
| `qwen_3_4b.safetensors` | 7.5 GB | Qwen3-4B text encoder |
| `ae.safetensors` | 320 MB | Flux-style VAE (16 latent channels) |

---

## Installation

### 1. One-time setup

```powershell
# creates .venv, installs torch (cu130, auto-falls back to cu128),
# requirements, and diffusers from git
.\setup.ps1
```

If the model repo is gated on HuggingFace, export your token before starting so the small config/tokenizer/scheduler files can be downloaded:

```powershell
$env:HF_TOKEN = "hf_..."
```

### 2. Start

```powershell
.\start.ps1
# or: .\start.ps1 -Port 9000 -ModelsDir D:\models
```

`start.ps1` defaults `LOCAL_MODELS_DIR` to `C:\Users\Ken Bai\ComfyUI-Shared\models` (ComfyUI's shared folder). On first load the server downloads tiny config/tokenizer files from HuggingFace, then loads the local weights (~2 minutes) and runs a warmup inference. You'll see:

```
=== Z-Image Studio (Windows) ===
  UI:       http://localhost:8100
```

### 3. Open the UI

- On this machine: `http://localhost:8100`
- From the LAN: `http://<host-ip>:8100`
- API docs (Swagger): `http://localhost:8100/docs`

### 4. Stop

Press `Ctrl+C` in the terminal running `start.ps1`.

---

## Configuration

All settings are env-var overridable before `start.ps1` (or edit the script).

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8100` | Port the server listens on |
| `PUBLIC_PORT` | `{PORT}` | Port used in result URLs |
| `ZIMAGE_LOAD_MODE` | `local` (set by `start.ps1`) | `local` = load from `.safetensors`; `hf` = download full model from HuggingFace |
| `LOCAL_MODELS_DIR` | auto-detect `../models` | Folder containing `diffusion_models/`, `text_encoders/`, `vae/` |
| `MAX_IMAGE_SIDE` | `1024` | Longest side (px) any request is scaled down to, to protect 12 GB VRAM |
| `OUTPUT_DIR` | `.\output` | Where generated images are saved |
| `HF_TOKEN` | _(empty)_ | HuggingFace token for gated downloads / higher rate limits |
| `MAX_QUEUE_SIZE` | `100` | Max queued jobs before rejecting with 429 |
| `MAX_RETRIES` | `2` | Retries per job on transient failure |
| `JOB_TTL_SECONDS` | `3600` | How long completed/failed jobs stay queryable |
| `HOST_IP` | _(empty)_ | LAN IP used in result URLs (empty ⇒ `localhost`) |

---

## API Reference

Base URL: `http://localhost:8100`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI (gold dark-theme interface) |
| `GET` | `/health` | Server health (`{"status":"ok"}` / `503` with error) |
| `POST` | `/generate` | Sync single-call generation — returns result inline |
| `POST` | `/submit` | Submit a generation job → returns `request_id` |
| `GET` | `/status/{request_id}` | Poll job status (`IN_QUEUE` / `IN_PROGRESS` / `COMPLETED` / `FAILED`) |
| `GET` | `/result/{request_id}` | Fetch result (images, seed, timings) when `COMPLETED` |
| `POST` | `/cancel/{request_id}` | Cancel a queued job |
| `GET` | `/api/history` | List recent output images (max 20) |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/output/{filename}` | Static image file |

### `POST /generate` — synchronous single-call

```bash
curl -X POST http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cute red panda reading a book in a library","num_inference_steps":4,"num_images":2,"output_format":"png"}'
```

Response: `{ "images": [ {url, width, height, content_type} ], "seed", "prompt", "timings" }`.
Use `"sync_mode": true` in the body to get base64 data URIs inline instead of file URLs.

### Submit parameters

| Field | Type | Default | Range |
|-------|------|---------|-------|
| `prompt` | string | _(required)_ | — |
| `image_size` | preset or `{width, height}` | `square_hd` | `square_hd`, `square`, `portrait_4_3`, `portrait_16_9`, `landscape_4_3`, `landscape_16_9`, or custom `512–2048` (auto-scaled to `MAX_IMAGE_SIDE`) |
| `num_inference_steps` | int | `9` | 1–10 |
| `guidance_scale` | float | `0.0` | 0–20 |
| `negative_prompt` | string | `""` | ignored when `guidance_scale=0` |
| `seed` | int\|null | `null` (random) | — |
| `num_images` | int | `1` | 1–4 (batches >1 use much more VRAM — keep low on 12 GB) |
| `output_format` | string | `png` | `png` or `jpeg` |
| `sync_mode` | bool | `false` | return base64 data URIs instead of file URLs |

### Example: full async flow

```bash
# 1. Submit
RID=$(curl -s -X POST localhost:8100/submit \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a serene mountain lake at dawn","num_inference_steps":4}' \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['request_id'])")

# 2. Poll until COMPLETED
curl -s localhost:8100/status/$RID

# 3. Fetch result
curl -s localhost:8100/result/$RID
```

---

## Web UI

The UI (`http://localhost:8100`) provides:

- **Prompt** textarea with example chips and `Cmd/Ctrl+Enter` shortcut
- **Advanced settings** (collapsible): negative prompt, guidance scale, seed + randomize
- **Size presets**: 6 aspect-ratio buttons (sizes > 1024 are downscaled on a 12 GB GPU)
- **Inference steps** slider (1–10)
- **Image count** (1–4) and **format** (PNG/JPEG)
- **Generate** button with live elapsed timer and skeleton loading state
- **History strip** from `/api/history`, click to open full-size
- **Theme toggle**: dark (default, gold) / light, persisted in `localStorage`

---

## Repo Layout

```
z-image-inference/
├── app.py                  # ZImageApp: model loading (_load_local/_load_hf) + generate()
├── config.py               # Env-overridable configuration (+ MAX_IMAGE_SIDE)
├── server.py               # FastAPI app: queue endpoints, UI serving, /api/history
├── requirements_zimage.txt # Python dependencies
├── setup.ps1               # One-time venv + deps install
├── start.ps1               # Windows launcher (shared ComfyUI models, port 8100)
├── .gitignore              # output/, .venv/, __pycache__/
├── _archive/               # Docker/Linux artifacts from the GB10 build (optional)
└── static/
    └── index.html          # Single-page web UI (Tailwind CDN + vanilla JS)
```

---

## Troubleshooting

**`running scripts is disabled on this system`** — PowerShell's default execution policy blocks `.ps1` files. Run setup/start with a one-off bypass: `powershell -ExecutionPolicy Bypass -File .\setup.ps1` (same form for `.\start.ps1`). Or enable local scripts once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then call `.\setup.ps1` normally.

**`No .venv found.`** — run `.\setup.ps1` first.

**Torch install fails with `cu130`** — run `.\setup.ps1 -CudaIndex cu128`. RTX 40/50-series cards work fine with either; your driver must support that CUDA version (≥ 580).

**VRAM out of memory at 1024 × 1024** — lower `MAX_IMAGE_SIDE` (e.g. `$env:MAX_IMAGE_SIDE = 896` before `start.ps1`), reduce to 4 inference steps, and keep `num_images` at 1. Close other GPU apps (Discord/Electron/browser hardware acceleration all hold VRAM).

**`CUDA unavailable` | `torch.cuda.is_available()` is False** — update the NVIDIA driver, then confirm `nvidia-smi` shows your GPU. Reinstall torch for a CUDA build (never the CPU-only wheel).

**Health returns `503` with a model error** — look at the terminal running `start.ps1` for the traceback. Common causes: missing model files, wrong `LOCAL_MODELS_DIR`, gated HuggingFace config (set `HF_TOKEN`), GPU OOM.

**Slow first request after startup** — the warmup runs at 512 × 512; the first real request at a larger size re-compiles kernels and offloads modules, so allow a few extra seconds. Subsequent requests are fast.

**"HuggingFace hub: 401"** — the config/tokenizer/scheduler repos are gated or rate-limited. Set `$env:HF_TOKEN = "hf_..."` before `start.ps1`.

---

## License

This repo is a server wrapper. The Z-Image-Turbo model is © Tongyi-MAI / Alibaba — refer to the [model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) for model licensing terms.