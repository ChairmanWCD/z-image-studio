import os

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
PUBLIC_PORT = int(os.environ.get("PUBLIC_PORT", str(PORT)))

LOAD_MODE = os.environ.get("ZIMAGE_LOAD_MODE", "hf")

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


def _default_models_dir():
    sibling = os.path.join(os.path.dirname(_REPO_DIR), "models")
    if os.path.isdir(sibling):
        return sibling
    return os.path.join(_REPO_DIR, "models")


LOCAL_MODELS_DIR = os.environ.get("LOCAL_MODELS_DIR") or _default_models_dir()
LOCAL_TRANSFORMER_PATH = os.path.join(
    LOCAL_MODELS_DIR, "diffusion_models", "z_image_turbo_bf16.safetensors"
)
LOCAL_VAE_PATH = os.path.join(LOCAL_MODELS_DIR, "vae", "ae.safetensors")
LOCAL_TEXT_ENCODER_PATH = os.path.join(
    LOCAL_MODELS_DIR, "text_encoders", "qwen_3_4b.safetensors"
)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(_REPO_DIR, "output"))
TORCH_DTYPE = "bfloat16"

# Longest side (px) any requested image is scaled down to. On a 12GB GPU,
# sizes above ~1024 risk out-of-memory even with CPU offload enabled.
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1024"))

MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "100"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "3600"))

HOST_IP = os.environ.get("HOST_IP", "")
BASE_URL = os.environ.get(
    "BASE_URL", f"http://{HOST_IP or 'localhost'}:{PUBLIC_PORT}"
)
