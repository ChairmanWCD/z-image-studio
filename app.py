import os
import time
import uuid
import base64
import logging
from enum import Enum
from typing import Optional, List, Union

from pydantic import BaseModel, Field

import config

logger = logging.getLogger("zimage.app")


class ImageSizePreset(str, Enum):
    square_hd = "square_hd"
    square = "square"
    portrait_4_3 = "portrait_4_3"
    portrait_16_9 = "portrait_16_9"
    landscape_4_3 = "landscape_4_3"
    landscape_16_9 = "landscape_16_9"


IMAGE_SIZE_PRESETS = {
    ImageSizePreset.square_hd: (1024, 1024),
    ImageSizePreset.square: (1024, 1024),
    ImageSizePreset.portrait_4_3: (896, 1152),
    ImageSizePreset.portrait_16_9: (768, 1344),
    ImageSizePreset.landscape_4_3: (1152, 896),
    ImageSizePreset.landscape_16_9: (1344, 768),
}


def fit_to_vram(width: int, height: int, max_side: int, multiple: int = 16):
    """Scale dimensions down proportionally so the longest side fits max_side.

    Keeps values on the `multiple` grid that diffusion models expect.
    """
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / longest
        width = int(round(width * scale / multiple) * multiple)
        height = int(round(height * scale / multiple) * multiple)
    else:
        width = int(round(width / multiple) * multiple)
        height = int(round(height / multiple) * multiple)
    return max(multiple, width), max(multiple, height)


class CustomImageSize(BaseModel):
    width: int = Field(ge=512, le=2048)
    height: int = Field(ge=512, le=2048)


class Input(BaseModel):
    prompt: str = Field(description="Text prompt for image generation")
    image_size: Union[ImageSizePreset, CustomImageSize] = Field(
        default=ImageSizePreset.square_hd,
        description="Image size preset or custom {width, height}",
    )
    num_inference_steps: int = Field(default=9, ge=1, le=10)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    negative_prompt: str = Field(
        default="", description="Negative prompt (ignored when guidance_scale=0)"
    )
    seed: Optional[int] = None
    num_images: int = Field(default=1, ge=1, le=4)
    output_format: str = Field(default="png", pattern="^(png|jpeg)$")
    sync_mode: bool = Field(
        default=False, description="Return base64 data URIs instead of file URLs"
    )
    enable_safety_checker: bool = Field(
        default=True,
        description="Z-Image has no built-in safety checker; kept for API compatibility",
    )


class OutputImage(BaseModel):
    url: Optional[str] = None
    base64_data: Optional[str] = None
    width: int
    height: int
    content_type: str


class Output(BaseModel):
    images: List[OutputImage]
    seed: int
    prompt: str
    timings: dict


class ZImageApp:
    """Mirrors fal.App: setup() for one-time init, generate() per request."""

    machine_type = "GPU-GB10"

    def __init__(self):
        self.pipe = None
        self.warm = False

    def setup(self):
        import torch

        logger.info("Loading Z-Image-Turbo pipeline (mode=%s)...", config.LOAD_MODE)
        t0 = time.time()

        if config.LOAD_MODE == "local":
            self._load_local()
        else:
            self._load_hf()

        self._configure_vram_saving()
        logger.info("Pipeline loaded in %.1fs", time.time() - t0)

        logger.info("Running warmup inference...")
        t0 = time.time()
        self.pipe(
            prompt="a cat sitting on a windowsill",
            num_inference_steps=2,
            guidance_scale=0.0,
            height=512,
            width=512,
            num_images_per_prompt=1,
        )
        logger.info("Warmup complete in %.1fs", time.time() - t0)
        self.warm = True

    def _configure_vram_saving(self):
        logger.info("Enabling VRAM-saving mode (targets 8-12GB consumer GPUs)...")
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()
        vae = getattr(self.pipe, "vae", None)
        if vae is not None:
            try:
                vae.enable_tiling()
                vae.enable_slicing()
            except (AttributeError, NotImplementedError):
                logger.info("VAE tiling/slicing not available on this VAE; skipping")

    def _load_hf(self):
        import torch
        from diffusers import ZImagePipeline

        self.pipe = ZImagePipeline.from_pretrained(
            config.MODEL_ID,
            torch_dtype=torch.bfloat16,
        )

    def _load_local(self):
        import torch
        from diffusers import (
            ZImagePipeline,
            ZImageTransformer2DModel,
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
        )
        from safetensors.torch import load_file

        logger.info("Loading transformer from %s", config.LOCAL_TRANSFORMER_PATH)
        transformer = ZImageTransformer2DModel.from_single_file(
            config.LOCAL_TRANSFORMER_PATH,
            torch_dtype=torch.bfloat16,
        )

        logger.info("Loading VAE from %s", config.LOCAL_VAE_PATH)
        vae = AutoencoderKL.from_single_file(
            config.LOCAL_VAE_PATH,
            config=config.MODEL_ID,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )

        logger.info("Loading text encoder from local %s", config.LOCAL_TEXT_ENCODER_PATH)
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        te_config = AutoConfig.from_pretrained(
            config.MODEL_ID, subfolder="text_encoder"
        )
        text_encoder = AutoModelForCausalLM.from_config(
            te_config, torch_dtype=torch.bfloat16
        )
        state_dict = load_file(config.LOCAL_TEXT_ENCODER_PATH)
        missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info("Text encoder missing keys (expected): %s", missing)
        if unexpected:
            logger.warning("Text encoder unexpected keys: %s", unexpected)

        logger.info("Loading tokenizer from HuggingFace (small download)...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_ID, subfolder="tokenizer"
        )

        logger.info("Loading scheduler config from HuggingFace (small download)...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.MODEL_ID, subfolder="scheduler"
        )

        self.pipe = ZImagePipeline(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

    def generate(self, input: Input) -> Output:
        import torch

        if isinstance(input.image_size, CustomImageSize):
            width, height = input.image_size.width, input.image_size.height
        else:
            width, height = IMAGE_SIZE_PRESETS[input.image_size]

        orig_w, orig_h = width, height
        width, height = fit_to_vram(width, height, config.MAX_IMAGE_SIDE)
        if (width, height) != (orig_w, orig_h):
            logger.info(
                "Downscaling %dx%d -> %dx%d (max side %d)",
                orig_w,
                orig_h,
                width,
                height,
                config.MAX_IMAGE_SIDE,
            )

        seed = input.seed if input.seed is not None else int(torch.seed())
        generator = torch.Generator("cuda").manual_seed(seed)

        logger.info(
            "Generating: prompt=%r size=%dx%d steps=%d guidance=%.1f seed=%d",
            input.prompt[:80],
            width,
            height,
            input.num_inference_steps,
            input.guidance_scale,
            seed,
        )
        t0 = time.time()

        images = self.pipe(
            prompt=input.prompt,
            negative_prompt=input.negative_prompt or None,
            num_inference_steps=input.num_inference_steps,
            guidance_scale=input.guidance_scale,
            height=height,
            width=width,
            num_images_per_prompt=input.num_images,
            generator=generator,
        ).images

        gen_time = time.time() - t0
        logger.info("Generation complete in %.1fs", gen_time)

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        return self._build_output(images, seed, gen_time, input)

    def _build_output(self, images, seed, gen_time, input):
        output_images = []
        for i, img in enumerate(images):
            filename = f"{uuid.uuid4().hex}_batch_{i}.{input.output_format}"
            filepath = os.path.join(config.OUTPUT_DIR, filename)
            img.save(filepath, format=input.output_format.upper())

            if input.sync_mode:
                with open(filepath, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                data_uri = f"data:image/{input.output_format};base64,{data}"
                output_images.append(
                    OutputImage(
                        base64_data=data_uri,
                        width=images[0].width,
                        height=images[0].height,
                        content_type=f"image/{input.output_format}",
                    )
                )
            else:
                url = f"{config.BASE_URL}/output/{filename}"
                output_images.append(
                    OutputImage(
                        url=url,
                        width=images[0].width,
                        height=images[0].height,
                        content_type=f"image/{input.output_format}",
                    )
                )

        return Output(
            images=output_images,
            seed=seed,
            prompt=input.prompt,
            timings={
                "generation_seconds": round(gen_time, 2),
                "num_images": input.num_images,
                "num_inference_steps": input.num_inference_steps,
            },
        )

    def generate_sync(self, input: Input) -> Output:
        return self.generate(input)
