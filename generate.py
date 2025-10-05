import torch
from huggingface_hub import snapshot_download
from diffusers import (
    QwenImageEditPipeline,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

def load_qwen_lightning():
    """Load Qwen Image Edit + Lightning LoRA model"""
    repo_id = "Qwen/Qwen-Image-Edit"
    cache_dir = snapshot_download(repo_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    transformer = QwenImageTransformer2DModel.from_pretrained(
        cache_dir + "/transformer", torch_dtype=dtype
    ).to(device)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cache_dir, subfolder="scheduler"
    )

    vae = AutoencoderKLQwenImage.from_pretrained(
        cache_dir + "/vae", torch_dtype=dtype
    ).to(device)

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cache_dir + "/text_encoder", torch_dtype=dtype
    ).to(device)

    tokenizer = Qwen2Tokenizer.from_pretrained(cache_dir + "/tokenizer")
    processor = Qwen2VLProcessor.from_pretrained(cache_dir + "/processor")

    pipe = QwenImageEditPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
    ).to(device)

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # Load Lightning LoRA
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name="Qwen-Image-Lightning-4steps-V1.0.safetensors",
    )
    try:
        pipe.fuse_lora()
    except Exception:
        pass

    return pipe
