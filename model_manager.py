import torch
from huggingface_hub import snapshot_download
from diffusers import (
    QwenImageEditPipeline,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
)

# Global variable to keep single instance
_model_instance = None

def load_qwen_model(repo_id="Qwen/Qwen-Image-Edit", lora_repo="lightx2v/Qwen-Image-Lightning"):
    """
    Loads and returns a pre-trained Qwen Image Edit pipeline with Lightning LoRA weights.
    Ensures model loads only once using a global singleton.
    """
    global _model_instance
    if _model_instance is not None:
        print("⚡ Model already loaded — reusing existing instance.")
        return _model_instance

    print("🔄 Downloading model weights...")
    cache_dir = snapshot_download(repo_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load submodules
    transformer = QwenImageTransformer2DModel.from_pretrained(
        f"{cache_dir}/transformer", torch_dtype=dtype
    ).to(device)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(cache_dir, subfolder="scheduler")
    vae = AutoencoderKLQwenImage.from_pretrained(f"{cache_dir}/vae", torch_dtype=dtype).to(device)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        f"{cache_dir}/text_encoder", torch_dtype=dtype
    ).to(device)
    tokenizer = Qwen2Tokenizer.from_pretrained(f"{cache_dir}/tokenizer")
    processor = Qwen2VLProcessor.from_pretrained(f"{cache_dir}/processor")

    pipe = QwenImageEditPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
    )

    pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    print("⚡ Loading Lightning LoRA weights...")
    pipe.load_lora_weights(
        lora_repo, weight_name="Qwen-Image-Lightning-4steps-V1.0.safetensors"
    )
    try:
        pipe.fuse_lora()
    except Exception:
        pass

    print("✅ Qwen Lightning model loaded successfully.")
    _model_instance = pipe
    return pipe
