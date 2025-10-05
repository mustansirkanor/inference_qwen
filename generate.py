from PIL import Image
from .model_manager import load_qwen_model

def generate_image(image_path, prompt, output_path="output.png"):
    """
    Generates a novel view of the provided image using Qwen-Image-Lightning.
    Args:
        image_path (str): Path to input image.
        prompt (str): Text prompt for novel view generation.
        output_path (str): Path to save the generated image.
    """
    print(f"🧠 Loading model...")
    pipe = load_qwen_model()
    img = Image.open(image_path).convert("RGB")

    print(f"🎨 Generating novel view for {image_path}...")
    result = pipe(
        image=img,
        prompt=prompt,
        negative_prompt="cropped, warped, distorted, duplicate objects, noisy, blurry",
        height=1024,
        width=1024,
        num_inference_steps=8,
        num_images_per_prompt=1,
        true_cfg_scale=2.5,
    ).images[0]

    result.save(output_path)
    print(f"✅ Saved generated image at {output_path}")
    return output_path
