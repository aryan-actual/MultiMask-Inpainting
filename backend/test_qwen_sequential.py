import os
import torch
from PIL import Image, ImageDraw
from diffusers import QwenImageEditInpaintPipeline

def create_dummy_images():
    """Create a dummy original image, two masks, and one reference image."""
    # 1. Original Image: 512x512 blue square
    original = Image.new("RGB", (512, 512), color="blue")
    
    # 2. Mask 1: A white rectangle on the left side (representing e.g., a sofa)
    mask1 = Image.new("L", (512, 512), color="black")
    draw1 = ImageDraw.Draw(mask1)
    draw1.rectangle([50, 100, 200, 400], fill="white")
    
    # 3. Reference Image: A green pattern (representing a modern sofa material)
    ref1 = Image.new("RGB", (512, 512), color="green")
    
    # 4. Mask 2: A white rectangle on the right side (representing e.g., a table)
    mask2 = Image.new("L", (512, 512), color="black")
    draw2 = ImageDraw.Draw(mask2)
    draw2.rectangle([300, 100, 450, 400], fill="white")
    
    return original, mask1, ref1, mask2

def main():
    print("Loading Qwen-Image-Edit Pipeline...")
    
    pipe = QwenImageEditInpaintPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16
    )
    # Essential for 7B parameter models on consumer hardware
    pipe.enable_model_cpu_offload()
    
    print("Creating dummy inputs...")
    current_image, mask_1, ref_1, mask_2 = create_dummy_images()
    
    print("\n--- Starting Sequential Editing ---")
    
    # Edit 1: Replace left object using a reference image
    prompt_1 = "Replace the masked area with the green material from the reference image."
    print(f"\nStep 1: {prompt_1}")
    print("Using: Mask 1 + Reference Image 1")
    
    final_prompt_1 = f"In the first image, {prompt_1}. Use the second image as a visual reference."
    
    current_image = pipe(
        image=[current_image, ref_1],
        mask_image=mask_1,
        prompt=final_prompt_1,
        negative_prompt=" ",
        strength=1.0,
        num_inference_steps=20,
        true_cfg_scale=5.0
    ).images[0]
    
    current_image.save("qwen_step_1_output.png")
    print("Saved qwen_step_1_output.png")
    
    # Edit 2: Remove right object (no reference image)
    prompt_2 = "Remove the object and seamlessly blend with the blue background."
    print(f"\nStep 2: {prompt_2}")
    print("Using: Mask 2 (No Reference Image)")
    
    current_image = pipe(
        image=[current_image], # Only base image
        mask_image=mask_2,
        prompt=prompt_2,
        negative_prompt=" ",
        strength=1.0,
        num_inference_steps=20,
        true_cfg_scale=5.0
    ).images[0]
    
    current_image.save("qwen_step_2_final_output.png")
    print("Saved qwen_step_2_final_output.png")
    
    print("\nSequential editing completed successfully.")

if __name__ == "__main__":
    main()
