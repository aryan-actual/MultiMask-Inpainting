import os
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import QwenImageEditPlusPipeline

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
    print("Loading Qwen-Image-Edit-Plus Pipeline...")
    
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    
    print("Creating dummy inputs...")
    current_image, mask_1, ref_1, mask_2 = create_dummy_images()
    
    print("\n--- Starting Visual Instruction Editing (Single Pass) ---")
    
    annotated_image = current_image.convert("RGBA")
    
    # Colorize Mask 1 (Red)
    solid_red = Image.new("RGBA", current_image.size, (255, 0, 0, 160))
    transparent = Image.new("RGBA", current_image.size, (0, 0, 0, 0))
    layer_red = Image.composite(solid_red, transparent, mask_1)
    annotated_image = Image.alpha_composite(annotated_image, layer_red)
    
    # Colorize Mask 2 (Blue)
    solid_blue = Image.new("RGBA", current_image.size, (0, 0, 255, 160))
    layer_blue = Image.composite(solid_blue, transparent, mask_2)
    annotated_image = Image.alpha_composite(annotated_image, layer_blue)
    
    annotated_image = annotated_image.convert("RGB")
    annotated_image.save("debug_annotated_input.png")
    
    master_prompt = "In the first image:\n"
    master_prompt += "- Edit the region marked in red: Replace the masked area with a green modern sofa. Use the second image as a visual reference.\n"
    master_prompt += "- Edit the region marked in blue: Remove the object and seamlessly blend with the blue background.\n"
    
    print("Master Prompt:\n", master_prompt)
    
    output = pipe(
        image=[annotated_image, ref_1],
        prompt=master_prompt,
        negative_prompt=" ",
        num_inference_steps=20,
        true_cfg_scale=5.0
    ).images[0]
    
    # Combine masks for pristine compositing of background
    combined_mask = Image.composite(Image.new("L", current_image.size, 255), mask_2, mask_1)
    for _ in range(15):
        combined_mask = combined_mask.filter(ImageFilter.MaxFilter(3))
    combined_mask = combined_mask.filter(ImageFilter.GaussianBlur(radius=25))
    
    final_output = Image.composite(output, current_image, combined_mask)
    
    final_output.save("qwen_visual_instruction_output.png")
    print("Saved qwen_visual_instruction_output.png")
    print("\nEditing completed successfully.")

if __name__ == "__main__":
    main()
