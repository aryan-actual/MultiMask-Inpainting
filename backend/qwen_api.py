import io
import json
import os
from typing import List, Optional

import torch
import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# Ensure you have installed diffusers from source
from diffusers import QwenImageEditPlusPipeline

app = FastAPI(title="Qwen-Image-Edit-2511 Visual Instruction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing Qwen-Image-Edit-2511 Plus Pipeline...")
try:
    # We load the pipeline in bfloat16. 
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    print("Pipeline loaded successfully with CPU offloading.")
except Exception as e:
    import traceback
    print("Failed to load Qwen pipeline:")
    traceback.print_exc()
    pipe = None

@app.get("/health")
def health_check():
    return {"status": "ok", "pipeline_loaded": pipe is not None}

@app.post("/edit")
async def edit_image(
    original_image: UploadFile = File(...),
    edits: str = Form(...),
    files: List[UploadFile] = File(default=[])
):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")

    try:
        edit_operations = json.loads(edits)
        if not isinstance(edit_operations, list):
            raise ValueError("edits must be a JSON array of operations.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid edits JSON: {str(e)}")

    try:
        orig_img_bytes = await original_image.read()
        current_image = Image.open(io.BytesIO(orig_img_bytes)).convert("RGB")
        
        max_size = 1024
        if current_image.width > max_size or current_image.height > max_size:
            current_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        w, h = current_image.size
        w = (w // 16) * 16
        h = (h // 16) * 16
        current_image = current_image.resize((w, h), Image.Resampling.LANCZOS)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid original_image: {str(e)}")

    file_map = {}
    for f in files:
        if f.filename:
            file_map[f.filename] = f

    print(f"Starting Visual Instruction processing for {len(edit_operations)} edits in ONE pass...")

    annotated_image = current_image.convert("RGBA")
    
    COLORS_RGB = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (128, 0, 128)]
    COLOR_NAMES = ["red", "blue", "green", "yellow", "purple"]
    
    master_prompt = "In the first image:\n"
    ref_images = []
    masks_for_composite = []

    for i, edit in enumerate(edit_operations):
        if i >= len(COLORS_RGB):
            print(f"Warning: Only {len(COLORS_RGB)} edits supported in one pass. Skipping excess.")
            break

        mask_filename = edit.get("mask")
        prompt = edit.get("prompt", "")
        ref_filename = edit.get("reference")

        if not mask_filename or mask_filename not in file_map:
            raise HTTPException(status_code=400, detail=f"Mask file '{mask_filename}' not found.")
        
        mask_bytes = await file_map[mask_filename].read()
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_img = mask_img.resize(current_image.size, Image.Resampling.NEAREST)
        
        mask_array = np.array(mask_img)
        mask_array = np.where(mask_array > 128, 255, 0).astype(np.uint8)
        mask_img = Image.fromarray(mask_array)
        masks_for_composite.append(mask_img)
        
        # Convert mask to an OUTLINE annotation instead of a filled blob.
        # A filled semi-transparent blob alters the image pixels, confusing the VLM into 
        # thinking the object is tinted red/blue, and blinding it to details underneath.
        # An outline allows the VLM to see the original context perfectly while knowing the bounds.
        outline = mask_img.filter(ImageFilter.FIND_EDGES)
        outline = outline.filter(ImageFilter.MaxFilter(5)) # Thicken to 5px for clear VLM visibility
        
        # Draw solid 100% opacity colored outline
        solid_color = Image.new("RGBA", current_image.size, COLORS_RGB[i] + (255,))
        transparent = Image.new("RGBA", current_image.size, (0,0,0,0))
        color_layer = Image.composite(solid_color, transparent, outline)
        annotated_image = Image.alpha_composite(annotated_image, color_layer)

        color_name = COLOR_NAMES[i]
        prompt_part = f"- Replace the region enclosed by the {color_name} outline with: {prompt}. Ensure the {color_name} outline itself is completely removed."
        
        if ref_filename and len(ref_images) < 2:
            if ref_filename not in file_map:
                raise HTTPException(status_code=400, detail=f"Reference file '{ref_filename}' not found.")
            ref_bytes = await file_map[ref_filename].read()
            ref_img = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
            ref_images.append(ref_img)
            ref_idx = len(ref_images) + 1 # +1 because annotated_image is 1st
            image_word = ["second", "third"][ref_idx - 2]
            prompt_part += f" Use the {image_word} image as a visual reference."
        elif ref_filename:
            print(f"Warning: Reference image for step {i+1} ignored due to max 2 limit.")
            
        master_prompt += prompt_part + "\n"

    annotated_image = annotated_image.convert("RGB") # Convert back to RGB for model

    print(f"Master Prompt:\n{master_prompt}")

    try:
        images_list = [annotated_image] + ref_images
        
        # Save debug annotated image
        annotated_image.save("debug_annotated_input.png")

        # Qwen-Image-Edit-Plus Official Diffusers Generation Call
        output = pipe(
            image=images_list,
            prompt=master_prompt,
            negative_prompt=" ",
            num_inference_steps=20, 
            true_cfg_scale=5.0
        ).images[0]

        output = output.resize(current_image.size, Image.Resampling.LANCZOS)
        
        # We NO LONGER composite the output back onto the original image.
        # The Qwen-Image-Edit-Plus model is trained to globally preserve the unedited background 
        # while perfectly rewriting the annotated regions into seamless pixels.
        # If we manually composite, we cut off the generated shadows and leave ghosting artifacts 
        # from the semi-transparent red/blue annotations.
        final_output = output

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during edit generation: {str(e)}")

    out_byte_arr = io.BytesIO()
    final_output.save(out_byte_arr, format='PNG')
    out_byte_arr = out_byte_arr.getvalue()

    return Response(content=out_byte_arr, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
