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

# Ensure you have installed diffusers from source: pip install git+https://github.com/huggingface/diffusers.git
from diffusers import QwenImageEditInpaintPipeline

app = FastAPI(title="Qwen-Image-Edit-2511 Sequential Editing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing Qwen-Image-Edit-2511 Pipeline...")
try:
    # We load the pipeline in bfloat16. 
    # For models this large, CPU offloading is crucial for memory management.
    pipe = QwenImageEditInpaintPipeline.from_pretrained(
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
    """
    Perform sequential masked editing.
    
    `original_image`: The starting image.
    `edits`: A JSON string representing a list of edit operations.
             Format: [{"mask": "mask1.png", "prompt": "...", "reference": "ref1.png"}, ...]
    `files`: A flat list of all mask and reference image files referenced in the `edits` JSON.
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")

    try:
        # Parse edits JSON
        edit_operations = json.loads(edits)
        if not isinstance(edit_operations, list):
            raise ValueError("edits must be a JSON array of operations.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid edits JSON: {str(e)}")

    # Read the original image
    try:
        orig_img_bytes = await original_image.read()
        current_image = Image.open(io.BytesIO(orig_img_bytes)).convert("RGB")
        
        # Resize to max 1024 for memory efficiency and speed, maintaining aspect ratio
        max_size = 1024
        if current_image.width > max_size or current_image.height > max_size:
            current_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        # Ensure width and height are perfectly divisible by 16 for the VAE/Transformer
        w, h = current_image.size
        w = (w // 16) * 16
        h = (h // 16) * 16
        current_image = current_image.resize((w, h), Image.Resampling.LANCZOS)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid original_image: {str(e)}")

    # Map uploaded files by filename for easy access
    file_map = {}
    for f in files:
        if f.filename:
            file_map[f.filename] = f

    print(f"Starting sequential processing of {len(edit_operations)} edits...")

    # Process edits sequentially
    for i, edit in enumerate(edit_operations):
        mask_filename = edit.get("mask")
        prompt = edit.get("prompt", "")
        ref_filename = edit.get("reference")

        if not mask_filename or mask_filename not in file_map:
            raise HTTPException(status_code=400, detail=f"Mask file '{mask_filename}' not found in uploaded files.")
        
        # Load mask
        mask_bytes = await file_map[mask_filename].read()
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        
        # Binarize mask to ensure clean edges before processing
        mask_array = np.array(mask_img)
        mask_array = np.where(mask_array > 128, 255, 0).astype(np.uint8)
        mask_img = Image.fromarray(mask_array)

        # Load reference image if provided
        ref_img = None
        if ref_filename:
            if ref_filename not in file_map:
                raise HTTPException(status_code=400, detail=f"Reference file '{ref_filename}' not found in uploaded files.")
            ref_bytes = await file_map[ref_filename].read()
            ref_img = Image.open(io.BytesIO(ref_bytes)).convert("RGB")

        print(f"Executing Edit {i+1}/{len(edit_operations)}: Prompt='{prompt}'")

        try:
            # Resize mask to match current image if necessary
            if mask_img.size != current_image.size:
                mask_img = mask_img.resize(current_image.size, Image.Resampling.NEAREST)

            # --- Mask Processing (Dilation and Feathering) ---
            # Grow the mask so the model can seamlessly blend the edges into the background
            processed_mask = mask_img.copy()
            for _ in range(15):  # Dilate outward
                processed_mask = processed_mask.filter(ImageFilter.MaxFilter(3))
            processed_mask = processed_mask.filter(ImageFilter.GaussianBlur(radius=25)) # Blur the edges

            # Format prompt for Qwen VL (requires indicating which image is which)
            images_list = [current_image]
            final_prompt = prompt
            
            if ref_img:
                images_list.append(ref_img)
                # Prefix the prompt so the VL model knows to look at the second image for reference
                # Qwen uses <|image_pad|> internally when multiple images are passed.
                final_prompt = f"In the first image, {prompt}. Use the second image as a visual reference."

            # Qwen-Image-Edit Official Diffusers Generation Call
            # We pass `image=images_list` allowing it to ingest both the base and reference image
            output = pipe(
                image=images_list,
                mask_image=processed_mask,
                prompt=final_prompt,
                negative_prompt=" ",
                strength=1.0,           # Full replacement of the masked area
                num_inference_steps=20, # Standard steps for Qwen Edit
                true_cfg_scale=5.0      # Qwen uses true_cfg_scale for classifier-free guidance
            ).images[0]

            # --- Pristine Compositing ---
            # Even though FLUX generates the whole image, we stitch the edited portion back onto 
            # the original using the *processed_mask* (dilated and feathered).
            # This ensures the generated shadows/blending zones are preserved while 
            # keeping the rest of the background 100% untouched.
            output = output.resize(current_image.size, Image.Resampling.LANCZOS)
            current_image = Image.composite(output, current_image, processed_mask)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error during edit step {i+1}: {str(e)}")

    # Return final image
    out_byte_arr = io.BytesIO()
    current_image.save(out_byte_arr, format='PNG')
    out_byte_arr = out_byte_arr.getvalue()

    return Response(content=out_byte_arr, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
