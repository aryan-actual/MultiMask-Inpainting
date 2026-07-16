import io
import json
from typing import List, Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response

# Ensure you have installed diffusers from source: pip install git+https://github.com/huggingface/diffusers.git
from diffusers import FluxKontextInpaintPipeline

app = FastAPI(title="FLUX.1 Kontext Sequential Editing API")

print("Initializing FLUX.1 Kontext Dev Pipeline...")
try:
    # We load the pipeline in bfloat16. 
    # For FLUX models (~12B parameters), CPU offloading is crucial for memory management.
    pipe = FluxKontextInpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    print("Pipeline loaded successfully with CPU offloading.")
except Exception as e:
    import traceback
    print("Failed to load FLUX.1 Kontext pipeline:")
    traceback.print_exc()
    pipe = None

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

            # FLUX.1 Kontext Dev Official Diffusers Generation Call
            output = pipe(
                image=current_image,
                mask_image=mask_img,
                prompt=prompt,
                image_reference=ref_img,
                strength=1.0,           # Full replacement of the masked area
                num_inference_steps=28, # Standard steps for FLUX.1 dev
                guidance_scale=3.5      # Standard CFG for FLUX.1 dev
            ).images[0]

            # The output becomes the input for the next edit
            current_image = output

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
