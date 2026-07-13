import io
import os
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from diffusers import QwenImageEditInpaintPipeline, QwenImageControlNetModel, QwenImageControlNetInpaintPipeline
from download_models import ensure_model_downloaded, MODEL_PATH, QWEN_IMAGE_PATH, CONTROLNET_PATH, LORA_PATH

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the model is downloaded before trying to load it
ensure_model_downloaded()

# We only load the FAST pipeline to save VRAM and skip loading the standard 2511 pipeline.
pipe = None

print("Setting up Qwen Image ControlNet Inpaint Pipeline (Fast/Lightning)...")
try:
    # Load the base model with bitsandbytes 4-bit quantization to save massive VRAM
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    quantization_config = DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    controlnet = QwenImageControlNetModel.from_pretrained(
        CONTROLNET_PATH, 
        torch_dtype=torch.bfloat16
    )
    fast_pipe = QwenImageControlNetInpaintPipeline.from_pretrained(
        QWEN_IMAGE_PATH,
        controlnet=controlnet,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16
    )
    # Load the lightning 4-steps LoRA
    fast_pipe.load_lora_weights(LORA_PATH, weight_name="Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors")
    
    # When using 4-bit quantization, diffusers automatically handles device placement for the quantized models,
    # but we still move the non-quantized parts (like ControlNet) to CUDA if necessary, 
    # or use enable_model_cpu_offload() for extreme VRAM savings.
    fast_pipe.enable_model_cpu_offload()
    
    print("Fast Model loaded successfully (4-bit Quantized)!")
except Exception as e:
    import traceback
    print("=== FAST MODEL LOAD ERROR ===")
    traceback.print_exc()
    print("=============================")
    fast_pipe = None

@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...)
):
    if not pipe:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        image_data = await image.read()
        mask_data = await mask.read()
        
        base_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Open mask, convert to grayscale
        raw_mask = Image.open(io.BytesIO(mask_data)).convert("L")
        
        # Ensure mask is purely binary (0 or 255) where white > 128 is the drawn area
        mask_array = np.array(raw_mask)
        mask_array = np.where(mask_array > 128, 255, 0).astype(np.uint8)
        mask_img = Image.fromarray(mask_array)
        
        # Resize to max 1024 or 512 for memory efficiency, maintaining aspect ratio
        max_size = 1024
        if base_img.width > max_size or base_img.height > max_size:
            base_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        # Use NEAREST to prevent creating gray pixels along the edges of the mask
        mask_img = mask_img.resize(base_img.size, Image.Resampling.NEAREST)

        print(f"Running pipeline with prompt: '{prompt}'")
        
        # Save input images for debugging
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_outputs")
        os.makedirs(debug_dir, exist_ok=True)
        base_img.save(os.path.join(debug_dir, "debug_input_image.png"))
        mask_img.save(os.path.join(debug_dir, "debug_input_mask.png"))
        print(f"Saved debug images to {debug_dir}")

        out = pipe(
            prompt=prompt,
            image=base_img,
            mask_image=mask_img,
            num_inference_steps=25,
            guidance_scale=7.0,
            strength=1.0  # Force it to modify the masked area fully
        ).images[0]

        img_byte_arr = io.BytesIO()
        out.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")
    
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inpaint-fast")
async def inpaint_fast(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...)
):
    if not fast_pipe:
        raise HTTPException(status_code=503, detail="Fast Model not loaded")
        
    try:
        image_data = await image.read()
        mask_data = await mask.read()
        
        base_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        raw_mask = Image.open(io.BytesIO(mask_data)).convert("L")
        
        # Binarize mask
        mask_array = np.array(raw_mask)
        mask_array = np.where(mask_array > 128, 255, 0).astype(np.uint8)
        mask_img = Image.fromarray(mask_array)
        
        max_size = 1024
        if base_img.width > max_size or base_img.height > max_size:
            base_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        mask_img = mask_img.resize(base_img.size, Image.Resampling.NEAREST)

        print(f"Running fast pipeline with prompt: '{prompt}'")
        
        # Save debug images
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_outputs")
        os.makedirs(debug_dir, exist_ok=True)
        base_img.save(os.path.join(debug_dir, "debug_input_image_fast.png"))
        mask_img.save(os.path.join(debug_dir, "debug_input_mask_fast.png"))

        # In diffusers QwenImageControlNetInpaintPipeline, it expects `control_image` and `control_mask`
        out = fast_pipe(
            prompt=prompt,
            negative_prompt=" ",
            control_image=base_img,
            control_mask=mask_img,
            width=base_img.width,
            height=base_img.height,
            num_inference_steps=4,       # Lightning 4-steps
            true_cfg_scale=1.0,          # CFG is 1.0 for lightning models usually
            controlnet_conditioning_scale=1.0
        ).images[0]

        img_byte_arr = io.BytesIO()
        out.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")
    
    except Exception as e:
        print(f"Fast Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
