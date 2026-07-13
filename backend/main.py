import io
import os
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from diffusers import QwenImageEditInpaintPipeline
from download_models import ensure_model_downloaded, MODEL_PATH

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

print("Setting up Qwen Image Edit Inpaint Pipeline...")
try:
    # Use bfloat16 or float16 based on what your GPU supports
    # The official repo might suggest bfloat16
    pipe = QwenImageEditInpaintPipeline.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16
    )
    if torch.cuda.is_available():
        pipe.to("cuda")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None

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
        
        # Diffusers typically expects mask to be single channel (L) or RGB. Let's convert to L to be safe,
        # where white is the area to inpaint.
        mask_img = Image.open(io.BytesIO(mask_data)).convert("L")
        
        # Resize to max 1024 or 512 for memory efficiency, maintaining aspect ratio
        max_size = 1024
        if base_img.width > max_size or base_img.height > max_size:
            base_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        mask_img = mask_img.resize(base_img.size, Image.Resampling.LANCZOS)

        print(f"Running pipeline with prompt: {prompt}")
        out = pipe(
            prompt=prompt,
            image=base_img,
            mask_image=mask_img,
            num_inference_steps=25,
            guidance_scale=7.0
        ).images[0]

        img_byte_arr = io.BytesIO()
        out.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")
    
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
