import torch
from diffusers import QwenImageEditInpaintPipeline, AutoPipelineForInpainting

try:
    print("Trying QwenImageEditInpaintPipeline")
    pipe = QwenImageEditInpaintPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.float16)
    print("Success Qwen-Image-Edit-2511")
except Exception as e:
    print(f"Failed 2511: {e}")
    try:
        pipe = AutoPipelineForInpainting.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.float16)
        print("Success Qwen-Image-Edit auto")
    except Exception as e:
        print(f"Failed auto: {e}")
