import os
from huggingface_hub import snapshot_download, hf_hub_download

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "Qwen-Image-Edit-2511")
QWEN_IMAGE_PATH = os.path.join(MODELS_DIR, "Qwen-Image")
CONTROLNET_PATH = os.path.join(MODELS_DIR, "Qwen-Image-ControlNet-Inpainting")
LORA_PATH = os.path.join(MODELS_DIR, "LoRA")

def ensure_model_downloaded():
    # 1. Download original Qwen-Image-Edit
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Downloading Qwen/Qwen-Image-Edit-2511...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        try:
            snapshot_download(
                repo_id="Qwen/Qwen-Image-Edit-2511",
                local_dir=MODEL_PATH,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
    else:
        print(f"Model already exists at {MODEL_PATH}.")

    # 2. Download Base Qwen-Image (for ControlNet alternative)
    if not os.path.exists(QWEN_IMAGE_PATH) or not os.listdir(QWEN_IMAGE_PATH):
        print(f"Model not found at {QWEN_IMAGE_PATH}. Downloading Qwen/Qwen-Image...")
        os.makedirs(QWEN_IMAGE_PATH, exist_ok=True)
        try:
            snapshot_download(
                repo_id="Qwen/Qwen-Image",
                local_dir=QWEN_IMAGE_PATH,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print("Qwen-Image Base downloaded successfully!")
        except Exception as e:
            print(f"Error downloading base model: {e}")

    # 3. Download InstantX ControlNet
    if not os.path.exists(CONTROLNET_PATH) or not os.listdir(CONTROLNET_PATH):
        print(f"Model not found at {CONTROLNET_PATH}. Downloading InstantX/Qwen-Image-ControlNet-Inpainting...")
        os.makedirs(CONTROLNET_PATH, exist_ok=True)
        try:
            snapshot_download(
                repo_id="InstantX/Qwen-Image-ControlNet-Inpainting",
                local_dir=CONTROLNET_PATH,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print("ControlNet downloaded successfully!")
        except Exception as e:
            print(f"Error downloading controlnet: {e}")

    # 4. Download Lightning 4-Step LoRA
    if not os.path.exists(LORA_PATH) or not os.listdir(LORA_PATH):
        print(f"Model not found at {LORA_PATH}. Downloading Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors...")
        os.makedirs(LORA_PATH, exist_ok=True)
        try:
            hf_hub_download(
                repo_id="lightx2v/Qwen-Image-Lightning",
                filename="Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
                local_dir=LORA_PATH,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print("LoRA downloaded successfully!")
        except Exception as e:
            print(f"Error downloading lora: {e}")

if __name__ == "__main__":
    ensure_model_downloaded()

