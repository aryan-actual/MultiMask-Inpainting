import os
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "Qwen-Image-Edit-2511")

def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Downloading {MODEL_ID}...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        try:
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=MODEL_PATH,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
    else:
        print(f"Model already exists at {MODEL_PATH}.")

if __name__ == "__main__":
    ensure_model_downloaded()
