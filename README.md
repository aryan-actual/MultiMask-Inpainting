# MultiMask-Inpainting: Qwen-Image Editing

A full-stack web application that allows you to easily edit, inpaint, and replace objects in images using the powerful **Qwen-Image** diffusion model.

This repository features a highly optimized backend utilizing the **InstantX Qwen-Image Inpainting ControlNet** combined with the **Lightning 4-steps LoRA** to deliver extremely fast, photorealistic edits while managing GPU memory efficiently via CPU offloading.

![App Screenshot](./frontend/public/vite.svg) *(UI built with Vite + React + Tailwind)*

## 🚀 Prerequisites

To run this application, you need a machine (preferably a Linux GPU server) with the following:
*   **GPU:** An NVIDIA GPU (e.g., RTX 3090, 4090, A100, L40S). The backend leverages PyTorch `enable_model_cpu_offload()` to prevent Out-Of-Memory errors.
*   **Python:** v3.10 or higher. You will also need Python development headers to compile dependencies like `triton`. (e.g., run `sudo apt install python3-dev` or `sudo apt install python3.12-dev` on Ubuntu/Debian).
*   **Node.js:** v22 or higher (Node 22 is required to prevent Vite `node:util` styleText errors).

---

## 🛠️ 1. Backend Setup (FastAPI + PyTorch)

The backend handles model downloading, image masking, and inference using the `diffusers` library.

1. Open a terminal and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the required Python packages (including the main branch of diffusers):
   ```bash
   pip install -r requirements.txt
   ```
4. Start the backend server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   *Note: On the very first run, this will automatically trigger the downloading of the base Qwen model, the ControlNet, and the Lightning LoRA into the `backend/models/` folder. This might take a while depending on your internet connection (approx. 20GB).*

---

## 🎨 2. Frontend Setup (React + Vite)

The frontend provides the drawing canvas for users to mask objects effortlessly. Node v22 or higher is required. We recommend using `nvm` (Node Version Manager) to install it:

```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc  # Or restart your terminal

# Install and use Node 22
nvm install 22
nvm use 22
```

1. Open a **new** terminal session and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the Node dependencies. **(Important: You must use the `--legacy-peer-deps` flag because of `react-canvas-draw`)**:
   ```bash
   npm install --legacy-peer-deps
   ```
3. Start the Vite development server:
   ```bash
   npm run dev -- --host
   ```
   *The `--host` flag exposes the app to your network so you can access it from your local browser by typing in the server's IP address.*

---

## 💡 How to Use

1. Open your web browser and navigate to `http://<YOUR_SERVER_IP>:5173`.
2. **Upload an Image** using the file picker.
3. Use the slider to adjust your **Brush Size** and paint over the object/area you want to replace. 
4. Type a **Prompt** describing what should appear in the masked area (e.g., *"A pile of vintage books"*).
5. Ensure **"Use Fast Mode (ControlNet + Lightning 4-Steps)"** is checked (it is checked by default).
6. Click **Generate Edit**. In just a few seconds, the model will seamlessly composite the new generated object into your original image!
