# Qwen-Image Inpainting Frontend

This is the frontend application for the MultiMask-Inpainting project, built with React, Vite, and Tailwind CSS.

## 🚀 Prerequisites: Node.js and NVM Installation

We recommend using Node Version Manager (NVM) to install and manage your Node.js versions. **Node v22 or higher is required** to prevent Vite `node:util` styleText errors.

### 1. Install NVM (Node Version Manager)

Run the following command in your terminal to install `nvm` (Linux/macOS):
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```
*(For Windows, it is recommended to use [nvm-windows](https://github.com/coreybutler/nvm-windows))*

After the installation script finishes, load NVM by either closing and reopening your terminal, or by running:
```bash
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
```

### 2. Install Node.js

Now, install Node.js v22 using NVM:
```bash
nvm install 22
nvm use 22
```

Verify your installation:
```bash
node -v # Should print v22.x.x
npm -v
```

## 🛠️ Setup Instructions

1. Make sure you are in the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Install the required Node dependencies. **(Important: You must use the `--legacy-peer-deps` flag because of `react-canvas-draw`)**:
   ```bash
   npm install --legacy-peer-deps
   ```

3. Start the Vite development server:
   ```bash
   npm run dev -- --host
   ```
   *The `--host` flag exposes the app to your network so you can access it from your local browser via the server's IP address (e.g., `http://<YOUR_SERVER_IP>:5173`).*
