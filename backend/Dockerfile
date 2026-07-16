# Use the official Python 3.12 slim image as the base
FROM python:3.12-slim

# Set environment variables to prevent Python from writing pyc files to disc
# and to prevent stdout/stderr from being buffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for building Python packages (like triton, numpy, PIL)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements_flux.txt .

# Install the Python dependencies
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_flux.txt

# Copy the rest of the application code
COPY . .

# Expose the port the FastAPI server will run on
EXPOSE 8000

# Command to run the FastAPI server using Uvicorn
CMD ["uvicorn", "flux_api:app", "--host", "0.0.0.0", "--port", "8000"]