# Use an official PyTorch runtime as a parent image (with CUDA if needed)
# You can change this to a CPU-only image if GPU is not available.
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PyVista (Xvfb for off-screen rendering) and basic tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variable for off-screen rendering
ENV PYVISTA_OFF_SCREEN=true
ENV DISPLAY=:99

# Copy the entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
