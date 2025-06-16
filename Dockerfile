FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # For PyBullet and OpenGL
    xvfb \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libxrender1 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    # For OpenCV
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    # Build tools
    build-essential \
    cmake \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints logs configs

# Set environment variables for headless rendering
ENV DISPLAY=:99
ENV PYBULLET_EGL=1
ENV MESA_GL_VERSION_OVERRIDE=3.3

# Expose port for TensorBoard
EXPOSE 6006

# Keep container running
CMD ["tail", "-f", "/dev/null"]
