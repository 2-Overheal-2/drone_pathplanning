FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    xvfb \
    libgl1 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    libglu1-mesa \
    libxrender1 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libxi6 \
    libxtst6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/checkpoints /app/logs /app/configs

ENV DISPLAY=:99
ENV PYBULLET_EGL=1
ENV MESA_GL_VERSION_OVERRIDE=3.3

EXPOSE 6006

CMD ["tail", "-f", "/dev/null"]