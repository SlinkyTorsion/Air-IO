FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    gsfonts \
    fontconfig \
    fonts-dejavu-core \
    fonts-liberation \
    fonts-urw-base35 \
    && fc-cache -fv && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire Air-IO project
COPY . .

# Create directories
RUN mkdir -p /app/experiments /app/EKFresult

# Set permissions
RUN chmod +x /app

# Default command to keep container running
CMD ["python", "--version"]
