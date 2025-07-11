FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app
ENV SHELL=/bin/zsh

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    zsh \
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

# Install oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install zsh plugins
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Configure zsh plugins
RUN sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

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

# Default command to keep container running with zsh
CMD ["zsh", "-c", "python --version && zsh"]
