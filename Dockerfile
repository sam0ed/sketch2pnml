# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set up a new user named "user" with user ID 1000 (HF requirement)
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Install system dependencies for OpenCV, Graphviz, and other packages
# Switch to root temporarily for system package installation
USER root
RUN apt-get update && apt-get install -y \
    graphviz \
    libgraphviz-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Switch back to user
USER user

# Copy requirements first for better caching (with proper ownership)
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (with proper ownership)
COPY --chown=user . $HOME/app

# Create necessary directories with proper permissions
RUN mkdir -p data/demos data/working data/output data/templates

# Set environment variables
ENV PYTHONPATH=$HOME/app
ENV PYTHONUNBUFFERED=1

# Expose the port that Gradio uses
EXPOSE 7860

# Run the application
CMD ["python", "app.py"] 