# Use NVIDIA's PyTorch base image which comes with CUDA toolkit and PyTorch pre-installed
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a requirements file with proper newlines using printf
RUN printf "transformers==4.48.2\ndatasets\npeft==0.13.0\ntrl\nwandb\nvllm==0.6.6.post1\n" > requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python && pip install opencv-python-headless==4.7.0.72

# Copy your Python script into the container
COPY grpo_demo.py .

# Set the default command to run your script
CMD ["python", "grpo_demo.py"]
