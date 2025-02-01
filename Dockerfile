# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python buffering and ensure output appears in real time
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose any ports if necessary (e.g., for WandB monitoring or dashboards; adjust if needed)
EXPOSE 8080

# Set the default command to run your training script
CMD ["python", "grpo_demo.py"]
