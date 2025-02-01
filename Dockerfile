FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Install software-properties-common and add deadsnakes PPA for Python 3.10
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

# Install system dependencies: Python 3.10, pip, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*
