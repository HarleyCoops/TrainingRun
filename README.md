# GRPO Finetuning Docker Setup

This repository contains the necessary files to build and run a Docker container for GRPO finetuning.

## Building the Docker Image

To build the Docker image, run:

    docker build -t grpo-finetuning .

## Running the Docker Container

### Without GPU Support

You can run the container without GPU support by executing:

    docker run -it grpo-finetuning bash

### With GPU Support

To enable GPU access in the Docker container, ensure that the NVIDIA Container Toolkit is installed on your host machine. Then run:

    docker run --gpus all -it grpo-finetuning bash

This command starts the container with access to the available NVIDIA GPUs.

## Finetuning Job

Once inside the container, start the GRPO finetuning job by running:

    python3 grpo_demo.py

This command uses Python 3 (as typically expected on Ubuntu) to launch the finetuning process.

## Additional Information

- The Dockerfile is configured for Ubuntu 20.04 and installs Python 3.10 using the deadsnakes PPA.
- For full GPU functionality, verify that your NVIDIA drivers and the NVIDIA Container Toolkit are properly installed and up-to-date.
- More information about the NVIDIA Container Toolkit can be found here: https://docs.nvidia.com/datacenter/cloud-native/

Happy fine-tuning!
