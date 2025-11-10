#!/bin/bash
set -e  # Exit on error

DOCKERFILE="runtime/dockerfiles/nlpka-pytorch-cuda11.7-cudnn8.dockerfile"
IMAGE_NAME="nlpka-pytorch-cuda11.7-cudnn8:latest"

echo "🚀 Building Podman image: $IMAGE_NAME from $DOCKERFILE..."
podman build -f $DOCKERFILE -t $IMAGE_NAME .
# podman build -f dockerfiles/nlpka-pytorch-cuda11.7-cudnn8.dockerfile -t nlpka-pytorch-cuda11.7-cudnn8:latest .

echo "✅ Build completed!"
