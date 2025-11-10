#!/bin/bash
set -e  # Exit on error

IMAGE_NAME="nlpka-pytorch-cuda11.7-cudnn8:latest"
CONTAINER_NAME="nlpka_runner"
DEFAULT_CONDA_PATH="$HOME/Documents/site-packages"  # Default Conda path

# Check if a Conda path argument is provided
if [ "$1" == "--no-conda" ]; then
    USE_CONDA=false
    echo "⚠️  Running without mounting Conda site-packages."
else
    USE_CONDA=true
    CONDA_PATH=${1:-$DEFAULT_CONDA_PATH}  # Use provided path or default
    echo "📦 Using Conda site-packages from: $CONDA_PATH"
fi

echo "🚀 Starting Podman container from image: $IMAGE_NAME"

PODMAN_CMD="podman run \
    --device nvidia.com/gpu=all \
    --group-add keep-groups \
    --runtime=/usr/bin/crun \
    --security-opt label=type:nvidia_container_t \
    -v \"$PWD:/nlpka_runner:z\" \
    -it --rm $IMAGE_NAME bash"

# Add Conda mount if enabled
if [ "$USE_CONDA" = true ]; then
    PODMAN_CMD="podman run \
        --device nvidia.com/gpu=all \
        --group-add keep-groups \
        --runtime=/usr/bin/crun \
        --security-opt label=type:nvidia_container_t \
        -v \"$PWD:/nlpka_runner:z\" \
        -v \"$CONDA_PATH:/opt/conda/lib/python3.10/site-packages\" \
        -it --rm $IMAGE_NAME bash"
fi

# Execute the final command
eval $PODMAN_CMD

echo "✅ Podman container exited."

# Podman Commands ======== >
#
# podman run --device nvidia.com/gpu=all --group-add keep-groups --runtime=/usr/bin/crun --security-opt label=type:nvidia_container_t -v $PWD:/nlpka_runner:z -it --rm e2aa616b86e0 bash # nlpka-pytorch-cuda11.7-cudnn8:latest bash
# podman run --device nvidia.com/gpu=all --group-add keep-groups --runtime=/usr/bin/crun --security-opt label=type:nvidia_container_t -v ~/Documents/site-packages:/opt/conda/lib/python3.10/site-packages -v $PWD:/nlpka_runner:z -it --rm e2aa616b86e0 bash # nlpka-pytorch-cuda11.7-cudnn8:latest bash
# podman run \
#     --device nvidia.com/gpu=all \
#     --group-add keep-groups \
#     --runtime=/usr/bin/crun \
#     --security-opt label=type:nvidia_container_t \
#     -v ~/Documents/site-packages:/opt/conda/lib/python3.10/site-packages \
#     -v $PWD:/nlpka_runner:z -it --rm nlpka-pytorch-cuda11.7-cudnn8:latest bash
# 
# podman cp <container_id>:/opt/conda/lib/python3.1/site-packages/ /home/mikaberidze/Documents
# 
# podman login -u besom 
# podman image tag nlpka-pytorch-cuda11.7-cudnn8:latest besom/nlpka-pytorch-cuda11.7-cudnn8:latest
# podman image push besom/nlpka-pytorch-cuda11.7-cudnn8:latest
# 
# --security-opt label=disable
# 
# Free up the space
# 
# podman system prune -a && restorecon -RvF ~/.local && df -h ~/ 
# podman system reset && df -h ~/ 
## rm -r ~/.local/share/containers
# 
# df -h ~/                                          # Check that your home folder is mapped to an additional disk.
# du -sh ~/                                         # Shows the size of youre home folder.
# du -sh ~/*                                        # Shows the sizes of folders and files in youre home folder.
# du -h --max-depth=1 ~ | sort -hr                  # lagre dirs
# find ~ -type f -size +1000M -exec ls -lh {} +     # large files

# nvidia-container-toolkit --version
# 
# nvidia-smi 
# watch -n 1 nvidia-smi
# nvidia-smi -L
# nvidia-smi -L
# nvidia-smi nvlink -s
# nvidia-smi topo -m