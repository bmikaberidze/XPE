#!/bin/bash

# Set paths
IMAGE_PATH="/enroot/nvcr.io_nvidia_pytorch_24.08-py3.sqsh"
SAVE_PATH="/netscratch/bmikaberidze/nlpka.nvcr.io_nvidia_pytorch_24.08-py3.sqsh"
MOUNT_DIRS="/netscratch/bmikaberidze:/netscratch/bmikaberidze,/fscratch/bmikaberidze:/fscratch/bmikaberidze,$(pwd):$(pwd)"

echo "🚀 Launching SLURM job to modify container and save as new nlpka iamge ..."
srun --job-name="create_nlpka_image" \
     --container-image="$IMAGE_PATH" \
     --container-save="$SAVE_PATH" \
     --container-mounts="$MOUNT_DIRS" \
     --container-workdir="$(pwd)" \
     --mem=100000 \
     --time=01:00:00 \
     --immediate=3600 \
     --mail-type="BEGIN" \
     --mail-user="beso.mikaberidze@gmail.com" \
     --pty /bin/bash << 'EOF'

echo "📦 Updating and modifying the container environment..."
# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r runtime/requirements/nlpka.pip.require.txt

# Install system dependencies
apt-get update -y
apt-get install -y --no-install-recommends strace
apt-get clean

echo "✅ Container modification complete. Exiting..."
exit
EOF

echo "🎉 Container has been successfully modified and saved as new image to: $SAVE_PATH"


### List existing squash images
# ls -1 /enroot/
# 

### 
# ls -l /enroot/nvcr.io_nvidia_pytorch_24.08-py3.sqsh
# ls -l /netscratch/bmikaberidze/nlpka.nvcr.io_nvidia_pytorch_24.08-py3.sqsh

### Docker Hub image to squashfs
# srun \
#     --mem=64000 \
#     enroot import \
#     -o /netscratch/bmikaberidze/nlpka-pytorch-cuda11.7-cudnn8:latest.sqsh \
#     docker://besom/nlpka-pytorch-cuda11.7-cudnn8:latest

#
### Build custom Docker / OCI images
### Build the image into the enroot container and save it as a squashfs file:
# srun \
#     --mem=164000 \
#     --container-image=/enroot/podman+enroot.sqsh \
#     --container-mounts=/dev/fuse:/dev/fuse,/netscratch/bmikaberidze:/netscratch/bmikaberidze,/fscratch/bmikaberidze:/fscratch/bmikaberidze,"`pwd`":"`pwd`" \
#     --container-workdir="`pwd`" \
#     --pty bash
#
# podman build -f dockerfiles/nlpka-pytorch-cuda11.7-cudnn8.dockerfile -t nlpka-pytorch-cuda11.7-cudnn8:latest .
# enroot import -o /netscratch/bmikaberidze/nlpka-pytorch-cuda11.7-cudnn8.sqsh podman://nlpka-pytorch-cuda11.7-cudnn8
#
### Run the enroot container:
# srun --container-image=/netscratch/bmikaberidze/nlpka-pytorch-cuda11.7-cudnn8.sqsh --container-workdir="`pwd`" --container-mounts="`pwd`":"`pwd`" --pty /bin/bash
#
# scp -r /Users/besom/Documents/Projects/AI/group5_nlp/NLPKAInDocker/nlpka/datasets/storage/raw/all/text_near/subset/0.2 bmikaberidze@login1.pegasus.kl.dfki.de:/home/bmikaberidze/group5_nlp/NLPKAInDocker/nlpka/datasets/storage/raw/all/text_near/subset/0.2

