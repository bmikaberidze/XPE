# nvidia-smi
# nvidia-container-toolkit --version
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel


# Create working directory
ENV WORKDIR=/xpe_runner
WORKDIR ${WORKDIR}

# Activate new env and install requirenments
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# RUN pip install -r requirements.txt --use-feature=2020-resolver

# Install additional packages (git needed for GitPython; strace for debugging)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        git \
        strace \
        apt-transport-https \
        ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the local project files into the Docker image:
# COPY ../ .

# Check if CUDA is available:
CMD [ "/bin/bash", "-c",  "python -c \"import torch; print(f'CUDA is available: {torch.cuda.is_available()}')\"" ]