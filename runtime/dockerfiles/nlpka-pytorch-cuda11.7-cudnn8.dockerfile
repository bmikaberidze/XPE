# nvidia-smi
# nvidia-container-toolkit --version
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Create working directory
ENV WORKDIR=/nlpka_runner
WORKDIR ${WORKDIR}

# Activate new env and install requirenments
COPY ../requirements/nlpka.pip.require.txt .
RUN pip install --upgrade pip
RUN pip install -r nlpka.pip.require.txt
# RUN pip install -r nlpka.pip.require.txt --use-feature=2020-resolver

# Install additional packages
RUN apt-get install -y apt-transport-https ca-certificates
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends strace && \
    apt-get clean

# Copy the local project files into the Docker image:
# COPY ../ .

# Check if CUDA is available:
CMD [ "/bin/bash", "-c",  "python -c \"import torch; print(f'CUDA is available: {torch.cuda.is_available()}')\"" ]