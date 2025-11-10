FROM continuumio/miniconda3

# Create working directory
WORKDIR /app

# Create the conda environment:
COPY ../requirements/nlpka.conda.env.yml .
RUN conda env create -f nlpka.conda.env.yml

# Make RUN commands use the new environment:
RUN echo "conda activate nlpka" >> ~/.profile
RUN echo "conda activate nlpka" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
RUN echo $SHELL

# Demonstrate the environment is activated:
RUN echo "Make sure nlpka is installed:"
# RUN conda run -n nlpka python -c "import transformers"
# RUN /bin/bash -c "source ~/.bashrc && python -c 'import transformers'"
# RUN source ~/.bashrc && python -c "import transformers"
RUN python -c "import transformers"

# Copy the local project files into the Docker image:
COPY ../ .

# Windows OS only
# Solution for files transferred from Windows
# Install dos2unix command and convert files to the Unix/Linux format:
RUN apt-get update && \
    apt-get install -y dos2unix
RUN find ../ -type f -name '*' -exec dos2unix {} \;

# run docker.entrypoint.sh bash script on container run:
# ENTRYPOINT ["./docker.entrypoint.sh"]

# podman build --security-opt label=disable -f nlpka.dockerfile -t nlpka
# podman run --security-opt label=disable -v $PWD:/app -it besom/nlpka
