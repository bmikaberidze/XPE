#!/bin/bash
#SBATCH --overcommit                    # Allow resource sharing
#SBATCH --wait                          # Ensure job waits instead of failing
#SBATCH --partition=RTXA6000
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user="beso.mikaberidze@gmail.com"
#SBATCH --output="runtime/clusters/pegasus/shell/logs/sbatch/%A_%a.out"     # %x = job name, %A = job ID, %a = array task ID
#SBATCH --error="runtime/clusters/pegasus/shell/logs/sbatch/%A_%a.err"      # %x = job name, %A = job ID, %a = array task ID
#SBATCH --job-name="sbatch_nlpka"
#SBATCH --array=1-4 #--array=0-2,5,7-9 %10

## Partitions:
# ~7:   V100-16GB | V100-32GB
# ~15:  RTX3090 | RTXA6000 | batch (Quadro RTX 6000)
# ~25:  L40S
# ~80:  A100-40GB | A100-PCI | A100-80GB
# ~200: H100 | H100-PCI | H100-Trails | H200

# runtime/clusters/pegasus/shell/run.sh --no-gpu                                            ## Run container without GPU
# runtime/clusters/pegasus/shell/run.sh --site-packages                                     ## Run container with site-packages mounted
# squeue -u bmikaberidze -l                                                                 ## List slurm jobs
# cp -r /usr/local/lib/python3.10/dist-packages/* /fscratch/bmikaberidze/site-packages/     ## Copy site-packages from running container
# ln -s /fscratch/bmikaberidze/project_name ~/                                              ## Symlink project to home directory

PARTITION="RTXA6000"
CPUS_PER_TASK=4
GPUS=1
MEMORY=35000
TIME="04:00:00" # Must be <= 4 hours
INTERACTIVE_GPU_TIME="01:50:00"
NODES=1
NTASKS_PER_NODE=1
JOB_NAME="containerize_nlpka_script"
TIME_ID=$(date +"%Y%m%d_%H%M%S")
LOG_PATH="runtime/clusters/pegasus/shell/logs/interactive/$TIME_ID.out"
IMMEDIATE=300 # Must be <= 3600
SRUN_ARGS=""

# Container and Mount Paths
UNAME="bmikaberidze"
IMAGE_NAME="nlpka.nvcr.io_nvidia_pytorch_24.08-py3.sqsh"
IMAGE_PATH="/netscratch/$UNAME/$IMAGE_NAME"
MOUNT_DIRS="/dev/fuse:/dev/fuse,/home/$UNAME:/home/$UNAME,/fscratch/$UNAME:/fscratch/$UNAME,/netscratch/$UNAME:/netscratch/$UNAME"
WORKDIR="$(pwd)"

# Check if site-packages mount argument is provided
if [ "$1" == "--site-packages" ]; then
     echo "⚠️  Running with site-packages mounted."
     MOUNT_PCKAGES="/fscratch/bmikaberidze/site-packages:/usr/local/lib/python3.10/dist-packages"
     MOUNT_DIRS="$MOUNT_DIRS,$MOUNT_PCKAGES"
     shift 1 # Shift arguments so that $2 becomes $1 (remove --site-packages from args)
fi

# Check if site-packages mount argument is provided
if [ "$1" == "--no-gpu" ]; then
     echo "⚠️  Running without GPU."
     GPUS=0
     shift 1
fi

# Default command: Launch interactive bash shell
DEFAULT_CMD="/bin/bash"
CMD=${1:-$DEFAULT_CMD}

# Detect SBATCH or interactive shell
if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "📅 Running in SBATCH mode"

else
    echo "🖥️  Running in INTERACTIVE mode"

    # Append redirection or prompt for interactive GPU shell
    if [[ "$CMD" != "$DEFAULT_CMD" ]]; then
        # Case 1: User provided a real command → redirect output
        echo "📝 Redirecting output to log file: $LOG_PATH"
        CMD+=" > \"$LOG_PATH\" 2>&1"
    else
        # Case 2: Default shell command → check for interactive GPU usage
        if [[ "$GPUS" -gt 0 ]]; then
            read -p "⚠️  You are running a GPU job in INTERACTIVE mode. Proceed? (y/n): " CONFIRM
            if [[ "$CONFIRM" != "y" ]]; then
                echo "❌ Aborting SLURM job."
                exit 1
            fi
            TIME=$INTERACTIVE_GPU_TIME
            echo "⏰ Job time limited to $TIME minutes!"
        fi
    fi

    # Build SRUN_ARGS for interactive mode
    SRUN_ARGS="--partition=$PARTITION --nodelist=$NODELIST --time=$TIME --immediate=$IMMEDIATE \
               --mem=$MEMORY --gpus=$GPUS --cpus-per-task=$CPUS_PER_TASK \
               --ntasks-per-node=$NTASKS_PER_NODE --nodes=$NODES --pty"
fi

# Final execution info
echo "🚀 Starting SLURM job with container..."
# echo "🖼️  Using container image: $IMAGE_PATH"
# echo "📂 Mounted directories: $MOUNT_DIRS"
# echo "🔧 Running command: $CMD"
echo "      Using image: $IMAGE_PATH"
echo "      Running command: $CMD"
echo "      Mounted directories: $MOUNT_DIRS"

srun --job-name="$JOB_NAME" \
     --container-image="$IMAGE_PATH" \
     --container-mounts="$MOUNT_DIRS" \
     --container-workdir="$WORKDIR" \
     $SRUN_ARGS \
     bash -c "$CMD"

# -n, --ntasks=ntasks         number of tasks to run
#     --ntasks-per-node=n     number of tasks to invoke on each node
# -N, --nodes=N               number of nodes on which to run (N = min[-max])
# -c, --cpus-per-task=ncpus   number of cpus required per task
#     --cpus-per-gpu=n        number of CPUs required per allocated GPU
# -G, --gpus=n                count of GPUs required for the job
#     --gpus-per-node=n       number of GPUs required per allocated node
#     --gpus-per-task=n       number of GPUs required per spawned task
#     --mem=MB                minimum amount of real memory
#     (--mem-per-gpu=MB)      DO NOT USE, BUGGY!
#                             real memory required per allocated GPU
#     --mem-per-cpu=MB        maximum amount of real memory per allocated
#                             cpu required by the job.
#                             --mem >= --mem-per-cpu if --mem is specified.
#     --time=1-00:00          job runtime limit [d-hh:mm] (default 7 days for
#                             non-privileged partitions, 1-3 for A100)
#     -K, --kill-on-bad-exit      kill the job if any task terminates with a
#                                 non-zero exit code

echo "✅ Podman container exited."

# 
# Sorted by BF16 Speed (Slowest → Fastest):
# 
# V100-16       — ~7.8      TFLOPS (BF16 not natively supported, emulated via FP16)
# V100-32       — ~7.8      TFLOPS (same as above; only VRAM differs)
# RTX 3090      — ~16–17    TFLOPS (BF16 support via Ampere arch)
# RTX A6000     — ~19.5     TFLOPS (better than 3090 due to higher bandwidth, though similar core arch)
# L40S          — ~24       TFLOPS (Ada Lovelace; mixed precision improvements)
# A100-40GB     — ~78       TFLOPS (native BF16 via Tensor Cores)
# A100-80GB     — ~78       TFLOPS (same compute, double VRAM)
# H100          — ~198      TFLOPS (Tensor Core BF16, Hopper arch)
# H200          — ~198      TFLOPS (same as H100 but with faster HBM3 memory and larger capacity)