# Cross-Prompt Encoder for Low-Performing Languages

This repository contains the **code and experimental setup** for our paper accepted at *Findings of IJCNLP–AACL 2025*:  
📄 Preprint: [arXiv:2508.10352](https://arxiv.org/abs/2508.10352)

**Authors**  
Beso Mikaberidze†, Teimuraz Saghinadze†, Simon Ostermann\*+, Philipp Müller\*°

† Muskhelishvili Institute of Computational Mathematics, GTU (MICM)  
\* Deutsches Forschungszentrum für Künstliche Intelligenz (DFKI)  
\+ Center for European Research in Trusted AI (CERTAIN)  
° Max Planck Institute for Intelligent Systems

The paper studies **cross-lingual transfer learning** for low-performing languages using **parameter-efficient prompt-based methods**.
It presents an empirical study showing that a prompt-encoder with multi-source training improves transfer on low-performing languages in SIB-200, while a hybrid approach with a standard soft prompt broadens applicability.

The recommended and canonical way to run the code is via **Docker**, which ensures reproducibility
across both CPU-only and NVIDIA GPU environments.

**Contents**  
- [Setup](#setup)  
- [Usage](#usage)  
- [Cite](#cite)  
- [Contact](#contact)


---

## Setup

### Clone the repository

```bash
git clone https://github.com/bmikaberidze/XPE.git
cd XPE
```

---

### Environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Set your Weights & Biases API key:

1. Obtain a key from: https://wandb.ai/authorize  
2. Add it to `.env`:
   ```
   WANDB_API_KEY=your_key_here
   ```

---

### Local Python environment (Optional)

> ⚠️ Local installation is **not guaranteed** to work on all platforms.
> The Docker setup below is the **officially supported** environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Docker-based setup (recommended)

#### Build the Docker image

```bash
docker build -t xpe .
```

#### Run an interactive container (CPU)

```bash
docker run -it --rm \
  -v $(pwd):/xpe_runner \
  -w /xpe_runner \
  xpe \
  bash
```

#### Run with NVIDIA GPU support

Requires:
- NVIDIA GPU
- NVIDIA drivers
- `nvidia-container-toolkit`

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/xpe_runner \
  -w /xpe_runner \
  xpe \
  bash
```

Inside the container, your project files are available at `/xpe_runner`.

---

## Usage

### Dataset preparation

Download preprocessed SIB-200 dataset:

```bash
python -m nlpka.datasets.scripts.sib200.download_tokenized
```

---

### Running experiments

Use the same entrypoint, changing only `--supervision_regime` and the trailing arguments that specify the source dataset and methodology type:

```bash
python -m nlpka.models.scripts.peft.xpe.run \
   --config xlmr/finetune/peft/sib200_hybrid.xpe \
   --supervision_regime=<0|1> <dataset_arg> <setup_id>
```

- `--supervision_regime=0` (Zero-Shot XLT):
  - `<dataset_arg>`: one of `sib200_enarzho`, `sib200_joshi5`, `sib200_xlmr_seen` (source dataset names).
  - `<setup_id>`: choose one of:
    - `1` = SPT: standard soft prompt.
    - `2` = D30: dual approach with 30% XPE.
    - `3` = D70: dual approach with 70% XPE.
    - `4` = XPE: full Cross-Prompt Encoder.

- `--supervision_regime=1` (Fully Supervised XLT):
  - `<dataset_arg>`: use `sib200_joshi5_divers_24`.
  - `<setup_id>`: required; choose 1–4 as above (same meaning).

---

### Notes on reproducibility

- Experiments were run inside a Docker container based on an official PyTorch runtime image.
- The same container supports **CPU-only** and **NVIDIA GPU** execution.
- GPU usage is enabled by running Docker with `--gpus all`.
- No support is provided for non-NVIDIA GPUs.

---

## Cite

If you use this code, please cite:

B. Mikaberidze, T. Saghinadze, S. Ostermann, and P. Müller. 2025. *Cross-Prompt Encoder for Low-Performing Languages.* Findings of AACL 2025. arXiv:2508.10352.

BibTeX:
```bibtex
@misc{mikaberidze2025crosspromptencoderlowperforminglanguages,
  title         = {Cross-Prompt Encoder for Low-Performing Languages},
  author        = {Beso Mikaberidze and Teimuraz Saghinadze and Simon Ostermann and Philipp Muller},
  year          = {2025},
  eprint        = {2508.10352},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2508.10352},
}
```

---

## Contact
- besik.mikaberidze@dfki.de
- beso.mikaberidze@gmail.com
- mueller@is.mpg.de

Feel free to reach out with questions, issues running the code, or requests for clarifications about the experiments.
