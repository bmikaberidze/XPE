# Cross-Prompt Encoder for Low-Performing Languages

This repository contains the **code and experimental setup** for our paper accepted at  
*Findings of IJCNLP–AACL 2025*, with a preprint available on [arXiv:2508.10352](https://arxiv.org/abs/2508.10352).

**Authors:**  
Beso Mikaberidze†, Teimuraz Saghinadze†, Simon Ostermann\*+, Philipp Müller\*°

† Muskhelishvili Institute of Computational Mathematics, GTU (MICM)  
\* Deutsches Forschungszentrum für Künstliche Intelligenz (DFKI)  
\+ Center for European Research in Trusted AI (CERTAIN)  
° Max Planck Institute for Intelligent Systems

The paper studies **cross-lingual transfer learning** for low-performing languages using **parameter-efficient prompt-based methods**.
It presents an empirical study showing that a prompt-encoder with multi-source training improves transfer on low-performing languages in SIB-200, while a hybrid approach with a standard soft prompt broadens applicability.

The recommended and canonical way to run the code is via **Docker**, which ensures reproducibility
across both CPU-only and NVIDIA GPU environments.

**Contents:**  
[Setup](#setup) | [Usage](#usage) | [Artifacts](#artifacts) | [Reproducibility Notes](#reproducibility-notes) | [Cite](#cite)  | [Contact](#contact)

---

## Setup

### Clone the repository

```bash
git clone https://github.com/bmikaberidze/XPE.git
cd XPE
```

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

### Local Python environment (Optional)

> ⚠️ Local installation is **not guaranteed** to work on all platforms.
> The Docker setup below is the **officially supported** environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Docker-based setup (Recommended)

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

Download the XLM-R–tokenized SIB-200 dataset (no auth required); it will be stored at:
`./nlpka/datasets/storage/benchmarks/text_classification/topic/sib200_tokenized_xlmr`

```bash
python -m nlpka.datasets.scripts.sib200.download_tokenized
```

### Running experiments

All experiments are run via a single entrypoint.  
Change only `--supervision_regime` and the trailing arguments that specify the source dataset and methodology type:

```bash
python -m nlpka.models.scripts.peft.xpe.run \
   --config xlmr/finetune/peft/sib200_hybrid.xpe \
   --supervision_regime=<0|1> <source_dataset> <setup_id>
```

- `--supervision_regime`:
  - `0` → Zero-Shot XLT
  - `1` → Fully Supervised XLT
- `<source_dataset>`:
  - `sib200_enarzho`, `sib200_joshi5`, `sib200_xlmr_seen` (used in Zero-Shot XLT)
  - `sib200_joshi5_divers_24` (used in Fully Supervised XLT).
- `<setup_id>`:
  - `1` → SPT (Standard Soft Prompt)
  - `2` → D30 (DUAL, 30% XPE)
  - `3` → D70 (DUAL, 70% XPE)
  - `4` → XPE (Cross-Prompt Encoder)

Example (Zero-Shot XLT with XLM-R seen source languages and XPE):

```bash
python -m nlpka.models.scripts.peft.xpe.run \
  --config xlmr/finetune/peft/sib200_hybrid.xpe \
  --supervision_regime=0 sib200_xlmr_seen 4
```

---

## Artifacts

In addition to the code in this repository, we release the following research artifacts on
Hugging Face to support reproducibility and further analysis:

- **SIB-200 (XLM-R tokenized)**  
  Preprocessed and XLM-R–tokenized version of SIB-200 used in experiments.  
  Dataset: [mikaberidze/sib200_tokenized_xlmr](https://huggingface.co/datasets/mikaberidze/sib200_tokenized_xlmr)

- **LID-200 (derived from SIB-200)**  
  Language identification dataset constructed from SIB-200 and used for auxiliary LID experiments.  
  Dataset: [mikaberidze/lid200](https://huggingface.co/datasets/mikaberidze/lid200)

- **Trained soft prompts and prompt encoders**  
  Models will be made publicly available on Hugging Face (coming soon).

---

## Reproducibility Notes

- Experiments were run inside a Docker container based on an official PyTorch image.
- The same container supports **CPU-only** and **NVIDIA GPU** execution.
- GPU usage is enabled by running Docker with `--gpus all`; **full training is intended for GPU**.
- CPU runs are supported but are meant for debugging or small-scale sanity checks.
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
