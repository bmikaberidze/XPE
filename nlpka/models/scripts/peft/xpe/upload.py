'''
usage: 
python -m nlpka.models.scripts.peft.xpe.upload
'''
import os
import shutil
import tempfile
from huggingface_hub import HfApi, upload_folder
from nlpka.configs.scripts.xpe_utils import grouped_trained_models, trained_model_paths

HF_TOKEN = os.environ.get("HF_TOKEN")  # optional
api = HfApi(token=HF_TOKEN)

BASE_MODEL = "FacebookAI/xlm-roberta-large"
MODEL_GROUPS = grouped_trained_models
MODEL_PATHS = trained_model_paths
REPOS = {
    "XPE_SEEN": "mikaberidze/xlmr-large-sib200-peft-xpe-seen",
    "SPT_SEEN": "mikaberidze/xlmr-large-sib200-peft-spt-seen",
    "XPE_J5":   "mikaberidze/xlmr-large-sib200-peft-xpe-joshi5",
    "SPT_J5":   "mikaberidze/xlmr-large-sib200-peft-spt-joshi5",
}

ALLOWED_FILES = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "README.md",
}

RECREATE_REPOS = False
UPLOAD_ALL_SEEDS = True

# === CREATE REPOS IF NEEDED ===
for repo in REPOS.values():
    try:
        if RECREATE_REPOS:
            try:
                api.delete_repo(repo_id=repo, repo_type="model")
                print(f"Deleted repo: {repo}")
            except Exception:
                pass
        api.create_repo(repo_id=repo, repo_type="model", private=False)
        print(f"Created repo: {repo}")
    except Exception:
        print(f"Repo already exists: {repo}")

def patch_adapter_config(
    path: str,
    group: str,
    base_model: str = BASE_MODEL,
):
    """
    Patch adapter_config.json to:
    - fix base_model_name_or_path
    - remove redundant / implicit XPE options
    - deduplicate modules_to_save
    - add encoder_ratio flag (0=SPT, 1=XPE)
    """

    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return

    import json

    with open(cfg_path) as f:
        cfg = json.load(f)

    # -------------------------------------------------
    # 1) Canonical base model (MANDATORY)
    # -------------------------------------------------
    cfg["base_model_name_or_path"] = base_model

    # -------------------------------------------------
    # 2) Remove redundant / implicit options
    #    (they are either defaults or encoded elsewhere)
    # -------------------------------------------------
    REDUNDANT_KEYS = [
        "encoder_embedding_type",                # implied by encoder_ratio
        "encoder_embedding_dedicated_ratio",     # implied
        "encoder_embedding_neutral_dedication",  # implied
        "encoder_embedding_skip_for_dedicated",  # implied
        "encoder_embedding_dedicated_init",      # implied
        "encoder_embedding_dedicated_init_key",  # implied
        "num_tasks",                             # task_keys already define it
    ]

    for k in REDUNDANT_KEYS:
        cfg.pop(k, None)

    # -------------------------------------------------
    # 3) Deduplicate modules_to_save (clean PEFT config)
    # -------------------------------------------------
    if "modules_to_save" in cfg:
        cfg["modules_to_save"] = sorted(set(cfg["modules_to_save"]))

    # -------------------------------------------------
    # 4) Add new explicit XPE/SPT switch
    # -------------------------------------------------
    encoder_ratio = 1 if "XPE" in group else 0
    cfg["encoder_ratio"] = int(encoder_ratio)  # 0 = SPT, 1 = XPE

    # -------------------------------------------------
    # 5) Write back (stable formatting)
    # -------------------------------------------------
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def patch_readme(path, repo_id, group, seed_id):
    readme = os.path.join(path, "README.md")

    method = "XPE" if "XPE" in group else "SPT"
    method_title = "Cross-Prompt Encoder (XPE)" if method == "XPE" else "Standard Soft Prompt (SPT)"
    source_language_group = "XLM-R Seen Languages" if "SEEN" in group else "Joshi5 (7 Highest Resource Languages)"

    content = f"""
---
base_model: FacebookAI/xlm-roberta-large
library_name: peft
tags:
- peft
- soft-prompt
- prompt-encoder
- xlm-roberta
- sib200
- multilingual
- cross-lingual-transfer
---

# {method_title} for XLM-R (large) — SIB-200

This model is released as part of the paper:  
**Cross-Prompt Encoder for Low-Performing Languages**  
*Findings of IJCNLP–AACL 2025*; preprint at [arXiv:2508.10352](https://arxiv.org/abs/2508.10352).

The paper studies cross-lingual transfer learning for low-performing languages using
parameter-efficient, prompt-based methods on the SIB-200 benchmark.

This repository provides the trained **{method_title}** adapter used in the study.
It is a parameter-efficient soft-prompt model designed to be loaded on top of a
frozen XLM-R (large) backbone, and contains the learned:
- Soft Prompt  
{"- Prompt Encoder  " if method == "XPE" else ""}
- Classification Head  

---

## Model Details

- **Adaptation:** Parameter-Efficient Fine-Tuning (PEFT), {method_title}  
- **Backbone:** [`FacebookAI/xlm-roberta-large`](https://huggingface.co/FacebookAI/xlm-roberta-large)  
- **Task:** Multilingual Topic Classification  
- **Training Regime:** Zero-Shot Cross-Lingual Transfer  
- **Source Language Group:** {source_language_group}  
- **Benchmark:** [`Davlan/sib200`](https://huggingface.co/datasets/Davlan/sib200)  

### Seeds

The **`main`** branch corresponds to **`seed-01`**  
Additional random seeds are available as branches: **`seed-02`**, **`seed-03`**, …, **`seed-10`**

---

## Usage


This model is part of the experimental framework introduced in the paper and is
intended to be loaded and used via the canonical
[codebase](https://github.com/bmikaberidze/XPE). 

---

## Related Resources
- **Research Paper:** [`Cross-Prompt Encoder for Low-Performing Languages`](https://arxiv.org/abs/2508.10352)  
- **Code Repository:** [`bmikaberidze/XPE`](https://github.com/bmikaberidze/XPE)  
- **Benchmark:** [`Davlan/sib200`](https://huggingface.co/datasets/Davlan/sib200)  
- **Preprocessed Dataset:** [`mikaberidze/sib200-xlmr-tokenized`](https://huggingface.co/datasets/mikaberidze/sib200-xlmr-tokenized)
- **Related Models:** 
    - [`mikaberidze/xlmr-large-sib200-peft-xpe-seen`](https://huggingface.co/mikaberidze/xlmr-large-sib200-peft-xpe-seen)
    - [`mikaberidze/xlmr-large-sib200-peft-spt-seen`](https://huggingface.co/mikaberidze/xlmr-large-sib200-peft-spt-seen)
    - [`mikaberidze/xlmr-large-sib200-peft-xpe-joshi5`](https://huggingface.co/mikaberidze/xlmr-large-sib200-peft-xpe-joshi5)
    - [`mikaberidze/xlmr-large-sib200-peft-spt-joshi5`](https://huggingface.co/mikaberidze/xlmr-large-sib200-peft-spt-joshi5)

---

## Citation

If you use this model, please cite:

""" + \
"""
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

**Beso Mikaberidze** — beso.mikaberidze@gmail.com  
**Philipp Muller** — mueller@is.mpg.de

"""

    with open(readme, "w") as f:
        f.write(content)

def ensure_branch(repo_id: str, branch: str):
    try:
        api.create_branch(
            repo_id=repo_id,
            branch=branch,
            repo_type="model",
        )
        print(f"Created branch {branch} in {repo_id}")
    except Exception:
        # branch already exists
        pass

# === UPLOAD ===
for group, uuids in MODEL_GROUPS.items():
    repo_id = REPOS[group]

    for i, uuid in enumerate(uuids, start=1):
        seed_branch = f"seed-{i:02d}"
        local_path = MODEL_PATHS[uuid]
        
        is_first_seed = (i == 1)

        print(f"Uploading {uuid} → {repo_id}@{seed_branch}")
        ensure_branch(repo_id, seed_branch) 

        with tempfile.TemporaryDirectory() as tmp:
            for fname in ALLOWED_FILES:
                src = os.path.join(local_path, fname)
                if os.path.exists(src):
                    shutil.copy(src, tmp)

            patch_readme(tmp, repo_id, group, i)
            patch_adapter_config(tmp, group, base_model=BASE_MODEL)

            upload_folder(
                folder_path=tmp,
                repo_id=repo_id,
                repo_type="model",
                revision=seed_branch,
                commit_message=f"Add {group} seed {i:02d}",
            )

            # ALSO upload seed-01 to main
            if is_first_seed:
                print(f"→ Setting {repo_id}@main = seed-01")
                upload_folder(
                    folder_path=tmp,
                    repo_id=repo_id,
                    repo_type="model",
                    revision="main",
                    commit_message=f"Set main to {group} seed 01",
                )
        if not UPLOAD_ALL_SEEDS: break