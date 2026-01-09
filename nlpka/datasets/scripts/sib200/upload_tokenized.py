"""
Usage:
    python -m nlpka.datasets.scripts.sib200.upload_tokenized

Prereqs (ONE of these):
    huggingface-cli login
    # OR
    export HF_TOKEN=hf_xxx
"""

import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

import os
from datasets import load_from_disk
from huggingface_hub import create_repo, whoami
from nlpka.configs.scripts.xpe_utils import get_ds_names
from datasets import get_dataset_config_names
import nlpka.datasets.storage as ds_stor
ds_stor_path = common.get_module_location(ds_stor)

# -----------------------------
# Config
# -----------------------------
REPO_ID = "mikaberidze/sib200-xlmr-tokenized"
PRIVATE = False

# Read existing configs on the Hub to avoid re-uploading
try:
    existing_configs = set(get_dataset_config_names(REPO_ID))
    print(f"[HUB CONFIGS] {len(existing_configs)} already exist")
except Exception:
    existing_configs = set()
    print("[HUB CONFIGS] none found (fresh repo)")

def path(name: str) -> str:
    return f"{ds_stor_path}/benchmarks/text_classification/topic/sib200_hf/{name}/tokenized|FacebookAI|xlm-roberta-large"


def ensure_auth():
    """Ensure HF auth exists (CLI login or env token)."""
    try:
        user = whoami()
        print(f"Logged in as: {user.get('name') or user['username']}")
        return None
    except Exception:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "No Hugging Face token found.\n"
                "Run `huggingface-cli login` OR set HF_TOKEN env var."
            )
        return token


def main():
    token = ensure_auth()

    # Create repo once
    print(f"Creating or ensuring repo: {REPO_ID} (private={PRIVATE})")
    create_repo(
        REPO_ID,
        repo_type="dataset",
        private=PRIVATE,
        exist_ok=True,
        token=token,
    )

    source_ds_names = ['source_xlmr_enarzho', 'source_xlmr_seen', 'source_xlmr_joshi5']
    _, _, target_ds_names = get_ds_names('sib200_enarzho', 'xlmr')
    ds_names = source_ds_names + target_ds_names
    # ds_names = ['eng_Latn', 'kmb_Latn']
    # print(f"[DS NAMES] {len(ds_names)}")

    for ds_name in ds_names:        
        if ds_name in existing_configs:
            print(f"[SKIP] already on hub: {ds_name}")
            continue
        
        ds_path = path(ds_name)

        if not os.path.exists(ds_path):
            print(f"[SKIP] missing path: {ds_path}")
            continue

        print(f"[LOAD] {ds_name}")
        ds = load_from_disk(ds_path)

        print(f"[PUSH] config={ds_name}")
        ds.push_to_hub(
            REPO_ID,
            config_name=ds_name,
            token=token,
        )

    print("Done.")
    print(f"https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
