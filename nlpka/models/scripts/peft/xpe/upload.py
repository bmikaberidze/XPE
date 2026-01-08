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

MODEL_GROUPS = grouped_trained_models
MODEL_PATHS = trained_model_paths
REPOS = {
    "XPE_SEEN": "mikaberidze/xlmr-large-sib200-softprompt-xpe-seen",
    "SPT_SEEN": "mikaberidze/xlmr-large-sib200-softprompt-spt-seen",
    "XPE_J5":   "mikaberidze/xlmr-large-sib200-softprompt-xpe-joshi5",
    "SPT_J5":   "mikaberidze/xlmr-large-sib200-softprompt-spt-joshi5",
}

# === CREATE REPOS IF NEEDED ===
for repo in REPOS.values():
    try:
        api.create_repo(repo_id=repo, repo_type="model", private=False)
        print(f"Created repo: {repo}")
    except Exception:
        print(f"Repo already exists: {repo}")

# === UPLOAD ===
for group, uuids in MODEL_GROUPS.items():
    repo_id = REPOS[group]

    for i, uuid in enumerate(uuids, start=1):
        seed_branch = f"seed-{i:02d}"
        local_path = MODEL_PATHS[uuid]

        print(f"Uploading {uuid} → {repo_id}@{seed_branch}")

        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(local_path, tmp, dirs_exist_ok=True)

            upload_folder(
                folder_path=tmp,
                repo_id=repo_id,
                repo_type="model",
                revision=seed_branch,
                commit_message=f"Add {group} seed {i:02d}",
            )
