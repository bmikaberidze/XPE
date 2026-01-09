"""
Usage:
    python -m nlpka.datasets.scripts.sib200.download_tokenized

"""

import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

import os
from datasets import get_dataset_config_names, load_dataset
import nlpka.datasets.storage as ds_stor
ds_stor_path = common.get_module_location(ds_stor)

# Define the dataset name and the local save path
dataset_name = "mikaberidze/sib200-xlmr-tokenized"
save_path = f"{ds_stor_path}/benchmarks/text_classification/topic/sib200_tokenized_xlmr"
os.makedirs(save_path, exist_ok=True)

# Download the dataset
# 1. Get config names
config_names = get_dataset_config_names(dataset_name)

# 2. Download each config and 3. Save in a separate folder
for config_name in config_names:
    
    config_save_path = os.path.join(save_path, config_name)
    if os.path.exists(config_save_path):
        print(f"Skipping {config_name}; already exists at {config_save_path}")
        continue

    print(f"Downloading config: {config_name}")
    ds = load_dataset(dataset_name, name=config_name)
    os.makedirs(config_save_path, exist_ok=True)
    ds.save_to_disk(config_save_path)
    print(f"Saved {config_name} to {config_save_path}")

print(f"Dataset downloaded and saved to {save_path}")