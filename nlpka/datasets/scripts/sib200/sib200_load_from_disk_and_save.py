import os
from datasets import load_dataset, DatasetDict

# Path to the dataset folder
dataset_path = "/home/bmikaberidze/group5_nlp/nlpka/datasets/storage/benchmarks/text_classification/topic/sib200/data"
save_path = "/home/bmikaberidze/group5_nlp/nlpka/datasets/storage/benchmarks/sib200_hf"

# Get all subset names (directories)
subsets = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

# Dictionary to store all loaded datasets
all_datasets = {}

# Load each dataset
for subset in subsets:
    print(f"Loading {subset}...")
    try:
        dataset = load_dataset(
            "csv",
            data_files={
                "train": f"{dataset_path}/{subset}/train.tsv",
                "dev": f"{dataset_path}/{subset}/dev.tsv",
                "test": f"{dataset_path}/{subset}/test.tsv",
            },
            delimiter="\t",
        )
        all_datasets[subset] = dataset
        print(f"✅ Loaded {subset}")
    except Exception as e:
        print(f"❌ Failed to load {subset}: {e}")

# Save all datasets together as a single DatasetDict
if all_datasets:
    print("\nSaving all datasets...")
    DatasetDict(all_datasets).save_to_disk(save_path)
    print(f"✅ All datasets saved to {save_path}")
else:
    print("❌ No datasets were loaded, skipping save step.")
