import os
import datasets

# List of GLUE datasets
# name = 'glue'
# tasks = [
#     "cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli", "ax"
# ]
# name = 'super_glue'
# tasks = [
#     "axb", "axg", "boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"
# ]

# name = 'squad'
# name = 'rajpurkar/squad_v2'
# tasks = [ None ]

name = 'mrqa'
tasks = [
    "nq", "hp", "sqa", "news" # 
]

# Define the base path
base_path = f"/home/bmikaberidze/group5_nlp/nlpka/datasets/storage/benchmarks/{name}"
os.makedirs(base_path, exist_ok=True)

# Loop through datasets and save them
for i, task in enumerate(tasks, start=1):
    try:
        # Load dataset # Define dataset path
        if task:
            dataset_path = os.path.join(base_path, task)
            dataset = datasets.load_dataset(name, task)
        else:
            dataset_path = base_path
            dataset = datasets.load_dataset(name)

        os.makedirs(dataset_path, exist_ok=True)
        
        # Save in Hugging Face format
        dataset.save_to_disk(dataset_path)
        
        print(f"Saved {task} to {dataset_path}")
    except Exception as e:
        print(f"Error loading {task}: {e}")