'''
python -m nlpka.datasets.scripts.lid200.download_from_hf
'''

from datasets import load_dataset

# Define the dataset name and the local save path
dataset_name = "mikaberidze/lid200"
save_path = "/home/bmikaberidze/group5_nlp/nlpka/datasets/storage/benchmarks/text_classification/lid/lid200_hf"


def download_and_save_dataset(dataset_name, save_path):
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset(dataset_name)
        
        # Save the dataset locally
        dataset.save_to_disk(save_path)
        print(f"✅ Dataset '{dataset_name}' downloaded and saved to {save_path}")
    except Exception as e:
        print(f"❌ Failed to download or save the dataset '{dataset_name}': {e}")

# Execute the function
download_and_save_dataset(dataset_name, save_path)
