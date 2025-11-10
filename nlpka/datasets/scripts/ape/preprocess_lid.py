"""
The dataset is preprocessed by the following steps:
    1. Tokenize
    2. Subset datasets

Usage:
    -
    runtime/clusters/pegasus/shell/run.sh --no-gpu "python -m nlpka.datasets.scripts.ape.preprocess_lid --config '' xlmr"
"""

import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

from nlpka.tools.enums import ConfigTypeSE
from nlpka.configs.config import CONFIG
from nlpka.datasets.dataset import DATASET
from nlpka.tokenizers.tokenizer import TOKENIZER
from nlpka.tools.enums import SaveDatasetAsSE
from nlpka.configs.scripts.xpe_utils import update_config

llm = None
config_prefix = None

conf_tokenize = 'tokenize'
conf_subset = 'subset'

def preprocess(config_name):
    
    # Tokenize all datasets
    if config_name == conf_tokenize:
        run(config_name, ds_name = None, task_id = None)

    # Subset mixed source dataset
    elif config_name == conf_subset:
        suffix_dirs = 'tokenized|FacebookAI|xlm-roberta-large'    
        for preproc_subset_ratio in [ 0.001, 0.01, 0.1, \
                                    #   0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, \
                                    #   0.1, 0.16, 0.2, 0.3, 0.32, 0.4, 0.5, 0.6, 0.64, 0.7, 0.8, 0.9 
                                    ]:
            run(config_name, ds_name = None, task_id = None, suffix_dirs = suffix_dirs, preproc_subset_ratio = preproc_subset_ratio)

def run(config_name, ds_name, task_id, suffix_dirs = None, hf_datasets = None, preproc_subset_ratio = None):
    config = CONFIG.load(f'{config_prefix}/{config_name}', ConfigTypeSE.DATASET)
    tokenizer = TOKENIZER.load(config)
    config = update_config(config, ds_name, task_id, suffix_dirs = suffix_dirs, preproc_subset_ratio = preproc_subset_ratio)
    dataset = DATASET(config, tokenizer, hf_datasets)
    return dataset

if __name__ == '__main__':

    # from datasets import load_dataset
    # data = load_dataset("mikaberidze/lid200")  # unified dataset (no per-language configs)
    # example = data["train"][0]
    # print(example)
    # exit()

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('llm', help = 'LLM Name')
    args, config_name = common.parse_script_args(ap)
    llm = args.llm
    
    config_prefix = f'ape/lid200/{llm}'

    print(f'llm: {llm}')
    print(f'config_prefix: {config_prefix}')
    # exit()

    preprocess(conf_tokenize)
    preprocess(conf_subset)

