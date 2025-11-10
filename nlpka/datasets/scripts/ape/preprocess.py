"""
This script preprocesses the GLUE, SuperGLUE, and SQuAD datasets for comparison between APE and MPT approaches.

The dataset is preprocessed by the following steps:
    1. Tokenize
    2. Mix source datasets
    3. Subset datasets

Usage:
    -
    runtime/clusters/pegasus/shell/run.sh --no-gpu "python -m nlpka.datasets.scripts.ape.preprocess --config '' sib200_all xglm"
    runtime/clusters/pegasus/shell/run.sh --no-gpu "python -m nlpka.datasets.scripts.ape.preprocess --config '' sib200_enarzho xglm"
    runtime/clusters/pegasus/shell/run.sh --no-gpu "python -m nlpka.datasets.scripts.ape.preprocess --config '' sib200_joshi5 xglm"
    -
    runtime/clusters/pegasus/shell/run.sh --no-gpu "python -m nlpka.datasets.scripts.ape.preprocess --config '' sib200_all xlmr"
    runtime/clusters/pegasus/shell/run.sh --no-gpu "python -m nlpka.datasets.scripts.ape.preprocess --config '' sib200_enarzho xlmr"
    runtime/clusters/pegasus/shell/run.sh --no-gpu "python -m nlpka.datasets.scripts.ape.preprocess --config '' sib200_joshi5 xlmr"
    runtime/clusters/pegasus/shell/run.sh --no-gpu "python -m nlpka.datasets.scripts.ape.preprocess --config '' sib200_xlmr_seen xlmr"

"""
import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

from nlpka.tools.enums import ConfigTypeSE
from nlpka.configs.config import CONFIG
from nlpka.datasets.dataset import DATASET
from nlpka.tokenizers.tokenizer import TOKENIZER
from nlpka.tools.enums import SaveDatasetAsSE
from nlpka.configs.scripts.ape_utils import get_ds_names, update_config, multi_sa_ds_name_groups, xnli_ds_name_groups, get_sib200_LID_LFID

ds_name = None
target_id = 0
source_ds_name, source_ds_names, target_ds_names = None, None, None

llm = None
suffix_dirs = None
config_prefix = None

conf_tokenize = 'tokenize'
conf_mix = 'mix_source_datasets'
conf_subset = 'subset'

def run(config_name, ds_name, task_id, suffix_dirs = None, hf_datasets = None, preproc_subset_ratio = None):
    config = CONFIG.load(f'{config_prefix}/{config_name}', ConfigTypeSE.DATASET)
    tokenizer = TOKENIZER.load(config)
    config = update_config(config, ds_name, task_id, suffix_dirs = suffix_dirs, preproc_subset_ratio = preproc_subset_ratio)
    dataset = DATASET(config, tokenizer, hf_datasets)
    return dataset

def preprocess(config_name):
    global source_ds_names, source_ds_name, suffix_dirs, benchmark_name

    suffix_dirs = None if config_name in [ conf_tokenize ] else suffix_dirs    
    
    # Tokenize all datasets
    if config_name == conf_tokenize:
        all_datasets = list(set(source_ds_names + target_ds_names))
        # all_datasets = [ 'eng_Latn' ]
        for ds_name in all_datasets:
            run(config_name, ds_name, task_id = None)

    # Mix all source datasets
    elif config_name == conf_mix:
        # load all source dataset
        def mix_source_datasets(source_ds_name, source_ds_names):
            task_id = 0
            source_hf_datasets = []
            for ds_name in source_ds_names:
                if benchmark_name.startswith('sib200'):
                    LID, LFID = get_sib200_LID_LFID(ds_name)
                    task_id = LFID if LFID_as_task_id else LID
                dataset = run(config_name, ds_name, task_id, suffix_dirs)
                source_hf_datasets.append(dataset.hf)
                if not benchmark_name.startswith('sib200'):
                    task_id += 1
            # create and save mixed dataset
            dataset = run(config_name, source_ds_name, task_id = None, suffix_dirs = suffix_dirs, hf_datasets = source_hf_datasets)
            dataset.save(save_as = SaveDatasetAsSE.HUGGINGFACE, dirs = None)
            print(f'mix of {len(source_hf_datasets)} source_hf_datasets')

        if benchmark_name.startswith('sib200'):
            mix_source_datasets(source_ds_name, source_ds_names)
        else:
            target_num = len(multi_sa_ds_name_groups) if benchmark_name.startswith('multi_sa') else \
                         len(xnli_ds_name_groups)
            for target_id in range(target_num):
                source_ds_name, source_ds_names, _ = get_ds_names(benchmark_name, llm, target_id)
                mix_source_datasets(source_ds_name, source_ds_names)

    # Subset mixed source dataset
    elif config_name == conf_subset:
        for preproc_subset_ratio in [ 0.001, 0.01, 0.1, \
                                    #   0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, \
                                    #   0.1, 0.16, 0.2, 0.3, 0.32, 0.4, 0.5, 0.6, 0.64, 0.7, 0.8, 0.9 
                                    ]:
            run(config_name, source_ds_name, task_id = None, suffix_dirs = suffix_dirs, preproc_subset_ratio = preproc_subset_ratio)

if __name__ == '__main__':

    LFID_as_task_id = False

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('benchmark_name', help = 'Benchmark Name')
    ap.add_argument('llm', help = 'LLM Name')
    args, config_name = common.parse_script_args(ap)
    benchmark_name = args.benchmark_name
    llm = args.llm

    source_ds_name, source_ds_names, target_ds_names = get_ds_names(benchmark_name, llm, target_id)

    if LFID_as_task_id:
        source_ds_name = f'{source_ds_name}_LFID'
    
    suffix_dirs = None
    if llm == 'xglm':
        suffix_dirs = 'tokenized|facebook|xglm-564M'
    elif llm == 'aya':
        suffix_dirs = 'tokenized|CohereForAI|aya-101'
    elif llm == 'xlmr':
        suffix_dirs = 'tokenized|FacebookAI|xlm-roberta-large'

    config_prefix = f'ape/sib200/{llm}' if benchmark_name.startswith('sib200') else \
                    f'ape/multi_sa/{llm}' if benchmark_name.startswith('multi_sa') else \
                    f'ape/xnli/{llm}'


    print(f'llm: {llm}')
    print(f'benchmark_name: {benchmark_name}')
    print(f'source_ds_name: {source_ds_name}')
    print(f'source_ds_names: {len(source_ds_names)}')
    print(f'target_ds_names: {len(target_ds_names)}')
    print(f'config_prefix: {config_prefix}')
    print(f'suffix_dirs: {suffix_dirs}')
    # exit()

    # preprocess(conf_tokenize)
    preprocess(conf_mix)
    preprocess(conf_subset)

