'''
Run Source Pre-Training and Target Fine-Tuning for Cross Prompt Encoder (XPE)

Usage:
    --
    # Zero-Shot XLT
    python -m nlpka.models.scripts.peft.xpe.run --config xlmr/finetune/peft/sib200_hybrid.xpe --supervision_regime=0 sib200_enarzho # 1-4
    python -m nlpka.models.scripts.peft.xpe.run --config xlmr/finetune/peft/sib200_hybrid.xpe --supervision_regime=0 sib200_joshi5 # 1-4
    python -m nlpka.models.scripts.peft.xpe.run --config xlmr/finetune/peft/sib200_hybrid.xpe --supervision_regime=0 sib200_xlmr_seen # 1-4

    # Full-Shot XLT
    python -m nlpka.models.scripts.peft.xpe.run --config xlmr/finetune/peft/sib200_hybrid.xpe --supervision_regime=1 sib200_joshi5_divers_24 # 1-4
    --
    squeue -u bmikaberidze --long
    scancel
'''
import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

import os
import wandb
import pandas as pd
from nlpka.models.model import MODEL
from nlpka.models.scripts.run import run
from nlpka.configs.scripts.xpe_utils import get_ds_names, get_config_by_slurm_task, update_config

import nlpka.evaluations.storage as eval_stor
eval_stor_path = common.get_module_location(eval_stor)

SLURM_ARRAY_TASK_ID = 'SLURM_ARRAY_TASK_ID'
xlmr_tokenized_dirs = 'tokenized|FacebookAI|xlm-roberta-large'
model_paths = {
    # (model_uuid4, path) pairs ...
    'd8349d5f-f2dc-445a-94da-c409c7ca9c11': '/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/d8349d5f-f2dc-445a-94da-c409c7ca9c11_FacebookAI|xlm-roberta-large_1982369_12_40_32_text_classification|topic|sib200_hf|source_xlmr_joshi5_LFID|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5_LFID',
}
# 
# Store Model Paths to Environment Variables
# 
def store_model_paths_to_envs():
    for uuid4, path in model_paths.items():
        MODEL.store_path_by_uuid4_in_envs(uuid4, path)
# 
# Save Results
# 
all_res = []
def save_results(results, model, source_model_uuid4, source_subset, ds_name):
    global all_res, run_name, llm, main_ds_name, run_group, s_job_id, s_task_conf_name

    def get_res(results):
        acc = [
            v for k, v in results.metrics.items() 
            if 'accuracy' in k
        ][0]
        return acc

    # === Header and DataFrame ===
    header = [
        "job_id", "task_conf_name", "trainable_param_size", "model_uuid4", 
        "source_model_uuid4", "source_subset", "ds_name", "z_result", "f_result"
    ]
    res = (
        f'{s_job_id}_', s_task_conf_name, model.trainable_param_size, model.uuid4, 
        source_model_uuid4, 
        source_subset if source_subset else 'full', # source_subset
        ds_name,
        get_res(results.full_shot) if results.full_shot else None, # z_result
        get_res(results.zero_shot) if results.zero_shot else None # f_result
    )
    all_res.append(res)
    df = pd.DataFrame(all_res, columns=header)
    common.p('\n', res)

    # === Set Path and Save Raw CSV ===
    base_path = f'{eval_stor_path}/{llm}/{main_ds_name}/{run_group}/{run_name}'
    os.makedirs(base_path, exist_ok=True)
    raw_path = f'{base_path}/raw.csv'
    df.to_csv(raw_path, index=False)
    common.p(f'Raw results saved to {raw_path}')

    return all_res
# 
# Run Source Pre-Training
# 
def run_source_training(config, source_subsets):
    global source_model_uuid4
    task_id = config.task.id
    for source_subset in source_subsets:
        trained_on_source = False
        for _ in range(cycles):
            if not trained_on_source or cycle_source_training:
                trained_on_source = True
                config = update_config(
                    config, source_ds_name, task_id, suffix_dirs = suffix_dirs, 
                    subset_dirs = source_subset, source_model_uuid4 = None, source_training = True)
                common.p(f'\nSource PEFT Config: ', config.task.peft)
                exit() if exit_before_run else None
                model, results = run(config)
            source_model_uuid4 = model.uuid4
            save_results(results, model, source_model_uuid4, source_subset, source_ds_name)
            run_target_training(config, target_ds_names, source_model_uuid4, source_subset) if target_training else None
#
# Run Target Fine-Tuning
# 
def run_target_training(config, target_ds_names, source_model_uuid4, source_subset):
    task_id = 0
    for ds_name in target_ds_names:
        config = update_config(config, ds_name, task_id, suffix_dirs = suffix_dirs, source_model_uuid4 = source_model_uuid4)
        common.p(f'\nTarget PEFT Config: ', config.task.peft)
        exit() if exit_before_run else None
        model, results = run(config)
        save_results(results, model, source_model_uuid4, source_subset, ds_name)
# 
# Main
# 
if __name__ == '__main__':

    # Parse Config Name and Load Configuration
    import argparse
    ap = argparse.ArgumentParser()
    # Explicitly parse as int so CLI input "0"/"1" matches numeric choices
    ap.add_argument('--supervision_regime', type=int, choices=[0, 1], default=0, help='Supervision regime: Zero Shot = 0 or Full = 1')
    ap.add_argument('main_ds_name', nargs='?', help = 'Main DS Name')
    ap.add_argument('s_task_id', nargs='?', default=1, type=int, help = 'Custom SLURM_ARRAY_TASK_ID')
    args, config_name = common.parse_script_args(ap)
    supervision_regime = args.supervision_regime
    main_ds_name = args.main_ds_name
    llm = config_name.split('/')[0]
    suffix_dirs = None
    # suffix_dirs = xlmr_tokenized_dirs
    
    # # # # # # # # # #
    # Run Configuration
    # 
    run_group = 'hybrid_6' # hybrid_6 
    exit_before_run = False
    # 
    # SBATCH
    if SLURM_ARRAY_TASK_ID in os.environ:
        test_run = False
        source_training = True
        target_training = True
        exit_before_run = False
    # 
    # INTERACTIVE
    else:
        test_run = True
        source_training = True
        target_training = True
        run_group = f'test_{run_group}'
        os.environ[SLURM_ARRAY_TASK_ID] = f'{args.s_task_id}'
    # 
    cycles = 10 if not test_run else 1
    cycle_source_training = True
    # source_subsets = ['0.01'] if test_run else [ None ]
    source_subsets = [ None ]
    custom_target_ds_names = None if not test_run else ['ace_Latn', 'ckb_Arab']
    config, s_task_conf_name, s_job_id, source_model_uuid4 = get_config_by_slurm_task(config_name, main_ds_name, supervision_regime)
    source_ds_name, source_ds_names, target_ds_names = get_ds_names(main_ds_name, llm)
    run_name = f'{common.get_time_id()}_{s_task_conf_name}'
    if test_run and custom_target_ds_names: target_ds_names = custom_target_ds_names

    print(f'\nCONFIG: {config_name}')
    print(f'SLURM: {s_job_id}')
    print(f'RUN: {main_ds_name}/{run_group}/{run_name}')
    print(f'\nsource_training: {source_training}')
    print(f'source_ds_name: {source_ds_name}')
    print(f'source_ds_names len: {len(source_ds_names)}')
    print(f'source_subsets: {source_subsets}')
    print(f'\ntarget_training: {target_training}')
    print(f'target_ds_names len: {len(target_ds_names)}')
    print(f'source_model_uuid4: {source_model_uuid4}')

    # Run Source Pre-Training
    run_source_training(config, source_subsets) if source_training else None

    # Run Target Fine-Tuning
    if target_training and not source_training:
        store_model_paths_to_envs()
        run_target_training(config, target_ds_names, source_model_uuid4)

    # Finish WandB Run If Active
    wandb.finish()
