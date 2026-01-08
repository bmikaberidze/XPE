'''
Run Source Pre-Training and Target Fine-Tuning for Cross Prompt Encoder (XPE)

Usage:
    --
    runtime/clusters/pegasus/shell/run.sh  "python -m nlpka.models.scripts.peft.xpe.run_lid --config xlmr/finetune/peft/lid200 lid200" # 1,2
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
from nlpka.configs.scripts.xpe_utils import get_config_by_slurm_task, update_config, trained_model_paths, grouped_trained_models
import nlpka.evaluations.storage as eval_stor
eval_stor_path = common.get_module_location(eval_stor)

SLURM_ARRAY_TASK_ID = 'SLURM_ARRAY_TASK_ID'

xlmr_suffix_dirs = 'tokenized|FacebookAI|xlm-roberta-large'
aya_suffix_dirs = 'tokenized|CohereForAI|aya-101'
xglm_suffix_dirs = 'tokenized|facebook|xglm-564M'

def gen_run_name(s_task_conf_name):
    run_time_id = common.get_time_id()
    run_name = f'{run_time_id}_{s_task_conf_name}'
    return run_name

def get_res(output):
    acc = [
        v for k, v in output.metrics.items() 
        if 'accuracy' in k
    ][0]
    recall = [
        v for k, v in output.metrics.items() 
        if 'recall' in k
    ][0]
    return acc, recall

def save_results(output, model, source_model_uuid4, subset_dirs, ds_name, dedicated_init = None, **kwargs):
    global all_res, run_name, llm, main_ds_name, run_group, s_job_id, s_task_conf_name, approach
    # Check if each global variable is None or not defined, then set from kwargs
    if 'all_res' not in globals() or all_res is None:
        all_res = kwargs.get('all_res', [])
    if 'run_name' not in globals() or run_name is None:
        run_name = kwargs.get('run_name', '')
    if 'llm' not in globals() or llm is None:
        llm = kwargs.get('llm', '')
    if 'main_ds_name' not in globals() or main_ds_name is None:
        main_ds_name = kwargs.get('main_ds_name', '')
    if 'run_group' not in globals() or run_group is None:
        run_group = kwargs.get('run_group', '')
    if 's_job_id' not in globals() or s_job_id is None:
        s_job_id = kwargs.get('s_job_id', 0)
    if 's_task_conf_name' not in globals() or s_task_conf_name is None:
        s_task_conf_name = kwargs.get('s_task_conf_name', '')
    if 'approach' not in globals() or approach is None:
        approach = kwargs.get('approach', 'xpe')

    # approach = kwargs.get('approach', 'xpe')
    # s_job_id = kwargs.get('s_job_id', 0)
    # s_task_conf_name = kwargs.get('s_task_conf_name', '')

    subset_dirs = subset_dirs if subset_dirs else 'full'

    f_acc, f_recall = get_res(output.full_shot) if output.full_shot else None
    # z_res = get_res(output.zero_shot) if output.zero_shot else None
    res = (
        approach, f'{s_job_id}_', s_task_conf_name, model.trainable_param_size, model.uuid4, 
        source_model_uuid4, subset_dirs, ds_name, dedicated_init, 
        # z_res,
        f_acc,
        *f_recall
    )
    common.p('\n', res)
    all_res.append(res)

    # === Paths ===
    base_path = f'{eval_stor_path}/{llm}/{main_ds_name}/{run_group}/{run_name}'
    os.makedirs(base_path, exist_ok=True)
    raw_path = f'{base_path}/raw.csv'
    # grouped_path = f'{base_path}/grouped.csv'
    # pivoted_path = f'{base_path}/pivoted.csv'

    labels = model._dataset.train.features['labels']

    # === Header and DataFrame ===
    header = [
        "approach", "job_id", "task_conf_name", "trainable_param_size", 
        "model_uuid4", "source_model_uuid4", 
        "subset_dirs", "ds_name", 
        "dedicated_init",
        # "z_result", 
        "f_acc",
        *[labels.int2str(i) for i in range(len(f_recall))]
    ]

    print('>>>>>>>>>>>>>>>>>>>>>>', len(header), len(res), len(f_recall))
    df = pd.DataFrame(all_res, columns=header)

    # === Raw CSV ===
    df.to_csv(raw_path, index=False)
    common.p(f'Raw results saved to {raw_path}')
    
    return all_res

# 
# Run Source Pre-Training
# 
def run_training(config, subset_dirs_list):
    task_id = -1
    num_tasks = 1
    ds_name = None
    for subset_dirs in subset_dirs_list:
        for source_model_uuid4 in grouped_trained_models[s_task_conf_name]:
            for _ in range(cycles):
                config = update_config(
                    config, ds_name, task_id, num_tasks, task_keys = None,
                    suffix_dirs = suffix_dirs, subset_dirs = subset_dirs, source_model_uuid4 = source_model_uuid4, source_training = False)

                common.p(f'\nSource Peft Config: ', config.task.peft); exit() if exit_after_peft_conf_print else None

                model, output = run(config)
                save_results(output, model, source_model_uuid4, subset_dirs, ds_name = None, dedicated_init = 'NONE')

if __name__ == '__main__':

    # Parse Config Name and Load Configuration
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('main_ds_name', nargs='?', help = 'Main DS Name')
    ap.add_argument('s_task_id', nargs='?', default=1, type=int, help = 'Custom SLURM_ARRAY_TASK_ID')
    args, config_name = common.parse_script_args(ap)
    main_ds_name = args.main_ds_name
    llm = config_name.split('/')[0]
    suffix_dirs = xlmr_suffix_dirs
    
    # # # # # # # # # #
    # Run Configuration
    # 
    run_group = 'lid_4_last' # 'fully_dedicated', 'family_groups'
    # 
    exit_after_peft_conf_print = False
    # 
    # SBATCH
    if SLURM_ARRAY_TASK_ID in os.environ:
        test_run = False
    # 
    # INTERACTIVE
    else:
        test_run = True
        run_group = f'test_{run_group}'
        os.environ[SLURM_ARRAY_TASK_ID] = f'{args.s_task_id}'
    #

    cycles = 5
    subset_dirs_list = [ '0.1' ] if test_run else [None]

    # === RUN VARIABLES ===
    all_res = []
    config, config_name, approach, target_id, s_task_conf_name, s_job_id, source_model_uuid4 = get_config_by_slurm_task(config_name, main_ds_name)
    run_name = gen_run_name(s_task_conf_name)

    print(f'\nconfig_name: {config_name}')
    print(f'source_model_uuid4: {source_model_uuid4}')
    print(f'\nSLURM: {s_job_id}')
    print(f'RUN: {main_ds_name}/{run_group}/{run_name}')

    print('\n')

    for model_uuid4 in trained_model_paths.keys():
        if trained_model_paths[model_uuid4]:
            MODEL.store_path_by_uuid4_in_envs(model_uuid4, trained_model_paths[model_uuid4])

    # Run LID Fine-Tuning
    run_training(config, subset_dirs_list)

    # Finish WandB Run If Active
    wandb.finish()
