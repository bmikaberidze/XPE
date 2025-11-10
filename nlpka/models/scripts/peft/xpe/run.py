'''
Run Source Pre-Training and Target Fine-Tuning for Cross Prompt Encoder (XPE)

Usage:
    --
    # Zero-Shot XLT
    runtime/clusters/pegasus/shell/run.sh  "python -m nlpka.models.scripts.peft.xpe.run --config xlmr/finetune/peft/sib200_hybrid sib200_enarzho" # 1-4
    runtime/clusters/pegasus/shell/run.sh  "python -m nlpka.models.scripts.peft.xpe.run --config xlmr/finetune/peft/sib200_hybrid sib200_xlmr_seen" # 1-4
    runtime/clusters/pegasus/shell/run.sh  "python -m nlpka.models.scripts.peft.xpe.run --config xlmr/finetune/peft/sib200_hybrid sib200_joshi5" # 1-4

    # Full-Shot XLT
    runtime/clusters/pegasus/shell/run.sh  "python -m nlpka.models.scripts.peft.xpe.run --config xlmr/finetune/peft/sib200_hybrid sib200_joshi5_ablation" # 11-14
    runtime/clusters/pegasus/shell/run.sh  "python -m nlpka.models.scripts.peft.xpe.run --config xlmr/finetune/peft/sib200_hybrid sib200_joshi5_divers_ablation" # 11-14
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
from nlpka.configs.scripts.xpe_utils import get_ds_names, get_config_by_slurm_task, update_config, get_sib200_LID_LFID
from nlpka.models.cross_prompt_encoder import CrossPromptEncoderEmbeddingType as XPE_ET
from nlpka.models.cross_prompt_encoder import CrossPromptEncoderEmbeddDedicInit as XPE_DI
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
    return acc*100

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

    f_res = get_res(output.full_shot) if output.full_shot else None
    z_res = get_res(output.zero_shot) if output.zero_shot else None
    res = (
        approach, f'{s_job_id}_', s_task_conf_name, model.trainable_param_size, model.uuid4, 
        source_model_uuid4, subset_dirs, ds_name, dedicated_init, z_res, f_res
    )
    common.p('\n', res)
    all_res.append(res)

    # === Paths ===
    base_path = f'{eval_stor_path}/{llm}/{main_ds_name}/{run_group}/{run_name}'
    os.makedirs(base_path, exist_ok=True)
    raw_path = f'{base_path}/raw.csv'
    grouped_path = f'{base_path}/grouped.csv'
    pivoted_path = f'{base_path}/pivoted.csv'

    # === Header and DataFrame ===
    header = [
        "approach", "job_id", "task_conf_name", "trainable_param_size", 
        "model_uuid4", "source_model_uuid4", 
        "subset_dirs", "ds_name", 
        "dedicated_init",
        "z_result", "f_result"
    ]
    df = pd.DataFrame(all_res, columns=header)

    # === Raw CSV ===
    df.to_csv(raw_path, index=False)
    common.p(f'Raw results saved to {raw_path}')

    # === GROUPED ===
    grouped_df = df.drop(columns=["model_uuid4", "job_id", "source_model_uuid4"]).groupby(
        ["approach", "task_conf_name", "subset_dirs", "trainable_param_size", "ds_name", "dedicated_init"], as_index=False
    ).agg(
        z_result_mean=("z_result", "mean"),
        z_result_std=("z_result", "std"),
        z_result_count=("z_result", "count"),
        f_result_mean=("f_result", "mean"),
        f_result_std=("f_result", "std"),
        f_result_count=("f_result", "count"),
    )
    for col in grouped_df.columns:
        if col.endswith("_mean"):
            grouped_df[col] = grouped_df[col]
        elif col.endswith("_std"):
            grouped_df[col] = grouped_df[col]
    grouped_df.to_csv(grouped_path, index=False)
    common.p(f'Grouped results saved to {grouped_path}')

    # === PIVOTED LONG FORMAT (rows for f_result and z_result separately) ===
    # First: melt the grouped_df into long format
    long_df = grouped_df.melt(
        id_vars=["approach", "task_conf_name", "subset_dirs", "trainable_param_size", "ds_name", "dedicated_init"],
        value_vars=["z_result_mean", "f_result_mean"],
        var_name="shot_type",
        value_name="score"
    )

    # Normalize shot_type values
    long_df["shot_type"] = long_df["shot_type"].map({
        "f_result_mean": "full_shot",
        "z_result_mean": "zero_shot"
    })

    # Pivot to get ds_name as columns and one row per shot_type
    pivoted_df = long_df.pivot_table(
        index=["approach", "task_conf_name", "subset_dirs", "trainable_param_size", "shot_type", "dedicated_init"],
        columns="ds_name",
        values="score",
        aggfunc="first"
    ).reset_index()

    # Save pivoted
    pivoted_df.to_csv(pivoted_path, index=False)
    common.p(f'Pivoted results saved to {pivoted_path}')

    # Pretty print
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print("\n=== Pivoted Results Table (Rows per Shot) ===")
        print(pivoted_df.fillna("").to_string(index=False))
    
    return all_res
# 
# Run Source Pre-Training
# 
def run_source_training(config, subset_dirs_list):
    global source_model_uuid4
    num_tasks = len(source_keys) if source_keys else len(source_ds_names)
    task_id = None if num_tasks > 1 else source_keys[0] if source_keys else config.task.id
    # print(f'\nnum_tasks: {num_tasks}')
    # print(f'source_keys: {source_keys}')
    # print(f'source_ds_names: {source_ds_names}')
    # print(f'task_id: {task_id}')
    # exit()
    trained_on_source = False
    for subset_dirs in subset_dirs_list:
        for _ in range(cycles):
            if not trained_on_source or cycle_source_training:
                trained_on_source = True
                config = update_config(
                    config, source_ds_name, task_id, num_tasks = num_tasks, task_keys = source_keys,
                    suffix_dirs = suffix_dirs, subset_dirs = subset_dirs, source_model_uuid4 = None, source_training = True)

                common.p(f'\nSource Peft Config: ', config.task.peft); exit() if exit_after_peft_conf_print else None

                model, output = run(config)
            source_model_uuid4 = model.uuid4
            save_results(output, model, source_model_uuid4, subset_dirs, source_ds_name, dedicated_init = 'NONE')
            run_target_training(config, target_ds_names, source_model_uuid4, subset_dirs) if target_training else None
#
# Run Target Fine-Tuning
# 
def run_target_training(config, target_ds_names, source_model_uuid4, subset_dirs = None):
    task_id = 0
    num_tasks = 1
    def run_target_training_by_dedicated_init(dedicated_init = 'NONE'):
        global config
        for ds_name in target_ds_names:
            fallback = False
            dedicated_init_key = None
            if dedicated_init == XPE_DI.SPECIFIC:
                if main_ds_name.startswith('sib200'):
                    key = get_sib200_LID_LFID(ds_name)[LFID_as_task_id]
                    dedicated_init_key = -1
                    # if key in source_keys:
                    #     dedicated_init_key = key
                    # else:
                    #     fallback = True
                
                common.p(f'\n[yellow]================================================================================[/yellow]')
                print(f'Target_ds_name: {ds_name}')
                print(f'target_key: {key}')
                print(f'source_keys: {source_keys}')
                print(f'dedicated_init: {dedicated_init}')
                print(f'dedicated_init_key: {dedicated_init_key}')
                common.p(f'[yellow]================================================================================[/yellow]')
            
            final_dedicated_init = dedicated_init if not fallback else XPE_DI.AVERAGE
            fallback = False
            
            config = update_config(
                config, ds_name, task_id, num_tasks = num_tasks, task_keys = None,
                suffix_dirs = suffix_dirs, source_model_uuid4 = source_model_uuid4,
                dedicated_init = final_dedicated_init, dedicated_init_key = dedicated_init_key)
            
            # if test_run:
            #     config.task.peft.encoder_embedding_dedicated_init = 'AVERAGE'

            common.p(f'\nTarget Peft Config: ', config.task.peft); exit() if exit_after_peft_conf_print else None
            # exit()
            
            model, output = run(config)
            save_results(output, model, source_model_uuid4, subset_dirs, ds_name, dedicated_init)
    
    if config.task.peft.encoder_embedding_type == XPE_ET.FULLY_SHARED:
        run_target_training_by_dedicated_init()
    else:
        for dedicated_init in [
            # XPE_DI.ZEROS, 
            # XPE_DI.RANDOM, 
            # XPE_DI.AVERAGE, 
            XPE_DI.SPECIFIC,
        ]:
            run_target_training_by_dedicated_init(dedicated_init)

if __name__ == '__main__':

    # Parse Config Name and Load Configuration
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('main_ds_name', nargs='?', help = 'Main DS Name')
    ap.add_argument('s_task_id', nargs='?', default=1, type=int, help = 'Custom SLURM_ARRAY_TASK_ID')
    args, config_name = common.parse_script_args(ap)
    main_ds_name = args.main_ds_name
    llm = config_name.split('/')[0]
    if llm == 'xlmr':
        suffix_dirs = xlmr_suffix_dirs
    elif llm == 'xglm':
        suffix_dirs = xglm_suffix_dirs
    elif llm == 'aya':
        suffix_dirs = aya_suffix_dirs
    else:
        raise ValueError(f'Invalid LLM: {llm}')
    
    # # # # # # # # # #
    # Run Configuration
    # 
    run_group = 'hybrid_6' # 'fully_dedicated', 'family_groups'
    # 
    LFID_as_task_id = 0
    # 
    unified_target = False
    target_ds_suffix = False # 'ablation'
    custom_target_ds_names = None # ['ace_Latn', 'ckb_Arab', 'tat_Cyrl', 'aeb_Arab', 'arb_Latn']
    # 
    exit_after_peft_conf_print = True
    # 
    # SBATCH
    if SLURM_ARRAY_TASK_ID in os.environ:
        test_run = False
        source_training = False
        target_training = True
    # 
    # INTERACTIVE
    else:
        test_run = True
        source_training = False
        target_training = True
        run_group = f'test_{run_group}'
        os.environ[SLURM_ARRAY_TASK_ID] = f'{args.s_task_id}'
    #

    # === RUN CONFIG ===
    # 
    # LID-200
    if main_ds_name.startswith('lid200'):
        cycles = 10
        cycle_source_training = True
        full_ds = 'only' # prepend | append | only
        custom_subset_dirs_list = None # [ '0.01'] if test_run else None

        # main_ds_name = 'sib200_en_on'
    
        if target_ds_suffix:
            main_ds_name = f'{main_ds_name}_{target_ds_suffix}'

        target_id = None

    # 
    # SIB-200
    elif main_ds_name.startswith('sib200'):
        cycles = 10
        cycle_source_training = True
        full_ds = 'only' # prepend | append | only
        custom_subset_dirs_list = None # [ '0.01'] if test_run else None

        # main_ds_name = 'sib200_en_on'
    
        if target_ds_suffix:
            main_ds_name = f'{main_ds_name}_{target_ds_suffix}'

        target_id = None
        
    # 
    # Multi-SA
    elif main_ds_name.startswith('multi_sa'):
        cycles = 3
        cycle_source_training = False
        full_ds = 'only' # prepend | append | only
        custom_subset_dirs_list = None
        target_id = 0
    # 
    # XNLI
    elif main_ds_name.startswith('xnli'):
        cycles = 3
        cycle_source_training = False
        full_ds = 'only' # prepend | append | only
        custom_subset_dirs_list = None # ['0.01']
        target_id = 0

    subset_dirs_list = [
        # '0.01', 
        '0.02',
        # '0.03', 
        '0.04', 
        # '0.05', 
        # '0.06', 
        # '0.07', 
        '0.08', 
        # '0.09', 
        # '0.1', 
        '0.16',
        # '0.2', 
        # '0.3',
        '0.32',
        # '0.4', 
        # '0.5', 
        # '0.6', 
        '0.64',
        # '0.7', 
        # '0.8', 
        # '0.9', 
    ]
    subset_dirs_list =  custom_subset_dirs_list if custom_subset_dirs_list else \
                        [None] + subset_dirs_list if full_ds == 'prepend' else \
                        subset_dirs_list + [None] if full_ds == 'append' else \
                        [None]

    # === RUN VARIABLES ===
    all_res = []
    config, config_name, approach, target_id, s_task_conf_name, s_job_id, source_model_uuid4 = get_config_by_slurm_task(config_name, main_ds_name)
    source_ds_name, source_ds_names, target_ds_names = get_ds_names(main_ds_name, llm, target_id)
    run_name = gen_run_name(s_task_conf_name)

    if test_run and custom_target_ds_names:
        target_ds_names = custom_target_ds_names

    source_keys = []
    target_keys = []
    if main_ds_name.startswith('sib200') and config.task.peft.encoder_embedding_type in [XPE_ET.PARTIALLY_DEDICATED, XPE_ET.FULLY_DEDICATED]:
        source_keys = list(set([get_sib200_LID_LFID(ds_name)[LFID_as_task_id] for ds_name in source_ds_names]))
        target_keys = list(set([get_sib200_LID_LFID(ds_name)[LFID_as_task_id] for ds_name in target_ds_names]))

        source_ds_name = f'{source_ds_name}_LFID' if LFID_as_task_id else source_ds_name

        # Neutral Dedication is set to 0 if all target_keys are present in source_keys
        neutral_dedication = config.task.peft.encoder_embedding_neutral_dedication
        if neutral_dedication and set(target_keys).issubset(set(source_keys)):
            config.task.peft.encoder_embedding_neutral_dedication = 0
            common.p(f'\n[red]Neutral Dedication is set to 0 because all target_keys present in source_keys[/red]')

    if unified_target: target_ds_names = [ 'source_xlmr_unseen_LFID' ]

    print(f'\nconfig_name: {config_name}')

    print(f'\nsource_training: {source_training}')
    print(f'target_training: {target_training}')
    print(f'source_model_uuid4: {source_model_uuid4}')

    print(f'\nSLURM: {s_job_id}')
    print(f'RUN: {main_ds_name}/{run_group}/{run_name}')

    print(f'\nsource_ds_name: {source_ds_name}')
    print(f'source_ds_names len: {len(source_ds_names)}')
    print(f'source_keys ({len(source_keys)}): {source_keys}')

    print('\n')
    if unified_target: print(f'unified_target_ds_names: {target_ds_names}')
    print(f'target_ds_suffix: {target_ds_suffix}')
    print(f'target_ds_names len: {len(target_ds_names)}')
    print(f'target_keys ({len(target_keys)}): {target_keys}')

    # Run Source Pre-Training
    run_source_training(config, subset_dirs_list) if source_training else None

    # Run Target Fine-Tuning
    if target_training and not source_training:

        model_paths = {
            'd8349d5f-f2dc-445a-94da-c409c7ca9c11': '/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/d8349d5f-f2dc-445a-94da-c409c7ca9c11_FacebookAI|xlm-roberta-large_1982369_12_40_32_text_classification|topic|sib200_hf|source_xlmr_joshi5_LFID|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5_LFID',
            'f09de413-0df6-49a8-be2f-f48c814b3170': '/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/f09de413-0df6-49a8-be2f-f48c814b3170_FacebookAI|xlm-roberta-large_1982371_9_40_32_text_classification|topic|sib200_hf|source_xlmr_joshi5_LFID|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5_LFID',
            'e3ec585b-a2fb-492c-9666-e8b5772209da': '/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/e3ec585b-a2fb-492c-9666-e8b5772209da_FacebookAI|xlm-roberta-large_1982372_10_40_32_text_classification|topic|sib200_hf|source_xlmr_joshi5_LFID|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5_LFID',
            '2c618551-1f7a-4cd0-864a-5b6e24a18eb4': '/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/4852c7ce-ac0d-4402-8355-2be33575b031_FacebookAI|xlm-roberta-large_1984989_3_40_32_text_classification|topic|sib200_hf|source_xlmr_joshi5_LFID|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5_LFID',
            '4852c7ce-ac0d-4402-8355-2be33575b031': '/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/4852c7ce-ac0d-4402-8355-2be33575b031_FacebookAI|xlm-roberta-large_1984989_3_40_32_text_classification|topic|sib200_hf|source_xlmr_joshi5_LFID|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5_LFID',
            '70c7459c-a9f1-41dd-b1b2-a5834dcf6134': '/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/70c7459c-a9f1-41dd-b1b2-a5834dcf6134_FacebookAI|xlm-roberta-large_2014022_54_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5_LFID|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5_LFID',
            '399d74eb-41f3-4fdc-a79c-71d0029f201c': '/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xglm/facebook/xglm-564M/topic/399d74eb-41f3-4fdc-a79c-71d0029f201c_facebook|xglm-564M_2018748_23_10_32_text_classification|topic|sib200_hf|source_xglm_joshi5|tokenized|facebook|xglm-564M_source_xglm_joshi5', 
        }
        for model_uuid4 in model_paths.keys():
            if model_paths[model_uuid4]:
                MODEL.store_path_by_uuid4_in_envs(model_uuid4, model_paths[model_uuid4])

        run_target_training(config, target_ds_names, source_model_uuid4)

    # Finish WandB Run If Active
    wandb.finish()
