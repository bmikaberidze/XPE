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
from nlpka.configs.scripts.xpe_utils import get_config_by_slurm_task, update_config
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


paths_by_id = {
    "SPT_SEEN": [
        # "70c56fb4-f6e5-40b5-bfbe-5fdd0587cc7d",
        # "932b3f65-0ca4-4e07-9abe-a0e2fbfbaa6f",
        # "78520010-9059-4daa-8b6a-86de2fffd6c6",
        # "ad872348-a9fb-46ac-a59d-722fb6f0262b",
        # "5bc5659f-c51c-4b67-9c6c-78416fcb8d4f",
        # "4de99566-48da-4f22-8273-619f6d601361",
        # "7bed6746-baae-4716-a14b-2073fd9a13b5",
        # "d5352222-c549-4861-b4e9-353fc59eed2f",
        # "8f405d0e-d752-4008-b710-e90bb1248ee2",
        "b078674c-086e-4326-8354-139638092aaa"
    ],
    "XPE_SEEN": [
        # "10e025f6-1da6-44a4-9f96-b8893e65c2c5",
        # "6ff52a90-aaa0-4d1a-b0b2-926bd57cd5c5",
        # "aef2b130-debc-499e-aec3-f14d6c5fbb1a",
        # "e8a21d23-789e-4cbf-82e4-27072ded5f3e",
        # "40d575f5-05d8-45a0-b599-36a593e37959",
        # "e10d1637-0452-4e4b-a339-37e5d6134fec",
        # "a6096917-2d98-47ed-92e3-ba00ef530f48",
        # "03a9d451-3312-4748-810d-eedc8a4009be",
        # "ad56d282-e924-4026-b8fb-691721266098",
        "ea1f72c6-0721-487c-bfa6-114a6cb4d6ce"
    ],
    "SPT_J5": [
        # "c417a084-008a-4cbc-86b8-d703a2aad405",
        # "aab0f095-9be9-493e-8cb4-d2e0f671f7fd",
        # "29405be4-0b55-4982-8acc-759bed8adbb9",
        # "bf5be0bb-ed5e-4d13-a095-2b3a1dbd5562",
        # "e3ae555d-7516-4ec7-92ee-65e7c8a8da61",
        # "270a6eda-5c6f-4dd2-8d65-8050c07e6e92",
        # "cffc931d-59e1-434b-86c4-29a0b45b58f1",
        # "cb9acdd3-fbe1-4ca8-aeec-c5c1d7dab2e0",
        # "dc1c7cfa-ef0a-4ad1-88f4-d7f8df06bdbf",
        "a749d626-a7f5-40dd-a4e1-678bf7868c67"
    ],
    "XPE_J5": [
        # "ed723b64-47e9-4ad7-ab61-f75c3f2cfd2b",
        # "a3b67bca-6382-45dd-a991-2467c27483b1",
        # "4a72123f-17c4-4565-a9ff-a76ada3eb6a1",
        # "922c5a8f-f0b4-40c6-abab-a60f785151be",
        # "932bb7f9-ee32-421c-90c0-fad65ce26038",
        # "387097bd-6c91-42d1-8565-72a03cbd0703",
        # "61a28706-ef12-46ec-b0b8-431199c272b9",
        # "3253362d-a2b8-42a1-b1cb-ef28dd199324",
        # "6ff741ed-ad58-4ac9-9bb6-03b0e46a9535",
        "7b0367b9-888f-4f17-96cd-463010d1e36a"
    ]
}

# 
# Run Source Pre-Training
# 
def run_training(config, subset_dirs_list):
    task_id = -1
    num_tasks = 1
    ds_name = None
    for subset_dirs in subset_dirs_list:
        for source_model_uuid4 in paths_by_id[s_task_conf_name]:
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

    model_paths = {
        # Source models for LID
        # "spt_seen": {
        "70c56fb4-f6e5-40b5-bfbe-5fdd0587cc7d": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/70c56fb4-f6e5-40b5-bfbe-5fdd0587cc7d_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "932b3f65-0ca4-4e07-9abe-a0e2fbfbaa6f": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/932b3f65-0ca4-4e07-9abe-a0e2fbfbaa6f_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "78520010-9059-4daa-8b6a-86de2fffd6c6": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/78520010-9059-4daa-8b6a-86de2fffd6c6_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "ad872348-a9fb-46ac-a59d-722fb6f0262b": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/ad872348-a9fb-46ac-a59d-722fb6f0262b_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "5bc5659f-c51c-4b67-9c6c-78416fcb8d4f": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/5bc5659f-c51c-4b67-9c6c-78416fcb8d4f_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "4de99566-48da-4f22-8273-619f6d601361": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/4de99566-48da-4f22-8273-619f6d601361_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "7bed6746-baae-4716-a14b-2073fd9a13b5": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/7bed6746-baae-4716-a14b-2073fd9a13b5_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "d5352222-c549-4861-b4e9-353fc59eed2f": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/d5352222-c549-4861-b4e9-353fc59eed2f_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "8f405d0e-d752-4008-b710-e90bb1248ee2": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/8f405d0e-d752-4008-b710-e90bb1248ee2_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "b078674c-086e-4326-8354-139638092aaa": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/b078674c-086e-4326-8354-139638092aaa_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        # "xpe_seen": {
        "10e025f6-1da6-44a4-9f96-b8893e65c2c5": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/10e025f6-1da6-44a4-9f96-b8893e65c2c5_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "6ff52a90-aaa0-4d1a-b0b2-926bd57cd5c5": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/6ff52a90-aaa0-4d1a-b0b2-926bd57cd5c5_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "aef2b130-debc-499e-aec3-f14d6c5fbb1a": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/aef2b130-debc-499e-aec3-f14d6c5fbb1a_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "e8a21d23-789e-4cbf-82e4-27072ded5f3e": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/e8a21d23-789e-4cbf-82e4-27072ded5f3e_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "40d575f5-05d8-45a0-b599-36a593e37959": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/40d575f5-05d8-45a0-b599-36a593e37959_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "e10d1637-0452-4e4b-a339-37e5d6134fec": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/e10d1637-0452-4e4b-a339-37e5d6134fec_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "a6096917-2d98-47ed-92e3-ba00ef530f48": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/a6096917-2d98-47ed-92e3-ba00ef530f48_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "03a9d451-3312-4748-810d-eedc8a4009be": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/03a9d451-3312-4748-810d-eedc8a4009be_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "ad56d282-e924-4026-b8fb-691721266098": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/ad56d282-e924-4026-b8fb-691721266098_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        "ea1f72c6-0721-487c-bfa6-114a6cb4d6ce": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/ea1f72c6-0721-487c-bfa6-114a6cb4d6ce_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
        # "spt_joshi5": {
        "c417a084-008a-4cbc-86b8-d703a2aad405": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/c417a084-008a-4cbc-86b8-d703a2aad405_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "aab0f095-9be9-493e-8cb4-d2e0f671f7fd": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/aab0f095-9be9-493e-8cb4-d2e0f671f7fd_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "29405be4-0b55-4982-8acc-759bed8adbb9": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/29405be4-0b55-4982-8acc-759bed8adbb9_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "bf5be0bb-ed5e-4d13-a095-2b3a1dbd5562": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/bf5be0bb-ed5e-4d13-a095-2b3a1dbd5562_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "e3ae555d-7516-4ec7-92ee-65e7c8a8da61": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/e3ae555d-7516-4ec7-92ee-65e7c8a8da61_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "270a6eda-5c6f-4dd2-8d65-8050c07e6e92": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/270a6eda-5c6f-4dd2-8d65-8050c07e6e92_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "cffc931d-59e1-434b-86c4-29a0b45b58f1": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/cffc931d-59e1-434b-86c4-29a0b45b58f1_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "cb9acdd3-fbe1-4ca8-aeec-c5c1d7dab2e0": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/cb9acdd3-fbe1-4ca8-aeec-c5c1d7dab2e0_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "dc1c7cfa-ef0a-4ad1-88f4-d7f8df06bdbf": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/dc1c7cfa-ef0a-4ad1-88f4-d7f8df06bdbf_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "a749d626-a7f5-40dd-a4e1-678bf7868c67": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/a749d626-a7f5-40dd-a4e1-678bf7868c67_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        # "xpe_joshi5": {
        "ed723b64-47e9-4ad7-ab61-f75c3f2cfd2b": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/ed723b64-47e9-4ad7-ab61-f75c3f2cfd2b_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "a3b67bca-6382-45dd-a991-2467c27483b1": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/a3b67bca-6382-45dd-a991-2467c27483b1_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "4a72123f-17c4-4565-a9ff-a76ada3eb6a1": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/4a72123f-17c4-4565-a9ff-a76ada3eb6a1_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "922c5a8f-f0b4-40c6-abab-a60f785151be": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/922c5a8f-f0b4-40c6-abab-a60f785151be_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "932bb7f9-ee32-421c-90c0-fad65ce26038": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/932bb7f9-ee32-421c-90c0-fad65ce26038_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "387097bd-6c91-42d1-8565-72a03cbd0703": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/387097bd-6c91-42d1-8565-72a03cbd0703_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "61a28706-ef12-46ec-b0b8-431199c272b9": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/61a28706-ef12-46ec-b0b8-431199c272b9_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "3253362d-a2b8-42a1-b1cb-ef28dd199324": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/3253362d-a2b8-42a1-b1cb-ef28dd199324_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "6ff741ed-ad58-4ac9-9bb6-03b0e46a9535": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/6ff741ed-ad58-4ac9-9bb6-03b0e46a9535_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
        "7b0367b9-888f-4f17-96cd-463010d1e36a": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/7b0367b9-888f-4f17-96cd-463010d1e36a_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    }
    for model_uuid4 in model_paths.keys():
        if model_paths[model_uuid4]:
            MODEL.store_path_by_uuid4_in_envs(model_uuid4, model_paths[model_uuid4])

    # Run LID Fine-Tuning
    run_training(config, subset_dirs_list)

    # Finish WandB Run If Active
    wandb.finish()
