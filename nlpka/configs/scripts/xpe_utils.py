import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

import os
import random
import pandas as pd
from peft import PeftType
from types import SimpleNamespace
from nlpka.models.model import MODEL
from nlpka.configs.config import CONFIG
from nlpka.tools.enums import ConfigTypeSE, ModelArchSE
from nlpka.models.cross_prompt_encoder import CrossPromptEncoderReparameterizationType as XPE_RE

import nlpka.datasets.storage as ds_stor
ds_path = common.get_module_location(ds_stor)
def get_sib200_meta():
    meta_path = f'{ds_path}/benchmarks/text_classification/topic/sib200_tokenized_xlmr/meta.csv'
    meta = pd.read_csv(meta_path)
    return meta

sib200_meta = get_sib200_meta()
sib200_ds_names = sib200_meta['code'].tolist()
xlmr_seen_sib200_ds_names = sib200_meta[sib200_meta['xlmr'] == 1]['code'].tolist()
xlmr_unseen_sib200_ds_names = list(set(sib200_ds_names) - set(xlmr_seen_sib200_ds_names))

xlmr_unseen_script_sib200_ds_names = ['nqo_Nkoo', 'sat_Olck', 'taq_Tfng', 'tzm_Tfng', 'bod_Tibt', 'dzo_Tibt']

# arb_Arab, deu_Latn, eng_Latn, fra_Latn, spa_Latn, jpn_Jpan, zho_Hans
joshi5_sib200_ds_names = sib200_meta[sib200_meta['class'] == 5]['code'].tolist()
not_joshi5_sib200_ds_names = list(set(sib200_ds_names) - set(joshi5_sib200_ds_names))

Sib200_family_to_id = {
    'Afro-Asiatic': 0,
    'Atlantic-Congo': 1,
    'Austroasiatic': 2,
    'Austronesian': 3,
    'Aymaran': 4,
    'Basque': 5,
    'Constructed': 6,
    'Dravidian': 7,
    'Indo-European': 8,
    'Japonic': 9,
    'Kartvelian': 10,
    'Koreanic': 11,
    'Mande': 12,
    'Mongolic-Khitan': 13,
    'Nilotic': 14,
    'Quechuan': 15,
    'Sino-Tibetan': 16,
    'Tai-Kadai': 17,
    'Tupian': 18,
    'Turkic': 19,
    'Uralic': 20
}

def get_sib200_LID_LFID(lang_code):
    lang_row = sib200_meta[sib200_meta['code'] == lang_code].iloc[0]
    LFID = Sib200_family_to_id[lang_row['family']]
    LID = int(lang_row.name)
    return LID, LFID

# print(get_sib200_LID_LFID('afr_Latn'))
# exit()

# xlmr_seen_lrl_sib200_ds_names  = sib200_meta[(sib200_meta['class'].isin([0, 1, 2])) & (sib200_meta['xlmr'] == 1)]['code'].tolist()
# xlmr_unseen_lrl_sib200_ds_names = sib200_meta[(sib200_meta['class'].isin([0, 1, 2])) & (sib200_meta['xlmr'] == 0)]['code'].tolist()
# random_seen = random.sample(xlmr_seen_lrl_sib200_ds_names, min(25, len(xlmr_seen_lrl_sib200_ds_names)))
# random_unseen = random.sample(xlmr_unseen_lrl_sib200_ds_names, min(25, len(xlmr_unseen_lrl_sib200_ds_names)))
# print("Random 25 from seen:", random_seen)
# print("Random 25 from unseen:", random_unseen)
xlmr_25_seen_lrl_sib200_ds_names = ['asm_Beng', 'swh_Latn', 'plt_Latn', 'ory_Orya', 'guj_Gujr', 'khm_Khmr', 'zho_Hant', 'uig_Arab', 'nob_Latn', 'xho_Latn', 'sun_Latn', 'jav_Latn', 'mal_Mlym', 'pan_Guru', 'amh_Ethi', 'mar_Deva', 'cym_Latn', 'som_Latn', 'ydd_Hebr', 'pbt_Arab', 'gle_Latn', 'snd_Arab', 'gaz_Latn', 'nno_Latn', 'hye_Armn']
xlmr_25_unseen_lrl_sib200_ds_names = ['szl_Latn', 'yor_Latn', 'ssw_Latn', 'ace_Latn', 'lmo_Latn', 'kas_Arab', 'pag_Latn', 'min_Arab', 'tuk_Latn', 'ltg_Latn', 'run_Latn', 'lin_Latn', 'fij_Latn', 'tum_Latn', 'ckb_Arab', 'tat_Cyrl', 'kac_Latn', 'tsn_Latn', 'ltz_Latn', 'kik_Latn', 'lim_Latn', 'crh_Latn', 'twi_Latn', 'scn_Latn', 'sot_Latn']

xlmr_24_divers_seen_sib200_ds_names = [
    "tir_Ethi", "arz_Arab", #"kab_Latn", issue
    "swh_Latn", "xho_Latn",
    "khm_Khmr", "vie_Latn",
    "ind_Latn", "zsm_Latn",
    "eus_Latn",
    "epo_Latn",
    "hye_Armn", "bos_Latn",
    "kat_Geor",
    "kor_Hang",
    "khk_Cyrl",
    "mya_Mymr", "zho_Hant",
    "lao_Laoo", "tha_Thai",
    "uig_Arab", "kaz_Cyrl",
    "fin_Latn", "hun_Latn"
]

xlmr_24_divers_unseen_sib200_ds_names = [
    "ajp_Arab", "amh_Ethi", "kab_Latn"
    "yor_Latn", "sna_Latn", "run_Latn",
    "ceb_Latn", "bjn_Arab", "ban_Latn",
    "ayr_Latn",
    "ckb_Arab", "mag_Deva", "sin_Sinh",
    "dyu_Latn",
    "knc_Arab",
    "quy_Latn",
    "lus_Latn", "mni_Beng", "yue_Hant",
    "shn_Mymr",
    "grn_Latn",
    "bak_Cyrl", "crh_Latn", "tat_Cyrl"
]

xlmr_seen_family_groups_sib200_ds_names = [
    # Afro-Asiatic
    'hau_Latn',
    'heb_Hebr',
    'arb_Arab',
    # Atlantic-Congo
    'swh_Latn',
    'xho_Latn',
    # Austroasiatic
    'khm_Khmr',
    'vie_Latn',
    # Austronesian
    'sun_Latn',
    'ind_Latn',
    'zsm_Latn',
    # Indo-European
    'eng_Latn',
    'fra_Latn',
    'spa_Latn',
    # Sino-Tibetan
    'mya_Mymr',
    'zho_Hant',
    'zho_Hans',
    # Tai-Kadai
    'lao_Laoo',
    'tha_Thai',
    # Turkic
    'kaz_Cyrl',
    'uzn_Latn',
    'tur_Latn'
]

enarzho_sib200_ds_names = [ 'eng_Latn', 'arb_Arab', 'zho_Hans' ]

source_ds_name = 'source'
def get_ds_names(benchmark_name, llm):

    # EN ================================================
    if benchmark_name == 'sib200_en':
        return 'eng_Latn', ['eng_Latn'], list(set(xlmr_seen_sib200_ds_names) - set(['eng_Latn'])) + xlmr_unseen_sib200_ds_names

    if benchmark_name == 'sib200_en_seen':
        return 'eng_Latn', ['eng_Latn'], list(set(xlmr_seen_sib200_ds_names) - set(['eng_Latn']))
        
    if benchmark_name == 'sib200_en_unseen':
        return 'eng_Latn', ['eng_Latn'], xlmr_unseen_sib200_ds_names

    if benchmark_name == 'sib200_en_ablation':
        return 'eng_Latn', ['eng_Latn'], xlmr_25_seen_lrl_sib200_ds_names + xlmr_25_unseen_lrl_sib200_ds_names

    # ENARZHO ================================================
    if benchmark_name == 'sib200_enarzho':
        return f'{source_ds_name}_{llm}_enarzho', enarzho_sib200_ds_names, not_joshi5_sib200_ds_names
    
    if benchmark_name == 'sib200_enarzho_ablation':
        return f'{source_ds_name}_{llm}_enarzho', enarzho_sib200_ds_names, xlmr_25_seen_lrl_sib200_ds_names + xlmr_25_unseen_lrl_sib200_ds_names

    if benchmark_name == 'sib200_enarzho_divers_ablation':
        return f'{source_ds_name}_{llm}_enarzho', enarzho_sib200_ds_names, xlmr_24_divers_seen_sib200_ds_names + xlmr_24_divers_unseen_sib200_ds_names
    
    # Joshi5 ================================================
    if benchmark_name == 'sib200_joshi5':
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, not_joshi5_sib200_ds_names
    
    if benchmark_name == 'sib200_joshi5_ablation':
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, xlmr_25_seen_lrl_sib200_ds_names + xlmr_25_unseen_lrl_sib200_ds_names
    
    if benchmark_name in [
        'sib200_joshi5_divers_ablation', 
        'sib200_joshi5_divers_24'
    ]:
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, xlmr_24_divers_seen_sib200_ds_names + xlmr_24_divers_unseen_sib200_ds_names
    
    if benchmark_name == 'sib200_joshi5_divers_ablation_unseen':
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, xlmr_24_divers_unseen_sib200_ds_names

    if benchmark_name == 'sib200_joshi5_abl_unseen':
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, xlmr_25_unseen_lrl_sib200_ds_names

    if benchmark_name == 'sib200_joshi5_unseen':
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, xlmr_unseen_sib200_ds_names
    
    if benchmark_name == 'sib200_joshi5_joshi5':
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, joshi5_sib200_ds_names

    # XLM-R seen ================================================
    if benchmark_name == 'sib200_xlmr_seen':
        return f'{source_ds_name}_{llm}_seen', xlmr_seen_sib200_ds_names, xlmr_unseen_sib200_ds_names

    if benchmark_name == 'sib200_xlmr_seen_family_groups':
        return f'{source_ds_name}_{llm}_seen_family_groups', xlmr_seen_family_groups_sib200_ds_names, xlmr_unseen_sib200_ds_names

    if benchmark_name == 'sib200_xlmr_seen_seen':
        return f'{source_ds_name}_{llm}_seen', xlmr_seen_sib200_ds_names, xlmr_seen_sib200_ds_names
    
    if benchmark_name == 'sib200_xlmr_seen_abl_unseen':
        return f'{source_ds_name}_{llm}_seen', xlmr_seen_sib200_ds_names, xlmr_25_unseen_lrl_sib200_ds_names
    
    if benchmark_name == 'sib200_xlmr_seen_ablation':
        return f'{source_ds_name}_{llm}_seen', xlmr_seen_sib200_ds_names, xlmr_25_seen_lrl_sib200_ds_names + xlmr_25_unseen_lrl_sib200_ds_names

    # XLM-R unseen ================================================
    if benchmark_name == 'sib200_xlmr_unseen':        
        return f'{source_ds_name}_{llm}_unseen', xlmr_unseen_sib200_ds_names, xlmr_seen_sib200_ds_names

    # All ================================================
    if benchmark_name == 'sib200_all_xlmr_unseen':
        return f'{source_ds_name}_{llm}_all', sib200_ds_names, xlmr_unseen_sib200_ds_names

    if benchmark_name == 'sib200_all_ablation':
        return f'{source_ds_name}_{llm}_all', sib200_ds_names, xlmr_25_seen_lrl_sib200_ds_names + xlmr_25_unseen_lrl_sib200_ds_names

    if benchmark_name == 'sib200_all':
        return f'{source_ds_name}_{llm}_all', sib200_ds_names, sib200_ds_names
    
    raise ValueError(f'Invalid benchmark_name: {benchmark_name}')

# 
# Get Configuration by SLURM Task
# 

def get_config_by_slurm_task(config_name, ds_name, supervision_regime):
    source_model_uuid4 = None

    s_job_id = int(os.getenv('SLURM_JOB_ID', 0))
    s_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))

    config = CONFIG.load(config_name, ConfigTypeSE.LANGUAGE_MODEL)
    s_task_conf_id = s_task_id

    s_task_configs = {
        # test run
        0: (f'TEST', 20, 0, 2, None, 10, 32, 0.00005, 'adafactor', 24, XPE_RE.NONE, '', 0.005, 0.0, False),

        # XLT run on enarzho, joshi5, and seen
        1: (f'SPT', 20, 0, 2, None, 10, 32, 0.00005, 'adafactor', 24000, XPE_RE.NONE, '', 0.005, 0.0, False),
        2: (f'D30', 20, 0.3, 2, None, 10, 32, 0.00005, 'adafactor', 24000, XPE_RE.MLP, '', 0.005, 0.0, False),
        3: (f'D70', 20, 0.7, 2, None, 10, 32, 0.00005, 'adafactor', 24000, XPE_RE.MLP, '', 0.005, 0.0, False),
        4: (f'XPE', 20, 1, 2, None, 10, 32, 0.00005, 'adafactor', 24000, XPE_RE.MLP, '', 0.005, 0.0, False),
        
        # LID over joshi5, and seen.
        101: (f'LID_SPT_SEEN', 20, 0, 2, None, 5, 32, 0.00005, 'adafactor', 0, XPE_RE.NONE, '', 0, 0.0, True),
        102: (f'LID_XPE_SEEN', 20, 1, 2, None, 5, 32, 0.00005, 'adafactor', 0, XPE_RE.MLP, '', 0, 0.0, True),
        103: (f'LID_SPT_J5', 20, 0, 2, None, 5, 32, 0.00005, 'adafactor', 0, XPE_RE.NONE, '', 0, 0.0, True),
        104: (f'LID_XPE_J5', 20, 1, 2, None, 5, 32, 0.00005, 'adafactor', 0, XPE_RE.MLP, '', 0, 0.0, True),
    }

    conf = s_task_configs[s_task_conf_id]

    if supervision_regime:
        s_task_conf_name = f'F_{conf[0]}'
        config.test.zero_shot_only = False
    else:
        s_task_conf_name = f'Z_{conf[0]}'
        config.test.zero_shot = True
        config.test.zero_shot_only = True

    config.task.peft.num_virtual_tokens = conf[1]
    config.task.peft.encoder_ratio = conf[2]
    config.task.peft.encoder_num_layers = conf[3]
    config.task.peft.encoder_input_size = conf[4]
    config.training_args.num_train_epochs = conf[5]
    config.training_args.per_device_train_batch_size = conf[6]
    config.training_args.learning_rate = conf[7]
    lr_type = conf[8]
    if lr_type == 'adafactor':
        config.training_args.optim = lr_type
        config.training_args.optim_args = SimpleNamespace(
            beta1=None,
            decay_rate=-0.8,
            weight_decay=0.01,
            clip_threshold=1.0,
            scale_parameter=True,
            relative_step=True,
            warmup_init=True
        )

    max_steps = conf[9]
    if max_steps: config.training_args.max_steps = max_steps

    encoder_type = conf[10]
    config.task.peft.encoder_reparameterization_type = encoder_type

    if len(conf) > 11:
        source_model_uuid4 = conf[11]
    if len(conf) > 12:
        spec_lr = conf[12]
        if spec_lr:
            config.custom_training_args.optimizer_grouped_parameters[0].lr = spec_lr
        spec_wd = conf[13]
        config.custom_training_args.optimizer_grouped_parameters[0].weight_decay = spec_wd
        if encoder_type == XPE_RE.NONE or config.model.architecture == ModelArchSE.XGLM:
            config.custom_training_args.optimizer_grouped_parameters[0].param_name_parts = ['embedding'] # concerns all prompt encoder embeddings
        elif encoder_type == XPE_RE.MLP:
            config.custom_training_args.optimizer_grouped_parameters[0].param_name_parts = ['xpe_embedding'] # concerns only dedicated embeddings
    if len(conf) > 14:
        config.task.peft.encoder_freeze = conf[14]

    return config, s_task_conf_name, f'{s_job_id}_{s_task_id}', source_model_uuid4

# 

ds_base_dirs = ''
def set_ds_dirs(config, name, suffix_dirs = None, subset_dirs = False):
    global ds_base_dirs
    if not ds_base_dirs: ds_base_dirs = config.ds.dirs
    config.ds.descriptive_name = name
    config.ds.dirs = f'{ds_base_dirs}/{name}' if name else ds_base_dirs
    config.ds.dirs = f'{config.ds.dirs}/{suffix_dirs}' if suffix_dirs else config.ds.dirs
    config.ds.dirs = f'{config.ds.dirs}/subset/{subset_dirs}' if subset_dirs else config.ds.dirs
    return config

encoder_freeze = None
num_train_epochs = 0
max_steps = 0
zero_shot_only = None
zero_shot = None
batch_size = 0
def update_config(config, ds_name, task_id, suffix_dirs = None, 
                  preproc_subset_ratio = None, subset_dirs = False, 
                  source_model_uuid4 = None, source_training = False):
    
    config = set_ds_dirs(config, ds_name, suffix_dirs, subset_dirs)
    if preproc_subset_ratio:
        config.ds.preproc_rules.subset.use = preproc_subset_ratio

    # Tasks
    config.task.id = task_id
            
    # Target Preparations
    if getattr(config.task, 'peft', None):

        global num_train_epochs, max_steps, zero_shot, zero_shot_only, encoder_freeze, batch_size
        batch_size = batch_size or config.training_args.per_device_train_batch_size
        zero_shot = zero_shot or config.test.zero_shot
        zero_shot_only = zero_shot_only if zero_shot_only is not None else config.test.zero_shot_only
        encoder_freeze = encoder_freeze if encoder_freeze is not None else config.task.peft.encoder_freeze
        num_train_epochs = num_train_epochs or config.training_args.num_train_epochs
        max_steps = max_steps or getattr(config.training_args, 'max_steps', None)

        if not source_training:
            common.p(f'\n[bold green]Target Training Phase![/bold green]')

            assert source_model_uuid4, 'source_model_uuid4 is required for target training'
            common.p(f'\nsource_model_uuid4: {source_model_uuid4}')

            config.test.zero_shot = zero_shot
            config.test.zero_shot_only = zero_shot_only
            config.task.peft.encoder_freeze = encoder_freeze
            config.custom_training_args.early_stopping_patience = 30
            config.training_args.per_device_train_batch_size = batch_size

            config.training_args.warmup_ratio = 0.05
            config.training_args.weight_decay = 0.01
            config.training_args.max_grad_norm = 1.0

            if max_steps:
                config.training_args.max_steps = int(max_steps/4)
            else:
                config.training_args.num_train_epochs = num_train_epochs * 5

            if config.task.peft.peft_type == PeftType.P_TUNING:
                config.task.peft.encoder_dropout = 0.0
                config.task.peft.encoder_init_state_dict_path = MODEL.get_last_checkpoint_path_by_uuid4(source_model_uuid4)
        
        elif source_training:
            
            common.p(f'\n[bold green]Source Training Phase![/bold green]')

            config.test.zero_shot = False
            config.test.zero_shot_only = False
            config.task.peft.encoder_freeze = False
            config.custom_training_args.early_stopping_patience = 20
            config.training_args.per_device_train_batch_size = 32

            config.training_args.warmup_ratio = 0.1
            config.training_args.weight_decay = 0.01
            config.training_args.max_grad_norm = 1.0

            if max_steps:
                config.training_args.max_steps = max_steps
            else:
                config.training_args.num_train_epochs = num_train_epochs

            if config.task.peft.peft_type == PeftType.P_TUNING:
                config.task.peft.encoder_dropout = 0.1
                config.task.peft.encoder_init_state_dict_path = None

    return config
