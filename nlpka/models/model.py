import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

import os
import math
import copy
import uuid
import torch
import wandb
from wandb.sdk.wandb_settings import Settings
import random
import shutil
import logging
import numpy as np
import pandas as pd
# from termcolor import colored
from types import SimpleNamespace
from transformers import logging as hf_logging
# from transformers.trainer_utils import get_last_checkpoint
from datetime import datetime
import nlpka.models.storage as model_stor
from nlpka.models.peft import PEFT
from nlpka.evaluations.evaluate import EVALUATE
from nlpka.tokenizers.tokenizer import TOKENIZER
from nlpka.models.logits_processors import ConstrainedPrefixLogitsProcessor
from nlpka.models.trainer_callbacks import DownstreamFineTuningCallback, EmptyCudaCacheCallback, ParamNormLogger, NormalizePromptEncoderEmbeddings, LossEarlyStoppingCallback
from nlpka.models.cross_prompt_encoder import is_cross_prompt_encoder

from nlpka.models.custom_trainer import custom_trainer_class_factory
from nlpka.models.custom_models import CustomT5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoModel, BertConfig, XLNetConfig, ElectraConfig, RobertaConfig, GenerationConfig
from transformers import AutoModel, BertForMaskedLM, XLNetLMHeadModel, ElectraForMaskedLM, RobertaForMaskedLM
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, AutoModelForSeq2SeqLM
from transformers import AutoModelForTokenClassification, XLNetForTokenClassification
from transformers import DataCollatorForLanguageModeling, DataCollatorForPermutationLanguageModeling, DataCollatorWithPadding, DataCollatorForTokenClassification, DataCollatorForSeq2Seq
from nlpka.models.custom_data_collators import DataCollatorForPLMWithPadding, DataCollatorTaskIDDecorator, ShiftLabelsDataCollatorForSeq2Seq, CustomDataCollatorWithPadding

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from nlpka.tools.enums import DeviceSE, DsSplitSE, ModeSE, ModelArchSE, PretSourceSE, TaskCatSE

class MODEL:

    _logs_dir = 'logs'
    checkpoint_pref = 'checkpoint-'
    stor_path = common.get_module_location(model_stor)

    def __init__(self, config, tokenizer, dataset):
        self._config = copy.deepcopy(config)
        self._tokenizer = tokenizer
        self._dataset = dataset
        self._set_paths()
        # self._setup_logs()
        self._setup_model()
        self._setup_trainer()
        self.print_details()

    def reinit(self, config, tokenizer, dataset):
        self.__init__(config, tokenizer, dataset)

    def _use_seq2seq_trainer(self):
        return self._config.model.architecture in [ ModelArchSE.T5 ]

    def get(self):
        '''
        Get huggingface's model object
        '''
        return self._model

    def run(self):
        '''
        Run Model According to the Configuration
        '''

        full_shot_res = None
        zero_shot_res = None
        test_pref = 'test'
        test_z_pref = 'test_zero'
        run_test = getattr(self._config.test, 'run', True)
        zero_shot = getattr(self._config.test, 'zero_shot', False)
        zero_shot_only = getattr(self._config.test, 'zero_shot_only', False)
        run_eval_before_train = getattr(self._config.eval, 'before_training', False)
        run_eval_after_train = getattr(self._config.eval, 'after_training', False)
        run_eval_before_train_on_test = getattr(self._config.eval, 'before_training_on_test', False)
        run_eval_after_train_on_test = getattr(self._config.eval, 'after_training_on_test', False)

        # Initialize Weights and Biases
        self._model.wandb_run = self._init_wandb() if not wandb.run else None

        # Zero Shot Testing
        if run_test and zero_shot:
            zero_shot_res = self._test(test_z_pref)

        if not zero_shot or (zero_shot and not zero_shot_only):

            # Training and Evaluation
            if self._config.mode in [ ModeSE.TRAIN, ModeSE.FINETUNE ]:

                # Zero Shot Evaluation
                self._evaluate() if run_eval_before_train else None
                self._evaluate(DsSplitSE.TEST, test_z_pref) if run_eval_before_train_on_test else None

                # Training
                self._train()

                # Final Evaluation
                self._evaluate() if run_eval_after_train else None
                self._evaluate(DsSplitSE.TEST, test_pref) if run_eval_after_train_on_test else None

            # Evaluation
            elif self._config.mode == ModeSE.EVALUATE:
                self._evaluate()
                self._evaluate(DsSplitSE.TEST, test_pref)

            # Final Testing
            if run_test: 
                full_shot_res = self._test(test_pref)

        # Finish Weights and Biases
        if self._model.wandb_run:
            self._model.wandb_run.finish()

        return SimpleNamespace(
            full_shot = full_shot_res,
            zero_shot = zero_shot_res,
        )

    def _train(self):
        common.p('\n[green]Train Model...[/green]')
        print(f'Task name:  {self._config.task.name}')
        self.trainer.compute_metrics = EVALUATE.get_compute_metrics(self._config, self.label_pad_id, self._metric_prefix, self.path, self._tokenizer, self._dataset.validation)
        self.trainer.train()

        self._load_best_model()

        if getattr(self._config.custom_training_args, 'save_final_model', False):
            shutil.rmtree(self.path) if getattr(self._config.custom_training_args, 'keep_only_final_model', False) else None
            self.trainer.save_model(self.path)

    def _evaluate(self, ds_split_name = DsSplitSE.VALIDATION, metric_key_prefix = 'eval'):
       
        eval_res = None
        ds_split = self._dataset.validation if ds_split_name == DsSplitSE.VALIDATION else self._dataset.test
        if ds_split:
            common.p(f'\n[green]Evaluate Model on [bold]{ds_split_name}[/bold] set...[/green]')
            print(f'Task name: {self._config.task.name}')

            if self._config.model.architecture == ModelArchSE.XLNET:
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
                print('Suppressing Errors for XLNET: torch._dynamo.config.suppress_errors = True')
                
            self.trainer.compute_metrics = EVALUATE.get_compute_metrics(
                self._config, self.label_pad_id, self._metric_prefix, self.path, self._tokenizer, ds_split
            )
            eval_res = self.trainer.evaluate(ds_split, metric_key_prefix = metric_key_prefix)
        return eval_res

    def _test(self, metric_key_prefix = 'test'):
        test_res = None
        if self._dataset.test: 
            common.p('\n[green]Test Model...[/green]')
            print('Task name: ', self._config.task.name)

            self.trainer.compute_metrics = EVALUATE.get_compute_metrics(
                self._config, self.label_pad_id, self._metric_prefix, self.path, self._tokenizer, self._dataset.test
            )

            if self._config.mode in [ ModeSE.TRAIN, ModeSE.FINETUNE, ModeSE.EVALUATE ]:
                test_res = self.trainer.predict(self._dataset.test, metric_key_prefix=metric_key_prefix)
                common.p(test_res.metrics)

                if self._config.test.save_predictions:
                    predictions_to_save = []
                    labels = test_res.label_ids
                    predictions = test_res.predictions
                    # print(len(predictions), len(labels), len(self._dataset.test))
                    # print(predictions, labels)
                    # exit()
                    if self._config.eval.filter_padded:
                        masked_preds = []
                        masked_labs = []
                        for preds, labs in zip(predictions, labels):
                            mask = (labs != self.label_pad_id)
                            masked_preds.append(preds[mask])
                            masked_labs.append(labs[mask])
                        predictions = masked_preds
                        labels = masked_labs
                    if self._config.eval.label_id_to_name:
                        label_names = np.array(self._config.ds.label.names)
                        named_preds = []
                        named_labs = []
                        for preds, labs in zip(predictions, labels):
                            named_preds.append(label_names[preds])
                            named_labs.append(label_names[labs])
                        predictions = named_preds
                        labels = named_labs
                    # print(len(predictions), len(labels), len(self._dataset.test))
                    # print(predictions, labels)
                    # exit()
                    print_num = 5
                    for prediction, label, sample in zip(predictions, labels, self._dataset.test):
                        input_ids = sample[self._dataset.keys.input_ids]
                        readable_tokens = [self._tokenizer.decode(tok_id) for tok_id in input_ids]
                        if self._config.task.category in [TaskCatSE.TEXT_CLASSIFICATION, TaskCatSE.TEXT_PAIR_CLASSIFICATION]:
                            readable_tokens = ' '.join(readable_tokens)
                        elif self._config.task.category == TaskCatSE.TOKEN_CLASSIFICATION:
                            # predictions_to_save.extend([prediction, label, readable_tokens, []])
                            # Convert numpy arrays to lists if not already in list format
                            predictions_row = prediction.tolist() if isinstance(prediction, np.ndarray) else list(prediction)
                            labels_row = label.tolist() if isinstance(label, np.ndarray) else list(label)
                            tokens_row = readable_tokens[1:-1]  # Token texts are already in list format
                            # Append each row to the predictions_to_save list
                            predictions_to_save.extend([predictions_row, labels_row, tokens_row, []])  # Include an empty list for separation

                        if print_num: 
                            print_num -= 1
                            common.p('Predicted: ', prediction, 'True: ', label, 'Text: ', readable_tokens)
                
                    eval_stor_path = EVALUATE.stor_path
                    csv_file_path = f'{eval_stor_path}/predictions/{self.name}.csv'
                    
                    if self._config.task.category in [TaskCatSE.TEXT_CLASSIFICATION, TaskCatSE.TEXT_PAIR_CLASSIFICATION, TaskCatSE.TOKEN_CLASSIFICATION]:
                        if self._config.task.category in [TaskCatSE.TEXT_CLASSIFICATION, TaskCatSE.TEXT_PAIR_CLASSIFICATION]:
                            df = pd.DataFrame(predictions_to_save, columns=['Predicted', 'True', 'Text'])
                            df.to_csv(csv_file_path, index=False)
                        elif self._config.task.category == TaskCatSE.TOKEN_CLASSIFICATION:
                            # Convert the list of data to a DataFrame
                            # The maximum length of row determines the number of columns
                            max_len = max(len(row) for row in predictions_to_save)
                            # Standardize the length of each row
                            standardized_rows = [row + [''] * (max_len - len(row)) for row in predictions_to_save]
                            df = pd.DataFrame(standardized_rows)
                            df.to_csv(csv_file_path, index=False)
                        print(f"Predictions data to {csv_file_path}")
                    
            elif self._config.mode == ModeSE.TEST:
                if self._config.task.category == TaskCatSE.TEXT_SIMILARITIY:
                    EVALUATE.compute_word_similarity(self._model, self._dataset.test, self._tokenizer)
        return test_res

    def _load_best_model(self):
        best_checkpoint_path = self.trainer.state.best_model_checkpoint
        if best_checkpoint_path:
            best_checkpoint = int(best_checkpoint_path.split('/')[-1].replace(self.checkpoint_pref, ''))
            current_checkpoint = self.trainer.state.global_step
            if best_checkpoint != current_checkpoint:
                common.p('\n[green]Custom load of the best model...[/green]')
                print('Path: ', best_checkpoint_path)
                if PEFT.is_peft(self):
                    self.trainer.model = PEFT.from_pretrained(self.trainer.model.base_model, best_checkpoint_path)
                else:
                    self.trainer.model = self.trainer.model.from_pretrained(best_checkpoint_path)
        else: 
            common.p('\n[red]No best model found...[/red]')

        temp_wandb_run = self._model.wandb_run
        self._model = self.trainer.model
        self._model.wandb_run = temp_wandb_run

    def _get_init_path(self, name = ''):
        init_path = f'{self.stor_path}/{self._config.model.architecture}'
        return f'{init_path}/{name}' if name else init_path

    def _set_paths(self):
        mode = self._config.mode
        # Set Pretrained Model Path
        if mode in [ ModeSE.TEST, ModeSE.FINETUNE, ModeSE.EVALUATE ]:
            self.pret_path = self._get_pret_path()
        # Set Model Path
        if mode in [ ModeSE.TRAIN, ModeSE.FINETUNE, ModeSE.EVALUATE ]:
            self._set_name()
            if mode == ModeSE.TRAIN:
                self.path = self._get_init_path(self.name)
            elif mode in [ ModeSE.FINETUNE, ModeSE.EVALUATE ]:
                task_name = self._config.task.name
                if self._config.model.pretrained.source == PretSourceSE.HUGGINGFACE:
                    self.path = self._get_init_path(self.pret_path)
                else:
                    self.path = self.pret_path
                self.path = f'{self.path}/{task_name}/{self.name}'
            self.logs_path = f'{self.path}/{self._logs_dir}'
            MODEL.store_path_by_uuid4_in_envs(self.uuid4, self.path)

    def _set_name(self):
        # Gnerate a unic time ID for the model
        self.uuid4 = str(uuid.uuid4())
        self._config.model.uuid4 = self.uuid4
        slurm_job_id = os.environ.get('SLURM_JOB_ID', None)
        slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', None)
        # Get the pretrained model time ID if available
        pret = self._config.model.pretrained
        self.pret_uuid4 = self._get_pret_uuid4()
        pret_info = self.pret_uuid4 if self.pret_uuid4 else pret.name.replace('/', '|')
        # Calculate the effective batch size (TO DO: Consider also the parrallelism)
        self.effective_batch_size = (
            self._config.training_args.per_device_train_batch_size
            * 
            self._config.training_args.gradient_accumulation_steps
        )
        self.training_steps = math.ceil(len(self._dataset.train) / self.effective_batch_size) * self._config.training_args.num_train_epochs
        # print(f'Num Train Epochs: {self._config.training_args.num_train_epochs} | effective_batch_size: {self.effective_batch_size} | Training Steps: {self.training_steps} | Dataset Length: {len(self._dataset.train)}')
        # Generate the model name
        name_params = [
            self.uuid4,
            pret_info,
            slurm_job_id,
            slurm_task_id,
            self._config.training_args.num_train_epochs,
            self.effective_batch_size,
            self._config.ds.dirs.replace('/','|'),
            self._dataset.name,
        ]
        self.name = '_'.join([ str(p) for p in name_params if p is not None])

    def _get_pret_uuid4(self):
        pret_uuid = None
        pret = self._config.model.pretrained
        if pret.source == PretSourceSE.LOCAL:
            pret_uuid4 = getattr(pret, 'uuid4', None)
            if not pret_uuid4 and pret.name:
                name_prefix = pret.name[:38]
                if common.is_valid_uuid(name_prefix):
                    pret_uuid4 = name_prefix
        return pret_uuid

    def _get_pret_path(self, pret = None):

        pret_path = None
        pret = pret if pret else self._config.model.pretrained

        if pret.source == PretSourceSE.HUGGINGFACE:
            pret_path = pret.name

        elif pret.source == PretSourceSE.LOCAL:
            pret_path = self._get_init_path(pret.name) if pret.name else MODEL.find_path_by_uuid4(pret.uuid4)
            pret_path = MODEL.get_last_checkpoint_path(pret_path, pret.checkpoint)

        return pret_path

    def _set_model_properties(self):
        # Set model embedding dimention and sequence length
        self._set_max_length()
        self._set_embedding_dim()
        # Count and Set the total number of parameters in the model
        self.param_size = sum(p.numel() for p in self._model.parameters())
        self.trainable_param_size = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self.trainable_param_size_ratio = self.trainable_param_size/self.param_size*100
        self._config.model.param_size = f'{self.param_size:,}'
        self._config.model.trainable_param_size = f'{self.trainable_param_size:,}'
        self._config.model.trainable_param_size_ratio = f'{self.trainable_param_size_ratio:.4f}'
        # Set label padding Id
        self.label_pad_id = getattr(self._config.ds.label, 'padded', -100)

    def _set_max_length(self):
        self.max_length = \
            getattr(self._model.config, "max_position_embeddings", 
                getattr(self._model.config, "n_positions", 
                    getattr(self._model.config, "mem_len", 
                        getattr(self._model.config, "max_length", None)
                    )
                )
            )
        if self.max_length is None:
            print("⚠️ Warning: max_length is not set in the model configuration.")
        
    def _set_embedding_dim(self):
        self.embedding_dim = \
            getattr(self._model.config, "hidden_size", 
                getattr(self._model.config, "d_model", None)
            )
        if self.embedding_dim is None:
            print("⚠️ Warning: embedding_dim is not set in the model configuration.")

    def _setup_logs(self):
        '''
        Setup Logs
        '''
        # logging.basicConfig(level=logging.DEBUG)
        hf_logging.set_verbosity_info()
        hf_logging.enable_default_handler()
        hf_logging.enable_explicit_format()

    def _setup_model(self):
        print('Setup Model...')

        if self._config.mode in [ ModeSE.FINETUNE, ModeSE.EVALUATE, ModeSE.TEST ]:
            if not self.pret_path: raise Exception('Pretrained model path not set')

        if not hasattr(self, '_model'):
            task = self._config.task.category
            if task == TaskCatSE.LANGUAGE_MODELING:
                self._setup_language_model()
            elif task in [TaskCatSE.TEXT_CLASSIFICATION, TaskCatSE.TEXT_PAIR_CLASSIFICATION]:
                self._setup_text_classif_model()
            elif task == TaskCatSE.TOKEN_CLASSIFICATION:
                self._setup_token_classif_model()
            elif task == TaskCatSE.TEXT_TO_TEXT:
                self._setup_text_to_text_model()
            elif task == TaskCatSE.TEXT_SIMILARITIY:
                self._model = AutoModel.from_pretrained(self.pret_path)

            if not self._model: raise Exception('Model not set')

            if getattr(self._config.model.pretrained, 'save_as', None) == PretSourceSE.HUGGINGFACE:
                print(f'Saving model to {self.path}')
                self._model.save_pretrained(self.path)

            PEFT.setup_model(self) if PEFT.is_peft(self) else None

            self._set_model_properties()
            self._setup_model_device()

    def _setup_model_device(self):
        """
        Set model to available device (CUDA or CPU)
        """
        device = DeviceSE.CUDA if torch.cuda.is_available() else DeviceSE.CPU
        print(f'\nSending Model to available {device} device...')
        self.device = torch.device(device)
        self._model.to(self.device)

    def _setup_language_model(self):

        if self._config.model.architecture == ModelArchSE.BERT:
            ModelConfig = BertConfig
            LanguageModel = BertForMaskedLM

        elif self._config.model.architecture == ModelArchSE.ELECTRA:
            ModelConfig = ElectraConfig
            LanguageModel = ElectraForMaskedLM

        elif self._config.model.architecture == ModelArchSE.ROBERTA:
            ModelConfig = RobertaConfig
            LanguageModel = RobertaForMaskedLM

        elif self._config.model.architecture == ModelArchSE.XLNET:
            ModelConfig = XLNetConfig
            LanguageModel = XLNetLMHeadModel
        
        else:
            ModelConfig = AutoModel
            LanguageModel = AutoModel
        
        # Set Model 
        if self._config.mode in [ ModeSE.FINETUNE, ModeSE.EVALUATE ]:
            self._model = LanguageModel.from_pretrained(self.pret_path)

        elif self._config.mode == ModeSE.TRAIN:

            model_config = ModelConfig(
                # vocab_size = self._tokenizer.vocab_size,
                vocab_size = len(self._tokenizer.get_vocab()),
                **vars(self._config.model_config),
            )
            self._model = LanguageModel(config=model_config)

    def _setup_text_classif_model(self):
        # Set Model
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.pret_path, num_labels=self._config.ds.label.number,
        )
        # self._model = T5ForConditionalGeneration.from_pretrained(
        #     self.pret_path, #num_labels=self._config.ds.label.number,
        # )
        # if self._config.model.pretrained.source == PretSourceSE.HUGGINGFACE:
        #     self._model = AutoModelForSequenceClassification.from_pretrained(self.pret_path, num_labels=self._config.ds.label.number)
        # elif self._config.model.pretrained.source == PretSourceSE.LOCAL:
        #     if self._config.model.architecture == ModelArchSE.BERT:
        #         self._model = BertForSequenceClassification.from_pretrained(self.pret_path, num_labels=self._config.ds.label.number)

    def _setup_token_classif_model(self):
        # Set Model 
        if self._config.model.architecture == ModelArchSE.XLNET:
            xlnet_config = XLNetConfig.from_pretrained(self.pret_path)
            xlnet_config.use_mems_eval = False
            self._model = XLNetForTokenClassification(config=xlnet_config)
        else:
            self._model = AutoModelForTokenClassification.from_pretrained(
                self.pret_path, num_labels=self._config.ds.label.number
            ) 

    def _setup_text_to_text_model(self):
        arch = self._config.model.architecture
        # Encoder-decoder: T5, mT5, BART, mBART
        # TODO: self._model.config.is_encoder_decoder
        if arch in [ ModelArchSE.T5 ]:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.pret_path) # For T5 it will load a T5ForConditionalGeneration model
        # Decoder-only: XGLM, GPT, BLOOM
        elif arch in [ ModelArchSE.XGLM ]:
            self._model = AutoModelForCausalLM.from_pretrained(self.pret_path)
        else:
            raise ValueError(f"Unsupported model architecture for text-to-text: {arch}")
        
        # common.p('self._model.config', self._model.config)
        # exit()

    def _setup_data_collator(self):

        print('Setup data collator...')
        model_arch = self._config.model.architecture
        task = self._config.task.category

        if task == TaskCatSE.LANGUAGE_MODELING:
            if model_arch in [ ModelArchSE.BERT, ModelArchSE.ELECTRA, ModelArchSE.ROBERTA ]:
                DataCollator = DataCollatorForLanguageModeling
            elif model_arch == ModelArchSE.XLNET:
                DataCollator = DataCollatorForPermutationLanguageModeling
                # DataCollator = DataCollatorForPermutationLanguageModelingWithPadding

        else:
            if task in [TaskCatSE.TEXT_CLASSIFICATION, TaskCatSE.TEXT_PAIR_CLASSIFICATION]:
                # DataCollator = CustomDataCollatorWithPadding
                DataCollator = DataCollatorWithPadding
                
            elif task == TaskCatSE.TOKEN_CLASSIFICATION:
                DataCollator = DataCollatorForTokenClassification
                self._config.data_collator.label_pad_token_id = self.label_pad_id

            elif task == TaskCatSE.TEXT_TO_TEXT:
                DataCollator = DataCollatorForSeq2Seq
                self._config.data_collator.label_pad_token_id = self.label_pad_id
                
                # Decoder-only models with prompt encoder and virtual tokens, 
                # need to shift the labels by the total number of virtual tokens
                if self._config.model.architecture in [ ModelArchSE.XGLM ]:
                    prompt_encoder = getattr(self._model, "prompt_encoder", None)
                    if isinstance(prompt_encoder, torch.nn.ModuleDict) and "default" in prompt_encoder:
                        prompt_encoder = prompt_encoder["default"]
                    
                    if prompt_encoder is not None:
                        total_virtual_tokens = getattr(prompt_encoder, "total_virtual_tokens", None)
                        if total_virtual_tokens:
                            DataCollator = ShiftLabelsDataCollatorForSeq2Seq
                            self._config.data_collator.shift_labels_by = total_virtual_tokens

            max_length = getattr(self._config.data_collator, 'max_length', self.max_length)
            self._config.data_collator.max_length = min(max_length, self.max_length)
            # self._config.data_collator.max_length = min(max_length, self.max_length, (self._dataset.longest_input_len()+5))
        
        # Set Data Collator
        self.data_collator = DataCollator(
            **vars(self._config.data_collator),
            tokenizer = self._tokenizer,
        )

        # task_id = getattr(self._config.task, 'id', None)
        # if task_id is not None:
        #     self.data_collator = DataCollatorTaskIDDecorator(self.data_collator, task_id)

    def _setup_metrics(self):
        print('Setup evaluation functions...')

        # Set Metric Prefix
        if self._config.task.category == TaskCatSE.LANGUAGE_MODELING:
            self._metric_prefix = ''
        else:
            config_name = self._config.file_path.split('/')[-1]
            self._metric_prefix = f'{config_name}/'
            # # Including Pretrained Model Checkpoint in Metric Prefix                                                             
            # if condition:
            #     pret_checkpoint = self._config.model.pretrained.checkpoint
            #     self._metric_prefix += f'{pret_checkpoint}_' if pret_checkpoint else ''

        # Set Compute Metrics Function
        self.compute_metrics = EVALUATE.get_compute_metrics(self._config, self.label_pad_id, self._metric_prefix, self.path, self._tokenizer, self._dataset.validation)

        # Set Preprocessing Logits Function
        self.preprocess_logits_for_metrics = EVALUATE.get_preprocess_logits_for_metrics(self._config)

    def _setup_wandb(self):
        '''
        Setup Weights and Biases
        '''
        print('Setup Weights and Biases...')
        wandb_proj_name_params = [
            self._config.task.category,
            self._config.task.name
        ]
        self.wandb_project_name = '_'.join([ str(p) for p in wandb_proj_name_params if p is not None])
        # os.environ['WANDB_PROJECT'] = self.wandb_project_name
        # os.environ['WANDB_LOG_MODEL'] = 'true'

        # Nested Runs Example
        # import wandb
        # # Start the pretraining run
        # wandb.init(project="bert_pretraining", entity="your_entity", name="pretraining_run", id="pretrain1", resume="allow")
        # # At some point, initiate a nested run for fine-tuning
        # with wandb.init(project="bert_finetuning", entity="your_entity", name="sentiment_analysis_ft", reinit=True) as child_run:
        #     # Load model from a checkpoint saved by the pretraining run
        #     model = load_model(checkpoint_path)
        #     # Fine-tune the model on sentiment analysis
        #     # [Your training code here]
        #     child_run.log({"accuracy": 0.92, "loss": 0.08})
        # # The pretraining run continues...
        # # Log additional pretraining metrics or checkpoints
        # wandb.log({"pretraining_metric": value})
        # # When pretraining is completely finished
        # wandb.finish()

    # def _init_wandb(self):
    #     '''
    #     Initialize Weights and Biases
    #     '''
    #     wandb_run = wandb.init(
    #         name=self.name,
    #         project=self.wandb_project_name, 
    #         config=common.simple_nsp_to_dict(self._config),
    #     )   
    #     return wandb_run
        
    def _init_wandb(self):
        '''
        Initialize Weights and Biases
        Fallback to offline mode if init fails
        '''

        try:
            wandb_run = wandb.init(
                name=self.name,
                project=self.wandb_project_name, 
                config=common.simple_nsp_to_dict(self._config),
                settings=Settings(init_timeout=300)
            )
        except wandb.errors.CommError as e:
            print(f"[W&B WARNING] Init failed with CommError: {e}. Retrying in offline mode.")
            os.environ["WANDB_MODE"] = "offline"
            wandb_run = wandb.init(
                name=self.name + "_offline",
                project=self.wandb_project_name, 
                config=common.simple_nsp_to_dict(self._config),
                settings=Settings(init_timeout=300)
            )
        return wandb_run


    def _setup_trainer_callbacks(self):
        '''
        Setup Trainer Callbacks
        '''
        print('Setup Trainer Callbacks...')
        self._trainer_callbacks = []

        # Log the norm of the training parameters and their update sizes on each step
        self._trainer_callbacks.append(ParamNormLogger())

        # Free up the VRAM in every X training steps
        if self.device == DeviceSE.CUDA:
            steps = self._config.cuda.empty_cache_steps
            if steps: self._trainer_callbacks.append(EmptyCudaCacheCallback(steps))
        
        # Set callback for finetuning checkpoints on downstream tasks
        if self._config.eval.downstream_tasks:
            self._trainer_callbacks.append(DownstreamFineTuningCallback(self._config, self.path))

        if self._config.custom_training_args.early_stopping_patience:
            self._trainer_callbacks.append(LossEarlyStoppingCallback(
                early_stopping_patience=self._config.custom_training_args.early_stopping_patience,
                early_stopping_threshold=self._config.custom_training_args.early_stopping_threshold,
                early_stopping_after=self._config.custom_training_args.early_stopping_after,
            ))

        # Set cross prompt encoder callbacks
        peft = getattr(self._config.task, 'peft', None)
        if peft and is_cross_prompt_encoder(peft):
            self._trainer_callbacks.append(NormalizePromptEncoderEmbeddings())
            # self._trainer_callbacks.append(PromptEncoderSaver())
        
        common.p('List of Callbacks: ', self._trainer_callbacks)
     
    def _calculate_eval_steps(self):
        targs = self._config.training_args
        eval_during_train = getattr(self._config.eval, 'during_training', None)
        if eval_during_train == 0:
            return 0
        elif eval_during_train is None:
            return targs.eval_steps
        else:
            base_interval = self.training_steps // (eval_during_train + 1)
            offset = max(1, int(self.training_steps * 0.01))
            eval_steps = base_interval + offset
            # print(f'\nEval During Training: {eval_during_train} \nTraining Steps: {self.training_steps} \nOffset: {offset} \nEval Steps: {eval_steps}')
            # exit()
            return eval_steps

    def _setup_eval_steps(self):
        print('Setup Eval Strategy...')
        targs = self._config.training_args
        eval_during_train = getattr(self._config.eval, 'during_training', None)
        if eval_during_train is not None:        

            if eval_during_train == 0:
                targs.eval_strategy = 'no'
                targs.load_best_model_at_end = False
            elif eval_during_train > 0:
                targs.eval_strategy = 'steps'
                targs.eval_steps = self._calculate_eval_steps()

        if targs.eval_strategy != 'no' and targs.save_strategy != 'no' and getattr(targs, 'load_best_model_at_end', False):
            targs.save_strategy = targs.eval_strategy
            targs.save_steps = targs.eval_steps

        print(f' Eval Strategy: {targs.eval_strategy}')
        print(f' Eval Steps: {targs.eval_steps}')
        print(f' Save Strategy: {targs.save_strategy}')
        print(f' Save Steps: {targs.save_steps}')

    def _setup_training_args(self):
        print('Setup TrainingArguments...')
        self._setup_eval_steps()

        targs = self._config.training_args
        targs.fp16 = targs.fp16 and self.device == torch.device(DeviceSE.CUDA)
        # targs.torch_compile = targs.torch_compile and self.device == DeviceSE.CUDA
        # print(targs.torch_compile)
        # exit()
        
        # Randomly generate a seed (it affects: weight initialization, training data ordering, dropout)
        #     Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
        #     https://github.com/huggingface/transformers/issues/14647
        #     https://ådiscuss.huggingface.co/t/fixing-the-random-seed-in-the-trainer-does-not-produce-the-same-results-across-runs/3442/3
        reproducibility = getattr(targs, 'full_determinism', False)
        if not reproducibility:
            targs.seed = random.randint(0, 2**32-1)
        
        if hasattr(targs, 'metric_for_best_model'):
            targs.metric_for_best_model = self._metric_prefix + targs.metric_for_best_model
        
        if hasattr(targs, 'optim_args'):
            targs.optim_args = ', '.join(f"{key}={val}" for key, val in vars(targs.optim_args).items())

        if hasattr(targs, 'lr_scheduler_kwargs'):
            targs.lr_scheduler_kwargs = vars(targs.lr_scheduler_kwargs)

        global TrainingArguments
        if self._use_seq2seq_trainer():
            TrainingArguments = Seq2SeqTrainingArguments
            if hasattr(self._config, 'generation_config'):
                max_length = getattr(self._config.generation_config, 'max_length', 0)
                max_new_tokens = getattr(self._config.generation_config, 'max_new_tokens', 0)
                if not (max_new_tokens + max_length): 
                    self._config.generation_config.max_new_tokens = self._dataset.longest_label_len() + 5
                force_words_ids = getattr(self._config.generation_config, 'force_words_ids', None)
                if force_words_ids:
                    print('\nGeneration config > force_words_ids: ', force_words_ids)
                    force_words_ids = [
                        self._tokenizer(word, add_special_tokens=False).input_ids 
                        for word in force_words_ids
                    ]
                    self._config.generation_config.force_words_ids = force_words_ids
                
                generation_whitelist = getattr(self._config.custom_training_args, 'generation_whitelist', None)
                if generation_whitelist:
                    self._config.generation_config.early_stopping = False
                    self._config.generation_config.do_sample = False
                    self._config.generation_config.num_beams = 1

                targs.generation_config = GenerationConfig(**vars(self._config.generation_config))
        
        self.training_args = TrainingArguments(
            run_name = self.name,
            output_dir = self.path,
            logging_dir = self.logs_path,
            **vars(targs)
        )
        # common.p('training_args', self.training_args)
        # exit()

    def _setup_trainer(self):
        if self._config.mode in [ ModeSE.TRAIN, ModeSE.FINETUNE, ModeSE.EVALUATE ]:
         
            self._setup_wandb()
            self._setup_metrics()
            self._setup_trainer_callbacks()
            self._setup_data_collator()
            self._setup_training_args()

            # Trainer
            print('Setup Trainer...')
            trainer_init_args = {
                'model': self._model,
                'args': self.training_args,
                'processing_class': self._tokenizer,
                'data_collator': self.data_collator,
                'train_dataset': self._dataset.train,
                'eval_dataset': self._dataset.validation,
                'callbacks': self._trainer_callbacks,
                'compute_metrics': self.compute_metrics,
            }
            if not self._use_seq2seq_trainer():
                trainer_init_args['preprocess_logits_for_metrics'] = self.preprocess_logits_for_metrics

            global Trainer
            if self._use_seq2seq_trainer(): 
                Trainer = Seq2SeqTrainer

            CustomTrainer = custom_trainer_class_factory(Trainer)
            self.trainer = CustomTrainer(
                custom_args = self._config.custom_training_args,
                **trainer_init_args,
            )

    def print_named_parameters(self, requires_grad = None, model = None):
        model = model if model else self._model
        for name, param in model.named_parameters():
            if requires_grad == None or requires_grad == param.requires_grad:
                print(f'name: {name} | grad: {param.requires_grad} | mean: {param.mean().item()}') 

    def print_batch_examples(self, split, batches=2, samples_per_batch=2):
        dataloader = None
        if split == DsSplitSE.TRAIN:
            dataloader = self.trainer.get_train_dataloader()
        elif split == DsSplitSE.VALIDATION:
            dataloader = self.trainer.get_eval_dataloader()
        elif split == DsSplitSE.TEST:
            dataloader = self.trainer.get_test_dataloader(self._dataset.test)
        if not dataloader: return
        for i, batch in enumerate(dataloader):
            if i == batches: break
            if self._dataset.keys.input_ids in batch:
                input_ids = batch[self._dataset.keys.input_ids]
                print()
                print(f'{split.upper()} Batch {i+1} - size: {len(input_ids)} | max_len: {len(input_ids[0])}')
                print(f'{split.upper()} Batch {i+1} - items: {len(batch)}, keys: {batch.keys()}')

                for key, value in batch.items():
                    if torch.is_tensor(value):
                        print(f"    Tensor Item: {key}, Tensor Shape: {value.shape}, Device: {value.device}")

                for j, ids in enumerate(input_ids):
                    if j == samples_per_batch:
                        break
                    ids = ids.tolist()
                    all_count = len(ids)
                    pad_count = ids.count(self._tokenizer.pad_token_id)
                    real_count = all_count - pad_count
                    mask_count = ids.count(self._tokenizer.mask_token_id)

                    print(f'{split.upper()} Sample {j+1} - len: {real_count} | pads: {pad_count} | masks: {mask_count} ({(mask_count/real_count*100):.2f}%)')
                    print(f'Inputs: {self._tokenizer.decode(ids, skip_special_tokens=False)}')

                    # Check for labels
                    if self._dataset.keys.labels in batch:
                        labels = batch[self._dataset.keys.labels][j]
                        if self._config.task.category == TaskCatSE.TEXT_TO_TEXT:
                            decoded_target = EVALUATE.decode(labels, self._tokenizer, self.label_pad_id, skip_special_tokens=False)

                            # decoded_target_tokens = self._tokenizer.convert_ids_to_tokens(labels)
                            print(f'Target: {decoded_target}')
                            # print(f'Target Tokens: {decoded_target_tokens}')
                            print(f'Labels: {labels}')
                        else:
                            print(f'Label: {labels}')
            #     # Specific checks right before the failing operation
            #     max_id = input_ids.max()
            #     min_id = input_ids.min()
            #     print()
            #     print(f"Max ID in input_ids: {max_id}")
            #     print(f"Min ID in input_ids: {min_id}")

            #     # Check if max_id is within the expected range (e.g., tokenizer's vocabulary size)
            #     vocab_size = self._model.config.vocab_size  # Assuming 'model' is your instantiated model
            #     print(f"Vocabulary Size: {vocab_size}")
            #     if max_id >= vocab_size:
            #         print("Error: max_id in input_ids exceeds vocabulary size")
            #         exit()
            #     if min_id < 0:
            #         print("Error: min_id in input_ids is negative")
            #         exit()

            # # This is to check the alignment and batching
            # if "attention_mask" in batch:
            #     attention_mask = batch["attention_mask"]
            #     if attention_mask.shape != input_ids.shape:
            #         print("Mismatch in shape between attention_mask and input_ids")
            #         exit()

    def print_details(self):

        print('\nModel Details >')
        if 'pret_path' in self.__dict__:
            print(f' - Pretrained Path: {self.pret_path}')
        print(f' - Arch: {self._config.model.architecture}')
        print(f' - Uuid4: {self.uuid4}')
        print(f' - Name: {self.name}')
        print(f' - Class: {type(self._model).__name__}')
        base_model = self.get_base_model(self._model)
        print(f' - Base Model Class: {type(base_model).__name__}')
        print(f' - Vocabulary size: {self._tokenizer.vocab_size}')
        print(f' - Embedding dimension: {self.embedding_dim}')
        print(f' - Max sequence length: {self.max_length}')
        print(f' - DataCollator max length: {self._config.data_collator.max_length}')
        # 
        common.p(f' - Parameters:')
        common.p(f' - - All: {self.param_size:,}')
        common.p(f' - - Trainable: [red]{self.trainable_param_size:,} ({self.trainable_param_size_ratio:.4f}%)[/red]')

        common.p('\n - Model Config: ', self._model.config)
        # common.dict_diff(self._model.config.__dict__, base_model.config.__dict__)

        print('\nRun Config >')
        common.p(self._config)

        print('\nEnvironment Details >')
        print(f' - Available Device: {self.device}')
        print(f' - CUDA_VISIBLE_DEVICES: {os.getenv("CUDA_VISIBLE_DEVICES")}')
        print(f' - MASTER_PORT: {os.getenv("MASTER_PORT")}')
        print(f' - Place Model on Device: {self.trainer.place_model_on_device}') # **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set to `False` if model parallel or deepspeed is used, or if the default `TrainingArguments.place_model_on_device` is overridden to return `False` .
        print(f' - Data Parralelism: {self.trainer.args.parallel_mode}')
        print(f' - ZeRO: {self.trainer.is_deepspeed_enabled}')
        print(f' - FSDP: {self.trainer.is_fsdp_enabled}')
        print(f' - FSDP_XLA: {self.trainer.is_fsdp_xla_enabled}')
        print(f' - Model Parralelism: {self.trainer.is_model_parallel}')

        print(f' - RANK: {os.getenv("RANK")}')
        print(f' - LOCAL_RANK: {os.getenv("LOCAL_RANK")}')
        print(f' - WORLD_SIZE: {os.getenv("WORLD_SIZE")}')
        print(f' - LOCAL_WORLD_SIZE: {os.getenv("LOCAL_WORLD_SIZE")}')
        print(f' - MASTER_ADDR: {os.getenv("MASTER_ADDR")}')
        print(f' - MASTER_PORT: {os.getenv("MASTER_PORT")}')

        print(f' - NCCL_DEBUG: {os.getenv("NCCL_DEBUG")}')
        print(f' - TORCH_CPP_LOG_LEVEL: {os.getenv("TORCH_CPP_LOG_LEVEL")}')
        print(f' - TORCH_DISTRIBUTED_DEBUG: {os.getenv("TORCH_DISTRIBUTED_DEBUG")}')
        print(f' - CUDA_HOME: {os.getenv("CUDA_HOME")}')
        print(f' - NCCL_CUDA_PATH: {os.getenv("NCCL_CUDA_PATH")}')
        print(f' - LD_LIBRARY_PATH: {os.getenv("LD_LIBRARY_PATH")}')
        
        print(f' - Distributed State: {self.trainer.args.distributed_state}')
        
        print('\nBatch Details >')
        self.print_batch_examples(DsSplitSE.TRAIN)
        self.print_batch_examples(DsSplitSE.VALIDATION, 1)
        self.print_batch_examples(DsSplitSE.TEST, 1)

        common.p(f'\n[green][bold]Model is ready for {self._config.mode}![/bold][/green]\n')
        # exit()

    def _print_input_output_details(self, base_model):
        print("\n=== Input Projection Layers ===")
        # Shared embeddings (vocab_size x hidden_size)
        print("Shared Embedding Weight:", base_model.shared.weight.shape)
        
        # Encoder layers
        if hasattr(base_model.encoder, 'block'):
            first_encoder_block = base_model.encoder.block[0]
            # T5LayerSelfAttention structure
            self_attention_layer = first_encoder_block.layer[0]
            # print("\nEncoder First Self-Attention Layer Components:")
            # print(dir(self_attention_layer))  # Debug: see available attributes
            
            # Correct path to layer norm in T5
            if hasattr(self_attention_layer, 'layer_norm'):
                print("Encoder Input Norm:", self_attention_layer.layer_norm.weight.shape)
            
            # Self-attention projections
            if hasattr(self_attention_layer.SelfAttention, 'q'):
                print("Query Projection:", self_attention_layer.SelfAttention.q.weight.shape)
        
        # Output layers
        print("\n=== Output Projection Layers ===")
        if hasattr(base_model, 'lm_head'):
            print("Output Head:", base_model.lm_head.weight.shape)
        else:
            print("Output uses shared embeddings:", base_model.shared.weight.shape)

    @staticmethod
    def get_base_model(model):
        if hasattr(model, 'get_base_model'):
            base_model = model.get_base_model()
        elif hasattr(model, 'base_model'):
            base_model = model.base_model
        else:
            base_model = model
        return base_model

    @staticmethod
    def get_last_checkpoint(path):
        # List all dirs from the saved models path
        dirs = os.listdir(path)
        # Filter for directories that start with self.checkpoint_pref
        checkpoints = [ int(dir.split('-')[-1]) for dir in dirs if dir.startswith(MODEL.checkpoint_pref)]
        # return the max checkpoint
        return max(checkpoints) if checkpoints else None
    
    @staticmethod
    def get_last_checkpoint_path(path, last_checkpoint = None):
        # from transformers.trainer_utils import get_last_checkpoint as get_last_checkpoint_hf
        # last_checkpoint_hf = get_last_checkpoint_hf(path) # returns path
        last_checkpoint = last_checkpoint if last_checkpoint else MODEL.get_last_checkpoint(path) # returns int
        return os.path.join(path, f'{MODEL.checkpoint_pref}{last_checkpoint}') if last_checkpoint else path
    
    @staticmethod
    def get_last_checkpoint_path_by_uuid4(source_model_uuid4):
        model_path = MODEL.find_path_by_uuid4(source_model_uuid4)
        last_checkpoint_path = MODEL.get_last_checkpoint_path(model_path)
        return last_checkpoint_path
    
    @staticmethod
    def store_path_by_uuid4_in_envs(uuid4, path):
        os.environ[f'MODEL_PATH_{uuid4}'] = path
        common.p(f'\n[green]Model path is stored in envvar MODEL_PATH_{uuid4}:[/green]\n {path}')
    
    @staticmethod
    def get_path_by_uuid4_from_envs(uuid4):
        return os.environ.get(f'MODEL_PATH_{uuid4}', None)
    
    @staticmethod
    def find_path_by_uuid4(uuid4, root_path = None):
        path = MODEL.get_path_by_uuid4_from_envs(uuid4)
        if not path:
            root_path = root_path if root_path else MODEL.stor_path
            paths = common.find_dir_path(root_dir=root_path, dir_prefix=str(uuid4))
            if 1 < len(paths): raise Exception('By this UUID4 multiple model directories are found!')
            if not len(paths): raise Exception('By this UUID4 no model directory is found!')
            path = paths[0]
            MODEL.store_path_by_uuid4_in_envs(uuid4, path)
        return path

    @staticmethod
    def get_uuid_path_dict(root_path = None):
        """
        Walk once through root_dir and build {uuid4: full_path} for all subdirs
        Assumes each directory name starts with or equals a UUID4 string.
        """
        uuid_path_dict = {}
        root_path = root_path if root_path else MODEL.stor_path
        from tqdm import tqdm
        for dirpath, dirnames, _ in tqdm(os.walk(root_path), desc="Walking through directories"):
            for dirname in dirnames:
                key = dirname.split("_")[0]   # if format is like "prefix_uuid"
                uuid_path_dict[key] = os.path.join(dirpath, dirname)
        return uuid_path_dict