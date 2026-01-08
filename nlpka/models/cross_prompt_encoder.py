import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

log = False

# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Based on https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/modules/common/prompt_encoder.py
# with some refactor
import warnings

import os
import sys
import peft
import copy
import enum
import torch
import random
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Union, Optional, List, Any
from dataclasses import dataclass, field
from transformers import PreTrainedModel
from peft.tuners import MultitaskPromptEmbedding
from peft.tuners.tuners_utils import BaseTuner
from accelerate.utils import get_balanced_memory, named_module_tensors
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from peft.utils import _prepare_prompt_learning_config, _get_batch_size, shift_tokens_right, CONFIG_NAME
from peft.utils.constants import PEFT_TYPE_TO_PREFIX_MAPPING
from peft.utils.save_and_load import (
    _find_mismatched_keys, 
    _insert_adapter_name_into_state_dict,
    has_valid_embedding_base_layer,
    get_embedding_layer_name,
    load_peft_weights,
)
from peft.utils.other import (
    check_file_exists_on_hf_hub,
    EMBEDDING_LAYER_NAMES,
)
from peft import (
    PeftConfig,
    PeftModel,
    PeftType, 
    TaskType, 
    PromptEncoderConfig, 
    PromptEncoderReparameterizationType,
    PromptEmbedding,
    PromptEncoder,
    PrefixEncoder,
    CPTEmbedding,
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    PeftModelForSeq2SeqLM,
)

def is_cross_prompt_encoder(config):
    return config.peft_type == PeftType.P_TUNING and hasattr(config, "encoder_ratio")
               
def get_cross_prompt_encoder(base_model, peft_config_vars):
    model_config = BaseTuner.get_model_config(base_model)
    # common.p('model_config', model_config)
    peft_config = CrossPromptEncoderConfig(**vars(peft_config_vars))
    # common.p('peft_config', peft_config)
    peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    PeftModel._setup_prompt_encoder = _setup_prompt_encoder
    PeftModel.from_pretrained = xpe_from_pretrained
    PeftModelSubClass = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type]
    common.p('PeftModelSubClass: ', PeftModelSubClass)
    if PeftModelSubClass == PeftModelForSeq2SeqLM:
        PeftModelSubClass.forward = seq2seq_forward
    adapter_name = 'default'
    peft_model = PeftModelSubClass(
                    base_model,
                    peft_config,
                    adapter_name=adapter_name,
                    autocast_adapter_dtype=True,
                    low_cpu_mem_usage=False,
                )
    prompt_encoder = peft_model.prompt_encoder[adapter_name]
    # print('prompt_encoder', prompt_encoder)
    # exit()
    prompt_encoder.set_grad_requirements()
    prompt_encoder.print_all_layers()
    # print('peft_model', peft_model, flush=True)
    PeftModel.get_prompt = get_prompt
    peft.utils.get_peft_model_state_dict = xpe_get_peft_model_state_dict
    peft.utils.set_peft_model_state_dict = xpe_set_peft_model_state_dict
    common.monkey_patch_globally("get_peft_model_state_dict", xpe_get_peft_model_state_dict)
    common.monkey_patch_globally("set_peft_model_state_dict", xpe_set_peft_model_state_dict)
    # print('peft_model', peft_model, flush=True)
    if peft_config.encoder_init_state_dict_path:
        peft_model = maybe_load_pretrained_classifier_state(peft_model, peft_config.encoder_init_state_dict_path)
    return peft_model

def _setup_prompt_encoder(self, adapter_name: str):
    config = self.peft_config[adapter_name]
    if not hasattr(self, "prompt_encoder"):
        self.prompt_encoder = torch.nn.ModuleDict({})
        self.prompt_tokens = {}
    transformer_backbone = None
    for name, module in self.base_model.named_children():
        for param in module.parameters():
            param.requires_grad = False
        if isinstance(module, PreTrainedModel):
            # Make sure to freeze Tranformers model
            if transformer_backbone is None:
                transformer_backbone = module
                self.transformer_backbone_name = name
    if transformer_backbone is None:
        transformer_backbone = self.base_model

    if config.num_transformer_submodules is None:
        config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1
    
    # Custom code start
    if config.peft_type == PeftType.P_TUNING:
        config.num_transformer_submodules = 1
    # Custom code end

    # determine the word embeddings
    word_embeddings = None
    try:
        # First try to find the word embeddings based on the module name, this should work for models like Bert,
        # Roberta, Deberta, etc.
        word_embeddings = self.base_model.get_submodule("embeddings.word_embeddings")
    except AttributeError:
        pass

    if word_embeddings is None:
        # Word embeddings could not be determined. Next try to guess them by checking which parameter has the size
        # of the vocab.
        for named_param, value in list(transformer_backbone.named_parameters()):
            # for ZeRO-3, the tensor is sharded across accelerators and deepspeed modifies it to a tensor with shape
            # [0] the actual unsharded shape is stored in "ds_shape" attribute special handling is needed in case
            # the model is initialized in deepspeed.zero.Init() context or HfDeepSpeedConfig has been called before
            # For reference refer to issue: https://github.com/huggingface/peft/issues/996
            deepspeed_distributed_tensor_shape = getattr(value, "ds_shape", None)

            if value.shape[0] == self.base_model.config.vocab_size or (
                deepspeed_distributed_tensor_shape is not None
                and deepspeed_distributed_tensor_shape[0] == self.base_model.config.vocab_size
            ):
                word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

    self.word_embeddings = word_embeddings

    if config.peft_type == PeftType.PROMPT_TUNING:
        prompt_encoder = PromptEmbedding(config, self.word_embeddings)
    elif config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
        prompt_encoder = MultitaskPromptEmbedding(config, self.word_embeddings)
    elif config.peft_type == PeftType.P_TUNING:
        if 'encoder_ratio' in vars(config):
            prompt_encoder = CrossPromptEncoder(config)
        else:
            prompt_encoder = PromptEncoder(config)
    elif config.peft_type == PeftType.PREFIX_TUNING:
        # prefix tuning now uses Cache but that won't work with gradient checkpointing
        if any(getattr(module, "gradient_checkpointing", False) for module in self.get_base_model().modules()):
            raise ValueError("Prefix tuning does not work with gradient checkpointing.")
        prompt_encoder = PrefixEncoder(config)
    elif config.peft_type == PeftType.CPT:
        prompt_encoder = CPTEmbedding(config, self.word_embeddings)
    else:
        raise ValueError("Not supported")

    prompt_encoder = prompt_encoder.to(self.device)
    self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
    self.prompt_tokens[adapter_name] = torch.arange(
        config.num_virtual_tokens * config.num_transformer_submodules
    ).long()

@classmethod
def xpe_from_pretrained(
    cls,
    model: torch.nn.Module,
    model_id: Union[str, os.PathLike],
    adapter_name: str = "default",
    is_trainable: bool = False,
    config: Optional[PeftConfig] = None,
    autocast_adapter_dtype: bool = True,
    ephemeral_gpu_offload: bool = False,
    low_cpu_mem_usage: bool = False,
    **kwargs: Any,
) -> PeftModel:
    r"""
    Instantiate a PEFT model from a pretrained model and loaded PEFT weights.

    Note that the passed `model` may be modified inplace.

    Args:
        model ([`torch.nn.Module`]):
            The model to be adapted. For 🤗 Transformers models, the model should be initialized with the
            [`~transformers.PreTrainedModel.from_pretrained`].
        model_id (`str` or `os.PathLike`):
            The name of the PEFT configuration to use. Can be either:
                - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                    Hub.
                - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                    method (`./my_peft_config_directory/`).
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter to be loaded. This is useful for loading multiple adapters.
        is_trainable (`bool`, *optional*, defaults to `False`):
            Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
            used for inference.
        config ([`~peft.PeftConfig`], *optional*):
            The configuration object to use instead of an automatically loaded configuration. This configuration
            object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
            loaded before calling `from_pretrained`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Only relevant for specific adapter types.
        ephemeral_gpu_offload (`bool`, *optional*):
            Whether to use ephemeral GPU offloading for partially loaded modules. Defaults to `False`. This is
            useful when parts of the model and/or components (such as adapters) are kept in CPU memory until they
            are needed. Rather than perform expensive operations on small data, the data is transferred to the GPU
            on-demand, the operation(s) performed, and the results moved back to CPU memory. This brings a slight
            momentary VRAM overhead but gives orders of magnitude speedup in certain cases.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device before loading the saved weights. Useful to speed up the
            process.
        torch_device (`str`, *optional*, defaults to None):
            The device to load the adapter on. If `None`, the device will be inferred.
        kwargs: (`optional`):
            Additional keyword arguments passed along to the specific PEFT configuration class.
    """
    from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING

    # load the config
    # print('model_id', model_id)
    # exit()
    if config is None:
        # config = PEFT_TYPE_TO_CONFIG_MAPPING[
        #     PeftConfig._get_peft_type(
        #         model_id,
        #         subfolder=kwargs.get("subfolder", None),
        #         revision=kwargs.get("revision", None),
        #         cache_dir=kwargs.get("cache_dir", None),
        #         use_auth_token=kwargs.get("use_auth_token", None),
        #         token=kwargs.get("token", None),
        #     )
        # ].from_pretrained(model_id, **kwargs)
        config = CrossPromptEncoderConfig.from_pretrained(model_id, **kwargs)
    elif isinstance(config, PeftConfig):
        config.inference_mode = not is_trainable
    else:
        raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

    common.p('\nxpe_from_pretrained.config: ', config)

    # Runtime configuration, if supported
    if hasattr(config, "runtime_config"):
        config.runtime_config.ephemeral_gpu_offload = ephemeral_gpu_offload
    else:
        if ephemeral_gpu_offload:
            warnings.warn("Ephemeral GPU offloading is not supported for this model. Ignoring.")

    if hasattr(model, "hf_device_map"):
        weight_map = dict(named_module_tensors(model, recurse=True))

        # recreate the offload_index for disk-offloaded modules: we need to know the location in storage of each weight
        # before the offload hook is removed from the model
        disk_modules = set()
        index = None
        for name, module in model.named_modules():
            if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "original_devices"):
                if hasattr(module._hf_hook.weights_map, "dataset"):
                    index = module._hf_hook.weights_map.dataset.index
                for key in module._hf_hook.original_devices.keys():
                    if module._hf_hook.original_devices[key] == torch.device("meta"):
                        disk_modules.add(str(name) + "." + str(key))

        if disk_modules and not kwargs.get("use_safetensors", True):
            raise ValueError("Disk offloading currently only supported for safetensors")

        if index:
            offload_index = {
                p: {
                    "safetensors_file": index[p]["safetensors_file"],
                    "weight_name": p,
                    "dtype": str(weight_map[p].dtype).replace("torch.", ""),
                }
                for p in weight_map.keys()
                if p in disk_modules
            }
            kwargs["offload_index"] = offload_index

    if (getattr(model, "hf_device_map", None) is not None) and len(
        set(model.hf_device_map.values()).intersection({"cpu", "disk"})
    ) > 0:
        remove_hook_from_submodules(model)

    if config.is_prompt_learning and is_trainable:
        raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
    else:
        config.inference_mode = not is_trainable
    # if isinstance(getattr(model, "base_model", None), XLoraModel):
    #     if not isinstance(config, XLoraConfig):
    #         raise TypeError(f"Expected 'XLoraConfig', got '{type(config)}' instead.")
    #     if "adapters" in kwargs:
    #         config.adapters = kwargs["adapters"]
    #     else:
    #         # If the path is on HF hub, then we get the adapter names to create a subfolders list which tells
    #         # `load_adapter` where the adapters are.
    #         if not os.path.exists(model_id):
    #             s = HfFileSystem()

    #             # The names of the adapters which must be in folders
    #             adapter_names = [
    #                 file["name"][len(model_id) + 1 :] for file in s.ls(model_id) if file["type"] == "directory"
    #             ]
    #             # Prepare a dict of adapter paths, which really just point to the hf id; we will use the subfolders
    #             adapter_paths = {}
    #             for adapter_name in adapter_names:
    #                 adapter_paths[adapter_name] = os.path.join(model_id, model_id)
    #             config.adapters = adapter_paths
    #             config._subfolders = adapter_names
    #         else:
    #             if "adapters" not in kwargs:
    #                 raise ValueError("If model_id is a local path, then `adapters` must be passed in kwargs.")

    print('\nxpe_from_pretrained.model: ', {
        # 'cls': cls,
        'cls_2': MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type],
        # 'model': model,
        # 'config': config,
        'adapter_name': adapter_name,
        'autocast_adapter_dtype': autocast_adapter_dtype,
        'low_cpu_mem_usage': low_cpu_mem_usage,
    })

    if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
        model = cls(
            model,
            config,
            adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    else:
        model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](
            model,
            config,
            adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    common.p('\nxpe_from_pretrained.model: ', model)


    load_result = model.load_adapter(
        model_id,
        adapter_name,
        is_trainable=is_trainable,
        autocast_adapter_dtype=autocast_adapter_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        **kwargs,
    )

    # The adapter weights loaded perfectly if load_result == {}
    common.p('\nxpe_from_pretrained.load_result: ', load_result)

    # 1. Remove VB-LoRA vector bank, since it's a shared parameter set via the VBLoRAModel
    # 2. Remove the prompt encoder, as it does not need to be part of the checkpoint
    missing_keys = [
        k for k in load_result.missing_keys if "vblora_vector_bank" not in k and "prompt_encoder" not in k
    ]
    if missing_keys:
        # Let's warn here since (in contrast to load_adapter) we don't return the load result, so it could be quite
        # difficult for users to even notice that something might have gone wrong here. As we filter out non PEFT
        # keys from the missing keys, this gives no false positives.
        warnings.warn(f"Found missing adapter keys while loading the checkpoint: {missing_keys}")

    return model

def xpe_get_peft_model_state_dict(
    model, state_dict=None, adapter_name="default", unwrap_compiled=False, save_embedding_layers="auto"
):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
            the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the passed model will be used.
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter whose state dict should be returned.
        unwrap_compiled (`bool`, *optional*, defaults to `False`):
            Whether to unwrap the model if torch.compile was used.
        save_embedding_layers (`Union[bool, str]`, , *optional*, defaults to `auto`):
            If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common embedding
            layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available. Based on it
            sets the boolean flag. This only works for 🤗 transformers models.
    """
    if log: common.p('\nxpe_get_peft_model_state_dict.state_dict: ', state_dict)

    if unwrap_compiled:
        model = getattr(model, "_orig_mod", model)

    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()

    to_return = {}
    # TUNER SPECIFIC CODE
    # ...removed

    # MODULES TO SAVE
    if log: common.p('\nxpe_get_peft_model_state_dict.model.modules_to_save: ', getattr(model, "modules_to_save", None))
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                if log: common.p('key: ', key.replace("modules_to_save.", ""))
                to_return[key.replace("modules_to_save.", "")] = value

    # DEAL WITH EMBEDDINGS
    # check the common embedding layers in `target_modules` to reset `save_embedding_layers` if necessary
    is_embedding_in_target_modules = False
    if (
        save_embedding_layers == "auto"
        and hasattr(config, "target_modules")
        and any(k in config.target_modules for k in EMBEDDING_LAYER_NAMES)
    ):
        warnings.warn("Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.")
        save_embedding_layers = is_embedding_in_target_modules = True
    elif save_embedding_layers == "auto":
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        model_id = getattr(config, "base_model_name_or_path", None)

        # For some models e.g. diffusers the text config file is stored in a subfolder
        # we need to make sure we can download that config.
        has_base_config = False
        # print('\nget_peft_model_state_dict.state_dict', state_dict.keys())
        # ensure that this check is not performed in HF offline mode, see #1452
        if model_id is not None:
            local_config_exists = os.path.exists(os.path.join(model_id, "config.json"))
            exists = local_config_exists or check_file_exists_on_hf_hub(model_id, "config.json")
            if exists is None:
                # check failed, could not determine if it exists or not
                warnings.warn(
                    f"Could not find a config file in {model_id} - will assume that the vocabulary was not modified."
                )
                has_base_config = False
            else:
                has_base_config = exists

        # check if the vocab size of the base model is different from the vocab size of the finetuned model
        if (
            vocab_size
            and model_id
            and has_base_config
            and (vocab_size != model.config.__class__.from_pretrained(model_id).vocab_size)
        ):
            warnings.warn(
                "Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning."
            )
            save_embedding_layers = True
        else:
            save_embedding_layers = False

    if save_embedding_layers and hasattr(model, "get_input_embeddings"):
        for layer in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if not is_embedding_in_target_modules or has_valid_embedding_base_layer(layer):
                # support from version >= 0.6.2
                embedding_module_name = get_embedding_layer_name(model, layer, is_embedding_in_target_modules)
                if embedding_module_name:
                    to_return.update({k: v for k, v in state_dict.items() if embedding_module_name in k})
    elif save_embedding_layers:
        warnings.warn("Could not identify embedding layer(s) because the model is not a 🤗 transformers model.")

    # REMOVE ADAPTER NAME
    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    if log: common.p('\nxpe_get_peft_model_state_dict.to_return', to_return.keys())
    # common.print_traceback(show_locals = False)
    return to_return

def xpe_set_peft_model_state_dict(
    model,
    peft_model_state_dict,
    adapter_name="default",
    ignore_mismatched_sizes: bool = False,
    low_cpu_mem_usage: bool = False,
):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]):
            The Peft model.
        peft_model_state_dict (`dict`):
            The state dict of the Peft model.
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter whose state dict should be set.
        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
            Whether to ignore mismatched in the state dict.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            This argument must be `True` if the `model` was loaded with adapter weights on the meta device, e.g. after
            calling `inject_adapter_in_model` with `low_cpu_mem_usage=True`. Otherwise, leave it as `False`.

    """
    if log: common.p('\nxpe_set_peft_model_state_dict.peft_model_state_dict', peft_model_state_dict.keys())
    config = model.peft_config[adapter_name]
    state_dict = {}
    if log: common.p('\nxpe_set_peft_model_state_dict.model.modules_to_save', getattr(model, "modules_to_save", None))
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in peft_model_state_dict.items():
            if log: common.p('\nold key: ', key)
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if log: common.p('module_name: ', module_name, '| module_name in key: ', module_name in key)
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            if log: common.p('new key: ', key)
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict
    
    if log: common.p('\nxpe_set_peft_model_state_dict.state_dict', state_dict.keys())
    # exit()

    if config.peft_type in PEFT_TYPE_TO_PREFIX_MAPPING:
        peft_model_state_dict = {}
        parameter_prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
        if config.peft_type == PeftType.VBLORA and config.save_only_topk_weights:
            num_vectors, _ = model.vblora_vector_bank[adapter_name].shape
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                # in save_only_topk_weights mode, only topk_indices and topk_weights are saved
                # note that topk_indices and topk_weights serve as an efficient representation of the logits
                # so we need to recover the logits from the topk_indices and topk_weights
                if "_topk_indices" in k:
                    v = state_dict[k].to(torch.long)
                    original_key = k.replace("_topk_indices", "")
                    # find the corresponding topk_weights from the state_dict
                    topk_weights = state_dict[k.replace("_topk_indices", "_topk_weights")]
                    # as we only save the first k-1 topk_weights, here we recover the last one
                    topk_weights = torch.cat([topk_weights, 1 - topk_weights.sum(-1, keepdim=True)], dim=-1)
                    # convert the weights to logits
                    topk_logits = torch.log(topk_weights)
                    matrix = (
                        torch.zeros([*(topk_logits.shape[:-1]), num_vectors])
                        .fill_(float("-inf"))
                        .to(topk_logits.device)
                        .scatter(-1, v, topk_logits)
                    )
                    # add logits to the state_dict
                    state_dict[original_key] = matrix
                    # delete the topk_indices and topk_weights from the state_dict
                    del state_dict[k]
                    del state_dict[k.replace("_topk_indices", "_topk_weights")]

        peft_model_state_dict = _insert_adapter_name_into_state_dict(
            state_dict, adapter_name=adapter_name, parameter_prefix=parameter_prefix
        )

        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
        elif config.peft_type == PeftType.VERA:
            if config.save_projection and "base_model.vera_A" not in peft_model_state_dict:
                raise ValueError(
                    "Specified to load vera_A and vera_B from state dictionary however they were not present!"
                )
            elif not config.save_projection and "base_model.vera_A" in peft_model_state_dict:
                warnings.warn(
                    "Specified to not load vera_A and vera_B from state dictionary however they are present in state"
                    " dictionary! Consider using them to ensure checkpoint loading is correct on all platforms using"
                    " `peft_config.save_projection = True`"
                )
            elif not config.save_projection:  # and no vera_A in state dictionary
                warnings.warn(
                    "Specified to not load vera_A and vera_B from state dictionary. This means we will be relying on"
                    " PRNG initialisation to restore these projections using `config.projection_prng_key`, which may"
                    " not be accurate on all system configurations."
                )
        elif config.peft_type == PeftType.LORA:
            # Here we take care of a refactor of DoRA which changed lora_magnitude_vector from a ParameterDict to a
            # ModuleDict with a DoraLayer instance. The old parameter is now the "weight" attribute of that layer.
            old_dora_suffix = f"lora_magnitude_vector.{adapter_name}"

            def renamed_dora_weights(k):
                if k.endswith(old_dora_suffix):
                    k = k + ".weight"
                return k

            peft_model_state_dict = {renamed_dora_weights(k): v for k, v in peft_model_state_dict.items()}
    
    elif config.is_prompt_learning or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    
    elif config.peft_type == PeftType.XLORA:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    peft_model_state_dict, mismatched_keys = _find_mismatched_keys(
        model, peft_model_state_dict, ignore_mismatched_sizes=ignore_mismatched_sizes
    )
    if log: common.p('\nxpe_set_peft_model_state_dict.mismatched_keys', mismatched_keys)
    if log: common.p('\nxpe_set_peft_model_state_dict.low_cpu_mem_usage', low_cpu_mem_usage)
    if low_cpu_mem_usage:
        load_result = model.load_state_dict(peft_model_state_dict, strict=False, assign=True)
        # ensure that the correct device is set
        for module in model.modules():
            if hasattr(module, "_move_adapter_to_device_of_base_layer"):
                module._move_adapter_to_device_of_base_layer(adapter_name)
    else:
        load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    
    # common.p(f'xpe_set_peft_model_state_dict.load_result.missing_keys: {load_result.missing_keys}')      # []
    # common.p(f'xpe_set_peft_model_state_dict.load_result.unexpected_keys: {load_result.unexpected_keys}')   # []

    # if config.is_prompt_learning:
    #     model.prompt_encoder[adapter_name].embedding.load_state_dict(
    #         {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
    #     )

    peft_model_state_dict = {k.replace(f"prompt_encoder.", ""): v for k, v in peft_model_state_dict.items()}
    if log: common.p('\nxpe_set_peft_model_state_dict.peft_model_state_dict', peft_model_state_dict.keys())
    # print('\nxpe_set_peft_model_state_dict.model.prompt_encoder[{adapter_name}]', model.prompt_encoder[adapter_name])
    load_pe_result = model.prompt_encoder[adapter_name].load_state_dict(peft_model_state_dict, strict=False)
    if log: common.p(f'load_pe_result.missing_keys: {load_pe_result.missing_keys}') # []
    if log: common.p(f'load_pe_result.unexpected_keys: {load_pe_result.unexpected_keys}') # []
    for name, param in model.prompt_encoder[adapter_name].named_parameters():
        if param.requires_grad:
            if log: common.p(name, param.shape)

    if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
        model.prompt_encoder[adapter_name].load_state_dict(peft_model_state_dict, strict=False)

    if mismatched_keys:
        # see https://github.com/huggingface/transformers/blob/09f9f566de83eef1f13ee83b5a1bbeebde5c80c1/src/transformers/modeling_utils.py#L4039
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        msg = (
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint "
            f"and are being ignored because you passed `ignore_mismatched_sizes=True`: {mismatched_warning}."
        )
        warnings.warn(msg)
    if log: common.p('\nxpe_set_peft_model_state_dict.load_result', load_result)
    # common.print_traceback(show_locals = False)
    return load_result

def get_prompt(self, batch_size: int, task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Returns the virtual prompts to use for Peft. Only applicable when using a prompt learning method.
    """
    prompt_encoder = self.prompt_encoder[self.active_adapter]
    # common.p('prompt_encoder', type(prompt_encoder).__name__)
    # common.p('self.prompt_tokens[self.active_adapter]', self.prompt_tokens[self.active_adapter])
    # common.p('prompt_encoder.embedding', prompt_encoder.embedding)
    prompt_tokens = (
        self.prompt_tokens[self.active_adapter]
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(prompt_encoder.get_device())
    )    
    # common.p(f"get_prompt.prompt_tokens.shape: {prompt_tokens.shape}")
    # exit()
    return prompt_encoder(prompt_tokens, task_ids)

def seq2seq_forward(
    self,
    input_ids=None,
    attention_mask=None,
    inputs_embeds=None,
    decoder_input_ids=None,
    decoder_attention_mask=None,
    decoder_inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    task_ids=None,
    **kwargs,
    ):
    # print("\n=== PEFT S2S FORWARD DEBUG ===")
    # print(f"Batch keys: {list(locals().keys())}")
    # print(f"input_ids: {True if input_ids is not None else False}")
    # print(f"inputs_embeds: {True if inputs_embeds is not None else False}")
    # print(f"labels: {True if labels is not None else False}")
    # for label in labels:
    #     print(label)

    peft_config = self.active_peft_config
    if not peft_config.is_prompt_learning:
        if peft_config.peft_type == PeftType.POLY:
            kwargs["task_ids"] = task_ids

        with self._enable_peft_forward_hooks(**kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

    batch_size = _get_batch_size(input_ids, inputs_embeds)
    if decoder_attention_mask is not None:
        # concat prompt attention mask
        prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
            decoder_attention_mask.device
        )
        if peft_config.peft_type not in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
            decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)

    if kwargs.get("position_ids", None) is not None:
        warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
        kwargs["position_ids"] = None
    if kwargs.get("token_type_ids", None) is not None:
        warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
        kwargs["token_type_ids"] = None
    kwargs.update(
        {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
    )

    if peft_config.peft_type == PeftType.PREFIX_TUNING:
        # overwrite past_kv in kwargs
        kwargs["past_key_values"] = self.get_prompt(batch_size)
        return self.base_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            **kwargs,
        )
    elif peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                attention_mask.device
            )
            kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
        # common.p(f"prompts.shape: {prompts.shape}")
        # common.p(f"peft_config.num_virtual_tokens: {peft_config.num_virtual_tokens}")
        # common.p(f"prompts: {prompts[:, : peft_config.num_virtual_tokens]}")
        # exit()

        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)


        res = self.base_model(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            **kwargs,
        )
        # common.print_traceback(show_locals = False)
        return res
    else:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if decoder_inputs_embeds is None and decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                attention_mask.device
            )
            kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        # concat prompt labels
        if labels is not None:
            if peft_config.num_transformer_submodules == 1:
                kwargs["labels"] = labels
            elif peft_config.num_transformer_submodules == 2:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
        prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)
        if peft_config.num_transformer_submodules == 1:
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
        elif peft_config.num_transformer_submodules == 2:
            decoder_inputs_embeds = torch.cat(
                (prompts[:, peft_config.num_virtual_tokens :], decoder_inputs_embeds), dim=1
            )
            return self.base_model(
                inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **kwargs
            )
        
class CrossPromptEncoderReparameterizationType(str, enum.Enum):
    MLP = "MLP"
    LSTM = "LSTM"
    ATTN = "ATTN"
    NONE = "NONE"

@dataclass
class CrossPromptEncoderConfig(PromptEncoderConfig):
    """
    This is the configuration class to store the configuration of a [`CrossPromptEncoder`].

    Args:
        encoder_embedding_init_type (`str`): The type of initialization to use for the embedding.
        encoder_init_state_dict_path (`str`): The path to pretraine encoder state for initialization shared embeddings and encoder heads.
        encoder_embedding_freeze (`bool`): The indicator of frozen or trainable embedding.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from CrossPromptEncoder layers to be set as trainable and saved in the final checkpoint.
        encoder_embedding_normalize (`str`): The type of normalization to use for the embedding (None,unit, clip).
        encoder_embedding_normalize_max_norm (`float`): The maximum norm for the embedding.
        encoder_input_size (`int`): The input size for the encoder.
        encoder_num_heads (`int`): The number of attention heads in the encoder.
    """
    encoder_embedding_init_type: str = field(
        default="hf_default",
        metadata={"help": "The type of initialization to use for the embedding (xavier_uniform, xavier_normal, hf_default)"},
    )
    encoder_init_state_dict_path: str = field(
        default=None,
        metadata={"help": "The path to pretrained encoder"},
    )
    encoder_freeze: bool = field(
        default=True,
        metadata={"help": "The indicator of frozen or trainable encoder"},
    )
    encoder_embedding_freeze: bool = field(
        default=True,
        metadata={"help": "The indicator of frozen or trainable embedding"},
    )
    encoder_ratio: float = field(
        default=0.25,
        metadata={"help": "The ratio of encoded vs standard input embeddings"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from CrossPromptEncoder layers to be set as trainable and saved in the final checkpoint."
        },
    )
    encoder_embedding_normalize: str = field(
        default="unit",
        metadata={"help": "The type of normalization to use for the embedding (None, unit, clip)"},
    )
    encoder_embedding_normalize_max_norm: float = field(
        default=1.0,
        metadata={"help": "The maximum norm for the embedding"},
    )
    encoder_input_size: Optional[int] = field(
        default=None,
        metadata={"help": "The input size for the encoder"},
    )
    encoder_num_heads: int = field(
        default=8,
        metadata={"help": "The number of attention heads in the encoder"},
    )

    def __post_init__(self):
        super().__post_init__()
        
        self.peft_type = PeftType.P_TUNING
        if self.modules_to_save is None:
            self.modules_to_save = []
            
            # Embeddings
            if self.encoder_ratio < 1:
                self.modules_to_save.append('embedding')
            if self.encoder_ratio > 0:
                self.modules_to_save.append('xpe_embedding')

            # Encoder Head
            if self.encoder_ratio > 0:
                self.modules_to_save.append('xpe_head')

    @classmethod
    def from_peft_type(cls, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a set of kwargs.

        The appropriate configuration type is determined by the `peft_type` argument. If `peft_type` is not provided,
        the calling class type is instantiated.

        Args:
            kwargs (configuration keyword arguments):
                Keyword arguments passed along to the configuration initialization.
        """
        return CrossPromptEncoderConfig(**kwargs)

# I fixed the saving of the entire prompt encoder and im using it for the best model saving and loading
class CrossPromptEncoder(torch.nn.Module):
    """
    The CrossPromptEncoder is a neural network module designed to generate virtual token embeddings, supporting various embedding strategies.

    Args:
        config ([`CrossPromptEncoderConfig`]): The configuration of the cross prompt encoder.

    Example:

    ```py
    >>> from nlpka.models.cross_prompt_encoder import CrossPromptEncoder, CrossPromptEncoderConfig

    >>> config = CrossPromptEncoderConfig(
    ...     peft_type="P_TUNING",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     encoder_num_layers=12,
    ...     encoder_reparameterization_type="MLP",
    ...     encoder_hidden_size=768,
    ...     encoder_embedding_type="FULLY_SHARED",
    ... )

    >>> prompt_encoder = CrossPromptEncoder(config)
    ```

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The Standard Soft Prompt (SPT) embedding layer (Skips XPE)
        - **xpe_embedding** (`torch.nn.Embedding`) -- The Cross Prompt Encoder (XPE) embedding layer (XPE input embeddings).
        - **xpe_head** (`torch.nn.Module`) -- The Cross Prompt Encoder (XPE) head of the prompt encoder if `encoder_reparameterization_type="MLP"`.
        - **token_dim** (`int`) -- The hidden embedding dimension of the base transformer model.
        - **input_size** (`int`) -- The input size of the prompt encoder.
        - **output_size** (`int`) -- The output size of the prompt encoder.
        - **hidden_size** (`int`) -- The hidden size of the prompt encoder.
        - **total_virtual_tokens** (`int`): The total number of virtual tokens of the prompt encoder.
        - **encoder_type** (Union[[`CrossPromptEncoderReparameterizationType`], `str`]): The encoder type of the prompt encoder.

    Input shape: (`batch_size`, `total_virtual_tokens`)

    Output shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config: CrossPromptEncoderConfig):
        super().__init__()
        self.token_dim = config.token_dim
        self.input_size = config.encoder_input_size or self.token_dim
        self.output_size = self.token_dim
        self.num_heads = config.encoder_num_heads
        self.encoder_num_layers = config.encoder_num_layers
        self.encoder_dropout = config.encoder_dropout
        self.hidden_size = config.encoder_hidden_size
        self.encoder_type = config.encoder_reparameterization_type
        self.embedding_init_type = config.encoder_embedding_init_type
        self.encoder_ratio = config.encoder_ratio
        
        # Number of total virtual tokens
        self.num_transformer_submodules = config.num_transformer_submodules
        self.total_virtual_tokens = config.num_virtual_tokens * self.num_transformer_submodules

        common.p(f"[yellow]num_virtual_tokens: {config.num_virtual_tokens}[/yellow]")
        common.p(f"[yellow]num_transformer_submodules: {self.num_transformer_submodules}[/yellow]")
        common.p(f"[yellow]total_virtual_tokens: {self.total_virtual_tokens}[/yellow]")

        # Initialize Embeddings and XPE Head, even if pretrained weights are provided.
        # This ensures that the model is always in a valid state, even if pretrained model is missing some weights.
        self.init_state_dict_path = config.encoder_init_state_dict_path
        
        # freeze
        self.encoder_freeze = config.encoder_freeze
        self.embedding_freeze = config.encoder_embedding_freeze

        # normalization
        self.embedding_normalize = config.encoder_embedding_normalize
        self.embedding_normalize_max_norm = config.encoder_embedding_normalize_max_norm

        # virtual tokens for XPE and SPT (Standard Soft Prompt)
        self.xpe_virtual_tokens = 0
        self.spt_virtual_tokens = 0
        if self.encoder_ratio == 0:
            self.spt_virtual_tokens = self.total_virtual_tokens
        elif self.encoder_ratio == 1:
            self.xpe_virtual_tokens = self.total_virtual_tokens
        elif self.encoder_ratio > 0 and self.encoder_ratio < 1:
            self.xpe_virtual_tokens = max(1, min(self.total_virtual_tokens - 1, int(self.total_virtual_tokens * self.encoder_ratio)))
            self.spt_virtual_tokens = self.total_virtual_tokens - self.xpe_virtual_tokens
        else:
            raise ValueError(f"Unknown encoder ratio: {self.encoder_ratio}")

        # Print virtual tokens
        common.p(f"[yellow]xpe_virtual_tokens: {self.xpe_virtual_tokens}[/yellow]")
        common.p(f"[yellow]spt_virtual_tokens: {self.spt_virtual_tokens}[/yellow]")

        # Initialize embeddings if virtual tokens are present
        if self.spt_virtual_tokens:
            self.embedding = self.init_embeddings(self.spt_virtual_tokens, self.input_size)
            common.p(f"[yellow]embedding: {self.embedding}[/yellow]")
        if self.xpe_virtual_tokens:
            self.xpe_embedding = self.init_embeddings(self.xpe_virtual_tokens, self.input_size)
            common.p(f"[yellow]xpe_embedding: {self.xpe_embedding}[/yellow]")
        
        if self.encoder_ratio > 0:
            
            # Initialize encoder heads
            common.p(f"[yellow]Randomly initialize {self.encoder_type} head:[/yellow]")
            
            if self.encoder_type == CrossPromptEncoderReparameterizationType.NONE:
                self.xpe_head = torch.nn.Identity()
            
            elif self.encoder_type == CrossPromptEncoderReparameterizationType.LSTM:
                lstm = torch.nn.LSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.encoder_num_layers,
                    dropout=self.encoder_dropout,
                    bidirectional=True,
                    batch_first=True,
                )
                mlp = self.gen_mlp_head(
                    input_size=self.hidden_size * 2,
                    hidden_size=self.hidden_size,
                    output_size=self.output_size,
                    num_layers=1,
                    dropout=self.encoder_dropout,
                )
                self.xpe_head = LSTMWrapper(lstm, mlp)

            elif self.encoder_type == CrossPromptEncoderReparameterizationType.MLP:
                self.xpe_head = self.gen_mlp_head(
                    self.input_size, 
                    self.hidden_size, 
                    self.output_size, 
                    self.encoder_num_layers,
                    self.encoder_dropout
                )

            elif self.encoder_type == CrossPromptEncoderReparameterizationType.ATTN:
                attn = LightweightSelfAttentionHead(
                    num_heads=self.num_heads,
                    embed_dim=self.input_size,
                    output_dim=self.input_size,
                    dropout=config.encoder_dropout,
                )
                mlp = self.gen_mlp_head(
                    self.input_size,
                    self.hidden_size,
                    self.output_size,
                    num_layers=1
                )
                self.xpe_head = nn.Sequential(attn, mlp)

            else:
                raise ValueError(f"Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")
            
            common.p(f"[yellow]{self.xpe_head}[/yellow]")
                
        # Load pretrained weights if provided
        if self.init_state_dict_path:
            self.load_pretrained_state()
            # self.print_all_layers()
            # exit()

        # Freeze embeddings or entire encoder if required
        self.set_grad_requirements()
        
        # # Print layers
        # self.print_all_layers()

    def init_embeddings(self, num: int, dim: int):
        embedding = torch.nn.Embedding(num, dim)
        if self.embedding_init_type  == "xavier_uniform":
            init.xavier_uniform_(embedding.weight)
        elif self.embedding_init_type  == "xavier_normal":
            init.xavier_normal_(embedding.weight)
        elif self.embedding_init_type == "hf_default":
            pass
        else:
            raise ValueError(f"Unknown initialization type: {self.embedding_init_type}")
        return embedding

    def print_trainable_layers(self):
        """
        Print all trainable layer names in the model.
        """
        trainable_params_count = 0
        trainable_layers = []
        trainable_data_norms = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_layers.append(name)
                trainable_data_norms.append(param.data.norm().detach().cpu())
                trainable_params_count += param.numel()
        common.p("[red]CrossPromptEncoder - Trainable params:[/red]", f'{trainable_params_count:,}')
        common.p("[red]CrossPromptEncoder - Trainable data norm:[/red]", round(torch.stack(trainable_data_norms).mean().item(), 3))
        common.p("[red]CrossPromptEncoder - Trainable layers:[/red]", trainable_layers)

    def print_all_layers(self):
        """
        Print all layers in the model, grouped by trainable and non-trainable.
        """
        trainable_layers = []
        trainable_data_norms = []
        trainable_params_count = 0

        non_trainable_layers = []
        non_trainable_data_norms = []
        non_trainable_params_count = 0

        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_layers.append(name)
                trainable_data_norms.append(param.data.norm().detach().cpu())
                trainable_params_count += param.numel()
            else:
                non_trainable_layers.append(name)
                non_trainable_data_norms.append(param.data.norm().detach().cpu())
                non_trainable_params_count += param.numel()

        # Print trainable parameters
        common.p("[red]\nXPE Trainable Parameters:[/red]")
        common.p(f"  Count: {trainable_params_count:,}")
        common.p(f"  Avg. Data Norm: {round(torch.stack(trainable_data_norms).mean().item(), 3) if trainable_data_norms else 'N/A'}")
        common.p("  Layers: ")
        for layer in trainable_layers:
            common.p(f"    {layer}")

        # Print non-trainable parameters
        common.p("[blue]\nXPE Frozen Parameters:[/blue]")
        common.p(f"  Count: {non_trainable_params_count:,}")
        common.p(f"  Avg. Data Norm: {round(torch.stack(non_trainable_data_norms).mean().item(), 3) if non_trainable_data_norms else 'N/A'}")
        common.p("  Layers: ")
        for layer in non_trainable_layers:
            common.p(f"    {layer}")


    def load_pretrained_state(self):
        """
        Load pretrained weights for embeddings and encoder heads (MLP or LSTM).
        Initialize embedding parameters only.
        """
        common.p(f"\nInitialize CrossPromptEncoder from pretrained: '{self.init_state_dict_path}'")
        
        # Load the pretrained state dictionary and remove the 'default.' prefix from the keys
        # pretrained_state_dict_2 = torch.load(self.init_state_dict_path, map_location=torch.device('cpu'))
        # common.p(f"Pretrained state dict keys: {pretrained_state_dict_2.keys()}")

        pretrained_state_dict = load_peft_weights(self.init_state_dict_path)
        pretrained_state_dict.pop('prompt_embeddings', None)
        pretrained_state_dict = {k.replace('prompt_encoder.', ''): v for k, v in pretrained_state_dict.items()}
        pretrained_state_dict = {k.replace('default.', ''): v for k, v in pretrained_state_dict.items()}
        common.p("Pretrained parameters keys and shapes: ", {k: v.shape for k, v in pretrained_state_dict.items()})

        # Initialize embedding parameters
        pret_embedding_dict = {
            k: v for k, v in pretrained_state_dict.items()
            if not k.startswith('base_model')
        }
        # common.p("Pretrained encoder parameters keys and shapes: ", {k: v.shape for k, v in pret_embedding_dict.items()})
        common.p("XPE parameters keys and shapes: ", {k: v.shape for k, v in self.state_dict().items()})
        for name, param in pret_embedding_dict.items():
            if name in self.state_dict():
                common.p(f'[yellow]Initialize: {name}[/yellow]   {param.shape}')
                self.state_dict()[name].copy_(param)

            # WARNING: Temporary workaround for loading old APE weights
            elif name == 'embedding.weight' and self.encoder_ratio == 1:
                common.p(f'[yellow]Initialize: {name}[/yellow]   {param.shape}')
                self.state_dict()['xpe_embedding.weight'].copy_(param)

            else:
                common.p(f"⚠️ Pretrained parameter {name} not found in model. Skipping.")
        
        # exit()

    def set_grad_requirements(self):

        common.p(f"[yellow]Setting embedding trainability to {not self.embedding_freeze}...[/yellow]")
        for name, param in self.named_parameters():
            if 'embedding' in name and 'original_module' not in name:
                param.requires_grad = False if self.embedding_freeze else True

        if self.encoder_ratio > 0:
            common.p(f"[yellow]Setting encoder trainability to {not self.encoder_freeze}...[/yellow]")
            for name, param in self.named_parameters():
                if 'xpe_head' in name and 'original_module' not in name:
                    param.requires_grad = False if self.encoder_freeze else True

    def forward(self, indices, task_ids=None):
        """
        Forward pass for the PromptEncoder.

        Args:
            indices (torch.Tensor): Indices of virtual tokens. Shape: (batch_size, num_virtual_tokens)
            task_ids (torch.Tensor): Task IDs for each example in the batch. Shape: (batch_size,)
        """
        # print(f"\nCrossPromptEncoder.forward() > Embedding type: {self.embedding_type}", flush=True)  # Debugging: Print embedding type
        # print(f"CrossPromptEncoder.forward() > Indices.shape: {indices.shape}", flush=True)  # Debugging: Print indices shape
        # print(f"CrossPromptEncoder.forward() > Indices: {indices}", flush=True)  # Debugging: Print indices
        # print(f"CrossPromptEncoder.forward() > Task IDs: {task_ids.shape}, {task_ids}", flush=True)  # Debugging: Print task_ids
        # print(f"CrossPromptEncoder.forward() > Unique Task IDs: {set(task_ids.tolist())}", flush=True)  # Debugging: Print unique task_ids
        # exit()

        assert indices.size(1) == self.total_virtual_tokens, \
            f"indices.shape[1] ({indices.size(1)}) does not match total_virtual_tokens ({self.total_virtual_tokens})"

        # batch_size = indices.shape[0]

        if self.encoder_ratio == 0:
            spt_embeds = self.embedding(indices) # Shape: (batch_size, num_virtual_tokens, token_dim)
            return spt_embeds

        if self.encoder_ratio == 1:
            xpe_embeds = self.xpe_embedding(indices)
            xpe_encoded_embeds = self.xpe_head(xpe_embeds)
            return xpe_encoded_embeds

        if self.encoder_ratio > 0 and self.encoder_ratio < 1:
            spt_embeds = self.embedding(indices[:, :self.spt_virtual_tokens])
            xpe_embeds = self.xpe_embedding(indices[:, self.spt_virtual_tokens:])
            xpe_encoded_embeds = self.xpe_head(xpe_embeds)
            embeds = torch.cat([spt_embeds, xpe_encoded_embeds], dim=1)
            return embeds

        raise ValueError(f"Unknown encoder ratio: {self.encoder_ratio}")


    def get_device(self):
        embedding = self.embedding if hasattr(self, 'embedding') else self.xpe_embedding
        return embedding.weight.device

    def normalize_embeddings(self):
        # common.p('[yellow]CrossPromptEncoder.normalize_embeddings...[yellow]')
        norms_after = []
        if self.embedding_normalize:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if 'embedding' in name and param.ndim == 2 and param.requires_grad:
                        norm = param.norm(dim=-1, keepdim=True).clamp(min=1e-6)

                        if self.embedding_normalize == 'unit':
                            param.div_(norm)
                        elif self.embedding_normalize == 'clip':
                            scale = (norm.clamp(max=self.embedding_normalize_max_norm) / norm)
                            param.mul_(scale)

                        # Store mean norm after normalization
                        norms_after.append(param.norm(dim=-1).mean().item())
        return sum(norms_after) / len(norms_after) if norms_after else 0.0
    
    def gen_mlp_head(self, input_size, hidden_size, output_size, num_layers, dropout=0.1):
        hidden_layers = []
        for _ in range(num_layers - 1):
            hidden_layers.append(torch.nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(torch.nn.ReLU())
            hidden_layers.append(torch.nn.Dropout(dropout))  # <--- Inject dropout here
        layers = [
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),    
            *hidden_layers,
            torch.nn.Linear(hidden_size, output_size),
        ]
        mlp_head = torch.nn.Sequential(*layers)
        return mlp_head
        
def maybe_load_pretrained_classifier_state(base_model,init_state_dict_path):
    """

    """
    pretrained_state_dict = load_peft_weights(init_state_dict_path)
    classifier_layer_key = 'base_model.classifier'
    classifier_params_dict = {
        k.replace(classifier_layer_key, f'{classifier_layer_key}.modules_to_save.default'): v for k, v in pretrained_state_dict.items()
        if k.startswith(classifier_layer_key)
    }
    if classifier_params_dict:
        common.p(f"\nInitialize Classifier from pretrained: '{init_state_dict_path}'")
        common.p("Pretrained base model classifier state dict keys: ", classifier_params_dict.keys())
        common.p(f"Base model state dict keys: {base_model.state_dict().keys()}")
        # exit()
        for name, param in classifier_params_dict.items():
            if name in base_model.state_dict():
                base_model.state_dict()[name].copy_(param)
                common.p(f'[yellow]Initialized with weight - {name}[/yellow]   {param.shape}')
            else:
                common.p(f"⚠️ Pretrained weight {name} not found in model. Skipping.")
    return base_model

class LightweightSelfAttentionHead(torch.nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.layernorm = torch.nn.LayerNorm(embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, num_tokens, hidden_size)
        attn_output, _ = self.self_attn(x, x, x)  # Self-attend
        x = self.layernorm(x + self.dropout(attn_output))  # Residual + LN
        x = self.out_proj(x)  # Project to output size
        return x
    
    class LSTMWrapper(torch.nn.Module):
        def __init__(self, lstm: torch.nn.LSTM, mlp: torch.nn.Module):
            super().__init__()
            self.lstm = lstm
            self.mlp = mlp

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.mlp(lstm_out)