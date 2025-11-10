import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

import os
import torch
from nlpka.tools.enums import DeviceSE
from safetensors.torch import load_file
import nlpka.models.storage as model_stor
from nlpka.models.cross_prompt_encoder import (
    is_cross_prompt_encoder,
    get_cross_prompt_encoder
)
from peft import (
    PeftModel,
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    MultitaskPromptTuningInit,
)

class PEFT:

    prompt_encoder_key = 'default'
    stor_path = common.get_module_location(model_stor)

    @staticmethod
    def is_peft(base):
        return hasattr(base._config.task, 'peft') or hasattr(base._config.model.pretrained, 'adapter')

    @staticmethod
    def from_pretrained(base_model, path):
        return PeftModel.from_pretrained(base_model, path)

    @staticmethod
    def setup_model(base):

        pret = base._config.model.pretrained
        peft = getattr(base._config.task, 'peft', None)
        
        if pret and hasattr(pret, 'adapter'):
            print('Load Pretrained PEFT Model...')
            pret_adapter_path = base._get_pret_path(pret.adapter)
            base._model = PEFT.from_pretrained(base._model, pret_adapter_path)
            print(base._model)
            exit()
        
        elif peft:
            # Load Cross Prompt Encoder
            if is_cross_prompt_encoder(peft):
                print('Setup Cross Prompt Encoder Model...')
                base._model = get_cross_prompt_encoder(base._model, peft)

            else:
                print('Setup PEFT Model...')
                peft_config = get_peft_config(vars(peft))
                base._model = get_peft_model(base._model, peft_config)

                # peft.mapping.get_peft_model
                #     peft.peft_model.PeftModelForSequenceClassification
                #         peft.peft_model.PeftModel
                #             peft.peft_model.PeftModel.add_adapter
                #                 peft.utils.others._prepare_prompt_learning_config
                #                 peft.peft_model.PeftModel._setup_prompt_encoder
                #                     peft.tuners.p_tuning.model.PromptEncoder(PromptEncoderConfig)
                #                 peft.peft_model.PeftModel.set_additional_trainable_modules(peft_config, adapter_name)

                if PEFT.is_mpt_target_tuning(base._config):
                    adapter = PEFT.load_adapter(peft.prompt_tuning_init_state_dict_path)
                    adapter["prefix_task_cols"] = adapter["prefix_task_cols"].mean(dim=0, keepdim=True)
                    adapter["prefix_task_rows"] = adapter["prefix_task_rows"].mean(dim=0, keepdim=True)
                    set_peft_model_state_dict(base._model, adapter)

    @staticmethod
    def load_adapter(path):
        """
        Automatically detects whether the provided path is a directory or a model file.
        If a directory is given, it searches for `adapter_model.bin` or `adapter_model.safetensors`.
        If a file is provided, it loads the model directly.

        Args:
            adapter_path (str): Path to the adapter model or directory containing it.

        Returns:
            dict: The loaded model state_dict.
        """
        # Check if the provided path is a directory
        if os.path.isdir(path):
            # Search for a valid adapter model file inside the directory
            model_files = [f for f in os.listdir(path) if f.startswith("adapter_model.")]

            if not model_files:
                raise FileNotFoundError(f"No adapter model found in directory: {path}")

            # Prioritize safetensors over bin for security
            model_files.sort(key=lambda x: (".safetensors" not in x, x))  # Prefers safetensors first
            model_file = os.path.join(path, model_files[0])
        else:
            # If it's already a file, use it directly
            model_file = path

        # Determine file extension
        ext = os.path.splitext(model_file)[-1].lower()
                
        try:
            if ext == ".safetensors":
                return load_file(model_file)  # SafeTensors secure loading
            elif ext in {".pth", ".pt", ".bin"}:
                return torch.load(model_file, map_location=DeviceSE.CPU)  # Load PyTorch model safely
            else:
                raise ValueError(f"Unsupported file format: {ext}")

        except Exception as e:
            raise RuntimeError(f"Error loading adapter model from {model_file}: {e}")

    @staticmethod
    def is_mpt_target_tuning(config):
        peft = getattr(config.task, 'peft', None)
        if peft:
            if peft.peft_type == PeftType.MULTITASK_PROMPT_TUNING and \
               peft.prompt_tuning_init in [
                    MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS,
                    MultitaskPromptTuningInit.EXACT_SOURCE_TASK,
                    MultitaskPromptTuningInit.ONLY_SOURCE_SHARED,
                ]:
                return True
        return False

    @staticmethod
    def print_prompt_encoder(base, key=None):
        """
        Dynamically print all layers, their parameter counts, and mean parameter values in the prompt encoder.
        """
        key = key if key else PEFT.prompt_encoder_key
        prompt_encoder = base._model.prompt_encoder[key]

        def count_parameters(module):
            """Counts trainable parameters in a module."""
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        def mean_parameter_value(module):
            """Computes the mean of all parameters in a module."""
            params = [p for p in module.parameters() if p.requires_grad]
            if not params:
                return 0.0  # No trainable parameters in this module
            return torch.cat([p.detach().flatten() for p in params]).mean().item()

        print(f"\n🔍 Listing layers in `prompt_encoder[{key}]`...\n")
        print(prompt_encoder)

        # ✅ Dynamically iterate through all submodules inside prompt_encoder
        for name, module in prompt_encoder.named_children():
            num_params = count_parameters(module)
            mean_value = mean_parameter_value(module)
            print(f"🔹 {name}: {num_params} parameters | Mean Value: {mean_value:.10f}")

        # ✅ Compute total trainable parameters in the entire prompt_encoder
        total_params = count_parameters(prompt_encoder)
        total_mean_value = mean_parameter_value(prompt_encoder)

        print(f"✅ Total trainable parameters in `prompt_encoder`: {total_params} | Mean Value: {total_mean_value:.10f}")
