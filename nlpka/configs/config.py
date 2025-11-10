import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

import os
import nlpka.configs.storage as config_stor
# from  transformers.utils import PaddingStrategy
from nlpka.tools.enums import ConfigTypeSE, ModeSE, PretSourceSE

ds_base_dirs = ''

class CONFIG:
    
    stor_path = common.get_module_location(config_stor)

    @staticmethod
    def load(path, type: ConfigTypeSE):
        config_path = f'{CONFIG.stor_path}/{type}/{path}.yml'
        config = common.yaml_file_to_simple_nsp(config_path)
        CONFIG._load_env_vars(config)
        CONFIG._validate(config, type)
        config.file_path = path
        return config
    
    @staticmethod
    def _load_env_vars(config):
        if hasattr(config, 'env'):
            for key, value in common.simple_nsp_to_dict(config.env).items():
                if value != None:
                    os.environ[key] = value
    
    @staticmethod
    def _validate(config, type: ConfigTypeSE):
        if type == ConfigTypeSE.LANGUAGE_MODEL:
            CONFIG._validate_model(config)
            CONFIG._validate_labels(config)

    @staticmethod
    def _validate_labels(config):
        Y = getattr(config.ds, 'Y', None)
        if Y and len(Y.names) != Y.number and (config.eval.label_id_to_name or Y.name_to_id): 
            raise Exception('Configured Label names and number mismatch')
        
    @staticmethod
    def _validate_model(config):
        architecture = config.model.architecture
        if not architecture:
            raise Exception('Model architecture must be set')
        if config.mode in [ ModeSE.FINETUNE, ModeSE.TEST ]:
            pretrained = getattr(config.model, 'pretrained', None)
            if not pretrained:
                raise Exception('Pretrained Model must be set')
            source = getattr(pretrained, 'source', None)
            name = getattr(pretrained, 'name', None)
            time_id = getattr(pretrained, 'time_id', None)
            if not source or (not name and not time_id):
                raise Exception('Pretrained Model source and (name or time_id) must be set')
            # elif source == PretSourceSE.LOCAL:
            #     checkpoint = getattr(pretrained, 'checkpoint', None)
                # if not checkpoint:
                #     raise Exception('Pretrained Model checkpoint must be set')