import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

from nlpka.tools.enums import ModelArchSE
from transformers import AutoTokenizer

class CustomXlmRoberta():
    '''
    '''
    hf_name = 'xlm-roberta-base'
    def __init__(self, model_arch = ModelArchSE.BERT):
        
        self.xlmr = AutoTokenizer.from_pretrained(self.hf_name)
        
        from nlpka.tokenizers.tokenizer import TOKENIZER
        TOKENIZER.add_special_tokens(self.xlmr, model_arch)
        TOKENIZER.add_post_processor(self.xlmr, model_arch)
        
        self.__dict__.update(self.xlmr.__dict__)
