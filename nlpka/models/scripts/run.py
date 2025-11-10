import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

import wandb
import warnings
warnings.filterwarnings('ignore', message="`max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.")
warnings.filterwarnings('ignore', message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.")
# warnings.filterwarnings('ignore', message="The following columns in the training set don't have a corresponding argument in `BertForMaskedLM.forward` and have been ignored: length. If length are not expected by `BertForMaskedLM.forward`,  you can safely ignore this message.")

def run(config):

    from nlpka.models.model import MODEL
    from nlpka.datasets.dataset import DATASET
    from nlpka.tokenizers.tokenizer import TOKENIZER
    
    # Load Tokenizer
    tokenizer = TOKENIZER.load(config)

    # Load, Split and Preprocess Dataset
    dataset = DATASET(config, tokenizer)

    # Configure Model and Trainer
    model = MODEL(config, tokenizer, dataset)

    # Run Model and Return
    test_output = model.run()
    
    return model, test_output


if __name__ == '__main__':

    from nlpka.tools.enums import ConfigTypeSE
    from nlpka.configs.config import CONFIG

    # Parse Config Name Argument
    config_name = common.parse_script_args()
   
    # Load Configuration
    config = CONFIG.load(config_name, ConfigTypeSE.LANGUAGE_MODEL)

    # Run Model
    run(config)
    
    # Finish WandB Run If Active
    wandb.finish()
    
