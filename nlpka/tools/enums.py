from nlpka.tools.common import SimpleEnum

# Device types:
class DeviceSE(SimpleEnum):
    CPU = 'cpu'
    GPU = 'gpu'
    CUDA = 'cuda'

# Config types:
class ConfigTypeSE(SimpleEnum):
    DATASET = 'dataset'
    TOKENIZER = 'tokenizer'
    LANGUAGE_MODEL = 'language_model'

# Config modes:
class ModeSE(SimpleEnum):
    TRAIN = 'train'
    FINETUNE = 'finetune'
    EVALUATE = 'evaluate'
    TEST = 'test'
    CLEAN = 'clean'
    PREPROCESS = 'preprocess'

class SentTokTypeSE(SimpleEnum):
    KA = 'kast'
    NLTK = 'nltkst'
    SPACY = 'spacyst'

class WordTokTypeSE(SimpleEnum):
    NLTK_WHITESPACE = 'nltk_whitespace'
    NLTK_PUNCT = 'nltk_punct'

class TokTypeSE(SimpleEnum):
    BPE = 'bpe'
    BYTE_LEVEL = 'byte_level'
    BYTE_LEVEL_BPE = 'byte_level_bpe'
    NATIVE_SENTPIECE = 'native_sentpiece'
    HUGGINGFACE_SENTPIECE = 'huggingface_sentpiece'
    WORDPIECE = 'wordpiece'
    
class TokAlgSE(SimpleEnum):
    BPE = 'bpe'
    UNIGRAM = 'unigram'

# Dataset splits:
class DsSplitSE(SimpleEnum):
    NONE = ''
    TRAIN = 'train'
    TEST = 'test'
    VALIDATION = 'validation'

class DsStateSE(SimpleEnum):
    TOKENIZED = 'tokenized'
    SPLITS = 'splits'
    SUBSET = 'subset'
    SHORT = 'short'
    LONG = 'long'

class SaveDatasetAsSE(SimpleEnum):
    CSV = 'csv'
    HUGGINGFACE = 'huggingface'

# Dataset categories:
class DsCatSE(SimpleEnum):
    RAW = 'raw'
    CORPORA = 'corpora'
    BENCHMARKS = 'benchmarks'
    COLLECTIOS = 'collections'

# Dataset types:
class DsTypeSE(SimpleEnum):
    TEXT = 'text'
    JSON = 'json'
    CSV = 'csv'
    HUGGINGFACE = 'huggingface'
    HUGGINGFACE_SAVED = 'huggingface_saved'

# Model architectures:
class ModelArchSE(SimpleEnum):
    BERT = 'bert'
    ROBERTA = 'roberta'
    ELECTRA = 'electra'
    XLNET = 'xlnet'
    XGLM = 'xglm'
    XLMR = 'xlmr'
    T5 = 't5'

# Pretrained model or tokenizer sources:
class PretSourceSE(SimpleEnum):
    HUGGINGFACE = 'huggingface'
    LOCAL = 'local'

# Downstream task categories:
class TaskCatSE(SimpleEnum):
    LANGUAGE_MODELING = 'language_modeling'
    TEXT_CLASSIFICATION = 'text_classification'
    TEXT_PAIR_CLASSIFICATION = 'text_pair_classification'
    TOKEN_CLASSIFICATION = 'token_classification'
    STRUCTURAL_ANALYSIS = 'structural_analysis'
    TEXT_SIMILARITIY = 'text_similarity'
    TEXT_TO_TEXT = 'text_to_text'

# Downstream tasks:
class TaskNameSE(SimpleEnum):
    MLM = 'mlm' # Masked Language Modeling
    DMLM = 'dmlm' # Dynmaic Masked Language Modeling
    SA = 'sa' # Sentiment Analysis
    NER = 'ner' # Named Entity Recognition
    POS = 'pos' # Part-of-Speech Tagging
    PLM = 'plm' # Permutation Language Modeling
    TOPIC = 'topic' # Topic Detection

# Huggingface evaluation types:
class EvalTypeSE(SimpleEnum):
    # A metric is used to evaluate a model’s performance and usually 
    # involves the model’s predictions as well as some ground truth labels. 
    METRIC = 'metric'
    # A comparison is used to compare two models. This can e.g. be done
    # by comparing their predictions to ground truth labels and computing their agreement. 
    COMPARISON = 'comparison'
    # With measurements, one can investigate a dataset’s properties. 
    MEASUREMENT = 'measurement'

# BertTokenizer = { unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]' }
class BertTokenSE(SimpleEnum):
    BOS = '[CLS]'
    EOS = '[SEP]'
    SEP = '[SEP]'
    CLS = '[CLS]'
    PAD = '[PAD]'
    UNK = '[UNK]'
    MASK = '[MASK]'
    ADDITIONAL = [ ]

# ElectraTokenizer = { unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]' }
class ElectraTokenSE(SimpleEnum):
    BOS = '[CLS]'
    EOS = '[SEP]'
    SEP = '[SEP]'
    CLS = '[CLS]'
    PAD = '[PAD]'
    UNK = '[UNK]'
    MASK = '[MASK]'
    ADDITIONAL = [ ]

# RobertaTokenizer = { bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>' }
class RobertaTokenSE(SimpleEnum):
    BOS = '<s>'
    EOS = '</s>'
    SEP = '</s>'
    CLS = '<s>'
    PAD = '<pad>'
    UNK = '<unk>'
    MASK = '<mask>'
    ADDITIONAL = [ ]
    
# XLMRobertaTokenizer = { bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>' }
class XLMRobertaTokenSE(SimpleEnum):
    BOS = '<s>'
    EOS = '</s>'
    SEP = '</s>'
    CLS = '<s>'
    PAD = '<pad>'
    UNK = '<unk>'
    MASK = '<mask>'
    ADDITIONAL = [ ]

# XLNetTokenizer = { bos_token='<s>', eos_token='</s>', unk_token='<unk>', sep_token='<sep>', pad_token='<pad>', cls_token='<cls>', mask_token='<mask>', additional_special_tokens=['<eop>', '<eod>'] }
class XLNetTokenSE(SimpleEnum):
    BOS = '<s>'
    EOS = '</s>'
    SEP = '<sep>'
    CLS = '<cls>'
    UNK = '<unk>'
    PAD = '<pad>'
    MASK = '<mask>'
    EOP = '<eop>'
    EOD = '<eod>'
    ADDITIONAL = [ EOP, EOD ]
    
class T5TokenSE(SimpleEnum):
    BOS = '<s>'  # Beginning of sequence (not commonly used in T5)
    EOS = '</s>'  # End of sequence token
    SEP = '</s>'  # Separator token (T5 does not use explicit separation like BERT)
    CLS = '<s>'  # Not explicitly used in T5, but kept for consistency
    PAD = '<pad>'  # Padding token
    UNK = '<unk>'  # Unknown token
    MASK = '<mask>'  # Used in span corruption during pretraining
    ADDITIONAL = []  # No additional special tokens in standard T5