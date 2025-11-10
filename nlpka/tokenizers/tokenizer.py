import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
# from nlpka.tokenizers.bert_byt5 import BertByT5Tokenizer
import nlpka.tokenizers.storage as tok_stor
import nlpka.datasets.storage as ds_stor

import sentencepiece as spm
from transformers.convert_slow_tokenizer import SpmConverter
from transformers import AutoTokenizer, PreTrainedTokenizerFast, BertTokenizerFast, ElectraTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast, XLNetTokenizerFast, T5TokenizerFast

from tokenizers import normalizers
from tokenizers.processors import TemplateProcessing, RobertaProcessing, BertProcessing

from tokenizers.implementations.char_level_bpe import CharBPETokenizer
from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
from tokenizers.implementations.bert_wordpiece import BertWordPieceTokenizer
from tokenizers.implementations.sentencepiece_bpe import SentencePieceBPETokenizer
from tokenizers.implementations.sentencepiece_unigram import SentencePieceUnigramTokenizer

from nlpka.tools.enums import ModelArchSE, PretSourceSE, SentTokTypeSE, WordTokTypeSE, TokTypeSE
from nlpka.tools.enums import BertTokenSE, ElectraTokenSE, RobertaTokenSE, XLMRobertaTokenSE, XLNetTokenSE, T5TokenSE

# Sentence Tokenization
import spacy
spacy = spacy.blank("en")
spacy.add_pipe('sentencizer')
from nltk.tokenize import sent_tokenize
# from nlpka.tokenizers.lib.sent.ka_sen_tok import KaSenTok

# Word Tokenization
from nltk.tokenize import word_tokenize, RegexpTokenizer
whitespace_tokenize = RegexpTokenizer(r'\w+')

class TOKENIZER:

    min_frequency = 2

    ds_stor_path = common.get_module_location(ds_stor)
    stor_path = common.get_module_location(tok_stor)
    temp_path = f'{stor_path}/temp'

    sentpiece_pref = 'sp'
    fasttokenizer_pref = 'tokenizer'
    unigram_pref = 'unigram'

    special_tokens_map = {
        ModelArchSE.BERT: BertTokenSE,
        ModelArchSE.ELECTRA: ElectraTokenSE,
        ModelArchSE.ROBERTA: RobertaTokenSE,
        ModelArchSE.XLMR: XLMRobertaTokenSE,
        ModelArchSE.XLNET: XLNetTokenSE,
        ModelArchSE.T5: T5TokenSE, 
    }

    tokenizers_map = {
        ModelArchSE.BERT: BertTokenizerFast,
        ModelArchSE.ELECTRA: ElectraTokenizerFast,
        ModelArchSE.ROBERTA: RobertaTokenizerFast,
        ModelArchSE.XLMR: XLMRobertaTokenizerFast,
        ModelArchSE.XLNET: XLNetTokenizerFast,
        ModelArchSE.T5: T5TokenizerFast, 
    }

    tokenizers_args_map = {
        ModelArchSE.BERT: {
            'do_lower_case': False,
            'tokenize_chinese_chars': False,
            'strip_accents': None,
        },
        ModelArchSE.ELECTRA: {
            'do_lower_case': False,
            'tokenize_chinese_chars': False,
            'strip_accents': None,
        },
        ModelArchSE.ROBERTA: {
            'add_prefix_space': True, # Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
            'trim_offsets': True, # Whether the post processing step should trim offsets to avoid including whitespaces.
        },
        ModelArchSE.XLMR: {},
        ModelArchSE.XLNET: {
            'do_lower_case': False,
            'remove_space': True, # Whether to strip the text when tokenizing (removing excess spaces before and after the string).
            'keep_accents': True, # Whether to keep accents when tokenizing.
        },
        ModelArchSE.T5: {
            'use_fast': True,  # Ensures fast tokenization
        },
    }

    post_processors_map = {
        ModelArchSE.BERT: BertProcessing,
        ModelArchSE.ELECTRA: BertProcessing,
        ModelArchSE.ROBERTA: RobertaProcessing,
        ModelArchSE.XLMR: TemplateProcessing,
        ModelArchSE.XLNET: TemplateProcessing,
        ModelArchSE.T5: TemplateProcessing, 
    }

    # ka_sen_tok = KaSenTok()

    def __init__(self, config):
        '''
        Initiate TONEIZER For 
        Training New Tokenizer Model
        '''
        self._config = config
        self.type = config.model.type
        self.vocab_size = config.model.vocab_size
        self._set_paths()
    # 
    def _set_paths(self):
        self.corpora_file_paths = []
        dirs = self._config.corpora.dirs
        files = self._config.corpora.files
        # Set Corpora Paths
        for i in range(0, len(dirs)):
            self.corpora_file_paths.append(f'{self.ds_stor_path}/{dirs[i]}/{files[i]}.txt')
        # Set Model Name and Path
        dirs_as_name = '|'.join(['|'.join(dir.split('/')[1:]) for dir in dirs])
        self.name = f'{dirs_as_name}_{int(self.vocab_size/1000)}k'
        algorithm = self._config.training_args.model_type if self.type == TokTypeSE.NATIVE_SENTPIECE else ''
        self.path = self.gen_path(PretSourceSE.LOCAL, self.name, self.type, algorithm)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        print('.................................')
        print(self.path)
        print('.................................')
    # 
    def _read_corpora_files(self):
        self.corpus = []
        for file_path in self.corpora_file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                texts = file.readlines()
            self.corpus.extend(texts)
    # 
    def train(self):
        if self.type == TokTypeSE.NATIVE_SENTPIECE:
            self.train_native_sentpiece()
        else:
            if self.type == TokTypeSE.WORDPIECE:
                self.train_hf_wordpiece()
            elif self.type in [ TokTypeSE.BYTE_LEVEL_BPE, TokTypeSE.BYTE_LEVEL]:
                self.train_hf_byte_level_bpe()
    # 
    def train_hf_wordpiece(self):
        '''
        Train BertWordPieceTokenizer
        '''

        # Initialize a tokenizer with Bert WordPiece
        tokenizer = BertWordPieceTokenizer(
            vocab=None,
            sep_token = BertTokenSE.SEP,
            cls_token = BertTokenSE.CLS,
            pad_token = BertTokenSE.PAD,
            mask_token = BertTokenSE.MASK,
            **vars(self._config.model_args),
        )
        
        # Train the tokenizer
        tokenizer.train(
            files = self.corpora_file_paths,
            vocab_size = self.vocab_size,
            special_tokens = [ 
                BertTokenSE.UNK, 
                BertTokenSE.CLS, 
                BertTokenSE.SEP, 
                BertTokenSE.PAD, 
                BertTokenSE.MASK 
            ],
            **vars(self._config.training_args),
        )

        self.save_lm_adapted_hf_tokenizers(tokenizer)
    # 
    def train_hf_byte_level_bpe(self):
        '''
        Train ByteLevelBPETokenizer
        '''
        
        # Initialize an empty tokenizer
        tokenizer = ByteLevelBPETokenizer(
            vocab = None, # default: None - A dictionnary of string keys and their ids
            merges = None, # default: None - A list of pairs of tokens
            **vars(self._config.model_args),
        )

        # Add Normalizers
        tokenizer.normalizer = normalizers.BertNormalizer(
            **vars(self._config.normalizer_args.bert)
        )

        # Train the tokenizer 
        tokenizer.train(
            files = self.corpora_file_paths,
            vocab_size = self.vocab_size,
            special_tokens = [ 
                RobertaTokenSE.UNK, 
                RobertaTokenSE.CLS, 
                RobertaTokenSE.SEP, 
                RobertaTokenSE.PAD, 
                RobertaTokenSE.MASK 
            ],
            **vars(self._config.training_args),
        )

        self.save_lm_adapted_hf_tokenizers(tokenizer)
    # 
    def save_lm_adapted_hf_tokenizers(self, tokenizer):
        '''
        Save WordPiece as fast tokenizers 
        For BERT and ELECTRA language model architecture
        '''
        for lm_arch in [ ModelArchSE.BERT, ModelArchSE.ELECTRA ]:
            print(lm_arch)
            tokenizer = TOKENIZER.to_lm_fasttok(tokenizer, lm_arch)
            tokenizer.save_pretrained(f'{self.path}/{lm_arch}')
    # 
    def train_native_sentpiece(self):
        '''
        Train SentencePiece tokenizer
        '''
        control_symbols_set = set()
        for TokenSE in self.special_tokens_map.values():
            control_symbols_set.add(TokenSE.UNK)
            control_symbols_set.add(TokenSE.BOS)
            control_symbols_set.add(TokenSE.EOS)
            control_symbols_set.add(TokenSE.PAD)
            control_symbols_set.add(TokenSE.CLS)
            control_symbols_set.add(TokenSE.SEP)
            control_symbols_set.add(TokenSE.MASK)
            for additional_token in TokenSE.ADDITIONAL:
                control_symbols_set.add(additional_token)
        control_symbols_set.remove(self._config.training_args.unk_piece)
        # control_symbols_set.remove(self._config.training_args.bos_piece)
        # control_symbols_set.remove(self._config.training_args.eos_piece)
        # control_symbols_set.remove(self._config.training_args.pad_piece)

        control_symbols = ','.join(control_symbols_set)

        # Train the model
        # https://github.com/google/sentencepiece?tab=readme-ov-file#train-sentencepiece-model
        spm.SentencePieceTrainer.Train(
            input=','.join(self.corpora_file_paths), 
            vocab_size=self.vocab_size,
            control_symbols=control_symbols,
            model_prefix=f'{self.path}/{self.sentpiece_pref}', 
            **vars(self._config.training_args)
        )

        self.save_lm_adapted_native_sentpieces()
    # 
    def save_lm_adapted_native_sentpieces(self):
        '''
        Convert Sentencepiece to fast tokenizers 
        And save for each language model architectures (BERT, ELECTRA, ROBERTA, XLMR, XLNET)
        '''
        sp = spm.SentencePieceProcessor()
        sp_model_file = f'{self.path}/{self.sentpiece_pref}.model'
        sp.vocab_file = sp_model_file
        sp.load(sp_model_file)
        tokenizer_fast = SpmConverter(sp).converted()
        # print(tokenizer_fast.unk_token, tokenizer_fast.unk_token_id)
        # exit()

        for lm_arch in ModelArchSE:
            lm_arch = lm_arch.value; print(lm_arch)
            tokenizer_fast = TOKENIZER.to_lm_fasttok(tokenizer_fast, lm_arch)
            tokenizer_fast.save_pretrained(f'{self.path}/{lm_arch}')
    # 
    @staticmethod
    def to_lm_fasttok(tokenizer, lm_arch):
        '''
        Save tokenizer as fast tokenizer
        '''
        temp_path = TOKENIZER.temp_path

        if 'save' in dir(tokenizer):
            Path(temp_path).mkdir(parents=True, exist_ok=True)
            ft_model_file = f'{temp_path}/{TOKENIZER.fasttokenizer_pref}.json'
            tokenizer.save(ft_model_file, pretty=True)
        elif 'save_pretrained' in dir(tokenizer):
            tokenizer.save_pretrained(temp_path)
        else:
            raise ValueError('Tokenizer does not have save or save_pretrained method')

        PreTrainedTokenizerFast = TOKENIZER.tokenizers_map[lm_arch]
        model_args = TOKENIZER.tokenizers_args_map[lm_arch]
        tokenizer_fast = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=common.absolute_to_root_relative(temp_path),
            **model_args
        )
        shutil.rmtree(temp_path)
        tokenizer_fast = TOKENIZER.replace_unk_token_manually(tokenizer_fast, lm_arch)
        TOKENIZER.add_special_tokens(tokenizer_fast, lm_arch)
        TOKENIZER.add_post_processor(tokenizer_fast, lm_arch)
        return tokenizer_fast
    #
    @staticmethod
    def add_special_tokens(tokenizer, lm_arch):
        '''
        Add special tokens to tokenizer
        '''
        print()
        TokenSE = TOKENIZER.special_tokens_map[lm_arch]
        special_tokens_dict = {
            'bos_token': TokenSE.BOS,
            'cls_token': TokenSE.CLS,
            'unk_token': TokenSE.UNK,
            'pad_token': TokenSE.PAD,
            'sep_token': TokenSE.SEP,
            'eos_token': TokenSE.EOS,
            'mask_token': TokenSE.MASK,
            'additional_special_tokens': TokenSE.ADDITIONAL,
        }
        tokenizer.add_special_tokens(special_tokens_dict)
    # 
    @staticmethod
    def add_post_processor(tokenizer, lm_arch):
        '''
        Add post processing to tokenizer
        tokenizer._tokenizer is the underlying Tokenizer (from tokenizers library)
        '''
        t = tokenizer
        post_processor = None
        PostProcessorClass = TOKENIZER.post_processors_map[lm_arch]
        if PostProcessorClass in [ BertProcessing, RobertaProcessing ]:
            post_processor = PostProcessorClass(
                (t.sep_token, t.sep_token_id),
                (t.cls_token, t.cls_token_id),
            )
        elif PostProcessorClass == TemplateProcessing and lm_arch == ModelArchSE.XLMR :
            post_processor = PostProcessorClass(
                single=f"{t.bos_token}:0 $A:0 {t.eos_token}:0",
                pair=f"{t.bos_token}:0 $A:0 {t.eos_token}:0 $B:1 {t.eos_token}:1",
                special_tokens=[
                    (t.bos_token, t.bos_token_id),
                    (t.eos_token, t.eos_token_id),
                ],
            )
        elif PostProcessorClass == TemplateProcessing and lm_arch == ModelArchSE.XLNET:
            post_processor = PostProcessorClass(
                single=f"$A:0 {t.sep_token}:0 {t.cls_token}:2",
                pair=f"$A:0 {t.sep_token}:0 $B:1 {t.sep_token}:1 {t.cls_token}:2",
                special_tokens=[
                    (t.cls_token, t.cls_token_id),
                    (t.sep_token, t.sep_token_id),
                ],
            )

        if hasattr(t, '_tokenizer'):
            t._tokenizer.post_processor = post_processor
        t.post_processor = post_processor
    # 
    @staticmethod
    def replace_unk_token_manually(tokenizer, lm_arch):
        '''
        Replace unk_token with new_unk_token
        '''
        old_unk_token = tokenizer.unk_token
        old_unk_token_id = tokenizer.unk_token_id
        # 
        TokenSE = TOKENIZER.special_tokens_map[lm_arch]
        new_unk_token = TokenSE.UNK
        # 
        if old_unk_token != new_unk_token:
            #
            temp_path = TOKENIZER.temp_path
            tokenizer.save_pretrained(temp_path)
            # 
            file_path = f'{temp_path}/vocab.txt'
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                for i in range(0, len(lines)):
                    if old_unk_token in lines[i]:
                        lines[i] = lines[i].replace(old_unk_token, new_unk_token)
                        break
                with open(file_path, 'w') as file:
                    file.writelines(lines)
            # 
            file_path = f'{temp_path}/tokenizer.json'
            if os.path.exists(file_path):
                file = common.json_file_to_simple_nsp(file_path)
                for added_token in file.added_tokens:
                    if added_token.id == old_unk_token_id:
                        added_token.content = new_unk_token
                        break
                print(type(file.model.vocab))
                if isinstance(file.model.vocab, list):
                    for vocab_item in file.model.vocab:
                        if vocab_item[0] == old_unk_token:
                            vocab_item[0] = new_unk_token
                            break
                elif isinstance(file.model.vocab, SimpleNamespace):
                    setattr(file.model.vocab, new_unk_token, old_unk_token_id)
                    delattr(file.model.vocab, old_unk_token)
                    file.model.unk_token = new_unk_token
                common.simple_nsp_to_json_file(file, file_path)
            # 
            file_path = f'{temp_path}/unigram.json'
            if os.path.exists(file_path):
                file = common.json_file_to_simple_nsp(file_path)
                for vocab_item in file.vocab:
                    if vocab_item[0] == old_unk_token:
                        vocab_item[0] = new_unk_token
                        break
                common.simple_nsp_to_json_file(file, file_path)
            # 
            file_path = f'{temp_path}/tokenizer_config.json'
            if os.path.exists(file_path):
                file = common.json_file_to_simple_nsp(file_path)
                getattr(file.added_tokens_decoder, str(old_unk_token_id)).content = new_unk_token
                file.unk_token = new_unk_token
                common.simple_nsp_to_json_file(file, file_path)
            # 
            file_path = f'{temp_path}/special_tokens_map.json'
            if os.path.exists(file_path):
                file = common.json_file_to_simple_nsp(file_path)
                file.unk_token = new_unk_token
                common.simple_nsp_to_json_file(file, file_path)
            #
            file_path = f'{temp_path}/vocab.json'
            if os.path.exists(file_path):
                file = common.json_file_to_simple_nsp(file_path)
                setattr(file, new_unk_token, old_unk_token_id)
                delattr(file, old_unk_token)
                common.simple_nsp_to_json_file(file, file_path)
            #
            tokenizer = AutoTokenizer.from_pretrained(temp_path)
            shutil.rmtree(temp_path)
        # 
        return tokenizer
    #
    @staticmethod
    def gen_path(source, name, type, algorithm = None):
        '''
        Get Tokenizer Path According Type, Algorithm and Name
        '''
        if source == PretSourceSE.HUGGINGFACE:
            tokenizer_path = name
        else:
            algorithm = f'/{algorithm}' if algorithm else ''
            tokenizer_path = f'{TOKENIZER.stor_path}/{type}{algorithm}/{name}'
        return tokenizer_path
    # 
    @staticmethod
    def vocab_size_by_lm_size(lm_param_size, lm_embedding_size, ratio = 0.2):
        '''
        Calculate vocabulary size according to language model size
        '''
        return int(lm_param_size * ratio / lm_embedding_size)
    # 
    @staticmethod
    def load(config):
        '''
        Load Pretrained AutoTokenizer According Language Models Configuration
        tokenizer:
            source:       local       | huggingface
            name:         local_dir   | hf_name
            type:         wordpiece   | sentpiece
            parralelism:  'false'     | 'true'
        '''
        tok_conf = config.tokenizer
        tokenizer_path = TOKENIZER.gen_path(tok_conf.source, tok_conf.name, tok_conf.type, tok_conf.algorithm)
        print()
        print('Load Tokenizer From: ', tokenizer_path)

        # if tok_conf.type == TokTypeSE.BYTE_LEVEL:
        #     tokenizer = BertByT5Tokenizer(tokenizer_path) # tokenizer_path == tok_conf.name
        # else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
        # Adapt to Language Model in case of HuggingFace source:
        if tok_conf.adapt_to_lm:
            tokenizer = TOKENIZER.to_lm_fasttok(tokenizer, config.model.architecture)

        tokenizer.path = tokenizer_path
        print('Source: ', tok_conf.source)
        print('Type: ', tok_conf.type)
        print('Algorithm: ', tok_conf.algorithm)
        print('Vocabulary Size: ', tokenizer.vocab_size)
        print('Vocabulary Actual Size: ', len(tokenizer.get_vocab()))
        print()

        return tokenizer

    @staticmethod
    def tokenize_sentences(text, method=SentTokTypeSE.KA):
        # if method == SentTokTypeSE.KA:
        #     return TOKENIZER.ka_sen_tok.tokenize(text)
        # el
        if method == SentTokTypeSE.NLTK:
            return sent_tokenize(text)
        elif method == SentTokTypeSE.SPACY:
            doc = spacy(text)
            return [sent.text for sent in doc.sents]
        else:
            raise ValueError('Invalid sentence tokenization method')
        
    @staticmethod
    def tokenize_words(text, method=WordTokTypeSE.NLTK_PUNCT):
        if method == WordTokTypeSE.NLTK_PUNCT:
            return word_tokenize(text)
        elif method == WordTokTypeSE.NLTK_WHITESPACE:
            return whitespace_tokenize.tokenize(text)
        else:
            raise ValueError('Invalid word tokenization method')