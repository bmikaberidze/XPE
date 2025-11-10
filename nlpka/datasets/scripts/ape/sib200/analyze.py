'''
use:
    runtime/clusters/pegasus/shell/run.sh "python -m nlpka.datasets.scripts.ape.sib200.analyze ape/sib200/tokenize"
    python -m nlpka.datasets.scripts.ape.sib200.analyze ape/sib200/tokenize
    --
    squeue -u bmikaberidze -l
    scancel
'''

import os
import csv
import math
import torch
from tqdm import tqdm
import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

from datasets import get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from nlpka.evaluations.evaluate import EVALUATE

# Choose model: 'aya' or 'xmlr'
USE_MODEL = "aya"  # Change to "xmlr" if needed

if USE_MODEL == "aya":
    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-101")
    model = AutoModelForSeq2SeqLM.from_pretrained("nlpka/models/storage/t5/aya101").eval().cuda()
elif USE_MODEL == "xmlr":
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base").eval().cuda()
else:
    raise ValueError(f"Unsupported model type: {USE_MODEL}")

sib200_langs = get_dataset_config_names("Davlan/sib200")
init_dirs = None
stats_list = []

def compute_stats(texts):
    total_tokens = 0
    total_words = 0
    total_unk = 0

    for text in texts:
        if not isinstance(text, str): continue
        words = text.split()
        tokens = tokenizer.tokenize(text, add_special_tokens=True)
        print(tokenizer.unk_token, tokens)
        exit()
        unks = sum(1 for t in tokens if t == tokenizer.unk_token_id)

        total_words += len(words)
        total_tokens += len(tokens)
        total_unk += unks

    tokens_per_word = total_tokens / total_words if total_words else 0
    unk_frac = total_unk / total_tokens if total_tokens else 0

    if USE_MODEL == "aya":
        perplexity = compute_perplexity(texts)
    elif USE_MODEL == "xmlr":
        perplexity = compute_pseudo_perplexity(texts)
    else:
        perplexity = None

    return {
        f'{USE_MODEL}_perplexity': round(perplexity, 2) if perplexity is not None else None,
        f'{USE_MODEL}_tokens_per_word': round(tokens_per_word, 2),
        f'{USE_MODEL}_unk_frac': round(unk_frac, 2),
    }

def compute_pseudo_perplexity(texts, max_length=256):
    total_log_likelihood = 0.0
    total_predictions = 0

    for text in tqdm(texts, desc="Processing texts"):
        if not isinstance(text, str) or not text.strip():
            continue

        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoding["input_ids"][0]
        if len(input_ids) < 3:
            continue  # skip short sequences

        for i in range(1, len(input_ids) - 1):  # skip CLS and SEP/EOS
            masked_input = input_ids.clone()
            masked_input[i] = tokenizer.mask_token_id
            with torch.no_grad():
                output = model(input_ids=masked_input.unsqueeze(0).to("cuda"))
                logits = output.logits[0, i]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                true_id = input_ids[i]
                prob = probs[true_id]
                log_prob = torch.log(prob + 1e-10)  # avoid log(0)
                total_log_likelihood += log_prob.item()
                total_predictions += 1
    if total_predictions == 0:
        return None
    return math.exp(-total_log_likelihood / total_predictions)

def compute_perplexity(texts, max_length=256):
    total_log_likelihood = 0.0
    total_tokens = 0

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        neg_log_likelihood = outputs.loss.item() * inputs["input_ids"].size(1)
        total_log_likelihood += neg_log_likelihood
        total_tokens += inputs["input_ids"].size(1)

    if total_tokens == 0:
        return float("inf")
    return round(math.exp(total_log_likelihood / total_tokens), 2)

def save_stats(stats_list):
    csv_path = f'{EVALUATE.stor_path}/{USE_MODEL}/sib200_stats.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ['id', 'lang'] + [
        key for key in stats_list[0].keys() if key not in ['id', 'lang']
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_list)
    print(f"✅ SIB-200 Lengths and {USE_MODEL} stats are saved to: {csv_path}")

if __name__ == '__main__':
    from nlpka.tools.enums import ConfigTypeSE
    from nlpka.tokenizers.tokenizer import TOKENIZER
    from nlpka.datasets.dataset import DATASET
    from nlpka.configs.config import CONFIG

    config_name = common.parse_script_args()
    config = CONFIG.load(config_name, ConfigTypeSE.DATASET)
    tokenizer_ds = TOKENIZER.load(config)

    for i, lang in tqdm(enumerate(sib200_langs)):
        if lang in ['nqo_Nkoo.zip']: continue

        init_dirs = init_dirs or config.ds.dirs
        config.ds.dirs = f'{init_dirs}/{lang}'
        config.ds.descriptive_name = f'sib200_{lang}'

        dataset = DATASET(config, tokenizer_ds)

        try:
            test_texts = dataset.test['inputs']
        except Exception as e:
            print(f"⚠️ Skipping {lang} due to error: {e}")
            continue

        stats = {'id': i, 'lang': lang}
        stats.update(compute_stats(test_texts))
        stats.update(dataset.analyze_lengths())

        stats_list.append(stats)
        common.p(stats)

        # if i > 5: break

    save_stats(stats_list)
