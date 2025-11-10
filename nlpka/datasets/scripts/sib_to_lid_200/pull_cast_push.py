'''
python -m nlpka.datasets.scripts.sib_to_lid_200.pull_cast_push
'''
# code.py
"""
Build and publish LID-200 (Language Identification) from SIB-200.

- Loads all language configs of Davlan/sib200
- Drops topic label/category, adds a `lang` label from the config name
- (Optionally) keeps `index_id` (recommended for provenance); can be dropped via --drop-index-id
- Concatenates all languages per split (train/validation/test)
- Pushes unified dataset to the Hugging Face Hub under --repo-id
- Uploads a comprehensive README.md (dataset card) modeled after SIB-200's card, adapted for LID

Usage:
  python code.py --repo-id mikaberidze/lid200
  # optional flags:
  #   --drop-index-id         -> do NOT keep index_id
  #   --private               -> create private repo
  #   --license cc-by-sa-4.0  -> set license tag in the README
  #   --max-shard-size 1GB    -> shard size for push_to_hub

Prereqs:
  pip install -U datasets huggingface_hub
  huggingface-cli login
"""

import argparse
from pathlib import Path
from typing import Dict, List

from datasets import (
    load_dataset,
    DatasetDict,
    concatenate_datasets,
    Features,
    Value,
    ClassLabel,
    get_dataset_config_names,
)
from huggingface_hub import HfApi, create_repo, upload_file, whoami


DEFAULT_SOURCE_DATASET = "Davlan/sib200"
DEFAULT_REPO_ID = "mikaberidze/lid200"
DEFAULT_LICENSE = "cc-by-sa-4.0"
DEFAULT_MAX_SHARD = "1GB"
README_PATH = Path("README.md")


# BibTeX block provided via a placeholder to avoid str.format parsing `{}` inside
BIBTEX_BLOCK = """\
```
@inproceedings{adelani2024sib200,
  title={{SIB-200}: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects},
  author={Adelani, David Ifeoluwa and Liu, Hannah and Shen, Xiaoyu and Vassilyev, Nikita and Alabi, Jesujoba O. and Mao, Yanke and Gao, Haonan and Lee, Annie En-Shiun and others},
  booktitle={EACL},
  year={2024}
}
```"""

README_TEMPLATE = """\
---
datasets:
- {repo_id}
language:
- multilingual
license: {license}
task_categories:
- text-classification
task_ids:
- language-identification
pretty_name: LID-200 (Language Identification from SIB-200)
size_categories:
- 100K<n<1M
tags:
- language-identification
- lid
- sib-200
- sib200
- flores-200
- news-topic
---

# Dataset Card for LID-200

**LID-200** is a unified language identification dataset derived from **SIB-200** by concatenating all language-specific subsets into a *single* dataset with the standard splits: `train`, `validation`, and `test`.
Each example contains:
- `text`: the sentence/text from SIB-200
- `lang`: a label indicating the language (e.g., `eng_Latn`, `kat_Geor`, …)
{index_id_sentence}

- **Source**: [{source_ds}](https://huggingface.co/datasets/{source_ds})
- **License**: {license} (inherited; share-alike)
- **Languages**: {num_langs}+ (see list below)

## Dataset Summary

LID-200 converts SIB-200 (topic classification over 200+ languages) into a *language identification* task. We remove topic labels and keep only the text plus a language label derived from the SIB-200 subset name. This enables training and evaluating LID systems across a very large set of languages and scripts using a consistent format.

## Supported Tasks and Leaderboards

- **language-identification**: predict the language ID for a given text.

## Languages

There are {num_langs}+ languages/dialects. See the full list below in **Languages list**.

## Dataset Structure

### Data Instances

Examples (using a random language code):

```python
from datasets import load_dataset
data = load_dataset("{this_repo}")  # unified dataset (no per-language configs)

example = data["train"][0]
example
```

A typical example looks like:

```json
{{
  "text": "…",
  "lang": "kat_Geor"{maybe_index_example}
}}
```

### Data Fields

- `text`: *(string)* the original text from SIB-200
- `lang`: *(ClassLabel)* language identifier derived from the SIB-200 subset name (e.g., `tur_Latn`){maybe_index_field}

### Data Splits

All languages are concatenated split-wise:

| Split       | # Examples |
|-------------|------------|
| train       | {n_train}  |
| validation  | {n_valid}  |
| test        | {n_test}   |
| **Total**   | {n_total}  |

> Note: In SIB-200, the original splits were named `train`, `dev`, and `test`. On the Hub they appear as `train`, `validation`, and `test`.

## Dataset Creation

### Curation Rationale

This dataset repackages SIB-200 into a format convenient for language identification research and preprocessing pipelines (e.g., filtering and routing).

### Source Data

SIB-200 originates from FLORES-200-like sentences and news-domain content; please refer to the original SIB-200 card and paper for details.

### Initial Data Collection and Normalization

We inherit preprocessing from SIB-200. No additional text normalization is applied beyond adding the `lang` column and removing the topic label.

### Who are the source language producers?

See SIB-200 documentation; sources include news organizations and Wikipedia-like text (see their card/paper).

## Annotations

### Annotation process & Annotators

See SIB-200; the topic labels were crowdsourced. LID-200 drops these labels.

## Personal and Sensitive Information

The source is news/public text. No additional annotation was added here.

## Considerations for Using the Data

### Social Impact

Large-scale LID enables better multilingual NLP pipelines for under-represented languages but should be used responsibly to avoid overgeneralization across domains not represented by news text.

### Discussion of Biases / Limitations

- Domain bias (news and related sources)
- Short texts can be challenging for LID in closely related languages or shared scripts

## Additional Information

### Dataset Curators

This dataset was created automatically from SIB-200 by [{author_handle}](https://huggingface.co/{author_handle}).

### Licensing Information

Inherited from SIB-200: **{license}** (share-alike).

### Citation Information

If you use **LID-200**, please **cite the original SIB-200 paper**:

> Adelani et al. (2024). “SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects.” EACL 2024. (arXiv:2309.07445)

{BIBTEX}

### Languages list

{LANGS_LIST}
"""


def build_and_push(
    repo_id: str = DEFAULT_REPO_ID,
    source_ds: str = DEFAULT_SOURCE_DATASET,
    keep_index_id: bool = True,
    license_tag: str = DEFAULT_LICENSE,
    private: bool = False,
    max_shard_size: str = DEFAULT_MAX_SHARD,
):
    # --- Auth ---
    api = HfApi()
    user = whoami()
    print(f"Logged in as: {user.get('name') or user['username']}")

    # --- Enumerate configs ---
    print(f"Fetching configs for {source_ds} ...")
    configs: List[str] = sorted(get_dataset_config_names(source_ds))
    configs = [cfg for cfg in configs if cfg != "nqo_Nkoo.zip"]
    # configs = [ 'ajp_Arab', 'ace_Latn' ]
    print(f"Found {len(configs)} language subsets.")

    # --- Accumulate per split ---
    per_split: Dict[str, list] = {"train": [], "validation": [], "test": []}
    langs_seen: List[str] = []

    for cfg in configs:
        print(f"Loading subset: {cfg}")
        ds_dict: DatasetDict = load_dataset(source_ds, cfg)

        for split in ["train", "validation", "test"]:
            if split not in ds_dict:
                continue

            ds = ds_dict[split]

            # Decide which columns to keep
            keep_cols = ["text"]
            if keep_index_id and "index_id" in ds.column_names:
                keep_cols.append("index_id")

            drop_cols = [c for c in ds.column_names if c not in keep_cols]
            if drop_cols:
                ds = ds.remove_columns(drop_cols)

            # Add language column with the config name
            ds = ds.add_column("lang", [cfg] * len(ds))

            per_split[split].append(ds)

        langs_seen.append(cfg)

    # --- Concatenate per split ---
    unified = {}
    for split in ["train", "validation", "test"]:
        if per_split[split]:
            print(f"Concatenating {split} with {len(per_split[split])} shards ...")
            unified[split] = concatenate_datasets(per_split[split])

    ddict = DatasetDict(unified)

    # --- Define features: text + lang (+ optional index_id) ---
    lang_names = sorted(set(langs_seen))
    features_dict = {
        "text": Value("string"),
        "lang": ClassLabel(names=lang_names),
    }
    if keep_index_id:
        features_dict["index_id"] = Value("int64")

    features = Features(features_dict)

    for split in ddict.keys():
        ddict[split] = ddict[split].cast(features)

    # Reorder columns to a consistent layout
    ordered_cols = ["text", "lang"] + (["index_id"] if keep_index_id else [])
    for split in ddict.keys():
        ddict[split] = ddict[split].select_columns(ordered_cols)

    print(ddict)

    # --- Create/ensure repo ---
    print(f"Creating or updating repo: {repo_id} (private={private})")
    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)

    # --- Push dataset ---
    print("Pushing dataset to the Hub ...")
    ddict.push_to_hub(repo_id, max_shard_size=max_shard_size)

    # --- README ---
    n_train = ddict["train"].num_rows if "train" in ddict else 0
    n_valid = ddict["validation"].num_rows if "validation" in ddict else 0
    n_test = ddict["test"].num_rows if "test" in ddict else 0
    n_total = n_train + n_valid + n_test

    index_id_sentence = (
        " - `index_id`: *(int64)* FLORES-200 sentence id retained for provenance."
        if keep_index_id else
        ""
    )
    maybe_index_example = (
        ',\n  "index_id": 1523'
        if keep_index_id else
        ""
    )
    maybe_index_field = (
        "\n- `index_id`: *(int64)* FLORES-200 sentence id (retained for traceability)"
        if keep_index_id else
        ""
    )

    readme_text = README_TEMPLATE.format(
        repo_id=repo_id,
        license=license_tag,
        source_ds=source_ds,
        num_langs=len(lang_names),
        this_repo=repo_id,
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        n_total=n_total,
        author_handle=repo_id.split("/")[0],
        index_id_sentence=index_id_sentence,
        maybe_index_example=maybe_index_example,
        maybe_index_field=maybe_index_field,
        LANGS_LIST="\n".join(f"- {x}" for x in lang_names),
        BIBTEX=BIBTEX_BLOCK,
    )

    README_PATH.write_text(readme_text, encoding="utf-8")
    print("Uploading README.md ...")
    upload_file(
        path_or_fileobj=str(README_PATH),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print("Done! Check your dataset page:")
    print(f"https://huggingface.co/datasets/{repo_id}")


def parse_args():
    p = argparse.ArgumentParser(description="Convert SIB-200 to unified LID-200 and push to HF Hub.")
    p.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Target dataset repo id, e.g., mikaberidze/lid200")
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE_DATASET, help="Source dataset (default: Davlan/sib200)")
    p.add_argument("--drop-index-id", action="store_true", help="If set, do NOT keep index_id")
    p.add_argument("--license", type=str, default=DEFAULT_LICENSE, help="License tag for README front-matter")
    p.add_argument("--private", action="store_true", help="Create a private repo")
    p.add_argument("--max-shard-size", type=str, default=DEFAULT_MAX_SHARD, help="Max shard size for push_to_hub (e.g., 500MB, 1GB)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_and_push(
        repo_id=args.repo_id,
        source_ds=args.source,
        keep_index_id=(not args.drop_index_id),
        license_tag=args.license,
        private=args.private,
        max_shard_size=args.max_shard_size,
    )
