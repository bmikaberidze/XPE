import nlpka.tools.common as common
common.info(__file__, __name__, __package__)

"""
Plot PCA of vocabulary embeddings versus soft prompt embeddings.

Usage (single config, as before):
    runtime/clusters/pegasus/shell/run.sh \
      "python -m nlpka.evaluations.scripts.run_2d_plot --config xlmr/finetune/peft/sib200_plot.xpe"

Usage (all id_by_groups):
    python -m nlpka.evaluations.scripts.run_2d_plot --config xlmr/finetune/peft/sib200_plot.xpe

    --
    squeue -u bmikaberidze --long
    scancel
"""
import os
import nlpka.tools.common as common
common.info(__file__, __name__, __package__)

import wandb
import warnings
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
import matplotlib.patheffects as pe
from nlpka.configs.scripts.xpe_utils import trained_model_paths, grouped_trained_models

import nlpka.evaluations.storage as eval_stor
eval_stor_path = common.get_module_location(eval_stor)
os.makedirs(os.path.dirname(f"{eval_stor_path}/temp/"), exist_ok=True)
os.makedirs(os.path.dirname(f"{eval_stor_path}/plots/"), exist_ok=True)
os.makedirs(os.path.dirname(f"{eval_stor_path}/plots/script_labels/"), exist_ok=True)



warnings.filterwarnings(
    "ignore",
    message="`max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.",
)
warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.",
)

import matplotlib
import itertools

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection
import numpy as np
import torch
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

FIGSIZE = (11, 7)
VOCAB_KEEP_ONLY = None
SAVE_SCRIPT_LABELS_SEPARATELY = False
# VOCAB_KEEP_ONLY = 9000 # 1000, 5000
VOCAB_SCALE = 0.15
SOFT_SCALE = 10
LANGUAGES = 'seen' # number or group of languages (seen/unseen)
CACHE_PATH = f"{eval_stor_path}/temp/sib200_script_cache_{LANGUAGES}.pkl"
VOCAB_MARKER = "o"
SOFT_MARKER = "*"
COLORS = {
    "SPT": "black",
    "XPE": "goldenrod",
    # 
    "SPT_J5": "green",
    "SPT_SEEN": "blue",
    "XPE_J5": "orange",
    "XPE_SEEN": "red",
}
DECOMPOSITION_TYPE = "tsne"
DECOMPOSITION_DICT = {
    "pca": PCA(n_components=2),
    "tsne": TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto"),
}

def get_peft_model_path_by_id(id):
    if id in trained_model_paths:
        return trained_model_paths[id]
    else:
        raise ValueError(f"PEFT model path not found for id: {id}")

id_by_groups = {
    "SPT": grouped_trained_models["SPT_SEEN"] + grouped_trained_models["SPT_J5"],
    "XPE": grouped_trained_models["XPE_SEEN"] + grouped_trained_models["XPE_J5"]
}

def project_and_plot_scripts(
    script_to_embs: dict[str, np.ndarray],
    mixed_embs: np.ndarray,
    soft_embs: np.ndarray,
    soft_group_labels: np.ndarray | None,
    output_path: str):
    """
    Plots:
      - Vocab tokens grouped by script (colored)
      - Mixed tokens in gray
      - Soft prompt groups in COLORS[...] (like original project_and_plot)
      - Soft prompts if ungrouped → black
    """

    # ---------------------------------------------
    # 0. Merge ALL vocab embeddings (scripts + mixed)
    # ---------------------------------------------
    all_vocab_blocks = []
    all_vocab_labels = []

    # Scripts
    for script, emb in script_to_embs.items():
        all_vocab_blocks.append(emb)
        all_vocab_labels.extend([script] * len(emb))

    # Mixed
    if mixed_embs is not None and len(mixed_embs) > 0:
        all_vocab_blocks.append(mixed_embs)
        all_vocab_labels.extend(["Mixed"] * len(mixed_embs))

    vocab_embs = np.vstack(all_vocab_blocks)
    vocab_labels = np.array(all_vocab_labels, dtype=object)

    # ---------------------------------------------
    # 1. Apply VOCAB_KEEP_ONLY exactly like project_and_plot
    # ---------------------------------------------
    if VOCAB_KEEP_ONLY and vocab_embs.shape[0] > VOCAB_KEEP_ONLY:
        idx = np.random.choice(vocab_embs.shape[0], VOCAB_KEEP_ONLY, replace=False)
        vocab_embs = vocab_embs[idx]
        vocab_labels = vocab_labels[idx]

    print(
        f"[INFO] vocab={vocab_embs.shape}, soft={soft_embs.shape} "
        f"scripts_labels={len(vocab_labels)} soft_labels={len(soft_group_labels)}"
    )

    # ---------------------------------------------
    # 2. Stack for PCA/TSNE
    # ---------------------------------------------
    stacked = np.vstack([vocab_embs, soft_embs])

    if DECOMPOSITION_TYPE == "tsne":
        print("[TSNE] PCA")
        X50 = PCA(n_components=50).fit_transform(stacked)
        print("[TSNE] TSNE")
        proj = DECOMPOSITION_DICT["tsne"].fit_transform(X50)
    else:
        print("[PCA] PCA only")
        proj = DECOMPOSITION_DICT["pca"].fit_transform(stacked)

    vocab_compr = proj[:len(vocab_embs)]
    soft_compr = proj[len(vocab_embs):]

    # ---------------------------------------------
    # 3. PLOTTING 2D
    # ---------------------------------------------
    plt.figure(figsize=FIGSIZE)

    # colors for script groups
    unique_scripts = sorted(set(vocab_labels.tolist()))
    color_map = build_script_color_map(unique_scripts)

    # ---- vocab tokens (scripts + mixed)
    for script in unique_scripts:
        mask = vocab_labels == script
        plt.scatter(
            vocab_compr[mask, 0],
            vocab_compr[mask, 1],
            s=VOCAB_SCALE,
            alpha=0.5,
            color=color_map[script],
            label=script,
        )

    # =============================================
    # SAVE LABELS
    # =============================================
    HALO_LW = 2.5 
    if not SAVE_SCRIPT_LABELS_SEPARATELY:
        labels = []
        for script in unique_scripts:
            if script in ["Mixed"]:
                continue

            mask = vocab_labels == script
            if mask.sum() == 0:
                continue

            pts = vocab_compr[mask]
            cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])

            txt = plt.text(
                cx, cy, script,
                fontsize=12,
                fontweight="bold",
                color=color_map[script],
                ha="center",
                va="center",
                zorder=10,
            )

            txt.set_path_effects([
                pe.Stroke(linewidth=HALO_LW, foreground="white"),
                pe.Normal()
            ])

            labels.append({
                "text": txt,
                "x": cx,
                "y": cy,
                "anchor": np.array([cx, cy])  # attract back to cluster
            })

            relax_labels(labels)

    else:

        FONTSIZE = 48   # big = crisp when pasted/scaled down
        DPI = 300
        script_labels_dir = f"{eval_stor_path}/plots/script_labels/"

        for script in unique_scripts:

            fig, ax = plt.subplots(figsize=(2, 1), dpi=300)
            fig.patch.set_alpha(0)     # transparent figure background
            ax.set_axis_off()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            txt = ax.text(
                0.5, 0.5, script,
                ha="center", va="center",
                fontsize=FONTSIZE,
                fontweight="bold",
                color=color_map[script],
            )

            # thin white outline around letters (no bbox!)
            txt.set_path_effects([
                pe.Stroke(linewidth=HALO_LW, foreground="white"),
                pe.Normal()
            ])

            # Save tightly cropped, transparent
            svg_path = os.path.join(script_labels_dir, f"{script}.svg")
            png_path = os.path.join(script_labels_dir, f"{script}.png")

            fig.savefig(svg_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
            fig.savefig(png_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)

    # ---- SOFT PROMPT GROUPS
    if soft_group_labels is None:
        plt.scatter(
            soft_compr[:, 0],
            soft_compr[:, 1],
            s=SOFT_SCALE,
            alpha=0.9,
            color="black",
            label="soft prompts",
            marker=SOFT_MARKER,
            edgecolors="none"
        )
    else:
        unique_groups = sorted(set(soft_group_labels.tolist()))
        for g in unique_groups:
            mask = soft_group_labels == g
            color = COLORS.get(g, "black")
            plt.scatter(
                soft_compr[mask, 0],
                soft_compr[mask, 1],
                s=SOFT_SCALE,
                alpha=0.9,
                color=color,
                label=g,
                marker=SOFT_MARKER,
                edgecolors="none"
            )

    # Legend outside + bigger markers
    handles, labels = plt.gca().get_legend_handles_labels()

    soft_handles = []
    soft_labels = []

    for h, l in zip(handles, labels):
        # if l in COLORS or l == "Mixed":
        if l in COLORS:
            soft_handles.append(h)
            soft_labels.append(l)

    plt.legend(
        soft_handles,
        soft_labels,
        markerscale=5,
        scatterpoints=1,
        # bbox_to_anchor=(1.05, 1),
        loc="lower right"
    )

    # Standardize figure size and export vector + raster outputs
    plt.gcf().set_size_inches(*FIGSIZE)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    base, _ = os.path.splitext(output_path)
    svg_path = f"{base}.svg"
    png_path = f"{base}.png"

    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {os.path.abspath(svg_path)} and {os.path.abspath(png_path)}")

def relax_labels(labels, iters=200, repel=0.02, attract=0.01,
                 step=0.05, max_radius=0.25):

    for _ in range(iters):
        for i, a in enumerate(labels):
            force = np.zeros(2)
            pa = np.array([a["x"], a["y"]])

            # repel from other labels
            for j, b in enumerate(labels):
                if i == j:
                    continue
                pb = np.array([b["x"], b["y"]])
                d = pa - pb
                dist = np.linalg.norm(d)
                if dist < 0.12:
                    force += (d / (dist + 1e-6)) * repel

            # attraction to cluster anchor
            force += (a["anchor"] - pa) * attract

            # apply damping
            delta = np.clip(force, -step, step)
            a["x"] += delta[0]
            a["y"] += delta[1]

            # hard constraint: stay near anchor
            offset = np.array([a["x"], a["y"]]) - a["anchor"]
            dist = np.linalg.norm(offset)
            if dist > max_radius:
                a["x"], a["y"] = a["anchor"] + offset / dist * max_radius

    for a in labels:
        a["text"].set_position((a["x"], a["y"]))

def build_script_color_map(unique_labels):
    """
    Assigns each script a stable, visually high-contrast color.
    'Mixed' is always gray.
    """
    palette = (
        list(plt.cm.tab20.colors) +
        list(plt.cm.Set1.colors) +
        list(plt.cm.Set2.colors) +
        list(plt.cm.Dark2.colors)
    )

    palette_iter = itertools.cycle(palette)
    color_map = {}

    for label in unique_labels:
        if label == "Mixed":
            color_map[label] = "gray"
        else:
            color_map[label] = next(palette_iter)

    return color_map

def collect_soft_embeddings_by_groups(config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all PEFT models listed in id_by_groups using the provided config,
    and collect:
      - one shared vocab_embeddings (from the first model),
      - stacked soft embeddings for all models,
      - group_labels (string per soft token row).
    """
    from nlpka.models.model import MODEL
    from nlpka.datasets.dataset import DATASET
    from nlpka.tokenizers.tokenizer import TOKENIZER

    tokenizer = TOKENIZER.load(config)
    dataset = DATASET(config, tokenizer)

    calc_vocab_embeddings = True
    vocab_embeddings = None
    soft_list = []
    group_labels = []

    for group_name, uuid_list in id_by_groups.items():
        for uid in uuid_list:
            encoder_path = get_peft_model_path_by_id(uid)
            # Update config to point to this adapter
            config.task.peft.encoder_init_state_dict_path = encoder_path
            config.task.peft.encoder_ratio = 0 if group_name.startswith("SPT") else 1
            model = MODEL(config, tokenizer, dataset)  # NOTE: same config/tokenizer/dataset reused

            ve, se = extract_embeddings(model.get(), calc_vocab_embeddings)
            if calc_vocab_embeddings:
                vocab_embeddings = ve  # keep vocab only from the first model
                calc_vocab_embeddings = False

            soft_list.append(se)
            group_labels.extend([group_name] * se.shape[0])

            # free model ASAP
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if vocab_embeddings is None:
        raise RuntimeError("No vocab embeddings collected – check id_by_groups / paths.")

    all_soft = np.vstack(soft_list)
    group_labels_arr = np.array(group_labels, dtype=object)
    print("[STEP] collect_soft_embeddings_by_groups: Completed collection.", flush=True)
    print(f"  - all_soft shape: {all_soft.shape}", flush=True)
    print(f"  - group_labels shape: {group_labels_arr.shape}", flush=True)
    return vocab_embeddings, all_soft, group_labels_arr

def extract_vocab_groups_by_script(
    base_model_name="FacebookAI/xlm-roberta-large",
    sib200_name="Davlan/sib200_14classes",
    max_texts_per_lang=200,
    verbose=True):
    """
    Returns:
        script_to_tokens: dict {script_name → set(token_ids)}
        mixed_tokens: set(token_ids shared across scripts)
        tokenizer: the XLM-R tokenizer
    """

    # -----------------------------------------------------
    # 0. Check for CACHE
    # -----------------------------------------------------
    if os.path.exists(CACHE_PATH):
        if verbose:
            print(f"[CACHE] Loading cached vocab groups from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        script_to_tokens = cached["script_to_tokens"]
        mixed_tokens = cached["mixed_tokens"]
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        return script_to_tokens, mixed_tokens, tokenizer

    if verbose:
        print("[CACHE] No cache found. Running extraction...")

    # -----------------------------------------------------
    # 1. Load tokenizer
    # -----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # -----------------------------------------------------
    # 2. Read dataset configs
    # -----------------------------------------------------
    configs = get_dataset_config_names(sib200_name)
    configs = [c for c in configs if not c.endswith("Nkoo.zip")]
    if LANGUAGES: 
        if isinstance(LANGUAGES, int):
            configs = configs[:LANGUAGES]
        elif LANGUAGES == "seen":
            # Import here to avoid top-level import cycles (see xpe_utils.py for seen SIB-200)
            from nlpka.configs.scripts.xpe_utils import xlmr_seen_sib200_ds_names
            configs = xlmr_seen_sib200_ds_names

    if verbose:
        print(f"[INFO] Found {len(configs)} SIB-200 language configs.")

    # Group languages by script suffix
    script_to_configs = defaultdict(list)

    for cfg in configs:
        if "_" in cfg:
            _, script = cfg.split("_", 1)
        else:
            script = "UNK"
        script_to_configs[script].append(cfg)

    if verbose:
        print(f"[INFO] Scripts: {list(script_to_configs.keys())}")

    # -----------------------------------------------------
    # 3. Tokenize datasets & gather token IDs per script
    # -----------------------------------------------------
    script_to_tokens = defaultdict(set)

    for script, cfg_list in script_to_configs.items():
        if verbose:
            print(f"\n[INFO] Processing script '{script}' ({len(cfg_list)} languages)")

        for cfg in cfg_list:
            try:
                dataset = load_dataset(sib200_name, cfg, split="train")
            except Exception as e:
                print(f"[WARN] Failed to load {cfg}: {e}")
                continue

            for i, example in enumerate(dataset):
                text = (
                    example.get("text")
                    or example.get("sentence")
                    or example.get("inputs")
                )

                if not text:
                    continue

                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=128,
                    add_special_tokens=True,
                )

                ids = encoded["input_ids"]

                if isinstance(ids[0], list):
                    for seq in ids:
                        script_to_tokens[script].update(seq)
                else:
                    script_to_tokens[script].update(ids)

                if i + 1 >= max_texts_per_lang:
                    break

        if verbose:
            print(f"  → {len(script_to_tokens[script])} unique tokens")

    # -----------------------------------------------------
    # 4. Compute mixed tokens (appear in >1 script)
    # -----------------------------------------------------
    token_to_scripts = defaultdict(set)
    for script, tokens in script_to_tokens.items():
        for tid in tokens:
            token_to_scripts[tid].add(script)

    mixed_tokens = {tid for tid, scrs in token_to_scripts.items() if len(scrs) > 1}

    if verbose:
        print(f"[INFO] Mixed tokens: {len(mixed_tokens)}")

    # Remove mixed tokens from individual script sets
    for script in script_to_tokens:
        script_to_tokens[script] -= mixed_tokens

    if verbose:
        for script, toks in script_to_tokens.items():
            print(f"Final unique tokens for '{script}': {len(toks)}")

    # -----------------------------------------------------
    # 5. SAVE CACHE
    # -----------------------------------------------------
    cache_obj = {
        "script_to_tokens": script_to_tokens,
        "mixed_tokens": mixed_tokens,
    }

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache_obj, f)

    if verbose:
        print(f"[CACHE] Saved cache to {CACHE_PATH}")

    return script_to_tokens, mixed_tokens, tokenizer

def extract_embeddings(peft_model, calc_vocab_embeddings: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return base vocab embeddings and soft prompt embeddings from a PEFT model."""
    base_model = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    if calc_vocab_embeddings:
        vocab_embeddings = base_model.get_input_embeddings().weight.detach().cpu().numpy()
    else:
        vocab_embeddings = None

    with torch.no_grad():
        prompt = peft_model.get_prompt(batch_size=1).detach().cpu().squeeze(0)
    soft_embeddings = prompt.numpy()

    return vocab_embeddings, soft_embeddings

def run_colorful_langs(config, output_path=f"{eval_stor_path}/plots/plot_scripts.png"):
    print("[STEP] run_colorful_langs: Starting...")

    # Get base vocab + soft prompt embeddings + soft prompt group labels
    vocab_embeddings_full, soft_embeddings, soft_group_labels = collect_soft_embeddings_by_groups(config)

    # Token IDs grouped by script
    script_to_tokens, mixed_tokens, tokenizer = extract_vocab_groups_by_script()

    # Token IDs → embedding rows
    script_to_embs = {
        script: vocab_embeddings_full[list(token_ids)]
        for script, token_ids in script_to_tokens.items()
        if len(token_ids) > 0
    }

    # Mixed token embeddings
    mixed_embs = (
        vocab_embeddings_full[list(mixed_tokens)]
        if len(mixed_tokens) > 0
        else np.zeros((0, vocab_embeddings_full.shape[1]))
    )

    # Plot with script colors + mixed + soft prompt groups
    project_and_plot_scripts(
        script_to_embs=script_to_embs,
        mixed_embs=mixed_embs,
        soft_embs=soft_embeddings,
        soft_group_labels=soft_group_labels,
        output_path=output_path,
    )

def main():
    from nlpka.tools.enums import ConfigTypeSE
    from nlpka.configs.config import CONFIG

    config_name = common.parse_script_args()
    config = CONFIG.load(config_name, ConfigTypeSE.LANGUAGE_MODEL)

    run_colorful_langs(config)

    wandb.finish()


if __name__ == "__main__":
    main()
