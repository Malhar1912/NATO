"""
squad_noise_nato_experiment.py

Run multiple trials comparing NATO-OPT vs Adam on SQuAD v1.1
(noisy QA training), measure EM/F1, compute statistical significance.

Requirements:
    pip install transformers datasets torch scipy nato_opt nltk
    python -c "import nltk; nltk.download('wordnet')"
"""

import os
import random
import csv
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    default_data_collator
)

from nltk.corpus import wordnet
from scipy.stats import ttest_rel, wilcoxon

# NATO optimizer
from nato_opt import NATOOptimizer, fourier_spectral_penalty, low_pass_filter_gradients

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
LR = 3e-5
EPOCHS = 1     # 1 epoch is enough for repeated trials
BATCH_SIZE = 16
N_TRIALS = 5
RESULTS_CSV = "results_squad_noise.csv"

NOISE_LEVEL = 0.15   # fraction of tokens modified

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)


# --------------------------------------------------
# TEXT NOISE FUNCTIONS
# --------------------------------------------------
def random_char_swap(text):
    chars = list(text)
    if len(chars) > 1:
        idx = random.randint(0, len(chars)-2)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    return "".join(chars)


def random_word_delete(words):
    if len(words) < 2:
        return words
    idx = random.randint(0, len(words)-1)
    return words[:idx] + words[idx+1:]


def synonym_replacement(words):
    idx = random.randint(0, len(words)-1)
    syns = wordnet.synsets(words[idx])
    if syns:
        lemmas = syns[0].lemma_names()
        if lemmas:
            words[idx] = lemmas[0].replace("_", " ")
    return words


def apply_noise(text, noise_level=NOISE_LEVEL):
    words = text.split()
    n_changes = max(1, int(len(words) * noise_level))

    for _ in range(n_changes):
        r = random.random()
        if r < 0.33:
            text = random_char_swap(text)
        elif r < 0.66:
            words = random_word_delete(words)
            text = " ".join(words)
        else:
            words = synonym_replacement(words)
            text = " ".join(words)

    return text


# --------------------------------------------------
# LOAD SQuAD
# --------------------------------------------------
def load_squad_with_noise():
    ds = load_dataset("squad")

    # Add noise to questions + contexts
    def add_noise(example):
        example["context"] = apply_noise(example["context"])
        example["question"] = apply_noise(example["question"])
        return example

    noisy_train = ds["train"].map(add_noise)
    noisy_val = ds["validation"].map(add_noise)
    return noisy_train, noisy_val


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# --------------------------------------------------
# TOKENIZATION + LABEL ALIGNMENT
# --------------------------------------------------
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )

    # Build start/end positions
    start_positions = []
    end_positions = []

    for i in range(len(examples["answers"]["text"])):
        answer = examples["answers"]["text"][i][0]
        start_char = examples["answers"]["answer_start"][i][0]

        context = examples["context"][i]

        end_char = start_char + len(answer)

        # Token alignment
        token_start = inputs.char_to_token(i, start_char)
        token_end = inputs.char_to_token(i, end_char-1)

        if token_start is None:
            token_start = 0
        if token_end is None:
            token_end = 0

        start_positions.append(token_start)
        end_positions.append(token_end)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


# --------------------------------------------------
# EVALUATION (EM/F1)
# --------------------------------------------------
def compute_em_f1(pred_start, pred_end, start_pos, end_pos):
    # exact match = 1 if identical span index
    em = 1.0 if (pred_start == start_pos and pred_end == end_pos) else 0.0

    # F1 token-overlap
    pred_span = set(range(pred_start, pred_end+1))
    gold_span = set(range(start_pos, end_pos+1))

    if len(pred_span) == 0 or len(gold_span) == 0:
        return em, 0.0

    overlap = len(pred_span & gold_span)
    if overlap == 0:
        return em, 0.0

    precision = overlap / len(pred_span)
    recall = overlap / len(gold_span)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1


# --------------------------------------------------
# TRAIN + EVAL FUNCTIONS
# --------------------------------------------------
def run_trial(seed, optimizer_name):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load noisy SQuAD fresh each time (noise is stochastic)
    train_ds, val_ds = load_squad_with_noise()
    train_ds = train_ds.map(preprocess_function, batched=True)
    val_ds   = val_ds.map(preprocess_function, batched=True)

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=default_data_collator)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=default_data_collator)

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)

    if optimizer_name.lower() == "nato":
        optimizer = NATOOptimizer(model.parameters(), lr=LR)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ---- TRAIN ----
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                start_positions=batch["start_positions"],
                end_positions=batch["end_positions"]
            )

            loss = outputs.loss

            try:
                fsp = fourier_spectral_penalty(model, lambda_fsp=1e-6)
                loss = loss + fsp
            except:
                pass

            loss.backward()

            try:
                low_pass_filter_gradients(model, cutoff_ratio=0.5)
            except:
                pass

            try:
                optimizer.step(epoch)
            except:
                optimizer.step()

            optimizer.zero_grad()

    # ---- EVAL ----
    model.eval()
    em_scores = []
    f1_scores = []

    with torch.no_grad():
        for batch in val_loader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            start_preds = outputs.start_logits.argmax(-1).cpu().numpy()
            end_preds   = outputs.end_logits.argmax(-1).cpu().numpy()
            start_true  = batch["start_positions"].cpu().numpy()
            end_true    = batch["end_positions"].cpu().numpy()

            for ps, pe, ts, te in zip(start_preds, end_preds, start_true, end_true):
                em, f1 = compute_em_f1(ps, pe, ts, te)
                em_scores.append(em)
                f1_scores.append(f1)

    return np.mean(em_scores), np.mean(f1_scores)


# --------------------------------------------------
# EXPERIMENT LOOP
# --------------------------------------------------
def run_experiment():
    seeds = [5555 + i * 13 for i in range(N_TRIALS)]
    nato_em, adam_em = [], []
    nato_f1, adam_f1 = [], []

    for i, seed in enumerate(seeds, 1):
        print(f"\n--- Trial {i}/{N_TRIALS} (seed={seed}) NATO ---")
        em, f1 = run_trial(seed, "nato")
        nato_em.append(em); nato_f1.append(f1)
        print(f" NATO EM={em:.4f}  F1={f1:.4f}")

        print(f"--- Trial {i}/{N_TRIALS} (seed={seed}) Adam ---")
        em, f1 = run_trial(seed, "adam")
        adam_em.append(em); adam_f1.append(f1)
        print(f" Adam EM={em:.4f}  F1={f1:.4f}")

    # statistics
    em_t, em_p = ttest_rel(nato_em, adam_em)
    f1_t, f1_p = ttest_rel(nato_f1, adam_f1)

    try:
        em_w, em_pw = wilcoxon(np.array(nato_em)-np.array(adam_em))
        f1_w, f1_pw = wilcoxon(np.array(nato_f1)-np.array(adam_f1))
    except:
        em_w = em_pw = f1_w = f1_pw = None

    diff_em = np.array(nato_em) - np.array(adam_em)
    diff_f1 = np.array(nato_f1) - np.array(adam_f1)

    d_em = diff_em.mean() / diff_em.std(ddof=1)
    d_f1 = diff_f1.mean() / diff_f1.std(ddof=1)

    print("\n===== SQuAD Noise Results =====")
    print(f"NATO EM: {mean(nato_em):.4f} ± {stdev(nato_em):.4f}")
    print(f"Adam EM: {mean(adam_em):.4f} ± {stdev(adam_em):.4f}")
    print(f"NATO F1: {mean(nato_f1):.4f} ± {stdev(nato_f1):.4f}")
    print(f"Adam F1: {mean(adam_f1):.4f} ± {stdev(adam_f1):.4f}")

    print(f"Paired t-test EM:  t={em_t:.4f}  p={em_p:.4e}")
    print(f"Paired t-test F1: t={f1_t:.4f}  p={f1_p:.4e}")
    print(f"Cohen's d EM:  {d_em:.4f}")
    print(f"Cohen's d F1: {d_f1:.4f}")

    # Save CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial","seed","nato_em","adam_em","nato_f1","adam_f1"])
        for i, s in enumerate(seeds):
            writer.writerow([i+1, s, nato_em[i], adam_em[i], nato_f1[i], adam_f1[i]])

    print(f"\nSaved trial results to {RESULTS_CSV}")


if __name__ == "__main__":
    run_experiment()
